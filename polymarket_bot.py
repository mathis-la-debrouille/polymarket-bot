#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║              Polymarket Live Trading Bot                         ║
║                                                                  ║
║  Scans live markets → computes EV + Kelly sizing →               ║
║  submits orders via Polymarket CLOB API                          ║
║                                                                  ║
║  SAFETY DEFAULTS:                                                ║
║    • Paper trading ON by default (--live flag required to trade) ║
║    • Hard kill switch at configurable max drawdown               ║
║    • Per-trade stake cap (default: $2 max / trade)               ║
║    • Full audit log written to bot_log.jsonl                     ║
╚══════════════════════════════════════════════════════════════════╝

SETUP (run these commands in your terminal before using):
    pip install py-clob-client python-dotenv requests

USAGE:
    # Paper mode (no real money, just shows what would be traded):
    python polymarket_bot.py

    # Live mode (REAL orders, REAL money):
    python polymarket_bot.py --live

    # Run once, then exit (cron-friendly):
    python polymarket_bot.py --live --once

    # Adjust scan interval and bankroll:
    python polymarket_bot.py --live --interval 300 --bankroll 20
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import numpy as np

# ──────────────────────────────────────────────────────────────────
# DEPENDENCIES CHECK
# ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[!] python-dotenv not found. Run: pip install python-dotenv")
    sys.exit(1)

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
    from py_clob_client.order_builder.constants import BUY, SELL
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False
    BUY  = "BUY"   # fallback constants so paper-mode doesn't crash
    SELL = "SELL"
    print("[!] py-clob-client not found. Install with: pip install py-clob-client")
    print("    Running in paper-mode only until installed.\n")

from signal_manifold import fetch_manifold_price

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
GAMMA_API    = "https://gamma-api.polymarket.com"
CLOB_HOST    = "https://clob.polymarket.com"
CHAIN_ID     = 137   # Polygon mainnet

# Risk limits
EV_THRESHOLD      = 0.06    # minimum EV to place a bet
MAX_STAKE_USD     = 2.00    # absolute max bet size in USD
MIN_STAKE_USD     = 0.50    # minimum order size
MAX_DRAWDOWN_PCT  = 30.0    # kill switch: stop trading if down this % from peak
KELLY_FRACTION    = 0.20    # conservative fractional Kelly
MAX_MARKETS_SCAN  = 50      # max markets to evaluate per run
MIN_VOLUME_USD    = 20_000  # skip thin markets
SLIPPAGE_CAP_PCT  = 2.0     # skip if LMSR price impact > 2%

# FIX 3: hourly candles, minimum 24 hours of history
MIN_HISTORY_POINTS = 24
PRICE_FIDELITY     = 60     # minutes per candle

# FIX 2: cap traded_markets list
MAX_TRADED_MARKETS = 200

SCAN_INTERVAL_SEC = 300
LOG_FILE          = "bot_log.jsonl"
STATE_FILE        = "bot_state.json"

# ──────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot_output.log"),
    ],
)
log = logging.getLogger(__name__)


def audit(event: str, data: dict) -> None:
    """Append a structured event to the JSONL audit log."""
    record = {
        "ts":    datetime.now(timezone.utc).isoformat(),
        "event": event,
        **data,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────────
# STATE (persisted across runs)
# ──────────────────────────────────────────────────────────────────
@dataclass
class BotState:
    starting_bankroll:  float = 20.0
    current_bankroll:   float = 20.0
    peak_bankroll:      float = 20.0
    total_trades:       int   = 0
    total_pnl:          float = 0.0
    active_positions:   Dict  = None   # market_id → {question, side, price, stake, ...}
    traded_markets:     List  = None   # market IDs already traded (avoid re-entry)
    resolved_positions: List  = None   # FIX 1: last 100 resolved positions

    def __post_init__(self):
        if self.active_positions is None:
            self.active_positions = {}
        if self.traded_markets is None:
            self.traded_markets = []
        if self.resolved_positions is None:
            self.resolved_positions = []


def load_state(bankroll: float) -> BotState:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                d = json.load(f)
            # resolved_positions may not exist in older state files
            d.setdefault("resolved_positions", [])
            state = BotState(**d)
            log.info(f"Resumed state: bankroll=${state.current_bankroll:.2f}, "
                     f"trades={state.total_trades}, pnl={state.total_pnl:+.2f}")
            return state
        except Exception as e:
            log.warning(f"Could not load state ({e}), starting fresh")
    state = BotState(starting_bankroll=bankroll, current_bankroll=bankroll, peak_bankroll=bankroll)
    log.info(f"New session: starting bankroll=${bankroll:.2f}")
    return state


def save_state(state: BotState) -> None:
    # FIX 2: cap traded_markets at MAX_TRADED_MARKETS, drop oldest
    if len(state.traded_markets) > MAX_TRADED_MARKETS:
        state.traded_markets = state.traded_markets[-MAX_TRADED_MARKETS:]
    with open(STATE_FILE, "w") as f:
        json.dump(asdict(state), f, indent=2)


# ──────────────────────────────────────────────────────────────────
# FIX 1 — POSITION RESOLUTION TRACKING
# ──────────────────────────────────────────────────────────────────
def resolve_positions(state: BotState) -> None:
    """
    Check every open position against the Gamma API.
    If a market has resolved, compute P&L, update bankroll, and archive it.
    Called at the top of every scan, before market fetching.
    """
    if not state.active_positions:
        return

    to_resolve = list(state.active_positions.items())
    resolved_count = 0
    net_pnl = 0.0

    for market_id, pos in to_resolve:
        try:
            resp = requests.get(
                f"{GAMMA_API}/markets/{market_id}",
                timeout=8,
            )
            resp.raise_for_status()
            market = resp.json()
        except Exception as e:
            log.debug(f"  Could not fetch resolution for {market_id}: {e}")
            continue

        resolved = market.get("resolved", False)
        closed   = market.get("closed",   False)
        if not (resolved or closed):
            continue

        # Parse outcomePrices — JSON string like '["0.95","0.05"]'
        try:
            raw_prices = market.get("outcomePrices", "[]")
            if isinstance(raw_prices, str):
                outcome_prices = json.loads(raw_prices)
            else:
                outcome_prices = raw_prices
            outcome_prices = [float(p) for p in outcome_prices]
        except Exception:
            outcome_prices = []

        if not outcome_prices:
            log.warning(f"  Could not parse outcomePrices for {market_id}, skipping resolution")
            continue

        # Find winning outcome index (price > 0.9 means that outcome won)
        winning_index = next(
            (i for i, p in enumerate(outcome_prices) if p > 0.9),
            None,
        )
        if winning_index is None:
            # Market closed but no clear winner yet (e.g. ambiguous)
            log.debug(f"  {market_id}: closed but no decisive outcome yet")
            continue

        # YES=index 0, NO=index 1
        bet_side    = pos.get("side", "YES")
        bet_index   = 0 if bet_side == "YES" else 1
        won         = (winning_index == bet_index)
        entry_price = pos.get("price", 0.5)
        stake       = pos.get("stake", 0.0)

        if won:
            pnl = stake * (1.0 / entry_price - 1.0)
        else:
            pnl = -stake

        state.total_pnl        += pnl
        state.current_bankroll += pnl
        state.peak_bankroll     = max(state.peak_bankroll, state.current_bankroll)

        resolved_record = {
            "market_id":   market_id,
            "question":    pos.get("question", "?"),
            "side":        bet_side,
            "entry_price": entry_price,
            "stake":       stake,
            "won":         won,
            "pnl":         round(pnl, 4),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Keep last 100 resolved positions
        state.resolved_positions.append(resolved_record)
        if len(state.resolved_positions) > 100:
            state.resolved_positions = state.resolved_positions[-100:]

        # Remove from active tracking; allow re-entry if market re-opens
        del state.active_positions[market_id]
        if market_id in state.traded_markets:
            state.traded_markets.remove(market_id)

        audit("position_resolved", resolved_record)

        result_str = f"{'WON' if won else 'LOST'}  pnl={pnl:+.4f}"
        log.info(f"  [RESOLVED] {pos.get('question','?')[:55]}  {result_str}")
        resolved_count += 1
        net_pnl += pnl

    if resolved_count > 0:
        log.info(f"  Resolved {resolved_count} position(s) | net PnL: {net_pnl:+.2f}")
        save_state(state)


# ──────────────────────────────────────────────────────────────────
# MARKET DATA
# ──────────────────────────────────────────────────────────────────
def fetch_live_markets(limit: int = MAX_MARKETS_SCAN) -> List[Dict]:
    """Fetch currently open (non-resolved) markets sorted by volume."""
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "active":    "true",
                "closed":    "false",
                "limit":     limit,
                "order":     "volume",
                "ascending": "false",
            },
            timeout=15,
        )
        resp.raise_for_status()
        markets = resp.json()
        markets = [m for m in markets if float(m.get("volume", 0)) >= MIN_VOLUME_USD]
        log.info(f"Fetched {len(markets)} live markets")
        return markets
    except Exception as e:
        log.error(f"Failed to fetch markets: {e}")
        return []


def fetch_price_history(token_id: str, fidelity: int = PRICE_FIDELITY) -> List[Dict]:
    """
    Fetch price history for a market token.
    FIX 3: fidelity=60 (hourly candles).
    The CLOB API requires 'interval' or startTs/endTs alongside fidelity.
    We request 2 weeks of hourly data (interval=2w).
    """
    if not token_id:
        return []
    try:
        resp = requests.get(
            f"{CLOB_HOST}/prices-history",
            params={"market": token_id, "fidelity": fidelity, "interval": "1w"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return [{"t": h["t"], "p": float(h["p"])} for h in data.get("history", [])
                if "t" in h and "p" in h]
    except Exception:
        return []


def get_orderbook_midprice(token_id: str) -> Optional[float]:
    """Get current best bid/ask midprice from CLOB orderbook."""
    try:
        resp = requests.get(f"{CLOB_HOST}/midpoint", params={"token_id": token_id}, timeout=8)
        resp.raise_for_status()
        return float(resp.json().get("mid", 0))
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# SIGNAL ENGINE
# ──────────────────────────────────────────────────────────────────
def model_probability(history: List[Dict]) -> Optional[float]:
    """
    MACD momentum model on hourly candles.
    FIX 3: tuned alphas for hourly data.
      fast alpha=0.15  ≈ 6-hour EMA
      slow alpha=0.03  ≈ 32-hour EMA
    """
    if len(history) < MIN_HISTORY_POINTS:
        return None
    prices = [h["p"] for h in history]
    fast, slow = prices[0], prices[0]
    for p in prices:
        fast = 0.15 * p + 0.85 * fast   # FIX 3: was 0.25
        slow = 0.03 * p + 0.97 * slow   # FIX 3: was 0.06
    momentum = fast - slow
    model_p  = prices[-1] + momentum * 0.6
    return float(np.clip(model_p, 0.02, 0.98))


def compute_ev(model_p: float, market_price: float, fee: float = 0.02) -> float:
    """
    FIX 4: standard binary bet EV.
    EV = model_p / market_price - 1 - fee
    (replaces wrong formula that amplified long shots)
    """
    if not (0 < market_price < 1):
        return 0.0
    return model_p / market_price - 1.0 - fee


def kelly_size(model_p: float, market_price: float, bankroll: float) -> float:
    """Fractional Kelly stake in USD, hard-capped at MAX_STAKE_USD."""
    if not (0 < market_price < 1):
        return 0.0
    odds  = (1.0 - market_price) / market_price
    f     = (model_p * odds - (1 - model_p)) / odds
    f    *= KELLY_FRACTION
    stake = max(0.0, f) * bankroll
    return min(stake, MAX_STAKE_USD)


def lmsr_slippage(current_price: float, trade_usd: float, liquidity: float) -> float:
    """Estimate LMSR slippage for a USD trade at given liquidity depth."""
    b = max(liquidity, 10)
    shares = trade_usd / current_price
    p = np.clip(current_price, 1e-6, 1 - 1e-6)
    q_yes = b * np.log(p / (1 - p))
    denom_after = np.exp((q_yes + shares) / b) + 1
    p_after = np.exp((q_yes + shares) / b) / denom_after
    return abs(p_after - p)


# ──────────────────────────────────────────────────────────────────
# TRADE EXECUTION
# ──────────────────────────────────────────────────────────────────
def submit_order(
    client:   "ClobClient",
    token_id: str,
    side:     str,
    price:    float,
    size_usd: float,
    paper:    bool,
) -> Dict:
    side_const = BUY if side == "BUY" else SELL

    if paper:
        log.info(f"  [PAPER] Would {side} {size_usd:.2f} USD @ {price:.3f} (token {token_id[:12]}…)")
        return {"status": "paper", "price": price, "size": size_usd}

    if not CLOB_AVAILABLE:
        log.warning("  [!] py-clob-client not available — cannot submit real order")
        return {"status": "error", "reason": "clob_unavailable"}

    try:
        order_args   = OrderArgs(token_id=token_id, price=round(price, 4),
                                 size=round(size_usd, 2), side=side_const)
        signed_order = client.create_order(order_args)
        resp         = client.post_order(signed_order)
        log.info(f"  [LIVE] Order submitted: {resp}")
        return {"status": "submitted", "response": str(resp)}
    except Exception as e:
        log.error(f"  [!] Order failed: {e}")
        return {"status": "error", "reason": str(e)}


# ──────────────────────────────────────────────────────────────────
# KILL SWITCH
# ──────────────────────────────────────────────────────────────────
def check_kill_switch(state: BotState) -> bool:
    if state.peak_bankroll <= 0:
        return False
    drawdown_pct = (state.peak_bankroll - state.current_bankroll) / state.peak_bankroll * 100
    if drawdown_pct >= MAX_DRAWDOWN_PCT:
        log.error(
            f"🛑  KILL SWITCH TRIGGERED: drawdown={drawdown_pct:.1f}% "
            f"(limit={MAX_DRAWDOWN_PCT}%). Halting all trading."
        )
        audit("kill_switch", {"drawdown_pct": drawdown_pct, "bankroll": state.current_bankroll})
        return True
    return False


# ──────────────────────────────────────────────────────────────────
# REAL BALANCE SYNC
# ──────────────────────────────────────────────────────────────────
def sync_real_balance(state: BotState, clob_client: Optional["ClobClient"] = None) -> None:
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        if clob_client is None:
            pk = os.environ.get("PRIVATE_KEY", "")
            if not pk:
                return
            clob_client = ClobClient(CLOB_HOST, key=pk, chain_id=CHAIN_ID, signature_type=1)
            creds = clob_client.create_or_derive_api_creds()
            clob_client.set_api_creds(creds)

        bal  = clob_client.get_balance_allowance(BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL, signature_type=1
        ))
        real = float(bal.get("balance", 0)) / 1_000_000
        if real <= 0:
            return
        old  = state.current_bankroll
        state.current_bankroll = real
        state.peak_bankroll    = max(state.peak_bankroll, real)
        log.info(f"  [✓] Balance synced from CLOB: ${real:.4f}  (was ${old:.4f})")
        audit("balance_sync", {"real": real, "was_tracked": old})
    except Exception as e:
        log.debug(f"  Balance sync skipped ({e}) — using tracked value")


# ──────────────────────────────────────────────────────────────────
# MAIN SCAN LOOP
# ──────────────────────────────────────────────────────────────────
def run_scan(client: Optional["ClobClient"], state: BotState, paper: bool) -> None:
    """One full scan: resolve positions → fetch markets → evaluate → place trades."""
    log.info(f"{'─'*55}")
    log.info(f"Scan started | bankroll=${state.current_bankroll:.2f} | "
             f"{'PAPER' if paper else '🔴 LIVE'}")
    log.info(f"{'─'*55}")

    # FIX 1: resolve any positions before scanning new ones
    resolve_positions(state)

    # Sync real balance
    sync_real_balance(state, clob_client=client)

    if check_kill_switch(state):
        return

    raw_markets = fetch_live_markets()
    if not raw_markets:
        log.warning("No markets returned — API may be down")
        return

    signals_found = 0

    for raw in raw_markets:
        market_id = raw.get("id", "")
        question  = raw.get("question", "?")[:65]

        if market_id in state.traded_markets:
            continue

        # Extract YES/NO token IDs from clobTokenIds field (index 0=YES, 1=NO)
        try:
            clob_ids = raw.get("clobTokenIds", "[]")
            if isinstance(clob_ids, str):
                clob_ids = json.loads(clob_ids)
            yes_token_id = clob_ids[0] if len(clob_ids) > 0 else ""
            no_token_id  = clob_ids[1] if len(clob_ids) > 1 else ""
            # Keep yes_token/no_token dicts for order submission compatibility
            yes_token = {"token_id": yes_token_id, "outcome": "YES"} if yes_token_id else None
            no_token  = {"token_id": no_token_id,  "outcome": "NO"}  if no_token_id  else None
        except Exception:
            yes_token_id, no_token_id = "", ""
            yes_token, no_token = None, None
        history = fetch_price_history(yes_token_id)
        if len(history) < MIN_HISTORY_POINTS:
            log.info(f"  {question[:48]:48s}  ✗ not enough price history ({len(history)} pts, need {MIN_HISTORY_POINTS})")
            continue

        current_price = history[-1]["p"]
        if not (0.05 < current_price < 0.95):
            log.info(f"  {question[:48]:48s}  ✗ near-resolved (price={current_price:.2f})")
            continue

        # FIX 3: MACD with hourly-tuned alphas
        model_p = model_probability(history)
        if model_p is None:
            continue

        # FIX 5: Manifold cross-platform signal
        manifold_p = fetch_manifold_price(question)
        if manifold_p is not None:
            log.info(
                f"  Manifold: {manifold_p:.3f} | Poly: {current_price:.3f} "
                f"| diff: {manifold_p - current_price:+.3f}"
            )
            # Blend: weight Manifold 60%, MACD 40%
            model_p = 0.4 * model_p + 0.6 * manifold_p
            model_p = float(np.clip(model_p, 0.02, 0.98))
        else:
            log.info(f"  Manifold: no match for '{question[:40]}'")

        # FIX 4: correct binary EV formula
        ev_yes  = compute_ev(model_p,       current_price)
        ev_no   = compute_ev(1 - model_p,   1 - current_price)
        best_ev = max(ev_yes, ev_no)

        verdict = "✓ EDGE" if best_ev >= EV_THRESHOLD else f"✗ EV too low (need {EV_THRESHOLD:.0%})"
        log.info(
            f"  {question[:48]:48s}  "
            f"price={current_price:.2f}  model={model_p:.2f}  "
            f"EV={best_ev:+.3f}  {verdict}"
        )

        if ev_yes >= ev_no and ev_yes >= EV_THRESHOLD:
            bet_side  = "YES"
            ev        = ev_yes
            bet_price = current_price
            token     = yes_token
            used_p    = model_p
        elif ev_no > ev_yes and ev_no >= EV_THRESHOLD:
            bet_side  = "NO"
            ev        = ev_no
            bet_price = 1 - current_price
            token     = no_token
            used_p    = 1 - model_p
        else:
            continue

        # Slippage check
        liquidity = float(raw.get("liquidity", 100))
        stake     = kelly_size(used_p, bet_price, state.current_bankroll)
        if stake < MIN_STAKE_USD:
            log.info(f"  Skip (Kelly stake ${stake:.2f} < min ${MIN_STAKE_USD}): {question[:40]}")
            continue

        slip = lmsr_slippage(bet_price, stake, liquidity)
        if slip / bet_price > (SLIPPAGE_CAP_PCT / 100):
            log.info(f"  Skip (slippage {slip/bet_price*100:.1f}% > {SLIPPAGE_CAP_PCT}%): {question[:40]}")
            continue

        signals_found += 1
        token_id = token.get("token_id", "") if token else ""

        log.info(f"\n  ✦ SIGNAL FOUND")
        log.info(f"    Market    : {question}")
        log.info(f"    Side      : {bet_side}")
        log.info(f"    Price     : {bet_price:.3f}  |  Model p: {used_p:.3f}  |  EV: {ev:.4f}")
        log.info(f"    Manifold  : {manifold_p:.3f}" if manifold_p else "    Manifold  : no match")
        log.info(f"    Stake     : ${stake:.2f}  |  Slippage: {slip/bet_price*100:.2f}%")
        log.info(f"    Token ID  : {token_id[:20]}…")

        audit("signal", {
            "market_id":  market_id,
            "question":   question,
            "side":       bet_side,
            "price":      bet_price,
            "model_p":    used_p,
            "manifold_p": manifold_p,
            "ev":         ev,
            "stake":      stake,
            "token_id":   token_id,
            "paper":      paper,
        })

        result = submit_order(
            client   = client,
            token_id = token_id,
            side     = "BUY",
            price    = bet_price,
            size_usd = stake,
            paper    = paper,
        )

        if result.get("status") in ("paper", "submitted"):
            state.traded_markets.append(market_id)
            state.total_trades += 1
            if paper:
                state.active_positions[market_id] = {
                    "question":  question,
                    "side":      bet_side,
                    "token_id":  token_id,
                    "price":     bet_price,
                    "stake":     stake,
                    "ev":        ev,
                    "paper":     True,
                }
                state.current_bankroll -= stake
                state.current_bankroll  = max(0.0, state.current_bankroll)

            audit("order_placed", {
                "market_id": market_id,
                "stake":     stake,
                "result":    result,
            })

        save_state(state)

    if signals_found == 0:
        log.info("  No EV signals found this scan.")

    save_state(state)

    log.info(
        f"\n  Bankroll: ${state.current_bankroll:.2f} | "
        f"Trades: {state.total_trades} | Active: {len(state.active_positions)} | "
        f"Resolved: {len(state.resolved_positions)}"
    )


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────
def main():
    global MAX_STAKE_USD, EV_THRESHOLD

    parser = argparse.ArgumentParser(description="Polymarket Live Trading Bot")
    parser.add_argument("--live",      action="store_true",               help="Enable real order submission")
    parser.add_argument("--once",      action="store_true",               help="Run one scan then exit")
    parser.add_argument("--bankroll",  type=float, default=20.0,          help="Starting bankroll in USD")
    parser.add_argument("--interval",  type=int,   default=SCAN_INTERVAL_SEC, help="Seconds between scans")
    parser.add_argument("--max-stake", type=float, default=MAX_STAKE_USD, help="Max USD per trade")
    parser.add_argument("--ev-min",    type=float, default=EV_THRESHOLD,  help="Minimum EV to trade")
    args = parser.parse_args()

    paper = not args.live

    if not paper:
        if not CLOB_AVAILABLE:
            log.error("Cannot run in live mode: py-clob-client not installed.")
            sys.exit(1)
        private_key = os.environ.get("PRIVATE_KEY", "")
        if not private_key or not private_key.startswith("0x"):
            log.error("PRIVATE_KEY not set or invalid in .env file.")
            sys.exit(1)
        log.warning("=" * 55)
        log.warning("  🔴  LIVE MODE — real money will be spent")
        log.warning(f"  Bankroll: ${args.bankroll:.2f}")
        log.warning(f"  Max per trade: ${args.max_stake:.2f}")
        log.warning(f"  Kill switch: -{MAX_DRAWDOWN_PCT:.0f}% drawdown")
        log.warning("=" * 55)
        time.sleep(3)
    else:
        log.info("=" * 55)
        log.info("  📄  PAPER MODE — no real orders will be placed")
        log.info("  (run with --live to trade real money)")
        log.info("=" * 55)

    MAX_STAKE_USD = args.max_stake
    EV_THRESHOLD  = args.ev_min

    client = None
    if not paper and CLOB_AVAILABLE:
        try:
            client = ClobClient(
                CLOB_HOST,
                key            = os.environ["PRIVATE_KEY"],
                chain_id       = CHAIN_ID,
                signature_type = 1,
            )
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            log.info(f"  [✓] Authenticated with Polymarket CLOB")
            audit("session_start", {"mode": "live", "bankroll": args.bankroll})
        except Exception as e:
            log.error(f"Authentication failed: {e}")
            sys.exit(1)
    else:
        audit("session_start", {"mode": "paper", "bankroll": args.bankroll})

    state = load_state(args.bankroll)

    try:
        while True:
            run_scan(client, state, paper)
            if args.once:
                log.info("--once flag set, exiting after single scan.")
                break
            log.info(f"\n  Sleeping {args.interval}s until next scan …\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("\n  Bot stopped by user (Ctrl+C)")
        audit("session_end", {
            "reason":   "user_interrupt",
            "bankroll": state.current_bankroll,
            "trades":   state.total_trades,
        })
        save_state(state)


if __name__ == "__main__":
    main()
