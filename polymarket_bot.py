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
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from signal_manifold import fetch_manifold_price, _cache_ratio as _manifold_cache_ratio
try:
    from signal_updown import compute_updown_signal, is_updown_market
    UPDOWN_AVAILABLE = True
except ImportError:
    UPDOWN_AVAILABLE = False
    log.warning("[!] signal_updown.py not found — crypto Up/Down pass disabled")
from signal_metaculus import fetch_metaculus_price

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
GAMMA_API    = "https://gamma-api.polymarket.com"
CLOB_HOST    = "https://clob.polymarket.com"
CHAIN_ID     = 137   # Polygon mainnet

# Proxy wallet — set FUNDER_ADDRESS in .env to enable live orders.
# This is the address of the Polymarket proxy wallet that holds USDC.
# Find yours via: call getPolyProxyWalletAddress(your_eoa) on the CTFExchange contract
# CTFExchange: 0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E (selector 0xedef7d8e)
FUNDER_ADDRESS = os.environ.get("FUNDER_ADDRESS", "")

# Risk limits
EV_THRESHOLD      = 0.06    # minimum EV to place a bet
MAX_STAKE_USD     = 2.00    # absolute max bet size in USD
MIN_STAKE_USD     = 1.00    # Polymarket minimum order is $1 USDC
MAX_DRAWDOWN_PCT  = 30.0    # kill switch: stop trading if down this % from peak
KELLY_FRACTION    = 0.35    # conservative fractional Kelly
MAX_MARKETS_SCAN  = 1000    # max markets to evaluate per run
MIN_VOLUME_USD    = 20_000  # skip thin markets
SLIPPAGE_CAP_PCT  = 2.0     # skip if LMSR price impact > 2%

# FIX 3: hourly candles, minimum 24 hours of history
MIN_HISTORY_POINTS = 24
PRICE_FIDELITY     = 60     # minutes per candle

# FIX 2: cap traded_markets list
MAX_TRADED_MARKETS = 200

SCAN_INTERVAL_SEC = 60
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
        # NOTE: no FileHandler — stdout is redirected to bot_output.log
        # by the shell (>> bot_output.log), so FileHandler would duplicate every line.
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
    clob_cash:          float = 0.0    # last known on-chain USDC cash (excludes position value)
    scans_since_sl_check: int = 0      # counter for stop-loss check frequency

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
            d.setdefault("clob_cash", None)  # may not exist in old state files
            d.setdefault("scans_since_sl_check", 0)
            known = {f.name for f in BotState.__dataclass_fields__.values()}
            state = BotState(**{k: v for k, v in d.items() if k in known})
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
    # Atomic write: write to temp file then rename so api_server never reads a partial file
    tmp = str(STATE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(state), f, indent=2)
    os.replace(tmp, STATE_FILE)


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
# STOP-LOSS CHECK
# ──────────────────────────────────────────────────────────────────
def check_stop_loss(state, clob_client) -> None:
    """Check open positions for >30% loss; sell if edge is gone."""
    state.scans_since_sl_check = getattr(state, 'scans_since_sl_check', 0) + 1
    if state.scans_since_sl_check < 3:
        return
    state.scans_since_sl_check = 0

    if not state.active_positions:
        return

    for mid, pos in list(state.active_positions.items()):
        token_id    = pos.get("token_id", "")
        shares      = pos.get("shares", 0)
        entry_price = pos.get("price", 0)
        question    = pos.get("question", "")
        side        = pos.get("side", "YES")

        if not token_id or shares <= 0 or entry_price <= 0:
            continue

        try:
            r = requests.get("https://clob.polymarket.com/midpoint",
                             params={"token_id": token_id}, timeout=5)
            curr_price = float(r.json().get("mid", entry_price))
        except Exception:
            continue

        entry_value   = shares * entry_price
        current_value = shares * curr_price
        loss_pct = (entry_value - current_value) / entry_value if entry_value > 0 else 0

        if loss_pct <= 0.30:
            continue

        log.info(f"  [SL] {question[:50]} — loss {loss_pct:.0%}, re-evaluating edge…")

        try:
            history = fetch_price_history(token_id)
            new_model_p = model_probability(history)
            if new_model_p is None:
                log.warning(f"  [SL] Could not get model_p for {question[:40]}, skipping")
                continue
            manifold_p  = fetch_manifold_price(question)
            metaculus_p = fetch_metaculus_price(question)
            sources = [("macd", new_model_p, 0.30)]
            if manifold_p:
                sources.append(("manifold", manifold_p, 0.40))
            if metaculus_p:
                sources.append(("metaculus", metaculus_p, 0.30))
            total_w = sum(w for _, _, w in sources)
            new_final_p = sum(p * w / total_w for _, p, w in sources)
            bet_price = curr_price if side == "YES" else (1 - curr_price)
            new_ev = (new_final_p / bet_price) - 1 if bet_price > 0 else 0
        except Exception as e:
            log.warning(f"  [SL] Could not re-evaluate {question[:40]}: {e}")
            continue

        original_ev = pos.get("ev", 0)

        if new_ev < 0.03:
            log.info(f"  [SL] Edge gone (new EV={new_ev:.1%}), submitting sell order…")
            try:
                sell_price   = max(0.01, curr_price - 0.01)
                sell_side    = BUY if side == "NO" else SELL
                order_args   = OrderArgs(token_id=token_id, price=round(sell_price, 4),
                                         size=float(shares), side=sell_side)
                signed_order = clob_client.create_order(order_args)
                clob_client.post_order(signed_order)
                audit("stop_loss_triggered", {
                    "market_id":    mid,
                    "question":     question,
                    "entry_price":  entry_price,
                    "current_price": curr_price,
                    "loss_pct":     round(loss_pct, 4),
                    "original_ev":  round(original_ev, 4),
                    "new_ev":       round(new_ev, 4),
                    "shares_sold":  shares,
                })
                del state.active_positions[mid]
                save_state(state)
                log.info(f"  [SL] Sold {shares} shares of {question[:40]}")
            except Exception as e:
                log.error(f"  [SL] Sell order failed: {e}")
        else:
            log.info(f"  [SL] Edge still holds (new EV={new_ev:.1%}), holding.")
            audit("stop_loss_evaluated", {
                "market_id": mid,
                "question":  question,
                "loss_pct":  round(loss_pct, 4),
                "new_ev":    round(new_ev, 4),
            })

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
                "order":     "volumeNum",
                "ascending": "false",
            },
            timeout=15,
        )
        resp.raise_for_status()
        markets = resp.json()
        markets = [m for m in markets if float(m.get("volumeNum", m.get("volume", 0))) >= MIN_VOLUME_USD]
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
        # size must be in SHARES (not USD): minimum 5 shares on Polymarket
        MIN_POLY_SHARES = 5
        MIN_ORDER_USD   = 1.00   # Polymarket minimum $1 per order
        shares = max(math.ceil(size_usd / price), MIN_POLY_SHARES, math.ceil(MIN_ORDER_USD / price))
        actual_cost = shares * price
        if actual_cost > max(size_usd * 3, MIN_ORDER_USD * 1.5):
            log.warning(f"  [!] Skipping order: min-order cost ${actual_cost:.2f} is >3x Kelly ${size_usd:.2f}")
            return {"status": "error", "reason": f"min_order_cost_too_high (${actual_cost:.2f} vs ${size_usd:.2f})"}
        order_args   = OrderArgs(token_id=token_id, price=round(price, 4),
                                 size=shares, side=side_const)
        signed_order = client.create_order(order_args)
        resp         = client.post_order(signed_order)
        log.info(f"  [LIVE] Order submitted: {resp}")

        # Confirm fill: wait 4s then check — cancel if not matched (avoids ghost positions)
        order_id = (resp or {}).get("orderID") or (resp or {}).get("id", "")
        if order_id:
            time.sleep(4)
            try:
                status = client.get_order(order_id)
                matched = float(status.get("size_matched", 0))
                remaining = float(status.get("size_remaining", status.get("original_size", shares)))
                if matched < 1:
                    client.cancel(order_id)
                    log.warning(f"  [!] Order {order_id[:12]} unmatched (0 shares filled) — cancelled")
                    return {"status": "error", "reason": "order_unmatched_cancelled"}
                log.info(f"  [✓] Confirmed {matched} shares filled (of {shares} ordered)")
                if remaining > 0:
                    client.cancel(order_id)  # cancel any remaining partial
            except Exception as ce:
                log.debug(f"  Order confirm check failed ({ce}) — assuming filled")

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
        clob_cash = float(bal.get("balance", 0)) / 1_000_000
        if clob_cash < 0:
            return

        # Sum up position values using midpoint prices
        pos_value = 0.0
        for mid, pos in state.active_positions.items():
            token_id = pos.get("token_id", "")
            shares   = pos.get("shares", 0)
            fallback = pos.get("price", 0)
            if shares <= 0:
                continue
            try:
                r = requests.get(
                    "https://clob.polymarket.com/midpoint",
                    params={"token_id": token_id},
                    timeout=5,
                )
                mid_price = float(r.json().get("mid", fallback))
            except Exception:
                mid_price = fallback
            pos_value += shares * mid_price

        portfolio = clob_cash + pos_value
        old = state.current_bankroll
        state.current_bankroll = portfolio
        state.clob_cash        = clob_cash
        state.peak_bankroll    = max(state.peak_bankroll, portfolio)
        log.info(
            f"  [✓] Balance synced: cash=${clob_cash:.4f}  "
            f"positions=${pos_value:.4f}  portfolio=${portfolio:.4f}  (was ${old:.4f})"
        )
        audit("balance_sync", {
            "clob_cash": clob_cash,
            "pos_value": pos_value,
            "portfolio": portfolio,
            "was_tracked": old,
        })
    except Exception as e:
        log.debug(f"  Balance sync skipped ({e}) — using tracked value")


# ──────────────────────────────────────────────────────────────────
# MAIN SCAN LOOP
# ──────────────────────────────────────────────────────────────────
def fetch_updown_markets() -> list:
    """Fetch active Bitcoin/Ethereum Up or Down markets from Gamma API."""
    try:
        # Order by endDate ascending to get soon-expiring windows first.
        # No volume filter: these 5-min markets have ~$10 volume by design.
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={"active": "true", "closed": "false",
                    "limit": 200, "order": "endDate", "ascending": "true"},
            timeout=15,
        )
        resp.raise_for_status()
        all_markets = resp.json()
        updown = [m for m in all_markets if is_updown_market(m.get("question", ""))]
        log.info(f"  [Pass A] Up/Down markets found: {len(updown)}/{len(all_markets)}")
        return updown
    except Exception as e:
        log.error(f"fetch_updown_markets: {e}")
        return []

def run_scan(client: Optional["ClobClient"], state: BotState, paper: bool) -> None:
    """One full scan: resolve positions → fetch markets → evaluate → place trades."""
    scan_start_time = time.time()
    log.info(f"{'─'*55}")
    log.info(f"Scan started | bankroll=${state.current_bankroll:.2f} | "
             f"{'PAPER' if paper else '🔴 LIVE'}")
    log.info(f"{'─'*55}")

    # FIX 1: resolve any positions before scanning new ones
    resolve_positions(state)
    check_stop_loss(state, client)

    # Sync real balance
    sync_real_balance(state, clob_client=client)

    if check_kill_switch(state):
        return

    # ===============================================================
    # PASS A -- Crypto Up/Down markets (Binance signal)
    # ===============================================================
    if UPDOWN_AVAILABLE:
        updown_found = 0
        for raw in fetch_updown_markets():
            market_id = raw.get("id", "")
            question  = raw.get("question", "?")
            if market_id in state.traded_markets:
                continue
            try:
                tokens    = json.loads(raw["tokens"]) if isinstance(raw.get("tokens"), str) else raw.get("tokens", [])
                yes_token = next((t for t in tokens if t.get("outcome","").upper()=="YES"), None)
                no_token  = next((t for t in tokens if t.get("outcome","").upper()=="NO"),  None)
            except Exception:
                yes_token, no_token = None, None
            if not yes_token:
                continue
            yes_token_id = yes_token.get("token_id", "")
            yes_midprice = get_orderbook_midprice(yes_token_id) or 0.50
            if not (0.03 < yes_midprice < 0.97):
                continue
            sig = compute_updown_signal(raw, yes_midprice)
            if sig is None:
                continue
            model_p    = sig["model_p"]
            confidence = sig["confidence"]
            if confidence < 0.35:
                log.debug(f"  [Pass A] Skip low-conf ({confidence:.2f}): {question[:50]}")
                continue
            no_midprice = 1.0 - yes_midprice
            ev_yes = compute_ev(model_p,         yes_midprice)
            ev_no  = compute_ev(1.0 - model_p,   no_midprice)
            if sig["arb_detected"] and sig["arb_ev"] > 0.02:
                log.info(f"  ARB EV={sig['arb_ev']:.3f}  {question[:50]}")
                ev_yes = sig["arb_ev"]
                ev_no  = sig["arb_ev"]
                model_p = 0.97
            if ev_yes >= ev_no and ev_yes >= EV_THRESHOLD:
                bet_side, ev, bet_price, token, used_p = "YES", ev_yes, yes_midprice, yes_token, model_p
            elif ev_no > ev_yes and ev_no >= EV_THRESHOLD:
                bet_side, ev, bet_price, token, used_p = "NO", ev_no, no_midprice, no_token, 1.0 - model_p
            else:
                continue
            if not token:
                continue
            token_id = token.get("token_id", "")
            stake = kelly_size(used_p, bet_price, state.current_bankroll)
            if stake < MIN_STAKE_USD:
                log.info(f"  [Pass A] Skip (stake ${stake:.2f} < min ${MIN_STAKE_USD}): {question[:50]}")
                continue
            log.info(
                f"\n  CRYPTO SIGNAL  {question[:60]}\n"
                f"    {bet_side} price={bet_price:.3f} model_p={used_p:.3f} EV={ev:.4f} "
                f"conf={confidence:.2f} stake=${stake:.2f}\n"
                f"    move={sig['pct_move']*100:+.3f}% {sig['minutes_left']:.1f}min [{sig['symbol']}]"
            )
            audit("signal", {
                "market_id": market_id, "question": question[:80],
                "side": bet_side, "price": bet_price, "model_p": used_p,
                "ev": ev, "stake": stake, "signal_type": "crypto_updown",
                "confidence": confidence, "pct_move": sig["pct_move"],
                "minutes_left": sig["minutes_left"], "paper": paper,
            })
            result = submit_order(client=client, token_id=token_id,
                                  side="BUY", price=bet_price, size_usd=stake, paper=paper)
            if result.get("status") in ("paper", "submitted"):
                state.traded_markets.append(market_id)
                state.total_trades += 1
                state.active_positions[market_id] = {
                    "question": question, "side": bet_side,
                    "token_id": token_id, "yes_token_id": yes_token_id,
                    "price": bet_price, "stake": stake, "ev": ev,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "paper": paper, "signal_type": "crypto_updown",
                }
                state.current_bankroll = max(0.0, state.current_bankroll - stake)
                audit("order_placed", {"market_id": market_id, "stake": stake,
                                       "signal_type": "crypto_updown", "result": result})
                save_state(state)
                updown_found += 1
        if updown_found == 0:
            log.info("  [Pass A] No crypto signals this scan.")
    # ===============================================================
    # PASS B -- General prediction markets (existing logic)
    # ===============================================================
    raw_markets = fetch_live_markets()
    if not raw_markets:
        log.warning("No markets returned — API may be down")
        return

    signals_found       = 0
    n_skipped_traded    = 0
    n_skipped_history   = 0
    n_skipped_price     = 0
    n_evaluated         = 0
    n_manifold_match    = 0
    n_manifold_rejected = 0
    scan_market_log: List[Dict] = []   # per-market data written to scan_markets.json

    # Probability band filter: evaluate mid-prob markets first (bigger Kelly stakes),
    # then long shots only if Bucket A yields < 3 signals.
    BUCKET_A_CAP = 100   # max Bucket A markets to evaluate per scan
    BUCKET_B_CAP = 50    # max Bucket B markets to evaluate per scan
    LONGSHOT_STAKE_CAP = 0.50  # cap stake on long shots regardless of Kelly

    def _get_last_price(m: Dict) -> float:
        try:
            return float(m.get("lastTradePrice") or m.get("outcomePrices", "[0.5]").strip("[]").split(",")[0])
        except Exception:
            return 0.5

    bucket_a = [m for m in raw_markets if 0.20 <= _get_last_price(m) <= 0.80][:BUCKET_A_CAP]
    bucket_b = [
        m for m in raw_markets
        if m not in bucket_a and 0.05 <= _get_last_price(m) <= 0.95
    ][:BUCKET_B_CAP]

    # ── Parallel prefetch: price histories + manifold cache warm ──
    def _token_ids(m):
        try:
            ids = m.get("clobTokenIds", "[]")
            if isinstance(ids, str):
                ids = json.loads(ids)
            return ids[0] if ids else ""
        except Exception:
            return ""

    _all_prefetch = bucket_a + bucket_b
    _token_map    = {m.get("id", ""): _token_ids(m) for m in _all_prefetch}
    _question_map = {m.get("id", ""): m.get("question", "")[:65] for m in _all_prefetch}

    history_cache: Dict[str, List] = {}
    with ThreadPoolExecutor(max_workers=30) as _ex:
        _futs = {_ex.submit(fetch_price_history, tid): (mid, tid)
                 for mid, tid in _token_map.items() if tid}
        for _f in as_completed(_futs):
            _mid, _tid = _futs[_f]
            history_cache[_tid] = _f.result() or []

    with ThreadPoolExecutor(max_workers=15) as _ex:
        list(_ex.map(fetch_manifold_price, _question_map.values()))

    signals_from_a = 0
    active_longshot_cap = False

    def _markets_to_scan():
        nonlocal active_longshot_cap
        yield from bucket_a
        if signals_from_a < 3:
            active_longshot_cap = True
            yield from bucket_b

    for raw in _markets_to_scan():
        market_id = raw.get("id", "")
        question  = raw.get("question", "?")[:65]
        volume    = float(raw.get("volume", 0))

        if market_id in state.traded_markets:
            n_skipped_traded += 1
            if market_id in state.active_positions:
                skip_reason  = "position_open"
                skip_detail  = "Position currently open — bot holds this market and is waiting for resolution."
            else:
                skip_reason  = "already_traded"
                skip_detail  = (
                    "Processed this session (order attempted or placed). "
                    "Skipped to avoid re-entry. Only successful trades appear in Recent Trades."
                )
            scan_market_log.append({
                "market_id": market_id, "question": question,
                "volume": volume, "skip_reason": skip_reason,
                "skip_detail": skip_detail,
                "signal": False,
            })
            continue

        # Extract YES/NO token IDs from clobTokenIds field (index 0=YES, 1=NO)
        try:
            clob_ids = raw.get("clobTokenIds", "[]")
            if isinstance(clob_ids, str):
                clob_ids = json.loads(clob_ids)
            yes_token_id = clob_ids[0] if len(clob_ids) > 0 else ""
            no_token_id  = clob_ids[1] if len(clob_ids) > 1 else ""
            yes_token = {"token_id": yes_token_id, "outcome": "YES"} if yes_token_id else None
            no_token  = {"token_id": no_token_id,  "outcome": "NO"}  if no_token_id  else None
        except Exception:
            yes_token_id, no_token_id = "", ""
            yes_token, no_token = None, None

        history = history_cache.get(yes_token_id, [])
        if len(history) < MIN_HISTORY_POINTS:
            log.info(f"  {question[:48]:48s}  ✗ not enough price history ({len(history)} pts, need {MIN_HISTORY_POINTS})")
            n_skipped_history += 1
            scan_market_log.append({
                "market_id": market_id, "question": question, "volume": volume,
                "skip_reason": "no_history",
                "skip_detail": f"Only {len(history)} hourly candles available. Need {MIN_HISTORY_POINTS}h for MACD signal.",
                "signal": False,
            })
            continue

        current_price = history[-1]["p"]
        if not (0.05 < current_price < 0.95):
            log.info(f"  {question[:48]:48s}  ✗ near-resolved (price={current_price:.2f})")
            n_skipped_price += 1
            scan_market_log.append({
                "market_id": market_id, "question": question, "volume": volume,
                "price": round(current_price, 4),
                "skip_reason": "near_resolved",
                "skip_detail": f"Price is {current_price:.1%} — market is near resolution. Bot only trades 5%–95% range.",
                "signal": False,
            })
            continue

        n_evaluated += 1
        in_bucket_a = raw in bucket_a

        # Resolution date scoring
        end_date_str = raw.get("endDate") or raw.get("end_date_min") or ""
        days_until = None
        if end_date_str:
            try:
                from datetime import datetime as _dt, timezone as _tz
                end_dt = _dt.fromisoformat(end_date_str.replace("Z", "+00:00"))
                days_until = (end_dt - _dt.now(_tz.utc)).days
            except Exception:
                pass

        if days_until is not None:
            if days_until <= 30:
                date_score = 1.5
            elif days_until <= 90:
                date_score = 1.0
            else:
                date_score = max(0.6, 1.0 - (days_until - 90) / 1000)
        else:
            date_score = 1.0

        log.info(f"  endDate={end_date_str[:10] if end_date_str else 'N/A'}  days_until={days_until}  date_score={date_score:.2f}")

        # FIX 3: MACD with hourly-tuned alphas
        model_p = model_probability(history)
        if model_p is None:
            scan_market_log.append({
                "market_id": market_id, "question": question, "volume": volume,
                "price": round(current_price, 4),
                "skip_reason": "no_model",
                "skip_detail": "MACD model returned no probability (insufficient price variation).",
                "signal": False,
            })
            continue

        # FIX 5: Multi-source signal blending (Manifold + Metaculus)
        manifold_raw = fetch_manifold_price(question)
        manifold_p   = manifold_raw
        manifold_rejected = False
        if manifold_p is not None:
            n_manifold_match += 1
            diff = abs(manifold_p - current_price)
            if diff > 0.50:
                log.info(
                    f"  Manifold: {manifold_p:.3f} REJECTED (divergence {diff:.2f} > 0.50 "
                    f"from Poly {current_price:.3f})"
                )
                manifold_rejected = True
                manifold_p = None
                n_manifold_rejected += 1
            else:
                log.info(
                    f"  Manifold: {manifold_p:.3f} | Poly: {current_price:.3f} "
                    f"| diff: {manifold_p - current_price:+.3f}"
                )
        else:
            log.info(f"  Manifold: no match for '{question[:40]}'")

        metaculus_p = fetch_metaculus_price(question)
        if metaculus_p is not None:
            log.info(f"  Metaculus: {metaculus_p:.3f}")
        else:
            log.info(f"  Metaculus: no match for '{question[:40]}'")

        sources = [("macd", model_p, 0.30)]
        if manifold_p is not None:
            sources.append(("manifold", manifold_p, 0.40))
        if metaculus_p is not None:
            sources.append(("metaculus", metaculus_p, 0.30))

        total_w = sum(w for _, _, w in sources)
        final_model_p = sum(p * w / total_w for _, p, w in sources)
        final_model_p = float(np.clip(final_model_p, 0.02, 0.98))

        log.info("  Signal sources: " + ", ".join(
            f"{name}={p:.3f}(w={w/total_w:.0%})" for name, p, w in sources
        ))

        model_p = final_model_p

        # FIX 4: correct binary EV formula
        ev_yes  = compute_ev(model_p,       current_price)
        ev_no   = compute_ev(1 - model_p,   1 - current_price)
        best_ev = max(ev_yes, ev_no)
        best_side = "YES" if ev_yes >= ev_no else "NO"

        effective_ev_yes = ev_yes * date_score
        effective_ev_no  = ev_no  * date_score
        best_effective_ev = max(effective_ev_yes, effective_ev_no)

        verdict = "✓ EDGE" if best_effective_ev >= EV_THRESHOLD else f"✗ EV too low (need {EV_THRESHOLD:.0%})"
        log.info(
            f"  {question[:48]:48s}  "
            f"price={current_price:.2f}  model={model_p:.2f}  "
            f"EV={best_ev:+.3f}  effEV={best_effective_ev:+.3f}  {verdict}"
        )

        if effective_ev_yes >= effective_ev_no and effective_ev_yes >= EV_THRESHOLD:
            bet_side  = "YES"
            ev        = ev_yes
            bet_price = current_price
            token     = yes_token
            used_p    = model_p
        elif effective_ev_no > effective_ev_yes and effective_ev_no >= EV_THRESHOLD:
            bet_side  = "NO"
            ev        = ev_no
            bet_price = 1 - current_price
            token     = no_token
            used_p    = 1 - model_p
        else:
            manifold_note = ""
            if manifold_rejected:
                manifold_note = f" Manifold returned {manifold_raw:.3f} but was rejected (>{50}% divergence from Poly)."
            elif manifold_p is not None:
                manifold_note = f" Manifold blended: {manifold_p:.3f}."
            scan_market_log.append({
                "market_id": market_id, "question": question, "volume": volume,
                "price": round(current_price, 4), "model_p": round(model_p, 4),
                "manifold_p": round(manifold_p, 4) if manifold_p else None,
                "ev": round(best_ev, 4), "best_side": best_side,
                "endDate": end_date_str[:10] if end_date_str else None,
                "days_until_resolution": days_until,
                "date_score": round(date_score, 3),
                "skip_reason": "ev_too_low",
                "skip_detail": (
                    f"Best EV is {best_ev:.1%} on {best_side} — below the {EV_THRESHOLD:.0%} threshold. "
                    f"MACD model_p={model_p:.3f} vs market price={current_price:.3f}.{manifold_note}"
                ),
                "signal": False,
            })
            continue

        # Slippage check
        liquidity = float(raw.get("liquidity", 100))
        stake     = kelly_size(used_p, bet_price, state.current_bankroll)
        # Long-shot stake cap: Bucket B markets capped at $0.50
        if not in_bucket_a:
            stake = min(stake, LONGSHOT_STAKE_CAP)
        min_order_cost = max(5 * bet_price, 1.00)  # Polymarket min: 5 shares AND $1
        if stake < min_order_cost:
            if min_order_cost <= stake * 4:  # allow up to 4x Kelly to meet minimum
                log.info(f"  Bumping stake ${stake:.2f} → ${min_order_cost:.2f} to meet $1 minimum: {question[:40]}")
                stake = min_order_cost
            else:
                log.info(f"  Skip (min order ${min_order_cost:.2f} is >4x Kelly ${stake:.2f}): {question[:40]}")
            scan_market_log.append({
                "market_id": market_id, "question": question, "volume": volume,
                "price": round(current_price, 4), "model_p": round(model_p, 4),
                "manifold_p": round(manifold_p, 4) if manifold_p else None,
                "ev": round(best_ev, 4), "best_side": best_side,
                "skip_reason": "stake_too_small",
                "skip_detail": f"Kelly-sized stake ${stake:.2f} is below the ${MIN_STAKE_USD} minimum. Edge exists but bankroll too small for this bet.",
                "signal": False,
            })
            continue

        slip = lmsr_slippage(bet_price, stake, liquidity)
        if slip / bet_price > (SLIPPAGE_CAP_PCT / 100):
            log.info(f"  Skip (slippage {slip/bet_price*100:.1f}% > {SLIPPAGE_CAP_PCT}%): {question[:40]}")
            scan_market_log.append({
                "market_id": market_id, "question": question, "volume": volume,
                "price": round(current_price, 4), "model_p": round(model_p, 4),
                "manifold_p": round(manifold_p, 4) if manifold_p else None,
                "ev": round(best_ev, 4), "best_side": best_side,
                "skip_reason": "slippage",
                "skip_detail": f"Market impact {slip/bet_price*100:.1f}% exceeds the {SLIPPAGE_CAP_PCT}% slippage cap. Low liquidity (${liquidity:,.0f}).",
                "signal": False,
            })
            continue

        signals_found += 1
        if in_bucket_a:
            signals_from_a += 1
        token_id = token.get("token_id", "") if token else ""

        bucket_label = "A (mid-prob)" if in_bucket_a else "B (long-shot)"
        log.info(f"\n  ✦ SIGNAL FOUND  [{bucket_label}]")
        log.info(f"    Market    : {question}")
        log.info(f"    Side      : {bet_side}")
        log.info(f"    Price     : {bet_price:.3f}  |  Model p: {used_p:.3f}  |  EV: {ev:.4f}")
        log.info(f"    Manifold  : {manifold_p:.3f}" if manifold_p else "    Manifold  : no match")
        log.info(f"    Stake     : ${stake:.2f}  |  Slippage: {slip/bet_price*100:.2f}%")
        log.info(f"    Token ID  : {token_id[:20]}…")

        result = submit_order(
            client   = client,
            token_id = token_id,
            side     = "BUY",
            price    = bet_price,
            size_usd = stake,
            paper    = paper,
        )

        # Always blacklist this market to avoid re-evaluation (shows as Attempted in dashboard)
        if market_id not in state.traded_markets:
            state.traded_markets.append(market_id)

        if result.get("status") in ("paper", "submitted"):
            state.total_trades += 1
            _shares = math.ceil(stake / bet_price) if bet_price > 0 else 0
            _real_cost = round(_shares * bet_price, 6)
            state.active_positions[market_id] = {
                "question":  question,
                "side":      bet_side,
                "token_id":  token_id,
                "price":     bet_price,
                "shares":    _shares,
                "stake":     _real_cost,
                "ev":        ev,
                "paper":     paper,
            }
            if paper:
                state.current_bankroll -= stake
                state.current_bankroll  = max(0.0, state.current_bankroll)

            audit("signal", {
                "market_id":   market_id,
                "question":    question,
                "side":        bet_side,
                "price":       bet_price,
                "model_p":     used_p,
                "manifold_p":  manifold_p,
                "metaculus_p": metaculus_p,
                "ev":          ev,
                "stake":       stake,
                "token_id":    token_id,
                "paper":       paper,
            })
            audit("order_placed", {
                "market_id": market_id,
                "stake":     stake,
                "result":    result,
            })
            scan_market_log.append({
                "market_id": market_id, "question": question, "volume": volume,
                "price": round(bet_price, 4), "model_p": round(used_p, 4),
                "manifold_p": round(manifold_p, 4) if manifold_p else None,
                "ev": round(ev, 4), "best_side": bet_side,
                "stake": round(stake, 2),
                "skip_reason": None,
                "skip_detail": f"Signal placed! {bet_side} at {bet_price:.3f} — EV {ev:.1%}, stake ${stake:.2f}.",
                "signal": True,
            })
        else:
            log.warning(f"  Order failed ({result.get('reason','?')}) — market skipped for this session")
            scan_market_log.append({
                "market_id": market_id, "question": question, "volume": volume,
                "price": round(bet_price, 4), "model_p": round(used_p, 4),
                "manifold_p": round(manifold_p, 4) if manifold_p else None,
                "ev": round(ev, 4), "best_side": bet_side,
                "skip_reason": "order_failed",
                "skip_detail": f"Signal found (EV {ev:.1%}) but order submission failed: {result.get('reason','unknown')}.",
                "signal": False,
            })

        save_state(state)

    if signals_found == 0:
        log.info("  No EV signals found this scan.")

    save_state(state)

    # Write per-market scan data for dashboard
    try:
        with open("scan_markets.json", "w") as f:
            json.dump({
                "ts":      datetime.now(timezone.utc).isoformat(),
                "markets": scan_market_log,
            }, f)
    except Exception as e:
        log.debug(f"Could not write scan_markets.json: {e}")

    duration_sec = round(time.time() - scan_start_time, 1)
    audit("scan_complete", {
        "markets_fetched":      len(raw_markets),
        "markets_skipped_traded":  n_skipped_traded,
        "markets_skipped_history": n_skipped_history,
        "markets_skipped_price":   n_skipped_price,
        "markets_evaluated":    n_evaluated,
        "manifold_matched":     n_manifold_match,
        "manifold_rejected":    n_manifold_rejected,
        "signals_found":        signals_found,
        "duration_sec":         duration_sec,
        "paper":                paper,
    })

    # Daily summary at midnight UTC (within one scan window)
    now_utc = datetime.now(timezone.utc)
    if now_utc.hour == 0 and now_utc.minute < 6:
        try:
            today = now_utc.date().isoformat()
            trades_today, pnl_today, signals_today, scans_today = 0, 0.0, 0, 0
            with open(LOG_FILE) as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        if ev.get("ts", "").startswith(today):
                            if ev.get("event") == "order_placed":
                                trades_today += 1
                            elif ev.get("event") == "position_resolved":
                                pnl_today += ev.get("data", {}).get("pnl", 0)
                            elif ev.get("event") == "signal":
                                signals_today += 1
                            elif ev.get("event") == "scan_complete":
                                scans_today += 1
                    except Exception:
                        pass
            audit("daily_summary", {
                "date":             today,
                "trades_today":     trades_today,
                "pnl_today":        round(pnl_today, 4),
                "scans_today":      scans_today,
                "signals_today":    signals_today,
                "bankroll":         state.current_bankroll,
                "paper":            paper,
            })
            log.info(f"  [daily] {today}: {trades_today} trades | PnL ${pnl_today:+.2f} | {signals_today} signals | bankroll ${state.current_bankroll:.2f}")
        except Exception as e:
            log.debug(f"Daily summary failed: {e}")

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
            if not FUNDER_ADDRESS:
                log.error("FUNDER_ADDRESS not set in .env — cannot submit live orders.")
                log.error("Set FUNDER_ADDRESS to your Polymarket proxy wallet address.")
                sys.exit(1)
            client = ClobClient(
                CLOB_HOST,
                key            = os.environ["PRIVATE_KEY"],
                chain_id       = CHAIN_ID,
                signature_type = 1,
                funder         = FUNDER_ADDRESS,
            )
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            log.info(f"  [✓] Authenticated with Polymarket CLOB (funder={FUNDER_ADDRESS[:10]}…)")
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
