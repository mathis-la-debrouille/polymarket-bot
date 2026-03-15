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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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
    print("[!] py-clob-client not found. Install with: pip install py-clob-client")
    print("    Running in paper-mode only until installed.\n")

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
GAMMA_API    = "https://gamma-api.polymarket.com"
CLOB_HOST    = "https://clob.polymarket.com"
CHAIN_ID     = 137   # Polygon mainnet

# Risk limits
EV_THRESHOLD      = 0.06    # minimum EV to place a bet (higher than backtest — real fees hurt more)
MAX_STAKE_USD     = 2.00    # absolute max bet size in USD (hard cap for $20 bankroll)
MIN_STAKE_USD     = 0.50    # minimum order size (Polymarket rejects very small orders)
MAX_DRAWDOWN_PCT  = 30.0    # kill switch: stop trading if down this % from peak
KELLY_FRACTION    = 0.20    # conservative fractional Kelly
MAX_MARKETS_SCAN  = 50      # max markets to evaluate per run
MIN_VOLUME_USD    = 20_000  # skip thin markets
SLIPPAGE_CAP_PCT  = 2.0     # skip if LMSR price impact > 2%

SCAN_INTERVAL_SEC = 300     # default: scan every 5 minutes
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
    starting_bankroll: float = 20.0
    current_bankroll:  float = 20.0
    peak_bankroll:     float = 20.0
    total_trades:      int   = 0
    total_pnl:         float = 0.0
    active_positions:  Dict  = None   # token_id → {question, side, price, size, stake}
    traded_markets:    List  = None   # market IDs already traded (avoid re-entry)

    def __post_init__(self):
        if self.active_positions is None:
            self.active_positions = {}
        if self.traded_markets is None:
            self.traded_markets = []


def load_state(bankroll: float) -> BotState:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                d = json.load(f)
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
    with open(STATE_FILE, "w") as f:
        json.dump(asdict(state), f, indent=2)


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


def fetch_price_history(condition_id: str, fidelity: int = 60) -> List[Dict]:
    """Fetch recent price history for a market's YES token."""
    if not condition_id:
        return []
    try:
        resp = requests.get(
            f"{CLOB_HOST}/prices-history",
            params={"market": condition_id, "fidelity": fidelity},
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
# SIGNAL ENGINE  (same logic as backtest, tuned for live use)
# ──────────────────────────────────────────────────────────────────
def model_probability(history: List[Dict]) -> Optional[float]:
    """
    MACD momentum model (same as backtest framework).
    Replace this function with your real signal source.
    """
    if len(history) < 10:
        return None
    prices = [h["p"] for h in history]
    fast, slow = prices[0], prices[0]
    for p in prices:
        fast = 0.25 * p + 0.75 * fast
        slow = 0.06 * p + 0.94 * slow
    momentum = fast - slow
    model_p  = prices[-1] + momentum * 0.6
    return float(np.clip(model_p, 0.02, 0.98))


def compute_ev(model_p: float, market_price: float, fee: float = 0.02) -> float:
    """EV = (model_p - market_price) * (1/market_price) - fee"""
    if not (0 < market_price < 1):
        return 0.0
    return (model_p - market_price) * (1.0 / market_price) - fee


def kelly_size(model_p: float, market_price: float, bankroll: float) -> float:
    """Fractional Kelly stake in USD, hard-capped at MAX_STAKE_USD."""
    if not (0 < market_price < 1):
        return 0.0
    odds = (1.0 - market_price) / market_price
    f    = (model_p * odds - (1 - model_p)) / odds
    f   *= KELLY_FRACTION
    stake = max(0.0, f) * bankroll
    return min(stake, MAX_STAKE_USD)


def lmsr_slippage(current_price: float, trade_usd: float, liquidity: float) -> float:
    """Estimate LMSR slippage for a USD trade at given liquidity depth."""
    b = max(liquidity, 10)
    shares = trade_usd / current_price  # approximate share count
    p = np.clip(current_price, 1e-6, 1 - 1e-6)
    q_yes = b * np.log(p / (1 - p))
    denom_before = np.exp(q_yes / b) + 1
    denom_after  = np.exp((q_yes + shares) / b) + 1
    p_after = np.exp((q_yes + shares) / b) / denom_after
    return abs(p_after - p)


# ──────────────────────────────────────────────────────────────────
# TRADE EXECUTION
# ──────────────────────────────────────────────────────────────────
def submit_order(
    client: "ClobClient",
    token_id:    str,
    side:        str,   # "BUY" or "SELL"
    price:       float,
    size_usd:    float,
    paper:       bool,
) -> Dict:
    """
    Submit a limit order to Polymarket CLOB.
    In paper mode, logs the intended trade without touching the API.
    """
    side_const = BUY if side == "BUY" else SELL

    if paper:
        log.info(f"  [PAPER] Would {side} {size_usd:.2f} USD @ {price:.3f} (token {token_id[:12]}…)")
        return {"status": "paper", "price": price, "size": size_usd}

    if not CLOB_AVAILABLE:
        log.warning("  [!] py-clob-client not available — cannot submit real order")
        return {"status": "error", "reason": "clob_unavailable"}

    try:
        order_args = OrderArgs(
            token_id = token_id,
            price    = round(price, 4),
            size     = round(size_usd, 2),
            side     = side_const,
        )
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
    """Returns True if the bot should halt trading due to drawdown."""
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
# MAIN SCAN LOOP
# ──────────────────────────────────────────────────────────────────
def run_scan(client: Optional["ClobClient"], state: BotState, paper: bool) -> None:
    """One full scan: fetch markets → evaluate → place trades."""
    log.info(f"{'─'*55}")
    log.info(f"Scan started | bankroll=${state.current_bankroll:.2f} | "
             f"{'PAPER' if paper else '🔴 LIVE'}")
    log.info(f"{'─'*55}")

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

        # Skip markets we've already traded
        if market_id in state.traded_markets:
            continue

        # Extract YES token ID (needed for CLOB orders)
        try:
            tokens = json.loads(raw.get("tokens", "[]")) if isinstance(raw.get("tokens"), str) \
                     else raw.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
            no_token  = next((t for t in tokens if t.get("outcome", "").upper() == "NO"),  None)
        except Exception:
            yes_token, no_token = None, None

        # Current midprice
        condition_id = raw.get("conditionId", "")
        history = fetch_price_history(condition_id)
        if len(history) < 10:
            continue

        current_price = history[-1]["p"]
        if not (0.05 < current_price < 0.95):
            continue  # skip near-resolved

        # Model probability
        model_p = model_probability(history)
        if model_p is None:
            continue

        # Evaluate both YES and NO
        ev_yes = compute_ev(model_p,       current_price)
        ev_no  = compute_ev(1 - model_p,   1 - current_price)

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
            continue  # no edge

        # Slippage check
        liquidity = float(raw.get("liquidity", 100))
        stake     = kelly_size(used_p, bet_price, state.current_bankroll)
        if stake < MIN_STAKE_USD:
            continue  # Kelly says too small to bother

        slip = lmsr_slippage(bet_price, stake, liquidity)
        if slip / bet_price > (SLIPPAGE_CAP_PCT / 100):
            log.info(f"  Skip (slippage {slip/bet_price*100:.1f}% > {SLIPPAGE_CAP_PCT}%): {question}")
            continue

        signals_found += 1
        token_id = token.get("token_id", "") if token else ""

        log.info(f"\n  ✦ SIGNAL FOUND")
        log.info(f"    Market    : {question}")
        log.info(f"    Side      : {bet_side}")
        log.info(f"    Price     : {bet_price:.3f}  |  Model p: {used_p:.3f}  |  EV: {ev:.4f}")
        log.info(f"    Stake     : ${stake:.2f}  |  Slippage: {slip/bet_price*100:.2f}%")
        log.info(f"    Token ID  : {token_id[:20]}…")

        audit("signal", {
            "market_id": market_id,
            "question":  question,
            "side":      bet_side,
            "price":     bet_price,
            "model_p":   used_p,
            "ev":        ev,
            "stake":     stake,
            "token_id":  token_id,
            "paper":     paper,
        })

        # Execute
        result = submit_order(
            client    = client,
            token_id  = token_id,
            side      = "BUY",
            price     = bet_price,
            size_usd  = stake,
            paper     = paper,
        )

        if result.get("status") in ("paper", "submitted"):
            # Update state (in paper mode, optimistically record as filled)
            state.traded_markets.append(market_id)
            state.total_trades += 1
            if paper:
                # Paper: we don't know outcome yet — track as pending
                state.active_positions[market_id] = {
                    "question":  question,
                    "side":      bet_side,
                    "token_id":  token_id,
                    "price":     bet_price,
                    "stake":     stake,
                    "ev":        ev,
                    "paper":     True,
                }
                # Simulate bankroll debit in paper mode
                state.current_bankroll -= stake
                state.current_bankroll = max(0.0, state.current_bankroll)

            audit("order_placed", {
                "market_id": market_id,
                "stake":     stake,
                "result":    result,
            })

        save_state(state)

    if signals_found == 0:
        log.info("  No EV signals found this scan.")

    save_state(state)

    log.info(f"\n  Bankroll: ${state.current_bankroll:.2f} | "
             f"Trades: {state.total_trades} | Active positions: {len(state.active_positions)}")


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────
def main():
    global MAX_STAKE_USD, EV_THRESHOLD

    parser = argparse.ArgumentParser(description="Polymarket Live Trading Bot")
    parser.add_argument("--live",       action="store_true",              help="Enable real order submission (default: paper mode)")
    parser.add_argument("--once",       action="store_true",              help="Run one scan then exit")
    parser.add_argument("--bankroll",   type=float, default=20.0,         help="Starting bankroll in USD (default: 20)")
    parser.add_argument("--interval",   type=int,   default=SCAN_INTERVAL_SEC, help="Seconds between scans (default: 300)")
    parser.add_argument("--max-stake",  type=float, default=MAX_STAKE_USD,  help="Max USD per trade (default: 2.0)")
    parser.add_argument("--ev-min",     type=float, default=EV_THRESHOLD, help="Minimum EV to trade (default: 0.06)")
    args = parser.parse_args()

    paper = not args.live

    # Warn loudly if live mode
    if not paper:
        if not CLOB_AVAILABLE:
            log.error("Cannot run in live mode: py-clob-client not installed.")
            log.error("Run: pip install py-clob-client")
            sys.exit(1)
        private_key = os.environ.get("PRIVATE_KEY", "")
        if not private_key or not private_key.startswith("0x"):
            log.error("PRIVATE_KEY not set or invalid in .env file.")
            log.error("Add:  PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE  to your .env file")
            sys.exit(1)
        log.warning("=" * 55)
        log.warning("  🔴  LIVE MODE — real money will be spent")
        log.warning(f"  Bankroll: ${args.bankroll:.2f}")
        log.warning(f"  Max per trade: ${args.max_stake:.2f}")
        log.warning(f"  Kill switch: -{MAX_DRAWDOWN_PCT:.0f}% drawdown")
        log.warning("=" * 55)
        time.sleep(3)   # brief pause to allow Ctrl+C abort
    else:
        log.info("=" * 55)
        log.info("  📄  PAPER MODE — no real orders will be placed")
        log.info("  (run with --live to trade real money)")
        log.info("=" * 55)

    # Override config from args
    MAX_STAKE_USD = args.max_stake
    EV_THRESHOLD  = args.ev_min

    # Authenticate CLOB client
    client = None
    if not paper and CLOB_AVAILABLE:
        try:
            client = ClobClient(
                CLOB_HOST,
                key      = os.environ["PRIVATE_KEY"],
                chain_id = CHAIN_ID,
            )
            # Derive or load L2 API credentials
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            log.info(f"  [✓] Authenticated with Polymarket CLOB")
            audit("session_start", {"mode": "live", "bankroll": args.bankroll})
        except Exception as e:
            log.error(f"Authentication failed: {e}")
            sys.exit(1)
    else:
        audit("session_start", {"mode": "paper", "bankroll": args.bankroll})

    # Load or create persistent state
    state = load_state(args.bankroll)

    # Scan loop
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
