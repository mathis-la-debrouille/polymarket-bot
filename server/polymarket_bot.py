#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║              Polymarket Bot — Template                           ║
║                                                                  ║
║  Handles all Polymarket infrastructure:                          ║
║    • Wallet management & live USDC balance sync                  ║
║    • Order placement via Polymarket CLOB API                     ║
║    • Position tracking, resolution, and P&L accounting           ║
║    • Stop-loss and kill-switch safety                            ║
║    • Structured audit logging (JSONL)                            ║
║    • Paper / live trading modes                                  ║
║                                                                  ║
║  TO ADD YOUR STRATEGY:                                           ║
║    Implement your signal logic in run_scan() where indicated.    ║
║    See the marked section and the docstring for the interface.   ║
║                                                                  ║
║  SAFETY DEFAULTS:                                                ║
║    • Paper trading ON by default (--live flag required to trade) ║
║    • Hard kill switch at configurable max drawdown               ║
║    • Per-trade stake cap (default: $5 max / trade)               ║
║    • Full audit log written to bot_log.jsonl                     ║
╚══════════════════════════════════════════════════════════════════╝

SETUP:
    pip install -r requirements.txt
    cp .env.example .env   # fill in your keys

USAGE:
    # Paper mode (no real money, just shows what would be traded):
    python polymarket_bot.py

    # Live mode (REAL orders, REAL money):
    python polymarket_bot.py --live

    # Run once then exit (cron-friendly):
    python polymarket_bot.py --live --once

    # Adjust scan interval and bankroll:
    python polymarket_bot.py --live --interval 300 --bankroll 100
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import numpy as np

# ──────────────────────────────────────────────────────────────────
# DEPENDENCIES
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
    BUY  = "BUY"
    SELL = "SELL"
    print("[!] py-clob-client not found. Install with: pip install py-clob-client")
    print("    Running in paper-mode only until installed.\n")

try:
    from signal_updown import (
        compute_updown_signal,
        is_updown_market,
        start_rtds_stream,
    )
    SIGNAL_AVAILABLE = True
except ImportError:
    SIGNAL_AVAILABLE = False
    log_bootstrap = logging.getLogger(__name__)
    log_bootstrap.warning("[!] signal_updown.py not found — bot will run but place no trades")

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
GAMMA_API  = "https://gamma-api.polymarket.com"
CLOB_HOST  = "https://clob.polymarket.com"
CHAIN_ID   = 137   # Polygon mainnet

# Proxy wallet — set FUNDER_ADDRESS in .env to enable live orders.
# This is the address of the Polymarket proxy wallet that holds USDC.
# Find yours via: call getPolyProxyWalletAddress(your_eoa) on the CTFExchange contract
# CTFExchange: 0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E (selector 0xedef7d8e)
FUNDER_ADDRESS = os.environ.get("FUNDER_ADDRESS", "")

# ── Strategy parameters — tune these for your algorithm ──────────
EV_THRESHOLD      = 0.05    # minimum expected value to place a trade
MAX_STAKE_USD     = 1.00    # max USD stake per trade
MIN_STAKE_USD     = 1.00    # Polymarket minimum per order
KELLY_FRACTION    = 0.25    # fraction of Kelly to use (0.25 = quarter-Kelly)
MAX_DRAWDOWN_PCT  = 30.0    # kill switch: stop if down 30% from peak

# ── Bot parameters ────────────────────────────────────────────────
MAX_TRADED_MARKETS = 200    # cap on traded_markets history
SCAN_INTERVAL_SEC  = 60     # seconds between scans
LOG_FILE           = "bot_log.jsonl"
STATE_FILE         = "bot_state.json"

# ──────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def audit(event: str, data: dict) -> None:
    """Append a structured event to the JSONL audit log."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **data}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────────
# STATE (persisted across runs)
# ──────────────────────────────────────────────────────────────────
@dataclass
class BotState:
    starting_bankroll:    float = 20.0
    current_bankroll:     float = 20.0
    peak_bankroll:        float = 20.0
    total_trades:         int   = 0
    total_pnl:            float = 0.0
    active_positions:     Dict  = None   # market_id → position dict
    traded_markets:       List  = None   # market IDs already traded (avoid re-entry)
    resolved_positions:   List  = None   # last 100 resolved positions
    clob_cash:            float = 0.0    # last known on-chain USDC cash
    scans_since_sl_check: int   = 0
    daily_start_bankroll: float = 0.0   # bankroll at 6 AM — daily drawdown reference
    daily_reset_date:     str   = ""    # YYYY-MM-DD of last daily reset

    def __post_init__(self):
        if self.active_positions  is None: self.active_positions  = {}
        if self.traded_markets    is None: self.traded_markets    = []
        if self.resolved_positions is None: self.resolved_positions = []


def load_state(bankroll: float) -> BotState:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                d = json.load(f)
            d.setdefault("resolved_positions", [])
            d.setdefault("clob_cash", 0.0)
            d.setdefault("scans_since_sl_check", 0)
            d.setdefault("daily_start_bankroll", 0.0)
            d.setdefault("daily_reset_date", "")
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
    """Atomic write: write to temp file then rename so api_server never reads a partial file."""
    if len(state.traded_markets) > MAX_TRADED_MARKETS:
        state.traded_markets = state.traded_markets[-MAX_TRADED_MARKETS:]
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(state), f, indent=2)
    os.replace(tmp, STATE_FILE)


# ──────────────────────────────────────────────────────────────────
# POSITION RESOLUTION
# ──────────────────────────────────────────────────────────────────
def resolve_positions(state: BotState) -> None:
    """
    Check every open position against the Gamma API.
    If a market has resolved, compute P&L, update bankroll, and archive it.
    Called at the top of every scan, before fetching new markets.
    """
    if not state.active_positions:
        return

    to_resolve = list(state.active_positions.items())
    resolved_count = 0
    net_pnl = 0.0

    for market_id, pos in to_resolve:
        try:
            resp = requests.get(f"{GAMMA_API}/markets/{market_id}", timeout=8)
            resp.raise_for_status()
            market = resp.json()
        except Exception as e:
            log.debug(f"  Could not fetch resolution for {market_id}: {e}")
            continue

        resolved = market.get("resolved", False)
        closed   = market.get("closed",   False)
        if not (resolved or closed):
            continue

        # Parse outcomePrices — Gamma API returns a JSON string like '["0.95","0.05"]'
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
        winning_index = next((i for i, p in enumerate(outcome_prices) if p > 0.9), None)
        if winning_index is None:
            log.debug(f"  {market_id}: closed but no decisive outcome yet")
            continue

        # YES = index 0, NO = index 1
        bet_side    = pos.get("side", "YES")
        bet_index   = 0 if bet_side == "YES" else 1
        won         = (winning_index == bet_index)
        entry_price = pos.get("price", 0.5)
        stake       = pos.get("stake", 0.0)

        pnl = stake * (1.0 / entry_price - 1.0) if won else -stake

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

        state.resolved_positions.append(resolved_record)
        if len(state.resolved_positions) > 100:
            state.resolved_positions = state.resolved_positions[-100:]

        del state.active_positions[market_id]
        if market_id in state.traded_markets:
            state.traded_markets.remove(market_id)

        audit("position_resolved", resolved_record)
        log.info(f"  [RESOLVED] {pos.get('question','?')[:55]}  "
                 f"{'WON' if won else 'LOST'}  pnl={pnl:+.4f}")
        resolved_count += 1
        net_pnl += pnl

    if resolved_count > 0:
        log.info(f"  Resolved {resolved_count} position(s) | net PnL: {net_pnl:+.2f}")
        save_state(state)


# ──────────────────────────────────────────────────────────────────
# STOP-LOSS CHECK
# ──────────────────────────────────────────────────────────────────
def check_stop_loss(state: BotState, clob_client: Optional["ClobClient"]) -> None:
    """
    Check open positions for >30% loss; sell if edge is gone.
    Runs every 3 scans to avoid excessive API calls.
    """
    state.scans_since_sl_check = getattr(state, "scans_since_sl_check", 0) + 1
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

        if not token_id or entry_price <= 0:
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

        log.info(f"  [SL] {question[:50]} — loss {loss_pct:.0%}, edge gone. Selling…")

        if clob_client is not None:
            try:
                sell_price = max(0.01, curr_price - 0.01)
                sell_side  = BUY if side == "NO" else SELL
                order_args = OrderArgs(
                    token_id=token_id,
                    price=round(sell_price, 4),
                    size=float(shares),
                    side=sell_side,
                )
                signed_order = clob_client.create_order(order_args)
                clob_client.post_order(signed_order)
                audit("stop_loss_triggered", {
                    "market_id":     mid,
                    "question":      question,
                    "entry_price":   entry_price,
                    "current_price": curr_price,
                    "loss_pct":      round(loss_pct, 4),
                    "shares_sold":   shares,
                })
                del state.active_positions[mid]
                save_state(state)
                log.info(f"  [SL] Sold {shares} shares of {question[:40]}")
            except Exception as e:
                log.error(f"  [SL] Sell order failed: {e}")
        else:
            # Paper mode: just remove the position
            log.info(f"  [SL] Paper mode — removing stale position: {question[:40]}")
            del state.active_positions[mid]
            save_state(state)


# ──────────────────────────────────────────────────────────────────
# DAILY DRAWDOWN RESET (6 AM UTC)
# ──────────────────────────────────────────────────────────────────
DAILY_RESET_HOUR = 6  # UTC

def reset_daily_drawdown_if_needed(state: BotState) -> None:
    """Reset the drawdown reference to current bankroll at 6 AM UTC each day."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    if now.hour >= DAILY_RESET_HOUR and state.daily_reset_date != today:
        prev = state.daily_start_bankroll
        state.daily_start_bankroll = state.current_bankroll
        state.daily_reset_date = today
        log.info(
            f"  [DAY RESET] Daily drawdown reference: ${prev:.2f} → ${state.current_bankroll:.2f}"
        )
        audit("daily_reset", {
            "date": today,
            "bankroll": state.current_bankroll,
            "prev_reference": prev,
        })
        save_state(state)


# ──────────────────────────────────────────────────────────────────
# KILL SWITCH
# ──────────────────────────────────────────────────────────────────
def check_kill_switch(state: BotState) -> bool:
    # Use today's 6 AM bankroll as reference; fall back to peak if not yet set
    ref = state.daily_start_bankroll if state.daily_start_bankroll > 0 else state.peak_bankroll
    if ref <= 0:
        return False
    drawdown_pct = (ref - state.current_bankroll) / ref * 100
    if drawdown_pct >= MAX_DRAWDOWN_PCT:
        log.error(
            f"KILL SWITCH TRIGGERED: drawdown={drawdown_pct:.1f}% from today's 6AM ref "
            f"${ref:.2f} (limit={MAX_DRAWDOWN_PCT}%). Halting all trading."
        )
        audit("kill_switch", {
            "drawdown_pct": drawdown_pct,
            "reference_bankroll": ref,
            "bankroll": state.current_bankroll,
        })
        return True
    return False


# ──────────────────────────────────────────────────────────────────
# REAL BALANCE SYNC
# ──────────────────────────────────────────────────────────────────
def sync_real_balance(state: BotState, clob_client: Optional["ClobClient"] = None) -> None:
    """Fetch live USDC balance from the CLOB and sync it to state."""
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        if clob_client is None:
            pk = os.environ.get("PRIVATE_KEY", "")
            if not pk:
                return
            clob_client = ClobClient(CLOB_HOST, key=pk, chain_id=CHAIN_ID, signature_type=1)
            creds = clob_client.create_or_derive_api_creds()
            clob_client.set_api_creds(creds)

        bal = clob_client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=1)
        )
        clob_cash = float(bal.get("balance", 0)) / 1_000_000
        if clob_cash < 0:
            return

        pos_value = 0.0
        for mid, pos in state.active_positions.items():
            token_id = pos.get("token_id", "")
            shares   = pos.get("shares", 0)
            fallback = pos.get("price", 0)
            if shares <= 0:
                continue
            try:
                r = requests.get("https://clob.polymarket.com/midpoint",
                                 params={"token_id": token_id}, timeout=5)
                mid_price = float(r.json().get("mid", fallback))
            except Exception:
                mid_price = fallback
            pos_value += shares * mid_price

        portfolio = clob_cash + pos_value
        old = state.current_bankroll
        state.current_bankroll = portfolio
        state.clob_cash        = clob_cash
        state.peak_bankroll    = max(state.peak_bankroll, portfolio)
        log.info(f"  [✓] Balance synced: cash=${clob_cash:.4f}  "
                 f"positions=${pos_value:.4f}  portfolio=${portfolio:.4f}  (was ${old:.4f})")
        audit("balance_sync", {
            "clob_cash":   clob_cash,
            "pos_value":   pos_value,
            "portfolio":   portfolio,
            "was_tracked": old,
        })
        save_state(state)
    except Exception as e:
        log.debug(f"  Balance sync skipped ({e}) — using tracked value")


# ──────────────────────────────────────────────────────────────────
# MARKET DATA HELPERS
# ──────────────────────────────────────────────────────────────────
def get_orderbook_midprice(token_id: str) -> Optional[float]:
    """Get current best bid/ask midprice from the CLOB orderbook."""
    try:
        resp = requests.get(f"{CLOB_HOST}/midpoint",
                            params={"token_id": token_id}, timeout=8)
        resp.raise_for_status()
        return float(resp.json().get("mid", 0))
    except Exception:
        return None


def fetch_markets(active: bool = True, limit: int = 100,
                  order: str = "volume", **kwargs) -> List[dict]:
    """
    Generic helper to fetch markets from the Gamma API.

    Args:
        active:  only return active/open markets
        limit:   max results (up to 500)
        order:   sort field ("volume", "liquidity", "end_date_min", ...)
        **kwargs: any additional Gamma API filter params

    Returns:
        list of market dicts
    """
    try:
        params = {
            "active": "true" if active else "false",
            "closed": "false",
            "limit":  limit,
            "order":  order,
            "ascending": "false",
            **{k: str(v) for k, v in kwargs.items()},
        }
        resp = requests.get(f"{GAMMA_API}/markets", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else []
    except Exception as e:
        log.warning(f"fetch_markets failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────────
# EV & SIZING UTILITIES
# ──────────────────────────────────────────────────────────────────
def compute_ev(model_p: float, market_price: float, fee: float = 0.02) -> float:
    """
    Expected value of a binary bet.
    EV = model_p / market_price - 1 - fee
    Returns 0 if market_price is out of (0, 1).
    """
    if not (0 < market_price < 1):
        return 0.0
    return model_p / market_price - 1.0 - fee


def kelly_size(model_p: float, market_price: float, bankroll: float) -> float:
    """
    Fractional Kelly stake in USD, hard-capped at MAX_STAKE_USD.
    Uses KELLY_FRACTION for partial Kelly sizing.
    """
    if not (0 < market_price < 1):
        return 0.0
    odds  = (1.0 - market_price) / market_price
    f     = (model_p * odds - (1 - model_p)) / odds * KELLY_FRACTION
    stake = max(0.0, f) * bankroll
    return min(stake, MAX_STAKE_USD)


# ──────────────────────────────────────────────────────────────────
# ORDER EXECUTION
# ──────────────────────────────────────────────────────────────────
def submit_order(
    client:   Optional["ClobClient"],
    token_id: str,
    side:     str,
    price:    float,
    size_usd: float,
    paper:    bool,
) -> Dict:
    """
    Submit an order to the Polymarket CLOB.

    Args:
        client:   authenticated ClobClient (None in paper mode)
        token_id: Polymarket token ID (from clobTokenIds)
        side:     "BUY" or "SELL"
        price:    limit price (0–1)
        size_usd: desired USD spend
        paper:    if True, simulate only (no real order)

    Returns:
        dict with "status" key: "paper" | "submitted" | "error"
    """
    side_const = BUY if side == "BUY" else SELL

    if paper:
        log.info(f"  [PAPER] Would {side} ${size_usd:.2f} @ {price:.3f} (token {token_id[:12]}…)")
        return {"status": "paper", "price": price, "size": size_usd}

    if not CLOB_AVAILABLE:
        log.warning("  [!] py-clob-client not available — cannot submit real order")
        return {"status": "error", "reason": "clob_unavailable"}

    try:
        MIN_POLY_SHARES = 5
        MIN_ORDER_USD   = 1.00
        shares = max(math.ceil(size_usd / price), MIN_POLY_SHARES,
                     math.ceil(MIN_ORDER_USD / price))
        actual_cost = shares * price
        if actual_cost > max(size_usd * 3, MIN_ORDER_USD * 1.5):
            log.warning(f"  [!] Skipping: min-order cost ${actual_cost:.2f} is >3x target ${size_usd:.2f}")
            return {"status": "error",
                    "reason": f"min_order_cost_too_high (${actual_cost:.2f} vs ${size_usd:.2f})"}

        order_args   = OrderArgs(token_id=token_id, price=round(price, 4),
                                 size=shares, side=side_const)
        signed_order = client.create_order(order_args)
        resp         = client.post_order(signed_order)
        log.info(f"  [LIVE] Order submitted: {resp}")

        # Confirm fill — cancel if not matched (avoids ghost positions)
        order_id = (resp or {}).get("orderID") or (resp or {}).get("id", "")
        if order_id:
            time.sleep(4)
            try:
                status    = client.get_order(order_id)
                matched   = float(status.get("size_matched", 0))
                remaining = float(status.get("size_remaining", shares))
                if matched < 1:
                    client.cancel(order_id)
                    log.warning(f"  [!] Order {order_id[:12]} unmatched — cancelled")
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


def purge_legacy_positions(state: BotState) -> None:
    """Remove positions that have no entry_time — these are legacy/manual positions
    not placed by the current bot and will never be resolved automatically."""
    stale = [mid for mid, pos in state.active_positions.items()
             if not pos.get("entry_time")]
    if stale:
        for mid in stale:
            q = state.active_positions[mid].get("question", mid)[:50]
            log.info(f"  [PURGE] Removing legacy position: {q}")
            del state.active_positions[mid]
        save_state(state)
        log.info(f"  [PURGE] Removed {len(stale)} legacy position(s)")


# ──────────────────────────────────────────────────────────────────
# MAIN SCAN LOOP
# ──────────────────────────────────────────────────────────────────
def run_scan(client: Optional["ClobClient"], state: BotState, paper: bool) -> None:
    """
    One full scan cycle:
      1. Resolve any pending positions
      2. Check stop-losses
      3. Sync real USDC balance
      4. Check kill switch
      5. Run your strategy (fetch markets → evaluate → place trades)
      6. Emit scan_complete audit event

    YOUR STRATEGY INTERFACE
    ───────────────────────
    Fetch the markets you care about, compute a probability estimate,
    then call the helpers below. Minimal example:

        markets = fetch_markets(limit=50, tag="crypto")
        for market in markets:
            if market["id"] in state.traded_markets:
                continue

            token_id = json.loads(market.get("clobTokenIds", "[]"))[0]
            price    = get_orderbook_midprice(token_id) or 0.5
            model_p  = your_signal(market)          # your estimate of YES probability
            ev       = compute_ev(model_p, price)
            stake    = kelly_size(model_p, price, state.current_bankroll)

            if ev >= EV_THRESHOLD and stake >= MIN_STAKE_USD:
                result = submit_order(client, token_id, "BUY", price, stake, paper)
                if result["status"] in ("paper", "submitted"):
                    state.active_positions[market["id"]] = {
                        "question":   market.get("question", "?"),
                        "side":       "YES",
                        "token_id":   token_id,
                        "price":      price,
                        "stake":      stake,
                        "shares":     math.ceil(stake / price),
                        "ev":         ev,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "paper":      paper,
                        "signal_type": "my_strategy",
                    }
                    state.traded_markets.append(market["id"])
                    state.total_trades    += 1
                    state.current_bankroll = max(0.0, state.current_bankroll - stake)
                    audit("signal", {
                        "market_id":   market["id"],
                        "question":    market.get("question", "?")[:80],
                        "side":        "YES",
                        "price":       price,
                        "model_p":     model_p,
                        "ev":          ev,
                        "stake":       stake,
                        "signal_type": "my_strategy",
                        "paper":       paper,
                    })
                    save_state(state)
    """
    scan_start = time.time()
    log.info("─" * 55)
    log.info(f"Scan started | bankroll=${state.current_bankroll:.2f} | "
             f"{'PAPER' if paper else 'LIVE'}")
    log.info("─" * 55)

    purge_legacy_positions(state)
    resolve_positions(state)
    check_stop_loss(state, client)
    sync_real_balance(state, clob_client=client)
    reset_daily_drawdown_if_needed(state)

    if check_kill_switch(state):
        return

    # ══════════════════════════════════════════════════════════════
    # STRATEGY — Crypto Up/Down markets via signal_updown.py
    # ══════════════════════════════════════════════════════════════
    signals_found = 0

    if not SIGNAL_AVAILABLE:
        log.warning("  signal_updown not available — no trades this scan")
    else:
        markets = fetch_markets(
            active=True, limit=500,
            order="volume", tag="crypto",
        )
        updown_markets = [m for m in markets if is_updown_market(m.get("question", ""))]
        log.info(f"  Found {len(updown_markets)} Up/Down markets")

        for raw in updown_markets:
            market_id = raw.get("id", "")
            question  = raw.get("question", "?")

            if market_id in state.traded_markets:
                continue

            # Parse clobTokenIds (YES = index 0, NO = index 1)
            try:
                clob_ids = raw.get("clobTokenIds") or "[]"
                if isinstance(clob_ids, str):
                    clob_ids = json.loads(clob_ids)
            except Exception:
                clob_ids = []
            if len(clob_ids) < 2:
                continue

            yes_token_id = clob_ids[0]
            no_token_id  = clob_ids[1]

            # Get YES midprice
            try:
                op = raw.get("outcomePrices") or "[]"
                if isinstance(op, str):
                    op = json.loads(op)
                yes_midprice = float(op[0]) if op else 0.0
            except Exception:
                yes_midprice = 0.0
            if yes_midprice <= 0:
                yes_midprice = get_orderbook_midprice(yes_token_id) or 0.50
            if not (0.03 < yes_midprice < 0.97):
                continue

            # Compute signal
            sig = compute_updown_signal(raw, yes_midprice)
            if sig is None:
                continue

            # ── Spread Arb ───────────────────────────────────────────
            if sig.get("strategy") == "ARB":
                stake = min(state.current_bankroll * 0.10, MAX_STAKE_USD)
                if stake >= MIN_STAKE_USD:
                    no_midprice = 1.0 - yes_midprice
                    r_yes = submit_order(client, yes_token_id, "BUY",
                                         yes_midprice, stake / 2, paper)
                    r_no  = submit_order(client, no_token_id,  "BUY",
                                         no_midprice,  stake / 2, paper)
                    if r_yes.get("status") in ("paper", "submitted"):
                        state.total_trades    += 1
                        state.current_bankroll = max(0.0, state.current_bankroll - stake)
                        state.traded_markets.append(market_id)
                        audit("spread_arb", {
                            "market_id": market_id, "question": question[:80],
                            "net_ev":    sig["signals"].get("arb_spread", 0),
                            "stake":     stake, "paper": paper,
                        })
                        save_state(state)
                        signals_found += 1
                continue

            # ── Directional / Oracle Latency ─────────────────────────
            side = sig.get("side", "PASS")
            if side == "PASS":
                conf = sig.get("confidence", 0)
                t_rem = sig.get("T_remaining", 0)
                ev_best = max(sig.get("ev_yes", 0), sig.get("ev_no", 0))
                reasons = []
                if conf < 0.40: reasons.append(f"conf={conf:.2f}<0.40")
                if not (1.0 <= t_rem <= 3.5): reasons.append(f"T={t_rem:.1f}min")
                if ev_best < 0.05: reasons.append(f"ev={ev_best:.3f}<0.05")
                reason_str = ", ".join(reasons) if reasons else "signal too weak"
                log.debug(f"  [PASS] {question[:40]} — {reason_str}")
                continue

            kelly_frac = sig.get("kelly_fraction", 0.0)
            stake = min(kelly_frac * state.current_bankroll, MAX_STAKE_USD)
            if stake < MIN_STAKE_USD:
                log.debug(f"  [PASS] {question[:40]} — kelly stake ${stake:.3f} < ${MIN_STAKE_USD} min")
                continue
            if state.current_bankroll < MIN_STAKE_USD:
                log.info("  Bankroll below minimum — skipping")
                break

            token_id  = yes_token_id if side == "YES" else no_token_id
            bet_price = yes_midprice  if side == "YES" else (1.0 - yes_midprice)
            ev        = sig.get("ev_yes" if side == "YES" else "ev_no", 0.0)

            mode_str = "PAPER" if paper else "LIVE"
            log.info(
                f"\n  ✦ [{mode_str}] SIGNAL  {question[:55]}\n"
                f"    {side} @ {bet_price:.3f}  ev={ev:.4f}  conf={sig['confidence']:.2f}  "
                f"T={sig.get('T_remaining',0):.1f}min  strategy={sig['strategy']}"
            )

            audit("signal", {
                "market_id":    market_id,
                "question":     question[:80],
                "side":         side,
                "price":        bet_price,
                "model_p":      sig["model_p"],
                "ev":           ev,
                "stake":        stake,
                "signal_type":  "updown",
                "strategy":     sig.get("strategy", "DIRECTIONAL"),
                "confidence":   sig["confidence"],
                "kelly":        kelly_frac,
                "regime":       sig.get("regime", ""),
                "T_remaining":  sig.get("T_remaining", 0),
                "paper":        paper,
                "signals":      sig.get("signals", {}),
            })

            result = submit_order(client, token_id, "BUY", bet_price, stake, paper)

            if result.get("status") in ("paper", "submitted"):
                state.active_positions[market_id] = {
                    "question":    question,
                    "side":        side,
                    "token_id":    token_id,
                    "yes_token_id": yes_token_id,
                    "no_token_id":  no_token_id,
                    "price":       bet_price,
                    "stake":       stake,
                    "shares":      math.ceil(stake / bet_price),
                    "ev":          ev,
                    "entry_time":  datetime.now(timezone.utc).isoformat(),
                    "paper":       paper,
                    "signal_type": "updown",
                    "strategy":    sig.get("strategy", "DIRECTIONAL"),
                }
                state.traded_markets.append(market_id)
                state.total_trades    += 1
                state.current_bankroll = max(0.0, state.current_bankroll - stake)
                save_state(state)
                signals_found += 1

    # ══════════════════════════════════════════════════════════════

    if signals_found == 0:
        log.info("  No signals this scan.")

    audit("scan_complete", {
        "signals_found": signals_found,
        "duration_sec":  round(time.time() - scan_start, 1),
        "paper":         paper,
    })
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

    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument("--live",      action="store_true",                 help="Enable real order submission")
    parser.add_argument("--once",      action="store_true",                 help="Run one scan then exit")
    parser.add_argument("--bankroll",  type=float, default=20.0,            help="Starting bankroll in USD")
    parser.add_argument("--interval",  type=int,   default=SCAN_INTERVAL_SEC, help="Seconds between scans")
    parser.add_argument("--max-stake", type=float, default=MAX_STAKE_USD,   help="Max USD per trade")
    parser.add_argument("--ev-min",    type=float, default=EV_THRESHOLD,    help="Minimum EV to trade")
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
        log.warning("  LIVE MODE — real money will be spent")
        log.warning(f"  Bankroll: ${args.bankroll:.2f}")
        log.warning(f"  Max per trade: ${args.max_stake:.2f}")
        log.warning(f"  Kill switch at: -{MAX_DRAWDOWN_PCT:.0f}% drawdown from peak")
        log.warning("=" * 55)
        time.sleep(3)
    else:
        log.info("=" * 55)
        log.info("  PAPER MODE — no real orders will be placed")
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

    if SIGNAL_AVAILABLE:
        try:
            start_rtds_stream()
        except Exception as e:
            log.warning(f"  RTDS start failed ({e}) — using Binance REST fallback")

    state = load_state(args.bankroll)

    try:
        while True:
            run_scan(client, state, paper)
            if args.once:
                log.info("--once flag set, exiting after single scan.")
                break
            log.info(f"\n  Sleeping {args.interval}s until next scan…\n")
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
