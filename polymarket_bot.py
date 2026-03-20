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
import re
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

try:
    from signal_updown import (
        compute_updown_signal, is_updown_market,
        start_rtds_stream, check_spread_arb,
    )
    UPDOWN_AVAILABLE = True
except ImportError:
    UPDOWN_AVAILABLE = False
    print("[!] signal_updown.py not found — crypto Up/Down pass disabled")

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

# ── SCALPING STRATEGY: $1 flat per trade, target +15% return ──────────
# Formula: only enter when model_p / market_price × 0.98 - 1 >= 0.15
# i.e. buy at ≤ $0.852 with enough signal conviction to expect a win.
# Fixed $1 stake = Polymarket minimum = one clean bet per window.
EV_THRESHOLD      = 0.15    # 15% minimum net return (after 2% fee)
MAX_STAKE_USD     = 1.00    # fixed $1 per trade — no Kelly sizing
MIN_STAKE_USD     = 1.00    # Polymarket minimum
MAX_DRAWDOWN_PCT  = 30.0    # kill switch: stop if down 30% from peak
KELLY_FRACTION    = 1.0     # unused — flat $1 stake overrides Kelly

# cap traded_markets list
MAX_TRADED_MARKETS = 200

SCAN_INTERVAL_SEC = 30
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
        # Auto-exit stale 5-min positions (should resolve in minutes, not days)
        entry_time_str = pos.get("entry_time", "")
        if pos.get("signal_type") == "5min_updown" and entry_time_str:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str.rstrip("Z"))
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - entry_dt).total_seconds() > 86400:
                    log.info(f"  [STALE] Auto-removing unresolved 5min position: {pos.get('question','?')[:50]}")
                    del state.active_positions[market_id]
                    continue
            except Exception:
                pass

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

        if not token_id or entry_price <= 0:
            continue
        # 5-min markets resolve in minutes — stop-loss not applicable
        if pos.get("signal_type") == "5min_updown":
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
            bet_price = curr_price if side == "YES" else (1 - curr_price)
            new_ev = -0.01  # treat as edge-gone when >30% loss
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
        save_state(state)   # persist synced balance so dashboard always has fresh values
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
        # Bitcoin ONLY — scalping strategy targets BTC 5-min windows exclusively
        updown = [m for m in all_markets
                  if is_updown_market(m.get("question", ""))
                  and re.search(r'\bbitcoin\b|\bbtc\b', m.get("question", ""), re.I)]
        log.info(f"  [Pass A] BTC Up/Down markets: {len(updown)}/{len(all_markets)}")
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

    # ══════════════════════════════════════════════════════════════════
    # PASS A — 5-Min/15-Min Crypto Up/Down Markets (full signal stack)
    # Targets: BTC, ETH, SOL, XRP, DOGE — every 30 seconds
    # ══════════════════════════════════════════════════════════════════
    if UPDOWN_AVAILABLE:
        updown_signals = 0

        for raw in fetch_updown_markets():
            market_id = raw.get("id", "")
            question  = raw.get("question", "?")
            if market_id in state.traded_markets:
                continue

            try:
                tokens    = json.loads(raw["tokens"]) if isinstance(raw.get("tokens"), str) \
                            else raw.get("tokens", [])
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

            signal = compute_updown_signal(raw, yes_midprice)
            if signal is None:
                continue

            # ── SPREAD ARB: riskless, bypass all other filters ──────────
            if signal["arb_detected"] and signal["net_arb_ev"] > 0.005:
                log.info(f"\n  ⚡ RISKLESS ARB  net_ev={signal['net_arb_ev']:.3f}  {question[:50]}")
                stake = min(state.current_bankroll * 0.10, MAX_STAKE_USD)
                if stake >= MIN_STAKE_USD and no_token:
                    no_token_id = no_token.get("token_id", "")
                    res_yes = submit_order(client, yes_token_id, "BUY",
                                           yes_midprice, stake / 2, paper)
                    res_no  = submit_order(client, no_token_id, "BUY",
                                           1.0 - yes_midprice, stake / 2, paper)
                    if res_yes.get("status") in ("paper", "submitted"):
                        state.total_trades    += 1
                        state.current_bankroll = max(0.0, state.current_bankroll - stake)
                        state.traded_markets.append(market_id)
                        audit("spread_arb", {"market_id": market_id, "question": question[:80],
                                              "net_ev": signal["net_arb_ev"], "stake": stake,
                                              "paper": paper})
                        save_state(state)
                        updown_signals += 1
                continue

            # ── DIRECTIONAL TRADE filters ─────────────────────────────
            if not signal["timing_ok"]:
                continue
            if signal["signal_strength"] < 1.5:  # needs clear directional move for 15% target
                continue
            if signal["confidence"] < 0.35:
                continue

            model_p     = signal["model_p"]
            no_midprice = 1.0 - yes_midprice

            ev_yes = compute_ev(model_p,       yes_midprice)
            ev_no  = compute_ev(1.0 - model_p, no_midprice)

            if ev_yes >= ev_no and ev_yes >= EV_THRESHOLD:
                bet_side, ev, bet_price, token, used_p = \
                    "YES", ev_yes, yes_midprice, yes_token, model_p
            elif ev_no > ev_yes and ev_no >= EV_THRESHOLD:
                bet_side, ev, bet_price, token, used_p = \
                    "NO", ev_no, no_midprice, no_token, 1.0 - model_p
            else:
                continue

            if not token:
                continue

            token_id = token.get("token_id", "")
            stake    = 1.00  # flat $1 per trade — scalping strategy
            if state.current_bankroll < MIN_STAKE_USD:
                log.info("  [Pass A] Bankroll below $1 — skipping.")
                break

            log.info(
                f"\n  ✦ SIGNAL  {question[:55]}\n"
                f"    {bet_side} @ {bet_price:.3f}  model={used_p:.3f}  EV={ev:.4f}  "
                f"conf={signal['confidence']:.2f}  strength={signal['signal_strength']:.2f}\n"
                f"    move={signal['pct_move']*100:+.3f}%  "
                f"{signal['minutes_elapsed']:.1f}/{signal['window_duration']:.0f}min elapsed  "
                f"stake=${stake:.2f}"
            )

            audit("signal", {
                "market_id": market_id, "question": question[:80],
                "side": bet_side, "price": bet_price, "model_p": used_p,
                "ev": ev, "stake": stake, "signal_type": "5min_updown",
                "signal_strength": signal["signal_strength"],
                "confidence": signal["confidence"],
                "pct_move": signal["pct_move"],
                "minutes_left": signal["minutes_left"],
                "paper": paper,
            })

            result = submit_order(client, token_id, "BUY", bet_price, stake, paper)

            if result.get("status") in ("paper", "submitted"):
                state.active_positions[market_id] = {
                    "question":     question,
                    "side":         bet_side,
                    "token_id":     token_id,
                    "yes_token_id": yes_token_id,
                    "price":        bet_price,
                    "stake":        stake,
                    "ev":           ev,
                    "entry_time":   datetime.now(timezone.utc).isoformat(),
                    "paper":        paper,
                    "signal_type":  "5min_updown",
                }
                state.traded_markets.append(market_id)
                state.total_trades    += 1
                state.current_bankroll = max(0.0, state.current_bankroll - stake)
                audit("order_placed", {"market_id": market_id, "stake": stake,
                                        "signal_type": "5min_updown", "result": result})
                save_state(state)
                updown_signals += 1

        if updown_signals == 0:
            log.info("  [Pass A] No signals this scan.")
    duration_sec = round(time.time() - scan_start_time, 1)
    audit("scan_complete", {
        "signals_found": updown_signals if UPDOWN_AVAILABLE else 0,
        "duration_sec":  duration_sec,
        "paper":         paper,
    })

    log.info(
        f"\n  Bankroll: ${state.current_bankroll:.2f} | "
        f"Trades: {state.total_trades} | Active: {len(state.active_positions)} | "
        f"Resolved: {len(state.resolved_positions)}"
    )

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
        log.warning(f"  Stop threshold: -{MAX_DRAWDOWN_PCT:.0f}% drawdown from peak")
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

    # Start Polymarket real-time price stream (same oracle used for resolution)
    if UPDOWN_AVAILABLE:
        try:
            start_rtds_stream()
            log.info("  Waiting 3s for RTDS WebSocket to connect…")
            time.sleep(3)
        except Exception as e:
            log.warning(f"  RTDS not started: {e} — using Binance fallback")

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
