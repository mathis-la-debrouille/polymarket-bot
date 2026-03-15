#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║           Polymarket Bot — Performance Dashboard API             ║
║                                                                  ║
║  Reads the bot's state + audit log and exposes them via HTTP.    ║
║  Run alongside polymarket_bot.py on the server.                  ║
║                                                                  ║
║  Endpoints:                                                      ║
║    GET /              — health check                             ║
║    GET /status        — bankroll, mode, uptime                   ║
║    GET /metrics       — win rate, Sharpe, P&L, drawdown          ║
║    GET /trades        — trade history (last N)                   ║
║    GET /positions     — open positions                           ║
║    GET /log           — raw audit log (last N events)            ║
║    GET /chart/pnl     — cumulative P&L as JSON series            ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    pip install fastapi uvicorn python-dotenv

    # Start API server (runs on port 8000 by default):
    python api_server.py

    # Custom port / host:
    python api_server.py --host 0.0.0.0 --port 8000

    # Query from your local machine:
    curl http://YOUR_SERVER_IP:8000/metrics
    curl http://YOUR_SERVER_IP:8000/trades?n=20
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from fastapi import FastAPI, HTTPException, Depends, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    import uvicorn
except ImportError:
    print("[!] FastAPI not installed. Run:  pip install fastapi uvicorn")
    raise

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
BOT_DIR    = Path(os.environ.get("BOT_DIR", "/home/polybot/app"))
STATE_FILE = BOT_DIR / "bot_state.json"
LOG_FILE   = BOT_DIR / "bot_log.jsonl"

# Optional API token for security (set API_TOKEN in .env)
API_TOKEN  = os.environ.get("API_TOKEN", "")

START_TIME = time.time()

# ──────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Polymarket Bot Dashboard",
    description = "Live performance metrics for the Polymarket trading bot",
    version     = "1.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten this if you add a frontend
    allow_methods  = ["GET"],
    allow_headers  = ["*"],
)

# ──────────────────────────────────────────────────────────────────
# AUTH  (optional bearer token)
# ──────────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

def check_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_TOKEN:
        return   # no token configured → open access
    if not credentials or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────
def read_state() -> Dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def read_log(n: int = 200) -> List[Dict]:
    if not LOG_FILE.exists():
        return []
    try:
        lines = LOG_FILE.read_text().strip().splitlines()
        parsed = []
        for line in lines[-n:]:
            try:
                parsed.append(json.loads(line))
            except Exception:
                pass
        return parsed
    except Exception:
        return []


def compute_metrics(state: Dict, log_entries: List[Dict]) -> Dict:
    """Compute performance metrics from state + trade log."""
    trades = [e for e in log_entries if e.get("event") == "order_placed"]
    signals = [e for e in log_entries if e.get("event") == "signal"]

    starting = state.get("starting_bankroll", 0)
    current  = state.get("current_bankroll",  0)
    peak     = state.get("peak_bankroll",      current)

    total_pnl    = current - starting
    return_pct   = (total_pnl / starting * 100) if starting > 0 else 0
    drawdown_pct = ((peak - current) / peak * 100) if peak > 0 else 0

    # Win/loss from active_positions (paper) or log
    # We approximate from total_pnl and total_trades
    total_trades = state.get("total_trades", len(trades))

    # EV stats from signals
    evs = [s.get("ev", 0) for s in signals]
    avg_ev    = float(np.mean(evs)) if evs else 0.0
    stakes    = [s.get("stake", 0) for s in signals]
    avg_stake = float(np.mean(stakes)) if stakes else 0.0

    # Uptime
    uptime_sec = time.time() - START_TIME
    uptime_str = _fmt_duration(uptime_sec)

    # Kill switch check
    kill_triggered = any(e.get("event") == "kill_switch" for e in log_entries)

    return {
        "starting_bankroll": starting,
        "current_bankroll":  round(current, 2),
        "peak_bankroll":     round(peak, 2),
        "total_pnl":         round(total_pnl, 2),
        "return_pct":        round(return_pct, 2),
        "drawdown_pct":      round(drawdown_pct, 2),
        "total_trades":      total_trades,
        "avg_ev":            round(avg_ev, 4),
        "avg_stake_usd":     round(avg_stake, 2),
        "kill_switch_triggered": kill_triggered,
        "uptime":            uptime_str,
    }


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ──────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    """Server health check."""
    return {
        "status":    "ok",
        "ts":        datetime.now(timezone.utc).isoformat(),
        "uptime":    _fmt_duration(time.time() - START_TIME),
        "state_ok":  STATE_FILE.exists(),
        "log_ok":    LOG_FILE.exists(),
    }


@app.get("/status", tags=["Bot"], dependencies=[Depends(check_auth)])
def status():
    """Current bot state: bankroll, mode, active positions."""
    state = read_state()
    if not state:
        return {"error": "Bot state file not found. Is the bot running?"}

    return {
        "current_bankroll":   round(state.get("current_bankroll", 0), 2),
        "starting_bankroll":  state.get("starting_bankroll", 0),
        "peak_bankroll":      round(state.get("peak_bankroll", 0), 2),
        "total_trades":       state.get("total_trades", 0),
        "active_positions":   len(state.get("active_positions", {})),
        "traded_markets":     len(state.get("traded_markets", [])),
        "api_uptime":         _fmt_duration(time.time() - START_TIME),
        "state_last_modified": (
            datetime.fromtimestamp(STATE_FILE.stat().st_mtime, tz=timezone.utc).isoformat()
            if STATE_FILE.exists() else None
        ),
    }


@app.get("/metrics", tags=["Bot"], dependencies=[Depends(check_auth)])
def metrics():
    """Computed performance metrics: P&L, return %, drawdown, EV stats."""
    state = read_state()
    log   = read_log(500)
    return compute_metrics(state, log)


@app.get("/trades", tags=["Bot"], dependencies=[Depends(check_auth)])
def trades(n: int = Query(default=50, le=500, description="Number of recent trades to return")):
    """Recent trade signals with EV, stake, side, market."""
    log     = read_log(1000)
    signals = [e for e in log if e.get("event") == "signal"]
    signals = signals[-n:]
    signals.reverse()   # newest first

    return {
        "count":  len(signals),
        "trades": [
            {
                "ts":        e.get("ts"),
                "market":    e.get("question", "?")[:60],
                "side":      e.get("side"),
                "price":     round(e.get("price", 0), 3),
                "model_p":   round(e.get("model_p", 0), 3),
                "ev":        round(e.get("ev", 0), 4),
                "stake_usd": round(e.get("stake", 0), 2),
                "paper":     e.get("paper", True),
            }
            for e in signals
        ],
    }


@app.get("/positions", tags=["Bot"], dependencies=[Depends(check_auth)])
def positions():
    """Currently open / tracked positions."""
    state = read_state()
    active = state.get("active_positions", {})

    return {
        "count":     len(active),
        "positions": [
            {
                "market_id": mid,
                "question":  pos.get("question", "?")[:60],
                "side":      pos.get("side"),
                "price":     round(pos.get("price", 0), 3),
                "stake_usd": round(pos.get("stake", 0), 2),
                "ev":        round(pos.get("ev", 0), 4),
                "paper":     pos.get("paper", True),
            }
            for mid, pos in active.items()
        ],
    }


@app.get("/log", tags=["Bot"], dependencies=[Depends(check_auth)])
def audit_log(n: int = Query(default=50, le=500, description="Number of log entries to return")):
    """Raw audit log — every bot decision as structured JSON."""
    entries = read_log(n)
    entries.reverse()  # newest first
    return {
        "count":   len(entries),
        "entries": entries,
    }


@app.get("/balance/real", tags=["Bot"], dependencies=[Depends(check_auth)])
def real_balance():
    """
    Query the actual USDC balance from two sources:
      • wallet_usdc  — undeposited USDC sitting in the EOA wallet (Polygon on-chain)
      • clob_usdc    — USDC deposited into Polymarket and available to trade (CLOB)
    The tracked_bankroll is what the bot computed locally (may drift from reality).
    """
    pk = os.environ.get("PRIVATE_KEY", "")
    if not pk:
        return {"error": "PRIVATE_KEY not set in server .env"}

    # ── 1. Derive wallet address ────────────────────────────────────
    wallet_address = None
    try:
        from eth_account import Account
        wallet_address = Account.from_key(pk).address
    except Exception as e:
        return {"error": f"Could not derive wallet address: {e}"}

    # ── 2. On-chain wallet USDC balance (Polygon RPC) ───────────────
    wallet_usdc = None
    try:
        USDC_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e on Polygon
        padded = wallet_address[2:].lower().zfill(64)
        data   = "0x70a08231" + padded  # balanceOf(address)
        resp   = requests.post("https://polygon-rpc.com", json={
            "jsonrpc": "2.0", "method": "eth_call",
            "params": [{"to": USDC_POLYGON, "data": data}, "latest"], "id": 1,
        }, timeout=8)
        raw = resp.json().get("result", "0x0")
        wallet_usdc = int(raw, 16) / 1_000_000  # USDC has 6 decimals
    except Exception as e:
        wallet_usdc = None

    # ── 3. CLOB balance (deposited into Polymarket) ─────────────────
    clob_usdc = None
    try:
        from py_clob_client.client import ClobClient
        client = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137)
        creds  = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        bal = client.get_balance_allowance({"asset_type": "COLLATERAL", "signature_type": "EOA"})
        clob_usdc = float(bal.get("balance", 0)) / 1_000_000
    except Exception:
        clob_usdc = None

    tracked = read_state().get("current_bankroll", 0)
    total   = (wallet_usdc or 0) + (clob_usdc or 0)

    return {
        "wallet_address":   wallet_address,
        "wallet_usdc":      round(wallet_usdc, 4) if wallet_usdc is not None else None,
        "clob_usdc":        round(clob_usdc,   4) if clob_usdc   is not None else None,
        "total_real_usdc":  round(total, 4),
        "tracked_bankroll": round(tracked, 4),
        "drift":            round(total - tracked, 4),
    }


@app.get("/chart/pnl", tags=["Charts"], dependencies=[Depends(check_auth)])
def chart_pnl():
    """
    Returns a time-series of cumulative P&L for charting.
    Format: [{ts, bankroll, pnl_cumulative}, ...]
    """
    log     = read_log(2000)
    state   = read_state()
    start   = state.get("starting_bankroll", 20.0)

    events = [e for e in log if e.get("event") == "order_placed"]

    series = []
    running = start
    for e in events:
        stake  = e.get("stake", 0)
        result = e.get("result", {})
        status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
        # In paper mode we record the debit; actual PnL is not known until resolution
        running -= stake if status in ("paper", "submitted") else 0
        series.append({
            "ts":              e.get("ts"),
            "bankroll":        round(running, 2),
            "pnl_cumulative":  round(running - start, 2),
        })

    return {
        "starting_bankroll": start,
        "data_points":       len(series),
        "series":            series,
    }


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Bot Dashboard API")
    parser.add_argument("--host", default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8000, type=int, help="Port (default: 8000)")
    args = parser.parse_args()

    print(f"\n  Polymarket Dashboard API")
    print(f"  Running on  http://{args.host}:{args.port}")
    print(f"  Docs at     http://{args.host}:{args.port}/docs")
    print(f"  Bot dir     {BOT_DIR}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
