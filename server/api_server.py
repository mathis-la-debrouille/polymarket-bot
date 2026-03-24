#!/usr/bin/env python3
"""
Polymarket Bot — Dashboard API

Endpoints:
  GET /          — health check
  GET /status    — bankroll, uptime, active positions count
  GET /metrics   — portfolio value, P&L, drawdown, win rate
  GET /trades    — recent trade signals (all strategies)
  GET /positions — open positions with live prices
  GET /kpi       — win rate, resolved stats, avg EV
  GET /chart/pnl — cumulative P&L time series
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

BOT_DIR    = Path(os.environ.get("BOT_DIR", "/home/polybot/app"))
STATE_FILE = BOT_DIR / "bot_state.json"
LOG_FILE   = BOT_DIR / "bot_log.jsonl"
API_TOKEN  = os.environ.get("API_TOKEN", "")
START_TIME = time.time()

app = FastAPI(title="Polymarket Bot Dashboard", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

security = HTTPBearer(auto_error=False)

def check_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_TOKEN:
        return
    if not credentials or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")

_LAST_GOOD_STATE: Dict = {}

def read_state() -> Dict:
    global _LAST_GOOD_STATE
    if not STATE_FILE.exists():
        return _LAST_GOOD_STATE
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        if data.get("starting_bankroll", 0) > 0:
            _LAST_GOOD_STATE = data
        return _LAST_GOOD_STATE if _LAST_GOOD_STATE else data
    except Exception:
        return _LAST_GOOD_STATE

def read_log(n: int = 500) -> List[Dict]:
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

def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f"{h}h {m}m"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

def _get_live_price(token_id: str, fallback: float) -> float:
    if not token_id:
        return fallback
    try:
        r = requests.get("https://clob.polymarket.com/midpoint",
                         params={"token_id": token_id}, timeout=4)
        return float(r.json().get("mid", fallback))
    except Exception:
        return fallback

@app.get("/", tags=["Health"])
def health():
    return {
        "status":   "ok",
        "ts":       datetime.now(timezone.utc).isoformat(),
        "uptime":   _fmt_duration(time.time() - START_TIME),
        "state_ok": STATE_FILE.exists(),
        "log_ok":   LOG_FILE.exists(),
    }

@app.get("/status", tags=["Bot"], dependencies=[Depends(check_auth)])
def status():
    state = read_state()
    if not state:
        return {"error": "Bot state file not found"}
    return {
        "current_bankroll":    round(state.get("current_bankroll", 0), 2),
        "starting_bankroll":   state.get("starting_bankroll", 0),
        "peak_bankroll":       round(state.get("peak_bankroll", 0), 2),
        "total_trades":        state.get("total_trades", 0),
        "active_positions":    len(state.get("active_positions", {})),
        "clob_cash":           round(state.get("clob_cash", 0), 2),
        "api_uptime":          _fmt_duration(time.time() - START_TIME),
        "state_last_modified": (
            datetime.fromtimestamp(STATE_FILE.stat().st_mtime, tz=timezone.utc).isoformat()
            if STATE_FILE.exists() else None
        ),
    }

@app.get("/metrics", tags=["Bot"], dependencies=[Depends(check_auth)])
def metrics():
    state = read_state()
    log   = read_log(1000)

    starting = state.get("starting_bankroll", 20.0)
    current  = state.get("current_bankroll",  starting)
    peak     = state.get("peak_bankroll",     current)
    clob_cash = state.get("clob_cash", 0.0)

    # Compute live position value
    active_pos = state.get("active_positions", {})
    position_value = 0.0
    for mid, pos in active_pos.items():
        token_id    = pos.get("token_id", "")
        entry_price = pos.get("price", 0)
        stake       = pos.get("stake", 0)
        shares      = pos.get("shares") or (math.ceil(stake / entry_price) if entry_price > 0 else 0)
        curr_price  = _get_live_price(token_id, entry_price)
        position_value += shares * curr_price

    portfolio    = current  # bot keeps this synced via balance_sync
    total_pnl    = portfolio - starting
    return_pct   = (total_pnl / starting * 100) if starting > 0 else 0
    drawdown_pct = ((peak - portfolio) / peak * 100) if peak > 0 else 0

    resolved = [e for e in log if e.get("event") == "position_resolved"]
    wins     = [e for e in resolved if e.get("pnl", 0) > 0]
    win_rate = round(len(wins) / len(resolved), 3) if resolved else None

    signals  = [e for e in log if e.get("event") == "signal"]
    scans    = [e for e in log if e.get("event") == "scan_complete"]
    last_scan_ago = None
    if scans:
        try:
            last_ts = datetime.fromisoformat(scans[-1]["ts"].rstrip("Z"))
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            last_scan_ago = int((datetime.now(timezone.utc) - last_ts).total_seconds())
        except Exception:
            pass

    return {
        "starting_bankroll":  starting,
        "current_bankroll":   round(current, 2),
        "peak_bankroll":      round(peak, 2),
        "portfolio_value":    round(portfolio, 2),
        "clob_cash":          round(clob_cash, 2),
        "position_value":     round(position_value, 2),
        "total_pnl":          round(total_pnl, 2),
        "return_pct":         round(return_pct, 2),
        "drawdown_pct":       round(max(0, drawdown_pct), 2),
        "total_trades":       state.get("total_trades", 0),
        "active_positions":   len(active_pos),
        "win_rate":           win_rate,
        "total_resolved":     len(resolved),
        "total_signals":      len(signals),
        "last_scan_ago_sec":  last_scan_ago,
        "kill_switch":        any(e.get("event") == "kill_switch" for e in log),
        "uptime":             _fmt_duration(time.time() - START_TIME),
    }

@app.get("/trades", tags=["Bot"], dependencies=[Depends(check_auth)])
def trades(n: int = Query(default=50, le=500)):
    log = read_log(2000)
    signals = [e for e in log if e.get("event") == "signal"]
    signals = signals[-n:]
    signals.reverse()

    # Match with resolved positions for outcome
    resolved_map = {e.get("market_id"): e for e in log if e.get("event") == "position_resolved"}

    result = []
    for e in signals:
        mid  = e.get("market_id", "")
        res  = resolved_map.get(mid)
        result.append({
            "ts":             e.get("ts"),
            "market":         e.get("question", "?")[:60],
            "side":           e.get("side"),
            "price":          round(e.get("price", 0), 3),
            "model_p":        round(e.get("model_p", 0), 3),
            "ev":             round(e.get("ev", 0), 4),
            "stake_usd":      round(e.get("stake", 0), 2),
            "signal_type":    e.get("signal_type", ""),
            "paper":          e.get("paper", True),
            "outcome":        ("won" if res and res.get("pnl", 0) > 0
                               else "lost" if res and res.get("pnl", 0) <= 0
                               else "open"),
            "pnl":            round(res["pnl"], 4) if res else None,
        })
    return {"count": len(result), "trades": result}

@app.get("/positions", tags=["Bot"], dependencies=[Depends(check_auth)])
def positions():
    state  = read_state()
    active = state.get("active_positions", {})
    rows   = []
    for mid, pos in active.items():
        entry    = pos.get("price", 0)
        stake    = pos.get("stake", 0)
        token_id = pos.get("token_id", "")
        shares   = pos.get("shares") or (math.ceil(stake / entry) if entry > 0 else 0)
        curr     = _get_live_price(token_id, entry)
        upnl     = round(shares * (curr - entry), 4)
        rows.append({
            "market_id":      mid,
            "question":       pos.get("question", "?")[:65],
            "side":           pos.get("side"),
            "signal_type":    pos.get("signal_type", "legacy"),
            "entry_price":    round(entry, 3),
            "current_price":  round(curr, 3),
            "shares":         shares,
            "stake_usd":      round(stake, 2),
            "unrealized_pnl": upnl,
            "ev":             round(pos.get("ev", 0), 4),
            "paper":          pos.get("paper", True),
            "entry_time":     pos.get("entry_time"),
        })
    rows.sort(key=lambda r: r.get("entry_time", ""), reverse=True)
    return {"count": len(rows), "positions": rows}

@app.get("/kpi", tags=["Bot"], dependencies=[Depends(check_auth)])
def kpi():
    state   = read_state()
    entries = read_log(5000)

    resolved  = [e for e in entries if e.get("event") == "position_resolved"]
    signals   = [e for e in entries if e.get("event") == "signal"]
    scans     = [e for e in entries if e.get("event") == "scan_complete"]

    wins   = [e for e in resolved if e.get("pnl", 0) > 0]
    losses = [e for e in resolved if e.get("pnl", 0) <= 0]

    evs    = [e.get("ev",    0) for e in signals]
    stakes = [e.get("stake", 0) for e in signals]

    last_scan = scans[-1] if scans else {}

    return {
        "win_rate":       round(len(wins) / len(resolved), 3) if resolved else None,
        "total_resolved": len(resolved),
        "total_wins":     len(wins),
        "total_losses":   len(losses),
        "total_won_usd":  round(sum(e.get("pnl", 0) for e in wins),   2),
        "total_lost_usd": round(sum(e.get("pnl", 0) for e in losses), 2),
        "avg_win_usd":    round(sum(e.get("pnl", 0) for e in wins)   / len(wins),   2) if wins   else None,
        "avg_loss_usd":   round(sum(e.get("pnl", 0) for e in losses) / len(losses), 2) if losses else None,
        "total_signals":  len(signals),
        "avg_ev":         round(float(np.mean(evs)),    4) if evs    else None,
        "avg_stake":      round(float(np.mean(stakes)), 2) if stakes else None,
        "total_scans":    len(scans),
        "last_scan":      last_scan,
    }

@app.get("/chart/pnl", tags=["Charts"], dependencies=[Depends(check_auth)])
def chart_pnl():
    log   = read_log(5000)
    state = read_state()
    start = state.get("starting_bankroll", 20.0)

    resolved = [e for e in log if e.get("event") == "position_resolved"]
    balance_syncs = [e for e in log if e.get("event") == "balance_sync"]

    series  = []
    running = start

    for e in sorted(resolved, key=lambda x: x.get("ts", "")):
        running += e.get("pnl", 0)
        series.append({
            "ts":             e.get("ts"),
            "bankroll":       round(running, 2),
            "pnl_cumulative": round(running - start, 2),
            "type":           "resolved",
        })

    # Also add balance_sync points for a smoother curve
    for e in sorted(balance_syncs, key=lambda x: x.get("ts", "")):
        portfolio = e.get("portfolio", 0)
        if portfolio > 0:
            series.append({
                "ts":             e.get("ts"),
                "bankroll":       round(portfolio, 2),
                "pnl_cumulative": round(portfolio - start, 2),
                "type":           "sync",
            })

    series.sort(key=lambda x: x.get("ts", ""))

    # Deduplicate by minute
    seen = set()
    deduped = []
    for pt in series:
        key = (pt.get("ts", "")[:16], pt.get("type"))
        if key not in seen:
            seen.add(key)
            deduped.append(pt)

    if not deduped:
        deduped = [{"ts": datetime.now(timezone.utc).isoformat(),
                    "bankroll": round(start, 2), "pnl_cumulative": 0.0, "type": "start"}]

    return {
        "starting_bankroll": start,
        "data_points":       len(deduped),
        "series":            deduped,
    }

@app.get("/signals/recent", tags=["Bot"], dependencies=[Depends(check_auth)])
def signals_recent(n: int = Query(default=100, le=500)):
    """Recent traded signals with full signal breakdown (bm, ofi, mom, mc)."""
    log = read_log(3000)
    events = [e for e in log if e.get("event") == "signal"]
    events = events[-n:]
    events.reverse()
    resolved_map = {e.get("market_id"): e for e in log if e.get("event") == "position_resolved"}
    result = []
    for e in events:
        mid = e.get("market_id", "")
        res = resolved_map.get(mid)
        sigs = e.get("signals", {})
        result.append({
            "ts":          e.get("ts"),
            "market_id":   mid,
            "market":      e.get("question", "?")[:65],
            "side":        e.get("side"),
            "price":       round(e.get("price", 0), 4),
            "model_p":     round(e.get("model_p", 0), 4),
            "confidence":  round(e.get("confidence", 0), 4),
            "ev":          round(e.get("ev", 0), 4),
            "stake_usd":   round(e.get("stake", 0), 2),
            "kelly":       round(e.get("kelly", 0), 4),
            "strategy":    e.get("strategy", ""),
            "regime":      e.get("regime", ""),
            "T_remaining": round(e.get("T_remaining", 0), 2),
            "paper":       e.get("paper", True),
            "bm":          round(sigs.get("bm", 0), 4) if sigs else None,
            "ofi":         round(sigs.get("ofi", 0), 4) if sigs else None,
            "mom":         round(sigs.get("mom", 0), 4) if sigs else None,
            "mc":          round(sigs.get("mc", 0), 4) if sigs else None,
            "outcome":     ("won" if res and res.get("pnl", 0) > 0
                            else "lost" if res and res.get("pnl", 0) <= 0
                            else "open"),
            "pnl":         round(res["pnl"], 4) if res else None,
        })
    return {"count": len(result), "signals": result}


@app.get("/debug/state", tags=["Debug"], dependencies=[Depends(check_auth)])
def debug_state():
    """Raw bot state JSON."""
    return read_state()


@app.get("/debug/events", tags=["Debug"], dependencies=[Depends(check_auth)])
def debug_events(
    event: str = Query(default="", description="filter by event type"),
    n: int = Query(default=100, le=1000),
):
    """Recent audit log entries, optionally filtered by event type."""
    entries = read_log(max(n * 3, 1000))
    if event:
        entries = [e for e in entries if e.get("event") == event]
    entries = list(reversed(entries[-n:]))
    return {"count": len(entries), "events": entries}


@app.get("/debug/scans", tags=["Debug"], dependencies=[Depends(check_auth)])
def debug_scans(n: int = Query(default=20, le=100)):
    """Recent scan_complete events."""
    entries = read_log(2000)
    scans = [e for e in entries if e.get("event") == "scan_complete"]
    scans = list(reversed(scans[-n:]))
    return {"count": len(scans), "scans": scans}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    print(f"\n  Polymarket Dashboard API — http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
