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
OUTPUT_LOG = Path(os.environ.get("BOT_OUTPUT_LOG", str(BOT_DIR / "bot_output.log")))
API_TOKEN  = os.environ.get("API_TOKEN", "")
START_TIME = time.time()

app = FastAPI(title="Polymarket Bot Dashboard", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

def _live_stats_all() -> Dict:
    """Scan full audit log for live wallet stats, portfolio series, and win/loss buckets."""
    wins = 0
    losses = 0
    pnl_series = []        # {ts, pnl: cumulative realized PnL}
    portfolio_series = []  # {ts, portfolio: from balance_sync events}
    hour_buckets: Dict[str, Dict] = {}
    cumulative = 0.0

    if not LOG_FILE.exists():
        return {"wins": 0, "losses": 0, "resolved": 0, "win_rate": None,
                "pnl_series": [], "portfolio_series": [], "hour_buckets": [],
                "starting_bankroll": 0, "current_bankroll": 0}
    try:
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                ev = e.get("event", "")
                if ev == "position_resolved":
                    pnl = e.get("pnl", 0)
                    ts  = e.get("ts", "")
                    cumulative += pnl
                    won = pnl > 0
                    if won: wins += 1
                    else:   losses += 1
                    pnl_series.append({"ts": ts, "pnl": round(cumulative, 4)})
                    hour_key = ts[:13] if len(ts) >= 13 else ts[:10]
                    if hour_key not in hour_buckets:
                        hour_buckets[hour_key] = {"hour": hour_key, "wins": 0, "losses": 0}
                    hour_buckets[hour_key]["wins" if won else "losses"] += 1
                elif ev == "balance_sync" and (e.get("portfolio") or 0) > 0:
                    portfolio_series.append({"ts": e.get("ts"), "portfolio": round(e.get("portfolio", 0), 2)})
    except Exception:
        pass

    # Deduplicate portfolio by minute
    seen: set = set()
    deduped: List[dict] = []
    for pt in portfolio_series:
        key = (pt.get("ts") or "")[:16]
        if key not in seen:
            seen.add(key)
            deduped.append(pt)

    state = read_state()
    resolved = wins + losses
    return {
        "wins":              wins,
        "losses":            losses,
        "resolved":          resolved,
        "win_rate":          round(wins / resolved, 3) if resolved else None,
        "pnl_series":        pnl_series,
        "portfolio_series":  deduped,
        "hour_buckets":      sorted(hour_buckets.values(), key=lambda x: x["hour"]),
        "starting_bankroll": round(state.get("starting_bankroll", 0), 2),
        "current_bankroll":  round(state.get("current_bankroll", 0), 2),
    }


def _paper_stats_all() -> Dict:
    """Scan full audit log for paper wallet stats, portfolio series, and win/loss buckets."""
    wins = 0
    losses = 0
    pnl_series = []        # {ts, pnl: cumulative_pnl}
    portfolio_series = []  # {ts, portfolio: starting + cumulative_pnl}
    hour_buckets: Dict[str, Dict[str, int]] = {}  # "YYYY-MM-DD HH" → {wins, losses}
    PAPER_START = 100.0
    cumulative = 0.0

    if not LOG_FILE.exists():
        return {"wins": 0, "losses": 0, "resolved": 0, "win_rate": None,
                "pnl_series": [], "portfolio_series": [], "hour_buckets": []}
    try:
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("event") != "paper_position_resolved":
                    continue
                pnl = e.get("pnl", 0)
                ts  = e.get("ts", "")
                cumulative += pnl
                won = pnl > 0
                if won:
                    wins += 1
                else:
                    losses += 1
                pnl_series.append({"ts": ts, "pnl": round(cumulative, 4)})
                portfolio_series.append({"ts": ts, "portfolio": round(PAPER_START + cumulative, 4)})
                # Hourly win/loss bucket
                hour_key = ts[:13] if len(ts) >= 13 else ts[:10]  # "YYYY-MM-DDTHH"
                if hour_key not in hour_buckets:
                    hour_buckets[hour_key] = {"hour": hour_key, "wins": 0, "losses": 0}
                if won:
                    hour_buckets[hour_key]["wins"] += 1
                else:
                    hour_buckets[hour_key]["losses"] += 1
    except Exception:
        pass
    resolved = wins + losses
    buckets_sorted = sorted(hour_buckets.values(), key=lambda x: x["hour"])
    return {
        "wins":             wins,
        "losses":           losses,
        "resolved":         resolved,
        "win_rate":         round(wins / resolved, 3) if resolved else None,
        "pnl_series":       pnl_series,
        "portfolio_series": portfolio_series,
        "hour_buckets":     buckets_sorted,
    }

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
                         params={"token_id": token_id}, timeout=2)
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

    starting  = state.get("starting_bankroll", 20.0)
    current   = state.get("current_bankroll", starting)
    clob_cash = state.get("clob_cash", 0.0)

    # Daily reference: value at 6 AM UTC today (resets each day)
    daily_start = state.get("daily_start_bankroll") or 0.0
    if daily_start <= 0:
        daily_start = starting

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

    # portfolio = cash + open positions mark-to-market
    portfolio    = clob_cash + position_value
    # All daily metrics relative to 6 AM UTC baseline
    total_pnl    = portfolio - daily_start
    return_pct   = (total_pnl / daily_start * 100) if daily_start > 0 else 0
    drawdown_pct = (max(0, daily_start - portfolio) / daily_start * 100) if daily_start > 0 else 0

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

    # Paper wallet metrics
    paper_bankroll   = round(state.get("paper_bankroll",   100.0), 2)
    paper_starting   = round(state.get("paper_daily_start", 100.0), 2)
    paper_pnl        = round(paper_bankroll - paper_starting, 2)
    paper_trades     = state.get("paper_trades", 0)
    paper_active     = len(state.get("paper_active_positions", {}))
    paper_return_pct = round((paper_pnl / paper_starting * 100) if paper_starting > 0 else 0, 2)

    # Paper wallet — scan full log for accurate stats
    p_stats        = _paper_stats_all()
    paper_win_rate = p_stats["win_rate"]

    return {
        "starting_bankroll":  daily_start,
        "current_bankroll":   round(current, 2),
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
        "live_mode":          state.get("paper_mode", True) is False,
        # Paper wallet
        "paper_bankroll":     paper_bankroll,
        "paper_starting":     paper_starting,
        "paper_pnl":          paper_pnl,
        "paper_return_pct":   paper_return_pct,
        "paper_trades":       paper_trades,
        "paper_active":       paper_active,
        "paper_win_rate":     paper_win_rate,
        "paper_resolved":     p_stats["resolved"],
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
    rows.sort(key=lambda r: r.get("entry_time") or "", reverse=True)
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
        "win_rate":         round(len(wins) / len(resolved), 3) if resolved else None,
        "total_resolved":   len(resolved),
        "total_wins":       len(wins),
        "total_losses":     len(losses),
        "total_won_usd":    round(sum(e.get("pnl", 0) for e in wins),   2),
        "total_lost_usd":   round(sum(e.get("pnl", 0) for e in losses), 2),
        "avg_win_usd":      round(sum(e.get("pnl", 0) for e in wins)   / len(wins),   2) if wins   else None,
        "avg_loss_usd":     round(sum(e.get("pnl", 0) for e in losses) / len(losses), 2) if losses else None,
        "total_signals":    len(signals),
        "avg_ev":           round(float(np.mean(evs)),    4) if evs    else None,
        "avg_stake":        round(float(np.mean(stakes)), 2) if stakes else None,
        "total_scans":      len(scans),
        "last_scan":        last_scan,
        # Daily P&L from state (6 AM UTC reference)
        "daily_start_bankroll": round(state.get("daily_start_bankroll") or state.get("starting_bankroll", 0), 2),
        "actual_pnl":       round(state.get("total_pnl", 0), 2),
        "actual_starting":  round(state.get("daily_start_bankroll") or state.get("starting_bankroll", 0), 2),
        "actual_current":   round(state.get("current_bankroll", 0), 2),
    }

@app.get("/chart/pnl", tags=["Charts"], dependencies=[Depends(check_auth)])
def chart_pnl():
    log   = read_log(5000)
    state = read_state()

    # Use daily_start_bankroll as zero reference (resets at 6 AM UTC each day)
    # Fall back to starting_bankroll if daily reset not yet set
    daily_ref = state.get("daily_start_bankroll") or state.get("starting_bankroll", 20.0)
    if daily_ref <= 0:
        daily_ref = state.get("starting_bankroll", 20.0)

    # Only show last 24 hours of data
    cutoff_ts = datetime.now(timezone.utc).isoformat()[:10]  # today's date prefix
    yesterday = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                 .isoformat())

    balance_syncs = [
        e for e in log
        if e.get("event") == "balance_sync"
        and (e.get("ts", "") or "") >= yesterday
        and e.get("portfolio", 0) > 0
    ]

    series = []
    for e in sorted(balance_syncs, key=lambda x: x.get("ts", "")):
        portfolio = e.get("portfolio", 0)
        series.append({
            "ts":             e.get("ts"),
            "bankroll":       round(portfolio, 2),
            "pnl_cumulative": round(portfolio - daily_ref, 2),
            "type":           "sync",
        })

    # Deduplicate by minute
    seen = set()
    deduped = []
    for pt in series:
        key = pt.get("ts", "")[:16]
        if key not in seen:
            seen.add(key)
            deduped.append(pt)

    if not deduped:
        current = state.get("current_bankroll", daily_ref)
        deduped = [{"ts": datetime.now(timezone.utc).isoformat(),
                    "bankroll": round(current, 2),
                    "pnl_cumulative": round(current - daily_ref, 2),
                    "type": "start"}]

    return {
        "starting_bankroll": daily_ref,
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


@app.get("/logs/tail", tags=["Debug"], dependencies=[Depends(check_auth)])
def logs_tail(n: int = Query(default=200, le=2000)):
    """Return the last N lines of the bot output log (human-readable stdout log)."""
    paths_to_try = [
        OUTPUT_LOG,
        Path("/home/polybot/logs/bot.log"),
        BOT_DIR / "bot_output.log",
    ]
    for p in paths_to_try:
        if p.exists():
            try:
                lines = p.read_text(errors="replace").splitlines()
                return {"lines": lines[-n:], "total": len(lines), "path": str(p)}
            except Exception as e:
                continue
    return {"lines": [], "total": 0, "path": "not found"}


# ── Paper mode toggle ─────────────────────────────────────────────────────────
@app.post("/bot/set-mode", tags=["Bot"], dependencies=[Depends(check_auth)])
def set_mode(mode: str = Query(..., description="'paper' or 'live'")):
    """Toggle the bot between paper-only mode and live trading mode."""
    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")
    state = read_state()
    if not state:
        raise HTTPException(status_code=503, detail="Bot state unavailable")
    # paper_mode=True → only paper trades; paper_mode=False → live trades allowed
    state["paper_mode"] = (mode == "paper")
    import tempfile, os
    tmp = str(STATE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, str(STATE_FILE))
    return {"ok": True, "mode": mode, "paper_mode": state["paper_mode"]}


# ── Paper P&L cumulative chart ────────────────────────────────────────────────
@app.get("/chart/paper-pnl", tags=["Charts"], dependencies=[Depends(check_auth)])
def chart_paper_pnl():
    """Cumulative paper wallet P&L, portfolio value, and hourly win/loss buckets."""
    stats = _paper_stats_all()

    def _downsample(s, n=600):
        if len(s) > n:
            step = max(1, len(s) // n)
            s = s[::step]
        return s

    return {
        "points":           len(stats["pnl_series"]),
        "wins":             stats["wins"],
        "losses":           stats["losses"],
        "resolved":         stats["resolved"],
        "win_rate":         stats["win_rate"],
        "series":           _downsample(stats["pnl_series"]),
        "portfolio_series": _downsample(stats["portfolio_series"]),
        "hour_buckets":     stats["hour_buckets"],
    }


@app.get("/chart/live-pnl", tags=["Charts"], dependencies=[Depends(check_auth)])
def chart_live_pnl():
    """Cumulative live wallet P&L, portfolio value, and hourly win/loss buckets (all-time)."""
    stats = _live_stats_all()

    def _downsample(s, n=600):
        if len(s) > n:
            step = max(1, len(s) // n)
            s = s[::step]
        return s

    return {
        "points":            len(stats["pnl_series"]),
        "wins":              stats["wins"],
        "losses":            stats["losses"],
        "resolved":          stats["resolved"],
        "win_rate":          stats["win_rate"],
        "series":            _downsample(stats["pnl_series"]),
        "portfolio_series":  _downsample(stats["portfolio_series"]),
        "hour_buckets":      stats["hour_buckets"],
        "starting_bankroll": stats["starting_bankroll"],
        "current_bankroll":  stats["current_bankroll"],
    }


# ── Paper trade analysis ──────────────────────────────────────────────────────
def _build_analysis() -> Dict:
    """
    Join signal events with paper_position_resolved events by market_id.
    For new trades (after bot enrichment), signal context is in the resolved event directly.
    For historical trades, fall back to the matching signal event in the log.
    """
    signals_by_market: Dict[str, dict] = {}
    resolved: List[dict] = []

    with open(LOG_FILE) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                e = json.loads(raw)
            except Exception:
                continue
            ev = e.get("event", "")
            if ev == "signal" and e.get("paper", True):
                mid = e.get("market_id", "")
                if mid:
                    signals_by_market[mid] = e   # last signal wins (overwrite)
            elif ev == "paper_position_resolved":
                resolved.append(e)

    trades = []
    ev_buckets:       Dict[str, dict] = {}
    regime_buckets:   Dict[str, dict] = {}
    strat_buckets:    Dict[str, dict] = {}
    conf_buckets:     Dict[str, dict] = {}
    symbol_buckets:   Dict[str, dict] = {}
    duration_buckets: Dict[str, dict] = {}
    t_rem_buckets:    Dict[str, dict] = {}
    pct_move_buckets: Dict[str, dict] = {}
    hour_of_day:      Dict[int, dict] = {h: {"wins":0,"losses":0,"pnl":0.0} for h in range(24)}

    for r in resolved:
        mid = r.get("market_id", "")
        sig = signals_by_market.get(mid, {})

        # Prefer fields embedded in resolved record (new format), fall back to signal event
        def _field(key, default=None):
            v = r.get(key)
            if v is not None:
                return v
            return sig.get(key, default)

        ev_val       = _field("ev")
        conf_val     = _field("confidence")
        regime       = _field("regime", "unknown") or "unknown"
        strategy     = _field("strategy", "DIRECTIONAL") or "DIRECTIONAL"
        t_rem        = _field("T_remaining")
        model_p      = _field("model_p")
        kelly        = _field("kelly")
        sigs         = _field("signals") or {}
        symbol       = _field("symbol", "unknown") or "unknown"
        duration_min = _field("duration_min")
        elapsed_min  = _field("elapsed_min")
        crypto_price = _field("crypto_price")
        ref_price    = _field("reference_price")
        pct_move     = _field("pct_move")
        vol_per_min  = _field("vol_per_min")
        ev_yes       = _field("ev_yes")
        ev_no        = _field("ev_no")
        weights      = _field("signal_weights") or {}
        yes_price    = _field("yes_price") or r.get("entry_price")
        no_price     = _field("no_price")
        duration_held = _field("duration_held_min")
        trade_num    = _field("trade_number")
        mkt_end      = _field("market_end_time")
        mkt_start    = _field("market_start_time")
        won          = r.get("won", False)
        pnl          = r.get("pnl", 0.0)
        ts           = r.get("resolved_at") or r.get("ts", "")

        trade = {
            # identity
            "market_id":    mid,
            "question":     r.get("question", "?"),
            "side":         r.get("side", "?"),
            "trade_number": trade_num,
            # outcome
            "won":          won,
            "pnl":          pnl,
            "payout":       r.get("payout"),
            "resolved_at":  ts,
            # position sizing
            "entry_price":  r.get("entry_price"),
            "yes_price":    yes_price,
            "no_price":     no_price,
            "stake":        r.get("stake"),
            "shares":       r.get("shares"),
            # timing
            "entry_time":      _field("entry_time"),
            "duration_held_min": duration_held,
            "T_remaining":     t_rem,
            "elapsed_min":     elapsed_min,
            "duration_min":    duration_min,
            "market_end_time": mkt_end,
            "market_start_time": mkt_start,
            # market context
            "symbol":         symbol,
            "crypto_price":   crypto_price,
            "reference_price": ref_price,
            "pct_move":       pct_move,
            "vol_per_min":    vol_per_min,
            # signal context
            "strategy":       strategy,
            "regime":         regime,
            "ev":             ev_val,
            "ev_yes":         ev_yes,
            "ev_no":          ev_no,
            "model_p":        model_p,
            "confidence":     conf_val,
            "kelly":          kelly,
            "signals":        sigs,
            "signal_weights": weights,
        }
        trades.append(trade)

        def _add(bucket_dict, key_val, label_key):
            if key_val is None:
                return
            bucket_dict.setdefault(key_val, {label_key: key_val, "wins":0, "losses":0, "pnl":0.0, "count":0})
            bucket_dict[key_val]["wins" if won else "losses"] += 1
            bucket_dict[key_val]["count"] += 1
            bucket_dict[key_val]["pnl"] = round(bucket_dict[key_val]["pnl"] + pnl, 4)

        # EV bucket (0.00-0.05, ...)
        if ev_val is not None:
            bk = f"{int(ev_val*20)/20:.2f}-{(int(ev_val*20)/20+0.05):.2f}"
            ev_buckets.setdefault(bk, {"bucket":bk,"wins":0,"losses":0,"pnl":0.0,"count":0})
            ev_buckets[bk]["wins" if won else "losses"] += 1
            ev_buckets[bk]["count"] += 1
            ev_buckets[bk]["pnl"] = round(ev_buckets[bk]["pnl"] + pnl, 4)

        # T_remaining bucket (0-1, 1-2, 2-3, ...)
        if t_rem is not None:
            tbk = f"{int(float(t_rem))}-{int(float(t_rem))+1}min"
            t_rem_buckets.setdefault(tbk, {"bucket":tbk,"wins":0,"losses":0,"pnl":0.0,"count":0})
            t_rem_buckets[tbk]["wins" if won else "losses"] += 1
            t_rem_buckets[tbk]["count"] += 1
            t_rem_buckets[tbk]["pnl"] = round(t_rem_buckets[tbk]["pnl"] + pnl, 4)

        # pct_move bucket (-4%...-2%, -2%...0%, 0%...+2%, ...)
        if pct_move is not None:
            pm_pct = float(pct_move) * 100
            pm_bk  = f"{int(pm_pct/2)*2}% to {int(pm_pct/2)*2+2}%"
            pct_move_buckets.setdefault(pm_bk, {"bucket":pm_bk,"wins":0,"losses":0,"pnl":0.0,"count":0})
            pct_move_buckets[pm_bk]["wins" if won else "losses"] += 1
            pct_move_buckets[pm_bk]["count"] += 1
            pct_move_buckets[pm_bk]["pnl"] = round(pct_move_buckets[pm_bk]["pnl"] + pnl, 4)

        _add(regime_buckets,   regime,        "regime")
        _add(strat_buckets,    strategy,      "strategy")
        _add(symbol_buckets,   symbol,        "symbol")
        if duration_min is not None:
            _add(duration_buckets, str(int(float(duration_min)))+"min", "duration")

        if conf_val is not None:
            cbk = f"{int(float(conf_val)*10)/10:.1f}"
            conf_buckets.setdefault(cbk, {"bucket":cbk,"wins":0,"losses":0,"pnl":0.0,"count":0})
            conf_buckets[cbk]["wins" if won else "losses"] += 1
            conf_buckets[cbk]["count"] += 1
            conf_buckets[cbk]["pnl"] = round(conf_buckets[cbk]["pnl"] + pnl, 4)

        # Hour of day (UTC)
        try:
            h = int(ts[11:13]) if len(ts) >= 13 else 0
            hour_of_day[h]["wins" if won else "losses"] += 1
            hour_of_day[h]["pnl"] = round(hour_of_day[h]["pnl"] + pnl, 4)
        except Exception:
            pass

    # Add win_rate to all bucket dicts
    def _wr(b): return round(b["wins"] / b["count"], 4) if b["count"] else None
    for b in ev_buckets.values():       b["win_rate"] = _wr(b)
    for b in regime_buckets.values():   b["win_rate"] = _wr(b)
    for b in strat_buckets.values():    b["win_rate"] = _wr(b)
    for b in symbol_buckets.values():   b["win_rate"] = _wr(b)
    for b in duration_buckets.values(): b["win_rate"] = _wr(b)
    for b in t_rem_buckets.values():    b["win_rate"] = _wr(b)
    for b in pct_move_buckets.values(): b["win_rate"] = _wr(b)
    for b in conf_buckets.values(): b["win_rate"] = _wr(b)
    for h, b in hour_of_day.items():
        total = b["wins"] + b["losses"]
        b["win_rate"] = round(b["wins"] / total, 4) if total else None
        b["count"] = total
        b["hour"] = h

    total     = len(trades)
    wins      = sum(1 for t in trades if t["won"])
    total_pnl = round(sum(t["pnl"] for t in trades), 4)
    enriched  = sum(1 for t in trades if t.get("ev") is not None)
    avg_hold  = None
    hold_vals = [t["duration_held_min"] for t in trades if t.get("duration_held_min") is not None]
    if hold_vals:
        avg_hold = round(sum(hold_vals) / len(hold_vals), 2)

    return {
        "total":            total,
        "wins":             wins,
        "losses":           total - wins,
        "win_rate":         round(wins / total, 4) if total else None,
        "total_pnl":        total_pnl,
        "enriched":         enriched,
        "avg_hold_min":     avg_hold,
        "by_ev":            sorted(ev_buckets.values(), key=lambda x: x["bucket"]),
        "by_regime":        sorted(regime_buckets.values(), key=lambda x: -x["count"]),
        "by_strategy":      sorted(strat_buckets.values(), key=lambda x: -x["count"]),
        "by_symbol":        sorted(symbol_buckets.values(), key=lambda x: -x["count"]),
        "by_duration":      sorted(duration_buckets.values(), key=lambda x: x["duration"]),
        "by_confidence":    sorted(conf_buckets.values(), key=lambda x: x["bucket"]),
        "by_t_remaining":   sorted(t_rem_buckets.values(), key=lambda x: x["bucket"]),
        "by_pct_move":      sorted(pct_move_buckets.values(), key=lambda x: x["bucket"]),
        "by_hour":          [hour_of_day[h] for h in range(24)],
        "trades":           trades[-500:],
    }


@app.get("/analysis/paper", tags=["Analysis"], dependencies=[Depends(check_auth)])
def analysis_paper():
    """
    Deep per-trade analysis: joins signal context with resolution outcome.
    Returns win-rate breakdowns by EV, regime, strategy, confidence, hour.
    """
    return _build_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    print(f"\n  Polymarket Dashboard API — http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
