"""
Unit + integration tests for api_server.py and dashboard proxy.

Run:
    pip install pytest httpx fastapi
    pytest tests/ -v
"""
import json
import math
import os
import sys
import tempfile
import pathlib
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_STATE = {
    "starting_bankroll": 10.0,
    "bankroll": 9.5,
    "paper": True,
    "active_positions": {
        "111": {
            "question": "BTC Up or Down?",
            "side": "YES",
            "token_id": "",
            "price": 0.5,
            "stake": 1.0,
            "ev": 0.1,
            "paper": True,
            "shares": 2,
            "entry_time": "2026-03-24T10:00:00+00:00",
            "signal_type": "DIRECTIONAL",
        },
        "222": {
            "question": "ETH Up or Down?",
            "side": "NO",
            "token_id": "",
            "price": 0.6,
            "stake": 1.0,
            "ev": 0.08,
            "paper": True,
            "shares": None,       # shares=None → should compute from stake/price
            "entry_time": None,   # None entry_time → was causing sort crash
            "signal_type": "ORACLE_LATENCY",
        },
        "333": {
            "question": "SOL Up?",
            "side": "YES",
            "token_id": "",
            "price": 0.0,         # zero price edge case
            "stake": 1.0,
            "ev": 0.05,
            "paper": True,
            "shares": 0,
            "entry_time": None,
        },
    },
    "total_pnl": -0.5,
    "peak_bankroll": 10.0,
    "uptime_start": "2026-03-24T09:00:00+00:00",
}

SAMPLE_LOG = [
    {"event": "signal", "ts": "2026-03-24T10:00:00+00:00", "market_id": "111",
     "market": "BTC Up or Down?", "side": "YES", "price": 0.5, "model_p": 0.6,
     "confidence": 0.55, "ev": 0.1, "stake": 1.0, "kelly": 0.05,
     "strategy": "DIRECTIONAL", "regime": "normal", "T_remaining": 2.5,
     "paper": True, "signals": {"bm": 0.6, "ofi": 0.55, "mom": 0.5, "mc": 0.58}},
    {"event": "position_resolved", "ts": "2026-03-24T10:05:00+00:00",
     "market_id": "000", "pnl": 0.5, "paper": True},
    {"event": "scan_complete", "ts": "2026-03-24T10:01:00+00:00",
     "markets_scanned": 10, "signals_fired": 1, "paper": True},
    {"event": "balance_sync", "ts": "2026-03-24T10:00:30+00:00",
     "bankroll": 9.5, "paper": True},
]


@pytest.fixture
def tmp_bot_dir(tmp_path):
    state_file = tmp_path / "bot_state.json"
    log_file   = tmp_path / "bot_log.jsonl"
    state_file.write_text(json.dumps(SAMPLE_STATE))
    log_file.write_text("\n".join(json.dumps(e) for e in SAMPLE_LOG) + "\n")
    return tmp_path


@pytest.fixture
def api_client(tmp_bot_dir, monkeypatch):
    """TestClient for api_server with temp state/log files and fixed token."""
    monkeypatch.setenv("BOT_DIR",   str(tmp_bot_dir))
    monkeypatch.setenv("API_TOKEN", "testtoken")

    # Force module reload so env vars are picked up
    if "api_server" in sys.modules:
        del sys.modules["api_server"]

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "server"))
    import api_server
    from fastapi.testclient import TestClient
    return TestClient(api_server.app)


# ── Helper ────────────────────────────────────────────────────────────────────

def auth(client, path, **params):
    return client.get(path, params=params, headers={"Authorization": "Bearer testtoken"})


# ── Auth ─────────────────────────────────────────────────────────────────────

def test_auth_required(api_client):
    r = api_client.get("/metrics")
    assert r.status_code == 401

def test_auth_invalid_token(api_client):
    r = api_client.get("/metrics", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401

def test_health_no_auth(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /metrics ──────────────────────────────────────────────────────────────────

def test_metrics(api_client):
    r = auth(api_client, "/metrics")
    assert r.status_code == 200
    d = r.json()
    assert "current_bankroll" in d
    assert "total_pnl" in d
    assert "drawdown_pct" in d


# ── /status ───────────────────────────────────────────────────────────────────

def test_status(api_client):
    r = auth(api_client, "/status")
    assert r.status_code == 200
    d = r.json()
    assert "current_bankroll" in d
    assert "active_positions" in d


# ── /positions ────────────────────────────────────────────────────────────────

def test_positions_ok(api_client):
    """Must not crash even when entry_time is None or shares is None."""
    r = auth(api_client, "/positions")
    assert r.status_code == 200, r.text
    d = r.json()
    assert "positions" in d
    assert d["count"] == 3

def test_positions_null_entry_time_no_crash(api_client):
    """Regression: entry_time=None was causing sort TypeError."""
    r = auth(api_client, "/positions")
    assert r.status_code == 200

def test_positions_null_shares_computed(api_client):
    """shares=None should be computed as ceil(stake/price)."""
    r = auth(api_client, "/positions")
    pos = {p["market_id"]: p for p in r.json()["positions"]}
    # 222: stake=1.0, price=0.6 → ceil(1.0/0.6)=2
    assert pos["222"]["shares"] == math.ceil(1.0 / 0.6)

def test_positions_zero_price_no_crash(api_client):
    """price=0 should not cause division by zero."""
    r = auth(api_client, "/positions")
    assert r.status_code == 200
    pos = {p["market_id"]: p for p in r.json()["positions"]}
    assert pos["333"]["shares"] == 0


# ── /trades ───────────────────────────────────────────────────────────────────

def test_trades(api_client):
    r = auth(api_client, "/trades")
    assert r.status_code == 200
    assert "trades" in r.json()


# ── /kpi ──────────────────────────────────────────────────────────────────────

def test_kpi(api_client):
    r = auth(api_client, "/kpi")
    assert r.status_code == 200
    d = r.json()
    assert "win_rate" in d
    assert "total_resolved" in d


# ── /chart/pnl ───────────────────────────────────────────────────────────────

def test_chart_pnl(api_client):
    r = auth(api_client, "/chart/pnl")
    assert r.status_code == 200
    d = r.json()
    assert "series" in d
    assert "starting_bankroll" in d


# ── /signals/recent ──────────────────────────────────────────────────────────

def test_signals_recent(api_client):
    r = auth(api_client, "/signals/recent", n=50)
    assert r.status_code == 200
    d = r.json()
    assert "signals" in d
    assert d["count"] >= 1
    sig = d["signals"][0]
    assert "bm" in sig   # sub-signal breakdown present


# ── /debug/state ─────────────────────────────────────────────────────────────

def test_debug_state(api_client):
    r = auth(api_client, "/debug/state")
    assert r.status_code == 200
    assert "bankroll" in r.json()


# ── /debug/events ────────────────────────────────────────────────────────────

def test_debug_events_all(api_client):
    r = auth(api_client, "/debug/events", n=100)
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == len(SAMPLE_LOG)

def test_debug_events_filtered(api_client):
    r = auth(api_client, "/debug/events", event="signal", n=100)
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == 1
    assert d["events"][0]["event"] == "signal"


# ── /debug/scans ─────────────────────────────────────────────────────────────

def test_debug_scans(api_client):
    r = auth(api_client, "/debug/scans", n=10)
    assert r.status_code == 200
    d = r.json()
    assert "scans" in d
    assert d["count"] == 1
