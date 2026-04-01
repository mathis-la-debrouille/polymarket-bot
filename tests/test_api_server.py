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
    "paper_mode": True,
    "paper_bankroll": 98.0,
    "paper_starting": 100.0,
    "paper_total_pnl": -2.0,
    "paper_trades": 5,
    "paper_active_positions": {},
    "paper_traded_markets": [],
    "paper_daily_start": 100.0,
    "paper_daily_reset": "",
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
    {"event": "paper_position_resolved", "ts": "2026-03-24T10:05:00+00:00",
     "market_id": "p1", "pnl": 0.8, "won": True, "paper": True},
    {"event": "paper_position_resolved", "ts": "2026-03-24T10:10:00+00:00",
     "market_id": "p2", "pnl": -1.0, "won": False, "paper": True},
    {"event": "paper_position_resolved", "ts": "2026-03-24T10:15:00+00:00",
     "market_id": "p3", "pnl": 0.9, "won": True, "paper": True},
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


# ── /chart/paper-pnl ─────────────────────────────────────────────────────────

def test_chart_paper_pnl(api_client):
    """Paper P&L chart returns series, portfolio, win/loss buckets."""
    r = auth(api_client, "/chart/paper-pnl")
    assert r.status_code == 200
    d = r.json()
    assert "series" in d
    assert "portfolio_series" in d
    assert "hour_buckets" in d
    assert "win_rate" in d
    assert "resolved" in d
    # 3 paper events in fixture: 2 wins, 1 loss
    assert d["resolved"] == 3
    assert d["wins"] == 2
    assert d["losses"] == 1
    assert abs(d["win_rate"] - 0.667) < 0.01
    # Last cumulative pnl = 0.8 - 1.0 + 0.9 = 0.7
    assert abs(d["series"][-1]["pnl"] - 0.7) < 0.01
    # Portfolio = 100 + cumulative pnl
    assert abs(d["portfolio_series"][-1]["portfolio"] - 100.7) < 0.01
    # hour_buckets must have wins+losses per bucket
    assert isinstance(d["hour_buckets"], list)
    total_wins   = sum(b["wins"]   for b in d["hour_buckets"])
    total_losses = sum(b["losses"] for b in d["hour_buckets"])
    assert total_wins == 2
    assert total_losses == 1


# ── /metrics paper fields ─────────────────────────────────────────────────────

def test_metrics_paper_fields(api_client):
    """Metrics endpoint must include accurate paper wallet fields from full log scan."""
    r = auth(api_client, "/metrics")
    assert r.status_code == 200
    d = r.json()
    assert "paper_bankroll" in d
    assert "paper_win_rate" in d
    assert "paper_resolved" in d
    assert "live_mode" in d
    # paper_mode=True in state → live_mode should be False
    assert d["live_mode"] is False
    # Win rate from full log: 2W / 3 resolved = 0.667
    assert d["paper_resolved"] == 3
    assert abs(d["paper_win_rate"] - 0.667) < 0.01


# ── /bot/set-mode ─────────────────────────────────────────────────────────────

def test_set_mode_paper(api_client, tmp_bot_dir):
    """POST /bot/set-mode?mode=paper sets paper_mode=True in state file."""
    r = api_client.post("/bot/set-mode", params={"mode": "paper"},
                        headers={"Authorization": "Bearer testtoken"})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["paper_mode"] is True
    state = json.loads((tmp_bot_dir / "bot_state.json").read_text())
    assert state["paper_mode"] is True

def test_set_mode_live(api_client, tmp_bot_dir):
    """POST /bot/set-mode?mode=live sets paper_mode=False in state file."""
    r = api_client.post("/bot/set-mode", params={"mode": "live"},
                        headers={"Authorization": "Bearer testtoken"})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["paper_mode"] is False
    state = json.loads((tmp_bot_dir / "bot_state.json").read_text())
    assert state["paper_mode"] is False

def test_set_mode_invalid(api_client):
    """Invalid mode value should return 400."""
    r = api_client.post("/bot/set-mode", params={"mode": "invalid"},
                        headers={"Authorization": "Bearer testtoken"})
    assert r.status_code == 400
