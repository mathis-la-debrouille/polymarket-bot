#!/usr/bin/env python3
"""
Local monitoring dashboard for the Polymarket trading bot.

Proxies the remote API, streams live logs via SSE, and provides SSH bot controls.

Usage:
    pip install -r requirements.txt
    python dashboard.py
    # Open http://localhost:3000
"""

import asyncio
import json as _json
import os
from pathlib import Path
from typing import Optional

import httpx
import paramiko
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env")

# ── Config ───────────────────────────────────────────────────────────────────
SSH_HOST       = os.environ.get("SSH_HOST",      "YOUR_SERVER_IP")
SSH_USER       = os.environ.get("SSH_USER",      "root")
SSH_PASSWORD   = os.environ.get("SSH_PASSWORD",  "")
REMOTE_API     = os.environ.get("REMOTE_API",    "http://YOUR_SERVER_IP:8000")
API_TOKEN      = os.environ.get("API_TOKEN",     "")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", 3000))

BOT_LOG   = "/home/polybot/logs/bot.log"
BOT_SVC   = "polymarket-bot"
API_SVC   = "polymarket-api"

# ── SSH helper ────────────────────────────────────────────────────────────────
def ssh_run(command: str, timeout: int = 15) -> tuple[int, str, str]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(SSH_HOST, username=SSH_USER, password=SSH_PASSWORD, timeout=timeout)
        _, stdout, stderr = client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")
    finally:
        client.close()

# ── Remote API proxy ──────────────────────────────────────────────────────────
async def remote_get(path: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{REMOTE_API}{path}",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
        resp.raise_for_status()
        return resp.json()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Polymarket Dashboard", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(_HERE / "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── Metrics proxy ─────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    try:
        return await remote_get("/status")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/metrics")
async def api_metrics():
    try:
        return await remote_get("/metrics")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/trades")
async def api_trades(n: int = Query(default=50)):
    try:
        return await remote_get("/trades", {"n": n})
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/positions")
async def api_positions():
    try:
        return await remote_get("/positions")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/chart/pnl")
async def api_chart_pnl():
    try:
        return await remote_get("/chart/pnl")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/kpi")
async def api_kpi():
    try:
        return await remote_get("/kpi")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/signals/recent")
async def api_signals_recent(n: int = Query(default=100)):
    try:
        return await remote_get("/signals/recent", {"n": n})
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/debug/state")
async def api_debug_state():
    try:
        return await remote_get("/debug/state")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/debug/events")
async def api_debug_events(event: str = Query(default=""), n: int = Query(default=100)):
    try:
        return await remote_get("/debug/events", {"event": event, "n": n})
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/debug/scans")
async def api_debug_scans(n: int = Query(default=20)):
    try:
        return await remote_get("/debug/scans", {"n": n})
    except Exception as e:
        raise HTTPException(502, str(e))


# ── Live log streaming (SSE) ──────────────────────────────────────────────────
@app.get("/stream/logs")
async def stream_logs(request: Request):
    """
    Server-Sent Events endpoint that streams the bot log file in real time.
    Sends the last 200 lines first, then follows new output.
    """
    async def event_gen():
        proc = None
        try:
            # tail -n 200 -f: last 200 historical lines + follow
            proc = await asyncio.create_subprocess_exec(
                "sshpass", "-p", SSH_PASSWORD,
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=15",
                "-o", "ConnectTimeout=10",
                f"{SSH_USER}@{SSH_HOST}",
                f"tail -n 200 -f {BOT_LOG}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            async for raw in proc.stdout:
                if await request.is_disconnected():
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line:
                    yield f"data: {_json.dumps(line)}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps(f'[stream error] {e}')}\n\n"
        finally:
            if proc and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Bot controls ──────────────────────────────────────────────────────────────
@app.get("/bot/service-status")
async def service_status():
    try:
        _, out, _ = ssh_run(
            f"systemctl is-active {BOT_SVC}; systemctl is-active {API_SVC}; "
            f"grep -o -- '--live' /etc/systemd/system/{BOT_SVC}.service 2>/dev/null || echo 'paper'"
        )
        lines = out.strip().split("\n")
        bot_active = lines[0].strip() == "active" if len(lines) > 0 else False
        api_active = lines[1].strip() == "active" if len(lines) > 1 else False
        live_mode  = "--live" in (lines[2] if len(lines) > 2 else "")
        return {"bot_running": bot_active, "api_running": api_active, "live_mode": live_mode}
    except Exception as e:
        return {"bot_running": False, "api_running": False, "live_mode": False, "error": str(e)}


@app.post("/bot/stop")
async def bot_stop():
    try:
        code, out, err = ssh_run(f"systemctl stop {BOT_SVC}")
        return {"ok": True, "output": "Stopped"}
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/bot/restart")
async def bot_restart():
    try:
        code, out, err = ssh_run(f"systemctl restart {BOT_SVC}")
        return {"ok": code == 0, "output": "Restarted" if code == 0 else err}
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/bot/start")
async def bot_start(mode: str = "paper"):
    try:
        _, existing, _ = ssh_run(f"systemctl is-active {BOT_SVC}")
        if existing.strip() == "active":
            return {"ok": False, "output": "Bot is already running"}
        code, out, err = ssh_run(f"systemctl start {BOT_SVC}")
        return {"ok": code == 0, "output": "Started"}
    except Exception as e:
        raise HTTPException(502, str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"\n  Polymarket Dashboard  →  http://localhost:{DASHBOARD_PORT}\n")
    uvicorn.run(app, host="127.0.0.1", port=DASHBOARD_PORT, log_level="warning")
