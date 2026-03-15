#!/usr/bin/env python3
"""
Local monitoring dashboard for the Polymarket trading bot.

Proxies the remote API and provides SSH controls to start/stop the bot.

Usage:
    pip install fastapi uvicorn httpx paramiko jinja2 python-dotenv
    python dashboard.py
    # Open http://localhost:3000
"""

import os
from typing import Optional

import httpx
import paramiko
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
SSH_HOST        = os.environ.get("SSH_HOST",     "31.97.93.12")
SSH_USER        = os.environ.get("SSH_USER",     "root")
SSH_PASSWORD    = os.environ.get("SSH_PASSWORD", "atJH;3j5qsIzw2p'")
REMOTE_API      = os.environ.get("REMOTE_API",   "http://31.97.93.12:8000")
API_TOKEN       = os.environ.get("API_TOKEN",    "ebc8cae89f6a3773666d3633326da5b5321a280998811926")
DASHBOARD_PORT  = int(os.environ.get("DASHBOARD_PORT", 3000))

# ── SSH helper ──────────────────────────────────────────────────────
def ssh_run(command: str) -> tuple[int, str, str]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(SSH_HOST, username=SSH_USER, password=SSH_PASSWORD, timeout=10)
        _, stdout, stderr = client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode(), stderr.read().decode()
    finally:
        client.close()

# ── Remote API proxy ────────────────────────────────────────────────
async def remote_get(path: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{REMOTE_API}{path}",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
        resp.raise_for_status()
        return resp.json()

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Polymarket Dashboard", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── API proxy routes ─────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    try:
        return await remote_get("/status")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/metrics")
async def metrics():
    try:
        return await remote_get("/metrics")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/trades")
async def trades(n: int = Query(default=30)):
    try:
        return await remote_get("/trades", {"n": n})
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/positions")
async def positions():
    try:
        return await remote_get("/positions")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/chart/pnl")
async def chart_pnl():
    try:
        return await remote_get("/chart/pnl")
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/logs")
async def logs():
    try:
        _, stdout, _ = ssh_run("tail -60 /home/polybot/app/bot_output.log 2>/dev/null || echo 'No log file found'")
        return {"lines": stdout.strip().split("\n")}
    except Exception as e:
        raise HTTPException(502, str(e))


# ── Bot control routes ───────────────────────────────────────────────
@app.get("/bot/service-status")
async def service_status():
    try:
        code, out, _ = ssh_run("systemctl is-active polymarket-bot")
        active = out.strip() == "active"
        _, unit, _ = ssh_run("cat /etc/systemd/system/polymarket-bot.service")
        live_mode = "--live" in unit
        return {"running": active, "live_mode": live_mode}
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/bot/start")
async def bot_start(mode: str = "paper"):
    try:
        exec_line = "/home/polybot/venv/bin/python polymarket_bot.py"
        if mode == "live":
            exec_line += " --live"

        ssh_run(
            f"sed -i 's|ExecStart=.*polymarket_bot.py.*|ExecStart={exec_line}|' "
            f"/etc/systemd/system/polymarket-bot.service && systemctl daemon-reload"
        )
        code, out, err = ssh_run("systemctl restart polymarket-bot")
        return {"ok": code == 0, "mode": mode, "output": out or err}
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/bot/stop")
async def bot_stop():
    try:
        code, out, err = ssh_run("systemctl stop polymarket-bot")
        return {"ok": code == 0, "output": out or err}
    except Exception as e:
        raise HTTPException(502, str(e))


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"\n  Polymarket Local Dashboard")
    print(f"  Open http://localhost:{DASHBOARD_PORT}\n")
    uvicorn.run(app, host="127.0.0.1", port=DASHBOARD_PORT, log_level="warning")
