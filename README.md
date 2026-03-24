# Polymarket Bot

A production-ready template for building automated trading bots on [Polymarket](https://polymarket.com). Handles all the hard infrastructure so you can focus on your edge.

---

## What this is

Two independent applications that work together:

| App | Location | Runs on | Purpose |
|-----|----------|---------|---------|
| **Server** | `server/` | VPS | Trading bot + REST API |
| **Dashboard** | `dashboard/` | Your machine | Local monitoring UI |

The **server** runs continuously on a remote VPS. It scans Polymarket markets, computes signals, places orders via the CLOB API, tracks positions, and exposes a REST API with your bot's state.

The **dashboard** runs locally on your laptop. It connects to the server over SSH and the REST API, and gives you a live web UI at `http://localhost:3000` to monitor P&L, open positions, recent signals, and live logs.

---

## Architecture

```
Your laptop                          VPS (Brazil / any)
─────────────                        ─────────────────────────────
dashboard/                           server/
  dashboard.py  ←── HTTP API ──────▶  api_server.py   (port 8000)
  (port 3000)   ←── SSH ──────────▶  polymarket_bot.py
                                       signal_updown.py
                                       bot_state.json
                                       bot_log.jsonl
```

- `polymarket_bot.py` — core bot loop: wallet auth, balance sync, position tracking, P&L, kill switch, order execution
- `signal_updown.py` — signal engine: Brownian motion, order flow imbalance, momentum hybrid, Monte Carlo GBM, spread arbitrage, oracle latency
- `api_server.py` — FastAPI server exposing bot state (status, metrics, trades, positions, P&L chart)
- `dashboard.py` — local proxy that serves the web UI and relays API calls + SSH bot controls

---

## Quick start

### 1. Deploy the server (VPS)

```bash
# Clone the repo on your VPS
git clone https://github.com/mathis-la-debrouille/polymarket-bot.git
cd polymarket-bot/server

# Run the setup script (Ubuntu/Debian)
chmod +x setup.sh
sudo ./setup.sh
```

The script will:
- Install Python and create the `polybot` system user
- Set up a virtual environment and install all dependencies
- Create two systemd services: `polybot` (bot) and `polybot-api` (REST API)

Then configure your credentials:

```bash
nano /home/polybot/app/.env
```

```env
PRIVATE_KEY=0x...          # your EOA private key
FUNDER_ADDRESS=0x...       # your Polymarket proxy wallet
API_TOKEN=your_secret      # protects the REST API
BOT_DIR=/home/polybot/app
```

> **Find your proxy wallet:** call `getPolyProxyWalletAddress(your_eoa)` on the CTFExchange contract (`0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`, selector `0xedef7d8e`) via Polygonscan.

Start the services:

```bash
sudo systemctl start polybot-api    # start REST API first
sudo systemctl start polybot        # start the trading bot
sudo systemctl status polybot       # verify running
tail -f /home/polybot/app/bot_output.log
```

**Paper mode (default):** the bot runs without placing real orders. To enable live trading, edit the service file:

```bash
sudo nano /etc/systemd/system/polybot.service
# Change: ExecStart=... polymarket_bot.py --live
sudo systemctl daemon-reload && sudo systemctl restart polybot
```

---

### 2. Set up the local dashboard

```bash
cd polymarket-bot/dashboard
chmod +x setup.sh
./setup.sh
```

Configure `.env`:

```env
SSH_HOST=YOUR_SERVER_IP
SSH_USER=root
SSH_PASSWORD=your_password
REMOTE_API=http://YOUR_SERVER_IP:8000
API_TOKEN=your_secret      # same as on the server
DASHBOARD_PORT=3000
```

Start:

```bash
source .venv/bin/activate
python dashboard.py
```

Open [http://localhost:3000](http://localhost:3000).

---

## Server app — what it does

### Bot (`polymarket_bot.py`)

Runs on a configurable interval (default 30s). Each scan:

1. **Resolves open positions** — checks Gamma API for settled markets, computes P&L
2. **Stop-loss check** — sells positions down >30% if edge is gone
3. **Balance sync** — fetches live USDC balance from Polygon/CLOB
4. **Kill switch** — halts if drawdown exceeds threshold (default 30%)
5. **Strategy** — calls `signal_updown.py` on all active crypto Up/Down markets

CLI options:

```bash
python polymarket_bot.py              # paper mode (no real money)
python polymarket_bot.py --live       # live trading
python polymarket_bot.py --once       # run one scan then exit (cron)
python polymarket_bot.py --bankroll 100 --max-stake 5 --ev-min 0.05
```

### Signal engine (`signal_updown.py`)

Targets 5-minute and 15-minute BTC/ETH/SOL/XRP/DOGE Up/Down markets.

| Signal | Weight | Source |
|--------|--------|--------|
| Brownian Motion | 30% | Binance spot price vs window reference |
| Order Flow Imbalance | 25% | Binance aggTrades (last 30s) |
| Momentum Hybrid | 20% | Fade 1-min move + follow 4-min trend |
| Monte Carlo GBM | 15% | 2000-path vectorized simulation |
| Regime gate | — | Adjusts weights in choppy/trending markets |
| Oracle Latency | bypass | Final 90s: exploit price update lag |
| Spread Arbitrage | bypass | YES+NO < $0.96 → buy both sides |

Entry conditions: EV ≥ 4%, confidence ≥ 40%, 1–3.5 minutes remaining.

### REST API (`api_server.py`)

Protected by a bearer token. Endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /status` | Bankroll, uptime, active positions |
| `GET /metrics` | P&L, drawdown, win rate, signals count |
| `GET /trades` | Recent trade signals |
| `GET /positions` | Open positions with live prices |
| `GET /kpi` | Win/loss stats, avg EV, avg stake |
| `GET /chart/pnl` | Cumulative P&L time series |

---

## Dashboard app — what it does

Local FastAPI app at `http://localhost:3000`. It:

- Proxies all API calls to the remote `api_server.py` (with auth)
- Provides SSH-based bot controls (start / stop / status)
- Serves the web UI from `templates/index.html`
- Tails live bot logs via SSH

Dashboard features:
- Portfolio value, P&L, drawdown, win rate, signals count — auto-refreshing every 15s
- Cumulative P&L chart
- Recent signals table with outcomes
- Open positions with live unrealized P&L
- Live log output

---

## Recommendations

**On the VPS:**
- Use a dedicated server in a low-latency region (São Paulo for Polymarket is fine)
- Keep `polybot-api` always running — the dashboard depends on it
- Monitor disk space: `bot_log.jsonl` and `bot_output.log` grow continuously. Add a logrotate rule.
- Never commit `.env` to git — it contains your private key

**On strategy:**
- Always run in paper mode first for at least a few hours before going live
- Watch the `confidence` field — below 0.40 the bot will not trade
- The oracle latency signal is your highest-edge play (final 90 seconds)
- The regime gate automatically reduces momentum weight in choppy markets

**On wallet:**
- Keep only what you're willing to risk on the bot's proxy wallet
- The kill switch at 30% drawdown is a hard stop — verify it works in paper mode
- `FUNDER_ADDRESS` must be the proxy wallet, not your EOA

---

## File structure

```
polymarket-bot/
├── server/                        # Runs on the VPS
│   ├── polymarket_bot.py          # Bot loop, wallet, orders, state
│   ├── signal_updown.py           # Signal engine (replace to add your strategy)
│   ├── api_server.py              # REST API exposing bot state
│   ├── requirements.txt
│   ├── .env.example
│   └── setup.sh                   # One-command VPS deploy
│
├── dashboard/                     # Runs locally
│   ├── dashboard.py               # Local proxy + SSH controls
│   ├── templates/
│   │   └── index.html             # Web UI
│   ├── requirements.txt
│   └── setup.sh                   # Local setup
│
└── README.md
```

---

## Replacing the signal engine

`signal_updown.py` is the only file you need to replace to change strategy. The interface:

```python
def compute_updown_signal(market: dict, yes_midprice: float) -> dict:
    return {
        "model_p":        float,   # 0–1, your P(UP) estimate
        "confidence":     float,   # 0–1, signal agreement
        "ev_yes":         float,   # expected value of YES bet
        "ev_no":          float,   # expected value of NO bet
        "side":           str,     # "YES", "NO", "PASS", or "BOTH"
        "kelly_fraction": float,   # fraction of bankroll to bet
        "signals":        dict,    # individual signal values (for logging)
        "strategy":       str,     # "DIRECTIONAL", "ARB", "ORACLE_LATENCY", "PASS"
    }
```

The bot also expects `is_updown_market(question)` and `start_rtds_stream()` to be importable from `signal_updown`.
