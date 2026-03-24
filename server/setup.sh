#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Polymarket Bot — VPS Setup Script
#  Run once on a fresh Ubuntu/Debian VPS to deploy the server stack.
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

APP_DIR="/home/polybot/app"
VENV_DIR="/home/polybot/venv"
SERVICE_BOT="polybot"
SERVICE_API="polybot-api"
PYTHON="python3"

echo ""
echo "══════════════════════════════════════════"
echo "  Polymarket Bot — Server Setup"
echo "══════════════════════════════════════════"
echo ""

# ── 1. System packages ────────────────────────────────────────────
echo "[1/6] Installing system packages…"
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl

# ── 2. Create bot user ────────────────────────────────────────────
echo "[2/6] Creating polybot user…"
if ! id -u polybot &>/dev/null; then
    useradd -m -s /bin/bash polybot
    echo "  ✓ User polybot created"
else
    echo "  ✓ User polybot already exists"
fi

# ── 3. App directory ──────────────────────────────────────────────
echo "[3/6] Setting up app directory…"
mkdir -p "$APP_DIR"
cp -n "$(dirname "$0")"/*.py "$APP_DIR/" 2>/dev/null || true
cp -n "$(dirname "$0")"/requirements.txt "$APP_DIR/" 2>/dev/null || true
chown -R polybot:polybot /home/polybot

# ── 4. Python virtual environment ─────────────────────────────────
echo "[4/6] Creating virtual environment and installing dependencies…"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
echo "  ✓ Dependencies installed"

# ── 5. Environment file ────────────────────────────────────────────
echo "[5/6] Checking .env file…"
if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$(dirname "$0")/.env.example" ]; then
        cp "$(dirname "$0")/.env.example" "$APP_DIR/.env"
    fi
    echo ""
    echo "  ⚠️  IMPORTANT: Edit $APP_DIR/.env and fill in:"
    echo "     PRIVATE_KEY=0x..."
    echo "     FUNDER_ADDRESS=0x..."
    echo "     API_TOKEN=your_secret"
    echo ""
fi

# ── 6. Systemd services ───────────────────────────────────────────
echo "[6/6] Installing systemd services…"

# polybot — the trading bot
cat > "/etc/systemd/system/${SERVICE_BOT}.service" <<EOF
[Unit]
Description=Polymarket Trading Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=polybot
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${VENV_DIR}/bin/python polymarket_bot.py --live
Restart=on-failure
RestartSec=15
StandardOutput=append:${APP_DIR}/bot_output.log
StandardError=append:${APP_DIR}/bot_output.log

[Install]
WantedBy=multi-user.target
EOF

# polybot-api — the dashboard REST API
cat > "/etc/systemd/system/${SERVICE_API}.service" <<EOF
[Unit]
Description=Polymarket Bot Dashboard API
After=network.target

[Service]
Type=simple
User=polybot
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${VENV_DIR}/bin/python api_server.py --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10
StandardOutput=append:${APP_DIR}/api_output.log
StandardError=append:${APP_DIR}/api_output.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_BOT}"
systemctl enable "${SERVICE_API}"

echo ""
echo "══════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Edit ${APP_DIR}/.env with your credentials"
echo "    2. Start the API:  sudo systemctl start ${SERVICE_API}"
echo "    3. Check API:      sudo systemctl status ${SERVICE_API}"
echo "    4. Start the bot:  sudo systemctl start ${SERVICE_BOT}"
echo "    5. Check bot:      sudo systemctl status ${SERVICE_BOT}"
echo "    6. View logs:      tail -f ${APP_DIR}/bot_output.log"
echo ""
echo "  Paper mode (safe default): bot runs without --live flag"
echo "  Edit the service file to add --live for real trading."
echo "══════════════════════════════════════════"
echo ""
