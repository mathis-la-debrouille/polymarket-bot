#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Polymarket Bot — Local Dashboard Setup Script
#  Run on your local machine to set up the monitoring dashboard.
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

DASHBOARD_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${DASHBOARD_DIR}/.venv"
PYTHON="python3"

echo ""
echo "══════════════════════════════════════════"
echo "  Polymarket Bot — Local Dashboard Setup"
echo "══════════════════════════════════════════"
echo ""

# ── 1. Python check ───────────────────────────────────────────────
echo "[1/4] Checking Python…"
if ! command -v $PYTHON &>/dev/null; then
    echo "  ✗ python3 not found. Install it from https://python.org"
    exit 1
fi
PY_VER=$($PYTHON --version 2>&1)
echo "  ✓ $PY_VER"

# ── 2. Virtual environment ────────────────────────────────────────
echo "[2/4] Creating virtual environment…"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    echo "  ✓ Virtual env created at $VENV_DIR"
else
    echo "  ✓ Virtual env already exists"
fi

# ── 3. Dependencies ───────────────────────────────────────────────
echo "[3/4] Installing dependencies…"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "${DASHBOARD_DIR}/requirements.txt"
echo "  ✓ Dependencies installed"

# ── 4. Environment file ───────────────────────────────────────────
echo "[4/4] Checking .env file…"
if [ ! -f "${DASHBOARD_DIR}/.env" ]; then
    if [ -f "${DASHBOARD_DIR}/../server/.env.example" ]; then
        cp "${DASHBOARD_DIR}/../server/.env.example" "${DASHBOARD_DIR}/.env"
    else
        cat > "${DASHBOARD_DIR}/.env" <<'ENVEOF'
SSH_HOST=YOUR_SERVER_IP
SSH_USER=root
SSH_PASSWORD=your_ssh_password
REMOTE_API=http://YOUR_SERVER_IP:8000
API_TOKEN=your_secret_token_here
DASHBOARD_PORT=3000
ENVEOF
    fi
    echo ""
    echo "  ⚠️  IMPORTANT: Edit ${DASHBOARD_DIR}/.env and fill in:"
    echo "     SSH_HOST      — your VPS IP address"
    echo "     SSH_PASSWORD  — your VPS root password"
    echo "     REMOTE_API    — http://YOUR_VPS_IP:8000"
    echo "     API_TOKEN     — same token as on the VPS"
    echo ""
fi

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Start dashboard:"
echo "    ${VENV_DIR}/bin/python ${DASHBOARD_DIR}/dashboard.py"
echo ""
echo "  Or activate the env first:"
echo "    source ${VENV_DIR}/bin/activate"
echo "    python dashboard.py"
echo ""
echo "  Then open:  http://localhost:3000"
echo "══════════════════════════════════════════"
echo ""

# Offer to start immediately
read -p "  Start the dashboard now? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Starting dashboard on http://localhost:3000 …"
    exec "$VENV_DIR/bin/python" "${DASHBOARD_DIR}/dashboard.py"
fi
