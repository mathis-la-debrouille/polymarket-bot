#!/usr/bin/env python3
"""
redeem_cron.py — Standalone on-chain token redemption script
============================================================
Run as a cron job every 5-10 minutes.
Scans all CLOB trades, finds resolved conditions with token balances, redeems them.

Usage:
    python3 redeem_cron.py
    # or via cron: */5 * * * * /home/polybot/venv/bin/python /home/polybot/app/redeem_cron.py >> /home/polybot/logs/redeem.log 2>&1
"""
import fcntl
import logging
import os
import sys
import time
from pathlib import Path

# ── Load .env ──────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PRIVATE_KEY    = os.environ.get("PRIVATE_KEY", "")
FUNDER_ADDRESS = os.environ.get("FUNDER_ADDRESS", "")
POLY_API_KEY   = os.environ.get("POLY_API_KEY", "")
POLY_ADDRESS   = os.environ.get("POLY_ADDRESS", "")
POLYGON_RPC    = os.environ.get("POLYGON_RPC", "https://polygon-bor-rpc.publicnode.com")

_REDEEM_CTF    = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
_REDEEM_USDC   = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
_PROXY_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"

_CTF_ABI = [
    {"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],"name":"payoutDenominator","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"account","type":"address"},{"name":"id","type":"uint256"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
]
_FACTORY_ABI = [
    {"inputs":[{"components":[{"name":"op","type":"uint8"},{"name":"to","type":"address"},{"name":"value","type":"uint256"},{"name":"data","type":"bytes"}],"name":"calls","type":"tuple[]"}],"name":"proxy","outputs":[],"stateMutability":"nonpayable","type":"function"}
]


_LOCK_FILE = "/tmp/redeem_cron.lock"


def main():
    # ── Single-instance lock (prevents overlapping cron runs) ─────────────────
    lock_fh = open(_LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info("Another redeem_cron instance is running — skipping")
        return

    if not PRIVATE_KEY:
        log.error("PRIVATE_KEY not set — cannot redeem")
        sys.exit(1)
    if not FUNDER_ADDRESS:
        log.error("FUNDER_ADDRESS not set — cannot redeem")
        sys.exit(1)

    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError:
        log.error("web3 not installed — pip install web3")
        sys.exit(1)

    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        log.error("py-clob-client not installed")
        sys.exit(1)

    # ── Connect ────────────────────────────────────────────────────────────────
    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    if not w3.is_connected():
        log.error(f"Cannot connect to Polygon RPC: {POLYGON_RPC}")
        sys.exit(1)

    account = Account.from_key(PRIVATE_KEY)
    eoa     = account.address

    CTF     = Web3.to_checksum_address(_REDEEM_CTF)
    USDC    = Web3.to_checksum_address(_REDEEM_USDC)
    FACTORY = Web3.to_checksum_address(_PROXY_FACTORY)
    PROXY   = Web3.to_checksum_address(FUNDER_ADDRESS)

    ctf     = w3.eth.contract(address=CTF,     abi=_CTF_ABI)
    factory = w3.eth.contract(address=FACTORY, abi=_FACTORY_ABI)

    # ── CLOB client for trade history ─────────────────────────────────────────
    clob = ClobClient(
        host="https://clob.polymarket.com",
        key=PRIVATE_KEY,
        chain_id=137,
        signature_type=1,
        funder=FUNDER_ADDRESS,
    )
    try:
        creds = clob.create_or_derive_api_creds()
        clob.set_api_creds(creds)
        log.info(f"API creds derived: key={creds.api_key}")
    except Exception as e:
        log.error(f"Failed to derive API creds: {e}")
        sys.exit(1)

    # ── Fetch trade history ────────────────────────────────────────────────────
    log.info("Fetching CLOB trade history…")
    try:
        trades = clob.get_trades()
    except Exception as e:
        log.error(f"Failed to fetch trades: {e}")
        sys.exit(1)

    if not trades:
        log.info("No trades found — nothing to redeem")
        return

    # ── Map condition_id → token_ids ──────────────────────────────────────────
    condition_tokens: dict = {}
    for t in trades:
        tid = t.get("asset_id", "")
        cid = t.get("market", "")
        if tid and cid:
            condition_tokens.setdefault(cid, [])
            if tid not in condition_tokens[cid]:
                condition_tokens[cid].append(tid)

    log.info(f"Found {len(condition_tokens)} unique conditions from {len(trades)} trades")

    # ── Find resolved conditions with balance ─────────────────────────────────
    resolved_cids = []
    checked = 0
    for cid, token_ids in condition_tokens.items():
        try:
            cid_bytes = bytes.fromhex(cid[2:].zfill(64))
            denom = ctf.functions.payoutDenominator(cid_bytes).call()
            if denom == 0:
                continue
            has_balance = any(
                ctf.functions.balanceOf(PROXY, int(tid)).call() > 100_000  # > $0.10
                for tid in token_ids
            )
            if has_balance:
                resolved_cids.append(cid)
        except Exception:
            continue
        checked += 1
        if checked % 20 == 0:
            log.info(f"  Checked {checked}/{len(condition_tokens)} conditions, {len(resolved_cids)} redeemable so far…")

    if not resolved_cids:
        log.info("No resolved conditions with token balance — nothing to redeem")
        return

    log.info(f"Found {len(resolved_cids)} conditions to redeem")

    # ── Batch-redeem in groups of 5 ───────────────────────────────────────────
    gas_price = min(w3.eth.gas_price * 2, Web3.to_wei(500, "gwei"))
    nonce     = w3.eth.get_transaction_count(eoa)
    redeemed  = 0
    batch_size = 5

    for i in range(0, len(resolved_cids), batch_size):
        batch = resolved_cids[i:i + batch_size]
        calls = []
        for cid in batch:
            cid_bytes = bytes.fromhex(cid[2:].zfill(64))
            inner = ctf.encode_abi("redeemPositions", args=[USDC, bytes(32), cid_bytes, [1, 2]])
            calls.append((1, CTF, 0, bytes.fromhex(inner[2:])))

        try:
            gas_est = factory.functions.proxy(calls).estimate_gas({"from": eoa})
            tx = factory.functions.proxy(calls).build_transaction({
                "from":     eoa,
                "nonce":    nonce,
                "gas":      gas_est + 50_000,
                "gasPrice": gas_price,
                "chainId":  137,
            })
            signed   = account.sign_transaction(tx)
            tx_hash  = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt  = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
            if receipt.status == 1:
                redeemed += len(batch)
                nonce    += 1
                log.info(f"  Batch {i//batch_size+1}: redeemed {len(batch)} conditions, tx={tx_hash.hex()[:20]}… gas={receipt.gasUsed}")
            else:
                log.warning(f"  Batch {i//batch_size+1}: TX reverted — {tx_hash.hex()[:20]}")
        except Exception as e:
            log.warning(f"  Batch {i//batch_size+1}: failed — {e}")
        time.sleep(1)

    log.info(f"Done — redeemed {redeemed} condition(s)")


if __name__ == "__main__":
    main()
