#!/usr/bin/env python3
"""Find proxy wallet address via on-chain factory + raw API."""
import os, json, requests
from dotenv import load_dotenv
load_dotenv()
from eth_account import Account
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams

pk  = os.environ["PRIVATE_KEY"]
eoa = Account.from_key(pk).address
print("EOA:", eoa)

# 1. Try web3 factory call
try:
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
    # Polymarket proxy wallet factory on Polygon
    FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
    ABI = [
        {"inputs":[{"name":"_funder","type":"address"}],
         "name":"computeProxyWalletAddress",
         "outputs":[{"name":"","type":"address"}],
         "stateMutability":"view","type":"function"},
        {"inputs":[{"name":"_funder","type":"address"}],
         "name":"getProxyAddress",
         "outputs":[{"name":"","type":"address"}],
         "stateMutability":"view","type":"function"},
    ]
    factory = w3.eth.contract(address=Web3.to_checksum_address(FACTORY), abi=ABI)
    checksum_eoa = Web3.to_checksum_address(eoa)
    for fn_name in ["computeProxyWalletAddress", "getProxyAddress"]:
        try:
            fn = getattr(factory.functions, fn_name)
            proxy = fn(checksum_eoa).call()
            print(f"Proxy wallet ({fn_name}):", proxy)
        except Exception as e:
            print(f"  {fn_name} failed:", e)
except ImportError:
    print("web3 not installed, trying pip install...")
    import subprocess
    subprocess.run(["/home/polybot/venv/bin/pip", "install", "web3"], capture_output=True)
    print("web3 installed, please re-run")
except Exception as e:
    print("web3 error:", e)

# 2. Raw balance-allowance to see what address the server sees
c = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137, signature_type=1)
creds = c.create_or_derive_api_creds()
c.set_api_creds(creds)

from py_clob_client.headers.headers import create_level_2_headers
from py_clob_client.signing.hmac import build_hmac_signature
import time

ts   = str(int(time.time()))
sig  = build_hmac_signature(creds.api_secret, ts, "GET", "/balance-allowance")
headers = {
    "POLY_ADDRESS":    eoa,
    "POLY_SIGNATURE":  sig,
    "POLY_TIMESTAMP":  ts,
    "POLY_API_KEY":    creds.api_key,
    "POLY_PASSPHRASE": creds.api_passphrase,
    "Content-Type":    "application/json",
}
resp = requests.get(
    "https://clob.polymarket.com/balance-allowance",
    headers=headers,
    params={"asset_type": "USDC", "signature_type": 1},
    timeout=10,
)
print("\nRaw balance-allowance (sig_type=1):", resp.status_code, resp.text[:500])

# 3. Try GET /user or /user/{address}
for url in [
    f"https://clob.polymarket.com/user",
    f"https://clob.polymarket.com/user/{eoa}",
    f"https://clob.polymarket.com/auth/user",
]:
    try:
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code < 404:
            print(f"\n{url}:", r.status_code, r.text[:300])
    except Exception:
        pass

# 4. Check py_order_utils for proxy wallet helper
try:
    from py_order_utils.config import get_contract_config
    cfg = get_contract_config(137)
    print("\npy_order_utils contract config:", cfg)
except Exception as e:
    print("py_order_utils config:", e)

try:
    from py_order_utils.builders import OrderBuilder
    import inspect
    src = inspect.getsource(OrderBuilder.__init__)
    print("\nOrderBuilder.__init__ source:\n", src[:800])
except Exception as e:
    print("OrderBuilder source error:", e)
