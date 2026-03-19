#!/usr/bin/env python3
"""Find the REAL Polymarket proxy wallet through the CLOB API."""
import os, json, requests
from dotenv import load_dotenv
load_dotenv("/home/polybot/app/.env")
from eth_account import Account
import time

pk  = os.environ["PRIVATE_KEY"]
EOA = Account.from_key(pk).address
print(f"EOA: {EOA}")

# Try to hit the Polymarket REST API directly for the proxy wallet
# The proxy wallet is deterministically tied to the EOA on Polymarket's system
headers_base = {"Content-Type": "application/json"}

# 1. Check Polymarket's user API (not CLOB)
urls = [
    f"https://strapi-matic.poly.market/users?address={EOA.lower()}",
    f"https://strapi-matic.poly.market/users?walletAddress={EOA.lower()}",
    f"https://gamma-api.polymarket.com/users?address={EOA.lower()}",
]
for url in urls:
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200 and len(r.text) > 10:
            print(f"\n{url}:")
            try:
                d = r.json()
                if isinstance(d, list) and d:
                    for item in d[:2]:
                        print(f"  {json.dumps({k:v for k,v in item.items() if 'proxy' in k.lower() or 'wallet' in k.lower() or 'address' in k.lower()})}")
                else:
                    print(f"  {str(d)[:200]}")
            except:
                print(f"  {r.text[:200]}")
    except Exception as e:
        print(f"  {url}: {e}")

# 2. Use py-clob-client to see if there's a way to derive the proxy wallet
from py_clob_client.client import ClobClient
from py_clob_client.signing.eip712 import sign_clob_auth
import inspect

c = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137, signature_type=1)

# Check all methods for proxy/funder/wallet
proxy_methods = [m for m in dir(c) if any(w in m.lower() for w in ['proxy','wallet','funder','address'])]
print(f"\nClobClient proxy-related methods: {proxy_methods}")

# 3. The Polymarket PolyProxy factory should have a deployProxyWallet event
# Try to find it via a contract event scan
# The factory IS at 0xaB45c5A4B0c941a2F231C04C3f49182e1A254052
# The event signature for ProxyWalletCreated(address indexed owner, address proxyWallet)
from web3 import Web3
from eth_utils import keccak

w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com", request_kwargs={"timeout":15}))
FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"

# Get factory creation block to scan the right range
try:
    factory_code = w3.eth.get_code(Web3.to_checksum_address(FACTORY))
    print(f"\nFactory code size: {len(factory_code)} bytes")
except Exception as e:
    print(f"Factory check: {e}")

# Compute all possible event topic signatures
for event_sig in [
    "ProxyWalletCreated(address,address)",
    "ProxyCreated(address,address)",
    "WalletDeployed(address,address)",
    "NewProxy(address,address)",
]:
    topic = "0x" + keccak(text=event_sig).hex()
    eoa_topic = "0x" + "0"*24 + EOA[2:].lower()
    try:
        latest = w3.eth.block_number
        logs = w3.eth.get_logs({
            "fromBlock": max(0, latest - 5000),
            "toBlock": "latest",
            "address": Web3.to_checksum_address(FACTORY),
            "topics": [topic],
        })
        # search all logs
        for log in logs:
            topics_str = [t.hex() for t in log['topics']]
            if EOA.lower()[2:] in " ".join(topics_str).lower():
                print(f"\nFOUND match in {event_sig}: {topics_str}")
                print(f"  data: {log['data'].hex()}")
        print(f"  {event_sig}: {len(logs)} events (none matched EOA)")
    except Exception as e:
        print(f"  {event_sig}: {str(e)[:80]}")

# 4. Try raw CLOB API with different approach — use the L2 auth to ask about proxy
creds = c.create_or_derive_api_creds()
c.set_api_creds(creds)
print(f"\nAPI key: {creds.api_key[:20]}...")

from py_clob_client.headers.headers import create_level_2_headers
from py_clob_client.clob_types import RequestArgs

for path in ["/auth/proxy-wallet", "/proxy-wallet", "/user/proxy-wallet",
             f"/proxy-wallet/{EOA}", "/wallet"]:
    try:
        args = RequestArgs(method="GET", request_path=path)
        hdrs = create_level_2_headers(c.signer, creds, args)
        r = requests.get(f"https://clob.polymarket.com{path}", headers=hdrs, timeout=6)
        print(f"  {path}: {r.status_code} {r.text[:200]}")
    except Exception as e:
        pass
