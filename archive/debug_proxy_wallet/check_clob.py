#!/usr/bin/env python3
import os, requests, json
from dotenv import load_dotenv
load_dotenv("/home/polybot/app/.env")
from py_clob_client.client import ClobClient

pk      = os.environ["PRIVATE_KEY"]
FUNDER  = os.environ.get("FUNDER_ADDRESS", "")
TOKEN   = "22262377482663885934165328376380902874834362930691871494750576625828775371280"
PRICE   = 0.104

import math
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY

shares = math.ceil(2.0 / PRICE)
print(f"shares={shares}, price={PRICE}, funder={FUNDER}")

# Test sig_type=0 (EOA)
print("\n=== sig_type=0 (EOA) ===")
c0 = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137, signature_type=0)
creds0 = c0.create_or_derive_api_creds()
c0.set_api_creds(creds0)
try:
    order = c0.create_order(OrderArgs(token_id=TOKEN, price=PRICE, size=shares, side=BUY))
    print("Order fields:", vars(order) if hasattr(order,'__dict__') else dir(order))
    resp = c0.post_order(order)
    print("SUCCESS:", resp)
except Exception as e:
    print(f"Error: {e}")

# Test sig_type=1 with correct funder
print(f"\n=== sig_type=1 funder={FUNDER[:12]}... ===")
c1 = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137, signature_type=1, funder=FUNDER)
creds1 = c1.create_or_derive_api_creds()
c1.set_api_creds(creds1)
try:
    order1 = c1.create_order(OrderArgs(token_id=TOKEN, price=PRICE, size=shares, side=BUY))
    d = vars(order1) if hasattr(order1,'__dict__') else {}
    print(f"maker={d.get('maker','?')}, signatureType={d.get('signatureType','?')}, signer={d.get('signer','?')}")
    resp1 = c1.post_order(order1)
    print("SUCCESS:", resp1)
except Exception as e:
    print(f"Error: {e}")

# Also check the balance allowance with type=0 to see if EOA has any funds
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
print("\n=== Balance check ===")
for st in [0, 1]:
    try:
        c_check = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137, signature_type=st, funder=FUNDER if st==1 else None)
        cr = c_check.create_or_derive_api_creds()
        c_check.set_api_creds(cr)
        bal = c_check.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=st))
        usdc = float(bal.get("balance", 0)) / 1_000_000
        print(f"  sig_type={st}: balance=${usdc:.4f}")
    except Exception as e:
        print(f"  sig_type={st}: {e}")
