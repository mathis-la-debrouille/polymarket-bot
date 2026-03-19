#!/usr/bin/env python3
"""Verify proxy wallet and test order signing."""
import os, math
from dotenv import load_dotenv
load_dotenv("/home/polybot/app/.env")
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType, OrderArgs, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY

pk  = os.environ["PRIVATE_KEY"]
PROXY = "0xd216153c06e857cd7f72665e0af1d7d82172f494"

print("Testing with proxy wallet:", PROXY)

# 1. Check balance with signature_type=1 and funder=proxy
c = ClobClient(
    "https://clob.polymarket.com",
    key=pk,
    chain_id=137,
    signature_type=1,
    funder=PROXY
)
creds = c.create_or_derive_api_creds()
c.set_api_creds(creds)

try:
    ba = c.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=1))
    print("Balance allowance (sig_type=1, with proxy funder):", ba)
except Exception as e:
    print("Balance error:", e)

# 2. Try creating an order (dry run - don't post it)
# Use a known test token that exists
# OKC Thunder NO token from previous error: 52114319501245915516055106046884209969926127482827954674443846427813813222426
TEST_TOKEN = "52114319501245915516055106046884209969926127482827954674443846427813813222426"
TEST_PRICE = 0.116

try:
    shares = math.ceil(2.0 / TEST_PRICE)  # $2 USD / price = shares
    print(f"\nCreating test order: {shares} shares @ {TEST_PRICE}")
    order = c.create_order(
        OrderArgs(
            token_id=TEST_TOKEN,
            price=TEST_PRICE,
            size=shares,
            side=BUY,
        )
    )
    print("Order created successfully!")
    print("  maker:", order.maker)
    print("  signatureType:", order.signatureType)
    print("  size:", order.size)
except Exception as e:
    print("Order creation error:", e)
