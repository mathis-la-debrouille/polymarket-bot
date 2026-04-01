# Polymarket On-Chain Token Redemption

## Problem

When a Polymarket market resolves, your winning (or losing) tokens remain in your **proxy wallet** as ERC1155 tokens. Until you call `redeemPositions`, the USDC stays locked in the Conditional Tokens Framework (CTF) contract. The Polymarket UI shows a "Claim" button, but automating this requires on-chain interaction.

---

## Architecture

```
EOA (your private key)
  └─► Proxy Factory (0xaB45c5A4B0c941a2F231C04C3f49182e1A254052)
        └─► proxy((uint8,address,uint256,bytes)[])
              └─► Proxy Wallet (EIP-1167 minimal proxy — your FUNDER_ADDRESS)
                    └─► CTF.redeemPositions(USDC, bytes32(0), conditionId, [1,2])
                          └─► USDC lands back in Proxy Wallet
```

### Key insight

The proxy wallet (`FUNDER_ADDRESS`) is an **EIP-1167 minimal proxy** that only implements:
- `onERC1155Received` / `onERC1155BatchReceived` (token callbacks)
- No `execute()`, no `owner()`, no Gnosis Safe interface

**You cannot call the proxy wallet directly.** Instead, the EOA calls the **Proxy Factory's `proxy()` function**, which forwards calls through the user's proxy wallet.

### Why the Relayer doesn't work (without special credentials)

`relayer-v2.polymarket.com/submit` requires HMAC auth with credentials that are separate from the CLOB API credentials. The endpoint returns `401 invalid authorization` with standard CLOB-derived keys. Bypass entirely — use the factory directly.

---

## Contract Addresses (Polygon Mainnet)

| Contract | Address |
|---|---|
| Proxy Factory | `0xaB45c5A4B0c941a2F231C04C3f49182e1A254052` |
| CTF (Conditional Tokens) | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` |
| USDC.e | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` |

---

## How to Find Resolved Conditions with Token Balance

1. Fetch all CLOB trades → extract `asset_id` (token_id) and `market` (condition_id)
2. For each condition, call `CTF.payoutDenominator(conditionId)` → returns `> 0` if resolved
3. Call `CTF.balanceOf(proxyWallet, tokenId)` → check for non-zero balance

```python
# Check if condition is resolved
denom = ctf.functions.payoutDenominator(cid_bytes).call()
is_resolved = denom > 0
```

---

## Redemption Call

```python
from web3 import Web3
from eth_account import Account

w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))
account = Account.from_key(PRIVATE_KEY)
eoa = account.address

CTF     = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
USDC    = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
FACTORY = Web3.to_checksum_address("0xaB45c5A4B0c941a2F231C04C3f49182e1A254052")

CTF_ABI = [{"inputs":[{"name":"collateralToken","type":"address"},
                       {"name":"parentCollectionId","type":"bytes32"},
                       {"name":"conditionId","type":"bytes32"},
                       {"name":"indexSets","type":"uint256[]"}],
            "name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"}]

FACTORY_ABI = [{"inputs":[{"components":[{"name":"op","type":"uint8"},
                                          {"name":"to","type":"address"},
                                          {"name":"value","type":"uint256"},
                                          {"name":"data","type":"bytes"}],
                            "name":"calls","type":"tuple[]"}],
                "name":"proxy","outputs":[],"stateMutability":"nonpayable","type":"function"}]

ctf     = w3.eth.contract(address=CTF,     abi=CTF_ABI)
factory = w3.eth.contract(address=FACTORY, abi=FACTORY_ABI)

# Build inner CTF calldata
cid_bytes = bytes.fromhex(condition_id[2:].zfill(64))
inner = ctf.encode_abi("redeemPositions", args=[USDC, bytes(32), cid_bytes, [1, 2]])
#                               parentCollectionId=bytes32(0) ^^^^^^^^
#                               indexSets=[1, 2] redeems both YES and NO tokens ^^^^

# Wrap in factory.proxy() call
calls = [(1, CTF, 0, bytes.fromhex(inner[2:]))]
#        ^op=1 (CALL), ^to=CTF, ^value=0

# Estimate gas and send
gas_est = factory.functions.proxy(calls).estimate_gas({"from": eoa})
tx = factory.functions.proxy(calls).build_transaction({
    "from":     eoa,
    "nonce":    w3.eth.get_transaction_count(eoa),
    "gas":      gas_est + 50_000,
    "gasPrice": w3.eth.gas_price * 2,  # 2x for fast inclusion
    "chainId":  137,
})
signed = account.sign_transaction(tx)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"Status: {receipt.status}")  # 1 = success
```

### Important notes

- **`indexSets=[1, 2]`** redeems both outcome positions. `1` = YES (outcome 0), `2` = NO (outcome 1). Always pass both — no harm in redeeming losing tokens (returns 0 USDC), but avoids leaving dust.
- **`parentCollectionId=bytes32(0)`** works for all standard and NegRisk Polymarket markets.
- **Gas**: ~100k per condition. Batch up to ~5 conditions per transaction.
- **Gas price**: Use `2x` the current gas price for Polygon to ensure quick inclusion.
- **USDC destination**: USDC lands back in the proxy wallet (FUNDER_ADDRESS), not the EOA.

---

## Batch Redemption (multiple conditions per TX)

```python
# Group multiple conditions in one transaction
calls = []
for cid in condition_ids:
    cid_bytes = bytes.fromhex(cid[2:].zfill(64))
    inner = ctf.encode_abi("redeemPositions", args=[USDC, bytes(32), cid_bytes, [1, 2]])
    calls.append((1, CTF, 0, bytes.fromhex(inner[2:])))

# One TX for all — gas estimate protects against reverts
gas_est = factory.functions.proxy(calls).estimate_gas({"from": eoa})
```

---

## Bot Integration (`_bulk_redeem` function)

The bot runs `_bulk_redeem()` every 20 scans (~10 minutes). It:

1. Fetches all CLOB trade history → maps condition_ids to token_ids
2. For each condition: checks `payoutDenominator` (resolved if `> 0`) and token balance
3. Batches resolved conditions (5 per TX) and redeems via `factory.proxy()`
4. After redemption, syncs the live USDC balance

This runs automatically in live mode only.

---

## Debugging

**"result for condition not received yet"** — Market not resolved yet. `payoutDenominator == 0`. Skip and retry later.

**"Could not decode contract function call"** — ABI output type mismatch. The `proxy()` function returns `tuple[]` which web3 may fail to decode. Ignore this — use `estimate_gas` instead of `call()` to validate, then send.

**Transaction dropped** — Gas price too low for Polygon. Use at least `2x` current gas price.

**401 from relayer** — Don't use the relayer. Use `factory.proxy()` directly from EOA instead.
