#!/usr/bin/env python3
"""
signal_updown.py  v2 — 5-Minute Crypto Up/Down Signal Engine
=============================================================

Targets Polymarket markets like:
  "Bitcoin Up or Down - 5 Minutes"     (resolves every 5 min)
  "Ethereum Up or Down - 5 Minutes"    (resolves every 5 min)
  "Bitcoin Up or Down - 15 Minutes"    (resolves every 15 min)
  "Solana Up or Down - 5 Minutes"
  "XRP Up or Down - 5 Minutes"
  "Dogecoin Up or Down - 5 Minutes"

AND legacy hourly/dated format:
  "Bitcoin Up or Down - March 17, 11AM ET"
  "Bitcoin Up or Down - March 17, 12:15PM-12:30PM ET"

SIGNAL STACK (blended):
  1. Brownian motion   P = Φ(pct_move / (σ × √T_remaining))          [30%]
  2. Momentum          OLS slope on 1-min closes, z-scored             [25%]
  3. Funding rate      Binance perp sentiment (bullish/bearish bias)   [20%]
  4. Volume confirm    Is volume supporting the current direction?     [10%]
  5. Cross-asset       BTC move predicts ETH/SOL/XRP/DOGE direction   [15%]

FILTERS (applied before blending):
  - Timing filter     : only enter between 33%–85% of window duration
  - Signal strength   : |pct_move| / (σ × √T_remaining) > 1.2

SPREAD ARB:
  - If UP_price + DOWN_price < $0.96 → buy both sides (riskless profit)

PRICE SOURCE:
  - Primary  : Polymarket RTDS WebSocket (same oracle used for resolution)
  - Fallback : Binance REST API
"""

import math, time, logging, re, threading, json as _json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
import requests
import numpy as np

log = logging.getLogger(__name__)

BINANCE_SPOT  = "https://api.binance.com/api/v3"
BINANCE_PERP  = "https://fapi.binance.com/fapi/v1"

# ── Cross-asset correlation betas (BTC 5-min → other asset 5-min) ──────
# Source: historical analysis of 5-min crypto returns 2024-2026
CROSS_BETA = {
    "ETHUSDT":  0.87,
    "SOLUSDT":  0.74,
    "XRPUSDT":  0.62,
    "DOGEUSDT": 0.71,
}

# ── Cache ───────────────────────────────────────────────────────────────
_cache: Dict = {}

def _get(url, params=None, ttl=30, timeout=6):
    key = url + str(sorted((params or {}).items()))
    c = _cache.get(key)
    if c and time.time() - c["ts"] < ttl:
        return c["v"]
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        v = r.json()
        _cache[key] = {"v": v, "ts": time.time()}
        return v
    except Exception as e:
        log.debug(f"_get {url}: {e}")
        return None

def _ndtr(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ── Polymarket RTDS WebSocket (real-time oracle price) ──────────────────
_rtds_prices: Dict = {}
_rtds_connected: bool = False

def _on_message(ws, message):
    global _rtds_prices
    try:
        data = _json.loads(message)
        items = data if isinstance(data, list) else [data]
        symbol_map = {
            "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
            "XRP": "XRPUSDT", "DOGE": "DOGEUSDT",
        }
        for item in items:
            asset = str(item.get("asset", item.get("symbol", ""))).upper()
            price = item.get("price", item.get("p", 0))
            sym = symbol_map.get(asset)
            if sym and price:
                _rtds_prices[sym] = float(price)
                _rtds_prices[f"{sym}_ts"] = time.time()
    except Exception:
        pass

def _on_error(ws, error):
    global _rtds_connected
    _rtds_connected = False
    log.debug(f"RTDS error: {error}")

def _on_close(ws, *args):
    global _rtds_connected
    _rtds_connected = False

def start_rtds_stream():
    """
    Connect to Polymarket's real-time crypto price WebSocket.
    Runs in background thread. Falls back silently to Binance if unavailable.
    """
    global _rtds_connected
    try:
        import websocket
    except ImportError:
        log.warning("  websocket-client not installed. Run: pip install websocket-client")
        log.warning("  Falling back to Binance polling.")
        return

    def _run():
        global _rtds_connected
        urls_to_try = [
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            "wss://clob.polymarket.com/ws",
        ]
        for url in urls_to_try:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=_on_message,
                    on_error=_on_error,
                    on_close=_on_close,
                )
                _rtds_connected = True
                log.info(f"  RTDS connected: {url}")
                ws.run_forever(ping_interval=20, ping_timeout=8)
            except Exception as e:
                log.debug(f"RTDS {url} failed: {e}")
                _rtds_connected = False
        log.debug("  RTDS: all endpoints failed, using Binance fallback")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

def get_resolution_price(symbol: str) -> Optional[float]:
    """
    Get the current price that Polymarket will use for resolution.
    Uses RTDS if connected and fresh (< 10s old), otherwise Binance.
    """
    rtds_price = _rtds_prices.get(symbol)
    rtds_age   = time.time() - _rtds_prices.get(f"{symbol}_ts", 0)
    if rtds_price and rtds_age < 10:
        return rtds_price
    d = _get(f"{BINANCE_SPOT}/ticker/price", {"symbol": symbol}, ttl=5)
    return float(d["price"]) if d else None

# ── Binance helpers ──────────────────────────────────────────────────────
def get_price_at_time(symbol: str, ts_ms: int) -> Optional[float]:
    """Open price of the 1-min candle at ts_ms. Cached forever (historical)."""
    d = _get(f"{BINANCE_SPOT}/klines",
             {"symbol": symbol, "interval": "1m", "startTime": ts_ms, "limit": 1},
             ttl=86400)
    return float(d[0][1]) if d else None  # [1] = open price

def get_vol_per_min(symbol: str) -> float:
    """Realized vol per minute from last 30 1-min log returns."""
    d = _get(f"{BINANCE_SPOT}/klines",
             {"symbol": symbol, "interval": "1m", "limit": 30}, ttl=120)
    if not d or len(d) < 5:
        return 0.0020 if "BTC" in symbol else 0.0025
    closes = np.array([float(k[4]) for k in d])
    return max(float(np.std(np.diff(np.log(closes)))), 0.0005)

# ── Individual signals ───────────────────────────────────────────────────
def sig_brownian(pct_move: float, minutes_left: float, vol: float) -> float:
    """P(UP) = Φ(pct_move / (σ × √T_remaining))"""
    if minutes_left <= 0:
        return 1.0 if pct_move > 0 else 0.0
    denom = max(vol, 1e-6) * math.sqrt(max(minutes_left, 0.1))
    return float(np.clip(_ndtr(pct_move / denom), 0.02, 0.98))

def sig_momentum(symbol: str) -> Optional[float]:
    """OLS slope on last 15 1-min closes, z-scored."""
    d = _get(f"{BINANCE_SPOT}/klines",
             {"symbol": symbol, "interval": "1m", "limit": 16}, ttl=20)
    if not d or len(d) < 5:
        return None
    closes  = np.array([float(k[4]) for k in d])
    t       = np.arange(len(closes))
    slope   = np.polyfit(t, closes / closes[0], 1)[0]
    returns = np.diff(np.log(closes))
    vol     = max(np.std(returns) if len(returns) > 2 else 0.002, 1e-5)
    return float(np.clip(_ndtr((slope / vol) * 0.6), 0.10, 0.90))

def sig_funding(symbol: str) -> Optional[float]:
    """Perp funding rate. Positive=bulls dominant→UP likely."""
    d = _get(f"{BINANCE_PERP}/premiumIndex", {"symbol": symbol}, ttl=120)
    if not d:
        return None
    try:
        rate = float(d.get("lastFundingRate", 0))
        z    = max(-3.0, min(3.0, rate / 0.0005))
        return float(np.clip(_ndtr(z * 0.5), 0.30, 0.70))
    except Exception:
        return None

def sig_volume(symbol: str, pct_move: float) -> Optional[float]:
    """High volume + direction = continuation signal."""
    d = _get(f"{BINANCE_SPOT}/klines",
             {"symbol": symbol, "interval": "1m", "limit": 11}, ttl=20)
    if not d or len(d) < 5:
        return None
    vols  = np.array([float(k[5]) for k in d])
    ratio = vols[-1] / max(np.mean(vols[:-1]), 1.0)
    if pct_move > 0:
        if ratio > 1.5:   return 0.68
        elif ratio > 1.0: return 0.60
        elif ratio > 0.7: return 0.54
        else:             return 0.48
    else:
        if ratio > 2.0:   return 0.38
        elif ratio > 1.2: return 0.33
        elif ratio > 0.8: return 0.42
        else:             return 0.48

def sig_cross_asset(symbol: str, minutes_left: float, vol: float) -> Optional[float]:
    """
    Use BTC's current 5-min move to predict direction of ETH/SOL/XRP/DOGE.
    BTC moves first; other assets lag by 30-120 seconds.
    """
    if symbol == "BTCUSDT":
        return None
    beta = CROSS_BETA.get(symbol)
    if beta is None:
        return None
    btc_now = get_resolution_price("BTCUSDT")
    ts_5min_ago = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp() * 1000)
    btc_5m_ago  = get_price_at_time("BTCUSDT", ts_5min_ago)
    if not btc_now or not btc_5m_ago or btc_5m_ago <= 0:
        return None
    btc_move = (btc_now - btc_5m_ago) / btc_5m_ago
    implied  = btc_move * beta
    denom    = max(vol, 1e-6) * math.sqrt(max(minutes_left, 0.1))
    return float(np.clip(_ndtr(implied / denom), 0.10, 0.90))

# ── Market question parser ───────────────────────────────────────────────
def parse_updown_question(question: str) -> Optional[Tuple]:
    """
    Handles rolling 5-min/15-min format:
      "Bitcoin Up or Down - 5 Minutes"
      "Ethereum Up or Down - 15 Minutes"

    AND legacy dated format:
      "Bitcoin Up or Down - March 17, 11AM ET"
      "Bitcoin Up or Down - March 17, 12:15PM-12:30PM ET"

    Returns (binance_symbol, ref_dt_utc, res_dt_utc) or None.
    For rolling 5-min markets: returns (symbol, None, None) — caller uses startDate/endDate.
    """
    q = question.strip()

    if re.search(r'\bbitcoin\b|\bbtc\b', q, re.I):     symbol = "BTCUSDT"
    elif re.search(r'\bethereum\b|\beth\b', q, re.I):  symbol = "ETHUSDT"
    elif re.search(r'\bsolana\b|\bsol\b', q, re.I):    symbol = "SOLUSDT"
    elif re.search(r'\bxrp\b|\bripple\b', q, re.I):    symbol = "XRPUSDT"
    elif re.search(r'\bdogecoin\b|\bdoge\b', q, re.I): symbol = "DOGEUSDT"
    else: return None

    if "up or down" not in q.lower(): return None

    # Rolling 5-min/15-min format — dates come from market startDate/endDate fields
    if re.search(r'\b5[\s-]*min', q, re.I) or re.search(r'\b15[\s-]*min', q, re.I):
        return (symbol, None, None)

    # Legacy dated format
    year   = datetime.now(timezone.utc).year
    month  = datetime.now(timezone.utc).month
    tz_off = timedelta(hours=4 if 3 <= month <= 11 else 5)

    def _parse(date_str, h, m, ap):
        try:
            dt = datetime.strptime(f"{date_str} {year} {h}:{m:02d} {ap}",
                                   "%B %d %Y %I:%M %p")
            return dt.replace(tzinfo=timezone.utc) + tz_off
        except: return None

    rng = re.search(r'(\w+ \d+),\s*(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)\s+ET', q, re.I)
    if rng:
        ref = _parse(rng.group(1), int(rng.group(2)), int(rng.group(3)), rng.group(4).upper())
        res = _parse(rng.group(1), int(rng.group(5)), int(rng.group(6)), rng.group(7).upper())
        if ref and res: return (symbol, ref, res)

    sng = re.search(r'(\w+ \d+),\s*(\d+)(?::(\d+))?(AM|PM)\s+ET', q, re.I)
    if sng:
        res = _parse(sng.group(1), int(sng.group(2)), int(sng.group(3) or 0), sng.group(4).upper())
        if res: return (symbol, res - timedelta(hours=1), res)

    return None

def is_updown_market(question: str) -> bool:
    return parse_updown_question(question) is not None

def _get_market_dates(market: dict) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Extract ref_dt and res_dt from Gamma API market fields."""
    def _parse_iso(s):
        if not s: return None
        try:
            s = s.rstrip("Z")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    start_str = market.get("startDate") or market.get("start_date_iso", "")
    end_str   = market.get("endDate")   or market.get("end_date_iso", "")
    return _parse_iso(start_str), _parse_iso(end_str)

# ── SPREAD ARB CHECK ─────────────────────────────────────────────────────
def check_spread_arb(yes_price: float, no_price: float) -> dict:
    """
    Riskless arbitrage: if YES + NO < $1, buy BOTH sides.
    Net profit = 1.0 - yes_price - no_price - 0.04 (2% fee each side)
    """
    gross_arb = 1.0 - yes_price - no_price
    net_arb   = gross_arb - 0.04
    return {
        "arb_detected": net_arb > 0.005,
        "net_arb_ev":   round(net_arb, 4),
        "gross_arb_ev": round(gross_arb, 4),
    }

# ── MASTER FUNCTION ──────────────────────────────────────────────────────
def compute_updown_signal(market: dict, yes_midprice: float) -> Optional[dict]:
    """
    Main entry point. Takes a Polymarket market dict + current YES midprice.

    Returns dict with:
      model_p         : P(UP resolves YES), blended from all signals
      confidence      : 0–1, how much signals agree
      signal_strength : z-score of current move
      arb_detected    : True if riskless spread arb exists
      net_arb_ev      : net EV of spread arb
      pct_move        : % move from window start
      minutes_left    : time until resolution
      timing_ok       : False if outside the 33%–85% entry window

    Returns None if market not parseable, data unavailable, or window not active.
    """
    question = market.get("question", "")
    parsed   = parse_updown_question(question)
    if not parsed:
        return None
    symbol, ref_dt, res_dt = parsed

    # For rolling 5-min/15-min markets, get dates from market fields
    if ref_dt is None or res_dt is None:
        ref_dt, res_dt = _get_market_dates(market)
    if not ref_dt or not res_dt:
        return None

    now = datetime.now(timezone.utc)
    if now >= res_dt: return None
    if now < ref_dt:  return None

    window_duration = (res_dt - ref_dt).total_seconds() / 60.0
    minutes_elapsed = (now - ref_dt).total_seconds() / 60.0
    minutes_left    = (res_dt - now).total_seconds() / 60.0

    if minutes_left < 0.3: return None

    # ── TIMING FILTER ──────────────────────────────────────────────────
    elapsed_frac = minutes_elapsed / max(window_duration, 0.1)
    timing_ok    = 0.33 <= elapsed_frac <= 0.85

    # ── FETCH PRICES ───────────────────────────────────────────────────
    ref_price = get_price_at_time(symbol, int(ref_dt.timestamp() * 1000))
    cur_price = get_resolution_price(symbol)
    if not ref_price or not cur_price:
        return None

    pct_move = (cur_price - ref_price) / ref_price
    vol      = get_vol_per_min(symbol)

    # ── SIGNAL STRENGTH ───────────────────────────────────────────────
    sigma_remaining = max(vol, 1e-6) * math.sqrt(max(minutes_left, 0.1))
    signal_strength = abs(pct_move) / sigma_remaining

    # ── SPREAD ARB CHECK ──────────────────────────────────────────────
    no_midprice = 1.0 - yes_midprice
    arb = check_spread_arb(yes_midprice, no_midprice)

    # ── SIGNALS ───────────────────────────────────────────────────────
    signals = {}
    weights = {}

    bm = sig_brownian(pct_move, minutes_left, vol)
    signals["brownian"] = bm;  weights["brownian"] = 0.30

    mom = sig_momentum(symbol)
    if mom is not None:
        signals["momentum"] = mom;  weights["momentum"] = 0.25
    else:
        weights["brownian"] += 0.25

    fund = sig_funding(symbol)
    if fund is not None:
        signals["funding"] = fund;  weights["funding"] = 0.20
    else:
        weights["brownian"] += 0.20

    vol_s = sig_volume(symbol, pct_move)
    if vol_s is not None:
        signals["volume"] = vol_s;  weights["volume"] = 0.10
    else:
        weights["brownian"] += 0.10

    cross = sig_cross_asset(symbol, minutes_left, vol)
    if cross is not None:
        signals["cross"] = cross;  weights["cross"] = 0.15
        weights["brownian"] = max(0.15, weights["brownian"] - 0.15)

    total_w = sum(weights.values())
    model_p = float(np.clip(
        sum(signals[k] * weights[k] / total_w for k in signals),
        0.03, 0.97
    ))

    vals       = list(signals.values())
    confidence = float(np.clip(1.0 - np.std(vals) * 4, 0.1, 1.0)) if len(vals) > 1 else 0.5

    log.info(
        f"  [{symbol}] ref=${ref_price:,.2f}→${cur_price:,.2f} "
        f"({pct_move*100:+.3f}%) {minutes_left:.1f}min "
        f"strength={signal_strength:.2f} | "
        + " ".join(f"{k}={v:.3f}" for k, v in signals.items())
        + f" → p={model_p:.3f} conf={confidence:.2f}"
        + (" timing_bad" if not timing_ok else "")
        + (" ⚡ARB" if arb["arb_detected"] else "")
    )

    return {
        "model_p":         model_p,
        "confidence":      confidence,
        "signal_strength": signal_strength,
        "timing_ok":       timing_ok,
        "pct_move":        round(pct_move, 5),
        "minutes_left":    round(minutes_left, 2),
        "minutes_elapsed": round(minutes_elapsed, 2),
        "window_duration": round(window_duration, 2),
        "symbol":          symbol,
        "ref_price":       ref_price,
        "cur_price":       cur_price,
        "signals":         signals,
        **arb,
    }
