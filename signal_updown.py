#!/usr/bin/env python3
"""
signal_updown.py
Signal engine for "Bitcoin/Ethereum Up or Down [time]" Polymarket markets.
"""
import math, time, logging, re
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
import requests
import numpy as np

log = logging.getLogger(__name__)

BINANCE_SPOT = "https://api.binance.com/api/v3"
BINANCE_PERP = "https://fapi.binance.com/fapi/v1"

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
        log.debug(f"signal_updown _get {url}: {e}")
        return None

def _ndtr(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def get_current_price(symbol: str) -> Optional[float]:
    d = _get(f"{BINANCE_SPOT}/ticker/price", {"symbol": symbol}, ttl=15)
    return float(d["price"]) if d else None

def get_price_at_time(symbol: str, ts_ms: int) -> Optional[float]:
    d = _get(f"{BINANCE_SPOT}/klines",
             {"symbol": symbol, "interval": "1m", "startTime": ts_ms, "limit": 1},
             ttl=3600)
    return float(d[0][1]) if d else None

def get_vol_per_min(symbol: str) -> float:
    d = _get(f"{BINANCE_SPOT}/klines",
             {"symbol": symbol, "interval": "1m", "limit": 30}, ttl=300)
    if not d or len(d) < 5:
        return 0.0020 if "BTC" in symbol else 0.0025
    closes = np.array([float(k[4]) for k in d])
    return max(float(np.std(np.diff(np.log(closes)))), 0.0005)

def sig_brownian(pct_move: float, minutes_left: float, vol: float) -> float:
    if minutes_left <= 0:
        return 1.0 if pct_move > 0 else 0.0
    denom = max(vol, 1e-6) * math.sqrt(max(minutes_left, 0.1))
    return float(np.clip(_ndtr(pct_move / denom), 0.02, 0.98))

def sig_momentum(symbol: str) -> Optional[float]:
    d = _get(f"{BINANCE_SPOT}/klines",
             {"symbol": symbol, "interval": "1m", "limit": 16}, ttl=30)
    if not d or len(d) < 5:
        return None
    closes  = np.array([float(k[4]) for k in d])
    t       = np.arange(len(closes))
    slope   = np.polyfit(t, closes / closes[0], 1)[0]
    returns = np.diff(np.log(closes))
    vol     = max(np.std(returns) if len(returns) > 2 else 0.002, 1e-5)
    z       = slope / vol
    return float(np.clip(_ndtr(z * 0.6), 0.10, 0.90))

def sig_funding(symbol: str) -> Optional[float]:
    d = _get(f"{BINANCE_PERP}/premiumIndex", {"symbol": symbol}, ttl=120)
    if not d:
        return None
    try:
        rate = float(d.get("lastFundingRate", 0))
    except Exception:
        return None
    z = max(-3.0, min(3.0, rate / 0.0005))
    return float(np.clip(_ndtr(z * 0.5), 0.30, 0.70))

def sig_volume(symbol: str, pct_move: float) -> Optional[float]:
    d = _get(f"{BINANCE_SPOT}/klines",
             {"symbol": symbol, "interval": "1m", "limit": 11}, ttl=30)
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

def parse_updown_question(question: str) -> Optional[Tuple]:
    q = question.strip()
    if re.search(r'\bbitcoin\b|\bbtc\b', q, re.I):    symbol = "BTCUSDT"
    elif re.search(r'\bethereum\b|\beth\b', q, re.I): symbol = "ETHUSDT"
    elif re.search(r'\bsolana\b|\bsol\b', q, re.I):   symbol = "SOLUSDT"
    else: return None
    if "up or down" not in q.lower(): return None

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

def compute_updown_signal(market: dict, yes_midprice: float) -> Optional[Dict]:
    """
    Main entry point. Returns dict with model_p, confidence, arb info.
    Returns None if market not parseable, Binance unavailable, or already resolved.
    """
    parsed = parse_updown_question(market.get("question", ""))
    if not parsed:
        return None
    symbol, ref_dt, res_dt = parsed
    now = datetime.now(timezone.utc)

    if now >= res_dt:  return None
    if now < ref_dt:   return None
    minutes_left = (res_dt - now).total_seconds() / 60.0
    if minutes_left < 0.5: return None

    ref_price = get_price_at_time(symbol, int(ref_dt.timestamp() * 1000))
    cur_price = get_current_price(symbol)
    if not ref_price or not cur_price:
        return None

    pct_move = (cur_price - ref_price) / ref_price
    vol      = get_vol_per_min(symbol)

    signals = {}
    weights = {}

    bm = sig_brownian(pct_move, minutes_left, vol)
    signals["brownian"] = bm
    weights["brownian"] = 0.30

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
        signals["volume"] = vol_s;  weights["volume"] = 0.25
    else:
        weights["brownian"] += 0.25

    total_w = sum(weights.values())
    model_p = float(np.clip(
        sum(signals[k] * weights[k] / total_w for k in signals),
        0.03, 0.97
    ))

    vals       = list(signals.values())
    confidence = float(np.clip(1.0 - np.std(vals) * 4, 0.1, 1.0)) if len(vals) > 1 else 0.5

    no_price  = 1.0 - yes_midprice
    arb_ev    = 1.0 - yes_midprice - no_price - 0.04
    arb_found = arb_ev > 0.01

    log.info(
        f"  [{symbol}] ref=${ref_price:,.0f}->>${cur_price:,.0f} "
        f"({pct_move*100:+.3f}%) {minutes_left:.1f}min sigma={vol*100:.3f}%/min | "
        f"BM={bm:.3f}"
        + (f" MOM={mom:.3f}" if mom is not None else "")
        + (f" FUND={fund:.3f}" if fund is not None else "")
        + (f" VOL={vol_s:.3f}" if vol_s is not None else "")
        + f" -> model_p={model_p:.3f} conf={confidence:.2f}"
        + (" ARB" if arb_found else "")
    )

    return {
        "model_p":      model_p,
        "confidence":   confidence,
        "arb_detected": arb_found,
        "arb_ev":       round(arb_ev, 4),
        "pct_move":     round(pct_move, 5),
        "minutes_left": round(minutes_left, 1),
        "symbol":       symbol,
        "ref_price":    ref_price,
        "cur_price":    cur_price,
    }
