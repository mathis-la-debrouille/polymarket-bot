"""
Manifold Markets probability signal.

Fetches the current probability from Manifold for a given question string,
using fuzzy title matching. Results are cached for 10 minutes.
"""

import difflib
import time
from typing import Optional

import requests

_cache: dict = {}   # {question_key: (timestamp, probability)}
CACHE_TTL = 600     # 10 minutes


def fetch_manifold_price(question: str) -> Optional[float]:
    """
    Search Manifold Markets for a binary market matching `question`.
    Returns the probability (0–1) if a match with ratio > 0.40 is found,
    otherwise None.
    """
    key = question[:60].lower()
    if key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]

    try:
        resp = requests.get(
            "https://api.manifold.markets/v0/search-markets",
            params={"term": question[:80], "limit": 5, "sort": "score"},
            timeout=6,
        )
        resp.raise_for_status()
        results = resp.json()
        for market in results:
            if market.get("outcomeType") != "BINARY":
                continue
            ratio = difflib.SequenceMatcher(
                None,
                question.lower()[:80],
                market.get("question", "").lower()[:80],
            ).ratio()
            if ratio > 0.40:
                prob = market.get("probability")
                if prob is not None:
                    _cache[key] = (time.time(), float(prob))
                    return float(prob)
    except Exception:
        pass

    return None
