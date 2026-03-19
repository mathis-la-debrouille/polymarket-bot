"""
Metaculus community forecast fetcher.
Returns the median community prediction (q2) for a matching question.
"""

import difflib
import time
from typing import Optional

import requests

_cache: dict = {}
CACHE_TTL = 900  # 15 min


def fetch_metaculus_price(question: str) -> Optional[float]:
    key = question[:60].lower()
    if key in _cache and time.time() - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]
    try:
        resp = requests.get(
            "https://www.metaculus.com/api2/questions/",
            params={
                "search": question[:80],
                "status": "open",
                "type": "forecast",
                "order_by": "-activity",
                "limit": 5,
            },
            headers={"Accept": "application/json"},
            timeout=6,
        )
        data = resp.json()
        for q in data.get("results", []):
            label = q.get("title", "")
            ratio = difflib.SequenceMatcher(
                None, question.lower()[:80], label.lower()[:80]
            ).ratio()
            if ratio > 0.50:
                cp = q.get("community_prediction", {})
                p = cp.get("full", {}).get("q2")  # median
                if p is not None:
                    _cache[key] = (time.time(), float(p))
                    return float(p)
    except Exception:
        pass
    _cache[key] = (time.time(), None)
    return None
