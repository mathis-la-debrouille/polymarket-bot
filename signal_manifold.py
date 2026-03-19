"""
Manifold Markets probability signal.

Fetches the current probability from Manifold for a given question string,
using fuzzy title matching. Results are cached for 10 minutes.
"""

import difflib
import re
import time
from typing import Optional

import requests

_cache: dict = {}        # {question_key: (timestamp, probability)}
_cache_ratio: dict = {}  # {question_key: ratio} for matched questions
CACHE_TTL = 600          # 10 minutes


def _extract_entities(text: str):
    """Extract words that start with uppercase and have length > 4."""
    words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
    return {w for w in words if len(w) > 4}


def fetch_manifold_price(question: str) -> Optional[float]:
    """
    Search Manifold Markets for a binary market matching `question`.
    Returns the probability (0–1) if a match with ratio >= 0.55 is found
    AND at least 1 common entity word exists, otherwise None.
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
        poly_entities = _extract_entities(question)
        for market in results:
            if market.get("outcomeType") != "BINARY":
                continue
            manifold_q = market.get("question", "")
            ratio = difflib.SequenceMatcher(
                None,
                question.lower()[:80],
                manifold_q.lower()[:80],
            ).ratio()
            if ratio < 0.55:
                continue
            # Entity check: require at least 1 common uppercase word (len > 4)
            manifold_entities = _extract_entities(manifold_q)
            common = poly_entities & manifold_entities
            if not common:
                continue
            prob = market.get("probability")
            if prob is not None:
                _cache[key] = (time.time(), float(prob))
                _cache_ratio[key] = ratio
                return float(prob)
    except Exception:
        pass

    return None
