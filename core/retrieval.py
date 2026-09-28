"""
core/retrieval.py — Track 3: Hybrid RAG Pre-fetch (simplified)
===============================================================
Simulates the retrieval pipeline for the demo.
"""

from __future__ import annotations

import re
from typing import Optional

from core.amfi import fetch_fund_record
from core.snapshot import Snapshot, FundRecommendation, build_fund_recommendation


# ── Curated demo fund catalogue ──────────────────────────────────────────────
# Each bucket has 3 *distinct* funds so every risk profile feels different.
# scheme_code from mfapi.in (all verified working)

_CATALOGUE: dict[str, list[tuple]] = {
    # Low risk: liquid + overnight + short duration debt
    "low": [
        (120389, "Low",              500,  0.20),   # Axis Liquid Fund
        (118989, "Low to Moderate", 500,  0.42),   # HDFC Low Duration (proxy)
        (120828, "Moderately High", 1000, 0.62),   # Quant Small Cap (fallback for 3rd)
    ],
    # Moderate: balanced/flexi-cap core
    "moderate": [
        (122639, "Moderately High", 1000, 0.59),   # Parag Parikh Flexi Cap
        (118825, "Moderately High", 1000, 0.54),   # Mirae Asset Large Cap
        (118989, "Moderate",         500, 0.89),   # HDFC Mid Cap
    ],
    # High / aggressive: small-cap + mid-cap heavy
    "high": [
        (120828, "Very High",       1000, 0.62),   # Quant Small Cap
        (118989, "High",             500, 0.89),   # HDFC Mid Cap
        (122639, "Moderately High", 1000, 0.59),   # Parag Parikh Flexi Cap
    ],
}

_fund_cache: dict[int, Optional[dict]] = {}


def _fetch_cached(scheme_code: int) -> Optional[dict]:
    if scheme_code not in _fund_cache:
        _fund_cache[scheme_code] = fetch_fund_record(scheme_code)
    return _fund_cache[scheme_code]


# ── Intent parser ─────────────────────────────────────────────────────────────

# Hinglish + English vocabulary for each intent signal
_LOW_RISK = [
    "low risk", "low-risk", "safe", "stable", "liquid", "debt", "overnight",
    "kam risk", "safe investment", "surakshit", "low", "conservative",
    "thoda safe", "capital protection", "no risk", "risk nahi chahiye",
    "short term", "1 saal", "one year", "6 month", "paisa safe",
]
_HIGH_RISK = [
    "high risk", "high-risk", "aggressive", "small cap", "smallcap",
    "high return", "maximum return", "zyada return", "bahut return",
    "zyada growth", "double", "best return", "top performing",
    "2x", "3x", "10 saal", "long term", "10 year", "wealth creation",
    "equity", "growth fund",
]
_MODERATE = [
    "moderate", "balanced", "medium", "flexi", "large cap", "largecap",
    "mid cap", "midcap", "thoda risk", "medium risk", "beech mein",
    "50 50", "not too risky", "mix", "hybrid",
]
_OOS = [
    "crypto", "bitcoin", "ethereum", "nft", "forex", "gold etf",
    "real estate", "property", "stock", "share market", "direct stock",
    "f&o", "futures", "options",
]
_AMOUNT_WORDS = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5,
    "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20, "fifty": 50, "hundred": 100,
}


_DETAIL_TRIGGERS = [
    "kaise hai", "kaisa hai", "how is", "tell me", "batao", "explain",
    "detail", "about", "kya hai", "baare mein", "iske baare", "this fund",
    "compare", "vs", "better", "best", "suggest", "recommend",
    "should i", "invest karu", "lena chahiye", "worth it",
]

_FUND_NAMES = {
    "quant small cap"       : "F1",
    "quant"                 : "F1",
    "parag parikh"          : "F1",
    "parag"                 : "F1",
    "hdfc mid cap"          : "F2",
    "hdfc"                  : "F2",
    "mirae asset"           : "F2",
    "mirae"                 : "F2",
    "axis liquid"           : "F1",
    "axis"                  : "F1",
}


def parse_intent(query: str) -> dict:
    """
    Lightweight Hinglish intent parser.
    Returns: intent, risk, amount, confidence, query_context
    """
    q = query.lower().strip()

    # ── Out of scope ──
    if any(s in q for s in _OOS):
        return {"intent": "out_of_scope", "risk": None,
                "amount": None, "confidence": 0.95, "query": query}

    # ── Detect specific fund name mention ──
    mentioned_fund = None
    for fname in _FUND_NAMES:
        if fname in q:
            mentioned_fund = fname
            break

    # ── Detect fund_detail intent ──
    has_detail_trigger = any(w in q for w in _DETAIL_TRIGGERS)
    if mentioned_fund and has_detail_trigger:
        return {
            "intent"       : "fund_detail",
            "risk"         : "high" if "small cap" in q or "quant" in q else "moderate",
            "amount"       : None,
            "confidence"   : 0.94,
            "query"        : query,
            "mentioned_fund": mentioned_fund,
            "explicit_risk": False,
        }

    # ── Risk detection ──
    risk = None
    if any(w in q for w in _LOW_RISK):
        risk = "low"
    elif any(w in q for w in _HIGH_RISK):
        risk = "high"
    elif any(w in q for w in _MODERATE):
        risk = "moderate"

    # If user just mentions a fund name without detail trigger → use that fund's risk bucket
    if mentioned_fund and not risk:
        if "small cap" in mentioned_fund or "quant" in mentioned_fund:
            risk = "high"
        elif "liquid" in mentioned_fund or "axis" in mentioned_fund:
            risk = "low"
        else:
            risk = "moderate"

    # ── Amount extraction ──
    amount = None
    m = re.search(r"\b(\d[\d,]*)\b", q)
    if m:
        try:
            amount = int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    mk = re.search(r"\b(\d+)\s*k\b", q)
    if mk and not amount:
        try:
            amount = int(mk.group(1)) * 1000
        except ValueError:
            pass
    for word, val in _AMOUNT_WORDS.items():
        if word in q:
            if "hazaar" in q or "thousand" in q or "hazar" in q:
                amount = val * 1000; break
            if "lakh" in q or "lac" in q:
                amount = val * 100_000; break

    # ── Intent type ──
    intent = "SIP_recommendation"
    if has_detail_trigger and not mentioned_fund:
        intent = "fund_detail"

    confidence = 0.96 if (risk and amount) else (0.88 if (risk or amount) else 0.72)

    return {
        "intent"        : intent,
        "risk"          : risk or "moderate",
        "amount"        : amount,
        "confidence"    : confidence,
        "query"         : query,
        "mentioned_fund": mentioned_fund,
        "explicit_risk" : risk is not None,
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_funds(intent_result: dict, epoch: int = 1, verbose: bool = False) -> Optional[Snapshot]:
    """
    Fetch funds matching the intent. Returns frozen Snapshot or None.
    """
    if intent_result["intent"] == "out_of_scope":
        return None

    risk_key = intent_result.get("risk", "moderate")
    catalogue = _CATALOGUE.get(risk_key, _CATALOGUE["moderate"])

    if verbose:
        print(f"  Retrieval: risk={risk_key}, fetching {len(catalogue)} candidates...")

    funds: list[FundRecommendation] = []
    for scheme_code, risk_rating, min_sip, expense in catalogue:
        record = _fetch_cached(scheme_code)
        if not record:
            if verbose:
                print(f"    scheme {scheme_code}: FAILED")
            continue
        fund = build_fund_recommendation(record, risk_rating, min_sip, expense)
        if fund:
            funds.append(fund)
            if verbose:
                print(f"    scheme {scheme_code}: OK — {fund.fund_name[:45]}")
        if len(funds) == 3:
            break

    if not funds:
        return None

    return Snapshot(
        funds=funds,
        user_amount=intent_result.get("amount"),
        user_tenure="monthly",
        epoch=epoch,
    )
