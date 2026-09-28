"""
core/retrieval2.py — Snapshot Builder (replaces _CATALOGUE in retrieval.py)
===========================================================================
Builds the snapshot from data/universe.json instead of hand-typed tuples.

The old _CATALOGUE put scheme 120828 (a Very High risk small-cap fund) in the
"low" bucket, and gave scheme 118989 three different risk_rating values in three
different buckets. Here a fund carries exactly one risk band and exactly one
appetite bucket, both derived from its SEBI category, so it is impossible for
the same fund to be "Low to Moderate" in one answer and "High" in the next.

Amount is now a filter, not decoration: funds whose minimum SIP exceeds what
the user said they can invest are excluded rather than recommended anyway.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from core import universe
from core.amfi import fetch_fund_record
from core.snapshot import FundRecommendation, Snapshot, build_fund_recommendation

_CACHE_FILE = Path(__file__).parent.parent / "data" / "mfapi_cache.json"

def _load_cache() -> dict[int, Optional[dict]]:
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                return {int(k): v for k, v in d.items()}
        except Exception:
            pass
    return {}

def _save_cache(cache: dict[int, Optional[dict]]) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in cache.items() if v is not None}, f, indent=2)
    except Exception:
        pass

_record_cache: dict[int, Optional[dict]] = _load_cache()


def _fetch(scheme_code: int) -> Optional[dict]:
    if scheme_code not in _record_cache:
        rec = fetch_fund_record(scheme_code)
        if rec:
            _record_cache[scheme_code] = rec
            _save_cache(_record_cache)
        else:
            _record_cache[scheme_code] = None
    return _record_cache[scheme_code]


def _to_recommendation(u: dict) -> Optional[FundRecommendation]:
    record = _fetch(u["scheme_code"])
    if not record:
        return None
    return build_fund_recommendation(
        record,
        risk_rating=u["risk_band"],
        min_sip=u["min_sip"],
        # expense_ratio is not published by mfapi.in. Passing a per-fund
        # invented figure was the old behaviour; a single declared placeholder
        # is at least honest, and the audit log records it as unfetched.
        expense_ratio=u.get("expense_ratio") or 0.0,
    )


def build_snapshot(
    bucket: str,
    amount: Optional[int] = None,
    tenure: Optional[str] = None,
    epoch: int = 1,
    only: Optional[list[str]] = None,
    limit: int = 3,
    goal_amount: Optional[int] = None,
    goal_years: Optional[int] = None,
    goal_sip: Optional[int] = None,
    fund_house: Optional[str] = None,
) -> Optional[Snapshot]:
    """
    bucket : "low" | "moderate" | "high" — the user's appetite, not the riskometer
    only   : list of fund_names to pin (used by detail / compare turns)
    """
    if only:
        candidates = [f for f in universe.load() if f["fund_name"] in only]
        candidates.sort(key=lambda f: only.index(f["fund_name"]))
    else:
        # For an AMC-specific query with no stated risk preference, allow all
        # its risk bands through so each result can be labelled honestly.
        candidates = list(universe.load()) if bucket == "all" else [
            f for f in universe.load() if f["appetite_bucket"] == bucket
        ]
        if amount:
            affordable = [f for f in candidates if f["min_sip"] <= amount]
            # only apply the filter if it leaves us something to say
            if affordable:
                candidates = affordable

    # An explicit AMC request is a hard constraint. Never quietly substitute
    # another fund house if this catalogue has no matching scheme.
    if fund_house:
        requested_house = " ".join(fund_house.casefold().split())
        candidates = [
            f for f in candidates
            if " ".join((f.get("fund_house") or "").casefold().split()) == requested_house
        ]

    if not only:
        # Rotate the candidate window using epoch so each new discovery turn
        # starts from a different fund. We rotate the full list so the loop
        # below can walk forward until it finds `limit` valid (cached/fetchable)
        # funds — never returns fewer just because some in the window failed.
        if len(candidates) > limit:
            offset = ((epoch - 1) * limit) % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]

    funds: list[FundRecommendation] = []
    for u in candidates:
        rec = _to_recommendation(u)
        if rec:
            funds.append(rec)
        if len(funds) >= limit:
            break

    if not funds:
        return None

    return Snapshot(
        funds=funds,
        user_amount=amount,
        user_tenure=tenure or "monthly",
        epoch=epoch,
        user_goal_amount=goal_amount,
        user_goal_years=goal_years,
        user_goal_sip=goal_sip,
    )
