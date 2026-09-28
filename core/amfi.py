"""
core/amfi.py — AMFI Data Fetcher
=================================
Fetches live fund data from mfapi.in (free, SEBI-mandated daily updates).
Computes 1yr / 3yr / 5yr CAGR from NAV history.
Builds the fund record that feeds into the snapshot.

All data is fetched at query time during Track 3 (RAG pre-fetch).
No LLM generates any of these numbers.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime, timedelta
from typing import Optional
import ssl
import urllib.request
import urllib.error
import json


# ── Constants ────────────────────────────────────────────────────────────────

MFAPI_BASE       = "https://api.mfapi.in/mf"
AMFI_SCHEME_URL  = "https://www.amfiindia.com/spages/NAVAll.txt"

# SEBI-approved risk rating labels (exact strings used in FundRecommendation)
RISK_LABELS = {"Low", "Low to Moderate", "Moderate", "Moderately High", "High", "Very High"}


# ── SSL context (handles Windows cert issues) ────────────────────────────────

def _ssl_ctx() -> ssl.SSLContext:
    return ssl.create_default_context()


def _get_json(url: str) -> dict | list:
    """HTTP GET → parsed JSON. Handles Windows SSL cert issues."""
    ctx = _ssl_ctx()
    req = urllib.request.Request(url, headers={"User-Agent": "VoiceFinAI/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── CAGR computation ─────────────────────────────────────────────────────────

def _cagr(nav_now: float, nav_then: float, years: float) -> Optional[float]:
    """
    Compute CAGR from two NAV points.
    Returns None if data is insufficient or invalid.
    Uses growth-plan NAV (not IDCW) — caller must ensure correct series.
    """
    if not nav_then or nav_then <= 0 or years <= 0:
        return None
    try:
        ratio = nav_now / nav_then
        return round((ratio ** (1.0 / years) - 1.0) * 100, 2)
    except (ZeroDivisionError, ValueError):
        return None


def _nav_on_or_before(nav_data: list[dict], target: date) -> Optional[float]:
    """
    Find the NAV value on or before `target` date.
    nav_data is list of {"date": "DD-MMM-YYYY", "nav": "123.45"}.
    mfapi returns newest-first.
    """
    for entry in nav_data:
        try:
            entry_date = datetime.strptime(entry["date"], "%d-%m-%Y").date()
            if entry_date <= target:
                return float(entry["nav"])
        except (ValueError, KeyError):
            continue
    return None


# ── Core fetch ───────────────────────────────────────────────────────────────

def fetch_fund_record(scheme_code: int) -> Optional[dict]:
    """
    Fetch a complete fund record from mfapi.in.
    Returns a dict with all fields needed for the FundRecord + CAGR values.
    Returns None if the fetch fails or data is insufficient.

    Fields returned:
        scheme_code, fund_name, isin_growth, nav, nav_date,
        cagr_1yr, cagr_3yr, cagr_5yr, fund_house, category
    """
    try:
        data = _get_json(f"{MFAPI_BASE}/{scheme_code}")
    except Exception as e:
        return None

    meta   = data.get("meta", {})
    prices = data.get("data", [])

    if not prices:
        return None

    # Latest NAV
    try:
        latest_nav  = float(prices[0]["nav"])
        latest_date = datetime.strptime(prices[0]["date"], "%d-%m-%Y").date()
    except (ValueError, KeyError, IndexError):
        return None

    today = latest_date  # treat latest available as "today"

    # CAGR from NAV history — independently computed, reconcilable against AMFI/VR
    cagr_1yr = _cagr(latest_nav, _nav_on_or_before(prices, today - timedelta(days=365)), 1.0)
    cagr_3yr = _cagr(latest_nav, _nav_on_or_before(prices, today - timedelta(days=365*3)), 3.0)
    cagr_5yr = _cagr(latest_nav, _nav_on_or_before(prices, today - timedelta(days=365*5)), 5.0)

    return {
        "scheme_code" : scheme_code,
        "fund_name"   : meta.get("scheme_name", ""),
        "fund_house"  : meta.get("fund_house", ""),
        "category"    : meta.get("scheme_category", ""),
        "isin_growth" : meta.get("isin_growth", ""),
        "isin_div"    : meta.get("isin_div", ""),
        "nav"         : latest_nav,
        "nav_date"    : latest_date.isoformat(),
        "cagr_1yr"    : cagr_1yr,
        "cagr_3yr"    : cagr_3yr,
        "cagr_5yr"    : cagr_5yr,
        "source"      : "mfapi.in",
        "as_of"       : latest_date.isoformat(),
    }



# ── Live AMFI master list ────────────────────────────────────────────────────
# NAVAll.txt is the SEBI-mandated daily feed. Every scheme is listed.
# We parse it to build the fund universe programmatically — no hand-typed codes.
#
# Format (pipe-separated):
#   Scheme Code;ISIN Div Payout/IDCW;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date
# Sections are separated by blank lines with an AMC header.

def fetch_amfi_scheme_list() -> list[dict]:
    """
    Fetch the full AMFI master scheme list from NAVAll.txt.
    Returns list of {scheme_code, scheme_name, isin_growth, nav_str} for
    Direct Plan – Growth schemes only.

    NAVAll.txt format (8 semicolon-separated columns):
      Scheme Code ; ISIN Div Payout ; ISIN Div Reinvestment ; Scheme Name ;
      Plan ; Option ; Net Asset Value ; Date

    Direct Plan Growth is col 4 == "Direct Plan" AND col 5 contains "growth"
    (case-insensitive). IDCW / Dividend / Reinvestment options are excluded.
    """
    ctx = _ssl_ctx()
    req = urllib.request.Request(
        AMFI_SCHEME_URL,
        headers={"User-Agent": "VoiceFinAI/2.0"},
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch AMFI master list: {e}") from e

    schemes = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or ";" not in line:
            continue
        parts = line.split(";")
        if len(parts) < 7:
            continue
        try:
            code = int(parts[0].strip())
        except ValueError:
            continue
        isin_div    = parts[1].strip()
        isin_growth = parts[2].strip()  # "-" if no growth ISIN separate from div
        name        = parts[3].strip()
        plan        = parts[4].strip().lower()
        option      = parts[5].strip().lower()
        nav_str     = parts[6].strip()

        # Only Direct Plan
        if "direct" not in plan:
            continue
        # Only Growth option — exclude IDCW, dividend, reinvestment, payout, bonus
        if not any(x in option for x in ("growth",)):
            continue
        if any(x in option for x in ("idcw", "dividend", "reinvestment", "payout", "bonus")):
            continue

        # isin_growth is col 2 for most funds; for some it's col 1 (isin_div)
        isin = isin_growth if isin_growth and isin_growth != "-" else isin_div

        schemes.append({
            "scheme_code": code,
            "scheme_name": name,
            "isin": isin,
            "nav_str": nav_str,
        })
    return schemes


# SEBI category keywords used to bucket schemes from NAVAll.txt.
# Keys must appear as substrings in the scheme name (case-insensitive).
# These are broad enough to catch standard naming conventions.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "liquid":     ["liquid fund", "liquid "],
    "low_dur":    ["low duration", "ultra short", "money market", "overnight"],
    "corp_bond":  ["corporate bond", "banking and psu", "banking & psu"],
    "balanced":   ["balanced advantage", "dynamic asset allocation"],
    "flexi_cap":  ["flexi cap"],
    "large_cap":  ["large cap", "largecap", "bluechip", "blue chip"],
    "mid_cap":    ["mid cap", "midcap"],
    "small_cap":  ["small cap", "smallcap"],
    "elss":       ["elss", "tax saver", "tax saving"],
}


def filter_universe(
    schemes: list[dict],
    top_n_per_category: int = 10,
) -> dict[str, list[dict]]:
    """
    From the full AMFI list, pick the top_n_per_category schemes per category.
    Returns dict {category_key: [scheme_dict, ...]} — no hand-typed codes.

    Selection: highest scheme_code first within each category (newer registrations
    tend to be Direct Plan – Growth; older codes are often legacy plans).
    A scheme is assigned to the FIRST matching category only (no duplicates).
    """
    seen: set[int] = set()
    result: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_KEYWORDS}

    # Sort descending by scheme_code so newer / more specific plans come first
    sorted_schemes = sorted(schemes, key=lambda s: s["scheme_code"], reverse=True)

    for s in sorted_schemes:
        if s["scheme_code"] in seen:
            continue
        name_lower = s["scheme_name"].lower()
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if len(result[cat]) >= top_n_per_category:
                continue
            if any(kw in name_lower for kw in keywords):
                result[cat].append(s)
                seen.add(s["scheme_code"])
                break  # first-match only — no duplicates across categories

    return result

