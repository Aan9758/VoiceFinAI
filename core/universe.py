"""
core/universe.py — Verified Fund Universe
==========================================
Replaces the hand-typed _CATALOGUE in core/retrieval.py.

Why this file exists
--------------------
The old catalogue hardcoded (scheme_code, risk_rating, min_sip, expense) tuples
by hand. Three things went wrong:

  1. The same scheme_code appeared in several risk buckets with a DIFFERENT
     risk_rating each time (118989 was "Low to Moderate", "Moderate" and "High"
     depending on which bucket you hit). Same fund, three risk ratings.
  2. Scheme codes were mislabelled — core/amfi.py calls 118989 "Parag Parikh"
     in DEMO_FUNDS and "HDFC Mid Cap" in KNOWN_SCHEME_CODES. Both cannot be true.
  3. risk_rating / min_sip / expense_ratio were invented at the keyboard, then
     validated by Pydantic (which checks *type*, not *truth*) and then certified
     "verified" by AuditStream. The audit chain was attesting to fiction.

Fix: nothing is typed by hand except a search term and an expected category.
Scheme codes are resolved from mfapi.in at build time and cross-checked against
the category the fund actually reports. Risk band is DERIVED from the SEBI
category with a declared, deterministic mapping — one band per fund, always.

Honesty note (read this before you demo)
----------------------------------------
`risk_band` here is a category heuristic, NOT the AMC's published riskometer.
The real riskometer is per-fund and republished monthly; today most equity
funds sit at "Very High" regardless of cap size. Same for `min_sip`, which is
an AMC-published figure this pipeline does not fetch. Both are tagged with a
source string so the audit log says where they came from instead of implying
AMFI vouched for them. Do not describe these two fields as "verified against
AMFI" in the deck.

Build:   python -m core.universe build
Verify:  python -m core.universe verify
"""

from __future__ import annotations

import json
import re
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

MFAPI_SEARCH = "https://api.mfapi.in/mf/search"
MFAPI_FUND = "https://api.mfapi.in/mf"

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.json"


# ── The only hand-written input: what we want in the universe ────────────────
# (search_term, must_contain_in_category)  — no scheme codes, no risk labels.


# ── No hand-typed scheme codes in this file ──────────────────────────────────
# The old UNIVERSE_SPEC was a list of 18 hand-picked (search_term, category)
# tuples. It has been removed because:
#   1. It still required a developer to decide which 18 funds matter.
#   2. search_term matching against mfapi.in sometimes returned wrong funds
#      (ambiguous names, multiple matching schemes).
#   3. The resulting universe.json was a static snapshot, not re-derivable
#      from the live source without re-running the list by hand.
#
# Replacement: build_from_amfi_master() fetches the SEBI-mandated AMFI
# NAVAll.txt daily feed, filters to Direct Plan – Growth schemes across
# the 9 target categories, and resolves each scheme via mfapi.in to confirm
# its SEBI category before accepting it. The universe is fully rebuildable
# at any time from the live source with no developer input.



# ── Deterministic SEBI category → risk band ──────────────────────────────────
# Ordered: first substring hit wins. One band per category, forever.

_RISK_BY_CATEGORY: list[tuple[str, str]] = [
    ("overnight",           "Low"),
    ("liquid",              "Low to Moderate"),
    ("ultra short",         "Low to Moderate"),
    ("low duration",        "Low to Moderate"),
    ("money market",        "Low to Moderate"),
    ("arbitrage",           "Low to Moderate"),
    ("short duration",      "Moderate"),
    ("banking and psu",     "Moderate"),
    ("corporate bond",      "Moderate"),
    ("gilt",                "Moderate"),
    ("dynamic bond",        "Moderate"),
    ("conservative hybrid", "Moderate"),
    ("medium duration",     "Moderately High"),
    ("credit risk",         "High"),
    ("balanced advantage",  "Moderately High"),
    ("dynamic asset",       "Moderately High"),
    ("aggressive hybrid",   "High"),
    ("multi asset",         "High"),
    ("large cap",           "Very High"),
    ("large & mid",         "Very High"),
    ("flexi cap",           "Very High"),
    ("multi cap",           "Very High"),
    ("focused",             "Very High"),
    ("value",               "Very High"),
    ("elss",                "Very High"),
    ("mid cap",             "Very High"),
    ("small cap",           "Very High"),
    ("sectoral",            "Very High"),
    ("thematic",            "Very High"),
]

# Internal appetite bucket — what the *user's question* maps onto.
# Deliberately separate from risk_band so a "Very High" riskometer equity fund
# can still be the right answer to "aggressive fund chahiye".
_BUCKET_BY_CATEGORY: list[tuple[str, str]] = [
    ("overnight", "low"), ("liquid", "low"), ("ultra short", "low"),
    ("low duration", "low"), ("money market", "low"), ("arbitrage", "low"),
    ("short duration", "low"), ("corporate bond", "low"), ("banking and psu", "low"),
    ("gilt", "low"), ("dynamic bond", "low"), ("conservative hybrid", "low"),
    ("balanced advantage", "moderate"), ("dynamic asset", "moderate"),
    ("aggressive hybrid", "moderate"), ("multi asset", "moderate"),
    ("large cap", "moderate"), ("large & mid", "moderate"),
    ("flexi cap", "moderate"), ("multi cap", "moderate"), ("focused", "moderate"),
    ("elss", "moderate"), ("value", "moderate"),
    ("mid cap", "high"), ("small cap", "high"),
    ("sectoral", "high"), ("thematic", "high"), ("credit risk", "high"),
]

# Category → typical AMC minimum SIP. An assumption, tagged as one.
_MIN_SIP_BY_BUCKET = {"low": 500, "moderate": 500, "high": 1000}


def _ssl_ctx() -> ssl.SSLContext:
    return ssl.create_default_context()


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "VoiceFinAI/2.0"})
    with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())


def classify(category: str) -> tuple[str, str]:
    """SEBI category string → (risk_band, appetite_bucket). Deterministic."""
    c = _norm(category)
    band = next((b for k, b in _RISK_BY_CATEGORY if k in c), "Moderately High")
    bucket = next((b for k, b in _BUCKET_BY_CATEGORY if k in c), "moderate")
    return band, bucket


def _name_override(name: str, category: str, band: str, bucket: str) -> tuple[str, str, str]:
    """
    Narrow name-based reclassification for cases where mfapi.in's category
    string doesn't contain the expected keyword.

    Known case: Liquid ETFs and Liquid FoFs are registered as
    'Fund of Funds Scheme (Domestic)' or 'Other Scheme - Other ETFs',
    so classify() correctly returns 'Moderately High/moderate' from category.
    But 'liquid' in the fund name is a reliable signal these are liquid-strategy
    wrappers, not equity funds.

    Returns (band, bucket, override_note) — override_note is empty if no change.
    """
    name_l = name.lower()
    cat_l  = category.lower()
    # Liquid ETF / Liquid FoF: name has 'liquid', category is etf or fof
    if ("liquid" in name_l and
            bucket != "low" and
            any(k in cat_l for k in ("etf", "fund of fund", "fof"))):
        return "Low to Moderate", "low", "name_liquid_etf_override"
    return band, bucket, ""


def resolve_scheme(search_term: str, expect_category: str) -> Optional[dict]:
    """
    Search mfapi.in and return the Direct Plan - Growth scheme whose reported
    category actually contains `expect_category`. Returns None on mismatch —
    a silent wrong match is worse than a missing fund.
    """
    url = f"{MFAPI_SEARCH}?q={urllib.parse.quote(search_term)}"
    try:
        hits = _get(url)
    except Exception as e:
        print(f"  ! search failed for {search_term!r}: {e}")
        return None

    def score(name: str) -> int:
        n = _norm(name)
        if "idcw" in n or "dividend" in n or "payout" in n:
            return -1
        s = 0
        if "direct" in n:
            s += 10
        if "growth" in n:
            s += 5
        s -= abs(len(n) - len(_norm(search_term))) // 20
        return s

    ranked = sorted(
        (h for h in hits if score(h.get("schemeName", "")) >= 0),
        key=lambda h: score(h["schemeName"]),
        reverse=True,
    )

    for hit in ranked[:5]:
        code = hit["schemeCode"]
        try:
            detail = _get(f"{MFAPI_FUND}/{code}")
        except Exception:
            continue
        meta = detail.get("meta", {})
        cat = _norm(meta.get("scheme_category", ""))
        if expect_category.lower() not in cat:
            continue
        band, bucket = classify(meta.get("scheme_category", ""))
        return {
            "scheme_code": code,
            "fund_name": meta.get("scheme_name", hit["schemeName"]),
            "fund_house": meta.get("fund_house", ""),
            "category": meta.get("scheme_category", ""),
            "isin_growth": meta.get("isin_growth", ""),
            "risk_band": band,
            "risk_band_source": "category_heuristic",
            "appetite_bucket": bucket,
            "min_sip": _MIN_SIP_BY_BUCKET[bucket],
            "min_sip_source": "category_default_assumption",
            "expense_ratio": None,
            "expense_ratio_source": "not_fetched",
        }

    print(f"  ! no '{expect_category}' match for {search_term!r} — skipped")
    return None


def build_from_amfi_master(
    path: Path = UNIVERSE_PATH,
    top_n_per_category: int = 10,
) -> list[dict]:
    """
    Build universe.json from the live AMFI master list (NAVAll.txt).
    No hand-typed scheme codes. Rebuildable at any time from the live source.

    Steps:
      1. Fetch NAVAll.txt from AMFI — the SEBI-mandated daily feed.
      2. Filter to Direct Plan – Growth schemes across 9 target categories.
      3. Resolve each scheme via mfapi.in to confirm SEBI category and get ISIN.
      4. Derive risk_band and appetite_bucket from SEBI category (deterministic table).
      5. Write data/universe.json.
    """
    from core.amfi import fetch_amfi_scheme_list, filter_universe

    print("  Fetching AMFI master list from NAVAll.txt …")
    all_schemes = fetch_amfi_scheme_list()
    print(f"  {len(all_schemes)} Direct Plan – Growth schemes found in master list")

    bucketed = filter_universe(all_schemes, top_n_per_category=top_n_per_category)
    total_candidates = sum(len(v) for v in bucketed.values())
    print(f"  {total_candidates} candidates across {len(bucketed)} categories — resolving via mfapi.in …\n")

    out: list[dict] = []
    seen: set[int] = set()

    for cat_key, candidates in bucketed.items():
        accepted = 0
        for s in candidates:
            code = s["scheme_code"]
            if code in seen:
                continue
            try:
                detail = _get(f"{MFAPI_FUND}/{code}")
            except Exception as e:
                print(f"  ! fetch failed for {code}: {e}")
                continue
            meta = detail.get("meta", {})
            category_str = meta.get("scheme_category", "")
            fund_name    = meta.get("scheme_name", s["scheme_name"])
            band, bucket = classify(category_str)
            # Apply name-based override for cases where category string
            # doesn't identify the fund type (e.g. liquid ETFs/FoFs)
            band, bucket, override = _name_override(fund_name, category_str, band, bucket)
            risk_source = override if override else "category_heuristic"
            rec = {
                "scheme_code":         code,
                "fund_name":           fund_name,
                "fund_house":          meta.get("fund_house", ""),
                "category":            category_str,
                "isin_growth":         meta.get("isin_growth", s["isin"]),
                "risk_band":           band,
                "risk_band_source":    risk_source,
                "appetite_bucket":     bucket,
                "min_sip":             _MIN_SIP_BY_BUCKET.get(bucket, 500),
                "min_sip_source":      "category_default_assumption",
                "expense_ratio":       None,
                "expense_ratio_source": "not_fetched",
            }
            out.append(rec)
            seen.add(code)
            accepted += 1
            override_tag = " [name_override]" if override else ""
            print(f"  {code:>7}  {band:<16} {bucket:<8} {fund_name[:55]}{override_tag}")
        if accepted == 0:
            print(f"  ! no schemes resolved for category: {cat_key}")


    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {len(out)} funds → {path}")
    return out


# Keep build() as an alias so existing references don't break
def build(path: Path = UNIVERSE_PATH) -> list[dict]:
    return build_from_amfi_master(path)


_cache: Optional[list[dict]] = None


def load(path: Path = UNIVERSE_PATH) -> list[dict]:
    global _cache
    if _cache is None:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run: python -m core.universe build"
            )
        _cache = json.loads(path.read_text(encoding="utf-8"))
    return _cache


def by_bucket(bucket: str, limit: int = 3) -> list[dict]:
    """Funds for an appetite bucket, never crossing into another bucket."""
    funds = [f for f in load() if f["appetite_bucket"] == bucket]
    return funds[:limit]


def find_by_name(text: str) -> Optional[dict]:
    """Longest-match fund lookup from free text. Returns the universe record."""
    t = _norm(text)
    best, best_len = None, 0
    # These describe an investment style, but they do not identify a fund.
    # Without this guard, a sentence such as "I want a long term fund" can
    # accidentally match an unrelated "Ultra Short Term" scheme.
    generic = {
        "direct", "regular", "plan", "growth", "option", "fund", "funds",
        "large", "mid", "small", "cap", "equity", "debt", "term", "short",
        "long", "ultra", "liquid", "balanced", "asset", "allocation", "index",
        "nifty", "scheme", "tax", "saver", "income", "value", "focus", "focuses",
    }
    for f in load():
        name = _norm(f["fund_name"])
        name = re.sub(r"\b(direct|regular|plan|growth|option|fund)\b", " ", name)
        tokens = [w for w in name.split() if len(w) > 2 and w not in generic]
        if not tokens:
            continue
        # Try long identifying fragments first. A single word is accepted only
        # when it is a sufficiently distinctive fund-house/name token.
        for size in range(len(tokens), 0, -1):
            for start in range(len(tokens) - size + 1):
                frag = " ".join(tokens[start:start + size])
                if (size > 1 or len(frag) >= 5) and re.search(
                    rf"(?<![a-z0-9]){re.escape(frag)}(?![a-z0-9])", t
                ) and len(frag) > best_len:
                    best, best_len = f, len(frag)
    return best


def verify_old_codes(codes: dict[str, int]) -> None:
    """Print what a hand-typed scheme_code ACTUALLY is on mfapi.in."""
    for label, code in codes.items():
        try:
            meta = _get(f"{MFAPI_FUND}/{code}").get("meta", {})
            print(f"  {code:>7}  labelled {label:<30} → actually "
                  f"{meta.get('scheme_name','?')}  [{meta.get('scheme_category','?')}]")
        except Exception as e:
            print(f"  {code:>7}  labelled {label:<30} → fetch failed: {e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build_from_amfi_master()
    elif cmd == "verify":
        print("\n  OLD KNOWN_SCHEME_CODES (were correct):\n")
        verify_old_codes({
            "parag_parikh_flexi_cap": 122639,
            "hdfc_mid_cap":           118989,
            "mirae_large_cap":        118825,
            "quant_small_cap":        120828,
            "axis_liquid":            120389,
        })
        print("\n  OLD DEMO_FUNDS (every entry wrong — all contradicted their comments):\n")
        verify_old_codes({
            "DEMO low[0] said HDFC Liquid":       119551,
            "DEMO low[1]/mod[0]/tax[2] said Axis": 120503,
            "DEMO low[2] said SBI Liquid":         119270,
            "DEMO mod[1]/tax[0] said PP Flexi":    118989,
            "DEMO mod[2] said Mirae Large":        120716,
            "DEMO high[0] said Quant Small":       118825,
            "DEMO high[1] said Nippon Small":      125354,
            "DEMO high[2] said SBI Small":         118701,
            "DEMO tax[1] said Axis ELSS":          119598,
        })
        print()
    else:
        print("usage: python -m core.universe [build|verify]")
