"""
core/dialogue.py — Session State + Intent Router
================================================
Replaces parse_intent() in core/retrieval.py.

What was broken
---------------
1. No session. _run_pipeline(query) was stateless, so "iske baare mein batao",
   "pehle wale", "compare karo", "start it" had no referent and all fell through
   to the same generic branch.
2. First-match-wins substring matching. `_LOW_RISK` contained the bare string
   "low", and it was tested before the other buckets — so "I don't want low
   risk" scored as low risk. `"ten" in "tenure"` also fired the amount parser.
3. Risk defaulted to "moderate" on every unmatched query, and the moderate
   branch produced one fixed sentence. That is the exact sentence the demo got
   stuck on. Every query the keyword lists didn't cover produced it.
4. Only two intents were actually routed (discover, fund_detail-with-a-name).
   compare / confirm / metric follow-ups fell through to discover.

What this does instead
----------------------
- Scored, word-boundary matching across all buckets — highest score wins, and a
  tie or a zero score routes to CLARIFY rather than silently defaulting.
- Negation-aware ("low risk nahi chahiye" does not score low).
- A Session object carrying the last snapshot, the focused fund, and sticky
  risk/amount so follow-up turns resolve pronouns and ordinals.
- Eight routed intents, each with its own downstream handler.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from core import universe

# ── IntentRouter integration ──────────────────────────────────────────────────
# Import the new, compiled-regex intent router. It replaces the hand-rolled
# substring matching for tiers 7b–10 (WHY/FUND_DETAIL/DISCOVER/CLARIFY).
# The router is stateless — one instance shared across all sessions.
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from intent_router import IntentRouter as _IntentRouter, Session as _IRSession, Risk as _IRRisk

def _build_router() -> _IntentRouter:
    """Build IntentRouter with the live fund catalog for name-recognition."""
    try:
        funds = universe.load()
        catalog = {}
        for rec in funds:
            fname = rec.get("fund_name", "")
            if not fname:
                continue
            clean = re.sub(r'\s*-\s*Direct\s+Plan.*$', '', fname, flags=re.IGNORECASE)
            clean = re.sub(r'\s*-\s*Regular\s+Plan.*$', '', clean, flags=re.IGNORECASE)
            clean = re.sub(r'\s+Fund$', '', clean, flags=re.IGNORECASE).strip()
            catalog[fname] = [fname, clean]
        return _IntentRouter(fund_catalog=catalog)
    except Exception:
        return _IntentRouter()

_INTENT_ROUTER: _IntentRouter = _build_router()


# ── Intents ───────────────────────────────────────────────────────────────────

GREET = "greeting"
DISCOVER = "discover"
FUND_DETAIL = "fund_detail"
COMPARE = "compare"
METRIC = "metric"
CONFIRM_SIP = "confirm_sip"
GRAPH = "graph"
GOAL = "goal"
OUT_OF_SCOPE = "out_of_scope"
CLARIFY = "clarify"
LANG_SWITCH = "language_switch"


# ── Language Detection & Switching ───────────────────────────────────────────

_LANG_SWITCH_EN = [
    "speak in english", "talk in english", "switch to english", "english please",
    "in english", "can you speak english", "change language to english", "use english",
    "speak english", "talk english",
]
_LANG_SWITCH_HI = [
    "speak in hindi", "talk in hindi", "hindi me bolo", "hindi mein bolo",
    "switch to hindi", "hindi please", "hinglish please", "in hindi", "in hinglish",
    "hindi me batao", "hindi mein batao", "switch to hinglish", "use hinglish",
    "speak hinglish", "talk hinglish", "hinglish me bolo", "hinglish mein bolo",
]

_HINGLISH_WORDS = frozenset({
    "karna", "karo", "kare", "karen", "chahiye", "hai", "hain", "batao", "bataiye",
    "saal", "rupaye", "rupiya", "paisa", "paise", "lagana", "lagau", "kisme", "kaunsa",
    "kaun", "hoga", "hogi", "mujhe", "mera", "meri", "mere", "aapka", "aapki", "aapke",
    "mein", "ko", "ke", "ki", "ka", "liye", "bhi", "toh", "yeh", "ye",
    "woh", "wo", "kya", "kyun", "kyu", "kaisa", "kaisi", "kaise", "ek", "teen",
    "chaar", "paanch", "panch", "chhe", "saat", "aath", "nau", "das", "pachees",
    "hazaar", "hazar", "dono", "sabse", "shuru", "pehle", "pehla", "doosra", "teesra",
    "theek", "acha", "accha", "sahi", "samjhao", "dikhao", "lekin", "magar", "chhod",
    "inme", "iska", "iski", "iske", "uska", "uski", "uske", "kuch", "aur", "tarah", "wala",
    "rakhein", "jaisa", "aise", "upar", "aakhri", "karein", "bata", "dikha", "samjha"
})

_ENGLISH_MARKERS = frozenset({
    "i", "want", "to", "invest", "in", "high", "risk", "funds", "what", "is",
    "the", "difference", "between", "which", "one", "should", "choose", "tell",
    "me", "more", "about", "show", "graph", "how", "much", "needed", "for",
    "my", "goal", "years", "crore", "crores", "lakh", "lakhs", "rupees",
    "accumulate", "wealth", "accumulation", "retirement", "child", "education",
    "house", "buying", "confirm", "start", "proceed", "portfolio", "returns",
    "safe", "balanced", "growth", "planning", "best", "option", "options",
    "can", "you", "explain", "details", "calculate", "monthly", "investment",
    "please", "help", "suggest", "compare", "performance", "equity", "debt"
})


def detect_language(text: str, default: str = "hinglish") -> str:
    """
    Detects whether text is 'hinglish' or 'english' using lexical marker frequency.
    """
    # A Devanagari character is an unambiguous user preference for a Hindi or
    # Hinglish reply, even when the same sentence includes English finance
    # terms such as "SIP" or "long term".
    if re.search(r"[\u0900-\u097f]", text):
        return "hinglish"
    tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
    hinglish_hits = len(tokens.intersection(_HINGLISH_WORDS))
    english_hits = len(tokens.intersection(_ENGLISH_MARKERS))
    if hinglish_hits > 0:
        return "hinglish"
    if english_hits > 0:
        return "english"
    return default



# ── Vocabulary. Phrases only — no bare ambiguous tokens like "low" or "ten". ──

_RISK_LEXICON: dict[str, list[str]] = {
    "low": [
        "low risk", "kam risk", "safe", "surakshit", "stable", "secure",
        "liquid fund", "debt fund", "debt", "bond", "overnight", "conservative",
        "capital protection", "koi risk nahi", "bilkul risk nahi", "guaranteed",
        "emergency fund", "short term", "paisa safe", "fd jaisa", "fd ke jaisa",
        "sahi", "seedha", "simple", "no risk", "risk nahi", "debt chahiye",
        "debt wala", "safe chahiye", "safe fund", "safe funds", "safe rakhna",
    ],
    "moderate": [
        "moderate", "balanced", "medium risk", "thoda risk", "beech ka",
        "flexi cap", "large cap", "largecap", "bluechip", "blue chip",
        "hybrid", "mix", "not too risky", "zyada risk nahi",
        "sip", "mutual fund", "invest karna", "invest karo", "lagana",
        "paisa lagana", "investment chahiye", "fund chahiye",
        "equity", "equity fund", "equity funds", "equity chahiye", "equities",
        "pure equity", "equity me", "equity mein", "equity wala", "equity market",
    ],
    "high": [
        "high risk", "aggressive", "zyada risk", "small cap", "smallcap",
        "mid cap", "midcap", "zyada return", "maximum return", "high return",
        "best return", "top performing", "wealth creation", "long term",
        "multibagger", "growth stock", "double karna",
        "growth", "growth fund", "growth funds", "growth chahiye", "high growth",
        "risk le sakte", "risk le sakta", "risk chalega", "risk le lenge",
        "can take risk", "willing to take risk", "high risk equity",
        "growth equity", "aggressive equity",
    ],
}

_OOS = [
    "crypto", "bitcoin", "btc", "ethereum", "eth", "nft", "forex", "real estate",
    "property", "plot kharidna", "share market", "direct stock", "stock price",
    "f&o", "futures and options", "intraday", "trading tips", "insurance policy", "lic",
    "gold rate", "silver rate", "weather", "mausam", "cricket", "who is", "who was",
    "kaun hai", "koun hai", "president", "prime minister", "pm modi", "elon musk",
    "recipe", "cook", "biryani", "joke", "chutkula", "movie", "film", "song", "gaana",
]

_GREET = [
    "hello", "hi", "hey", "hi madhur", "hey madhur", "hi mridul", "hey mridul", "namaste", "namaskar", "pranam",
    "good morning", "good afternoon", "good evening", "kaise ho", "how are you", "kya haal",
    "suno", "madhur", "mridul", "namste",
]

_DETAIL = ["kaise hai", "kaisa hai", "kaisi hai", "batao", "bataiye", "tell me",
           "explain", "detail", "baare mein", "bare mein", "about this",
           "iske baare", "uske baare", "kya hai ye", "more info", "elaborate"]

_COMPARE = [
    "compare", "comparison", "vs", "versus", "difference", "antar", "farak", "farq",
    "dono mein", "dono me", "dono fund", "compare karo", "comparison dikhao",
    "dono me se", "dono mein se", "tulna", "compare them", "competition",
    "competition show", "competition dikhao", "comparison show", "compition",
]

_ADVISE = [
    "kaunsa lu", "kaun sa lu", "kaunsa accha", "kaun sa accha", "kaunsa acha", "kaun sa acha",
    "kaunsa best", "kaun sa best", "kaunsa sahi", "kaun sa sahi", "kisme lagau", "kisme invest karu",
    "kisme dalu", "kisme daalu", "which one", "which is best", "which one is good", "which will be the best",
    "which would be the best", "which will be best", "best for me", "which is best for me",
    "from these three which", "from these three", "from these", "inme se kaunsa", "inme se kaun sa",
    "inme se mere liye", "mere liye kaunsa", "which fund is good", "which to choose", "which should i choose",
    "what do you suggest", "suggest one", "recommend one", "suggest karo", "recommend karo",
    "suggest", "recommend", "tell me more", "inme se", "best kaunsa", "best option",
    "aap batao", "tum batao", "kya suggest", "choose", "pick", "kisme",
]

_GRAPH = [
    "graph", "chart", "performance graph", "graph dikhao", "chart dikhao",
    "show me graph", "show chart", "returns chart", "growth chart", "plot",
    "graph bhi dikhao", "cagr graph", "sip graph", "show graph", "visualize",
    "graph dekhna", "chart dekhna", "graph show", "chart show",
]

_WEALTH_EXPLAIN = [
    "wealth accumulation", "wealth accumalation", "wealth acucmalation",
    "what is this wealth", "what is wealth", "what does wealth",
    "what is this wealth accumulation", "what is wealth accumulation",
    "what does wealth accumulation mean", "wealth accumulation kya hai",
    "ye wealth accumulation kya hai", "wealth accumulation kya hota hai",
    "explain wealth accumulation", "explain this wealth accumulation",
    "wealth accumulation samjhao", "wealth accumulation ka matlab",
    "explain graph", "explain the graph", "explain this graph",
    "graph explain", "graph explain karo", "graph samjhao",
    "chart explain", "chart explain karo", "chart samjhao", "ye chart kya hai",
    "what does this graph show", "how is this calculated", "calculation kaise",
    "ye calculation kaise", "future value", "what is future value", "future value kya hai",
    "what is gain", "gain kya hai", "gain ka matlab", "invested vs gain",
    "ye graph kya", "graph kya dikha", "graph kya hai", "chart kya dikha", "ye chart kya", "graph me kya",
]

_WHY = [
    "why did you choose", "why this", "kyun chuna", "kyu chuna", "why choose",
    "reason kya hai", "why this fund", "kyun recommend kiya", "why is this the best",
    "why best", "kyun liya", "kyun suggest kiya", "why suggest", "reason",
    "kyun accha hai", "why it choose", "why you choose",
    "how did you pick", "why did you pick", "what makes", "what criteria",
    "why have you chosen", "why was this chosen",
]
_CONFIRM = [
    "start it", "shuru karo", "start karo", "chalu karo", "haan kar do", "haan kardo",
    "yes start", "invest karo", "book karo", "confirm", "go ahead",
    "let's do it", "kar do", "kardo", "theek hai start", "start my sip", "start sip",
    "start investment", "start investing", "sip shuru", "sip start",
    "invest in this", "invest in this fund", "start in this", "start in this fund",
    "proceed with sip", "proceed", "lock it in",
    "haan", "haanji", "haan ji", "yes", "theek hai", "ok", "okay", "sure", "bilkul",
    "shuru kardo", "shuru kar do", "done", "laga do", "lagao", "ha", "haa",
    "kar dena", "proceed karo", "kar do start", "kardo start",
]

_NEGATORS = ["nahi", "nahin", "not ", "don't", "dont ", "no ", "mat ", "chhod",
             "avoid", "except", "alag", "koi aur", "dusra", "doosra dikha"]

# Which snapshot field a follow-up question is asking for
_METRIC_FIELDS: list[tuple[list[str], str]] = [
    (["expense ratio", "expense", "charges", "kitna charge", "fees", "fee", "cost"], "expense"),
    (["lock in", "lock-in", "lockin", "lock kitna", "kitne saal lock", "lock in period"], "lockin"),
    (["minimum sip", "min sip", "kitne se start", "kitna minimum", "minimum kitna",
      "kam se kam kitna", "starting amount", "minimum investment"], "minsip"),
    (["nav", "current nav", "nav kya", "nav kitna", "nav bata", "price kya", "latest nav", "share price"], "nav"),
    (["5 year", "five year", "paanch saal", "5 saal", "long term return", "5 yr return"], "cagr5"),
    (["1 year", "one year", "ek saal", "1 saal", "last year return", "1 yr return"], "cagr1"),
    (["3 year", "three year", "teen saal", "3 saal", "3 yr return"], "cagr3"),
    (["return kitna", "kitna return", "return kya", "performance kaisi",
      "kaisa perform", "returns", "performance"], "cagr3"),
    (["risk kitna", "kitna risk", "risk kya", "risky hai", "risk rating", "how risky", "what is the risk",
      "what risk does", "risk does it carry", "safe fund", "is this safe", "is it safe", "safe hai", "safe rahega",
      "risk level", "risk of this fund"], "risk"),
    (["fund house", "amc", "company kaunsi", "kis company", "which company", "who manages"], "house"),
    (["category", "type kya", "kis category", "which category", "fund type"], "category"),
]

_ORDINALS: list[tuple[list[str], int]] = [
    (["pehla", "pehle", "pehli", "first one", "first fund", "the first",
      "number one", "sabse upar", "upar wala"], 1),
    (["doosra", "doosre", "dusra", "dusre", "second one", "second fund",
      "the second", "beech wala", "number two"], 2),
    (["teesra", "teesre", "tisra", "tisre", "third one", "third fund",
      "the third", "last wala", "aakhri", "number three"], 3),
]

# Investment horizon also carries risk signal — "10 saal" implies equity tolerance
_HORIZON: list[tuple[list[str], str]] = [
    (["6 mahine", "6 month", "1 saal", "one year", "1 year", "short term",
      "jaldi nikaalna", "kuch mahine"], "low"),
    (["3 saal", "3 year", "three year", "5 saal", "5 year", "five year",
      "medium term"], "moderate"),
    (["10 saal", "10 year", "ten year", "15 saal", "20 saal", "lambe samay",
      "long term", "retirement", "bachche ki padhai", "bacche ki padhai"], "high"),
]


def _parse_horizon(padded: str) -> Optional[str]:
    for phrases, bucket in _HORIZON:
        for ph in phrases:
            if _hit(ph, padded) is not None:
                if ph in ("long term", "lambe samay", "retirement", "10 saal", "10 year", "ten year", "15 saal", "20 saal"):
                    return "long term"
                elif ph in ("short term", "jaldi nikaalna", "6 mahine", "6 month", "1 saal", "one year", "1 year", "kuch mahine"):
                    return "short term"
                elif ph in ("medium term", "3 saal", "3 year", "three year", "5 saal", "5 year", "five year"):
                    return "medium term"
                return ph
    return None

_PRONOUN_REF = ["iska", "iski", "iske", "is fund", "ye fund", "yeh fund",
                "uska", "uske", "that one", "this one", "it ",
                "inme se", "in funds", "isme", "usme", "ye wala", "yeh wala", "wala"]


def _norm_str(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_AMC_ALIASES = {
    "SBI Mutual Fund": ("state bank of india", "sbi"),
    "HDFC Mutual Fund": ("hdfc mutual fund", "hdfc"),
    "ICICI Prudential Mutual Fund": ("icici prudential", "icici pru", "icici"),
    "Axis Mutual Fund": ("axis mutual fund", "axis"),
    "Bank of India Mutual Fund": ("bank of india mutual fund", "bank of india", "boi"),
    "Nippon India Mutual Fund": ("nippon india mutual fund", "nippon india", "nippon", "reliance mutual fund"),
    "PPFAS Mutual Fund": ("parag parikh", "ppfas"),
    "Kotak Mahindra Mutual Fund": ("kotak mahindra", "kotak"),
    "UTI Mutual Fund": ("uti mutual fund", "uti"),
    "Tata Mutual Fund": ("tata mutual fund", "tata"),
    "DSP Mutual Fund": ("dsp mutual fund", "dsp"),
    "Aditya Birla Sun Life Mutual Fund": ("aditya birla sun life", "birla sun life", "absl"),
    "Canara Robeco Mutual Fund": ("canara robeco",),
    "quant Mutual Fund": ("quant mutual fund", "quant"),
    "Mirae Asset Mutual Fund": ("mirae asset", "mirae"),
}


def detect_fund_house(query: str) -> Optional[str]:
    """Return a canonical AMC when the user explicitly names one."""
    normalized = _norm_str(query)
    aliases = sorted(
        ((alias, canonical) for canonical, names in _AMC_ALIASES.items() for alias in names),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, canonical in aliases:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            return canonical
    return None


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9&' ]+", " ", text.lower()).strip() + " "


def _raw_clauses(text: str) -> list[str]:
    """Split on punctuation BEFORE normalisation strips it, then normalise each."""
    parts = [p for p in _CLAUSE_SPLIT.split(text.lower()) if p.strip()]
    return [_words(p) for p in parts] or [_words(text)]


def _span(phrase: str, padded: str) -> Optional[tuple[int, int]]:
    """(start, end) of phrase in padded text, honouring word boundaries."""
    p = phrase.strip()
    pat = " " + p if not p.endswith(" ") else " " + p.rstrip()
    idx = padded.find(pat + " ")
    if idx < 0:
        idx = padded.find(pat)
        if idx < 0 or (idx + len(pat) < len(padded) and padded[idx + len(pat)].isalnum()):
            return None
    return idx + 1, idx + 1 + len(p)


def _hit(phrase: str, padded: str) -> Optional[int]:
    sp = _span(phrase, padded)
    return sp[0] if sp else None


_CLAUSE_SPLIT = re.compile(r"\s*(?:,|;|\blekin\b|\bbut\b|\bbalki\b|\bmagar\b|\bhowever\b)\s*")


# (clause splitting lives in _raw_clauses — it must run before normalisation,
#  otherwise the comma is already gone and every sentence is one clause)


def _negated(clause: str, start: int, end: int) -> bool:
    """
    English negators precede ('not low risk'); Hindi ones follow
    ('low risk nahi chahiye'). Look back a little, forward a little, and never
    outside this clause.
    """
    before = clause[max(0, start - 14): start]
    after = clause[end: end + 18]
    return (any(n in before for n in _NEGATORS)
            or any(n in after for n in ("nahi", "nahin", "mat ", "chhod", "avoid")))


def _score_risk(padded: str, clauses: list[str]) -> tuple[Optional[str], float]:
    scores = {"low": 0.0, "moderate": 0.0, "high": 0.0}

    for clause in clauses:
        for bucket, phrases in _RISK_LEXICON.items():
            for ph in phrases:
                sp = _span(ph, clause)
                if sp is None:
                    continue
                if _negated(clause, sp[0], sp[1]):
                    for other in scores:
                        if other != bucket:
                            scores[other] += 0.4
                    continue
                scores[bucket] += 1.0 + (0.5 if " " in ph else 0.0)

    # horizon contributes a weaker signal than an explicit risk word
    for phrases, bucket in _HORIZON:
        for ph in phrases:
            if _hit(ph, padded) is not None:
                scores[bucket] += 0.7
                break

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return None, 0.0
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    conf = min(0.95, 0.62 + 0.12 * scores[best] + 0.08 * margin)
    return best, conf


_AMOUNT_WORDS = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
    "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10, "bees": 20, "pachas": 50,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "fifty": 50,
}
_SCALE = [("crore", 10_000_000), ("cr", 10_000_000), ("lakh", 100_000), ("lac", 100_000),
          ("hazaar", 1_000), ("hazar", 1_000), ("thousand", 1_000), ("k ", 1_000)]

# Number-shaped things that are NOT money
_NOT_MONEY = re.compile(
    r"\b\d+\s*(?:year|years|yr|saal|sal|month|months|mahine|maheene|%|percent)\b"
)

_YEAR_WORDS = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
    "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12,
    "pandrah": 15, "bees": 20, "pachees": 25,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20,
}

_GOAL_TRIGGERS = [
    "want", "need", "chahiye", "banana hai", "target", "goal",
    "reach", "accumulate", "bana sakein", "how much", "how much sip",
    "kitna invest", "kitni sip", "kitna jama", "bachche", "retirement", "ghar", "house",
    "car", "gadi", "gaadi", "vehicle", "bike", "wedding", "shaadi", "shadi"
]


def _scaled_amount(raw_value: str, unit: str | None) -> Optional[int]:
    """Parse one normalised money token without guessing its role in a sentence."""
    try:
        value = float(raw_value.replace(",", ""))
    except (TypeError, ValueError):
        return None

    unit = (unit or "").casefold()
    if unit.startswith(("crore", "cr")):
        value *= 10_000_000
    elif unit.startswith(("lakh", "lac", "laakh")):
        value *= 100_000
    elif unit.startswith(("hazaar", "hazar", "thousand", "k")):
        value *= 1_000

    amount = int(round(value))
    return amount if 100 <= amount <= 1_000_000_000 else None


def _parse_monthly_sip(text: str) -> Optional[int]:
    """Extract the amount tied to a monthly cadence.

    This deliberately looks at the words immediately around each amount.  A
    goal such as ``10 lakh`` can appear in the same utterance as ``har month
    20,000``; treating the first number as the SIP is the mistake that made
    goal conversations lose the user's actual plan.
    """
    t = text.casefold().strip()
    # Intent-router normalisation removes punctuation, so ₹20,000 reaches this
    # helper as "20 000".  Rejoin Indian-style digit groups before scanning.
    while re.search(r"(?<=\d)\s+(?=\d{3}\b)", t):
        t = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", t)
    money_re = re.compile(
        r"(?<!\d)(?P<value>\d[\d,]*(?:\.\d+)?)"
        r"(?:\s*(?P<unit>crores?|crs?|lakhs?|lacs?|laakhs?|"
        r"hazaars?|hazars?|thousands?|k|rs|rupees?))?"
    )
    before_month = re.compile(
        r"(?:har|per|every)?\s*(?:mahina|monthly)"
        r"(?:\s+(?:ka|ke|ki|mein|me|sip|investment|invest))?\s*$"
    )
    after_month = re.compile(
        r"\s*(?:rs|rupees)?\s*(?:a|per|every|har)?\s*"
        r"(?:mahina|monthly)\b"
    )

    for match in money_re.finditer(t):
        before = t[max(0, match.start() - 32):match.start()]
        after = t[match.end():match.end() + 32]
        if not (before_month.search(before) or after_month.match(after)):
            continue
        amount = _scaled_amount(match.group("value"), match.group("unit"))
        if amount:
            return amount
    return None


def _parse_goal(text: str) -> Optional[dict]:
    # parse_turn already supplies the canonical form, but normalising again
    # keeps this helper correct when it is used directly in a test or script.
    from intent_router import normalize as _ir_normalize
    t = _ir_normalize(text).strip()
    target_amount = None
    m_amt = re.search(r"\b(\d+(?:\.\d+)?)\s*(crores?|crs?|lakhs?|lacs?|laakhs?)\b", t)
    if m_amt:
        target_amount = _scaled_amount(m_amt.group(1), m_amt.group(2))
    else:
        for w, v in _AMOUNT_WORDS.items():
            m_w = re.search(rf"\b{w}\s*(crores?|crs?|lakhs?|lacs?|laakhs?)\b", t)
            if m_w:
                mult = 10_000_000 if m_w.group(1).startswith("cr") else 100_000
                target_amount = v * mult
                break

    target_years = None
    m_yr = re.search(r"\b(\d+)\s*(?:years?|yrs?|saal|sal)\b", t)
    if m_yr:
        target_years = int(m_yr.group(1))
    else:
        for w, y in _YEAR_WORDS.items():
            if re.search(rf"\b{w}\s*(?:years?|yrs?|saal|sal)\b", t):
                target_years = y
                break

    purpose = "Wealth Creation"
    if any(k in t for k in ("house", "ghar", "home", "flat", "property")):
        purpose = "Buying a House"
    elif any(k in t for k in ("retire", "retirement", "pension")):
        purpose = "Retirement Fund"
    elif any(k in t for k in ("child", "children", "kid", "kids", "education", "padhai", "bachhe", "bachho", "beti", "beta", "college", "school")):
        purpose = "Child Education"
    elif any(k in t for k in ("car", "gadi", "gaadi", "vehicle", "bike")):
        purpose = "Buying a Car"
    elif any(k in t for k in ("marriage", "wedding", "shadi", "shaadi", "biyaah")):
        purpose = "Wedding & Marriage"

    monthly_sip = _parse_monthly_sip(t)
    # A named purpose plus a target is a goal even if the user did not use the
    # literal word "goal".  This covers natural requests such as "car दस लाख
    # की है aur har month 20000 invest karunga".
    is_goal = target_amount and (
        target_years
        or monthly_sip
        or purpose != "Wealth Creation"
        or any(trig in t for trig in _GOAL_TRIGGERS)
    )
    if is_goal:
        return {
            "target_amount": target_amount,
            "target_years": target_years,
            "monthly_sip": monthly_sip,
            "purpose": purpose,
        }
    return None


def _parse_what_if(text: str) -> Optional[dict]:
    t = text.lower()
    is_what_if = any(k in t for k in (
        "what if", "agar", "maan lo", "suppose", "step up", "step-up", "stepup",
        "percent", "%", "change to", "increase to", "decrease to", "make it", "what about"
    ))
    if not is_what_if:
        return None

    res = {}
    m_amt = re.search(r"\b(\d+(?:\.\d+)?)\s*(crores?|crs?|lakhs?|lacs?|hazaars?|hazars?|thousands?|k)\b", t)
    if m_amt:
        val = float(m_amt.group(1))
        unit = m_amt.group(2)
        mult = 10_000_000 if unit.startswith("cr") else (100_000 if unit.startswith("la") else 1_000)
        res["amount"] = int(val * mult)
    else:
        for w, v in _AMOUNT_WORDS.items():
            m_w = re.search(rf"\b{w}\s*(crores?|crs?|lakhs?|lacs?|hazaars?|hazars?|thousands?|k)\b", t)
            if m_w:
                unit = m_w.group(1)
                mult = 10_000_000 if unit.startswith("cr") else (100_000 if unit.startswith("la") else 1_000)
                res["amount"] = v * mult
                break
        if "amount" not in res:
            m_bare = re.search(r"\b(\d{4,7})\b", t)
            if m_bare:
                res["amount"] = int(m_bare.group(1))

    m_cagr = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent\b)", t)
    if m_cagr:
        res["cagr"] = float(m_cagr.group(1))

    m_yr = re.search(r"\b(\d+)\s*(?:years?|yrs?|saal|sal)\b", t)
    if m_yr:
        res["years"] = int(m_yr.group(1))

    if "step up" in t or "step-up" in t or "stepup" in t or "top up" in t:
        res["step_up"] = True

    return res if res else None


def _parse_amount(padded: str, raw: str = "") -> Optional[int]:
    # Try raw first — _words() replaces ',' with space so "1,000" → "1 000"
    # and the bare-digit regex can't match "1" (too short). Raw has comma intact.
    for text in ([raw.lower()] if raw else []) + [padded]:
        cleaned = _NOT_MONEY.sub(" ", text)

        # Scale-word match: "5000 rupaye", "5k", "paanch hazaar", "2.5 cr", "1.5 lakh"
        m = re.search(r"\b(\d[\d,]*(?:\.\d+)?)\s*(crores?|crs?|lakhs?|lacs?|hazaars?|hazars?|thousands?|k)\b", cleaned)
        if m:
            unit = m.group(2)
            mult = 10_000_000 if unit.startswith("cr") else (100_000 if unit.startswith("la") else 1_000)
            return int(float(m.group(1).replace(",", "")) * mult)

        for word, val in _AMOUNT_WORDS.items():
            at = _hit(word, " " + cleaned + " ")
            if at is None:
                continue
            tail = cleaned[at: at + 40]
            for token, mult in _SCALE:
                if token in tail:
                    return val * mult

        # Bare digit — strip commas so "1,000" matches
        stripped = cleaned.replace(",", "")
        m = re.search(r"\b(\d{3,7})\b", stripped)
        if m:
            val = int(m.group(1))
            if 100 <= val <= 10_000_000:
                return val
    return None


# ── Session ───────────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str
    turn: int = 0
    epoch: int = 1
    risk: Optional[str] = None            # sticky across turns
    amount: Optional[int] = None          # sticky across turns
    horizon: Optional[str] = None         # sticky across turns (e.g. "long term")
    goal_amount: Optional[int] = None     # target goal corpus (e.g. 15000000)
    goal_years: Optional[int] = None      # target years (e.g. 7)
    goal_monthly_sip: Optional[int] = None  # user-stated monthly contribution
    goal_purpose: Optional[str] = None    # e.g. "Buying a House"
    what_if: Optional[dict] = None        # live what-if parameter overrides
    last_slots: dict = field(default_factory=dict)   # funds on screen right now
    last_list: dict = field(default_factory=dict)    # last multi-fund LIST shown
    focus_slot: Optional[str] = None      # which fund the user is talking about
    user_name: Optional[str] = None       # user's name e.g. 'Aman'
    language: str = "auto"                # "auto", "english", or "hinglish"
    said_templates: list[str] = field(default_factory=list)  # anti-repetition
    created_at: float = field(default_factory=time.time)

    def remember_snapshot(self, snapshot, is_list: bool) -> None:
        """
        A one-fund detail view must NOT wipe the list the user is still
        referring to. Otherwise "doosre wale se compare karo" right after a
        detail turn has no F2 to point at — which is exactly what broke compare.
        """
        self.last_slots = {k: v for k, v in snapshot.funds.items()}
        if is_list and len(snapshot.funds) > 1:
            self.last_list = dict(self.last_slots)
        self.focus_slot = "F1" if "F1" in self.last_slots else None

    def fund_at(self, n: int):
        """Ordinals always resolve against the last LIST, not the detail view."""
        source = self.last_list or self.last_slots
        return source.get(f"F{n}")

    def focused(self):
        return self.last_slots.get(self.focus_slot) if self.focus_slot else None


_SESSIONS: dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = Session(session_id=session_id)
    return _SESSIONS[session_id]


def reset_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


_NAME_STOP_WORDS = frozenset({
    'fund', 'funds', 'sip', 'nav', 'cagr', 'money', 'paisa', 'invest', 'investment',
    'nivesh', 'lumpsum', 'goal', 'large', 'mid', 'small', 'cap', 'debt', 'equity',
    'elss', 'axis', 'hdfc', 'sbi', 'mridul', 'madhur', 'yes', 'no', 'haan', 'nahi',
    'ok', 'okay', 'sure', 'kuch', 'kya', 'why', 'how', 'who', 'nothing', 'hello',
    'hi', 'hey', 'namaste', 'namaskar', 'pranam', 'good', 'morning', 'afternoon',
    'hai', 'hoon', 'hu', 'me', 'mera', 'meri', 'mere', 'batao', 'dikhao', 'karein',
    'karna', 'chahiye', 'kaunsa', 'kaise', 'bhi', 'toh', 'aur', 'and', 'the', 'is'
})

def extract_user_name(q: str, is_first_turn: bool = False) -> tuple[Optional[str], str]:
    q_trimmed = q.strip()
    name = None
    cleaned_query = q_trimmed

    # Pattern A: 'mera naam X hai' / 'my name is X' / 'i am X' / 'this is X'
    m = re.search(
        r'^(?:(?:hello|hi|hey|namaste|pranam)[,\s]+)?(?:mera\s+naam|my\s+name\s+is|my\s+name|i\s+am|i\'m|this\s+is)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)(?:\s+(?:hai|hoon|here))?(?:[,\s.]+(?:aur\s+|and\s+)?(.*))?$',
        q_trimmed,
        re.I
    )
    if m:
        cand = m.group(1).strip()
        rem = (m.group(2) or '').strip()
        words = [w for w in cand.split() if w.lower() not in _NAME_STOP_WORDS]
        if words:
            name = ' '.join(words).title()
            cleaned_query = rem

    # Pattern B: 'main X hoon'
    if not name:
        m2 = re.search(
            r'^(?:(?:hello|hi|hey|namaste|pranam)[,\s]+)?main\s+([a-zA-Z]+)\s+(?:hoon|hu)(?:[,\s.]+(?:aur\s+|and\s+)?(.*))?$',
            q_trimmed,
            re.I
        )
        if m2:
            cand = m2.group(1).strip()
            rem = (m2.group(2) or '').strip()
            if cand.lower() not in _NAME_STOP_WORDS:
                name = cand.title()
                cleaned_query = rem

    # Pattern C: first turn standalone 1-2 words name (e.g. 'Aman' or 'Rahul Sharma')
    if not name and is_first_turn:
        words = q_trimmed.split()
        if 1 <= len(words) <= 2:
            clean = [w for w in words if re.match(r'^[a-zA-Z]+$', w) and w.lower() not in _NAME_STOP_WORDS]
            if len(clean) == len(words):
                name = ' '.join(clean).title()
                cleaned_query = ''

    return name, cleaned_query


# ── The router ────────────────────────────────────────────────────────────────

def parse_turn(query: str, session: Session) -> dict:
    """
    Returns a fully-resolved turn:
      { intent, risk, amount, confidence, target_slots, metric_field,
        mentioned_fund, query, is_followup }

    Never silently defaults to a risk bucket. If nothing scores, intent is
    CLARIFY and the caller asks one question instead of guessing.
    """
    session.turn += 1

    # Check if user is introducing their name
    name_found, name_query_rem = extract_user_name(query, is_first_turn=(session.turn == 1 and session.user_name is None))
    if name_found:
        session.user_name = name_found
        if name_query_rem:
            query = name_query_rem
        else:
            if session.language == "auto":
                effective_lang = detect_language(query, default="hinglish")
            else:
                effective_lang = session.language
            return {
                "query": query,
                "turn": session.turn,
                "intent": GREET,
                "language": effective_lang,
                "confidence": 0.95,
                "target_slots": [],
                "metric_field": None,
                "mentioned_fund": None,
                "is_followup": False,
                "risk_changed": False,
                "risk": session.risk,
                "amount": session.amount,
                "horizon": session.horizon,
            }
    from intent_router import normalize as _ir_norm
    norm_query = _ir_norm(query).strip()
    padded = f" {norm_query} "

    # 0. explicit language switch command
    if any(_hit(s, padded) is not None for s in _LANG_SWITCH_EN):
        session.language = "english"
        return {
            "query": query,
            "turn": session.turn,
            "intent": LANG_SWITCH,
            "language": "english",
            "confidence": 0.99,
            "target_slots": [],
            "metric_field": None,
            "mentioned_fund": None,
            "is_followup": bool(session.last_slots),
            "risk_changed": False,
            "risk": session.risk,
            "amount": session.amount,
            "horizon": session.horizon,
        }

    if any(_hit(s, padded) is not None for s in _LANG_SWITCH_HI):
        session.language = "hinglish"
        return {
            "query": query,
            "turn": session.turn,
            "intent": LANG_SWITCH,
            "language": "hinglish",
            "confidence": 0.99,
            "target_slots": [],
            "metric_field": None,
            "mentioned_fund": None,
            "is_followup": bool(session.last_slots),
            "risk_changed": False,
            "risk": session.risk,
            "amount": session.amount,
            "horizon": session.horizon,
        }

    # Determine effective language for this turn
    if session.language == "auto":
        effective_lang = detect_language(query, default="hinglish")
    else:
        effective_lang = session.language

    out = {
        "query": query,
        "turn": session.turn,
        "intent": CLARIFY,
        "language": effective_lang,
        "risk": None,
        "amount": None,
        "confidence": 0.5,
        "target_slots": [],
        "metric_field": None,
        "mentioned_fund": None,
        "fund_house": detect_fund_house(query),
        "is_followup": bool(session.last_slots),
        "risk_changed": False,
    }

    # 1. out of scope — checked first, highest precedence
    for ph in _OOS:
        if _hit(ph, padded) is not None:
            out.update(intent=OUT_OF_SCOPE, confidence=0.95)
            return out

    # 2. greeting
    if len(padded.split()) <= 4 and any(_hit(g, padded) is not None for g in _GREET):
        out.update(intent=GREET, confidence=0.9)
        return out

    # 3. resolve which fund(s) the user means
    slots: list[str] = []
    named = None
    reference = session.last_list or session.last_slots

    # Check on-screen funds (reference) first
    q_clean = _norm_str(query)
    best_slot = None
    best_len = 0
    best_name = None

    if reference:
        for slot, rec in reference.items():
            fname = getattr(rec, "fund_name", str(rec) if rec else "") or ""
            name_clean = _norm_str(fname)
            name_clean = re.sub(r"\b(direct|regular|plan|growth|option|fund|schemes?)\b", " ", name_clean)
            tokens = [w for w in name_clean.split() if len(w) > 2]
            for size in range(len(tokens), 0, -1):
                for start in range(len(tokens) - size + 1):
                    frag = " ".join(tokens[start:start + size])
                    if len(frag) > 3 and frag in q_clean:
                        if len(frag) > best_len:
                            best_slot = slot
                            best_len = len(frag)
                            best_name = fname
        if best_slot:
            slots.append(best_slot)
            out["mentioned_fund"] = best_name

    if not slots:
        named = universe.find_by_name(query)
        if named:
            out["mentioned_fund"] = named["fund_name"]
            for slot, rec in reference.items():
                if rec and getattr(rec, "fund_name", None) == named["fund_name"]:
                    slots.append(slot)

    for phrases, n in _ORDINALS:
        if any(_hit(p, padded) is not None for p in phrases):
            if session.fund_at(n) is not None:
                slots.append(f"F{n}")

    if not slots and any(_hit(p, padded) is not None for p in _PRONOUN_REF):
        if session.focus_slot:
            slots.append(session.focus_slot)

    slots = list(dict.fromkeys(slots))
    out["target_slots"] = slots

    # 4. read risk + amount + horizon signals — but do NOT commit them yet.
    #    A follow-up like "iska 5 saal ka return kitna hai" contains a horizon
    #    phrase; committing it would silently flip the user's risk profile
    #    mid-conversation. Only discovery turns may change the profile.
    risk, risk_conf = _score_risk(padded, _raw_clauses(query))
    amount = _parse_amount(padded, raw=query)
    horizon = _parse_horizon(padded)

    def commit(is_discovery: bool) -> None:
        if amount:
            session.amount = amount
        if horizon:
            session.horizon = horizon
        if risk and is_discovery:
            out["risk_changed"] = session.risk is not None and risk != session.risk
            session.risk = risk
            if out["risk_changed"]:
                session.epoch += 1       # speculative rollback — discard prefetch
        out["risk"] = session.risk
        out["amount"] = session.amount
        out["horizon"] = session.horizon

    # 4b. goal-based planning (e.g., "1.5 crore in 7 years for house").
    # The canonical text handles mixed Hindi/English quantities consistently.
    goal_info = _parse_goal(norm_query)
    if not goal_info and session.goal_amount:
        followup_sip = _parse_monthly_sip(norm_query)
        asks_for_goal_timing = any(marker in padded for marker in (
            " kitna saal ", " kitne saal ", " how long ", " kab ",
            " reach ", " target ", " goal ", " ho jaye ",
        ))
        if followup_sip and asks_for_goal_timing:
            goal_info = {
                "target_amount": session.goal_amount,
                "target_years": None,
                "monthly_sip": followup_sip,
                "purpose": session.goal_purpose or "Wealth Creation",
            }
    if goal_info:
        commit(False)
        goal_purpose = goal_info["purpose"]
        # A follow-up often repeats the target and SIP but omits "car" or
        # "house".  Keep the named purpose from the plan already in memory.
        if goal_purpose == "Wealth Creation" and session.goal_purpose:
            goal_purpose = session.goal_purpose
        monthly_sip = goal_info.get("monthly_sip")
        if monthly_sip:
            session.amount = monthly_sip
            out["amount"] = monthly_sip
        out.update(
            intent=GOAL,
            confidence=0.96,
            goal_amount=goal_info["target_amount"],
            goal_years=goal_info["target_years"],
            goal_monthly_sip=monthly_sip,
            goal_purpose=goal_purpose,
            target_slots=slots or [session.focus_slot or "F1"]
        )
        session.goal_amount = goal_info["target_amount"]
        session.goal_years = goal_info["target_years"]
        session.goal_monthly_sip = monthly_sip
        session.goal_purpose = goal_purpose
        return out

    # 5. confirm
    if any(_hit(c, padded) is not None for c in _CONFIRM):
        if session.last_slots:
            commit(False)
            out.update(intent=CONFIRM_SIP, confidence=0.93,
                       target_slots=slots or [session.focus_slot or "F1"])
            return out

    # 5b. wealth accumulation & graph explanation (checked first as it is more specific)
    if any(_hit(w, padded) is not None for w in _WEALTH_EXPLAIN) and reference:
        commit(False)
        target = slots or [session.focus_slot or "F1"]
        out.update(intent=GRAPH, confidence=0.95, target_slots=target[:1], explain_wealth=True)
        session.focus_slot = target[0]
        return out

    # 5c. graph / visual performance & SIP projection chart
    if any(_hit(g, padded) is not None for g in _GRAPH) and reference:
        commit(False)
        target = slots or [session.focus_slot or "F1"]
        out.update(intent=GRAPH, confidence=0.94, target_slots=target[:1])
        session.focus_slot = target[0]
        return out

    # 5d. what-if parameter tweaking on active chart
    what_if_info = _parse_what_if(query)
    if what_if_info and reference:
        commit(False)
        target = slots or [session.focus_slot or "F1"]
        if "amount" in what_if_info:
            session.amount = what_if_info["amount"]
            out["amount"] = what_if_info["amount"]
        session.what_if = what_if_info
        out.update(intent=GRAPH, confidence=0.95, target_slots=target[:1], what_if=what_if_info)
        session.focus_slot = target[0]
        return out

    # 6. compare
    if any(_hit(c, padded) is not None for c in _COMPARE) and reference:
        commit(False)
        if len(slots) == 1 and session.focus_slot and session.focus_slot not in slots:
            slots = [session.focus_slot] + slots      # "isse compare karo" + one named
        targets = slots if len(slots) >= 2 else list(reference)[:2]
        out.update(intent=COMPARE, confidence=0.9, target_slots=targets)
        return out

    # 7. specific metric follow-up. Keep every requested field so compound
    # questions such as "risk and minimum SIP?" are answered together.
    matched_metrics = [
        fieldname for phrases, fieldname in _METRIC_FIELDS
        if any(_hit(p, padded) is not None for p in phrases)
    ]
    is_why_question = any(_hit(p, padded) is not None for p in _WHY)
    if matched_metrics and not is_why_question:
        target = slots or ([session.focus_slot] if session.focus_slot else [])
        if target:
            commit(False)
            out.update(intent=METRIC, metric_field=matched_metrics[0], metric_fields=matched_metrics,
                       confidence=0.91, target_slots=target)
            session.focus_slot = target[0]
            return out

    # 7b–10: delegate to IntentRouter — compiled-regex lexicons, Devanagari-aware,
    # fully unit-tested. We build a lightweight IR session that mirrors the
    # dialogue session's screen state, run the router, then copy results back.
    ir_session = _IRSession(
        visible_slots=tuple((session.last_list or session.last_slots).keys()),
        focus_slot=session.focus_slot,
        risk=_IRRisk(session.risk) if session.risk in ("low", "moderate", "high") else None,
        amount=session.amount,
        horizon_months=float(session.horizon.split()[0]) * 12
            if session.horizon and session.horizon[0].isdigit() else None,
        turn=session.turn,
    )
    routing = _INTENT_ROUTER.route(query, ir_session)

    # Translate IR intent → dialogue string constants
    _IR_INTENT_MAP = {
        "FUND_DETAIL": FUND_DETAIL,
        "DISCOVER":    DISCOVER,
        "CLARIFY":     CLARIFY,
    }
    ir_intent = _IR_INTENT_MAP.get(routing.intent.value, CLARIFY)
    routing_confidence = routing.confidence
    if ir_intent == CLARIFY and out.get("fund_house") and not routing.mentioned_fund:
        # A named AMC is a valid discovery constraint even when the query has
        # no separate appetite phrase (e.g. "SBI mutual funds").
        ir_intent = DISCOVER
        routing_confidence = max(routing_confidence, 0.85)

    # Commit profile signals discovered by the router
    if routing.risk is not None:
        session.risk = routing.risk.value
    if routing.amount is not None:
        session.amount = routing.amount
    if routing.horizon_months is not None:
        months = routing.horizon_months
        if months >= 60:
            session.horizon = "long term"
        elif months >= 24:
            session.horizon = f"{int(months // 12)} years"
        else:
            session.horizon = f"{int(months)} months"
    if ir_intent == DISCOVER:
        session.epoch += 1  # new query → fresh fund list

    # Copy focus slot update
    if ir_session.focus_slot and not routing.results_will_change:
        session.focus_slot = ir_session.focus_slot

    target_slots = list(routing.target_slots)
    mentioned_fund = routing.mentioned_fund or (out.get("mentioned_fund"))

    out.update(
        intent=ir_intent,
        confidence=routing_confidence,
        target_slots=target_slots,
        mentioned_fund=mentioned_fund,
        why_query=routing.why_query,
        best_pick=routing.best_pick,
        risk=routing.risk.value if routing.risk else session.risk,
        amount=routing.amount,
        horizon=session.horizon,
    )
    return out
