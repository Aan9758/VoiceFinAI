"""Deterministic intent router for a Hinglish/English mutual-fund assistant.

Design notes
------------
* **Deterministic cascade.** Rules are evaluated in a fixed priority order and the
  first match wins. This is intentional: a rule engine you can read top-to-bottom
  is debuggable at 2am in a way a classifier is not.
* **Every rule is a pure function** of (utterance features, session) -> Routing | None.
  That makes each tier independently unit-testable without driving the whole ladder.
* **Session mutation happens in exactly one place** (`IntentRouter.route`), never
  inside a rule. Rules return what they *want*; the router applies it.
* **Hinglish is normalised before matching**, not matched twice. Devanagari is
  transliterated to Latin, then spelling variants are folded to a canonical form,
  so `कौन सा`, `konsa`, `kaun-sa` and `kaunsa` all hit the same lexicon entry.
* **No magic numbers.** Confidences live in `Confidence` so they can be tuned,
  logged, and diffed.

Thread safety: `IntentRouter` is stateless and safe to share. `Session` is not —
own one per conversation and don't share it across threads.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "Intent",
    "Risk",
    "Session",
    "Routing",
    "IntentRouter",
    "normalize",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enums and tunables
# --------------------------------------------------------------------------- #


class Intent(str, Enum):
    """Terminal intents this router can emit.

    Extend here rather than returning bare strings, so the set stays closed and
    downstream handlers can exhaustively match on it.
    """

    FUND_DETAIL = "FUND_DETAIL"
    DISCOVER = "DISCOVER"
    CLARIFY = "CLARIFY"


class Risk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Confidence:
    """Hand-tuned confidences, named so they are greppable and tunable.

    These are *calibration targets*, not probabilities. If you start logging
    outcomes, treat them as priors to re-fit rather than as ground truth.
    """

    WHY_FOLLOWUP = 0.93
    FUND_DETAIL_SLOTTED = 0.92
    BEST_PICK = 0.92
    FUND_NAMED_OFFSCREEN = 0.88
    DISCOVER_SOFT = 0.80
    CLARIFY = 0.45


#: Utterances at or below this token count, when a slot is on screen, are treated
#: as terse detail requests ("F2?", "ye wala", "second one").
TERSE_UTTERANCE_TOKENS = 6

#: Default slot to fall back on when the user references the screen but the
#: session has no focus yet.
DEFAULT_FOCUS_SLOT = "F1"


# --------------------------------------------------------------------------- #
# Normalisation: Devanagari -> Latin, then variant folding
# --------------------------------------------------------------------------- #

_DEVA_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}

_DEVA_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}

_DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # Nukta forms, common in Hindi/Urdu vocabulary.
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}

_DEVA_SIGNS = {"ं": "n", "ँ": "n", "ः": "h", "़": ""}
_DEVA_VIRAMA = "्"
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


@dataclass
class _Syllable:
    """One Devanagari akshara flattened to Latin: onset + vowel + coda."""

    onset: str = ""
    vowel: str = ""
    coda: str = ""
    inherent: bool = False  # vowel is the implicit schwa, not a written matra

    def render(self) -> str:
        return self.onset + self.vowel + self.coda


def _syllabify(word: str) -> list[_Syllable]:
    """Break one Devanagari word into Latin syllables.

    Virama-joined consonants accumulate into the next syllable's onset, so
    `र्न` becomes a single `rn` onset rather than two syllables.
    """
    units: list[_Syllable] = []
    pending_onset = ""
    i, n = 0, len(word)

    while i < n:
        ch = word[i]
        pair = ch + (word[i + 1] if i + 1 < n else "")

        if pair in _DEVA_CONSONANTS:
            base, consumed = _DEVA_CONSONANTS[pair], 2
        elif ch in _DEVA_CONSONANTS:
            base, consumed = _DEVA_CONSONANTS[ch], 1
        else:
            base, consumed = None, 0

        if base is not None:
            i += consumed
            follower = word[i] if i < n else ""
            if follower == _DEVA_VIRAMA:
                pending_onset += base  # dead consonant; joins the next syllable
                i += 1
            elif follower in _DEVA_MATRAS:
                units.append(_Syllable(pending_onset + base, _DEVA_MATRAS[follower]))
                pending_onset = ""
                i += 1
            else:
                units.append(_Syllable(pending_onset + base, "a", inherent=True))
                pending_onset = ""
            continue

        if ch in _DEVA_VOWELS:
            units.append(_Syllable(pending_onset, _DEVA_VOWELS[ch]))
            pending_onset = ""
        elif ch in _DEVA_SIGNS:
            if units:
                units[-1].coda += _DEVA_SIGNS[ch]
            else:
                pending_onset += _DEVA_SIGNS[ch]
        elif ch in _DEVA_MATRAS:
            if units and not units[-1].vowel:
                units[-1].vowel = _DEVA_MATRAS[ch]
            else:
                units.append(_Syllable(pending_onset, _DEVA_MATRAS[ch]))
                pending_onset = ""
        else:
            units.append(_Syllable(pending_onset + ch))
            pending_onset = ""
        i += 1

    if pending_onset:
        units.append(_Syllable(pending_onset))
    return units


def _delete_schwa(units: list[_Syllable]) -> None:
    """Apply Hindi schwa deletion in place.

    Two rules, both necessary for lexicon hits:

    1. The word-final inherent vowel is silent, so `फंड` is `phand`, not `phanda`.
    2. If rule 1 did *not* apply (the word already ends in a written vowel), the
       penultimate inherent vowel drops instead, so `कितना` is `kitnaa`, not
       `kitanaa`.

    Rule 2 is gated on rule 1 not firing because applying both over-deletes:
    `रिटर्न` would become `ritrn` instead of `ritarn`.
    """
    if len(units) < 2:
        return

    last = units[-1]
    if last.inherent and last.onset:
        last.vowel = ""
        last.inherent = False
        return

    if len(units) >= 3:
        penult = units[-2]
        if penult.inherent and penult.onset and units[0] is not penult:
            penult.vowel = ""
            penult.inherent = False


def _transliterate_devanagari(text: str) -> str:
    """Phonetic Devanagari -> Latin transliteration.

    Deliberately lossy and not reversible. The only goal is to land Devanagari
    input in the same string space as the Hinglish lexicons, so that `कौन सा फंड`
    and `kaunsa fund` reach the same rules.
    """
    if not any("\u0900" <= ch <= "\u097F" for ch in text):
        return text

    text = text.translate(_DEVA_DIGITS).replace("।", " ").replace("॥", " ")
    out: list[str] = []
    for word in re.split(r"(\s+)", text):
        if not word or word.isspace():
            out.append(word)
            continue
        units = _syllabify(word)
        _delete_schwa(units)
        out.append("".join(u.render() for u in units))
    return "".join(out)


#: Hinglish spelling is not standardised, so fold the common variants to one
#: canonical token. Keys are what users type; values are what lexicons contain.
_VARIANTS: Mapping[str, str] = {
    # interrogatives
    "konsa": "kaunsa", "kaunsaa": "kaunsa", "kounsa": "kaunsa", "kon": "kaun",
    "kaunsi": "kaunsa", "konsi": "kaunsa", "kaunse": "kaunsa", "konse": "kaunsa",
    "kyu": "kyun", "kyo": "kyun", "kyun": "kyun", "kyonki": "kyunki",
    "kaise": "kaise", "kese": "kaise", "kaisa": "kaise", "kesa": "kaise",
    # verbs / modals
    "chahiye": "chahiye", "chaiye": "chahiye", "chahie": "chahiye",
    "lu": "lena", "lun": "lena", "loon": "lena", "lena": "lena", "le": "lena",
    "karu": "karna", "karun": "karna", "karoon": "karna", "kro": "karna",
    "batao": "batao", "bata": "batao", "btao": "batao", "bataiye": "batao",
    "samjhao": "samjhao", "smjhao": "samjhao", "samjha": "samjhao",
    # money / horizon
    # All money words fold to "rs" — the token `_AMOUNT_RE` looks for. Folding
    # them to "paisa" instead would silently break every amount parse.
    "paise": "rs", "paisa": "rs", "rupaye": "rs", "rupee": "rs",
    "rupees": "rs", "rs": "rs", "inr": "rs", "rupaya": "rs",
    "hazar": "hazaar", "hajar": "hazaar", "hazaar": "hazaar",
    "lac": "lakh", "lakhs": "lakh", "lacs": "lakh",
    "karod": "crore", "crores": "crore", "cr": "crore",
    "sal": "saal", "saal": "saal", "varsh": "saal", "years": "saal",
    "year": "saal", "yr": "saal", "yrs": "saal",
    "mahine": "mahina", "mahina": "mahina", "months": "mahina",
    "month": "mahina", "maheene": "mahina",
    # risk
    "surakshit": "safe", "safety": "safe", "sureksha": "safe",
    "jokhim": "risk", "jokhm": "risk", "risky": "risk",
    # deixis
    "isme": "ismein", "ismei": "ismein", "inme": "inmein", "inmei": "inmein",
    "yeh": "ye", "yh": "ye", "woh": "wo", "vo": "wo",
    "upar": "upar", "uper": "upar",
    # domain
    "fnd": "fund", "funds": "fund", "skim": "scheme", "schemes": "scheme",
    "sip": "sip", "mf": "fund", "mutual": "mutual",
    # Forms the Devanagari transliterator emits, folded back to Latin spellings.
    "phand": "fund", "fand": "fund", "saa": "sa", "myuchual": "mutual",
    "nivesh": "invest", "paisaa": "rs", "rupayaa": "rs",
    "achchhaa": "acha", "achchha": "acha", "accha": "acha", "achha": "acha",
    "kitnaa": "kitna", "kitne": "kitna", "ritarn": "return",
    "chaahie": "chahiye", "chaahiye": "chahiye", "inmen": "inmein",
    "ismen": "ismein", "kyon": "kyun", "laakh": "lakh", "hajaar": "hazaar",
    "jokhim": "risk", "surkshit": "safe", "behtr": "behtar",
    "namste": "namaste", "namaskar": "namaste", "pranam": "namaste",
    "pahlaa": "pehla", "vaalaa": "wala", "dikhaao": "dikhao", "dikhaie": "dikhao",
    "graaph": "graph", "chaart": "chart", "shuroo": "shuru",
    "esaaaeepee": "sip", "saaipee": "sip", "esip": "sip",
    "ha": "haan", "haa": "haan", "hn": "haan", "han": "haan", "h": "haan",
}

#: Number words, folded to digits only when a unit follows, so English "do" in
#: "do this" is never mistaken for the Hindi two.
_NUMBER_WORDS: Mapping[str, str] = {
    "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4",
    "paanch": "5", "panch": "5", "chhe": "6", "chah": "6", "saat": "7",
    "aath": "8", "nau": "9", "das": "10",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

_NUMBER_WORD_RE = re.compile(
    rf"(?<!\w)(?P<word>{'|'.join(_NUMBER_WORDS)})\s+"
    r"(?P<unit>lakh|crore|hazaar|k|saal|mahina|din|hafta)(?!\w)"
)

_PUNCT_RE = re.compile(r"[^\w\s%₹.]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(raw: str) -> str:
    """Return a padded, canonical form of `raw` suitable for lexicon matching.

    Padding with single spaces lets the matchers use cheap word-boundary logic
    without special-casing the start and end of the string.
    """
    if not raw:
        return " "
    text = unicodedata.normalize("NFKC", raw)
    text = _transliterate_devanagari(text)
    text = text.lower()
    text = text.replace("₹", " rs ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    tokens = [_VARIANTS.get(tok, tok) for tok in text.split(" ") if tok]
    folded = " ".join(tokens)
    folded = _NUMBER_WORD_RE.sub(
        lambda m: f"{_NUMBER_WORDS[m.group('word')]} {m.group('unit')}", folded
    )
    return " " + folded + " "


# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #


class Lexicon:
    """A set of phrases compiled into one alternation regex.

    One compiled pattern per lexicon beats N substring scans, and the
    longest-first ordering means "kaunsa lena" wins over "lena" when both match.
    """

    __slots__ = ("name", "_pattern", "_phrases")

    def __init__(self, name: str, phrases: Iterable[str]) -> None:
        self.name = name
        self._phrases = tuple(sorted({normalize(p).strip() for p in phrases}, key=len, reverse=True))
        if not self._phrases:
            raise ValueError(f"lexicon {name!r} is empty")
        alternation = "|".join(re.escape(p) for p in self._phrases)
        self._pattern = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")

    def hit(self, padded: str) -> str | None:
        """Return the matched phrase, or None. Mirrors the old `_hit` contract."""
        m = self._pattern.search(padded)
        return m.group(0) if m else None

    def __contains__(self, padded: str) -> bool:
        return self._pattern.search(padded) is not None


WHY = Lexicon("why", [
    "why", "why this", "why did you", "why these", "reason", "rationale",
    "on what basis", "how come", "justify", "explain why",
    "kyun", "kyunki", "kyun ye", "kyun isko", "kya reason", "reason kya",
    "kis basis pe", "kis aadhar par", "iska reason", "yahi kyun",
    "why did you pick", "how did you pick", "what makes", "what criteria",
    "why have you", "why was this", "why were these",
])

DETAIL = Lexicon("detail", [
    "detail", "details", "tell me about", "more about", "info", "information",
    "breakdown", "holdings", "returns", "performance", "expense ratio",
    "nav", "aum", "past performance", "track record", "risk level",
    "what is", "show me", "overview",
    "batao", "samjhao", "iske bare mein", "ke bare mein", "jankari",
    "puri jankari", "kya hai", "kaise hai", "return kitna", "kitna return",
])

ADVISE = Lexicon("advise", [
    "which one", "which is best", "best for me", "should i", "recommend",
    "suggest", "suitable", "right for me", "good for me", "worth it",
    "advice", "advise", "pick", "choose", "go with",
    "kaunsa lena", "kaunsa", "kaunsa acha", "konsa behtar", "mujhe kya",
    "kaun sa", "kaun sa lena", "kaun sa acha", "kaun sa behtar", "acha kaun sa",
    "kya lena", "sahi rahega", "theek rahega", "faayda", "fayda",
    "acha rahega", "behtar", "suggest karo", "salah",
])

REFERENCE = Lexicon("reference", [
    "these", "this", "that", "them", "above", "shown", "listed", "the list",
    "on screen", "first one", "second one", "third one", "last one",
    "out of these", "among these", "from these",
    "ismein", "inmein", "inmein se", "ismein se", "ye", "ye wala", "wo wala",
    "upar", "upar wale", "dikhaye", "dikha rahe", "list mein", "in sab mein",
    # Users space these out as often as they run them together.
    "in me", "in me se", "is me", "is me se", "in sab", "in dono", "in teeno",
])

RISK_HIGH = Lexicon("risk_high", [
    "high risk", "aggressive", "risk le sakta", "risk le sakte", "risky",
    "high return", "maximum return", "zyada return", "jokhim", "growth",
    "small cap", "smallcap", "bold", "risk hai to chalega",
])

RISK_LOW = Lexicon("risk_low", [
    "low risk", "safe", "no risk", "without risk", "conservative", "capital protection",
    "stable", "guaranteed", "fd jaisa", "fd ke jaisa", "bina risk", "risk nahi",
    "nuksan nahi", "debt fund", "liquid fund", "secure",
])

RISK_MODERATE = Lexicon("risk_moderate", [
    "moderate", "balanced", "medium risk", "thoda risk", "beech ka",
    "hybrid", "average risk", "neither", "mid cap", "midcap",
])

HORIZON_LONG = Lexicon("horizon_long", [
    "long term", "long run", "retirement", "wealth creation",
    "lambe samay", "lamba", "kaafi saal", "bahut saal", "future ke liye",
])

HORIZON_SHORT = Lexicon("horizon_short", [
    "short term", "quick", "soon", "emergency", "parking",
    "jaldi", "kam samay", "thode din", "turant",
])

#: Slot references: "F2", "second", "dusra", "option 3", "2nd".
_SLOT_RE = re.compile(
    r"(?<!\w)(?:"
    r"f\s?(?P<fnum>[1-9]\d?)"
    r"|(?:option|number|no|slot)\s+(?P<onum>[1-9]\d?)"
    r"|(?P<ord_en>first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)"
    r"|(?P<ord_hi>pehla|pehle|pahla|dusra|dusre|doosra|teesra|tisra|chautha|paanchva|panchva)"
    r")(?!\w)"
)

_ORDINALS: Mapping[str, int] = {
    "first": 1, "1st": 1, "pehla": 1, "pehle": 1, "pahla": 1,
    "second": 2, "2nd": 2, "dusra": 2, "dusre": 2, "doosra": 2,
    "third": 3, "3rd": 3, "teesra": 3, "tisra": 3,
    "fourth": 4, "4th": 4, "chautha": 4,
    "fifth": 5, "5th": 5, "paanchva": 5, "panchva": 5,
}

#: Amounts: "50k", "2 lakh", "rs 5000", "1.5 cr".
_AMOUNT_RE = re.compile(
    r"(?<!\w)(?:rs\s*)?(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>k|hazaar|thousand|lakh|crore)?(?!\w)"
)

_AMOUNT_MULTIPLIER: Mapping[str, int] = {
    "k": 1_000, "hazaar": 1_000, "thousand": 1_000,
    "lakh": 100_000, "crore": 10_000_000,
}

#: Horizon: "3 saal", "18 mahina".
_HORIZON_RE = re.compile(r"(?<!\w)(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>saal|mahina|din|hafta|week)(?!\w)")

_HORIZON_MONTHS: Mapping[str, float] = {
    "saal": 12.0, "mahina": 1.0, "din": 1 / 30.0, "hafta": 0.25, "week": 0.25,
}

#: Below this, an amount is almost certainly a slot index or a year, not money.
MIN_PLAUSIBLE_AMOUNT = 500


# --------------------------------------------------------------------------- #
# Session and result types
# --------------------------------------------------------------------------- #


@dataclass
class Session:
    """Per-conversation state. Mutated only by `IntentRouter.route`."""

    #: Slots currently rendered on screen, e.g. ("F1", "F2", "F3").
    visible_slots: tuple[str, ...] = ()
    #: The slot the user is currently talking about.
    focus_slot: str | None = None
    #: Accumulated profile signals.
    risk: Risk | None = None
    amount: int | None = None
    horizon_months: float | None = None
    last_intent: Intent | None = None
    turn: int = 0

    def commit(self, results_will_change: bool) -> None:
        """Settle screen state for the turn.

        `results_will_change=True` means the handler is about to repaint the fund
        list, so any slot focus is stale and must be dropped. This is the
        behaviour the original `commit(True/False)` call encoded.
        """
        if results_will_change:
            self.visible_slots = ()
            self.focus_slot = None


@dataclass(frozen=True)
class Features:
    """Everything extracted from one utterance. Immutable; rules only read it."""

    raw: str
    padded: str
    tokens: tuple[str, ...]
    slots: tuple[str, ...]
    named_funds: tuple[str, ...]
    onscreen_named: tuple[str, ...]
    #: A deictic word was actually said ("inmein se", "these", "upar wale").
    explicit_reference: bool
    #: Explicit reference, or the user is plainly talking about what's on screen
    #: because funds are rendered and they named no other target. This is what
    #: makes bare "which will be the best for me" route to the visible list
    #: instead of falling through to CLARIFY.
    has_reference: bool
    risk: Risk | None
    risk_confidence: float
    amount: int | None
    horizon_months: float | None

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def has_profile_signal(self) -> bool:
        return self.risk is not None or self.amount is not None or self.horizon_months is not None


@dataclass(frozen=True)
class Routing:
    """What the router decided. Safe to log wholesale — carries no PII beyond text."""

    intent: Intent
    confidence: float
    target_slots: tuple[str, ...] = ()
    why_query: bool = False
    best_pick: bool = False
    mentioned_fund: str | None = None
    risk: Risk | None = None
    amount: int | None = None
    horizon_months: float | None = None
    #: Name of the rule that fired. Log this; it is the single most useful field
    #: when a route surprises you in production.
    rule: str = ""
    #: Whether the handler is expected to repaint the fund list.
    results_will_change: bool = False

    def as_log_fields(self) -> dict[str, object]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "rule": self.rule,
            "slots": list(self.target_slots),
            "why": self.why_query,
            "best_pick": self.best_pick,
        }


Rule = Callable[[Features, Session], Routing | None]


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def _extract_slots(padded: str, visible: Sequence[str]) -> tuple[str, ...]:
    """Resolve slot references to canonical slot ids, preserving mention order.

    Ordinals are resolved against `visible` so "dusra" means the second thing the
    user can actually see, not literally "F2".
    """
    found: list[str] = []
    for m in _SLOT_RE.finditer(padded):
        slot: str | None = None
        if m.group("fnum"):
            slot = f"F{int(m.group('fnum'))}"
        elif m.group("onum"):
            idx = int(m.group("onum"))
            slot = visible[idx - 1] if 0 < idx <= len(visible) else f"F{idx}"
        else:
            word = m.group("ord_en") or m.group("ord_hi")
            idx = _ORDINALS.get(word or "", 0)
            if idx:
                slot = visible[idx - 1] if idx <= len(visible) else f"F{idx}"
        if slot and slot not in found:
            found.append(slot)
    return tuple(found)


def _extract_amount(padded: str) -> int | None:
    """Largest plausible money figure in the utterance.

    Largest, not first, because "invest 5000 in fund 2" should read 5000 and not 2.
    Bare numbers below `MIN_PLAUSIBLE_AMOUNT` are ignored as slot indices.
    """
    best: int | None = None
    for m in _AMOUNT_RE.finditer(padded):
        unit = m.group("unit")
        value = float(m.group("num")) * _AMOUNT_MULTIPLIER.get(unit or "", 1)
        amount = int(value)
        if unit is None and amount < MIN_PLAUSIBLE_AMOUNT:
            continue
        if best is None or amount > best:
            best = amount
    return best


def _extract_horizon(padded: str) -> float | None:
    for m in _HORIZON_RE.finditer(padded):
        return float(m.group("num")) * _HORIZON_MONTHS[m.group("unit")]
    if HORIZON_LONG.hit(padded):
        return 60.0
    if HORIZON_SHORT.hit(padded):
        return 6.0
    return None


def _extract_risk(padded: str) -> tuple[Risk | None, float]:
    """Risk plus a confidence.

    Ordered low -> high -> moderate: "no risk" and "risk nahi" must not be read as
    high-risk just because they contain the token `risk`.
    """
    if RISK_LOW.hit(padded):
        return Risk.LOW, 0.90
    if RISK_HIGH.hit(padded):
        return Risk.HIGH, 0.88
    if RISK_MODERATE.hit(padded):
        return Risk.MODERATE, 0.82
    return None, 0.0


# --------------------------------------------------------------------------- #
# Rules — tiers 7b through 10 of the original cascade
# --------------------------------------------------------------------------- #


def rule_why_followup(f: Features, s: Session) -> Routing | None:
    """7b — "why this one?" about something already on screen."""
    if not ((f.has_reference or f.slots or f.onscreen_named or f.named_funds) and WHY.hit(f.padded)):
        return None
    plural = any(term in f.padded for term in (" these ", " these funds ", " these options ", " inmein ", " in sab "))
    if plural and f.has_reference and not f.slots:
        targets = s.visible_slots[:3]
    else:
        target = f.slots[0] if f.slots else (s.focus_slot or DEFAULT_FOCUS_SLOT)
        targets = (target,)
    return Routing(
        intent=Intent.FUND_DETAIL,
        confidence=Confidence.WHY_FOLLOWUP,
        target_slots=targets,
        why_query=True,
        rule="why_followup",
    )


def rule_detail_on_slot(f: Features, s: Session) -> Routing | None:
    """8 — detail or advice about a specific, identified fund."""
    if not f.slots:
        if not f.has_profile_signal and (f.has_reference or bool(s.focus_slot)) and DETAIL.hit(f.padded) is not None:
            target = s.focus_slot or (s.visible_slots[0] if s.visible_slots else DEFAULT_FOCUS_SLOT)
            return Routing(
                intent=Intent.FUND_DETAIL,
                confidence=Confidence.FUND_DETAIL_SLOTTED,
                target_slots=(target,),
                rule="detail_on_slot",
            )
        return None
    triggered = (
        DETAIL.hit(f.padded) is not None
        or ADVISE.hit(f.padded) is not None
        or f.token_count <= TERSE_UTTERANCE_TOKENS
        or bool(f.onscreen_named)
    )
    if not triggered:
        return None
    return Routing(
        intent=Intent.FUND_DETAIL,
        confidence=Confidence.FUND_DETAIL_SLOTTED,
        target_slots=(f.slots[0],),
        mentioned_fund=f.onscreen_named[0] if f.onscreen_named else None,
        rule="detail_on_slot",
    )


def rule_best_pick(f: Features, s: Session) -> Routing | None:
    """8b — "inmein se kaunsa lu" / "which is best for me" over on-screen funds."""
    if not (f.has_reference and ADVISE.hit(f.padded)):
        return None
    target = s.focus_slot or (s.visible_slots[0] if s.visible_slots else DEFAULT_FOCUS_SLOT)
    return Routing(
        intent=Intent.FUND_DETAIL,
        confidence=Confidence.BEST_PICK,
        target_slots=(target,),
        best_pick=True,
        rule="best_pick",
    )


def rule_named_offscreen(f: Features, s: Session) -> Routing | None:
    """A fund named by name that isn't currently rendered. Still a detail request."""
    if f.slots or not f.named_funds:
        return None
    return Routing(
        intent=Intent.FUND_DETAIL,
        confidence=Confidence.FUND_NAMED_OFFSCREEN,
        target_slots=(),
        mentioned_fund=f.named_funds[0],
        rule="named_offscreen",
    )


def rule_discover(f: Features, s: Session) -> Routing | None:
    """9 — enough profile signal to go and fetch a fresh list."""
    if not f.has_profile_signal:
        return None
    confidence = max(
        f.risk_confidence,
        Confidence.DISCOVER_SOFT if (f.amount or f.horizon_months) else 0.0,
    )
    return Routing(
        intent=Intent.DISCOVER,
        confidence=confidence,
        risk=f.risk,
        amount=f.amount,
        horizon_months=f.horizon_months,
        rule="discover",
        results_will_change=True,
    )


def rule_clarify(f: Features, s: Session) -> Routing | None:
    """10 — nothing scored. Ask, don't guess."""
    return Routing(intent=Intent.CLARIFY, confidence=Confidence.CLARIFY, rule="clarify")


#: Evaluated in order; first non-None wins.
#:
#: Tiers 1-7 from the original cascade (greeting, reset, compare, buy, portfolio,
#: off-topic, ...) are not reproduced here because they weren't in the excerpt.
#: Insert them above `rule_why_followup` at their original priority — the contract
#: is `(Features, Session) -> Routing | None` and nothing else needs to change.
DEFAULT_RULES: tuple[Rule, ...] = (
    rule_why_followup,
    rule_detail_on_slot,
    rule_best_pick,
    rule_named_offscreen,
    rule_discover,
    rule_clarify,  # terminal; must stay last
)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


class IntentRouter:
    """Stateless router. Construct once, share freely.

    Parameters
    ----------
    fund_catalog:
        Maps a canonical fund id/name to the aliases users actually type. Used to
        detect `named_funds`. Pass your live catalog; aliases are normalised at
        construction time so lookup stays cheap.
    rules:
        Override to insert your tier 1-7 rules or to test a subset in isolation.
    """

    def __init__(
        self,
        fund_catalog: Mapping[str, Sequence[str]] | None = None,
        rules: Sequence[Rule] = DEFAULT_RULES,
    ) -> None:
        if not rules:
            raise ValueError("rules must not be empty")
        self._rules = tuple(rules)
        self._alias_to_fund: dict[str, str] = {}
        for canonical, aliases in (fund_catalog or {}).items():
            for alias in (canonical, *aliases):
                key = normalize(alias).strip()
                if key:
                    self._alias_to_fund[key] = canonical
        self._alias_pattern = self._compile_aliases()

    def _compile_aliases(self) -> re.Pattern[str] | None:
        if not self._alias_to_fund:
            return None
        alternation = "|".join(
            re.escape(a) for a in sorted(self._alias_to_fund, key=len, reverse=True)
        )
        return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")

    # -- feature extraction ------------------------------------------------- #

    def extract(self, utterance: str, session: Session) -> Features:
        padded = normalize(utterance)
        tokens = tuple(t for t in padded.split(" ") if t)
        slots = _extract_slots(padded, session.visible_slots)
        risk, risk_conf = _extract_risk(padded)

        named: list[str] = []
        if self._alias_pattern is not None:
            for m in self._alias_pattern.finditer(padded):
                canonical = self._alias_to_fund[m.group(0)]
                if canonical not in named:
                    named.append(canonical)

        onscreen = tuple(n for n in named if n in session.visible_slots)

        explicit_ref = REFERENCE.hit(padded) is not None
        implicit_ref = bool(session.visible_slots) and not slots and not named

        return Features(
            raw=utterance,
            padded=padded,
            tokens=tokens,
            slots=slots,
            named_funds=tuple(named),
            onscreen_named=onscreen,
            explicit_reference=explicit_ref,
            has_reference=explicit_ref or implicit_ref,
            risk=risk,
            risk_confidence=risk_conf,
            amount=_extract_amount(padded),
            horizon_months=_extract_horizon(padded),
        )

    # -- public API ---------------------------------------------------------- #

    def route(self, utterance: str, session: Session) -> Routing:
        """Classify one turn and apply its effects to `session`.

        This is the only function that mutates session state. Rules stay pure so
        they can be tested and reordered without side effects.
        """
        features = self.extract(utterance, session)

        routing = None
        for rule in self._rules:
            routing = rule(features, session)
            if routing is not None:
                break
        if routing is None:  # defensive: a custom rule list dropped the terminal rule
            routing = Routing(
                intent=Intent.CLARIFY, confidence=Confidence.CLARIFY, rule="fallthrough"
            )

        self._apply(routing, features, session)
        log.info("intent_routed", extra=routing.as_log_fields())
        return routing

    @staticmethod
    def _apply(routing: Routing, features: Features, session: Session) -> None:
        session.turn += 1
        session.last_intent = routing.intent
        session.commit(routing.results_will_change)

        if routing.target_slots and not routing.results_will_change:
            session.focus_slot = routing.target_slots[0]

        # Profile signals accumulate across turns — a later turn that only
        # mentions an amount should not wipe an earlier risk answer.
        if features.risk is not None:
            session.risk = features.risk
        if features.amount is not None:
            session.amount = features.amount
        if features.horizon_months is not None:
            session.horizon_months = features.horizon_months


# --------------------------------------------------------------------------- #
# Backwards-compatible shim for the original call site
# --------------------------------------------------------------------------- #


def _hit(lexicon: Lexicon, padded: str) -> str | None:
    """Deprecated. Use `lexicon.hit(padded)`. Kept so old call sites still import."""
    return lexicon.hit(padded)
