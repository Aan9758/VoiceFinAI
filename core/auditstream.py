"""
core/auditstream.py — AuditStream Validator
============================================
The verification firewall. Validates generated template text against
the snapshot BEFORE rendering to voice or screen.

AuditStream is a VALIDATOR, not a verifier:
- Every placeholder resolved  ✓
- Every index in range         ✓
- No bare digit in raw template (contract violation → hard fail)
- All rendered values sourced from snapshot (dict lookup, no network)

Three validation rules from §5.1:
  Rule 1 — Entity existence: fund slot must exist in snapshot
  Rule 2 — Figure accuracy: all numbers come from snapshot (never model weights)
  Rule 3 — Confidence gating: handled upstream (not in this module)

Audit log format (§5.2):
  snapshot_hash, pre_render_template, rendered_output, prev_hash → hash chain
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal

from core.snapshot import Snapshot


# ── Regex patterns ────────────────────────────────────────────────────────────

# Matches any placeholder: {F1}, {F1.nav}, {U.amount}, {U.goal_amount}, etc.
_PLACEHOLDER_RE = re.compile(r"\{([A-Z]\d*(?:\.[a-z0-9_]+)?|U\.[a-z0-9_]+)\}")

# Detects bare digits in text OUTSIDE of placeholders.
# Allowed: small cardinals "1"–"10" (for counting like "teen" approximation)
# Allowed: years like "2026" (date context) - we'll be strict and flag anything
_BARE_DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# Small cardinals whitelist (1–10) — allowed bare in voice text ("teen options hain")
_ALLOWED_BARE_CARDINALS = frozenset(str(i) for i in range(1, 11))

# Closed set of valid field names
_VALID_FIELDS = frozenset({
    "nav", "cagr1", "cagr3", "cagr5",
    "risk", "minsip", "expense", "category",
    "house", "lockin",
})


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed       : bool
    template     : str          # Pre-render template (placeholders intact)
    violations   : list[str]    # List of violation descriptions
    resolved     : dict[str, str] = field(default_factory=dict)  # slot → resolved value


@dataclass
class AuditEntry:
    """One entry in the tamper-proof audit log."""
    session_id      : str
    turn            : int
    epoch           : int
    user_input      : str
    intent          : str
    retrieval_confidence: float
    snapshot_hash   : str
    pre_render_template: str
    rendered_screen : str
    rendered_voice  : str
    rendered_audit  : str
    validation_passed: bool
    violations      : list[str]
    total_latency_ms: int
    timestamp       : str
    prev_entry_hash : str        # Hash of previous entry — forms chain
    entry_hash      : str = ""  # Computed after creation

    def compute_hash(self) -> str:
        """SHA256 of this entry (excluding entry_hash itself)."""
        d = {k: v for k, v in self.__dict__.items() if k != "entry_hash"}
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ── AuditStream validator ─────────────────────────────────────────────────────

class AuditStream:
    """
    Validates a generated template against the snapshot.

    Usage:
        auditor = AuditStream(snapshot)
        result  = auditor.validate(template_text)
        if result.passed:
            screen = auditor.render(template_text, "screen")
            voice  = auditor.render(template_text, "voice")
    """

    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot

    def validate(self, template: str) -> ValidationResult:
        """
        Validate a placeholder template against the snapshot.

        Checks:
        1. Every placeholder resolves to a non-None value
        2. Every fund index is in range
        3. Every field name is in the closed enum
        4. No bare digit appears outside a placeholder (except cardinals 1–10)
        """
        violations: list[str] = []
        resolved:   dict[str, str] = {}

        # ── Step 1: Extract all placeholders and validate each ───────────────
        placeholders = _PLACEHOLDER_RE.findall(template)

        for slot in placeholders:
            full_slot = f"{{{slot}}}"

            # Validate field name if present
            if "." in slot:
                parts = slot.split(".", 1)
                fund_key, field_name = parts[0], parts[1]

                if slot.startswith("U."):
                    field_name = slot[2:]
                    # User fields: amount, tenure, goal_amount, goal_years, goal_sip
                    if field_name not in ("amount", "tenure", "goal_amount", "goal_years", "goal_sip"):
                        violations.append(
                            f"UNKNOWN_USER_FIELD: {{U.{field_name}}} is not a valid user slot"
                        )
                        continue
                elif field_name not in _VALID_FIELDS:
                    violations.append(
                        f"INVALID_FIELD: {full_slot} — '{field_name}' not in closed field enum. "
                        f"Valid fields: {sorted(_VALID_FIELDS)}"
                    )
                    continue

            # Resolve the slot
            value = self.snapshot.resolve(slot)

            if value is None:
                # Distinguish between out-of-range index and missing field
                parts = slot.split(".", 1)
                fund_key = parts[0]
                if not slot.startswith("U.") and fund_key not in self.snapshot.funds:
                    violations.append(
                        f"INDEX_OUT_OF_RANGE: {full_slot} — only {len(self.snapshot.funds)} "
                        f"fund(s) in snapshot ({list(self.snapshot.funds.keys())})"
                    )
                else:
                    violations.append(
                        f"UNRESOLVED: {full_slot} resolved to None — field missing from snapshot"
                    )
            else:
                resolved[full_slot] = str(value)

        # ── Step 2: Check for bare digits outside placeholders ───────────────
        # Remove all placeholders from the template first
        stripped = _PLACEHOLDER_RE.sub("SLOT", template)

        for match in _BARE_DIGIT_RE.finditer(stripped):
            number_str = match.group()
            # Allow small cardinals (1–10) — for counting ("teen options")
            if number_str in _ALLOWED_BARE_CARDINALS:
                continue
            violations.append(
                f"BARE_DIGIT: '{number_str}' appears in template outside a placeholder. "
                f"All numbers must come from snapshot slots. "
                f"Context: '...{stripped[max(0,match.start()-20):match.end()+20]}...'"
            )

        passed = len(violations) == 0
        return ValidationResult(
            passed=passed,
            template=template,
            violations=violations,
            resolved=resolved,
        )

    def render(self, template: str, target: Literal["screen", "voice", "audit"], lang: str = "hinglish") -> str:
        """
        Render a validated template for the specified target.
        Rendering is a dict lookup — no network calls.

        Render-time failure rule (§16.5):
        1. Drop the clause if placeholder is missing
        2. Substitute a fixed hedge
        3. Abort the turn
        Never: Optional, N/A, fallback to model prose.
        """
        def replace_slot(match: re.Match) -> str:
            slot = match.group(1)
            value = self.snapshot.resolve(slot)

            if value is None:
                # Render-time failure — substitute hedge
                if target == "voice":
                    return "this exact figure cannot be confirmed right now" if lang == "english" else "iska exact figure abhi confirm nahi ho pa raha"
                return _HEDGE_BY_TARGET.get(target, "[data unavailable]")

            return _format_value(slot, value, target, lang=lang)

        return _PLACEHOLDER_RE.sub(replace_slot, template)


# ── Value formatting — render(tag, target) ────────────────────────────────────

_HEDGE_BY_TARGET = {
    "screen": "—",
    "voice" : "iska exact figure abhi confirm nahi ho pa raha",
    "audit" : "MISSING",
}

# Suffixes to strip from fund names for voice rendering (§22 — plan/option suffixes dropped)
_STRIP_SUFFIXES = [
    " - Direct Plan - Growth",
    " - Direct Plan - Growth Option",
    " - Regular Plan - Growth",
    " - Regular Plan - Growth Option",
    " - Direct Plan - IDCW",
    " - Regular Plan - IDCW",
    " - Direct Plan - Dividend",
    " - Regular Plan - Dividend",
    " - Direct Plan",
    " - Regular Plan",
    " - Growth Option",
    " - Growth",
    " Fund",        # keep at end — strips trailing "Fund" after other suffixes
]


def _spoken_fund_name(full_name: str) -> str:
    """
    Strip plan/option suffixes from fund name for voice rendering.
    'Parag Parikh Flexi Cap Fund - Direct Plan - Growth' → 'Parag Parikh Flexi Cap'
    Screen rendering keeps the full canonical name.
    """
    name = full_name.strip()
    name = re.sub(r'\s*\(\s*Form(?:erly)?\s+Know(?:n)?\s+as\s+[^)]+\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*Direct\s+Plan\s*-\s*(?:Direct\s+)?Growth(?:\s+Option)?\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*Regular\s+Plan\s*-\s*(?:Regular\s+)?Growth(?:\s+Option)?\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*Direct\s+Plan\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*Regular\s+Plan\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*(?:Direct\s+)?Growth(?:\s+Option)?\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+Fund$', '', name, flags=re.IGNORECASE)
    return name.strip(' -')


def _format_value(slot: str, value, target: Literal["screen", "voice", "audit"], lang: str = "hinglish") -> str:
    """
    Format a resolved value for the target channel.
    Three rules from §22:
      1. All quantities in English
      2. Small cardinals (1-10) may stay as-is
      3. Round for speech, log exact for audit
    """
    parts = slot.split(".", 1)
    field = parts[1] if len(parts) > 1 else None

    if target == "audit":
        # Exact values for audit — no rounding
        return str(value)

    if field in ("cagr1", "cagr3", "cagr5"):
        # Return percentage
        pct = float(value)
        if target == "screen":
            return f"{pct:.1f}%"
        else:  # voice
            return _verbalize_percent(pct)

    if field == "nav":
        nav = float(value)
        if target == "screen":
            return f"₹{nav:.2f}"
        else:
            return f"{nav:.2f} rupees" if lang == "english" else f"{nav:.2f} rupaye"

    if field == "minsip":
        amount = int(value)
        if target == "screen":
            return f"₹{amount:,}"
        else:
            return _verbalize_inr(amount, lang=lang)

    if field == "expense":
        exp = float(value)
        if target == "screen":
            return f"{exp:.2f}%"
        else:
            return f"{exp:.2f} percent"

    if field == "lockin":
        years = int(value)
        if years == 0:
            if target == "screen":
                return "No lock-in"
            return "no lock-in period" if lang == "english" else "koi lock-in nahi"
        elif years == 3:
            if target == "screen":
                return "3 years"
            return "three-year lock-in period" if lang == "english" else "teen saal ka lock-in"
        if target == "screen":
            return f"{years} years"
        return f"{_int_to_words(years)}-year lock-in period" if lang == "english" else f"{_int_to_words(years)} saal ka lock-in"

    if field == "risk":
        return str(value)  # Same in both targets

    if slot.startswith("U.amount") or slot.startswith("U.goal_sip"):
        amount = int(value)
        if target == "screen":
            return f"₹{amount:,}"
        else:
            return _verbalize_inr(amount, lang=lang)

    if slot.startswith("U.goal_amount"):
        amount = int(value)
        if target == "screen":
            if amount >= 10_000_000:
                return f"₹{amount / 10_000_000:.2f} Cr"
            elif amount >= 100_000:
                return f"₹{amount / 100_000:.2f} L"
            return f"₹{amount:,}"
        else:
            return _verbalize_inr(amount, lang=lang)

    if slot.startswith("U.goal_years"):
        yrs = int(value)
        if target == "screen":
            return f"{yrs} Years"
        else:
            return f"{_int_to_words(yrs)} years" if lang == "english" else f"{_int_to_words(yrs)} saal"

    # Fund name ({F1} with no field suffix) — strip suffixes for voice
    if field is None and not slot.startswith("U."):
        name = str(value)
        if target == "voice":
            return _spoken_fund_name(name)
        return name  # screen + audit: full canonical name

    # Fund house, category — same in screen and voice
    return str(value)


def _verbalize_percent(pct: float) -> str:
    """14.2 → 'fourteen point two percent'"""
    integer_part = int(pct)
    decimal_part = round((pct - integer_part) * 10)  # 1 decimal place
    words_int = _int_to_words(integer_part)
    if decimal_part > 0:
        return f"{words_int} point {_int_to_words(decimal_part)} percent"
    return f"{words_int} percent"


def _verbalize_inr(amount: int, lang: str = "hinglish") -> str:
    """5000 → 'five thousand rupaye' (HI) | 'five thousand rupees' (EN)"""
    curr = "rupees" if lang == "english" else "rupaye"
    if amount >= 10000000:
        crores = amount / 10000000
        if amount % 10000000 == 0:
            return f"{_int_to_words(int(crores))} crore {curr}"
        return f"{crores:.1f} crore {curr}"
    if amount >= 100000:
        lakhs = amount / 100000
        if amount % 100000 == 0:
            return f"{_int_to_words(int(lakhs))} lakh {curr}"
        return f"{lakhs:.1f} lakh {curr}"
    if amount >= 1000:
        thousands = amount // 1000
        remainder = amount % 1000
        if remainder == 0:
            return f"{_int_to_words(thousands)} thousand {curr}"
        return f"{_int_to_words(thousands)} thousand {_int_to_words(remainder)} {curr}"
    return f"{_int_to_words(amount)} {curr}"


_ONES = [
    "", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]


def _int_to_words(n: int) -> str:
    """Convert integer 0–999 to English words."""
    if n < 0:
        return f"minus {_int_to_words(-n)}"
    if n == 0:
        return "zero"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[ones] if ones else "")
    hundreds, rest = divmod(n, 100)
    return _ONES[hundreds] + " hundred" + (" " + _int_to_words(rest) if rest else "")
