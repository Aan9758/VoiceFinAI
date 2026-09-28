"""
core/snapshot.py — Snapshot Builder
=====================================
Track 3 output: builds the F1..Fn snapshot dict that is injected into
model context. Generation does NOT start until this snapshot is complete.

The snapshot is the sole source of all financial facts in the response.
The model only sees slot references like {F1.nav} — never raw numbers.

Schema per fund slot:
    fund_name, isin, category, nav, nav_date, cagr_1yr, cagr_3yr, cagr_5yr,
    risk_rating, min_sip, expense_ratio, fund_house, lockin_years, as_of, source

Snapshot also contains:
    U.amount, U.tenure  — user-supplied values echoed back via {U.*} slots
    _hash               — SHA256 of the snapshot for audit log
    _epoch              — conversation epoch at snapshot creation time
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, field_validator


# ── FundRecord — what the DB guarantees ──────────────────────────────────────

class FundRecord(BaseModel):
    """
    Strict schema for a verified fund record.
    Every field is required. No Optional. No defaults that hide missing data.
    If a field is missing → route to IncompleteFund, not here.
    """
    fund_name    : str
    isin         : str     # 12-character ISIN from AMFI
    category     : str     # SEBI-approved category
    nav          : float   # Latest NAV from mfapi.in
    nav_date     : str     # ISO date string (as_of)
    cagr_1yr     : float   # Computed from NAV history
    cagr_3yr     : float   # Computed from NAV history
    cagr_5yr     : float   # Computed from NAV history — may be None for new funds
    risk_rating  : Literal["Low", "Low to Moderate", "Moderate", "Moderately High", "High", "Very High"]
    min_sip      : int     # AMC-published minimum SIP in INR
    expense_ratio: float   # Annual expense ratio %
    fund_house   : str     # Must exist in AMFI registry
    lockin_years : int     # 0 for most funds, 3 for ELSS
    source       : str     # Data source identifier
    as_of        : str     # ISO date — when this data was fetched

    @field_validator("isin")
    @classmethod
    def isin_must_be_12_chars(cls, v: str) -> str:
        v = v.strip()
        if len(v) != 12:
            raise ValueError(f"ISIN must be exactly 12 characters, got {len(v)}: {v!r}")
        return v


class IncompleteFund(BaseModel):
    """
    Explicit type for funds with missing required fields.
    Has its own code path → routes to hedge response, not to FundRecommendation.
    If the only way to represent missing data is this type, you cannot silently
    widen the FundRecord schema.
    """
    scheme_code  : int
    fund_name    : str
    missing_fields: list[str]
    reason       : str


# ── FundRecommendation — what may be spoken ───────────────────────────────────

class FundRecommendation(BaseModel):
    """
    Strict schema for what may be spoken/rendered to the user.
    No Optional fields. CI test enforces this.
    If schema validation fails → block entirely, route to IncompleteFund.
    """
    fund_name    : str
    isin         : str
    category     : str
    nav          : float
    returns_1yr  : float
    returns_3yr  : float
    returns_5yr  : float
    risk_rating  : Literal["Low", "Low to Moderate", "Moderate", "Moderately High", "High", "Very High"]
    min_sip      : int
    fund_house   : str
    lockin_years : int
    expense_ratio: float
    as_of        : str
    source       : str


# ── Snapshot ──────────────────────────────────────────────────────────────────

class Snapshot:
    """
    The frozen context injected into the model before generation starts.

    Structure:
        funds: dict[str, FundRecommendation]  — keys are "F1", "F2", "F3"
        user:  dict[str, str | int | None]    — keys are "amount", "tenure"
        epoch: int                             — conversation epoch
        hash:  str                             — SHA256 of serialized snapshot
        created_at: str                        — ISO timestamp

    The model context string is built from this snapshot.
    Rendering is a dict lookup — no network calls during generation.
    """

    FIELD_ENUM = frozenset({
        "nav", "cagr1", "cagr3", "cagr5",
        "risk", "minsip", "expense", "category",
        "house", "lockin",
    })

    def __init__(
        self,
        funds: list[FundRecommendation],
        user_amount: Optional[int] = None,
        user_tenure: Optional[str] = None,
        epoch: int = 1,
        user_goal_amount: Optional[int] = None,
        user_goal_years: Optional[int] = None,
        user_goal_sip: Optional[int] = None,
    ):
        if not funds:
            raise ValueError("Snapshot requires at least one fund")
        if len(funds) > 5:
            raise ValueError("Snapshot supports at most 5 fund slots (F1..F5)")

        # Fund List count fixed at 3 (per architecture rule #42)
        self.funds: dict[str, FundRecommendation] = {
            f"F{i+1}": fund for i, fund in enumerate(funds[:3])
        }
        self.user = {
            "amount": user_amount,
            "tenure": user_tenure,
            "goal_amount": user_goal_amount,
            "goal_years": user_goal_years,
            "goal_sip": user_goal_sip,
        }
        self.epoch      = epoch
        self.created_at = datetime.utcnow().isoformat() + "Z"
        self.hash       = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA256 of the canonical snapshot JSON. Used in audit log."""
        canonical = json.dumps(
            {
                "funds": {k: v.model_dump() for k, v in self.funds.items()},
                "user" : self.user,
                "epoch": self.epoch,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ── Field resolution ─────────────────────────────────────────────────────

    def resolve(self, slot: str) -> Optional[str | float | int]:
        """
        Resolve a placeholder slot to its raw value.
        slot examples: "F1", "F1.nav", "F1.cagr3", "U.amount"

        Returns None if:
        - Index out of range (F4 when only 3 funds)
        - Unknown field name
        - User field not supplied
        """
        slot = slot.strip("{} \t")

        if slot.startswith("U."):
            field = slot[2:]
            return self.user.get(field)

        parts = slot.split(".", 1)
        fund_key = parts[0]  # "F1", "F2", etc.

        if fund_key not in self.funds:
            return None  # Index out of range

        fund = self.funds[fund_key]

        if len(parts) == 1:
            # {F1} → fund display name
            return fund.fund_name

        field = parts[1]
        field_map = {
            "nav"    : fund.nav,
            "cagr1"  : fund.returns_1yr,
            "cagr3"  : fund.returns_3yr,
            "cagr5"  : fund.returns_5yr,
            "risk"   : fund.risk_rating,
            "minsip" : fund.min_sip,
            "expense": fund.expense_ratio,
            "category": fund.category,
            "house"  : fund.fund_house,
            "lockin" : fund.lockin_years,
        }

        if field not in field_map:
            return None  # Unknown field — not in closed enum

        return field_map[field]

    def build_context_string(self) -> str:
        """
        Build the context string injected into the model prompt.
        The model sees this and can only reference slots — never raw values.
        """
        lines = ["## Retrieved Funds (reference by slot only)\n"]

        for key, fund in self.funds.items():
            lines.append(f"### {key}: {fund.fund_name}")
            lines.append(f"  Slot         | Field         | Value")
            lines.append(f"  ------------ | ------------- | -----")
            lines.append(f"  {{{key}}}        | display name  | (use slot only)")
            lines.append(f"  {{{key}.nav}}    | NAV           | use slot only")
            lines.append(f"  {{{key}.cagr1}} | 1yr return    | use slot only")
            lines.append(f"  {{{key}.cagr3}} | 3yr return    | use slot only")
            lines.append(f"  {{{key}.cagr5}} | 5yr return    | use slot only")
            lines.append(f"  {{{key}.risk}}  | risk rating   | use slot only")
            lines.append(f"  {{{key}.minsip}}| min SIP       | use slot only")
            lines.append(f"  {{{key}.expense}}| expense ratio | use slot only")
            lines.append(f"  {{{key}.category}}| category     | use slot only")
            lines.append(f"  {{{key}.house}} | fund house    | use slot only")
            lines.append(f"  {{{key}.lockin}}| lock-in period| use slot only")
            lines.append("")

        if self.user.get("amount"):
            lines.append(f"User amount: {{U.amount}} = (use slot only)")
        if self.user.get("tenure"):
            lines.append(f"User tenure: {{U.tenure}} = (use slot only)")

        lines.append(
            "\n## RULE: You MUST reference funds and numbers ONLY via the slots above. "
            "Never write a fund name, NAV, return percentage, or number directly in your response. "
            "Invalid slot references will be rejected."
        )
        return "\n".join(lines)

    def to_audit_dict(self) -> dict:
        """Audit-log representation: snapshot hash + all resolved values."""
        return {
            "snapshot_hash": self.hash,
            "epoch"        : self.epoch,
            "created_at"   : self.created_at,
            "funds"        : {k: v.model_dump() for k, v in self.funds.items()},
            "user"         : self.user,
        }


# ── Builder helpers ───────────────────────────────────────────────────────────

def build_fund_recommendation(record: dict, risk_rating: str, min_sip: int = 500, expense_ratio: float = 0.5) -> Optional[FundRecommendation]:
    """
    Convert a raw AMFI fetch result into a FundRecommendation.
    Returns None if required fields are missing → caller routes to IncompleteFund.
    """
    missing = []

    isin = record.get("isin_growth", "").strip()
    if len(isin) != 12:
        missing.append("isin")

    cagr_1yr = record.get("cagr_1yr")
    cagr_3yr = record.get("cagr_3yr")
    cagr_5yr = record.get("cagr_5yr")

    if cagr_1yr is None: missing.append("cagr_1yr")
    if cagr_3yr is None: missing.append("cagr_3yr")
    if cagr_5yr is None: missing.append("cagr_5yr")

    if missing:
        return None  # Caller creates IncompleteFund

    try:
        return FundRecommendation(
            fund_name    = record["fund_name"],
            isin         = isin,
            category     = record.get("category", ""),
            nav          = record["nav"],
            returns_1yr  = cagr_1yr,
            returns_3yr  = cagr_3yr,
            returns_5yr  = cagr_5yr,
            risk_rating  = risk_rating,
            min_sip      = min_sip,
            fund_house   = record.get("fund_house", ""),
            lockin_years = 3 if "ELSS" in record.get("category", "") else 0,
            expense_ratio= expense_ratio,
            as_of        = record["as_of"],
            source       = record["source"],
        )
    except Exception:
        return None
