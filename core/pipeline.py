"""
core/pipeline.py — One turn, end to end
========================================
This replaces the body of _run_pipeline() in app.py.

Old flow:   query -> keyword risk -> fixed catalogue -> one of 6 f-strings
New flow:   query -> session-aware intent -> bucket retrieval -> snapshot
                  -> language model (or rotating bank) emits a SLOT TEMPLATE
                  -> AuditStream validates the template
                  -> render voice + screen from the snapshot
                  -> audit entry appended to the hash chain

The ordering matters. Generation now happens against slots and validation
happens against generated text, which is what makes AuditStream a firewall
rather than a self-check on a string this file wrote a line earlier.
"""

from __future__ import annotations

import re
import time
import math
from typing import Optional

from core import dialogue, generate, retrieval2, universe
from core.auditstream import AuditStream, AuditEntry

_audit_log: list[AuditEntry] = []


def _prev_hash() -> str:
    return _audit_log[-1].entry_hash if _audit_log else "GENESIS"


def run_turn(query: str, session_id: str, synthesize) -> dict:
    """
    synthesize: callable(text) -> audio_token. Injected so this module stays
                free of Flask and of the TTS backend.
    """
    t0 = time.perf_counter()
    session = dialogue.get_session(session_id)
    turn = dialogue.parse_turn(query, session)

    effective_lang = turn.get("language") or getattr(session, "language", "hinglish")
    if effective_lang == "auto":
        effective_lang = "hinglish"

    intent = turn["intent"]

    # ── Smart promotion: CLARIFY with a named fund → FUND_DETAIL ─────────────
    # When the user mentions a specific fund by name (e.g. "Mirae Asset ke baare
    # mein batao") the dialogue router returns CLARIFY because there is no clean
    # intent signal, but mentioned_fund is already populated. Silently falling
    # through to a generic clarify template ignores the fund the user clearly
    # wants. Promote to FUND_DETAIL so the fund is fetched and spoken about.
    if intent == dialogue.CLARIFY and turn.get("mentioned_fund"):
        turn["intent"] = dialogue.FUND_DETAIL
        intent = dialogue.FUND_DETAIL

    # ── Turns that need no snapshot ──────────────────────────────────────────
    if intent in (dialogue.OUT_OF_SCOPE, dialogue.GREET, dialogue.CLARIFY, dialogue.LANG_SWITCH):
        text = generate.from_bank(turn, session)
        return _finish(intent, text, None, None, turn, session, t0, synthesize,
                       cards=[], view="message", language=effective_lang)

    # ── Which funds this turn is about ───────────────────────────────────────
    only: Optional[list[str]] = None
    limit = 3

    if intent == dialogue.FUND_DETAIL:
        if turn["target_slots"]:
            ref = session.last_list or session.last_slots
            resolved = [
                session.last_slots.get(slot) or ref.get(slot)
                for slot in turn["target_slots"]
            ]
            only = [rec.fund_name for rec in resolved if rec]
        elif turn["mentioned_fund"]:
            only = [turn["mentioned_fund"]]
            # Also inherit the bucket from the universe record so snapshot uses
            # the right risk band for this specific fund
            u_rec = universe.find_by_name(turn["mentioned_fund"])
            if u_rec and not turn.get("risk"):
                turn["risk"] = u_rec.get("appetite_bucket") or "moderate"
        limit = min(3, len(only)) if turn.get("why_query") and only else 1

    elif intent in (dialogue.METRIC, dialogue.CONFIRM_SIP, dialogue.GRAPH):
        slot = (turn["target_slots"] or [session.focus_slot or "F1"])[0]
        ref = session.last_list or session.last_slots
        rec = session.last_slots.get(slot) or ref.get(slot)
        only = [rec.fund_name] if rec else None
        limit = 1

    elif intent == dialogue.COMPARE:
        ref = session.last_list or session.last_slots
        names = [ref[s].fund_name for s in turn["target_slots"] if s in ref]
        only = names[:2] or None
        limit = 2

    # An AMC is a hard constraint for a fund search or goal request such as
    # "show SBI mutual funds".  It is not a constraint when the user names a
    # fund while asking to compare it with the currently focused fund: the
    # name can reveal its AMC, but that should not erase the other comparison
    # candidate.
    requested_house = turn.get("fund_house") if intent in (dialogue.DISCOVER, dialogue.GOAL) else None

    effective_amount = turn.get("amount") or session.amount
    effective_horizon = turn.get("horizon") or session.horizon
    what_if = turn.get("what_if") or getattr(session, "what_if", None)
    if what_if and "amount" in what_if:
        effective_amount = what_if["amount"]

    goal_data = None
    goal_plan_mode = None
    goal_months = None
    goal_monthly_sip = None
    if intent == dialogue.GOAL:
        goal_amount = turn.get("goal_amount") or getattr(session, "goal_amount", None) or 10_000_000
        previous_goal_amount = getattr(session, "goal_amount", None)
        turn_goal_sip = turn.get("goal_monthly_sip")
        # Reuse a previously stated SIP only when the user is continuing the
        # same corpus plan.  A new target must not silently inherit an old SIP.
        goal_monthly_sip = turn_goal_sip or (
            getattr(session, "goal_monthly_sip", None)
            if previous_goal_amount == goal_amount else None
        )
        stated_goal_years = turn.get("goal_years")
        goal_purpose = turn.get("goal_purpose") or getattr(session, "goal_purpose", None) or "Financial Goal"

        if goal_monthly_sip and not stated_goal_years:
            # The user supplied the contribution but not a deadline.  Solve
            # for time instead of inventing a five-year horizon.
            goal_plan_mode = "time_to_goal"
            goal_months = _calc_months_to_goal(goal_amount, goal_monthly_sip, 14.0)
            goal_years = max(1, math.ceil(goal_months / 12))
            effective_amount = goal_monthly_sip
        else:
            # The user supplied a deadline (or explicitly asks for a target
            # plan), so solve for the monthly SIP required for that deadline.
            goal_plan_mode = "sip_to_goal"
            goal_years = stated_goal_years or getattr(session, "goal_years", None) or 5
            balanced_sip = _calc_required_sip(goal_amount, goal_years, 14.0)
            goal_monthly_sip = balanced_sip
            effective_amount = balanced_sip

        session.amount = effective_amount
        session.goal_amount = goal_amount
        session.goal_years = goal_years
        session.goal_monthly_sip = effective_amount
        session.goal_purpose = goal_purpose
        effective_horizon = f"{goal_years} years"
        turn["goal_plan_mode"] = goal_plan_mode
        turn["goal_months"] = goal_months
        turn["goal_monthly_sip"] = effective_amount

        snapshot = retrieval2.build_snapshot(
            bucket="moderate",
            amount=effective_amount,
            tenure=effective_horizon,
            epoch=session.epoch,
            limit=3,
            goal_amount=goal_amount,
            goal_years=goal_years,
            goal_sip=effective_amount,
            fund_house=requested_house,
        )
    else:
        snapshot = retrieval2.build_snapshot(
            bucket=turn.get("risk") or session.risk or ("all" if requested_house else "moderate"),
            amount=effective_amount,
            tenure=effective_horizon,
            epoch=session.epoch,
            only=only,
            limit=limit,
            fund_house=requested_house,
        )

    if snapshot is None:
        turn["intent"] = dialogue.CLARIFY
        if requested_house:
            house_candidates = [
                item for item in universe.load()
                if " ".join((item.get("fund_house") or "").casefold().split())
                == " ".join(requested_house.casefold().split())
                and item.get("appetite_bucket") == (turn.get("risk") or session.risk or "moderate")
            ]
            if house_candidates:
                risk_label = turn.get("risk") or session.risk or "matching"
                if effective_lang == "english":
                    text = (
                        f"I found {len(house_candidates)} {requested_house} option(s) in the "
                        f"{risk_label} range, but my current records don't have enough verified "
                        "history to show their returns. I won't present unverified figures as "
                        "recommendations. Would you like another risk level, or should I include "
                        "other fund houses?"
                    )
                else:
                    text = (
                        f"{requested_house} ke {risk_label} range mein {len(house_candidates)} "
                        "option(s) mile, lekin unke returns verify karne ke liye meri current list "
                        "mein poori history nahi hai. Main unverified figures ko recommendation "
                        "ke roop mein nahi dikhाऊँगा. Kya doosra risk level dekhein, ya doosre fund "
                        "houses bhi shamil karun?"
                    )
            elif effective_lang == "english":
                text = (
                    f"I couldn't find a {turn.get('risk') or 'matching-risk'} fund from "
                    f"{requested_house} in my current list. Would you like another risk level, "
                    "or should I include other fund houses?"
                )
            else:
                text = (
                    f"Meri current list mein {requested_house} ka us risk level ka fund nahi mila. "
                    "Kya doosra risk level dekhein, ya doosre fund houses bhi dikhau?"
                )
        else:
            text = generate.from_bank(turn, session)
        return _finish(dialogue.CLARIFY, text, None, None, turn, session, t0,
                       synthesize, cards=[], view="message", language=effective_lang)

    # ── Language layer ───────────────────────────────────────────────────────
    template, source = generate.generate_template(turn, snapshot, session)
    if requested_house and _template_mentions_other_house(template, requested_house):
        template = _house_scoped_fallback(turn, snapshot)
        source = "bank_after_house_scope_violation"

    auditor = AuditStream(snapshot)
    result = auditor.validate(template)
    model_violations: list[str] = []

    if not result.passed:
        # The firewall did its job. Fall back to a template we know is clean
        # rather than speaking an unvalidated string.
        model_violations = result.violations
        print(f"  [pipeline] AuditStream REJECTED generated template:")
        for v in model_violations:
            print(f"    → {v}")
        template = generate.from_bank(turn, session)
        source = "bank_after_violation"
        result = auditor.validate(template)
        if not result.passed:
            text = "One moment, I am unable to verify this figure right now." if effective_lang == "english" else "Ek second, ye figure main abhi confirm nahi kar pa rahi."
            return _finish("error", text, None, snapshot, turn, session, t0,
                           synthesize, cards=[], view="message",
                           violations=result.violations, language=effective_lang)

    voice = auditor.render(template, "voice", lang=effective_lang)
    screen = auditor.render(template, "screen", lang=effective_lang)

    session.remember_snapshot(snapshot, is_list=(intent in (dialogue.DISCOVER, dialogue.GOAL)))
    if intent in (dialogue.FUND_DETAIL, dialogue.METRIC, dialogue.GRAPH, dialogue.GOAL) and snapshot.funds:
        session.focus_slot = "F1"

    view = {
        dialogue.FUND_DETAIL: "detail",
        dialogue.COMPARE: "comparison",
        dialogue.CONFIRM_SIP: "confirmation",
        dialogue.METRIC: "detail",
        dialogue.GRAPH: "chart",
        dialogue.GOAL: "goal",
    }.get(intent, "fund_list")

    cards = [_card(k, f) for k, f in snapshot.funds.items()]
    chart_data = None
    if intent == dialogue.GRAPH and snapshot.funds:
        chart_data = _build_chart_data(list(snapshot.funds.values())[0], effective_amount, effective_horizon, what_if=what_if)

    if intent == dialogue.GOAL:
        goal_data = _build_goal_data(
            goal_amount,
            goal_years,
            goal_purpose,
            snapshot,
            session,
            plan_mode=goal_plan_mode,
            monthly_sip=goal_monthly_sip,
            months_to_goal=goal_months,
        )

    return _finish("recommendation", voice, screen, snapshot, turn, session, t0,
                   synthesize, cards=cards, view=view,
                   template=template, gen_source=source,
                   violations=model_violations, chart_data=chart_data,
                   goal_data=goal_data, language=effective_lang)


def _template_mentions_other_house(template: str, requested_house: str) -> bool:
    """Reject a generated fund name from an AMC the user did not request."""
    normalized = re.sub(r"[^a-z0-9]+", " ", template.casefold()).strip()
    padded = f" {normalized} "
    for record in universe.load():
        other_house = record.get("fund_house", "")
        if other_house.casefold() == requested_house.casefold():
            continue
        brand = re.sub(r"\s+mutual fund$", "", other_house, flags=re.IGNORECASE).casefold().strip()
        if brand and f" {brand} " in padded:
            return True
        name = (record.get("fund_name") or "").split(" - Direct Plan", 1)[0]
        name = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
        if len(name) >= 7 and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", normalized):
            return True
    return False


def _house_scoped_fallback(turn: dict, snapshot) -> str:
    """Deterministic, slot-checked answer for an AMC-constrained request."""
    house = turn["fund_house"]
    count = len(snapshot.funds)
    if turn.get("language") == "english":
        if count == 1:
            return f"From {house}, {{F1}} is in my current list. It has {{F1.risk}} risk and a three-year return of {{F1.cagr3}}."
        return f"From {house}, {{F1}} is one option to review. It has {{F1.risk}} risk and a three-year return of {{F1.cagr3}}. {{F2}} is another option from {house}."
    if count == 1:
        return f"{house} mein {{F1}} available hai. Iska risk {{F1.risk}} hai aur teen saal ka return {{F1.cagr3}} raha."
    return f"{house} mein {{F1}} ek option hai. Iska risk {{F1.risk}} aur teen saal ka return {{F1.cagr3}} raha. {{F2}} doosra option hai."


def _calc_required_sip(target_fv: int, years: int, annual_cagr: float) -> int:
    years = max(1, years)
    target_fv = max(1000, target_fv)
    annual_cagr = max(-50.0, min(100.0, annual_cagr))
    n = years * 12
    if abs(annual_cagr) < 1e-6:
        return max(500, int(round(target_fv / n)))
    r = (1.0 + annual_cagr / 100.0) ** (1.0 / 12.0) - 1.0
    if abs(r) < 1e-7:
        return max(500, int(round(target_fv / n)))
    num = target_fv * r
    den = ((1.0 + r) ** n - 1.0) * (1.0 + r)
    if den <= 0:
        return max(500, int(round(target_fv / n)))
    return max(500, int(round(num / den)))


def _calc_months_to_goal(target_fv: int, monthly_sip: int, annual_cagr: float) -> int:
    """Return the whole number of monthly SIPs needed to reach a target.

    Uses the same beginning-of-month SIP convention as ``_calc_required_sip``
    so the two goal paths agree with one another.  The return assumption is a
    planning input, never a promised outcome.
    """
    target_fv = max(1_000, int(target_fv))
    monthly_sip = max(1, int(monthly_sip))
    annual_cagr = max(-50.0, min(100.0, float(annual_cagr)))

    if abs(annual_cagr) < 1e-6:
        return max(1, math.ceil(target_fv / monthly_sip))

    monthly_rate = (1.0 + annual_cagr / 100.0) ** (1.0 / 12.0) - 1.0
    if monthly_rate <= 1e-8:
        return max(1, math.ceil(target_fv / monthly_sip))

    # Future value for an annuity due:
    # SIP * (((1 + r)^n - 1) / r) * (1 + r) = target
    inside = 1.0 + (target_fv * monthly_rate) / (monthly_sip * (1.0 + monthly_rate))
    if inside <= 1.0:
        return 1
    return max(1, math.ceil(math.log(inside) / math.log(1.0 + monthly_rate)))


def _calc_stepup_factor(years: int, annual_cagr: float, step_up_pct: float = 10.0) -> float:
    years = max(1, years)
    annual_cagr = max(-50.0, min(100.0, annual_cagr))
    r = (1.0 + annual_cagr / 100.0) ** (1.0 / 12.0) - 1.0 if abs(annual_cagr) >= 1e-6 else 0.0
    n = years * 12
    total_fv = 0.0
    for y in range(years):
        sip_this_year = (1.0 + step_up_pct / 100.0) ** y
        for m in range(12):
            months_rem = n - (y * 12 + m)
            if abs(r) < 1e-7:
                fv_inst = sip_this_year
            else:
                fv_inst = sip_this_year * ((1.0 + r) ** months_rem)
            total_fv += fv_inst
    return max(1.0, total_fv)


def _calc_required_stepup_sip(target_fv: int, years: int, annual_cagr: float, step_up_pct: float = 10.0) -> int:
    factor = _calc_stepup_factor(years, annual_cagr, step_up_pct)
    return max(500, int(round(target_fv / factor)))


def _format_goal_duration(months: int) -> str:
    months = max(1, int(months))
    years, remaining = divmod(months, 12)
    if years and remaining:
        return f"{years} yr {remaining} mo"
    if years:
        return f"{years} yr"
    return f"{remaining} mo"


def _build_goal_data(
    target: int,
    years: int,
    purpose: str,
    snapshot,
    session,
    *,
    plan_mode: str = "sip_to_goal",
    monthly_sip: Optional[int] = None,
    months_to_goal: Optional[int] = None,
) -> dict:
    profiles = [
        {"name": "Conservative", "cagr": 10.0, "category": "Debt & Hybrid Funds", "risk": "Low to Moderate", "desc": "Capital preservation & steady growth", "recommended": False},
        {"name": "Balanced", "cagr": 14.0, "category": "Flexi Cap & Large Cap Funds", "risk": "Moderate", "desc": "Balanced growth and risk (Recommended)", "recommended": True},
        {"name": "Aggressive", "cagr": 18.0, "category": "Mid Cap & Small Cap Funds", "risk": "High / Very High", "desc": "Maximum long-term compounding", "recommended": False},
    ]

    fund_list = list(snapshot.funds.values()) if snapshot and snapshot.funds else []

    strategies = []
    for i, p in enumerate(profiles):
        cagr = p["cagr"]
        if plan_mode == "time_to_goal" and monthly_sip:
            sip_flat = monthly_sip
            timeline_months = _calc_months_to_goal(target, sip_flat, cagr)
            sip_stepup = None
            invested_flat = sip_flat * timeline_months
        else:
            sip_flat = _calc_required_sip(target, years, cagr)
            timeline_months = years * 12
            sip_stepup = _calc_required_stepup_sip(target, years, cagr, 10.0)
            invested_flat = sip_flat * timeline_months
        wealth_gain = max(0, target - invested_flat)
        f_rec = fund_list[i % len(fund_list)] if fund_list else None

        strategies.append({
            "name": p["name"],
            "cagr": cagr,
            "cagr_fmt": f"{cagr:.0f}%",
            "category": p["category"],
            "risk": p["risk"],
            "desc": p["desc"],
            "recommended": p["recommended"],
            "required_sip": sip_flat,
            "required_sip_fmt": _format_lakh_cr(sip_flat),
            "sip_label": "Your Monthly SIP" if plan_mode == "time_to_goal" else "Required Monthly SIP",
            "stepup_sip": sip_stepup,
            "stepup_sip_fmt": _format_lakh_cr(sip_stepup) if sip_stepup else None,
            "total_invested": invested_flat,
            "total_invested_fmt": _format_lakh_cr(invested_flat),
            "wealth_gain": wealth_gain,
            "wealth_gain_fmt": _format_lakh_cr(wealth_gain),
            "timeline_months": timeline_months,
            "timeline_fmt": _format_goal_duration(timeline_months),
            "fund_name": f_rec.fund_name if f_rec else "Top Rated Equity Fund",
            "fund_short": _short(f_rec.fund_name) if f_rec else "Flexi Cap Fund",
        })

    bal = strategies[1]
    return {
        "target_amount": target,
        "target_amount_fmt": _format_lakh_cr(target),
        "target_years": years,
        "purpose": purpose,
        "plan_mode": plan_mode,
        "timeline_months": months_to_goal or bal["timeline_months"],
        "timeline_fmt": _format_goal_duration(months_to_goal or bal["timeline_months"]),
        "user_monthly_sip": monthly_sip if plan_mode == "time_to_goal" else None,
        "user_monthly_sip_fmt": _format_lakh_cr(monthly_sip) if plan_mode == "time_to_goal" and monthly_sip else None,
        "strategies": strategies,
        "recommended_sip": bal["required_sip"],
        "recommended_sip_fmt": bal["required_sip_fmt"],
        "recommended_stepup_sip": bal["stepup_sip"],
        "recommended_stepup_sip_fmt": bal["stepup_sip_fmt"],
    }


def _build_chart_data(f, amount: Optional[int], horizon: Optional[str], what_if: Optional[dict] = None) -> dict:
    sip = amount or 10000
    r_annual = f.returns_5yr if (f.returns_5yr and f.returns_5yr > 0) else (f.returns_3yr or 12.0)

    # What-If overrides if provided
    if what_if:
        if "cagr" in what_if:
            r_annual = float(what_if["cagr"])
        if "amount" in what_if:
            sip = int(what_if["amount"])

    # Monthly compounding rate
    r_annual = max(-50.0, min(100.0, r_annual))
    r = (1.0 + r_annual / 100.0) ** (1.0 / 12.0) - 1.0 if abs(r_annual) >= 1e-6 else 0.0

    projections = []
    for yrs in [1, 3, 5, 10]:
        n = yrs * 12
        invested = sip * n
        if abs(r) < 1e-7:
            fv = invested
        else:
            fv = int(sip * (((1.0 + r) ** n - 1.0) / r) * (1.0 + r))
        gain = max(0, fv - invested)

        # 10% Step-Up SIP calculation
        invested_stepup = 0
        fv_stepup = 0.0
        for y in range(yrs):
            sip_y = int(sip * ((1.10) ** y))
            for m in range(12):
                invested_stepup += sip_y
                months_rem = n - (y * 12 + m)
                if abs(r) < 1e-7:
                    fv_stepup += sip_y
                else:
                    fv_stepup += sip_y * ((1.0 + r) ** months_rem)
        fv_stepup = int(fv_stepup)
        gain_stepup = max(0, fv_stepup - invested_stepup)

        projections.append({
            "years": yrs,
            "invested": invested,
            "fv": fv,
            "gain": gain,
            "invested_fmt": _format_lakh_cr(invested),
            "fv_fmt": _format_lakh_cr(fv),
            "gain_fmt": _format_lakh_cr(gain),
            "invested_stepup": invested_stepup,
            "fv_stepup": fv_stepup,
            "gain_stepup": gain_stepup,
            "invested_stepup_fmt": _format_lakh_cr(invested_stepup),
            "fv_stepup_fmt": _format_lakh_cr(fv_stepup),
            "gain_stepup_fmt": _format_lakh_cr(gain_stepup),
        })

    return {
        "fund_name": f.fund_name,
        "name_spoken": _short(f.fund_name),
        "house": f.fund_house,
        "category": f.category,
        "risk": f.risk_rating,
        "monthly_sip": sip,
        "monthly_sip_fmt": f"₹{sip:,}",
        "cagr_used": round(r_annual, 1),
        "cagr1": f"{f.returns_1yr:.1f}%",
        "cagr3": f"{f.returns_3yr:.1f}%",
        "cagr5": f"{f.returns_5yr:.1f}%",
        "cagr1_val": f.returns_1yr,
        "cagr3_val": f.returns_3yr,
        "cagr5_val": f.returns_5yr,
        "projections": projections,
        "what_if": what_if,
    }


def _format_lakh_cr(val: int) -> str:
    if val >= 10_000_000:
        return f"₹{val / 10_000_000:.2f} Cr"
    elif val >= 100_000:
        return f"₹{val / 100_000:.2f} L"
    else:
        return f"₹{val:,}"


def _finish(rtype, voice, screen, snapshot, turn, session, t0, synthesize,
            cards, view, violations=None, template=None, gen_source="bank",
            chart_data=None, goal_data=None, language="hinglish") -> dict:
    token = synthesize(voice)
    latency = int((time.perf_counter() - t0) * 1000)

    entry = AuditEntry(
        session_id=session.session_id,
        turn=session.turn,
        epoch=session.epoch,
        user_input=turn["query"],
        intent=turn["intent"],
        retrieval_confidence=turn["confidence"],
        snapshot_hash=snapshot.hash if snapshot else "none",
        pre_render_template=template or voice,
        rendered_screen=screen or voice,
        rendered_voice=voice,
        rendered_audit=voice,
        validation_passed=not violations,
        violations=violations or [],
        total_latency_ms=latency,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        prev_entry_hash=_prev_hash(),
    )
    entry.entry_hash = entry.compute_hash()
    _audit_log.append(entry)

    return {
        "type": rtype,
        "view": view,
        "voice_text": voice,
        "screen_text": screen or voice,
        "language": language,
        "cards": cards,
        "chart_data": chart_data,
        "goal_data": goal_data,
        "audio_token": token,
        "intent": turn["intent"],
        "risk": turn.get("risk") or session.risk,
        "amount": turn.get("amount") or session.amount,
        "horizon": turn.get("horizon") or session.horizon,
        "goal_amount": session.goal_amount,
        "goal_years": session.goal_years,
        "goal_purpose": session.goal_purpose,
        "confidence": turn["confidence"],
        "generator": gen_source,
        "snapshot_hash": (snapshot.hash[:16] + "...") if snapshot else None,
        "latency_ms": latency,
        "violations": violations or [],
    }


_STRIP = [" - Direct Plan - Growth Option", " - Direct Plan - Growth",
          " - Regular Plan - Growth Option", " - Regular Plan - Growth",
          " - Direct Plan", " - Regular Plan", " - Growth Option", " - Growth", " Fund"]


def _short(name: str) -> str:
    name = name.strip()
    name = re.sub(r'\s*\(\s*Form(?:erly)?\s+Know(?:n)?\s+as\s+[^)]+\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*Direct\s+Plan\s*-\s*(?:Direct\s+)?Growth(?:\s+Option)?\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*Regular\s+Plan\s*-\s*(?:Regular\s+)?Growth(?:\s+Option)?\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*Direct\s+Plan\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*Regular\s+Plan\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*-\s*(?:Direct\s+)?Growth(?:\s+Option)?\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+Fund$', '', name, flags=re.IGNORECASE)
    return name.strip(' -')


def _risk_class(risk: str) -> str:
    return {"Low": "risk-low", "Low to Moderate": "risk-low-mod",
            "Moderate": "risk-mod", "Moderately High": "risk-mod-high",
            "High": "risk-high", "Very High": "risk-very-high"}.get(risk, "risk-mod")


def _card(slot: str, f) -> dict:
    return {
        "slot": slot,
        "name": f.fund_name,
        "name_spoken": _short(f.fund_name),
        "nav": f"₹{f.nav:.2f}",
        "cagr1": f"{f.returns_1yr:.1f}%",
        "cagr3": f"{f.returns_3yr:.1f}%",
        "cagr5": f"{f.returns_5yr:.1f}%",
        "risk": f.risk_rating,
        "risk_class": _risk_class(f.risk_rating),
        "min_sip": f"₹{f.min_sip:,}",
        "expense": "—" if not f.expense_ratio else f"{f.expense_ratio:.2f}%",
        "house": f.fund_house,
        "category": f.category,
        "lockin": "No lock-in" if f.lockin_years == 0 else f"{f.lockin_years} years",
        "as_of": f.as_of,
        "isin": f.isin,
    }


def audit_log() -> list[dict]:
    return [e.__dict__ for e in _audit_log]
