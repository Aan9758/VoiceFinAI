"""Tests for intent_router.

Run: pytest test_intent_router.py -v
"""

from __future__ import annotations

import pytest

from intent_router import (
    Intent,
    IntentRouter,
    Risk,
    Session,
    normalize,
)

CATALOG = {
    "PARAG_FLEXI": ["parag parikh flexi cap", "parag parikh", "ppfas"],
    "HDFC_MIDCAP": ["hdfc mid cap opportunities", "hdfc midcap"],
    "SBI_SMALLCAP": ["sbi small cap", "sbi smallcap"],
}


@pytest.fixture
def router() -> IntentRouter:
    return IntentRouter(fund_catalog=CATALOG)


@pytest.fixture
def screen() -> Session:
    return Session(visible_slots=("F1", "F2", "F3"))


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected_substring",
    [
        ("Konsa achha hai?", "kaunsa"),
        ("KAUN SA lu", "kaun sa lena"),
        ("2 lac chahiye", "2 lakh chahiye"),
        ("₹5000", "rs 5000"),
        ("3 sal ke liye", "3 saal ke liye"),
    ],
)
def test_variant_folding(raw: str, expected_substring: str) -> None:
    assert expected_substring in normalize(raw)


def test_devanagari_reaches_same_space() -> None:
    assert "kaun" in normalize("कौन सा फंड")
    assert "fund" in normalize("कौन सा फंड")


def test_normalize_is_padded() -> None:
    out = normalize("hello")
    assert out.startswith(" ") and out.endswith(" ")


def test_empty_input_does_not_crash() -> None:
    assert normalize("") == " "


# --------------------------------------------------------------------------- #
# Tier 7b — why follow-up
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "utterance",
    ["why did you pick these", "inmein ye kyun hai", "in sab mein reason kya hai"],
)
def test_why_followup(router: IntentRouter, screen: Session, utterance: str) -> None:
    r = router.route(utterance, screen)
    assert r.intent is Intent.FUND_DETAIL
    assert r.why_query is True
    assert r.rule == "why_followup"


def test_why_without_reference_does_not_fire_7b(router: IntentRouter) -> None:
    r = router.route("why", Session())
    assert r.rule != "why_followup"


def test_why_targets_explicit_slot_over_focus(router: IntentRouter) -> None:
    s = Session(visible_slots=("F1", "F2", "F3"), focus_slot="F3")
    r = router.route("in me se F2 kyun", s)
    assert r.target_slots == ("F2",)


# --------------------------------------------------------------------------- #
# Tier 8 — detail on a specific slot
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "utterance,expected_slot",
    [
        ("F2 ke bare mein batao", "F2"),
        ("tell me about the second one", "F2"),
        ("teesra wala detail", "F3"),
        ("F1", "F1"),
    ],
)
def test_detail_on_slot(
    router: IntentRouter, screen: Session, utterance: str, expected_slot: str
) -> None:
    r = router.route(utterance, screen)
    assert r.intent is Intent.FUND_DETAIL
    assert r.target_slots == (expected_slot,)


def test_terse_utterance_with_slot_is_detail(router: IntentRouter, screen: Session) -> None:
    r = router.route("F2?", screen)
    assert r.intent is Intent.FUND_DETAIL
    assert r.rule == "detail_on_slot"


def test_focus_slot_is_updated(router: IntentRouter, screen: Session) -> None:
    router.route("F3 ke bare mein batao", screen)
    assert screen.focus_slot == "F3"


# --------------------------------------------------------------------------- #
# Tier 8b — best pick over on-screen funds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "utterance",
    [
        "inmein se kaunsa lu",
        "in me se konsa lu",
        "which will be the best for me",
        "out of these which one should i pick",
    ],
)
def test_best_pick(router: IntentRouter, screen: Session, utterance: str) -> None:
    r = router.route(utterance, screen)
    assert r.intent is Intent.FUND_DETAIL
    assert r.best_pick is True


def test_best_pick_falls_back_to_first_visible(router: IntentRouter) -> None:
    s = Session(visible_slots=("F4", "F5"))
    r = router.route("inmein se kaunsa lena chahiye", s)
    assert r.target_slots == ("F4",)


def test_best_pick_prefers_existing_focus(router: IntentRouter) -> None:
    s = Session(visible_slots=("F1", "F2"), focus_slot="F2")
    r = router.route("inmein se kaunsa acha", s)
    assert r.target_slots == ("F2",)


# --------------------------------------------------------------------------- #
# Named fund not on screen
# --------------------------------------------------------------------------- #


def test_named_offscreen_fund(router: IntentRouter) -> None:
    r = router.route("parag parikh flexi cap kaisa hai", Session())
    assert r.intent is Intent.FUND_DETAIL
    assert r.mentioned_fund == "PARAG_FLEXI"
    assert r.target_slots == ()


def test_unknown_catalog_yields_no_named_fund() -> None:
    r = IntentRouter().route("parag parikh kaisa hai", Session())
    assert r.mentioned_fund is None


# --------------------------------------------------------------------------- #
# Tier 9 — discovery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "utterance,risk",
    [
        ("mujhe safe fund chahiye", Risk.LOW),
        ("i want aggressive high return funds", Risk.HIGH),
        ("balanced type ka kuch dikhao", Risk.MODERATE),
    ],
)
def test_discover_from_risk(router: IntentRouter, utterance: str, risk: Risk) -> None:
    r = router.route(utterance, Session())
    assert r.intent is Intent.DISCOVER
    assert r.risk is risk


def test_no_risk_is_low_not_high(router: IntentRouter) -> None:
    """`risk nahi chahiye` contains 'risk' — must not read as high risk."""
    r = router.route("mujhe risk nahi chahiye", Session())
    assert r.risk is Risk.LOW


@pytest.mark.parametrize(
    "utterance,amount",
    [
        ("5000 invest karna hai", 5_000),
        ("50k lagana hai", 50_000),
        ("2 lakh ka plan", 200_000),
        ("rs 25000 monthly", 25_000),
    ],
)
def test_amount_extraction(router: IntentRouter, utterance: str, amount: int) -> None:
    r = router.route(utterance, Session())
    assert r.amount == amount


def test_small_bare_number_is_not_an_amount(router: IntentRouter, screen: Session) -> None:
    """'F2' style indices must not be read as money."""
    r = router.route("option 2 batao", screen)
    assert r.amount is None
    assert r.intent is Intent.FUND_DETAIL


@pytest.mark.parametrize(
    "utterance,months",
    [("3 saal ke liye", 36.0), ("18 mahina", 18.0), ("long term ke liye", 60.0)],
)
def test_horizon_extraction(router: IntentRouter, utterance: str, months: float) -> None:
    r = router.route(utterance, Session())
    assert r.horizon_months == pytest.approx(months)


def test_discover_clears_stale_slots(router: IntentRouter, screen: Session) -> None:
    screen.focus_slot = "F2"
    router.route("mujhe safe fund chahiye 2 lakh ka", screen)
    assert screen.visible_slots == ()
    assert screen.focus_slot is None


# --------------------------------------------------------------------------- #
# Tier 10 — clarify
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("utterance", ["hmm", "ok theek hai", "asdkjh", ""])
def test_clarify_fallback(router: IntentRouter, utterance: str) -> None:
    r = router.route(utterance, Session())
    assert r.intent is Intent.CLARIFY
    assert r.confidence == pytest.approx(0.45)


# --------------------------------------------------------------------------- #
# Session accumulation
# --------------------------------------------------------------------------- #


def test_profile_signals_accumulate(router: IntentRouter) -> None:
    s = Session()
    router.route("mujhe safe fund chahiye", s)
    router.route("2 lakh", s)
    router.route("5 saal ke liye", s)
    assert s.risk is Risk.LOW
    assert s.amount == 200_000
    assert s.horizon_months == pytest.approx(60.0)
    assert s.turn == 3


def test_priority_why_beats_advise(router: IntentRouter, screen: Session) -> None:
    """'kyun ye best hai' has both WHY and ADVISE tokens; 7b must win."""
    r = router.route("inmein ye kyun best hai", screen)
    assert r.why_query is True
    assert r.best_pick is False


def test_custom_rule_can_be_prepended(screen: Session) -> None:
    from intent_router import DEFAULT_RULES, Routing

    def always_clarify(f, s):
        return Routing(intent=Intent.CLARIFY, confidence=1.0, rule="custom")

    r = IntentRouter(rules=(always_clarify, *DEFAULT_RULES)).route("F2 batao", screen)
    assert r.rule == "custom"


# --------------------------------------------------------------------------- #
# Devanagari end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "utterance,intent",
    [
        ("मुझे सुरक्षित फंड चाहिए", Intent.DISCOVER),
        ("दो लाख निवेश करना है", Intent.DISCOVER),
        ("इनमें से कौन सा लूं", Intent.FUND_DETAIL),
    ],
)
def test_devanagari_routes(router: IntentRouter, screen: Session, utterance, intent) -> None:
    assert router.route(utterance, screen).intent is intent


def test_devanagari_amount(router: IntentRouter) -> None:
    assert router.route("दो लाख निवेश करना है", Session()).amount == 200_000


def test_number_word_only_folds_before_a_unit() -> None:
    assert " 2 lakh " in normalize("do lakh")
    assert " do " in normalize("do it now")
