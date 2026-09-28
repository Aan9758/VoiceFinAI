from __future__ import annotations

from core import dialogue, generate, pipeline, retrieval2, universe
from core.auditstream import AuditStream
from core.snapshot import FundRecommendation, Snapshot


def _fund(name: str, house: str) -> FundRecommendation:
    return FundRecommendation(
        fund_name=name,
        isin="INF000000000",
        category="Equity Scheme - Multi Cap Fund",
        nav=100.0,
        returns_1yr=8.0,
        returns_3yr=10.0,
        returns_5yr=12.0,
        risk_rating="Moderate",
        min_sip=500,
        fund_house=house,
        lockin_years=0,
        expense_ratio=0.0,
        as_of="2026-09-28",
        source="test fixture",
    )


def test_amc_name_is_parsed_from_discovery_request():
    turn = dialogue.parse_turn(
        "can you suggest me the sbi mutual funds", dialogue.Session("amc-parse")
    )
    assert turn["intent"] == dialogue.DISCOVER
    assert turn["fund_house"] == "SBI Mutual Fund"


def test_snapshot_never_returns_other_amcs_for_explicit_house(monkeypatch):
    records = [
        {"fund_name": "Axis Multi Cap", "fund_house": "Axis Mutual Fund", "appetite_bucket": "moderate", "scheme_code": 1, "risk_band": "Moderate", "min_sip": 500},
        {"fund_name": "SBI Multi Asset", "fund_house": "SBI Mutual Fund", "appetite_bucket": "moderate", "scheme_code": 2, "risk_band": "Moderate", "min_sip": 500},
        {"fund_name": "SBI Balanced", "fund_house": "SBI Mutual Fund", "appetite_bucket": "moderate", "scheme_code": 3, "risk_band": "Moderate", "min_sip": 500},
        {"fund_name": "Bank of India Flexi", "fund_house": "Bank of India Mutual Fund", "appetite_bucket": "moderate", "scheme_code": 4, "risk_band": "Moderate", "min_sip": 500},
    ]
    monkeypatch.setattr(universe, "load", lambda: records)
    monkeypatch.setattr(
        retrieval2,
        "_to_recommendation",
        lambda record: _fund(record["fund_name"], record["fund_house"]),
    )

    snapshot = retrieval2.build_snapshot(
        bucket="moderate", fund_house="SBI Mutual Fund", limit=3
    )

    assert snapshot is not None
    assert len(snapshot.funds) == 2
    assert {fund.fund_house for fund in snapshot.funds.values()} == {"SBI Mutual Fund"}


def test_explicit_house_with_no_matching_risk_returns_no_substitute(monkeypatch):
    monkeypatch.setattr(universe, "load", lambda: [
        {"fund_name": "SBI Multi Asset", "fund_house": "SBI Mutual Fund", "appetite_bucket": "moderate", "scheme_code": 2, "risk_band": "Moderate", "min_sip": 500},
        {"fund_name": "Axis Liquid", "fund_house": "Axis Mutual Fund", "appetite_bucket": "low", "scheme_code": 1, "risk_band": "Low", "min_sip": 500},
    ])
    monkeypatch.setattr(retrieval2, "_to_recommendation", lambda record: _fund(record["fund_name"], record["fund_house"]))

    assert retrieval2.build_snapshot(bucket="low", fund_house="SBI Mutual Fund") is None


def test_pipeline_passes_requested_house_as_hard_filter(monkeypatch):
    observed = {}
    snapshot = Snapshot([_fund("SBI Multi Asset", "SBI Mutual Fund")])

    def fake_build_snapshot(**kwargs):
        observed.update(kwargs)
        return snapshot

    monkeypatch.setattr(pipeline.retrieval2, "build_snapshot", fake_build_snapshot)
    monkeypatch.setattr(
        pipeline.generate,
        "generate_template",
        lambda *_args, **_kwargs: ("Axis ELSS - Tax Saver is better; I can show {F1}.", "test"),
    )
    dialogue.reset_session("amc-pipeline")

    result = pipeline.run_turn(
        "can you suggest me the sbi mutual funds", "amc-pipeline", lambda _text: "token"
    )

    assert observed["fund_house"] == "SBI Mutual Fund"
    assert {card["house"] for card in result["cards"]} == {"SBI Mutual Fund"}
    assert "Axis" not in result["voice_text"]
    assert "SBI Mutual Fund" in result["voice_text"]


def test_model_template_with_other_amc_is_rejected():
    assert pipeline._template_mentions_other_house(
        "I recommend Axis ELSS - Tax Saver.", "SBI Mutual Fund"
    )
    assert not pipeline._template_mentions_other_house(
        "I recommend {F1} from {F1.house}.", "SBI Mutual Fund"
    )


def test_why_followup_keeps_all_visible_funds_and_is_grounded(monkeypatch):
    session_id = "why-followup"
    dialogue.reset_session(session_id)
    funds = [
        _fund("SBI Large Cap", "SBI Mutual Fund"),
        _fund("SBI Balanced Advantage", "SBI Mutual Fund"),
    ]
    snapshot = Snapshot(funds, user_amount=10000, user_tenure="long term")
    dialogue.get_session(session_id).remember_snapshot(snapshot, is_list=True)
    monkeypatch.setattr(pipeline.retrieval2, "build_snapshot", lambda **_kwargs: snapshot)
    turn = dialogue.parse_turn("Why did you choose these funds?", dialogue.get_session(session_id))
    assert turn["intent"] == dialogue.FUND_DETAIL
    assert turn["why_query"] is True
    assert turn["target_slots"] == ["F1", "F2"]

    result = pipeline.run_turn(
        "Why did you choose these funds?", session_id, lambda _text: "token"
    )
    assert result["type"] == "recommendation"
    assert len(result["cards"]) == 2
    assert "not a personal ranking" in result["voice_text"]


def test_compound_metric_question_keeps_risk_and_sip_fields():
    session = dialogue.Session("compound-metric")
    session.last_slots = {"F1": _fund("SBI Large Cap", "SBI Mutual Fund")}
    session.last_list = dict(session.last_slots)
    session.focus_slot = "F1"
    turn = dialogue.parse_turn("What is the risk and minimum SIP?", session)
    assert turn["intent"] == dialogue.METRIC
    assert set(turn["metric_fields"]) == {"risk", "minsip"}


def test_safety_question_about_current_fund_is_not_new_discovery():
    session = dialogue.Session("fund-safety")
    session.last_slots = {"F1": _fund("SBI Large Cap", "SBI Mutual Fund")}
    session.last_list = dict(session.last_slots)
    session.focus_slot = "F1"
    turn = dialogue.parse_turn("Is this a safe fund?", session)
    assert turn["intent"] == dialogue.METRIC
    assert turn["metric_field"] == "risk"
    assert turn["target_slots"] == ["F1"]


def test_devanagari_request_prefers_hinglish_reply_language():
    assert dialogue.detect_language(
        "मुझे long term के लिए mutual fund suggest करो"
    ) == "hinglish"


def test_generic_horizon_phrase_is_not_mistaken_for_a_named_fund(monkeypatch):
    record = {
        "fund_name": "Edelweiss Ultra Short Term Fund - Direct Plan - Growth",
        "fund_house": "Edelweiss Mutual Fund",
    }
    monkeypatch.setattr(universe, "load", lambda: [record])

    assert universe.find_by_name("I need a long term mutual fund") is None
    assert universe.find_by_name("Tell me about Edelweiss Ultra Short Term Fund") == record


def test_discovery_reply_is_factual_and_audit_safe():
    snapshot = Snapshot([
        _fund("SBI Large Cap", "SBI Mutual Fund"),
        _fund("SBI Balanced Advantage", "SBI Mutual Fund"),
    ])
    session = dialogue.Session("factual-discovery")
    turn = {"intent": dialogue.DISCOVER, "language": "english", "risk": "moderate"}

    template, source = generate.generate_template(turn, snapshot, session)

    assert source == "factual_discovery"
    assert "best" not in template.casefold()
    assert "suitable" not in template.casefold()
    assert AuditStream(snapshot).validate(template).passed


def test_hinglish_car_goal_captures_target_and_monthly_sip():
    session = dialogue.Session("car-goal-hinglish")
    turn = dialogue.parse_turn(
        "मेरे को एक long term horizon के लिए invest करना है. मेरे को car खरीदनी है "
        "जो कि है दस लाख की और मैं हर month 20000 invest करना चाहता हूं.",
        session,
    )

    assert turn["intent"] == dialogue.GOAL
    assert turn["goal_amount"] == 1_000_000
    assert turn["goal_monthly_sip"] == 20_000
    assert turn["goal_years"] is None
    assert turn["goal_purpose"] == "Buying a Car"


def test_goal_followup_keeps_car_context_and_finds_timeline_sip():
    session = dialogue.Session("car-goal-followup")
    session.goal_amount = 1_000_000
    session.goal_purpose = "Buying a Car"

    turn = dialogue.parse_turn(
        "तो मैं यह वाला fund ले सकता हूँ और कितने साल के लिए invest करना पड़ेगा "
        "10000 rupees a month के हिसाब से जो कि मेरा 10 lakh हो जाए?",
        session,
    )

    assert turn["intent"] == dialogue.GOAL
    assert turn["goal_amount"] == 1_000_000
    assert turn["goal_monthly_sip"] == 10_000
    assert turn["goal_years"] is None
    assert turn["goal_purpose"] == "Buying a Car"


def test_months_to_goal_and_goal_reply_are_consistent_and_audited():
    months = pipeline._calc_months_to_goal(1_000_000, 20_000, 14.0)
    assert months == 40

    snapshot = Snapshot(
        [_fund("SBI Large Cap", "SBI Mutual Fund")],
        user_goal_amount=1_000_000,
        user_goal_years=4,
        user_goal_sip=20_000,
    )
    session = dialogue.Session("goal-template")
    session.goal_purpose = "Buying a Car"
    template, source = generate.generate_template(
        {
            "intent": dialogue.GOAL,
            "language": "hinglish",
            "goal_purpose": "Buying a Car",
            "goal_plan_mode": "time_to_goal",
        },
        snapshot,
        session,
    )

    assert source == "goal_plan"
    assert "car" in template.casefold()
    assert AuditStream(snapshot).validate(template).passed


def test_competition_word_routes_to_a_side_by_side_comparison():
    session = dialogue.Session("competition-phrase")
    session.last_slots = {
        "F1": _fund("BANK OF INDIA LARGE CAP", "Bank of India Mutual Fund"),
        "F2": _fund("Sundaram Large Cap", "Sundaram Mutual Fund"),
    }
    session.last_list = dict(session.last_slots)
    session.focus_slot = "F1"

    turn = dialogue.parse_turn("Sundaram Large Cap fund ka competition show karo", session)

    assert turn["intent"] == dialogue.COMPARE
    assert turn["target_slots"] == ["F1", "F2"]


def test_named_fund_comparison_does_not_filter_out_the_other_amc(monkeypatch):
    session_id = "comparison-house-scope"
    dialogue.reset_session(session_id)
    session = dialogue.get_session(session_id)
    visible = Snapshot([
        _fund("Axis Large Cap", "Axis Mutual Fund"),
        _fund("Mirae Asset Large Cap", "Mirae Asset Mutual Fund"),
    ])
    session.remember_snapshot(visible, is_list=True)
    session.focus_slot = "F1"
    observed = {}

    def fake_snapshot(**kwargs):
        observed.update(kwargs)
        return visible

    monkeypatch.setattr(pipeline.retrieval2, "build_snapshot", fake_snapshot)
    result = pipeline.run_turn(
        "Mirae Asset Large Cap fund ka competition show karo",
        session_id,
        lambda _text: "token",
    )

    assert observed["fund_house"] is None
    assert [card["name"] for card in result["cards"]] == [
        "Axis Large Cap", "Mirae Asset Large Cap"
    ]
