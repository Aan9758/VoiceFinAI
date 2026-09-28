"""
core/generate.py — The Language Layer
======================================
This is the piece the project never had.

The architecture doc describes Phi-3 Mini + LoRA + Medusa generating language
while the database supplies facts. In the shipped code there is no model of any
kind — app.py holds six hand-written f-strings and picks one by (risk, amount).
Six possible sentences is the hard ceiling on variety, which is why every query
produced the same reply.

This module restores the design without requiring a fine-tuned model:

  - The model receives snapshot.build_context_string() — slot names only, never
    a number, never a fund name.
  - It returns a TEMPLATE containing placeholders.
  - AuditStream validates that template. Now validation is meaningful: it is
    checking output that something else generated, not a string the same file
    wrote. A model that writes "12.3%" instead of "{F1.cagr3}" gets caught by
    the BARE_DIGIT rule and the turn is regenerated or dropped to the bank.

Set ANTHROPIC_API_KEY to enable the model path. With no key, the deterministic
bank runs — still varied, because it rotates per intent, per risk and per turn
instead of collapsing to one string.

Swapping in Phi-3 + LoRA later means replacing _call_model() only. Everything
around it — context building, slot discipline, validation, rendering — is
already model-agnostic.
"""

from __future__ import annotations

import json
import os
import random
import re
import ssl
import time
import urllib.request
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

_BARE_DIGIT = re.compile(r"\b\d+(?:\.\d+)?\b")
_ALLOWED_CARDINALS = frozenset(str(i) for i in range(1, 11))
_PLACEHOLDER = re.compile(r"\{([A-Z]\d*(?:\.[a-z0-9_]+)?|U\.[a-z0-9_]+)\}")


SYSTEM_PROMPT = """You are Mridul, a polite, professional, and knowledgeable male Indian mutual fund advisor.
Tone: fast, natural, courteous, respectful conversational Hinglish.
Always address the user with respect using 'Aap', 'Aapka', 'Aapke', 'bataiye', 'dekhiye', 'pooch sakte hain'.
If the user's name is known from context, you may address them warmly as '<Name> ji'.
NEVER use 'tu', 'tera', 'teri', 'pooch', 'bol bas', or dismissive words.
Use male grammatical forms: 'kar sakta hoon', 'bata deta hoon', 'recommend karunga', 'dikhata hoon', 'dekh raha hoon'.

You write ONE short spoken reply — two or three sentences, MAX 40 words.
You are speaking out loud — no bullet points, no markdown, no lists, no formal jargon.

ABSOLUTE RULE — you have no access to any financial figure.
Every fund name, return, NAV, expense ratio, minimum SIP, risk rating and amount must come from placeholder slots in the context.
Never write a fund name. Never write a digit. Never write a percentage.

Do NOT echo the fund data table. That is input context, not your reply.
Only use slots present in the context. If only {F1} exists, do not reference {F2} or {F3}.

Example of correct output:
  Aap {F1} dekh sakte hain — teen saal mein {F1.cagr3} return diya hai, risk {F1.risk} hai. Ye ek solid fund hai.

Example of WRONG output:
  F1: {F1} | 3yr: {F1.cagr3} | risk: {F1.risk}
  Quant Small Cap ka 3-year return 18.5% raha hai.

If a slot is missing, skip it — don't guess.
Write natural, respectful, conversational Hinglish in Latin script.
Output only the reply text — no preamble, no quotes."""


SYSTEM_PROMPT_EN = """You are Mridul, a male AI financial voice assistant for Indian mutual funds.
If the user's name is known, you may politely address them by name.

You write ONE short spoken reply — two or three sentences, under 55 words.
You are speaking out loud, so no bullet points, no markdown, no lists.

ABSOLUTE RULE — you have no access to any financial figure.
Every fund name, return, NAV, expense ratio, minimum SIP, risk rating and
amount must be written as a placeholder slot from the context below.
Never write a fund name. Never write a digit. Never write a percentage.

Do NOT repeat or echo the fund data table. That is input context, not your reply.
Your reply must always be spoken natural sentences — never a table, never pipe-separated lines.

Only use slots that appear in the context you are given. If only {F1} is available,
do not reference {F2} or {F3} — those funds do not exist in this turn's context.

Example of correct output (single-fund context):
  {F1} has delivered a 3-year return of {F1.cagr3}, with risk rating {F1.risk}.

Example of WRONG output (hallucinating slots not in context):
  F1: {F1} | 3yr: {F1.cagr3} | risk: {F1.risk}
  Quant Small Cap delivered 18.5% over 3 years.

If you cannot express something with an available slot, leave it out entirely.
Write clear, professional, spoken English as an Indian wealth advisor speaks.
Vary your phrasing between turns — do not reuse the opening of your last reply.
Output only the reply text. No preamble, no quotes, no explanation."""


INTENT_BRIEF = {
    "discover": "Present the available options factually. Do not rank them, call one best, or claim personal suitability. State only category, risk, and historical-return slots, and remind the user that historical returns are not forecasts.",
    "fund_detail": "The user asked about one specific fund or why it was chosen. Talk about that one fund only, using its slots. Do not list the others.",
    "compare": "The user asked to compare. Contrast the two funds in the context on return and risk, then say plainly which suits which kind of investor.",
    "metric": "The user asked one narrow question. Answer that one thing in a sentence, then offer one relevant follow-up.",
    "confirm_sip": "The user wants to start a SIP. Read back the fund and the amount from the slots and ask for a yes/no confirmation. Nothing else.",
    "graph": "The user asked to see or understand the performance and wealth accumulation chart. Explain the compounding trajectory, invested capital versus wealth gain shown on screen in 2 short spoken sentences. Under 45 words.",
    "clarify": "You do not have enough to go on. Ask exactly ONE short question — equity ya debt, ya kitna monthly. Do not recommend anything.",
    "out_of_scope": "Outside mutual funds. Say so warmly in one line and offer what you can help with.",
    "greeting": "Greet briefly and ask what they are looking for. One line.",
    "goal": "The user is planning for a financial goal (target amount and tenure). Explain the recommended monthly SIP required to achieve their target in 2 spoken sentences. Mention the balanced fund {F1} and required monthly SIP.",
}


# ── Deterministic bank — used when no model key is present ───────────────────
# Multiple variants per (intent, risk). Rotation is what kills the repetition.

BANK_HI: dict[str, list[str]] = {
    "discover:low": [
        "Safe investment ke liye {F1} sabse badhiya rahega — risk {F1.risk} hai aur return {F1.cagr3}. {F2} aur {F3} bhi stable options hain.",
        "{F1} conservative investment ke liye solid choice hai. Minimum SIP {F1.minsip} se start hoti hai, risk {F1.risk} hai. Saath mein {F2} bhi dekh sakte hain.",
        "Capital safe rakhni hai toh {F1} dekhiye — {F1.category} category, {F1.cagr3} return. {F2} aur {F3} bhi screen par hain.",
    ],
    "discover:moderate": [
        "{F1} ek balanced option hai — teen saal mein {F1.cagr3} return aur risk {F1.risk}. {F2} aur {F3} bhi similar range mein hain.",
        "Steady growth ke liye {F1} achha rahega — paanch saal mein {F1.cagr5} diya hai. {F2} bhi consider kar sakte hain.",
        "Main {F1} recommend karunga — {F1.house} ka established fund hai, SIP {F1.minsip} se shuru hoti hai. {F2} aur {F3} achhe alternatives hain.",
    ],
    "discover:high": [
        "High growth ke liye {F1} lead option hai — teen saal mein {F1.cagr3} return hai. Risk {F1.risk} hai toh long term ke liye hi invest karein. {F2} bhi strong fund hai.",
        "{F1} ka performance kafi strong raha hai, paanch saal mein {F1.cagr5}. Risk {F1.risk} hai, market volatility dhyan mein rakhein. {F2} aur {F3} bhi isi category mein hain.",
        "Agar aggressive growth chahiye toh {F1} dekhiye. Minimum SIP {F1.minsip} se, risk {F1.risk} hai.",
    ],
    "fund_detail": [
        "{U.amount} monthly investment ke liye {F1} sabse suitable lag raha hai — teen saal ka return {F1.cagr3}, paanch saal {F1.cagr5}, risk {F1.risk}.",
        "{F1} ko isliye recommend kiya hai kyunki paanch saal ka track record {F1.cagr5} raha hai aur ye {F1.house} ka established fund hai.",
        "Inme {F1} lead fund hai — teen saal ka return {F1.cagr3}, risk {F1.risk}. Minimum SIP {F1.minsip} se shuru hoti hai.",
        "{F1} — {F1.house} ka fund hai. Teen saal mein {F1.cagr3}, paanch saal {F1.cagr5}, risk {F1.risk}. Minimum SIP {F1.minsip}.",
        "{F1} ka current NAV {F1.nav} hai. Ek saal ka return {F1.cagr1}, teen saal {F1.cagr3}. Risk {F1.risk} hai.",
        "{F1} ne paanch saal mein {F1.cagr5} aur teen saal mein {F1.cagr3} return diya hai. Risk {F1.risk} hai, aur {F1.lockin} hai.",
    ],
    "graph": [
        "{F1} ka SIP growth graph screen par hai. Teen saal ka return {F1.cagr3} aur paanch saal ka {F1.cagr5} CAGR raha hai.",
        "Screen par {F1} ka growth projection chart dekh sakte hain — {U.amount} monthly SIP ka long term compounding dikhaya gaya hai.",
        "{F1} ka historical return aur wealth accumulation graph aapke saamne hai — paanch saal mein {F1.cagr5} return diya hai.",
    ],
    "wealth_explain": [
        "Gray hissa aapka invested capital hai, aur green hissa {F1} ke compounding returns se bana munafa hai. {U.amount} monthly invest karte rahein, toh das saal mein gain invested capital se bhi upar nikal jaata hai.",
        "Calculation compounding formula par based hai — {U.amount} monthly SIP par {F1} ka CAGR compound hota hai. Samay ke saath green bar tezi se badhta hai.",
        "Green section aapka total wealth gain hai — {U.amount} monthly SIP lagane se, compounding ki wajah se paanch aur das saal mein return kafi badh jaata hai.",
        "Total corpus dikhata hai ye graph — {F1} mein lagaya paisa invested capital hai, aur green hissa compounded returns hain. Lamba samay dene par compounding ka fayda kayi guna ho jata hai.",
        "Simple calculation hai — {F1} mein {U.amount} monthly investment par compounding chalti rehti hai. Green hissa wahi gain hai jo samay ke saath tezi se badhta hai.",
    ],
    "compare": [
        "{F1} ka teen saal ka return {F1.cagr3} hai aur {F2} ka {F2.cagr3}. Risk mein {F1} {F1.risk} hai jabki {F2} {F2.risk}. Stability ke liye doosra behtar hai.",
        "Dono mein farak ye hai — {F1} {F1.category} mein {F1.cagr3}, jabki {F2} {F2.category} mein {F2.cagr3}. Long term ke liye pehla option strong hai.",
        "Teen saal mein {F1} ne {F1.cagr3} diya aur {F2} ne {F2.cagr3}. Paanch saal mein {F1.cagr5} banaam {F2.cagr5}. Risk {F1.risk} aur {F2.risk} hai.",
    ],
    "confirm_sip": [
        "Toh {U.amount} monthly {F1} mein. Risk rating {F1.risk} hai aur minimum SIP {F1.minsip}. Confirm karein?",
        "{F1} mein {U.amount} ki monthly SIP set kar raha hoon. Kya aap proceed karna chahte hain?",
        # amount-free variants — used when the user never said a figure
        "{F1} mein SIP shuru karte hain. Minimum investment {F1.minsip} se start hoti hai — monthly kitna amount rakhein?",
        "Theek hai, selected fund {F1} hai — risk {F1.risk}. Aap monthly kitna amount invest karna chahenge?",
    ],
    "greeting": [
        "Namaste! Main Madhur hoon, aapka financial advisor. Bataiye, kya plan hai — SIP, lumpsum, ya goal-based investment?",
        "Namaste! Main Madhur hoon. Mutual funds mein aaj main aapki kya madad kar sakta hoon?",
    ],
    "out_of_scope": [
        "Maaf kijiye, main sirf mutual funds aur SIP ke baare mein guide kar sakta hoon.",
        "Is par main guide nahi kar paunga — mutual funds ya SIP ke baare mein poochiye, main poori madad karunga.",
    ],
    "goal": [
        "{U.goal_years} saal mein {U.goal_amount} ke target ke liye {F1} mein lagbhag {U.goal_sip} monthly SIP karni hogi. Screen par teen strategies aur step-up option dekh sakte hain.",
        "{U.goal_amount} ke target ke liye {F1} solid choice hai — isme lagbhag {U.goal_sip} monthly chahiye. Das percent yearly step-up se shuruat kam amount se bhi ho sakti hai.",
        "{F1} mein {U.goal_sip} monthly SIP se {U.goal_years} saal mein {U.goal_amount} ka corpus plan kar sakte hain. Screen par projections calculate kiye hain.",
    ],
    "language_switch": [
        "Theek hai, main ab Hinglish mein baat karunga. Bataiye, kis tarah ka investment dekh rahe hain?",
        "Hinglish mein continue karte hain. Aap mutual funds ke baare mein kya poochna chahte hain?",
    ],
}

BANK = BANK_HI  # Backward-compatible alias

BANK_EN: dict[str, list[str]] = {
    "discover:low": [
        "For capital protection, consider {F1} — risk is {F1.risk} with a 3-year return of {F1.cagr3}. {F2} and {F3} are also stable options.",
        "{F1} looks most suitable for conservative investing. Minimum SIP starts at {F1.minsip}, with {F1.risk} risk. You can also evaluate {F2}.",
        "Among these three, {F1} offers the highest stability in the {F1.category} category with a return of {F1.cagr3}. The other two are {F2} and {F3}.",
    ],
    "discover:moderate": [
        "{F1} is a well-balanced choice offering a 3-year return of {F1.cagr3} and {F1.risk} risk. {F2} and {F3} are in a similar range.",
        "If you want moderate risk with steady growth, {F1} is a solid pick. Its 5-year return stands at {F1.cagr5}. You may also look at {F2}.",
        "I would recommend {F1} from {F1.house}, with minimum SIP starting at {F1.minsip}. Reliable alternatives include {F2} and {F3}.",
    ],
    "discover:high": [
        "For higher growth potential, {F1} has delivered a 3-year return of {F1.cagr3}, but carries {F1.risk} risk. Best suited for long term horizons. {F2} is another strong option.",
        "{F1} has demonstrated strong momentum with a 5-year return of {F1.cagr5}. Risk is {F1.risk}, so expect market volatility. {F2} and {F3} are comparable.",
        "For aggressive wealth creation, {F1} is the lead option. Minimum SIP starts at {F1.minsip}. Please note the {F1.risk} risk rating before investing.",
    ],
    "fund_detail": [
        "For your monthly investment of {U.amount}, {F1} appears to be the strongest choice — 3-year return {F1.cagr3}, 5-year return {F1.cagr5}, and risk rating {F1.risk}.",
        "{F1} is recommended for its established 5-year track record of {F1.cagr5} and reputable management by {F1.house}.",
        "Among the selected funds, {F1} is the lead option with a 3-year return of {F1.cagr3} and {F1.risk} risk. Minimum SIP starts at {F1.minsip}.",
        "{F1} is managed by {F1.house}. Its 3-year return is {F1.cagr3}, 5-year return is {F1.cagr5}, and risk is {F1.risk}. Minimum SIP is {F1.minsip}.",
        "Current NAV for {F1} stands at {F1.nav}. 1-year return is {F1.cagr1}, 3-year return is {F1.cagr3}, with risk classified as {F1.risk}.",
        "{F1} has generated {F1.cagr5} over 5 years and {F1.cagr3} over 3 years. It has a {F1.risk} risk rating and {F1.lockin}.",
    ],
    "graph": [
        "The performance and SIP growth chart for {F1} is now on screen. It has delivered a CAGR of {F1.cagr3} over 3 years and {F1.cagr5} over 5 years.",
        "You can see the projected wealth accumulation on screen — showing estimated long-term compounding for a monthly SIP of {U.amount}.",
        "Here is the historical performance and wealth compounding chart for {F1}, delivering {F1.cagr5} over 5 years.",
    ],
    "wealth_explain": [
        "This chart illustrates total wealth accumulation from a monthly SIP of {U.amount}. The gray area represents your invested capital, while the green area highlights compound gains from {F1}. Over 10 years, compounding expands wealth gains significantly beyond total investment.",
        "The projection is based on monthly compounding where {F1}'s CAGR compounds on every installment of {U.amount}. Over time, the green gains bar accelerates exponentially.",
        "The green section represents your accumulated wealth gain from investing {U.amount} monthly. Compounding returns in {F1} cause the gains to surpass invested capital across 5 and 10 years.",
        "Wealth accumulation reflects your total corpus generated through disciplined SIPs and compound interest. Capital invested in {F1} is represented in gray, while returns compound exponentially in green over time.",
        "This graph highlights total accumulated corpus. Your contributions into {F1} form the invested base, while the green bar represents compounded wealth that multiplies over long horizons.",
    ],
    "compare": [
        "{F1} has a 3-year return of {F1.cagr3} compared to {F2.cagr3} for {F2}. In terms of risk, {F1} is rated {F1.risk} while {F2} is {F2.risk}. For lower volatility, the second is preferable.",
        "The primary difference lies in category and returns — {F1} is in {F1.category} with {F1.cagr3}, while {F2} is in {F2.category} with {F2.cagr3}. For long horizons, the first is attractive.",
        "Over 3 years, {F1} delivered {F1.cagr3} versus {F2.cagr3} for {F2}. Over 5 years, the comparison is {F1.cagr5} against {F2.cagr5}. Both have risk profiles of {F1.risk} and {F2.risk}.",
    ],
    "confirm_sip": [
        "Setting up a monthly investment of {U.amount} in {F1}. Risk is {F1.risk} and minimum SIP is {F1.minsip}. Shall I confirm?",
        "Ready to start your monthly SIP of {U.amount} in {F1}. Would you like to proceed?",
        "Let's begin your SIP in {F1}. Minimum investment starts at {F1.minsip} — what monthly amount would you prefer?",
        "Understood, selected fund is {F1} with risk rating {F1.risk}. What monthly amount would you like to invest?",
    ],
    "greeting": [
        "Hello! I am Madhur, your AI wealth advisor. What kind of mutual fund investment are you exploring today?",
        "Welcome! I am Madhur, your AI wealth advisor. Are you looking to start a SIP or a lumpsum investment today?",
    ],
    "out_of_scope": [
        "That falls outside my scope. I specialize exclusively in mutual funds and SIP investments.",
        "I can only advise on Indian mutual funds and SIPs. Please feel free to ask about fund options, returns, or goals.",
    ],
    "goal": [
        "To reach your target of {U.goal_amount} in {U.goal_years}, a balanced fund like {F1} would require approximately {U.goal_sip} monthly SIP. 3 tailored strategies are displayed on your screen.",
        "For your goal of {U.goal_amount}, {F1} is a suitable choice requiring an estimated monthly SIP of {U.goal_sip}. A 10 percent annual step-up allows you to start with a lower contribution.",
        "Investing approximately {U.goal_sip} monthly in {F1} can help you achieve your corpus of {U.goal_amount} in {U.goal_years}. Detailed projections are shown on screen.",
    ],
    "language_switch": [
        "Switched to English. How can I assist you with your investments?",
        "I will speak in English now. What kind of mutual funds are you looking for?",
    ],
}

# Context-aware clarify bank — never asks for risk or amount if already known
CLARIFY_BANK = {
    # On-screen funds already exist — guide user on the displayed funds
    "has_cards": [
        "Ye funds aapke screen par hain. Kisi fund ki detail dekhna chahenge ya compare karein?",
        "Aap kisi bhi fund ka return, risk ya SIP details pooch sakte hain — main bata deta hoon.",
        "Inme se kaunse fund ke baare mein aap aur detail mein samajhna chahte hain?",
    ],
    # Risk is known, monthly amount is missing
    "need_amount": [
        "Har mahine kitna invest karne ka plan hai? Bataiye, options finalize karte hain.",
        "Monthly SIP mein kitna amount lagana chahenge — jaise ek hazaar, do hazaar ya paanch hazaar?",
        "Monthly SIP amount bataiye, hum investment options finalize kar dete hain.",
    ],
    # Amount is known, risk is missing
    "need_risk": [
        "Safe funds dekhna chahte hain ya behtar return ke liye thoda risk le sakte hain?",
        "Equity funds mein invest karna hai growth ke liye, ya safe debt options?",
        "Aapka risk profile kaisa hai — safe, balanced, ya high growth?",
    ],
    # Neither risk nor amount is known
    "need_both": [
        "Thoda aur bataiye — equity funds chahiye ya safe debt options? Us hisaab se options dikhata hoon.",
        "Safe rakhna hai ya acche return ke liye thoda risk le sakte hain?",
        "Monthly kitna invest karna chahte hain, aur kis category ka fund pasand karenge?",
    ],
}

CLARIFY_BANK_EN = {
    "has_cards": [
        "These funds are on your screen. Would you like more details on any fund, or should we compare them?",
        "You can ask about any fund's returns, risk, expense ratio, or ask me to start a SIP.",
        "Which of these funds would you like to explore in more detail?",
    ],
    "need_amount": [
        "What is your target monthly investment? Based on that, we can finalize your fund options.",
        "How much would you like to invest in monthly SIP — for instance, one thousand, two thousand, or five thousand?",
        "Please specify your monthly SIP amount so we can structure your portfolio.",
    ],
    "need_risk": [
        "Would you prefer conservative capital protection, or are you comfortable with risk for higher returns?",
        "Are you looking for growth equity funds or safe debt options?",
        "What is your risk appetite — low risk, balanced, or high growth?",
    ],
    "need_both": [
        "Could you share a bit more — are you looking for equity funds or safe debt options, and what is your monthly budget?",
        "Would you prefer capital preservation or higher growth with some market risk?",
        "What is your intended monthly SIP amount, and which fund category do you prefer?",
    ],
}

_METRIC_SENTENCE = {
    "cagr1": "{F1} ka 1-year return {F1.cagr1} raha hai.",
    "cagr3": "{F1} ka 3-year return {F1.cagr3} hai.",
    "cagr5": "{F1} ne 5 saal mein {F1.cagr5} diya hai.",
    "nav": "{F1} ka current NAV {F1.nav} hai.",
    "expense": "Expense ratio abhi mere paas verified nahi hai — wo AMC factsheet se aata hai. Baaki {F1} ka 3-year return {F1.cagr3} hai.",
    "minsip": "{F1} mein minimum SIP {F1.minsip} se shuru hoti hai.",
    "risk": "{F1} ka risk rating {F1.risk} hai.",
    "lockin": "{F1} mein {F1.lockin} hai.",
    "house": "{F1} {F1.house} ka fund hai.",
    "category": "{F1} {F1.category} category mein aata hai.",
}
_METRIC_TAIL = [
    " Aur kuch jaanna hai iske baare mein?",
    " Doosre funds se compare karke dikhaun?",
    " Iska SIP shuru karna chahenge?",
]

_METRIC_SENTENCE_EN = {
    "cagr1": "{F1} delivered a 1-year return of {F1.cagr1}.",
    "cagr3": "{F1} has a 3-year return of {F1.cagr3}.",
    "cagr5": "{F1} delivered {F1.cagr5} over 5 years.",
    "nav": "Current NAV for {F1} is {F1.nav}.",
    "expense": "I don't have verified expense-ratio data for {F1} in this demo. Please check the latest AMC factsheet.",
    "minsip": "Minimum SIP for {F1} starts at {F1.minsip}.",
    "risk": "Risk rating for {F1} is {F1.risk}.",
    "lockin": "{F1} has {F1.lockin}.",
    "house": "{F1} is managed by {F1.house}.",
    "category": "{F1} belongs to the {F1.category} category.",
}
_METRIC_TAIL_EN = [
    " Would you like to know more about this fund?",
    " Shall I compare this with other funds on screen?",
    " Would you like to start a SIP in this fund?",
]


def _bank_key(intent: str, risk: Optional[str]) -> str:
    if intent == "discover":
        return f"discover:{risk or 'moderate'}"
    return intent


def from_bank(turn: dict, session) -> str:
    """Pick a template from the bank, avoiding the last two used."""
    intent = turn["intent"]
    lang = turn.get("language") or getattr(session, "language", "hinglish")
    if lang == "auto":
        lang = "hinglish"

    active_bank = BANK_EN if lang == "english" else BANK_HI
    active_clarify = CLARIFY_BANK_EN if lang == "english" else CLARIFY_BANK
    active_metric_sent = _METRIC_SENTENCE_EN if lang == "english" else _METRIC_SENTENCE
    active_metric_tail = _METRIC_TAIL_EN if lang == "english" else _METRIC_TAIL

    if intent == "greeting":
        user_name = getattr(session, "user_name", None)
        if lang == "english":
            if user_name:
                return f"Hello {user_name}! I am Mridul, your financial advisor. How can I assist you with your mutual funds or SIP investments today?"
            return "Hello! I am Mridul, your financial advisor. How can I assist you with your mutual funds or SIP investments today?"
        else:
            if user_name:
                return f"Namaste {user_name} ji! Main Mridul hoon, aapka financial advisor. Bataiye, mutual funds ya SIP mein aaj main aapki kya madad kar sakta hoon?"
            return "Namaste! Main Mridul hoon, aapka financial advisor. Bataiye, mutual funds ya SIP mein aaj main aapki kya madad kar sakta hoon?"

    if intent == "metric":
        base = active_metric_sent.get(turn.get("metric_field") or "cagr3",
                                      active_metric_sent["cagr3"])
        tail = active_metric_tail[session.turn % len(active_metric_tail)]
        tpl = base + tail
    elif intent == "clarify":
        has_cards = bool(session.last_slots or session.last_list)
        has_risk = bool(session.risk or turn.get("risk"))
        has_amount = bool(session.amount or turn.get("amount"))

        if has_cards:
            options = active_clarify["has_cards"]
        elif has_risk and not has_amount:
            options = active_clarify["need_amount"]
        elif has_amount and not has_risk:
            options = active_clarify["need_risk"]
        else:
            options = active_clarify["need_both"]

        fresh = [o for o in options if o not in session.said_templates[-2:]]
        tpl = random.choice(fresh or options)
    else:
        if turn.get("explain_wealth"):
            options = active_bank.get("wealth_explain", [])
        else:
            options = active_bank.get(_bank_key(intent, turn.get("risk"))) or active_clarify["need_both"]

        # A template referencing {U.amount} cannot render if neither turn nor session has an amount
        effective_amount = turn.get("amount") or getattr(session, "amount", None)
        if not effective_amount:
            usable = [o for o in options if "{U.amount}" not in o]
            if usable:
                options = usable
        elif intent == "confirm_sip" or turn.get("explain_wealth"):
            usable = [o for o in options if "{U.amount}" in o]
            if usable:
                options = usable

        # A template referencing {U.tenure} cannot render if session has no horizon
        if not getattr(session, "horizon", None):
            usable = [o for o in options if "{U.tenure}" not in o]
            if usable:
                options = usable

        fresh = [o for o in options if o not in session.said_templates[-2:]]
        tpl = random.choice(fresh or options)

    # Add amount context prefix if the chosen template doesn't already have one
    if turn.get("amount") and intent == "discover" and "{U.amount}" not in tpl:
        if lang == "english":
            tpl = "For your monthly investment of {U.amount} — " + tpl
        else:
            tpl = "Aapke {U.amount} monthly ke hisaab se — " + tpl

    # Append the FINAL (possibly prefixed) template so dedup tracks what was heard
    session.said_templates.append(tpl)
    return tpl



# ── Local model path (Phase 4 — base inference, not fine-tuned) ──────────────
# Uses microsoft/Phi-3-mini-4k-instruct from HuggingFace.
# This is the BASE model — no LoRA adapter, no domain-specific training.
# Do NOT describe this as "fine-tuned" or "domain-confined" anywhere.
# It only generates slot templates (never actual numbers) and the same
# AuditStream validation gate applies before any output reaches the user.

LOCAL_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None
_LOCAL_LOAD_FAILED = False   # set True if import/load fails so we don't retry


def _load_local_model():
    """
    Lazy-load Phi-3-mini once. Returns (tokenizer, model) or (None, None).
    Silently returns (None, None) if transformers is not installed.
    NOT fine-tuned — base model only.
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_LOAD_FAILED
    if _LOCAL_LOAD_FAILED:
        return None, None
    if _LOCAL_MODEL is not None:
        return _LOCAL_TOKENIZER, _LOCAL_MODEL
    try:
        # Windows SSL cert bypass — same issue as mfapi.in/Deepgram
        import os as _os
        _os.environ.setdefault("CURL_CA_BUNDLE", "")
        _os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
        _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Patch requests for HuggingFace hub SSL (must be before any HF import)
        import requests as _req, urllib3 as _u3
        _u3.disable_warnings(_u3.exceptions.InsecureRequestWarning)
        _orig = _req.Session.send
        def _no_verify(self, request, **kwargs):
            kwargs["verify"] = False
            return _orig(self, request, **kwargs)
        _req.Session.send = _no_verify

        # ── DynamicCache compatibility shim ──────────────────────────────
        # transformers 4.57 removed several DynamicCache methods that
        # Phi-3's bundled modeling_phi3.py (written for ~4.36) still calls.
        # Shim the full set at once so we don't hit one missing method at a time.
        # All implementations match the original transformers 4.36 source exactly.
        try:
            from transformers.cache_utils import DynamicCache

            if not hasattr(DynamicCache, "seen_tokens"):
                DynamicCache.seen_tokens = property(
                    lambda self: getattr(self, "_seen_tokens", 0)
                )

            if not hasattr(DynamicCache, "get_seq_length"):
                def _get_seq_length(self, layer_idx: int = 0) -> int:
                    if len(getattr(self, "key_cache", [])) <= layer_idx:
                        return 0
                    return self.key_cache[layer_idx].shape[-2]
                DynamicCache.get_seq_length = _get_seq_length

            if not hasattr(DynamicCache, "get_max_length"):
                def _get_max_length(self):
                    if not getattr(self, "key_cache", []):
                        return None
                    return self.key_cache[0].shape[-2]
                DynamicCache.get_max_length = _get_max_length

            if not hasattr(DynamicCache, "get_usable_length"):
                def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
                    max_length = self.get_max_length()
                    prev_length = self.get_seq_length(layer_idx)
                    if max_length is not None and prev_length + new_seq_length > max_length:
                        return max_length - new_seq_length
                    return prev_length
                DynamicCache.get_usable_length = _get_usable_length

        except ImportError:
            pass

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            print(f"  [generate] Local model skipped: no CUDA GPU available")
            _LOCAL_LOAD_FAILED = True
            return None, None

        print(f"  [generate] Loading {LOCAL_MODEL_NAME} in 4-bit on {torch.cuda.get_device_name(0)} ...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            quantization_config=bnb_cfg,
            device_map="auto",           # places all layers on cuda:0 with GPU present
            trust_remote_code=True,
            attn_implementation="eager", # avoids flash-attn / window-size errors
        )
        mdl.eval()
        _LOCAL_TOKENIZER = tok
        _LOCAL_MODEL = mdl
        vram_mb = round(torch.cuda.memory_allocated(0) / 1024**2, 1)
        print(f"  [generate] {LOCAL_MODEL_NAME} loaded — VRAM: {vram_mb} MB")
        return _LOCAL_TOKENIZER, _LOCAL_MODEL
    except Exception as e:
        print(f"  [generate] Local model unavailable: {e}")
        _LOCAL_LOAD_FAILED = True
        return None, None



def _call_local_model(context: str, brief: str, history: list[str], system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
    """
    Run one inference pass through Phi-3-mini (base, not fine-tuned).
    Measures and prints wall-clock latency.
    Returns a slot template string or None on failure/timeout.
    """
    import time
    tok, mdl = _load_local_model()
    if tok is None:
        return None

    try:
        import torch
        # ── Compatibility shim ──────────────────────────────────────────────
        # transformers >=4.39 renamed DynamicCache.seen_tokens → _seen_tokens.
        # Phi-3's modeling_phi3.py still uses the old name. Add the alias
        # if the installed version is missing it.
        try:
            from transformers.cache_utils import DynamicCache
            if not hasattr(DynamicCache, "seen_tokens"):
                DynamicCache.seen_tokens = property(
                    lambda self: getattr(self, "_seen_tokens", 0)
                )
        except ImportError:
            pass

        recent = ""
        if history:
            recent = ("\n\nYour last replies (do not reuse their opening):\n- "
                      + "\n- ".join(history[-2:]))

        user_msg = (f"{context}\n\nTurn instruction: {brief}{recent}"
                    "\n\nWrite the reply now, slots only.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Move to GPU — model is on cuda:0 via 4-bit bitsandbytes
        device = next(mdl.parameters()).device
        inputs = inputs.to(device)
        attention_mask = inputs.new_ones(inputs.shape)
        input_len = inputs.shape[-1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = mdl.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=60,
                do_sample=False,
                repetition_penalty=1.25,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        new_tokens = out[0][input_len:]
        reply = tok.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"  [generate] local model: {len(new_tokens)} tokens in {latency_ms}ms"
              f" ({latency_ms // max(len(new_tokens), 1)} ms/tok)")
        return reply or None
    except Exception as e:
        print(f"  [generate] local model inference failed: {e}")
        return None



def _call_groq_model(context: str, brief: str, history: list[str], system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
    """
    Call Groq LPU API for ultra-low-latency (~200ms) dynamic template generation.
    Returns slot template string or None.
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return None

    recent = ""
    if history:
        recent = ("\n\nRecent previous replies (do not repeat opening words):\n- " + "\n- ".join(history[-3:]))

    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"Turn instruction: {brief}{recent}\n\n"
                    "MANDATORY REQUIREMENT: Use ONLY slot placeholders like {F1}, {F1.cagr3}, {F1.risk}, {F2}. "
                    "Never output raw digits, percentages, or fund names directly. Reply in 2 short natural sentences."
                ),
            },
        ],
        "max_tokens": 80,
        "temperature": 0.2,
    }

    try:
        import httpx
        with httpx.Client(verify=True, timeout=4.0) as client:
            t0 = time.perf_counter()
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
            )
            ms = int((time.perf_counter() - t0) * 1000)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content:
                    print(f"  [generate] Groq ({model_name}) generated in {ms}ms: {content}")
                    return content
            else:
                print(f"  [generate] Groq returned HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"  [generate] Groq inference failed: {e}")
    return None


def _call_model(context: str, brief: str, history: list[str], system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None

    recent = ""
    if history:
        recent = ("\n\nYour last replies this session (do not repeat their "
                  "opening words):\n- " + "\n- ".join(history[-3:]))

    payload = {
        "model": os.getenv("VOICEFINAI_MODEL", "claude-sonnet-4-6"),
        "max_tokens": 300,
        "system": system_prompt,
        "messages": [{
            "role": "user",
            "content": f"{context}\n\nTurn instruction: {brief}{recent}\n\nWrite the reply now, slots only.",
        }],
    }

    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data.get("content", [])).strip()
    except Exception:
        return None


def _template_is_clean(tpl: str) -> bool:
    """Pre-flight check before handing to AuditStream — same rules, cheaper."""
    if not tpl or len(tpl) > 420:
        return False
    if not _PLACEHOLDER.search(tpl):
        return False
    # Must terminate cleanly on sentence-ending punctuation to reject mid-token cutoffs
    trimmed = tpl.strip()
    if trimmed[-1] not in (".", "!", "?", '"', "'"):
        return False
    # Reject degenerative loop repetitions (same 16+ char clause repeated 2+ times)
    if re.search(r"(.{16,}?)[\s.,!?;:\-—]+\1", tpl, re.IGNORECASE):
        return False
    stripped = _PLACEHOLDER.sub("SLOT", tpl)
    for m in _BARE_DIGIT.finditer(stripped):
        if m.group() not in _ALLOWED_CARDINALS:
            return False
    return True


def _factual_discover_template(turn: dict, snapshot, session, lang: str) -> str:
    """Return a concise, slot-backed discovery response without a personal ranking.

    A request for fund ideas does not provide enough information to declare a
    winner. This keeps the first answer useful while leaving the user free to
    ask about risk, past returns, or a comparison of the visible funds.
    """
    has_second = "F2" in snapshot.funds
    variant = getattr(session, "turn", 0) % 3

    if lang == "english":
        if variant == 0:
            opening = (
                "Here is an available option matching the details you shared: {F1} is in "
                "{F1.category}, carries {F1.risk} risk, and its past three-year return is {F1.cagr3}."
            )
        elif variant == 1:
            opening = (
                "For the criteria you mentioned, {F1} is one option on screen. It has "
                "{F1.risk} risk and returned {F1.cagr3} over the past three years."
            )
        else:
            opening = (
                "{F1} is available in the {F1.category} category. Its risk is {F1.risk}, "
                "and its past three-year return is {F1.cagr3}."
            )
        if has_second:
            return opening + " {F2} is another option to compare. Historical returns are not a forecast."
        return opening + " Historical returns are not a forecast; check the AMC factsheet and your risk tolerance before choosing."

    if variant == 0:
        opening = (
            "Aapki batayi hui details ke hisaab se {F1} ek available option hai. Ye "
            "{F1.category} category mein hai, risk {F1.risk} hai aur pichhle teen saal ka return {F1.cagr3} raha hai."
        )
    elif variant == 1:
        opening = (
            "{F1} screen par available option hai. Iska risk {F1.risk} hai aur pichhle "
            "teen saal ka return {F1.cagr3} raha hai."
        )
    else:
        opening = (
            "Aapke criteria ke liye {F1} {F1.category} category mein mil raha hai. Iska risk "
            "{F1.risk} aur pichhle teen saal ka return {F1.cagr3} hai."
        )
    if has_second:
        return opening + " {F2} ek aur option hai jise compare kar sakte hain. Purane returns future ki guarantee nahi hote."
    return opening + " Purane returns future ki guarantee nahi hote; AMC factsheet aur apni risk tolerance check karke hi chunav karein."


def _goal_purpose_label(purpose: str) -> str:
    """Turn stored plan labels into a short phrase that sounds natural aloud."""
    labels = {
        "Buying a Car": "car",
        "Buying a House": "house",
        "Retirement Fund": "retirement",
        "Child Education": "child education",
        "Wedding & Marriage": "wedding",
    }
    return labels.get(purpose or "", "financial")


def _factual_goal_template(turn: dict, snapshot, session, lang: str) -> str:
    """Explain the user's own corpus/SIP plan with snapshot-backed figures."""
    purpose = _goal_purpose_label(turn.get("goal_purpose") or getattr(session, "goal_purpose", ""))
    time_to_goal = turn.get("goal_plan_mode") == "time_to_goal"

    if lang == "english":
        if time_to_goal:
            return (
                f"For your {purpose} goal of {{U.goal_amount}}, investing {{U.goal_sip}} each month "
                "could reach that target in about {U.goal_years} under this planning assumption. "
                "{F1} is an available option in the displayed category; future returns are not guaranteed."
            )
        return (
            f"For your {purpose} goal of {{U.goal_amount}}, a monthly SIP of {{U.goal_sip}} "
            "is the planning estimate for {U.goal_years}. {F1} is an available option in the "
            "displayed category; future returns are not guaranteed."
        )

    if time_to_goal:
        return (
            f"Aapke {purpose} ke {{U.goal_amount}} goal ke liye, har mahine {{U.goal_sip}} invest "
            "karne par planning assumption ke hisaab se lagbhag {U.goal_years} lag sakte hain. "
            "{F1} screen par available option hai; future returns guaranteed nahi hote."
        )
    return (
        f"Aapke {purpose} ke {{U.goal_amount}} goal ke liye, {{U.goal_years}} mein pahunchne ka "
        "planning estimate har mahine {U.goal_sip} ka hai. {F1} screen par available option hai; "
        "future returns guaranteed nahi hote."
    )


def _factual_compare_template(snapshot, lang: str) -> str:
    """A closed-set comparison that cannot drift into an invented ranking."""
    if len(snapshot.funds) < 2:
        if lang == "english":
            return (
                "I have {F1} on screen. Its past three-year return is {F1.cagr3} and its risk is "
                "{F1.risk}. Add one more fund and I can compare them side by side."
            )
        return (
            "Screen par {F1} hai. Iska pichhle teen saal ka return {F1.cagr3} aur risk {F1.risk} hai. "
            "Ek aur fund add karenge toh side-by-side compare kar sakte hain."
        )
    if lang == "english":
        return (
            "{F1} has a past three-year return of {F1.cagr3}, while {F2} has {F2.cagr3}. "
            "Their risk labels are {F1.risk} and {F2.risk}. Historical returns are not a forecast."
        )
    return (
        "{F1} ka pichhle teen saal ka return {F1.cagr3} hai, jabki {F2} ka {F2.cagr3} hai. "
        "Risk labels {F1.risk} aur {F2.risk} hain. Purane returns future ki guarantee nahi hote."
    )


def generate_template(turn: dict, snapshot, session, attempts: int = 2) -> tuple[str, str]:
    """
    Returns (template, source) where source is one of:
      "model"       — Anthropic cloud model
      "local_model" — Phi-3-mini base model (not fine-tuned)
      "bank"        — deterministic rotating bank (always available)
    Template still goes through AuditStream — this is not a substitute for validation.
    """
    lang = turn.get("language") or getattr(session, "language", "hinglish")
    if lang == "auto":
        lang = "hinglish"

    if turn.get("intent") == "goal" and snapshot is not None and snapshot.funds:
        return _factual_goal_template(turn, snapshot, session, lang), "goal_plan"

    if turn.get("intent") == "compare" and snapshot is not None and snapshot.funds:
        return _factual_compare_template(snapshot, lang), "factual_comparison"

    # The demo rotates an eligible catalogue window. Explain the visible
    # screening facts without inventing a performance-ranking or suitability
    # process that did not actually happen.
    if turn.get("why_query") and snapshot is not None and snapshot.funds:
        if len(snapshot.funds) > 1:
            if lang == "english":
                details = "; ".join(
                    f"{{F{i}}} is {{F{i}.category}} with {{F{i}.risk}} risk"
                    for i in range(1, len(snapshot.funds) + 1)
                )
                return (
                    "These are available candidates from this search, not a personal ranking. "
                    f"{details}. Past returns are historical and don't prove future performance "
                    "or personal suitability. Which one would you like me to compare or explain?",
                    "grounded_why_reply",
                )
            details = "; ".join(
                f"{{F{i}}} {{F{i}.category}} category ka hai aur risk {{F{i}.risk}} hai"
                for i in range(1, len(snapshot.funds) + 1)
            )
            return (
                "Ye search mein mile available options hain, personal ranking nahi. "
                f"{details}. Purane returns future performance ya personal suitability ki "
                "guarantee nahi dete. Kis fund ko compare ya detail mein samjhun?",
                "grounded_why_reply",
            )
        if lang == "english":
            return (
                "I surfaced {F1} from the options matching this search; it is in the "
                "{F1.category} category and carries {F1.risk} risk. Its past three-year return "
                "was {F1.cagr3}, which is historical information, not a forecast or proof "
                "that it suits you. I can compare it with another option or explain its risk.",
                "grounded_why_reply",
            )
        return (
            "Is search ke matching options mein {F1} aaya hai. Ye {F1.category} category ka "
            "fund hai aur iska risk {F1.risk} hai; iska pichhle teen saal ka return {F1.cagr3} "
            "sirf historical jaankari hai, future ka vaada ya personal suitability ka proof nahi. "
            "Chahein toh main ise doosre option se compare kar sakta hoon.",
            "grounded_why_reply",
        )

    if turn.get("fund_house") and not (turn.get("risk") or getattr(session, "risk", None)) and snapshot is not None and snapshot.funds:
        if lang == "english":
            return (
                f"From {turn['fund_house']}, {{F1}} is in my currently verified list. "
                "It carries {F1.risk} risk; a long horizon by itself doesn't tell me whether "
                "that volatility suits you. I can explain its risk or include other fund houses.",
                "amc_without_risk_preference",
            )
        return (
            f"{turn['fund_house']} ka {{F1}} meri abhi verified list mein hai. Iska risk "
            "{F1.risk} hai; sirf long term kehne se mujhe aapki risk tolerance pata nahi chalti. "
            "Chahein toh iska risk samjhaun ya doosre fund houses bhi dikhaun.",
            "amc_without_risk_preference",
        )

    # Metric replies are closed-set lookups. Answer all fields requested in a
    # compound follow-up, and don't let a language model invent unavailable
    # data such as an expense ratio.
    if turn.get("intent") == "metric":
        metric_map = _METRIC_SENTENCE_EN if lang == "english" else _METRIC_SENTENCE
        tail_map = _METRIC_TAIL_EN if lang == "english" else _METRIC_TAIL
        fields = list(dict.fromkeys(turn.get("metric_fields") or [turn.get("metric_field") or "cagr3"]))
        lines = [metric_map[field] for field in fields if field in metric_map]
        if lines:
            tail = tail_map[(getattr(session, "turn", 0) or 0) % len(tail_map)]
            return " ".join(lines[:3]) + tail, "verified_metric_lookup"

    if turn.get("intent") == "discover" and snapshot is not None and snapshot.funds:
        return _factual_discover_template(turn, snapshot, session, lang), "factual_discovery"
    sys_prompt = SYSTEM_PROMPT_EN if lang == "english" else SYSTEM_PROMPT

    brief = INTENT_BRIEF.get(turn["intent"], INTENT_BRIEF["clarify"])

    # ── Fund-count-aware discover brief ──────────────────────────────────
    # The base model consistently enumerates every fund slot it sees, which
    # causes two bugs:
    #  - 3 funds → enumerate all 3 → hit 120-token limit → corrupt last one
    #  - 1 fund  → invent {F2} to pattern-match against the brief's "1 or 2"
    # Fix: tell the model exactly how many funds exist and what to do.
    if turn["intent"] == "discover" and snapshot is not None:
        n = len(snapshot.funds)
        if n == 1:
            brief = (
                "Only ONE fund is available in this context: {F1}. "
                "Recommend ONLY {F1} using its slots. "
                "Do NOT reference {F2}, {F3}, or any fund that is not in the context. "
                "Two spoken sentences, under 40 words."
            )
        elif n == 2:
            brief = (
                "Two funds are available: {F1} and {F2}. "
                "Pick the better one and explain in 2 sentences why. Mention the other briefly. "
                "Do NOT enumerate both in full detail. Under 45 words."
            )
        else:
            brief = (
                f"{n} funds are available. Pick the SINGLE best option and recommend it in 2 sentences. "
                "You may mention 1 other fund briefly for contrast. "
                "Do NOT enumerate every fund — that wastes words and truncates. Under 50 words."
            )

    requested_house = turn.get("fund_house")
    if requested_house and snapshot is not None:
        brief = (
            f"The user explicitly asked for {requested_house}. Recommend only fund slots from the "
            f"{requested_house} context. Never mention or recommend another fund house or any fund "
            f"outside these slots. {brief}"
        )

    if snapshot is not None:
        context = snapshot.build_context_string()

        # 1. Try ultra-fast Groq LPU inference first (~250-350ms)
        tpl = _call_groq_model(context, brief, session.said_templates, system_prompt=sys_prompt)
        if tpl and _template_is_clean(tpl):
            session.said_templates.append(tpl)
            return tpl, "groq_model"

        # 2. Try cloud model (Anthropic) if key is set
        for _ in range(attempts):
            tpl = _call_model(context, brief, session.said_templates, system_prompt=sys_prompt)
            if tpl and _template_is_clean(tpl):
                session.said_templates.append(tpl)
                return tpl, "model"

        # 2. Try local Phi-3-mini (base model, not fine-tuned)
        #    Only attempted if USE_LOCAL_LLM=1 (defaults to 0 for instant, snappy voice replies)
        if os.getenv("USE_LOCAL_LLM", "0").lower() in ("1", "true", "yes"):
            tpl = _call_local_model(context, brief, session.said_templates, system_prompt=sys_prompt)
            if tpl and _template_is_clean(tpl):
                session.said_templates.append(tpl)
                return tpl, "local_model"

    # 3. Bank fallback — always works, anti-repetition within session
    return from_bank(turn, session), "bank"
