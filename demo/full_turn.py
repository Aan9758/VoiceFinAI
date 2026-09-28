"""
demo/full_turn.py — End-to-End Turn Simulation
================================================
Shows the complete pipeline from user voice query to verified rendered output.

User query → Intent parser → Retrieval (Track 3) → Snapshot → AuditStream → Render

Run:
    python demo/full_turn.py
    python demo/full_turn.py "high risk SIP chahiye"
    python demo/full_turn.py "mujhe crypto fund chahiye"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval import parse_intent, retrieve_funds
from core.auditstream import AuditStream

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def h1(t): return f"\n{BOLD}{CYAN}{'━'*64}{RESET}\n{BOLD}{CYAN}  {t}{RESET}\n{BOLD}{CYAN}{'━'*64}{RESET}"
def h2(t): return f"\n{BOLD}  {t}{RESET}"
def ok(t): return f"  {GREEN}✅  {t}{RESET}"
def err(t): return f"  {RED}❌  {t}{RESET}"
def info(t): return f"  {DIM}{t}{RESET}"


# ── Response templates per intent ─────────────────────────────────────────────
# These are what the model generates — slot references ONLY, no raw values.
# The model never types a fund name or a number directly.

TEMPLATES = {
    "SIP_recommendation_3funds": (
        "Yahan {F1}, {F2}, aur {F3} hain aapke liye. "
        "{F1} mein {F1.cagr3} ka 3-year return hai, "
        "minimum SIP {F1.minsip} se start hoti hai. "
        "Risk {F1.risk} hai."
    ),
    "SIP_recommendation_with_amount": (
        "Aapka {U.amount} monthly ke liye {F1} ek achha option hai. "
        "3-year return {F1.cagr3}, NAV {F1.nav}, risk {F1.risk}. "
        "Minimum SIP {F1.minsip} se start hoti hai."
    ),
    "out_of_scope": None,   # Pre-generated refusal clip
}

OUT_OF_SCOPE_VOICE = (
    "Yeh mere domain se bahar hai. "
    "Main sirf mutual funds aur SIP ke baare mein help kar sakta hoon."
)


def run_turn(query: str, epoch: int = 1):
    print(h1("VoiceFinAI — Full Turn Simulation"))
    print(info(f"User query: \"{query}\"\n"))

    # ── Track 2: Intent detection ─────────────────────────────────────────────
    print(h2("Track 2 — Intent Detection"))
    intent_result = parse_intent(query)
    print(info(f"Intent:     {intent_result['intent']}"))
    print(info(f"Risk:       {intent_result['risk']}"))
    print(info(f"Amount:     {intent_result['amount']}"))
    print(info(f"Confidence: {intent_result['confidence']}"))

    # Out-of-scope → immediate refusal, no retrieval, no generation
    if intent_result["intent"] == "out_of_scope":
        print(h2("Track 2B — Domain Classifier"))
        print(err("OUT OF SCOPE — domain boundary enforced"))
        print(h2("Voice Response (pre-generated clip):"))
        print(f"\n    {RED}{OUT_OF_SCOPE_VOICE}{RESET}\n")
        print(info("No retrieval. No generation. No hallucination surface."))
        return

    # ── Track 3: Retrieval + Snapshot ────────────────────────────────────────
    print(h2("Track 3 — Retrieval (fires at intent confidence > 0.65)"))
    snapshot = retrieve_funds(intent_result, epoch=epoch, verbose=True)

    if not snapshot:
        print(err("No funds retrieved — confidence gating (Rule 3)"))
        print(info("Clarifying question: 'Equity ya debt funds?'"))
        return

    print(ok(f"Snapshot frozen: {len(snapshot.funds)} fund(s), hash {snapshot.hash[:16]}..."))

    # ── Pick response template ────────────────────────────────────────────────
    if intent_result.get("amount") and len(snapshot.funds) >= 1:
        template = TEMPLATES["SIP_recommendation_with_amount"]
    else:
        template = TEMPLATES["SIP_recommendation_3funds"]

    # ── Track 5: AuditStream validation ──────────────────────────────────────
    print(h2("Track 5 — AuditStream Validation"))
    auditor = AuditStream(snapshot)
    result  = auditor.validate(template)

    if not result.passed:
        print(err(f"BLOCKED — {len(result.violations)} violation(s):"))
        for v in result.violations:
            print(f"    {RED}→ {v}{RESET}")
        return

    print(ok("Template validated. No bare digits. All slots resolved."))

    # ── Render ────────────────────────────────────────────────────────────────
    print(h2("Raw Model Output (Clip 2 — Wall 2):"))
    print(f"\n    {YELLOW}{template}{RESET}\n")

    screen = auditor.render(template, "screen")
    voice  = auditor.render(template, "voice")
    audit  = auditor.render(template, "audit")

    print(h2("Rendered — Screen:"))
    print(f"\n    {GREEN}{screen}{RESET}\n")

    print(h2("Rendered — Voice:"))
    print(f"\n    {GREEN}{voice}{RESET}\n")

    print(h2("Audit Log Entry:"))
    print(info(f"  snapshot_hash:      {snapshot.hash}"))
    print(info(f"  pre_render_template: {template[:60]}..."))
    print(info(f"  rendered_audit:      {audit[:60]}..."))
    print(info(f"  Replay template → snapshot → byte-identical output guaranteed"))

    print(f"\n{ok('Turn complete.')}\n")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "low risk SIP 5000 per month"
    run_turn(query)
