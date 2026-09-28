"""
demo/voice_turn.py — Full Turn With Live Voice Output
======================================================
Same as full_turn.py but actually speaks the response using edge-tts.
en-IN-NeerjaNeural — Indian English neural voice (free, no API key).

Run:
    python demo/voice_turn.py
    python demo/voice_turn.py "high risk fund chahiye"
    python demo/voice_turn.py "mujhe crypto fund chahiye"

TTFB measured: P50 ~1073ms, P95 ~1361ms (edge-tts from India, 2026-09-14)
Production target: ElevenLabs Flash v2.5, Singapore, ~150ms warm TTFB
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval  import parse_intent, retrieve_funds
from core.auditstream import AuditStream
from core.tts        import speak, generate_filler, FILLER_THRESHOLD_MS, VOICE

# ANSI
GREEN  = "\033[92m"; RED  = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m";  DIM    = "\033[2m"; RESET = "\033[0m"

def h1(t): return f"\n{BOLD}{CYAN}{'━'*64}{RESET}\n{BOLD}{CYAN}  {t}{RESET}\n{BOLD}{CYAN}{'━'*64}{RESET}"
def h2(t): return f"\n{BOLD}  {t}{RESET}"
def ok(t): return f"  {GREEN}✅  {t}{RESET}"
def err(t): return f"  {RED}❌  {t}{RESET}"
def info(t): return f"  {DIM}{t}{RESET}"

TEMPLATES = {
    "3funds": (
        "Yahan {F1}, {F2}, aur {F3} hain aapke liye. "
        "{F1} mein {F1.cagr3} ka 3-year return hai, "
        "minimum SIP {F1.minsip} se start hoti hai. "
        "Risk {F1.risk} hai."
    ),
    "with_amount": (
        "Aapka {U.amount} monthly ke liye {F1} ek achha option hai. "
        "3-year return {F1.cagr3}, NAV {F1.nav}, risk {F1.risk}. "
        "Minimum SIP {F1.minsip} se start hoti hai."
    ),
}
OUT_OF_SCOPE_VOICE = (
    "Yeh mere domain se bahar hai. "
    "Main sirf mutual funds aur SIP ke baare mein help kar sakta hoon."
)


async def run_voice_turn(query: str):
    print(h1("VoiceFinAI — Live Voice Turn"))
    print(info(f"User query: \"{query}\""))
    print(info(f"TTS: edge-tts / {CYAN}en-IN-NeerjaNeural{DIM} (demo) | Production: ElevenLabs Flash v2.5\n"))

    # Pre-generate filler in background while we do intent + retrieval
    filler_task = asyncio.create_task(generate_filler())

    # ── Track 2: Intent ───────────────────────────────────────────────────────
    intent_result = parse_intent(query)
    print(h2("Track 2 — Intent"))
    print(info(f"  intent={intent_result['intent']}  risk={intent_result['risk']}  "
               f"amount={intent_result['amount']}  confidence={intent_result['confidence']}"))

    # Out-of-scope → speak refusal clip immediately
    if intent_result["intent"] == "out_of_scope":
        print(h2("Track 2B — Domain Boundary"))
        print(err("OUT OF SCOPE — speaking refusal clip"))
        t0 = time.perf_counter()
        _, ttfb = await speak(OUT_OF_SCOPE_VOICE, play=True)
        print(info(f"  TTFB: {ttfb:.0f}ms"))
        return

    # ── Track 3: Retrieval ────────────────────────────────────────────────────
    print(h2("Track 3 — Retrieval"))
    t_retrieval = time.perf_counter()
    snapshot = retrieve_funds(intent_result, verbose=True)
    retrieval_ms = (time.perf_counter() - t_retrieval) * 1000
    print(info(f"  Retrieval: {retrieval_ms:.0f}ms"))

    if not snapshot:
        clarify = "Kaunsa fund type chahiye — equity ya debt?"
        print(err("No confident match — speaking clarifying question"))
        await speak(clarify, play=True)
        return

    print(ok(f"Snapshot frozen: {len(snapshot.funds)} fund(s)"))

    # ── Pick template + validate ──────────────────────────────────────────────
    template = TEMPLATES["with_amount"] if intent_result.get("amount") else TEMPLATES["3funds"]
    auditor  = AuditStream(snapshot)
    result   = auditor.validate(template)

    if not result.passed:
        print(err(f"AuditStream BLOCKED ({len(result.violations)} violations):"))
        for v in result.violations:
            print(f"    {RED}→ {v}{RESET}")
        return

    # ── Render ────────────────────────────────────────────────────────────────
    voice_text = auditor.render(template, "voice")
    screen_text = auditor.render(template, "screen")

    print(h2("Raw template (Clip 2 — Wall 2):"))
    print(f"\n    {YELLOW}{template}{RESET}\n")

    print(h2("Screen render:"))
    print(f"\n    {GREEN}{screen_text}{RESET}\n")

    print(h2("Voice render (speaking now...):"))
    print(f"\n    {GREEN}{voice_text}{RESET}\n")

    # Wait for filler to be ready, then race TTS against filler threshold
    filler_path = await filler_task
    t_tts_start = time.perf_counter()

    # Synthesize the real response
    print(info(f"  Synthesizing via edge-tts ({VOICE})..."))
    audio_path, ttfb_ms = await speak(voice_text, play=True)

    total_ms = (time.perf_counter() - t_tts_start) * 1000
    print(info(f"  TTFB: {ttfb_ms:.0f}ms  |  Filler threshold: {FILLER_THRESHOLD_MS}ms"))

    if ttfb_ms > FILLER_THRESHOLD_MS:
        print(info(f"  (Filler would have fired at {FILLER_THRESHOLD_MS}ms — production gap covered)"))

    print(h2("Audit:"))
    print(info(f"  snapshot_hash: {snapshot.hash[:32]}..."))
    print(info(f"  template:      {template[:55]}..."))
    print(info(f"  audit_render:  {auditor.render(template, 'audit')[:55]}..."))
    print(f"\n{ok('Turn complete.')}\n")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "low risk SIP 5000 chahiye"
    asyncio.run(run_voice_turn(query))
