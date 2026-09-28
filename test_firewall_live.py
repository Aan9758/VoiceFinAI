"""
test_firewall_live.py — End-to-end AuditStream firewall verification.

This test goes through the REAL production pipeline (core.pipeline.run_turn),
not an isolated call to AuditStream.validate().

Setup:
  1. Monkey-patch generate._call_model → None  (no Anthropic key)
  2. Monkey-patch generate._call_local_model → returns a DELIBERATELY BAD
     template containing {F2} when only F1 exists in the snapshot
  3. Call pipeline.run_turn() with a query that produces a fund_detail intent
     (single fund, limit=1 → snapshot only has F1)
  4. Observe: does AuditStream catch it? Does the pipeline fall back to bank?
     What violations are reported?

This is the "watched the firewall work" test.
"""
import os, sys, json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
# Ensure no Anthropic key so cloud model path is skipped
os.environ.pop("ANTHROPIC_API_KEY", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Step 1: Import the real pipeline and generation modules ───────────────────
from core import pipeline, generate, dialogue

# ── Step 2: Monkey-patch the local model to return a BAD template ─────────────
# This simulates the exact bug we diagnosed: model writes {F2} in a single-fund
# context. The template is syntactically valid (_template_is_clean passes) but
# AuditStream must catch the out-of-range index.

BAD_TEMPLATE = (
    "{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai. "
    "{F2} bhi consider karo, iska return {F2.cagr3} aur risk {F2.risk} hai."
)

_original_call_local = generate._call_local_model

def _poisoned_local_model(context, brief, history):
    """Always returns the bad template — simulates the hallucination bug."""
    print(f"  [POISON] _call_local_model returning BAD template with {{F2}}", flush=True)
    return BAD_TEMPLATE

generate._call_local_model = _poisoned_local_model

# Also ensure _call_model returns None (no API key)
generate._call_model = lambda *a, **kw: None

# ── Step 3: Confirm the bad template passes _template_is_clean ────────────────
# This is the whole point — the format-only check passes, but AuditStream must
# still catch the semantic violation.
clean_check = generate._template_is_clean(BAD_TEMPLATE)
print(f"\n{'='*70}", flush=True)
print(f"BAD TEMPLATE: {repr(BAD_TEMPLATE)}", flush=True)
print(f"_template_is_clean: {clean_check}", flush=True)
print(f"  → This PASSES the format check. The question is: does AuditStream catch it?", flush=True)

# ── Step 4: Dummy synthesize (we don't need real TTS for this test) ───────────
def dummy_synth(text):
    return "dummy_audio_token"

# ── Step 5: Run a DISCOVER query through the real pipeline ────────────────────
# "low risk safe fund" → parse_turn should produce DISCOVER intent with risk="low"
# pipeline.run_turn will call retrieval2.build_snapshot(bucket="low", limit=3)
# BUT — the bad template always writes {F2}, and if the snapshot happens to have
# F1+F2+F3 for a discover, the violation wouldn't trigger.
#
# So we also test a FUND_DETAIL scenario where limit=1 is enforced.
# To do that we need a prior discover turn to populate session.last_slots,
# then a follow-up detail query.

print(f"\n{'='*70}", flush=True)
print("PHASE A: Initial discover turn (populates session with fund list)", flush=True)
print("="*70, flush=True)

# First, do a normal discover to populate the session
generate._call_local_model = lambda *a, **kw: None  # skip model, use bank
result_a = pipeline.run_turn("low risk safe fund dikhao", "firewall-test-001", dummy_synth)
print(f"  Intent:    {result_a['intent']}", flush=True)
print(f"  Generator: {result_a['generator']}", flush=True)
print(f"  Snapshot:  {result_a.get('snapshot_hash', 'none')}", flush=True)
print(f"  Voice:     {result_a['voice_text'][:100]}...", flush=True)
print(f"  Violations: {result_a['violations']}", flush=True)

# Now restore the poisoned model for the critical test
generate._call_local_model = _poisoned_local_model

print(f"\n{'='*70}", flush=True)
print("PHASE B: Fund detail follow-up — POISONED model returns {F2} on a 1-fund snapshot", flush=True)
print("="*70, flush=True)
print("  This is the real test: limit=1 → Snapshot has only F1.", flush=True)
print("  Poisoned model returns template with {F2}.", flush=True)
print("  AuditStream must fire INDEX_OUT_OF_RANGE and fall back to bank.", flush=True)
print("", flush=True)

# "pehla fund bata" → should trigger FUND_DETAIL with target F1, limit=1
result_b = pipeline.run_turn("pehla fund ke baare mein batao", "firewall-test-001", dummy_synth)

print(f"  Intent:      {result_b['intent']}", flush=True)
print(f"  Generator:   {result_b['generator']}", flush=True)
print(f"  Voice text:  {result_b['voice_text'][:150]}", flush=True)
print(f"  Violations:  {result_b['violations']}", flush=True)

# ── Step 6: Verdict ──────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print("VERDICT", flush=True)
print("="*70, flush=True)

if result_b["generator"] == "bank_after_violation" and result_b["violations"]:
    print("  ✅ FIREWALL CAUGHT THE VIOLATION.", flush=True)
    print(f"  Generator fell back to: {result_b['generator']}", flush=True)
    print(f"  Violations reported:", flush=True)
    for v in result_b["violations"]:
        print(f"    → {v}", flush=True)
    print("", flush=True)
    print("  The poisoned template passed _template_is_clean() but was REJECTED", flush=True)
    print("  by AuditStream.validate() in the live pipeline. The bank fallback", flush=True)
    print("  produced the response the user actually hears.", flush=True)
elif result_b["generator"] == "bank":
    print("  ⚠️  Bank was used, but not via violation fallback.", flush=True)
    print("  The poisoned model may not have been called (check [POISON] log above).", flush=True)
    print(f"  Violations: {result_b['violations']}", flush=True)
else:
    print("  ❌ FIREWALL DID NOT CATCH IT.", flush=True)
    print(f"  Generator: {result_b['generator']}", flush=True)
    print(f"  Voice text: {result_b['voice_text']}", flush=True)
    print(f"  Violations: {result_b['violations']}", flush=True)

# ── Step 7: Also show the full audit log for this session ─────────────────────
print(f"\n{'='*70}", flush=True)
print("AUDIT LOG (session firewall-test-001)", flush=True)
print("="*70, flush=True)
for entry in pipeline.audit_log():
    if entry.get("session_id") == "firewall-test-001":
        print(json.dumps({
            "turn": entry["turn"],
            "intent": entry["intent"],
            "validation_passed": entry["validation_passed"],
            "violations": entry["violations"],
            "pre_render_template": entry["pre_render_template"][:120],
        }, indent=2), flush=True)

print("\nDONE", flush=True)
