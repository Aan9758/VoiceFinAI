"""
verify_audit_chain.py — Verify Audit Log entry for Turn 2 and hash chain consistency.
"""
import os, sys, json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ.pop("ANTHROPIC_API_KEY", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import pipeline, generate

# 1. Reset pipeline audit log for a clean test
pipeline._audit_log.clear()

# 2. Poisoned model setup
BAD_TEMPLATE = (
    "{F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai. "
    "{F2} bhi consider karo, iska return {F2.cagr3} aur risk {F2.risk} hai."
)

generate._call_model = lambda *a, **kw: None
generate._call_local_model = lambda *a, **kw: None  # Turn 1: normal bank

def dummy_synth(text):
    return "token_123"

# Turn 1
print("Running Turn 1...", flush=True)
pipeline.run_turn("low risk safe fund dikhao", "chain-verify-001", dummy_synth)

# Turn 2 with poisoned model
generate._call_local_model = lambda context, brief, history: BAD_TEMPLATE
print("Running Turn 2 (poisoned)...", flush=True)
pipeline.run_turn("pehla fund ke baare mein batao", "chain-verify-001", dummy_synth)

# 3. Retrieve and inspect audit entries
entries = [e for e in pipeline._audit_log if e.session_id == "chain-verify-001"]
assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"

e1, e2 = entries[0], entries[1]

print("\n" + "=" * 70)
print("AUDIT ENTRY: TURN 1")
print("=" * 70)
print(json.dumps(e1.__dict__, indent=2))

print("\n" + "=" * 70)
print("AUDIT ENTRY: TURN 2 (AFTER FIREWALL INTERVENTION)")
print("=" * 70)
print(json.dumps(e2.__dict__, indent=2))

print("\n" + "=" * 70)
print("HASH CHAIN VERIFICATION")
print("=" * 70)

# Check 1: Genesis link
c1_pass = (e1.prev_entry_hash == "GENESIS")
print(f"1. Turn 1 prev_entry_hash == 'GENESIS': {c1_pass} ({e1.prev_entry_hash})")

# Check 2: Turn 1 self-hash
recomputed_hash_1 = e1.compute_hash()
c2_pass = (recomputed_hash_1 == e1.entry_hash)
print(f"2. Turn 1 entry_hash matches compute_hash(): {c2_pass}")
print(f"   stored:     {e1.entry_hash}")
print(f"   recomputed: {recomputed_hash_1}")

# Check 3: Turn 2 prev link matches Turn 1 entry_hash
c3_pass = (e2.prev_entry_hash == e1.entry_hash)
print(f"3. Turn 2 prev_entry_hash == Turn 1 entry_hash: {c3_pass}")
print(f"   Turn 1 entry_hash:      {e1.entry_hash}")
print(f"   Turn 2 prev_entry_hash: {e2.prev_entry_hash}")

# Check 4: Turn 2 self-hash with corrected violations / validation_passed
recomputed_hash_2 = e2.compute_hash()
c4_pass = (recomputed_hash_2 == e2.entry_hash)
print(f"4. Turn 2 entry_hash matches compute_hash(): {c4_pass}")
print(f"   stored:     {e2.entry_hash}")
print(f"   recomputed: {recomputed_hash_2}")

# Check 5: Turn 2 fields integrity
c5_pass = (e2.validation_passed is False and len(e2.violations) > 0)
print(f"5. Turn 2 records validation_passed=False and violations: {c5_pass}")
print(f"   validation_passed: {e2.validation_passed}")
print(f"   violations count:  {len(e2.violations)}")

# Check 6: Tamper-evidence demonstration
# If someone tampered with validation_passed to hide the violation:
tampered_entry = e2.__dict__.copy()
tampered_entry["validation_passed"] = True
canonical_tampered = json.dumps({k: v for k, v in tampered_entry.items() if k != "entry_hash"}, sort_keys=True, default=str)
import hashlib
tampered_hash = hashlib.sha256(canonical_tampered.encode()).hexdigest()
c6_pass = (tampered_hash != e2.entry_hash)
print(f"6. Tamper check: modifying validation_passed to True alters hash: {c6_pass}")
print(f"   original hash: {e2.entry_hash}")
print(f"   tampered hash: {tampered_hash}")

all_clean = all([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])
print("\nOVERALL HASH CHAIN CONSISTENCY: " + ("✅ PASS" if all_clean else "❌ FAIL"))
