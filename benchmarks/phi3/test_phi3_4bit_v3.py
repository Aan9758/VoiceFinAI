"""
test_phi3_4bit_v3.py — 6 test cases, both validators, real Snapshots.

Original 4:
  1. discover (3 funds) — retests Bug 2
  2. fund_detail
  3. compare
  4. discover_low (1 fund)

Adversarial 2:
  5. discover (3 funds, exact Bug 2 repro) — greedy 3-fund enumeration
  6. discover_low AFTER a prior 3-fund discover (retest Bug 1 Case B — does
     prior session context leak multi-fund slots into a single-fund turn?)

For each run: _template_is_clean() + AuditStream.validate() with matching Snapshot.
"""
import os, sys, time, re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_orig = requests.Session.send
def _nv(self, r, **kw): kw["verify"] = False; return _orig(self, r, **kw)
requests.Session.send = _nv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.generate import SYSTEM_PROMPT, INTENT_BRIEF
from core.snapshot import Snapshot, FundRecommendation
from core.auditstream import AuditStream

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def vram_mb():
    torch.cuda.synchronize()
    return round(torch.cuda.memory_allocated(0) / 1024**2, 1)

print(f"CUDA: {torch.cuda.get_device_name(0)}  VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)}GB", flush=True)
print(f"VRAM before load: {vram_mb()} MB", flush=True)

MODEL = "microsoft/Phi-3-mini-4k-instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
print(f"\nLoading {MODEL} 4-bit...", flush=True)
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, attn_implementation="eager").eval()
print(f"Load: {int((time.perf_counter()-t0)*1000)}ms  VRAM: {vram_mb()} MB", flush=True)


# ── Minimal mock FundRecommendations ─────────────────────────────────────────
def _mock_fund(n: int) -> FundRecommendation:
    return FundRecommendation(
        fund_name=f"Mock Fund {n} Direct Growth",
        isin=f"INF00000{n:04d}A1",
        category="Flexi Cap" if n == 1 else ("Small Cap" if n == 2 else "Overnight"),
        nav=float(50 + n * 10),
        returns_1yr=float(12 + n),
        returns_3yr=float(14 + n),
        returns_5yr=float(15 + n),
        risk_rating="Moderate" if n == 1 else ("High" if n == 2 else "Low"),
        min_sip=500,
        fund_house="Mock AMC",
        lockin_years=0,
        expense_ratio=0.5,
        as_of="2026-09-15",
        source="test",
    )

FUND1 = _mock_fund(1)
FUND2 = _mock_fund(2)
FUND3 = _mock_fund(3)

# Snapshots for each test
SNAP_3 = Snapshot([FUND1, FUND2, FUND3])   # F1 + F2 + F3
SNAP_1 = Snapshot([FUND1])                  # F1 only
SNAP_2 = Snapshot([FUND1, FUND2])           # F1 + F2


# ── Validators ────────────────────────────────────────────────────────────────
PLACEHOLDER = re.compile(r"\{([A-Z]\d*(?:\.[a-z0-9]+)?|U\.[a-z]+)\}")
BARE_DIGIT  = re.compile(r"\b\d+(?:\.\d+)?\b")
ALLOWED     = frozenset(str(i) for i in range(1, 11))

def is_clean(raw):
    if not raw: return "FAIL (empty)"
    if len(raw) > 420: return "FAIL (too long: %d)" % len(raw)
    if not PLACEHOLDER.search(raw): return "FAIL (no placeholder)"
    stripped = PLACEHOLDER.sub("SLOT", raw)
    for m in BARE_DIGIT.finditer(stripped):
        if m.group() not in ALLOWED:
            return "FAIL (bare digit: %r)" % m.group()
    return "PASS"

def audit_check(raw, snapshot):
    auditor = AuditStream(snapshot)
    result = auditor.validate(raw)
    if result.passed:
        return "PASS"
    return "FAIL — " + "; ".join(result.violations)


# ── Build context string matching the test's hand-crafted format ──────────────
def ctx_from_snapshot(snap, fields=("cagr3", "risk", "minsip")):
    """Simple pipe-separated context matching the format the model was trained on."""
    lines = []
    for key in snap.funds:
        parts = [f"{{{key}}}"]
        for f in fields:
            parts.append(f"{f}: {{{key}.{f}}}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


# ── Test cases ────────────────────────────────────────────────────────────────
TESTS = [
    # (label, intent, snapshot, extra_note)
    ("discover-3funds",   "discover",     SNAP_3, "Original Bug 2 scenario"),
    ("fund_detail",       "fund_detail",  SNAP_1, "Single fund — original Run 2"),
    ("compare",           "compare",      SNAP_2, "Two funds — original Run 3"),
    ("discover_low-1fund","discover",     SNAP_1, "Original Bug 1 scenario — single fund, discover intent"),
    # Adversarial
    ("ADV-discover-3funds","discover",    SNAP_3, "ADVERSARIAL: 3-fund discover, reproduce Bug 2 bare digit"),
    ("ADV-discover_after_3","discover",   SNAP_1, "ADVERSARIAL: single-fund discover AFTER seeing 3-fund context (Bug 1 Case B)"),
]

prior_3fund_context = None  # Will be set after Run 5 to simulate session context

for i, (label, intent, snap, note) in enumerate(TESTS, 1):
    brief = INTENT_BRIEF.get(intent, "")
    ctx = ctx_from_snapshot(snap)

    # Adversarial Run 6: inject the prior 3-fund context as recent history
    if label == "ADV-discover_after_3" and prior_3fund_context:
        user_msg = (
            f"{ctx}\n\nTurn instruction: {brief}"
            f"\n\nPrevious turn context (for reference only — do NOT copy its slots): {prior_3fund_context}"
            f"\n\nWrite the reply now, slots only."
        )
    else:
        user_msg = f"{ctx}\n\nTurn instruction: {brief}\n\nWrite the reply now, slots only."

    print(f"\n{'='*65}", flush=True)
    print(f"Run {i} — {label}", flush=True)
    print(f"  Note: {note}", flush=True)
    print(f"  Snapshot funds: {list(snap.funds.keys())}", flush=True)
    print(f"  brief: {brief[:70]}...", flush=True)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
    attn   = inputs.new_ones(inputs.shape)

    t1 = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(inputs, attention_mask=attn, max_new_tokens=120,
                           do_sample=False, eos_token_id=tok.eos_token_id,
                           pad_token_id=tok.eos_token_id)
    t2 = time.perf_counter()
    torch.cuda.synchronize()

    new_toks = out[0][inputs.shape[-1]:]
    raw = tok.decode(new_toks, skip_special_tokens=True).strip()
    gen_ms = int((t2 - t1) * 1000)
    ntoks  = len(new_toks)
    tps    = round(ntoks / max(t2 - t1, 0.001), 1)

    print(f"  gen_time={gen_ms}ms  ntoks={ntoks}  tok/s={tps}  VRAM={vram_mb()}MB", flush=True)
    print(f"  Raw: {repr(raw)}", flush=True)
    print(f"  _template_is_clean:  {is_clean(raw)}", flush=True)
    print(f"  AuditStream.validate: {audit_check(raw, snap)}", flush=True)

    if label == "ADV-discover-3funds":
        prior_3fund_context = ctx  # save for run 6

print("\nALL 6 RUNS COMPLETE", flush=True)
