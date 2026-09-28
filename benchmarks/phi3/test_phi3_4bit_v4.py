"""
test_phi3_4bit_v4.py — 6 test cases with dynamic discover brief.

Uses the SAME dynamic brief logic as generate_template() in generate.py:
  1 fund  → "Only ONE fund is available..."
  2 funds → "Two funds are available..."
  3 funds → "N funds are available. Pick the SINGLE best..."

Both validators: _template_is_clean() + AuditStream.validate() with real Snapshots.
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

MODEL = "microsoft/Phi-3-mini-4k-instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
print(f"Loading {MODEL} 4-bit...", flush=True)
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, attn_implementation="eager").eval()
print(f"Load: {int((time.perf_counter()-t0)*1000)}ms  VRAM: {vram_mb()} MB", flush=True)


# ── Mock funds ────────────────────────────────────────────────────────────────
def _mock(n):
    return FundRecommendation(
        fund_name=f"Mock Fund {n} Direct Growth", isin=f"INF00000{n:04d}A1",
        category=["Flexi Cap","Small Cap","Overnight"][n-1] if n<=3 else "Flexi Cap",
        nav=float(50+n*10), returns_1yr=float(12+n), returns_3yr=float(14+n),
        returns_5yr=float(15+n),
        risk_rating=["Moderate","High","Low"][n-1] if n<=3 else "Moderate",
        min_sip=500, fund_house="Mock AMC", lockin_years=0,
        expense_ratio=0.5, as_of="2026-09-15", source="test")

SNAP_3 = Snapshot([_mock(1), _mock(2), _mock(3)])
SNAP_1 = Snapshot([_mock(1)])
SNAP_2 = Snapshot([_mock(1), _mock(2)])


# ── Dynamic discover brief — same logic as generate_template() ────────────────
def discover_brief(n_funds):
    if n_funds == 1:
        return (
            "Only ONE fund is available in this context: {F1}. "
            "Recommend ONLY {F1} using its slots. "
            "Do NOT reference {F2}, {F3}, or any fund that is not in the context. "
            "Two spoken sentences, under 40 words."
        )
    elif n_funds == 2:
        return (
            "Two funds are available: {F1} and {F2}. "
            "Pick the better one and explain in 2 sentences why. Mention the other briefly. "
            "Do NOT enumerate both in full detail. Under 45 words."
        )
    else:
        return (
            f"{n_funds} funds are available. Pick the SINGLE best option and recommend it in 2 sentences. "
            "You may mention 1 other fund briefly for contrast. "
            "Do NOT enumerate every fund — that wastes words and truncates. Under 50 words."
        )


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
    result = AuditStream(snapshot).validate(raw)
    if result.passed:
        return "PASS"
    return "FAIL — " + "; ".join(result.violations)


def ctx_from_snapshot(snap):
    lines = []
    for key in snap.funds:
        lines.append(f"{{{key}}} | cagr3: {{{key}.cagr3}} | risk: {{{key}.risk}} | minsip: {{{key}.minsip}}")
    return "\n".join(lines)


# ── Test cases ────────────────────────────────────────────────────────────────
TESTS = [
    ("discover-3funds",      "discover",    SNAP_3, "Original Bug 2 — 3-fund discover"),
    ("fund_detail",          "fund_detail", SNAP_1, "Single fund — original Run 2"),
    ("compare",              "compare",     SNAP_2, "Two funds — original Run 3"),
    ("discover_low-1fund",   "discover",    SNAP_1, "Original Bug 1 — single fund discover"),
    ("ADV-discover-3funds",  "discover",    SNAP_3, "ADVERSARIAL: 3-fund discover repro"),
    ("ADV-discover_after_3", "discover",    SNAP_1, "ADVERSARIAL: 1-fund discover after prior 3-fund turn"),
]

prior_3fund_ctx = None

for i, (label, intent, snap, note) in enumerate(TESTS, 1):
    n = len(snap.funds)
    if intent == "discover":
        brief = discover_brief(n)
    else:
        brief = INTENT_BRIEF.get(intent, "")

    ctx = ctx_from_snapshot(snap)

    # Run 6: inject prior 3-fund context as history
    if label == "ADV-discover_after_3" and prior_3fund_ctx:
        user_msg = (
            f"{ctx}\n\nTurn instruction: {brief}"
            f"\n\nPrevious turn context (do NOT copy its slots): {prior_3fund_ctx}"
            f"\n\nWrite the reply now, slots only."
        )
    else:
        user_msg = f"{ctx}\n\nTurn instruction: {brief}\n\nWrite the reply now, slots only."

    print(f"\n{'='*65}", flush=True)
    print(f"Run {i} — {label}", flush=True)
    print(f"  Note: {note}", flush=True)
    print(f"  Snapshot funds: {list(snap.funds.keys())}", flush=True)
    print(f"  Brief: {brief[:80]}...", flush=True)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
    attn = inputs.new_ones(inputs.shape)

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
    ntoks = len(new_toks)
    tps = round(ntoks / max(t2 - t1, 0.001), 1)

    print(f"  gen_time={gen_ms}ms  ntoks={ntoks}  tok/s={tps}  VRAM={vram_mb()}MB", flush=True)
    print(f"  Raw: {repr(raw)}", flush=True)
    print(f"  _template_is_clean:   {is_clean(raw)}", flush=True)
    print(f"  AuditStream.validate: {audit_check(raw, snap)}", flush=True)

    if label == "ADV-discover-3funds":
        prior_3fund_ctx = ctx

print("\nALL 6 RUNS COMPLETE", flush=True)
