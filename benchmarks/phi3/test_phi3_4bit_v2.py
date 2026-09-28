"""
test_phi3_4bit_v2.py
Same as v1 but imports SYSTEM_PROMPT and INTENT_BRIEF directly from core.generate
so the test is guaranteed to use the production prompt, not a separate copy.
Verifies all 4 intents: discover, fund_detail, compare, discover_low.
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

# Add project root to path so we can import core.generate
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.generate import SYSTEM_PROMPT, INTENT_BRIEF

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def vram_mb():
    torch.cuda.synchronize()
    return round(torch.cuda.memory_allocated(0) / 1024**2, 1)

print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM total:  {round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)} GB", flush=True)
print(f"VRAM before load: {vram_mb()} MB", flush=True)

MODEL = "microsoft/Phi-3-mini-4k-instruct"
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print(f"\nLoading {MODEL} in 4-bit on cuda:0...", flush=True)
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
).eval()
load_ms = int((time.perf_counter() - t0) * 1000)
print(f"Load time:        {load_ms}ms ({load_ms/1000:.1f}s)", flush=True)
print(f"VRAM after load:  {vram_mb()} MB", flush=True)

# Same 4 test contexts as before
CONTEXTS = [
    ("discover",
     "F1: {F1} | 3yr: {F1.cagr3} | risk: {F1.risk} | minSIP: {F1.minsip}\n"
     "F2: {F2} | 3yr: {F2.cagr3} | risk: {F2.risk} | minSIP: {F2.minsip}\n"
     "F3: {F3} | 3yr: {F3.cagr3} | risk: {F3.risk} | minSIP: {F3.minsip}"),
    ("fund_detail",
     "F1: {F1} | NAV: {F1.nav} | 1yr: {F1.cagr1} | 3yr: {F1.cagr3} | risk: {F1.risk}"),
    ("compare",
     "F1: {F1} | risk: {F1.risk} | 3yr: {F1.cagr3}\n"
     "F2: {F2} | risk: {F2.risk} | 3yr: {F2.cagr3}"),
    ("discover_low",
     "F1: {F1} | 3yr: {F1.cagr3} | risk: {F1.risk} | minSIP: {F1.minsip}"),
]

PLACEHOLDER = re.compile(r"\{([A-Z]\d*(?:\.[a-z0-9]+)?|U\.[a-z]+)\}")
BARE_DIGIT  = re.compile(r"\b\d+(?:\.\d+)?\b")
ALLOWED     = frozenset(str(i) for i in range(1, 11))

def check(raw):
    if raw is None: return "FAIL (None)"
    if not raw: return "FAIL (empty)"
    if len(raw) > 420: return "FAIL (too long: %d)" % len(raw)
    if not PLACEHOLDER.search(raw): return "FAIL (no {Fx} placeholder)"
    stripped = PLACEHOLDER.sub("SLOT", raw)
    for m in BARE_DIGIT.finditer(stripped):
        if m.group() not in ALLOWED:
            return "FAIL (bare digit: %r)" % m.group()
    return "PASS"

for i, (intent, ctx) in enumerate(CONTEXTS, 1):
    brief = INTENT_BRIEF.get(intent, "")
    # Same user_msg format as _call_local_model() in generate.py
    user_msg = f"{ctx}\n\nTurn instruction: {brief}\n\nWrite the reply now, slots only."

    print(f"\n{'='*60}", flush=True)
    print(f"Run {i} — {intent}", flush=True)
    print(f"  brief: {brief[:80]}...", flush=True)

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to("cuda:0")
    attn   = inputs.new_ones(inputs.shape)

    print(f"  input_len={inputs.shape[-1]}  VRAM before gen: {vram_mb()} MB", flush=True)
    t1 = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(
            inputs, attention_mask=attn,
            max_new_tokens=120,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    t2 = time.perf_counter()
    torch.cuda.synchronize()

    new_toks = out[0][inputs.shape[-1]:]
    raw = tok.decode(new_toks, skip_special_tokens=True).strip()
    gen_ms = int((t2 - t1) * 1000)
    ntoks  = len(new_toks)
    tps    = round(ntoks / max((t2 - t1), 0.001), 1)

    print(f"  gen_time={gen_ms}ms  ntoks={ntoks}  tok/s={tps}", flush=True)
    print(f"  VRAM after gen: {vram_mb()} MB", flush=True)
    print(f"  Raw: {repr(raw)}", flush=True)
    print(f"  _template_is_clean: {check(raw)}", flush=True)

print("\nALL RUNS COMPLETE", flush=True)
