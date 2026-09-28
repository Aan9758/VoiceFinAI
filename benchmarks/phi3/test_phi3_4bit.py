"""
test_phi3_4bit.py
Loads Phi-3-mini-4k-instruct in 4-bit on cuda:0.
Reports: VRAM before/after load, VRAM during generation, tokens/sec, raw output, clean check.
"""
import os, time, sys
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

import torch

def vram_mb():
    torch.cuda.synchronize()
    return round(torch.cuda.memory_allocated(0) / 1024**2, 1)

print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM total:  {round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)} GB", flush=True)
print(f"VRAM before model load: {vram_mb()} MB", flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    device_map="auto",              # auto works correctly for GPU 4-bit; dict form triggers .to() error
    trust_remote_code=True,
    attn_implementation="eager",
)
mdl.eval()
load_ms = int((time.perf_counter() - t0) * 1000)
print(f"Load time:  {load_ms}ms ({load_ms/1000:.1f}s)", flush=True)
print(f"VRAM after model load: {vram_mb()} MB", flush=True)

SYSTEM = """You are Aria, a Hinglish voice assistant for Indian mutual funds.
Write ONE short spoken reply — 1-2 sentences, under 35 words.
ABSOLUTE RULE: every fund name, return figure, NAV, expense ratio, min SIP must be a slot placeholder.
Never write a real number. Never write a real fund name.
Correct: {F1} ka 3-year return {F1.cagr3} raha hai, risk {F1.risk} hai.
Wrong:   Quant ka return 18.5% raha hai.
Output only the reply. No quotes. No preamble."""

CONTEXTS = [
    ("discover",
     "F1: {F1} | 3yr: {F1.cagr3} | risk: {F1.risk} | minSIP: {F1.minsip}\nF2: {F2} | 3yr: {F2.cagr3} | risk: {F2.risk} | minSIP: {F2.minsip}\nF3: {F3} | 3yr: {F3.cagr3} | risk: {F3.risk} | minSIP: {F3.minsip}",
     "User wants moderate equity funds. Recommend from F1 F2 F3 using slots only."),
    ("fund_detail",
     "F1: {F1} | NAV: {F1.nav} | 1yr: {F1.cagr1} | 3yr: {F1.cagr3} | risk: {F1.risk}",
     "User asked about F1 specifically. Describe it using its slots only."),
    ("compare",
     "F1: {F1} | risk: {F1.risk} | 3yr: {F1.cagr3}\nF2: {F2} | risk: {F2.risk} | 3yr: {F2.cagr3}",
     "Compare F1 and F2 on return and risk. One sentence each. Slots only."),
    ("discover_low",
     "F1: {F1} | 3yr: {F1.cagr3} | risk: {F1.risk} | minSIP: {F1.minsip}",
     "User wants low risk safe option. Recommend F1 using slots."),
]

import re
PLACEHOLDER = re.compile(r"\{([A-Z]\d*(?:\.[a-z0-9]+)?|U\.[a-z]+)\}")
BARE_DIGIT  = re.compile(r"\b\d+(?:\.\d+)?\b")
ALLOWED     = frozenset(str(i) for i in range(1, 11))

def check(raw):
    if raw is None: return "FAIL (None)"
    if not raw: return "FAIL (empty)"
    if len(raw) > 420: return "FAIL (too long: %d chars)" % len(raw)
    if not PLACEHOLDER.search(raw): return "FAIL (no {Fx} placeholder)"
    stripped = PLACEHOLDER.sub("SLOT", raw)
    for m in BARE_DIGIT.finditer(stripped):
        if m.group() not in ALLOWED:
            return "FAIL (bare digit: %r)" % m.group()
    return "PASS"

for i, (label, ctx, brief) in enumerate(CONTEXTS, 1):
    print(f"\n{'='*60}", flush=True)
    print(f"Run {i} — {label}", flush=True)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"{ctx}\n\n{brief}\n\nReply now using slots only."},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to("cuda:0")
    attn   = inputs.new_ones(inputs.shape)
    print(f"  input_len={inputs.shape[-1]}  VRAM before gen: {vram_mb()} MB", flush=True)

    t1 = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(
            inputs,
            attention_mask=attn,
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
    print(f"  Raw output: {repr(raw)}", flush=True)
    print(f"  _template_is_clean: {check(raw)}", flush=True)

print("\nALL RUNS COMPLETE", flush=True)
