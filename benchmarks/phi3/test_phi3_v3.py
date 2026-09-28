"""
test_phi3_v3.py
- Model already cached locally (~7GB)
- No device_map=auto — loads entirely into CPU RAM
- Full DynamicCache shim (seen_tokens, get_seq_length, get_max_length, get_usable_length)
- attn_implementation=eager
- attention_mask explicit
"""
import os, time, sys

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_orig_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, request, **kwargs)
requests.Session.send = _patched_send
print("requests patched")

# Full DynamicCache shim — all methods Phi-3 needs that 4.57 removed
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: getattr(self, "_seen_tokens", 0))
    print("shim: seen_tokens")
if not hasattr(DynamicCache, "get_seq_length"):
    def _gsl(self, layer_idx=0):
        if len(getattr(self, "key_cache", [])) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]
    DynamicCache.get_seq_length = _gsl
    print("shim: get_seq_length")
if not hasattr(DynamicCache, "get_max_length"):
    def _gml(self):
        if not getattr(self, "key_cache", []):
            return None
        return self.key_cache[0].shape[-2]
    DynamicCache.get_max_length = _gml
    print("shim: get_max_length")
if not hasattr(DynamicCache, "get_usable_length"):
    def _gul(self, new_seq_length, layer_idx=0):
        ml = self.get_max_length()
        pl = self.get_seq_length(layer_idx)
        if ml is not None and pl + new_seq_length > ml:
            return ml - new_seq_length
        return pl
    DynamicCache.get_usable_length = _gul
    print("shim: get_usable_length")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/Phi-3-mini-4k-instruct"
print(f"\nLoading {MODEL} into CPU RAM (no disk offload)...")
t_load_start = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    attn_implementation="eager",
)
mdl = mdl.to("cpu").eval()
t_load_end = time.perf_counter()
print(f"Model loaded in {(t_load_end - t_load_start)*1000:.0f}ms")

SYSTEM = """You are Aria, a Hinglish voice assistant for Indian mutual funds.
Write ONE short spoken reply — two sentences, under 40 words.
ABSOLUTE RULE: every fund name, return, NAV, expense ratio, min SIP must be a placeholder slot.
Never write a digit. Never write a fund name directly.
Correct: {F1} ka 3-year return {F1.cagr3} raha hai.
Wrong: Quant ka 3-year return 18.5% raha hai.
Output only the reply text. No quotes."""

CONTEXTS = [
    ("discover",
     "F1: {F1} | 3yr: {F1.cagr3} | risk: {F1.risk} | minSIP: {F1.minsip}\nF2: {F2} | 3yr: {F2.cagr3} | risk: {F2.risk}",
     "User wants moderate equity. Recommend F1 and F2 using slots only."),
    ("fund_detail",
     "F1: {F1} | NAV: {F1.nav} | 1yr: {F1.cagr1} | 3yr: {F1.cagr3} | risk: {F1.risk}",
     "User asked about F1 specifically. Describe it using its slots only."),
    ("compare",
     "F1: {F1} | risk: {F1.risk} | 3yr: {F1.cagr3}\nF2: {F2} | risk: {F2.risk} | 3yr: {F2.cagr3}",
     "Compare F1 and F2 on return and risk. One sentence each."),
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
    if len(raw) > 420: return "FAIL (too long: %d)" % len(raw)
    if not PLACEHOLDER.search(raw): return "FAIL (no {Fx} placeholder)"
    stripped = PLACEHOLDER.sub("SLOT", raw)
    for m in BARE_DIGIT.finditer(stripped):
        if m.group() not in ALLOWED:
            return "FAIL (bare digit: %r)" % m.group()
    return "PASS"

for i, (label, ctx, brief) in enumerate(CONTEXTS, 1):
    print("\n" + "="*60, flush=True)
    print(f"Run {i} — {label}", flush=True)
    msgs = [{"role":"system","content":SYSTEM}, {"role":"user","content":f"{ctx}\n\n{brief}\n\nReply now, slots only."}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    attn   = inputs.new_ones(inputs.shape)
    print(f"Starting generate for Run {i} (input_len={inputs.shape[-1]})...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(inputs, attention_mask=attn, max_new_tokens=120,
                           do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    t1 = time.perf_counter()
    new_toks = out[0][inputs.shape[-1]:]
    raw = tok.decode(new_toks, skip_special_tokens=True).strip()
    ms  = int((t1-t0)*1000)
    print(f"Time: {ms}ms  ({ms//max(len(new_toks),1)} ms/tok)  ntoks={len(new_toks)}", flush=True)
    print(f"Raw:  {repr(raw)}", flush=True)
    print(f"Clean: {check(raw)}", flush=True)

print("\nDONE", flush=True)

