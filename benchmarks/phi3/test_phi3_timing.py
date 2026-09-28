"""
test_phi3_timing.py — Minimal sanity check.
Generates only 5 tokens to measure ms/tok and confirm the path works.
"""
import os, time

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_orig = requests.Session.send
def _nv(self, r, **kw): kw["verify"] = False; return _orig(self, r, **kw)
requests.Session.send = _nv

from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda s: getattr(s, "_seen_tokens", 0))
if not hasattr(DynamicCache, "get_seq_length"):
    def _gsl(s, l=0): return s.key_cache[l].shape[-2] if len(getattr(s,"key_cache",[]))>l else 0
    DynamicCache.get_seq_length = _gsl
if not hasattr(DynamicCache, "get_max_length"):
    def _gml(s): return s.key_cache[0].shape[-2] if getattr(s,"key_cache",[]) else None
    DynamicCache.get_max_length = _gml
if not hasattr(DynamicCache, "get_usable_length"):
    def _gul(s, n, l=0):
        ml = s.get_max_length(); pl = s.get_seq_length(l)
        return ml - n if ml is not None and pl + n > ml else pl
    DynamicCache.get_usable_length = _gul
print("shims applied")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/Phi-3-mini-4k-instruct"
print(f"Loading {MODEL}...")
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, trust_remote_code=True, attn_implementation="eager"
).to("cpu").eval()
load_ms = int((time.perf_counter() - t0) * 1000)
print(f"Load time: {load_ms}ms ({load_ms/1000:.1f}s)")

# Minimal prompt — just enough to trigger generation
msgs = [
    {"role": "system", "content": "You recommend mutual funds using slot placeholders only."},
    {"role": "user",   "content": "Recommend {F1}.\n\nReply now."},
]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
attn   = inputs.new_ones(inputs.shape)
print(f"Input tokens: {inputs.shape[-1]}")

print("Generating 5 tokens...")
t1 = time.perf_counter()
with torch.no_grad():
    out = mdl.generate(
        inputs, attention_mask=attn,
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
gen_ms = int((time.perf_counter() - t1) * 1000)
new_toks = out[0][inputs.shape[-1]:]
raw = tok.decode(new_toks, skip_special_tokens=True)
print(f"Generated {len(new_toks)} tokens in {gen_ms}ms")
print(f"ms/tok: {gen_ms // max(len(new_toks), 1)}")
print(f"Raw output: {repr(raw)}")
print(f"Extrapolated time for 120 tokens: {(gen_ms / max(len(new_toks),1) * 120 / 1000):.0f}s")
print("DONE")
