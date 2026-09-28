"""
test_phi3_v2.py — Re-run with DynamicCache shim. Model already cached locally.
"""
import os, time, sys

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Patch requests for SSL (needed even for cached load — HF checks for updates)
import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_orig_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, request, **kwargs)
requests.Session.send = _patched_send
print("requests SSL patched")

# Apply ALL DynamicCache shims BEFORE importing generate or transformers
try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: getattr(self, "_seen_tokens", 0))
        print("shim: seen_tokens")

    if not hasattr(DynamicCache, "get_seq_length"):
        def _get_seq_length(self, layer_idx=0):
            if len(getattr(self, "key_cache", [])) <= layer_idx:
                return 0
            return self.key_cache[layer_idx].shape[-2]
        DynamicCache.get_seq_length = _get_seq_length
        print("shim: get_seq_length")

    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            if not getattr(self, "key_cache", []):
                return None
            return self.key_cache[0].shape[-2]
        DynamicCache.get_max_length = _get_max_length
        print("shim: get_max_length")

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            max_length = self.get_max_length()
            prev_length = self.get_seq_length(layer_idx)
            if max_length is not None and prev_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return prev_length
        DynamicCache.get_usable_length = _get_usable_length
        print("shim: get_usable_length")

except Exception as e:
    print("DynamicCache shim error:", e)

import core.generate as gen
gen._LOCAL_LOAD_FAILED = False
gen._LOCAL_MODEL = None
gen._LOCAL_TOKENIZER = None

from core.generate import _call_local_model, _template_is_clean, _BARE_DIGIT, _PLACEHOLDER

def check_clean(raw):
    if raw is None:
        return "FAIL — None returned"
    if not raw:
        return "FAIL — empty string"
    if len(raw) > 420:
        return "FAIL — len=%d > 420 char limit" % len(raw)
    if not _PLACEHOLDER.search(raw):
        return "FAIL — no {Fx} placeholder in output"
    from core.generate import _ALLOWED_CARDINALS
    stripped = _PLACEHOLDER.sub("SLOT", raw)
    for m in _BARE_DIGIT.finditer(stripped):
        if m.group() not in _ALLOWED_CARDINALS:
            return "FAIL — bare digit found: %r" % m.group()
    return "PASS"

CTX_DISCOVER = (
    "## Retrieved Funds\n"
    "F1: {F1} | 3yr: {F1.cagr3} | risk: {F1.risk} | min SIP: {F1.minsip}\n"
    "F2: {F2} | 3yr: {F2.cagr3} | risk: {F2.risk} | min SIP: {F2.minsip}\n"
    "F3: {F3} | 3yr: {F3.cagr3} | risk: {F3.risk} | min SIP: {F3.minsip}\n"
    "User amount: {U.amount}"
)
CTX_DETAIL = (
    "## Fund Detail\n"
    "F1: {F1} | NAV: {F1.nav} | 1yr: {F1.cagr1} | 3yr: {F1.cagr3} | risk: {F1.risk}\n"
    "User asked specifically about this fund."
)
CTX_COMPARE = (
    "## Comparison\n"
    "F1: {F1} | risk: {F1.risk} | 3yr: {F1.cagr3}\n"
    "F2: {F2} | risk: {F2.risk} | 3yr: {F2.cagr3}\n"
    "User wants to compare these two."
)

runs = [
    ("discover",     CTX_DISCOVER, "User wants moderate equity growth funds. Recommend from slots only. 2-3 sentences, spoken Hinglish."),
    ("fund_detail",  CTX_DETAIL,   "User asked about one specific fund. Talk about F1 only using its slots."),
    ("compare",      CTX_COMPARE,  "User asked to compare. Contrast F1 and F2 on return and risk. Plain spoken Hinglish."),
    ("discover_low", CTX_DISCOVER, "User wants low risk safe investment. Recommend from slots. Mention risk and min SIP."),
]

t_start = time.perf_counter()
first_run = True

for i, (label, ctx, brief) in enumerate(runs, 1):
    print("=" * 60)
    print("Run %d — intent: %s" % (i, label))
    t0 = time.perf_counter()
    raw = _call_local_model(ctx, brief, [])
    t1 = time.perf_counter()
    elapsed_ms = int((t1 - t0) * 1000)

    if first_run:
        total_ms = int((t1 - t_start) * 1000)
        print("Wall-clock incl. model load (weights already cached): %dms (%.1fs)" % (total_ms, total_ms / 1000))
        first_run = False

    print("Generation time: %dms" % elapsed_ms)
    print("Raw output repr:")
    print("  %r" % raw)
    result = check_clean(raw)
    print("_template_is_clean: %s" % result)
    print()

print("ALL RUNS COMPLETE")
