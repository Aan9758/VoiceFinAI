"""
test_phi3_ssl.py — Phi-3-mini inference test with SSL bypass.
Must be imported/run BEFORE any huggingface_hub or transformers code.
"""
import os, time, sys

# ── 1. Env vars (belt) ────────────────────────────────────────────────────────
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ── 2. Monkey-patch requests (suspenders) ─────────────────────────────────────
# Must happen before huggingface_hub is imported — it creates sessions at
# module level. Patching Session.send forces verify=False on every call.
import requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_orig_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, request, **kwargs)
requests.Session.send = _patched_send
print("requests SSL patched")

# ── 3. Now import generate (which will import transformers lazily) ─────────────
import core.generate as gen
# Reset any prior failed-flag from earlier attempt in this session
gen._LOCAL_LOAD_FAILED = False
gen._LOCAL_MODEL = None
gen._LOCAL_TOKENIZER = None

from core.generate import _call_local_model, _template_is_clean, _BARE_DIGIT, _PLACEHOLDER

def check_clean(raw):
    if raw is None:
        return "FAIL (None returned)"
    if not raw or len(raw) > 420:
        return "FAIL (empty or len=%d > 420)" % len(raw)
    if not _PLACEHOLDER.search(raw):
        return "FAIL (no {Fx} placeholder found in output)"
    from core.generate import _ALLOWED_CARDINALS
    stripped = _PLACEHOLDER.sub("SLOT", raw)
    for m in _BARE_DIGIT.finditer(stripped):
        if m.group() not in _ALLOWED_CARDINALS:
            return "FAIL (bare digit found: %r)" % m.group()
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

t_script_start = time.perf_counter()
first_run = True

for i, (label, ctx, brief) in enumerate(runs, 1):
    print("=" * 60)
    print("Run %d — intent: %s" % (i, label))
    t0 = time.perf_counter()
    raw = _call_local_model(ctx, brief, [])
    t1 = time.perf_counter()
    elapsed_ms = int((t1 - t0) * 1000)

    if first_run:
        total_ms = int((t1 - t_script_start) * 1000)
        print("Wall-clock incl. model download+load: %dms (%.1fs)" % (total_ms, total_ms/1000))
        first_run = False

    print("Generation time: %dms" % elapsed_ms)
    print("Raw output repr: %r" % raw)
    result = check_clean(raw)
    print("_template_is_clean: %s" % result)
    print()

print("ALL RUNS COMPLETE")
