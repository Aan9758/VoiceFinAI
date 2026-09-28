from core.pipeline import run_turn
from core import dialogue

def tts(text):
    return "tok-test"

queries = [
    ("low risk fund chahiye", "sid-A"),
    ("aggressive fund batao", "sid-A"),
    ("safe investment chahiye 2000 monthly", "sid-A"),
    ("quant small cap kaise hai", "sid-A"),
    ("high return ke liye kya best hai", "sid-A"),
    ("debt fund chahiye", "sid-A"),
    ("balanced fund option batao", "sid-A"),
    ("1000 rupee me SIP kahan karo", "sid-A"),
]

queries_B = [
    ("aggressive fund chahiye", "sid-B"),
    ("low risk chahiye", "sid-B"),
]

seen_voices = []
print("=== 8-query single session (sid-A) ===")
for q, sid in queries:
    r = run_turn(q, sid, tts)
    v = r.get("voice_text", "")[:80]
    intent = r.get("intent", {}).get("intent", "?")[:8] if isinstance(r.get("intent"), dict) else "?"
    dup = v[:40] in [s[:40] for s in seen_voices]
    print("[%s] [%s] %s" % (r["type"][:4], intent, v))
    if dup:
        print("  !! DUPLICATE OPENING DETECTED !!")
    seen_voices.append(v)

print()
print("=== Session isolation test ===")
for q, sid in queries_B:
    run_turn(q, sid, tts)

sessions = list(dialogue._SESSIONS.keys())
print("Active sessions:", sessions)
assert "sid-A" in sessions and "sid-B" in sessions, "Session isolation broken!"
assert dialogue._SESSIONS.get("sid-A") is not dialogue._SESSIONS.get("sid-B"), "Same object!"
print("PASS: sid-A and sid-B are distinct Session objects")
print("sid-A said_templates count:", len(dialogue._SESSIONS["sid-A"].said_templates))
print("sid-B said_templates count:", len(dialogue._SESSIONS["sid-B"].said_templates))
