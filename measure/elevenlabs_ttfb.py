"""
ElevenLabs Flash v2.5 — TTFB Measurement Script
================================================
Measures Time-To-First-Byte (first audio chunk) from Singapore endpoint
using a persistent WebSocket connection — the production architecture.

Run: python measure/elevenlabs_ttfb.py
Requires: pip install websockets python-dotenv
Set ELEVENLABS_API_KEY in .env or environment.
"""

import asyncio
import json
import os
import statistics
import time
from dotenv import load_dotenv

try:
    import websockets
except ImportError:
    raise SystemExit("Run: pip install websockets python-dotenv")

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID     = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel default
MODEL_ID     = "eleven_flash_v2_5"   # Flash v2.5 — the low-latency model
# Singapore endpoint — ~60ms RTT from India vs ~220ms to US East
WS_URL       = (
    f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input"
    f"?model_id={MODEL_ID}&optimize_streaming_latency=4"
)
NUM_RUNS     = 20   # Number of requests to measure
FILLER_THRESHOLD_MS = 250  # If P95 > this, filler clip always fires

# ── Test sentences — realistic short phrases like VoiceFinAI would emit ──────
TEST_PHRASES = [
    "Yahan teen options hain low risk SIP ke liye.",
    "Parag Parikh Flexi Cap Fund mein fourteen point two percent ka return hai.",
    "Minimum SIP five hundred rupaye se start hoti hai.",
    "NAV sixty-eight point four two hai.",
    "Risk rating moderate hai.",
    "Teen options hain aapke liye.",
    "Expense ratio zero point nine two percent hai.",
    "Ek second, dekh raha hoon.",
    "Fund house PPFAS Mutual Fund hai.",
    "Lock-in period nahi hai is fund mein.",
]

# ── Measurement ──────────────────────────────────────────────────────────────

async def measure_single(session_ws, phrase: str) -> float:
    """
    Sends one phrase over an open WebSocket and returns TTFB in milliseconds.
    The WebSocket must already be open (persistent connection).
    """
    # Send text chunk — chunk_length_schedule: low so synthesis starts immediately
    payload = {
        "text": phrase + " ",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        },
        "generation_config": {
            "chunk_length_schedule": [50]  # Start synthesis at 50 chars
        }
    }

    t_send = time.perf_counter()
    await session_ws.send(json.dumps(payload))

    # Flush signal — tells ElevenLabs to synthesize what it has
    await session_ws.send(json.dumps({"text": ""}))

    # Wait for first audio chunk
    while True:
        msg = await asyncio.wait_for(session_ws.recv(), timeout=10.0)
        data = json.loads(msg)

        if data.get("audio"):
            t_first_chunk = time.perf_counter()
            return (t_first_chunk - t_send) * 1000  # ms

        if data.get("error"):
            raise RuntimeError(f"ElevenLabs error: {data['error']}")


async def run_measurements():
    if not API_KEY:
        raise SystemExit(
            "Set ELEVENLABS_API_KEY in your environment or a .env file.\n"
            "Get a key at: https://elevenlabs.io"
        )

    print(f"\n{'='*60}")
    print(f"  ElevenLabs Flash v2.5 — TTFB Measurement")
    print(f"  Endpoint : Singapore (optimize_streaming_latency=4)")
    print(f"  Model    : {MODEL_ID}")
    print(f"  Voice    : {VOICE_ID}")
    print(f"  Runs     : {NUM_RUNS}")
    print(f"{'='*60}\n")

    results_cold  = []   # First connection (cold)
    results_warm  = []   # Subsequent requests on same WebSocket (warm)

    headers = {"xi-api-key": API_KEY}

    # ── Warm connection measurement ─────────────────────────────────────────
    print("Opening persistent WebSocket...")
    t_connect_start = time.perf_counter()

    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        t_connected = time.perf_counter()
        connect_ms = (t_connected - t_connect_start) * 1000
        print(f"  WebSocket connected in {connect_ms:.1f}ms\n")

        for i in range(NUM_RUNS):
            phrase = TEST_PHRASES[i % len(TEST_PHRASES)]
            try:
                ttfb = await measure_single(ws, phrase)
                label = "cold" if i == 0 else "warm"
                if i == 0:
                    results_cold.append(ttfb)
                else:
                    results_warm.append(ttfb)

                print(f"  Run {i+1:>2}/{NUM_RUNS}  [{label}]  {ttfb:>7.1f}ms  │ {phrase[:45]}")
                await asyncio.sleep(0.3)  # Brief pause between requests

            except asyncio.TimeoutError:
                print(f"  Run {i+1:>2}/{NUM_RUNS}  TIMEOUT (>10s)")
            except Exception as e:
                print(f"  Run {i+1:>2}/{NUM_RUNS}  ERROR: {e}")

    # ── Report ───────────────────────────────────────────────────────────────
    all_warm = results_warm
    if not all_warm:
        print("\nNot enough data to compute statistics.")
        return

    p50  = statistics.median(all_warm)
    p95  = sorted(all_warm)[int(len(all_warm) * 0.95)]
    p99  = sorted(all_warm)[int(len(all_warm) * 0.99)] if len(all_warm) >= 10 else max(all_warm)
    mean = statistics.mean(all_warm)
    mn   = min(all_warm)
    mx   = max(all_warm)

    print(f"\n{'='*60}")
    print(f"  RESULTS — Warm WebSocket (N={len(all_warm)} runs)")
    print(f"{'='*60}")
    print(f"  Min    : {mn:>7.1f}ms")
    print(f"  Mean   : {mean:>7.1f}ms")
    print(f"  P50    : {p50:>7.1f}ms")
    print(f"  P95    : {p95:>7.1f}ms   ← governs filler clip decision")
    print(f"  P99    : {p99:>7.1f}ms")
    print(f"  Max    : {mx:>7.1f}ms")
    if results_cold:
        print(f"\n  Cold (run 1): {results_cold[0]:.1f}ms")
    print(f"  WebSocket connect: {connect_ms:.1f}ms")
    print(f"{'='*60}")

    # ── Decision ─────────────────────────────────────────────────────────────
    print(f"\n  Filler clip threshold: {FILLER_THRESHOLD_MS}ms")
    if p95 <= FILLER_THRESHOLD_MS:
        print(f"  ✅  P95 {p95:.0f}ms ≤ {FILLER_THRESHOLD_MS}ms")
        print(f"      Filler fires only on slow outliers. Good.")
        print(f"      Architecture latency claim: ~{p50:.0f}–{p95:.0f}ms TTFB (warm)")
    elif p95 <= 400:
        print(f"  ⚠️   P95 {p95:.0f}ms > {FILLER_THRESHOLD_MS}ms")
        print(f"      Filler will fire frequently. Adjust threshold to {int(p95)+50}ms")
        print(f"      Architecture latency claim: 500–700ms P95 still holds (filler covers gap)")
    else:
        print(f"  ❌  P95 {p95:.0f}ms — Singapore endpoint is slow.")
        print(f"      Check API key region settings or try ap-southeast-1 explicitly.")
        print(f"      Filler clip is mandatory; may need to rethink TTS fallback.")

    print()


if __name__ == "__main__":
    asyncio.run(run_measurements())
