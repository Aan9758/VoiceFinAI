"""
measure/edgetts_ttfb.py — edge-tts TTFB Measurement
=====================================================
Measures Time-To-First-Byte for edge-tts (free, no account needed).
Uses en-IN-NeerjaNeural — Indian English neural voice.

Run: python measure/edgetts_ttfb.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

import edge_tts

VOICE    = "en-IN-NeerjaNeural"   # Indian English female — best for Hinglish frames
NUM_RUNS = 15
FILLER_THRESHOLD_MS = 250

TEST_PHRASES = [
    "Parag Parikh Flexi Cap mein twelve point three percent ka return hai.",
    "Yahan teen options hain aapke liye.",
    "Minimum SIP one thousand rupaye se start hoti hai.",
    "NAV eighty-nine point five seven hai.",
    "Risk Moderately High hai.",
    "Ek second, dekh raha hoon.",
    "Five thousand rupaye monthly invest karna ek achha option hai.",
    "Quant Small Cap mein eighteen point five percent ka return hai.",
    "Koi lock-in nahi hai is fund mein.",
    "Yeh mere domain se bahar hai.",
]


async def measure_single(phrase: str) -> float:
    """Returns TTFB in milliseconds — time to first audio chunk."""
    communicate = edge_tts.Communicate(phrase, VOICE)
    t_send = time.perf_counter()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio" and chunk["data"]:
            t_first = time.perf_counter()
            return (t_first - t_send) * 1000

    return float("inf")   # No audio received


async def run_measurements():
    print(f"\n{'='*60}")
    print(f"  edge-tts TTFB Measurement")
    print(f"  Voice  : {VOICE}")
    print(f"  Runs   : {NUM_RUNS}")
    print(f"{'='*60}\n")

    results = []

    for i in range(NUM_RUNS):
        phrase = TEST_PHRASES[i % len(TEST_PHRASES)]
        try:
            ttfb = await measure_single(phrase)
            results.append(ttfb)
            print(f"  Run {i+1:>2}/{NUM_RUNS}  {ttfb:>7.1f}ms  │ {phrase[:50]}")
        except Exception as e:
            print(f"  Run {i+1:>2}/{NUM_RUNS}  ERROR: {e}")

    if not results:
        print("\nNo data collected.")
        return

    p50 = statistics.median(results)
    p95 = sorted(results)[int(len(results) * 0.95)]
    mn  = min(results)
    mx  = max(results)
    avg = statistics.mean(results)

    print(f"\n{'='*60}")
    print(f"  RESULTS (N={len(results)})")
    print(f"{'='*60}")
    print(f"  Min    : {mn:>7.1f}ms")
    print(f"  Mean   : {avg:>7.1f}ms")
    print(f"  P50    : {p50:>7.1f}ms")
    print(f"  P95    : {p95:>7.1f}ms   ← governs filler clip decision")
    print(f"  Max    : {mx:>7.1f}ms")
    print(f"{'='*60}")

    print(f"\n  Filler clip threshold : {FILLER_THRESHOLD_MS}ms")
    if p95 <= FILLER_THRESHOLD_MS:
        print(f"  ✅  P95 {p95:.0f}ms — filler fires on outliers only")
    elif p95 <= 500:
        print(f"  ⚠️   P95 {p95:.0f}ms — filler fires frequently. Adjust threshold.")
        print(f"       Recommended threshold: {int(p95)+50}ms")
    else:
        print(f"  ❌  P95 {p95:.0f}ms — too slow for production.")
        print(f"       edge-tts acceptable for demo recording only.")
    print()


if __name__ == "__main__":
    asyncio.run(run_measurements())
