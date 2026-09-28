"""Text-to-speech adapter for ElevenLabs natural voice and Edge TTS demo fallback."""

from __future__ import annotations

import asyncio
import json
import io
import logging
import os
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

VOICE             = "hi-IN-MadhurNeural"   # Default Indian Hindi/Hinglish male neural
CURRENT_VOICE     = os.getenv("TTS_VOICE", VOICE)
FILLER_THRESHOLD_MS = 800                  # edge-tts P50 ~1100ms — filler at 800ms

# Slower Edge fallback cadence avoids the rushed demo feel.
SPEECH_RATE = os.getenv("TTS_RATE", "+5%")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip()
TTS_PROVIDER = os.getenv(
    "TTS_PROVIDER",
    "elevenlabs" if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID else "edge",
).strip().lower()
TTS_FALLBACK_VOICE = os.getenv("TTS_FALLBACK_VOICE", "hi-IN-MadhurNeural").strip()
ELEVENLABS_BLOCKED = False


class SpeechProviderError(RuntimeError):
    """Safe, user-facing speech-provider failure."""

# Filler text — pre-generated once at startup
FILLER_TEXT = "Ek second, dekh raha hoon."

AVAILABLE_VOICES = [
    {
        "id": "en-IN-NeerjaNeural",
        "name": "Neerja",
        "gender": "Female",
        "accent": "Indian English / Hinglish",
        "tag": "Balanced & Clear",
        "desc": "Natural Indian English female voice with fluent Hinglish cadence (Default).",
        "sample_url": "/static/voice_previews/en-IN-NeerjaNeural.mp3"
    },
    {
        "id": "en-IN-NeerjaExpressiveNeural",
        "name": "Neerja Expressive",
        "gender": "Female",
        "accent": "Indian English / Hinglish",
        "tag": "Warm & Expressive",
        "desc": "Lively, expressive conversational tone with rich voice dynamics.",
        "sample_url": "/static/voice_previews/en-IN-NeerjaExpressiveNeural.mp3"
    },
    {
        "id": "en-IN-PrabhatNeural",
        "name": "Prabhat",
        "gender": "Male",
        "accent": "Indian English / Hinglish",
        "tag": "Confident Advisor",
        "desc": "Grounded, authoritative Indian English male financial advisor voice.",
        "sample_url": "/static/voice_previews/en-IN-PrabhatNeural.mp3"
    },
    {
        "id": "hi-IN-SwaraNeural",
        "name": "Swara",
        "gender": "Female",
        "accent": "Hindi / Hinglish",
        "tag": "Authentic Hindi",
        "desc": "Crisp and polite native Hindi cadence, excellent for Hindi & Hinglish.",
        "sample_url": "/static/voice_previews/hi-IN-SwaraNeural.mp3"
    },
    {
        "id": "hi-IN-MadhurNeural",
        "name": "Madhur",
        "gender": "Male",
        "accent": "Hindi / Hinglish",
        "tag": "Warm & Friendly",
        "desc": "Gentle, friendly Hindi male tone with natural Indian rhythm.",
        "sample_url": "/static/voice_previews/hi-IN-MadhurNeural.mp3"
    },
    {
        "id": "en-US-JennyNeural",
        "name": "Jenny",
        "gender": "Female",
        "accent": "US English",
        "tag": "Modern Digital",
        "desc": "Crisp, professional American English digital assistant.",
        "sample_url": "/static/voice_previews/en-US-JennyNeural.mp3"
    },
    {
        "id": "en-GB-RyanNeural",
        "name": "Ryan",
        "gender": "Male",
        "accent": "British English",
        "tag": "Polished & Formal",
        "desc": "Sophisticated British English voice with clear articulation.",
        "sample_url": "/static/voice_previews/en-GB-RyanNeural.mp3"
    },
]


def get_current_voice() -> str:
    return CURRENT_VOICE


def set_current_voice(voice_id: str) -> bool:
    global CURRENT_VOICE
    if any(v["id"] == voice_id for v in AVAILABLE_VOICES):
        CURRENT_VOICE = voice_id
        return True
    return False


async def _synthesize_to_file(text: str, output_path: str, voice: str | None = None) -> None:
    """Synthesize speech with the configured natural-voice provider."""
    import re

    cleaned = text.replace("\u2014", ", ").replace("\u2013", ", ").replace("--", ", ")
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if TTS_PROVIDER == "elevenlabs":
        if not ELEVENLABS_BLOCKED:
            try:
                _synthesize_elevenlabs(cleaned, output_path, voice or ELEVENLABS_VOICE_ID)
                return
            except SpeechProviderError as exc:
                logging.getLogger(__name__).warning(
                    "Natural speech provider unavailable; using the Indian male demo voice (%s)", exc
                )
        # A plan or permission rejection marks ElevenLabs unavailable for the
        # rest of this process. Every following turn must still use the same
        # Indian male fallback instead of failing as an "unknown provider".
        await _synthesize_edge(cleaned, output_path, TTS_FALLBACK_VOICE)
        return
    if TTS_PROVIDER != "edge":
        raise SpeechProviderError("Unknown speech provider. Set TTS_PROVIDER to elevenlabs or edge.")

    await _synthesize_edge(cleaned, output_path, voice or CURRENT_VOICE)


async def _synthesize_edge(text: str, output_path: str, voice_id: str) -> None:
    """Use Microsoft's Indian male neural voice as the reliable demo fallback."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice_id, rate=SPEECH_RATE)
    await communicate.save(output_path)


def _synthesize_elevenlabs(text: str, output_path: str, voice_id: str) -> None:
    """Generate expressive Hindi/Hinglish audio with the configured ElevenLabs voice."""
    if not ELEVENLABS_API_KEY or not voice_id:
        raise SpeechProviderError("Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID to enable natural voice.")

    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.42,
            "similarity_boost": 0.78,
            "style": 0.35,
            "use_speaker_boost": True,
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=45) as response:
            audio = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 402:
            global ELEVENLABS_BLOCKED
            ELEVENLABS_BLOCKED = True
            raise SpeechProviderError(
                "ElevenLabs does not allow this voice on the current plan. Enable API access for the voice or update the plan."
            ) from None
        if exc.code in (401, 403):
            ELEVENLABS_BLOCKED = True
            raise SpeechProviderError(
                "ElevenLabs rejected the key or its text-to-speech permission. Check the API key permissions."
            ) from None
        if exc.code == 429:
            raise SpeechProviderError("ElevenLabs is rate-limiting speech right now. Try again shortly.") from None
        raise SpeechProviderError(f"ElevenLabs speech generation failed (HTTP {exc.code}).") from None
    except (TimeoutError, OSError) as exc:
        raise SpeechProviderError("ElevenLabs could not be reached. Check the connection and try again.") from exc

    if not audio:
        raise SpeechProviderError("ElevenLabs returned an empty audio response.")
    with open(output_path, "wb") as output:
        output.write(audio)


async def speak(text: str, play: bool = True, voice: str | None = None) -> tuple[str, float]:
    """
    Synthesize text and optionally play it.
    Returns path to the generated audio file.

    If play=True, plays through the system's default audio player.
    TTFB is measured and logged.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name

    t0 = time.perf_counter()
    await _synthesize_to_file(text, output_path, voice=voice)
    ttfb_ms = (time.perf_counter() - t0) * 1000

    if play:
        _play_audio(output_path)

    return output_path, ttfb_ms


def _play_audio(path: str) -> None:
    """Play audio file using Windows built-in player."""
    import subprocess
    try:
        # Use Windows Media Player via PowerShell — no extra deps
        subprocess.Popen(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).wait()
    except Exception:
        # Fallback: open with default app
        os.startfile(path)


async def generate_filler(output_path: str | None = None) -> str:
    """
    Pre-generate the filler clip at session start.
    Returns path to the filler MP3.
    """
    if output_path is None:
        output_path = str(Path(tempfile.gettempdir()) / "voicefinai_filler.mp3")
    await _synthesize_to_file(FILLER_TEXT, output_path)
    return output_path
