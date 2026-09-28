from __future__ import annotations

import io
import json
import ssl
import urllib.error
import asyncio

import pytest

import core.tts as tts


def test_elevenlabs_synthesis_saves_audio_and_uses_natural_voice_settings(monkeypatch):
    observed = {}
    written = {}

    class FakeOutput:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, audio):
            written["audio"] = audio

    def fake_urlopen(request, *, context, timeout):
        observed.update(
            url=request.full_url,
            payload=json.loads(request.data),
            tls=context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname,
            timeout=timeout,
        )
        return io.BytesIO(b"mock-mp3-audio")

    monkeypatch.setattr(tts, "ELEVENLABS_API_KEY", "test-api-key")
    monkeypatch.setattr(tts, "ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    monkeypatch.setattr(tts, "ELEVENLABS_BLOCKED", False)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: FakeOutput())

    tts._synthesize_elevenlabs("Namaste", "voice.mp3", "voice-test")

    assert written["audio"] == b"mock-mp3-audio"
    assert observed["url"].endswith("/voice-test?output_format=mp3_44100_128")
    assert observed["payload"]["text"] == "Namaste"
    assert observed["payload"]["model_id"] == "eleven_multilingual_v2"
    assert observed["payload"]["voice_settings"]["stability"] < 0.5
    assert observed["tls"] is True
    assert observed["timeout"] == 45


def test_elevenlabs_plan_error_is_explained_without_edge_fallback(monkeypatch):
    def rejected(request, *, context, timeout):
        raise urllib.error.HTTPError(request.full_url, 402, "payment required", {}, io.BytesIO())

    monkeypatch.setattr(tts, "ELEVENLABS_API_KEY", "test-api-key")
    monkeypatch.setattr(tts, "ELEVENLABS_BLOCKED", False)
    monkeypatch.setattr("urllib.request.urlopen", rejected)

    with pytest.raises(tts.SpeechProviderError, match="current plan"):
        tts._synthesize_elevenlabs("Namaste", "voice.mp3", "voice-test")


def test_elevenlabs_plan_error_falls_back_to_indian_male_voice(monkeypatch):
    observed = {}

    def reject(*_args):
        raise tts.SpeechProviderError("voice unavailable on current plan")

    async def fake_edge(text, path, voice_id):
        observed.update(text=text, path=path, voice_id=voice_id)

    monkeypatch.setattr(tts, "TTS_PROVIDER", "elevenlabs")
    monkeypatch.setattr(tts, "TTS_FALLBACK_VOICE", "hi-IN-MadhurNeural")
    monkeypatch.setattr(tts, "ELEVENLABS_BLOCKED", False)
    monkeypatch.setattr(tts, "_synthesize_elevenlabs", reject)
    monkeypatch.setattr(tts, "_synthesize_edge", fake_edge)

    asyncio.run(tts._synthesize_to_file("Namaste", "voice.mp3"))

    assert observed["text"] == "Namaste"
    assert observed["voice_id"] == "hi-IN-MadhurNeural"


def test_blocked_elevenlabs_keeps_using_indian_male_fallback(monkeypatch):
    observed = {}

    async def fake_edge(text, path, voice_id):
        observed.update(text=text, path=path, voice_id=voice_id)

    def should_not_retry_elevenlabs(*_args):
        raise AssertionError("ElevenLabs must not be retried after a plan rejection")

    monkeypatch.setattr(tts, "TTS_PROVIDER", "elevenlabs")
    monkeypatch.setattr(tts, "TTS_FALLBACK_VOICE", "hi-IN-MadhurNeural")
    monkeypatch.setattr(tts, "ELEVENLABS_BLOCKED", True)
    monkeypatch.setattr(tts, "_synthesize_elevenlabs", should_not_retry_elevenlabs)
    monkeypatch.setattr(tts, "_synthesize_edge", fake_edge)

    asyncio.run(tts._synthesize_to_file("Second answer", "voice.mp3"))

    assert observed == {
        "text": "Second answer",
        "path": "voice.mp3",
        "voice_id": "hi-IN-MadhurNeural",
    }
