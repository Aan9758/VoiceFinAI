from __future__ import annotations

import io
import json
import os
import ssl
import tempfile

import app as voice_app


def test_home_shows_text_input_and_manual_voice_control():
    page = voice_app.app.test_client().get("/")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert 'id="queryInput"' in body
    assert 'id="voiceButton"' in body
    assert "Your mic stays off until you choose Start conversation" in body
    assert "Start conversation" in body
    assert "audio.src='/static/greetings/mridul-welcome-hi.mp3'" in body
    assert "speechSynthesis.speak" not in body
    assert 'id="aud" class="voice-player" controls hidden' in body
    assert "getUserMedia" in body
    assert "window.addEventListener('load', async" not in body


def test_startup_greeting_audio_asset_is_served():
    response = voice_app.app.test_client().get("/static/greetings/mridul-welcome-hi.mp3")
    assert response.status_code == 200
    assert response.mimetype == "audio/mpeg"
    assert len(response.data) > 10_000


def test_health_reports_configuration_without_exposing_key(monkeypatch):
    monkeypatch.setattr(voice_app, "DEEPGRAM_KEY", "private-test-key")
    response = voice_app.app.test_client().get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok", "transcription_configured": True}
    assert b"private-test-key" not in response.data


def test_text_query_passes_query_and_session_to_pipeline(monkeypatch):
    seen = {}

    def fake_pipeline(query, session_id):
        seen.update(query=query, session_id=session_id)
        return {"type": "message", "voice_text": "Sample reply"}

    monkeypatch.setattr(voice_app, "_run_pipeline", fake_pipeline)
    response = voice_app.app.test_client().post(
        "/query", json={"query": "  show funds  ", "session_id": "session-test"}
    )
    assert response.status_code == 200
    assert response.json["voice_text"] == "Sample reply"
    assert seen == {"query": "show funds", "session_id": "session-test"}


def test_text_query_rejects_non_text_and_oversized_input():
    client = voice_app.app.test_client()
    assert client.post("/query", json={"query": ["unexpected"]}).status_code == 400
    assert client.post("/query", json={"query": "x" * 501}).status_code == 413


def test_audit_requires_and_scopes_session_id(monkeypatch):
    monkeypatch.setattr(
        voice_app,
        "audit_log",
        lambda: [
            {"session_id": "one", "entry_hash": "a"},
            {"session_id": "two", "entry_hash": "b"},
        ],
    )
    client = voice_app.app.test_client()
    assert client.get("/audit").status_code == 400
    assert client.get("/audit?session_id=one").json == [
        {"session_id": "one", "entry_hash": "a"}
    ]


def test_transcription_requires_audio_and_provider_key(monkeypatch):
    client = voice_app.app.test_client()
    monkeypatch.setattr(voice_app, "DEEPGRAM_KEY", "")
    assert client.post("/transcribe").status_code == 400
    response = client.post(
        "/transcribe", data={"audio": (io.BytesIO(b"audio"), "sample.webm")}
    )
    assert response.status_code == 503
    assert "use text input" in response.json["error"].lower()


def test_transcription_uses_verified_tls_and_parses_transcript(monkeypatch):
    observed = {}

    def fake_urlopen(request, *, context, timeout):
        observed.update(
            verify_mode=context.verify_mode,
            check_hostname=context.check_hostname,
            timeout=timeout,
            content_type=request.get_header("Content-type"),
        )
        return io.BytesIO(
            json.dumps(
                {"results": {"channels": [{"alternatives": [{"transcript": "hello"}]}]}}
            ).encode()
        )

    monkeypatch.setattr(voice_app, "DEEPGRAM_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transcript = voice_app._transcribe_with_deepgram(b"sample", "audio/webm")
    assert transcript == "hello"
    assert observed == {
        "verify_mode": ssl.CERT_REQUIRED,
        "check_hostname": True,
        "timeout": 30,
        "content_type": "audio/webm",
    }


def test_transcribe_route_returns_mocked_transcript_without_network(monkeypatch):
    monkeypatch.setattr(voice_app, "DEEPGRAM_KEY", "test-key")
    monkeypatch.setattr(voice_app, "_transcribe_with_deepgram", lambda audio, mime: "namaste")
    response = voice_app.app.test_client().post(
        "/transcribe", data={"audio": (io.BytesIO(b"recorded bytes"), "turn.webm")}
    )
    assert response.status_code == 200
    assert response.json == {"transcript": "namaste"}


def test_audio_download_removes_temporary_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio:
        audio.write(b"fake mp3")
        path = audio.name
    token = "test-audio-token"
    voice_app._audio_store[token] = path
    response = voice_app.app.test_client().get(f"/audio/{token}")
    assert response.status_code == 200
    assert response.data == b"fake mp3"
    response.close()
    assert token not in voice_app._audio_store
    assert not os.path.exists(path)


def test_audio_route_reports_provider_error_to_browser():
    token = "failed-audio-token"
    voice_app._audio_errors[token] = "Natural voice is unavailable on this plan."
    response = voice_app.app.test_client().get(f"/audio/{token}")
    assert response.status_code == 503
    assert response.json == {"error": "Natural voice is unavailable on this plan."}
    assert token not in voice_app._audio_errors


def test_audio_route_reports_pending_synthesis_without_expiring_token():
    token = "pending-audio-token"
    voice_app._audio_pending.add(token)
    response = voice_app.app.test_client().get(f"/audio/{token}")
    assert response.status_code == 202
    assert response.json == {"status": "pending"}
    assert token in voice_app._audio_pending
    voice_app._audio_pending.discard(token)
