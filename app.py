"""Flask web app for the VoiceFinAI mutual-fund information demo.

The current browser records one utterance, posts it to ``/transcribe`` for
Deepgram transcription, sends the text to ``/query``, then displays the result
and plays server-generated speech. ``/ws/transcribe`` remains an experimental
transport and is not used by the current page.
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import sys
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_sock import Sock

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from core import dialogue
from core.pipeline import run_turn, audit_log
from core.tts import (
    speak, AVAILABLE_VOICES, get_current_voice, set_current_voice,
    SpeechProviderError,
)

app  = Flask(__name__)
sock = Sock(app)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# Temp audio store — {token: filepath}. Files are deleted after the client fetches them.
_audio_store: dict[str, str] = {}
_audio_errors: dict[str, str] = {}
_audio_pending: set[str] = set()



# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The request is too large. Please use a shorter query or recording."}), 413


@app.route("/health")
def health():
    return jsonify({"status": "ok", "transcription_configured": bool(DEEPGRAM_KEY)})


@app.route("/api/voices", methods=["GET"])
def get_voices():
    """List all available voices and the currently active voice."""
    return jsonify({
        "current_voice": get_current_voice(),
        "voices": AVAILABLE_VOICES,
    })


@app.route("/api/set_voice", methods=["POST"])
def set_voice_endpoint():
    """Set the active voice."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    v_id = data.get("voice", "")
    if not isinstance(v_id, str):
        return jsonify({"error": "voice must be text"}), 400
    v_id = v_id.strip()
    if not v_id:
        return jsonify({"error": "no voice specified"}), 400
    if set_current_voice(v_id):
        return jsonify({"status": "ok", "current_voice": get_current_voice()})
    return jsonify({"error": f"unknown voice: {v_id}"}), 400


@app.route("/audio/<token>")
def serve_audio(token: str):
    error = _audio_errors.pop(token, None)
    if error:
        _audio_pending.discard(token)
        return jsonify({"error": error}), 503
    path = _audio_store.get(token)
    if not path:
        if token in _audio_pending:
            return jsonify({"status": "pending"}), 202
        return jsonify({"error": "audio expired"}), 404
    if not os.path.exists(path):
        return jsonify({"status": "pending"}), 202

    def stream_audio():
        try:
            with open(path, "rb") as audio:
                while chunk := audio.read(64 * 1024):
                    yield chunk
        finally:
            _discard_audio(token, path)

    return Response(
        stream_audio(),
        mimetype="audio/mpeg",
        headers={"Content-Length": str(os.path.getsize(path))},
    )


def _discard_audio(token: str, path: str) -> None:
    if _audio_store.get(token) == path:
        _audio_store.pop(token, None)
        try:
            os.remove(path)
        except OSError:
            app.logger.warning("Could not remove temporary audio file")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """Synthesize a short response for the browser audio player."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    text = data.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "text must be a string"}), 400
    text = text.strip()
    voice = (data or {}).get("voice")
    if not text:
        return jsonify({"error": "no text"}), 400
    if len(text) > 1200:
        return jsonify({"error": "text must be 1,200 characters or fewer"}), 413
    if voice is not None and (
        not isinstance(voice, str)
        or voice not in {item["id"] for item in AVAILABLE_VOICES}
    ):
        return jsonify({"error": "unknown voice"}), 400
    token = _synthesize_async(text, voice=voice)
    return jsonify({"token": token})


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Receive audio blob from browser, send to Deepgram REST API server-side.
    Returns: { "transcript": "..." }
    """
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "no audio"}), 400
    if not DEEPGRAM_KEY:
        return jsonify({"error": "Voice transcription is not configured. Add a Deepgram API key or use text input."}), 503

    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error": "The audio recording was empty. Please try again."}), 400
    mime_type   = audio_file.mimetype or "audio/webm"

    try:
        transcript = _transcribe_with_deepgram(audio_bytes, mime_type)
        return jsonify({"transcript": transcript})
    except Exception:
        app.logger.exception("Speech transcription failed")
        return jsonify({"error": "Speech recognition failed. Please try again or type your question."}), 502


def _transcribe_with_deepgram(audio_bytes: bytes, mime_type: str) -> str:
    """Send one recorded utterance to Deepgram using verified TLS."""
    import urllib.request as urlreq

    dg_url = (
        "https://api.deepgram.com/v1/listen"
        "?model=nova-3&language=multi&smart_format=true&numerals=true"
        "&keyterm=SIP&keyterm=NAV&keyterm=ELSS&keyterm=mutual+fund"
    )

    req = urlreq.Request(
        dg_url,
        data    = audio_bytes,
        headers = {
            "Authorization": f"Token {DEEPGRAM_KEY}",
            "Content-Type" : mime_type,
        },
        method  = "POST",
    )

    with urlreq.urlopen(req, context=ssl.create_default_context(), timeout=30) as r:
        result = json.loads(r.read())
    return (
        result.get("results", {})
              .get("channels", [{}])[0]
              .get("alternatives", [{}])[0]
              .get("transcript", "")
              .strip()
    )


@app.route("/query", methods=["POST"])
def text_query():
    """Text query — session-aware via session_id field."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    query = data.get("query", "")
    if not isinstance(query, str):
        return jsonify({"error": "query must be text"}), 400
    query = query.strip()
    if len(query) > 500:
        return jsonify({"error": "query must be 500 characters or fewer"}), 413
    if not query or query == "__greet__":
        return jsonify({"type": "noop"})
    sid = data.get("session_id") or request.remote_addr or "default"
    if not isinstance(sid, str) or len(sid) > 128:
        return jsonify({"error": "invalid session id"}), 400
    lang = data.get("language")
    if lang and lang in ("english", "hinglish", "auto"):
        sess = dialogue.get_session(sid)
        sess.language = lang
    result = _run_pipeline(query, sid)
    return jsonify(result)


@app.route("/audit")
def audit():
    """Return audit entries for one client session only."""
    session_id = request.args.get("session_id", "").strip()
    if not session_id or len(session_id) > 128:
        return jsonify({"error": "session_id is required"}), 400
    return jsonify([entry for entry in audit_log() if entry.get("session_id") == session_id])


# ── WebSocket — Track 1: Deepgram STT ────────────────────────────────────────

@sock.route("/ws/transcribe")
def transcribe(ws):
    """
    Track 1: Browser streams raw audio → we forward to Deepgram Nova-3.
    Deepgram returns word tokens, pipeline fires on speech_final/UtteranceEnd.
    Session ID is sent by the browser as the first message: {"type":"init","session_id":"..."}.

    Message protocol (server → browser):
      {"type": "transcript", "text": "...", "is_final": bool}
      {"type": "result",     ...fund card data + audio_token}
      {"type": "error",      "message": "..."}
    """
    import websockets.sync.client as wsc

    # ── Read session ID from browser's first message ─────────────────────────
    # Browser sends {"type":"init","session_id":"<uuid>"} immediately on connect.
    # If the first message is audio bytes (old client) we fall back to IP.
    session_id = None
    try:
        first = ws.receive(timeout=2)
        if isinstance(first, str):
            data = json.loads(first)
            if data.get("type") == "init":
                session_id = data.get("session_id", "").strip() or None
                lang = data.get("language")
                if lang and session_id and lang in ("english", "hinglish", "auto"):
                    sess = dialogue.get_session(session_id)
                    sess.language = lang
    except Exception:
        pass
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    dg_url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?model=nova-3"
        f"&language=multi"
        f"&encoding=webm-opus"
        f"&sample_rate=48000"
        f"&channels=1"
        f"&smart_format=true"
        f"&numerals=true"
        f"&interim_results=true"
        f"&utterance_end_ms=1200"
        f"&endpointing=300"
        f"&keyterm=SIP&keyterm=ELSS&keyterm=NAV&keyterm=ULIP&keyterm=ARN"
        f"&keyterm=mutual+fund&keyterm=flexi+cap&keyterm=small+cap"
    )
    dg_headers = {"Authorization": f"Token {DEEPGRAM_KEY}"}

    ssl_ctx = ssl.create_default_context()

    full_transcript = ""
    pipeline_fired  = False
    pipeline_lock   = threading.Lock()

    try:
        with wsc.connect(dg_url, additional_headers=dg_headers, ssl=ssl_ctx) as dg_ws:

            def audio_sender():
                try:
                    while True:
                        chunk = ws.receive()
                        if chunk is None:
                            break
                        if isinstance(chunk, bytes):
                            dg_ws.send(chunk)
                        elif chunk == "STOP":
                            dg_ws.send(json.dumps({"type": "CloseStream"}))
                            break
                except Exception:
                    pass

            sender_thread = threading.Thread(target=audio_sender, daemon=True)
            sender_thread.start()

            for raw in dg_ws:
                msg = json.loads(raw)

                if msg.get("type") == "Results":
                    alt = (msg.get("channel", {})
                               .get("alternatives", [{}])[0])
                    text         = alt.get("transcript", "").strip()
                    is_final     = msg.get("is_final", False)
                    speech_final = msg.get("speech_final", False)

                    if not text:
                        continue

                    if is_final:
                        full_transcript = (full_transcript + " " + text).strip()

                    display_text = full_transcript if is_final else (full_transcript + " " + text).strip()

                    ws.send(json.dumps({
                        "type"    : "transcript",
                        "text"    : display_text,
                        "is_final": is_final,
                    }))

                    # Fire pipeline once per utterance on speech_final
                    if speech_final and not pipeline_fired:
                        with pipeline_lock:
                            if not pipeline_fired:
                                pipeline_fired = True
                        result = _run_pipeline(display_text, session_id)
                        ws.send(json.dumps({"type": "result", **result}))

                elif msg.get("type") == "UtteranceEnd":
                    if full_transcript and not pipeline_fired:
                        with pipeline_lock:
                            if not pipeline_fired:
                                pipeline_fired = True
                        result = _run_pipeline(full_transcript, session_id)
                        ws.send(json.dumps({"type": "result", **result}))

    except Exception as e:
        try:
            ws.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(query: str, session_id: str = "default") -> dict:
    """Delegate to core.pipeline.run_turn (session-aware, language-layer enabled)."""
    return run_turn(query, session_id, _synthesize_async)


def _synthesize_async(text: str, voice: str | None = None) -> str:
    token = uuid.uuid4().hex
    _audio_pending.add(token)
    def _worker():
        try:
            path, _ = asyncio.run(speak(text, play=False, voice=voice))
            _audio_store[token] = path
        except SpeechProviderError as exc:
            _audio_errors[token] = str(exc)
        except Exception:
            _audio_errors[token] = "Voice generation failed. Please try again."
            app.logger.exception("Speech synthesis failed")
        finally:
            _audio_pending.discard(token)
    threading.Thread(target=_worker, daemon=True).start()
    return token



def _warm_cache() -> None:
    """
    Background thread: pre-fetch NAV data for every fund in the universe
    that isn't already cached. Runs once at startup with throttling so it
    doesn't hammer mfapi.in. Each fetched record is saved to mfapi_cache.json
    so epoch rotation always has enough valid funds to choose from.
    """
    import time as _time
    _time.sleep(2)  # let Flask finish booting first
    try:
        from core import universe
        from core.retrieval2 import _fetch, _save_cache, _record_cache
        funds = universe.load()
        missing = [f for f in funds if f["scheme_code"] not in _record_cache]
        if not missing:
            return
        print(f"  [cache-warmer] Pre-fetching NAV for {len(missing)} uncached funds…")
        fetched = 0
        for f in missing:
            try:
                rec = _fetch(f["scheme_code"])
                if rec:
                    fetched += 1
                _time.sleep(0.12)  # ~8 requests/sec — polite to mfapi.in
            except Exception:
                pass
        _save_cache(_record_cache)
        print(f"  [cache-warmer] Done — {fetched}/{len(missing)} newly cached "
              f"({len(_record_cache)} total)")
    except Exception as e:
        print(f"  [cache-warmer] Failed: {e}")


if __name__ == "__main__":
    print("\n  VoiceFinAI — voice and text demo")
    print("  Open: http://localhost:5000\n")
    if os.getenv("VOICEFINAI_WARM_CACHE") == "1":
        threading.Thread(target=_warm_cache, daemon=True).start()
    app.run(debug=False, port=5000, threaded=True)
