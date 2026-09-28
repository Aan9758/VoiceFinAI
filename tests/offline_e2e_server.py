"""Local-only UI harness for exercising the browser capture/upload path.

This server deliberately replaces Deepgram, the model providers, and TTS with
fixed responses. Uploaded audio stays on localhost and is immediately
discarded; its contents are never logged or forwarded to a provider.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as voice_app

voice_app.DEEPGRAM_KEY = ""


def transcribe_local_clip():
    audio = request.files.get("audio")
    if audio is None:
        return jsonify({"error": "no audio"}), 400
    # Read and discard bytes to exercise multipart upload without storing them.
    if not audio.read():
        return jsonify({"error": "empty audio"}), 400
    return jsonify({"transcript": "Show me lower-risk mutual fund information"})


def answer_local_query():
    body = request.get_json(silent=True) or {}
    query = body.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "query is required"}), 400
    return jsonify({
        "type": "message",
        "intent": "offline_test",
        "view": "message",
        "cards": [],
        "voice_text": "Offline voice test complete. Audio stayed on this computer.",
        "screen_text": "Offline voice test complete. Audio stayed on this computer.",
        "audio_token": "",
    })


voice_app.app.view_functions["transcribe_audio"] = transcribe_local_clip
voice_app.app.view_functions["text_query"] = answer_local_query

if __name__ == "__main__":
    port = int(os.getenv("VOICEFINAI_E2E_PORT", "5001"))
    print(f"Offline UI harness: http://127.0.0.1:{port} (no provider calls)")
    voice_app.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
