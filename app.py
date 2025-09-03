# app.py — SL ↔ Flask Bridge (Echo + token no body/query/header)

import os
import time
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "change-me")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# ── helpers ────────────────────────────────────────────────────────────────
def _error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status

RATE_WINDOW_SECONDS = 3
_last_hit_by_ip = {}
def _rate_limit(ip: str):
    now = time.time()
    last = _last_hit_by_ip.get(ip, 0)
    if now - last < RATE_WINDOW_SECONDS:
        return False
    _last_hit_by_ip[ip] = now
    return True

# ── routes ────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "sl-flask-bridge", "time": int(time.time())})

@app.post("/chat")
def chat():
    # 1) rate limit
    if not _rate_limit(request.remote_addr or "?"):
        return _error("too_many_requests", 429)

    # 2) payload (se houver)
    payload = request.get_json(silent=True) or {}

    # 3) token: aceita header OU body OU query
    token = request.headers.get("X-Auth-Token") or request.headers.get("Authorization")
    if not token:
        token = payload.get("token") or request.args.get("token")

    if AUTH_TOKEN and token != AUTH_TOKEN:
        return _error("unauthorized", 401)

    # 4) campos
    message = str(payload.get("message", "")).strip()
    speaker = str(payload.get("speaker", ""))[:64]
    session_id = str(payload.get("session_id", ""))[:128]

    if not message:
        return _error("'message' is required")

    # 5) echo (por enquanto)
    reply = f"[ECHO] {speaker+': ' if speaker else ''}{message}"

    return jsonify({
        "ok": True,
        "reply": reply,
        "meta": {"mode": "echo", "session_id": session_id},
    })

# ── local run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
