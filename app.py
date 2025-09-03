# app.py — SL ↔ Flask Bridge (Passo 1: ECHO)
import os
import time
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "change-me")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Rate limit simples por IP (anti-spam)
RATE_WINDOW_SECONDS = 3
_last_hit_by_ip = {}

def _error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status

def _rate_limit(ip: str):
    now = time.time()
    last = _last_hit_by_ip.get(ip, 0)
    if now - last < RATE_WINDOW_SECONDS:
        return False
    _last_hit_by_ip[ip] = now
    return True

@app.get("/")
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "sl-flask-bridge", "time": int(time.time())})

@app.post("/chat")
def chat():
    # 1) Header de autenticação
    auth_header = request.headers.get("X-Auth-Token")
    if AUTH_TOKEN and auth_header != AUTH_TOKEN:
        return _error("unauthorized", 401)

    # 2) Rate limit
    if not _rate_limit(request.remote_addr or "?"):
        return _error("too_many_requests", 429)

    # 3) JSON básico
    if not request.is_json:
        return _error("expected application/json body")
    payload = request.get_json(silent=True) or {}

    message = str(payload.get("message", "")).strip()
    speaker = str(payload.get("speaker", ""))[:64]
    session_id = str(payload.get("session_id", ""))[:128]

    if not message:
        return _error("'message' is required")

    # 4) Modo ECHO (sem IA ainda)
    reply = f"[ECHO] {speaker+': ' if speaker else ''}{message}"

    return jsonify({
        "ok": True,
        "reply": reply,
        "meta": {"mode": "echo", "session_id": session_id},
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
