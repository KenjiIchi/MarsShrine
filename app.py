# app.py — SL ↔ Flask Bridge (Echo + Mars + /diag)
import os, time, json
import requests
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# ── ENV ────────────────────────────────────────────────────────────────────
AUTH_TOKEN        = os.getenv("AUTH_TOKEN", "change-me")

# Mars (OpenAI-compatível). Se faltarem, cai em ECHO.
MARS_API_KEY      = os.getenv("MARS_API_KEY")                 # CHK-...
MARS_API_URL      = os.getenv("MARS_API_URL", "")             # ex.: https://mars.chub.ai/chub/asha/v1
MARS_CHAT_PATH    = os.getenv("MARS_CHAT_PATH", "/chat/completions")
MARS_MODEL        = os.getenv("MARS_MODEL", "")               # opcional
MARS_TIMEOUT      = float(os.getenv("MARS_TIMEOUT", "25"))

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# ── helpers ────────────────────────────────────────────────────────────────
def _error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status

RATE_WINDOW_SECONDS = 2.5
_last_hit_by_ip = {}
def _rate_limit(ip: str):
    now = time.time()
    last = _last_hit_by_ip.get(ip, 0)
    if now - last < RATE_WINDOW_SECONDS:
        return False
    _last_hit_by_ip[ip] = now
    return True

def _read_token(payload: dict):
    tok = request.headers.get("X-Auth-Token") or request.headers.get("Authorization")
    if not tok:
        tok = payload.get("token") or request.args.get("token")
    return tok

def _mars_ready():
    return bool(MARS_API_KEY and MARS_API_URL)

def mars_chat(message: str, session_id: str, speaker: str):
    """Chama a IA Mars/Asha/etc. Retorna (reply, err)."""
    if not (MARS_API_KEY and MARS_API_URL):
        return None, "missing_mars_env"

    url = MARS_API_URL.rstrip("/") + MARS_CHAT_PATH
    payload = {
        "messages": [{"role": "user", "content": message}],
    }
    if MARS_MODEL:
        payload["model"] = MARS_MODEL

    try:
        headers = {
            "Authorization": f"Bearer {MARS_API_KEY}",
            "X-API-Key": MARS_API_KEY,             # alguns proxies pedem também
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        r = requests.post(url, headers=headers, json=payload, timeout=MARS_TIMEOUT)

        # Conteúdo do upstream (para debug controlado)
        ct = (r.headers.get("content-type") or "").lower()
        text_preview = r.text[:400] if r.text else ""

        # Sucesso
        if r.status_code == 200 and ct.startswith("application/json"):
            jr = r.json()
            reply = (
                jr.get("output")
                or jr.get("reply")
                or (jr.get("choices", [{}])[0].get("message", {}).get("content"))
            )
            if reply:
                return reply, None
            return None, "empty_json_body"

        # Falha: devolve status e trechinho do corpo para sabermos o motivo
        return None, f"upstream_{r.status_code}:{text_preview or 'no-body'}"

    except Exception as e:
        return None, f"exception:{type(e).__name__}"

# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "sl-flask-bridge", "time": int(time.time())})

@app.get("/diag")
def diag():
    # Não mostra segredos; só flags pra debug
    return jsonify({
        "ok": True,
        "service": "sl-flask-bridge",
        "has_requests": True,
        "env_flags": {
            "AUTH_TOKEN_set": bool(AUTH_TOKEN),
            "MARS_API_KEY_set": bool(MARS_API_KEY),
            "MARS_API_URL_set": bool(MARS_API_URL),
            "MARS_CHAT_PATH": MARS_CHAT_PATH,
            "MARS_MODEL_set": bool(MARS_MODEL),
        },
        "chat_url": (MARS_API_URL.rstrip("/") + MARS_CHAT_PATH) if MARS_API_URL else None,
        "ready_for_mars": _mars_ready(),
    })

@app.post("/chat")
def chat():
    if not _rate_limit(request.remote_addr or "?"):
        return _error("too_many_requests", 429)

    payload = request.get_json(silent=True) or {}

    # auth: header/body/query
    token = _read_token(payload)
    if AUTH_TOKEN and token != AUTH_TOKEN:
        return _error("unauthorized", 401)

    message    = str(payload.get("message", "")).strip()
    speaker    = str(payload.get("speaker", ""))[:64]
    session_id = str(payload.get("session_id", ""))[:128]
    if not message:
        return _error("'message' is required")

    # IA → fallback ECHO
    reply, err = mars_chat(message, session_id, speaker)
    mode = "mars"
    if reply is None:
        reply = f"[ECHO] {speaker+': ' if speaker else ''}{message}"
        mode = "echo"

    return jsonify({
        "ok": True,
        "reply": reply,
        "meta": {"mode": mode, "session_id": session_id, "mars_error": err},
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
