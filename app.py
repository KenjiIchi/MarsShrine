# app.py — SL ↔ Flask Bridge (Echo + Mars + /diag + persona opcional)
import os, time
import requests
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# ── ENV (Render/produção ou local) ──────────────────────────────────────────
AUTH_TOKEN        = os.getenv("AUTH_TOKEN", "change-me")

# API OpenAI-compatível (Chub/Mars, Asha/Mixtral/Soji, etc.)
MARS_API_KEY      = os.getenv("MARS_API_KEY")                 # ex.: CHK-xxxxxxxx
MARS_API_URL      = os.getenv("MARS_API_URL", "")             # ex.: https://api.chub.ai/chub/asha  (ou .../soji)
MARS_CHAT_PATH    = os.getenv("MARS_CHAT_PATH", "/v1/chat/completions")
MARS_MODEL        = os.getenv("MARS_MODEL", "")               # opcional (ex.: asha, mixtral, soji)
MARS_TIMEOUT      = float(os.getenv("MARS_TIMEOUT", "25"))

# Persona/controle de saída (opcionais)
MARS_SYSTEM       = os.getenv("MARS_SYSTEM", "")              # prompt de sistema (ou vazio p/ “livre”)
MARS_MAX_TOKENS   = int(os.getenv("MARS_MAX_TOKENS", "220"))
MARS_TEMPERATURE  = float(os.getenv("MARS_TEMPERATURE", "1.0"))

# ── Flask ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# ── Helpers ─────────────────────────────────────────────────────────────────
def _error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status

RATE_WINDOW_SECONDS = 2.5
_last_hit_by_ip = {}
def _rate_limit(ip: str) -> bool:
    now = time.time()
    last = _last_hit_by_ip.get(ip, 0)
    if now - last < RATE_WINDOW_SECONDS:
        return False
    _last_hit_by_ip[ip] = now
    return True

def _read_token(payload: dict):
    # header X-Auth-Token/Authorization → body.token → query ?token=
    tok = request.headers.get("X-Auth-Token") or request.headers.get("Authorization")
    if not tok:
        tok = payload.get("token") or request.args.get("token")
    return tok

def _clip(s: str, limit: int = 900) -> str:
    if not s:
        return ""
    return (s[: limit - 3] + "...") if len(s) > limit else s

def _mars_ready() -> bool:
    return bool(MARS_API_KEY and MARS_API_URL)

def mars_chat(message: str, session_id: str, speaker: str):
    """Chama a IA (API OpenAI-compatível). Retorna (reply, err)."""
    if not _mars_ready():
        return None, "missing_mars_env"

    url = MARS_API_URL.rstrip("/") + MARS_CHAT_PATH
    payload = {
        "messages": (
            [{"role": "system", "content": MARS_SYSTEM}] if MARS_SYSTEM else []
        ) + [
            {"role": "user", "content": message}
        ],
        "max_tokens": MARS_MAX_TOKENS,
        "temperature": MARS_TEMPERATURE,
    }
    if MARS_MODEL:
        payload["model"] = MARS_MODEL

    try:
        headers = {
            "Authorization": f"Bearer {MARS_API_KEY}",
            "X-API-Key": MARS_API_KEY,                     # cobre proxies que exigem header extra
            "User-Agent": "Mozilla/5.0 (compatible; MarsBridge/1.0)",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        r = requests.post(url, headers=headers, json=payload, timeout=MARS_TIMEOUT)

        ct = (r.headers.get("content-type") or "").lower()
        # Sucesso típico OpenAI-compatível
        if r.status_code == 200 and "application/json" in ct:
            jr = r.json()
            reply = (
                jr.get("output")
                or jr.get("reply")
                or (jr.get("choices", [{}])[0].get("message", {}).get("content"))
            )
            if reply:
                return reply, None
            return None, "empty_json_body"

        # Falha: devolve status + preview do corpo para depuração
        preview = (r.text or "")[:400]
        return None, f"upstream_{r.status_code}:{preview or 'no-body'}"

    except Exception as e:
        return None, f"exception:{type(e).__name__}"

# ── Rotas ───────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "sl-flask-bridge", "time": int(time.time())})

@app.get("/diag")
def diag():
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
            "MARS_SYSTEM_set": bool(MARS_SYSTEM),
            "MARS_MAX_TOKENS": MARS_MAX_TOKENS,
            "MARS_TEMPERATURE": MARS_TEMPERATURE,
        },
        "chat_url": (MARS_API_URL.rstrip("/") + MARS_CHAT_PATH) if MARS_API_URL else None,
        "ready_for_mars": _mars_ready(),
    })

@app.post("/chat")
def chat():
    # Anti-flood simples
    if not _rate_limit(request.remote_addr or "?"):
        return _error("too_many_requests", 429)

    payload = request.get_json(silent=True) or {}

    # Auth (header/body/query)
    token = _read_token(payload)
    if AUTH_TOKEN and token != AUTH_TOKEN:
        return _error("unauthorized", 401)

    # Campos
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

    # Evita estourar limite no chat do SL
    reply = _clip(reply, 900)

    return jsonify({
        "ok": True,
        "reply": reply,
        "meta": {"mode": mode, "session_id": session_id, "mars_error": err},
    })

# ── Local run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
