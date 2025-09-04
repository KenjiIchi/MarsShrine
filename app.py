# app.py — SL ↔ Flask Bridge (Echo + Mars + /diag + perfis via ENV)
# compatível com APIs estilo OpenAI (Chub Mars: Soji/Asha/Mixtral etc.)

import os, time, json
import requests
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# ── ENV BASICOS ─────────────────────────────────────────────────────────────
AUTH_TOKEN  = os.getenv("AUTH_TOKEN", "change-me")     # token que o SL envia

# Config da API (escolha do modelo é via URL + opcional MARS_MODEL)
MARS_API_KEY   = os.getenv("MARS_API_KEY")             # ex.: CHK-xxxxxxxx
MARS_API_URL   = os.getenv("MARS_API_URL", "")         # ex.: https://mars.chub.ai/chub/soji/v1
MARS_CHAT_PATH = os.getenv("MARS_CHAT_PATH", "/v1/chat/completions")
MARS_MODEL     = os.getenv("MARS_MODEL", "")           # ex.: soji, asha, mixtral
MARS_TIMEOUT   = float(os.getenv("MARS_TIMEOUT", "25"))

# Persona única (fallback) — deixe vazio se usar perfis
MARS_SYSTEM = os.getenv("MARS_SYSTEM", "")

# Perfis via ENV (JSON) + perfil padrão
PROFILES_JSON   = os.getenv("PROFILES_JSON", "{}")
DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "").strip()

# Knobs de geração (opcionais)
MARS_MAX_TOKENS  = int(float(os.getenv("MARS_MAX_TOKENS", "220")))
MARS_TEMPERATURE = float(os.getenv("MARS_TEMPERATURE", "1.0"))
MARS_FREQUENCY_PENALTY   = float(os.getenv("MARS_FREQUENCY_PENALTY", "0.0"))
MARS_PRESENCE_PENALTY    = float(os.getenv("MARS_PRESENCE_PENALTY", "0.0"))
MARS_REPETITION_PENALTY  = float(os.getenv("MARS_REPETITION_PENALTY", "1.0"))

# Parse do mapa de perfis
try:
    PROFILE_MAP = json.loads(PROFILES_JSON) if PROFILES_JSON else {}
    if not isinstance(PROFILE_MAP, dict):
        PROFILE_MAP = {}
except Exception:
    PROFILE_MAP = {}

# ── FLASK ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# ── HELPERS ─────────────────────────────────────────────────────────────────
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
    # header → body → query
    tok = request.headers.get("X-Auth-Token") or request.headers.get("Authorization")
    if not tok:
        tok = payload.get("token") or request.args.get("token")
    return tok

def _clip(s: str, limit: int = 900) -> str:
    s = s or ""
    return (s[:limit-3] + "...") if len(s) > limit else s

def _mars_ready() -> bool:
    return bool(MARS_API_KEY and MARS_API_URL)

def _pick_system_from(payload: dict) -> str:
    """
    Prioridade do system:
      1) profile enviado na requisição (body/profile ou ?profile=)
      2) DEFAULT_PROFILE (servidor)
      3) MARS_SYSTEM
      4) vazio
    Kill-switch por mensagem: ?sys=off ou "use_system": false
    """
    # kill-switch
    use_system = payload.get("use_system")
    if isinstance(use_system, str) and use_system.lower() in ("0", "false", "off", "no"):
        return ""
    if use_system is False or (request.args.get("sys", "").lower() == "off"):
        return ""

    # override por profile na requisição
    key = (payload.get("profile") or request.args.get("profile") or "").strip()
    if key and key in PROFILE_MAP:
        return str(PROFILE_MAP.get(key) or "")

    # perfil padrão
    if DEFAULT_PROFILE and DEFAULT_PROFILE in PROFILE_MAP:
        return str(PROFILE_MAP.get(DEFAULT_PROFILE) or "")

    # fallback único
    return MARS_SYSTEM

def mars_chat(message: str, system_text: str):
    """Chama a IA (API OpenAI-compatível). Retorna (reply, err)."""
    if not _mars_ready():
        return None, "missing_mars_env"

    url = MARS_API_URL.rstrip("/") + MARS_CHAT_PATH
    payload = {
        "messages": (
            [{"role": "system", "content": system_text}] if system_text else []
        ) + [{"role": "user", "content": message}],
        "max_tokens": MARS_MAX_TOKENS,
        "temperature": MARS_TEMPERATURE,
        "frequency_penalty": MARS_FREQUENCY_PENALTY,
        "presence_penalty":  MARS_PRESENCE_PENALTY,
    }
    if MARS_MODEL:
        payload["model"] = MARS_MODEL
    if MARS_REPETITION_PENALTY != 1.0:
        payload["repetition_penalty"] = MARS_REPETITION_PENALTY

    try:
        headers = {
            "Authorization": f"Bearer {MARS_API_KEY}",
            "X-API-Key": MARS_API_KEY,  # cobre proxies que exigem este header
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

        # Falha — devolve status + preview do corpo pro diag
        preview = (r.text or "")[:400]
        return None, f"upstream_{r.status_code}:{preview or 'no-body'}"

    except Exception as e:
        return None, f"exception:{type(e).__name__}"

# ── ROTAS ───────────────────────────────────────────────────────────────────
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
            "DEFAULT_PROFILE": DEFAULT_PROFILE,
            "PROFILE_KEYS": list(PROFILE_MAP.keys())[:24],  # só os nomes (sem conteúdo)
            "MARS_MAX_TOKENS": MARS_MAX_TOKENS,
            "MARS_TEMPERATURE": MARS_TEMPERATURE,
            "MARS_FREQ_PENALTY": MARS_FREQUENCY_PENALTY,
            "MARS_PRES_PENALTY": MARS_PRESENCE_PENALTY,
            "MARS_REP_PENALTY": MARS_REPETITION_PENALTY,
        },
        "chat_url": (MARS_API_URL.rstrip("/") + MARS_CHAT_PATH) if MARS_API_URL else None,
        "ready_for_mars": _mars_ready(),
    })

@app.post("/chat")
def chat():
    if not _rate_limit(request.remote_addr or "?"):
        return _error("too_many_requests", 429)

    payload = request.get_json(silent=True) or {}

    # Auth
    token = _read_token(payload)
    if AUTH_TOKEN and token != AUTH_TOKEN:
        return _error("unauthorized", 401)

    # Campos
    message    = str(payload.get("message", "")).strip()
    speaker    = str(payload.get("speaker", ""))[:64]
    session_id = str(payload.get("session_id", ""))[:128]
    if not message:
        return _error("'message' is required")

    # Escolhe system (perfil) — padrão já vem do servidor
    system_text = _pick_system_from(payload)

    # IA → fallback ECHO
    reply, err = mars_chat(message, system_text)
    mode = "mars"
    if reply is None:
        reply = f"[ECHO] {speaker+': ' if speaker else ''}{message}"
        mode = "echo"

    reply = _clip(reply, 900)  # evita estourar limite no chat do SL

    return jsonify({
        "ok": True,
        "reply": reply,
        "meta": {
            "mode": mode,
            "session_id": session_id,
            "mars_error": err,
            "profile_used": (payload.get("profile") or DEFAULT_PROFILE or ("MARS_SYSTEM" if MARS_SYSTEM else "")),
        },
    })

# ── LOCAL RUN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
