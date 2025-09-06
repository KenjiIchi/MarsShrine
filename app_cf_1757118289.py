# app.py — SL ↔ Flask Bridge
# - Perfis via PROFILES_JSON + DEFAULT_PROFILE
# - Memória com persistência opcional (SQLite) + criação automática da pasta
# - Charset UTF-8 (corrige JP no SL)
# - /health, /diag, /memory/export, /memory/clear
# - Fallback ECHO quando upstream falha
# - Sanitização de tokens (</s>, [INST], [/INST]) no reply
# - Pronto para Render + gunicorn

import os
import re
import time
import json
import sqlite3
import threading
import collections
import requests

from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# ───────────────────────────── ENVs BÁSICAS ─────────────────────────────

AUTH_TOKEN   = os.getenv("AUTH_TOKEN", "change-me")  # token que o SL envia

# Config da API OpenAI-compatível (Chub Mars: Soji/Asha/Mixtral etc.)
MARS_API_KEY   = os.getenv("MARS_API_KEY", "")
MARS_API_URL   = os.getenv("MARS_API_URL", "")       # ex.: https://mars.chub.ai/chub/soji/v1
MARS_CHAT_PATH = os.getenv("MARS_CHAT_PATH", "/v1/chat/completions")
MARS_MODEL     = os.getenv("MARS_MODEL", "")         # ex.: soji
MARS_TIMEOUT   = float(os.getenv("MARS_TIMEOUT", "25"))

# Persona única (fallback) — deixe vazio se usar PROFILES_JSON
MARS_SYSTEM = os.getenv("MARS_SYSTEM", "")

# Perfis por JSON (string ou objeto) + perfil padrão
PROFILES_JSON   = os.getenv("PROFILES_JSON", "{}")
DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "").strip()

# Knobs globais (sobrescritíveis por perfil)
MARS_MAX_TOKENS         = int(float(os.getenv("MARS_MAX_TOKENS", "220")))
MARS_TEMPERATURE        = float(os.getenv("MARS_TEMPERATURE", "1.0"))
MARS_FREQUENCY_PENALTY  = float(os.getenv("MARS_FREQUENCY_PENALTY", "0.0"))
MARS_PRESENCE_PENALTY   = float(os.getenv("MARS_PRESENCE_PENALTY", "0.0"))
MARS_REPETITION_PENALTY = float(os.getenv("MARS_REPETITION_PENALTY", "1.0"))

# Decoding (opcionais)
MARS_TOP_P      = os.getenv("MARS_TOP_P", "")
MARS_TOP_K      = int(float(os.getenv("MARS_TOP_K", "0")))
MARS_MIN_TOKENS = int(float(os.getenv("MARS_MIN_TOKENS", "0")))
MARS_STOP_RAW   = os.getenv("MARS_STOP", "")  # JSON array opcional (["###","User:"])

try:
    MARS_STOP = json.loads(MARS_STOP_RAW) if MARS_STOP_RAW else None
    if MARS_STOP is not None and not isinstance(MARS_STOP, list):
        MARS_STOP = None
except Exception:
    MARS_STOP = None

# Memória em RAM (limites padrão)
MEMORY_TURNS_DEFAULT = int(float(os.getenv("MEMORY_TURNS", "8")))   # pares user+assistant
MEMORY_MAX_CHARS     = int(float(os.getenv("MEMORY_MAX_CHARS", "3500")))

# Persistência (opcional)
PERSIST_ENABLED = os.getenv("PERSIST_ENABLED", "false").lower() in ("1","true","yes","on")
MEMORY_DB_PATH  = os.getenv("MEMORY_DB_PATH", "./memory.sqlite")  # use /data/memory.sqlite se tiver disco persistente

# ───────────────────────────── FLASK APP ─────────────────────────────

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# JSON bonito em UTF-8 (sem \uXXXX) + força charset no header
app.json.ensure_ascii = False

@app.after_request
def force_json_utf8(resp):
    ct = resp.headers.get("Content-Type", "")
    if ct.startswith("application/json") and "charset=" not in ct.lower():
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp

# ───────────────────────────── HELPERS ─────────────────────────────

def _error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status

def _mars_ready() -> bool:
    return bool(MARS_API_KEY and MARS_API_URL)

def _clip(s: str, limit: int = 900) -> str:
    s = s or ""
    return (s[:limit-3] + "...") if len(s) > limit else s

def _clean_artifacts(s: str) -> str:
    """Remove tokens de template comuns (</s>, [INST], [/INST]) e espaços invisíveis."""
    if not s:
        return s
    s = re.sub(r'(</s>|\[/?INST\])\s*', '', s, flags=re.IGNORECASE)
    s = s.replace('\u200b', '').strip()
    return s

def _read_token(payload: dict):
    tok = request.headers.get("X-Auth-Token") or request.headers.get("Authorization")
    if not tok:
        tok = payload.get("token") or request.args.get("token")
    return tok

# rate-limit simples por IP (anti-spam)
RATE_WINDOW_SECONDS = 2.5
_last_hit_by_ip = {}
def _rate_limit(ip: str) -> bool:
    now = time.time()
    last = _last_hit_by_ip.get(ip, 0)
    if now - last < RATE_WINDOW_SECONDS:
        return False
    _last_hit_by_ip[ip] = now
    return True

# parse perfis
def _load_profiles(raw: str):
    try:
        data = json.loads(raw) if raw else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
PROFILE_MAP = _load_profiles(PROFILES_JSON)

# ───────────────────────────── PERSISTÊNCIA (SQLite) ─────────────────────────────

_db_lock = threading.Lock()
_db_conn = None

def _db():
    """Abre/cria o SQLite quando necessário (cria diretório pai, se preciso)."""
    global _db_conn
    if _db_conn is None:
        parent = os.path.dirname(MEMORY_DB_PATH) or "."
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        _db_conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
        _db_conn.execute("PRAGMA journal_mode=WAL;")
        _db_conn.execute("""
          CREATE TABLE IF NOT EXISTS memory (
            session_id TEXT NOT NULL,
            ts        INTEGER NOT NULL,
            role      TEXT NOT NULL,   -- 'user' | 'assistant'
            content   TEXT NOT NULL
          )
        """)
        _db_conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_sid_ts ON memory(session_id, ts)")
        _db_conn.commit()
    return _db_conn

def _db_append_msg(sid: str, role: str, content: str):
    if not PERSIST_ENABLED:
        return
    with _db_lock:
        _db().execute(
            "INSERT INTO memory (session_id, ts, role, content) VALUES (?,?,?,?)",
            (sid, int(time.time()*1000), role, content)
        )
        _db().commit()

def _db_fetch_recent(sid: str, turns: int, char_cap: int):
    """Busca últimos eventos; devolve lista [{role, content}, ...] já podada."""
    if not PERSIST_ENABLED:
        return []
    with _db_lock:
        cur = _db().execute(
            "SELECT role, content FROM memory WHERE session_id=? ORDER BY ts DESC LIMIT ?",
            (sid, 2*max(1, int(turns))*3)
        )
        rows = cur.fetchall()
    rows = list(reversed([{"role": r[0], "content": r[1]} for r in rows]))  # ordem cronológica
    total = 0
    pruned = []
    for m in rows[::-1]:  # do fim pro começo
        total += len(m.get("content",""))
        if total > max(0, int(char_cap)):
            break
        pruned.append(m)
    pruned.reverse()
    return pruned

def _db_trim_to_limits(sid: str, turns: int, char_cap: int):
    """Mantém no máximo 2*turns e respeita char_cap, removendo os mais antigos."""
    if not PERSIST_ENABLED:
        return
    with _db_lock:
        cur = _db().execute(
            "SELECT rowid, length(content) FROM memory WHERE session_id=? ORDER BY ts DESC",
            (sid,)
        )
        rows = cur.fetchall()
        keep = 2*max(1, int(turns))
        total = 0
        kept = 0
        to_delete = []
        for rowid, ln in rows:
            if kept >= keep or total + (ln or 0) > max(0, int(char_cap)):
                to_delete.append(rowid)
            else:
                kept += 1
                total += (ln or 0)
        if to_delete:
            _db().executemany("DELETE FROM memory WHERE rowid=?", [(x,) for x in to_delete])
            _db().commit()

def _db_clear_session(sid: str):
    if not PERSIST_ENABLED:
        return 0
    with _db_lock:
        cur = _db().execute("DELETE FROM memory WHERE session_id=?", (sid,))
        _db().commit()
        return cur.rowcount

# ───────────────────────────── PERFIL + MEMÓRIA ─────────────────────────────

def _resolve_profile(payload: dict):
    """
    Retorna (system_text, overrides_dict)
    overrides_dict pode conter: temperature, frequency_penalty, presence_penalty,
    repetition_penalty, max_tokens, memory_turns.
    """
    # kill-switch do system
    use_system = payload.get("use_system")
    if isinstance(use_system, str) and use_system.lower() in ("0","false","off","no"):
        return "", {}
    if use_system is False or (request.args.get("sys","").lower() == "off"):
        return "", {}

    key = (payload.get("profile") or request.args.get("profile") or "").strip()
    source = None
    if key and key in PROFILE_MAP:
        source = PROFILE_MAP[key]
    elif DEFAULT_PROFILE and DEFAULT_PROFILE in PROFILE_MAP:
        source = PROFILE_MAP[DEFAULT_PROFILE]
    else:
        if MARS_SYSTEM:
            return str(MARS_SYSTEM), {}

    if source is None:
        return "", {}

    if isinstance(source, str):
        return source, {}

    if isinstance(source, dict):
        parts = []
        for field in ("system", "backstory", "style", "rules", "memory_hint"):
            txt = source.get(field)
            if isinstance(txt, str) and txt.strip():
                parts.append(txt.strip())
        system_text = "\n\n".join(parts)

        params = source.get("parameters") or {}
        overrides = {}
        for k in ("temperature","frequency_penalty","presence_penalty","repetition_penalty","max_tokens"):
            if k in params:
                overrides[k] = params[k]
        mem = source.get("memory") or {}
        if "turns" in mem:
            overrides["memory_turns"] = int(mem.get("turns", MEMORY_TURNS_DEFAULT))
        return system_text, overrides

    return "", {}

# Histórico em RAM: { sid: deque([...]) }
_history = {}

def _get_session_id(payload: dict) -> str:
    sid = (payload.get("session_id") or "").strip()
    if not sid:
        sid = request.remote_addr or "anon"
    return sid[:128]

def _get_history(sid: str):
    return _history.get(sid) or collections.deque(maxlen=2*MEMORY_TURNS_DEFAULT)

def _set_history(sid: str, dq):
    _history[sid] = dq

def _apply_memory(messages: list, sid: str, turns: int, use_memory_flag: bool):
    if not use_memory_flag:
        return messages
    dq = _get_history(sid)
    hist = list(dq)
    # em cold-start, carrega do banco
    if not hist and PERSIST_ENABLED:
        hist = _db_fetch_recent(sid, turns, MEMORY_MAX_CHARS)
    needed = max(0, min(len(hist), 2*turns))
    hist = hist[-needed:] if needed else []
    # poda por char_cap
    total = 0
    pruned = []
    for m in hist:
        c = m.get("content","")
        total += len(c)
        if total > MEMORY_MAX_CHARS:
            break
        pruned.append(m)
    return pruned + messages

def _push_history(sid: str, user_msg: str, assistant_msg: str, turns: int):
    dq = _get_history(sid)
    target_maxlen = 2*max(1, int(turns))
    if dq.maxlen != target_maxlen:
        dq = collections.deque(dq, maxlen=target_maxlen)
    dq.append({"role": "user", "content": user_msg})
    dq.append({"role": "assistant", "content": assistant_msg})
    _set_history(sid, dq)

    if PERSIST_ENABLED:
        _db_append_msg(sid, "user", user_msg)
        _db_append_msg(sid, "assistant", assistant_msg)
        _db_trim_to_limits(sid, turns, MEMORY_MAX_CHARS)

# ───────────────────────────── CHAMADA À IA ─────────────────────────────

def mars_chat(messages: list, params: dict):
    """Chama a IA (API OpenAI-compatível). Retorna (reply, err)."""
    if not _mars_ready():
        return None, "missing_mars_env"

    url = MARS_API_URL.rstrip("/") + MARS_CHAT_PATH

    payload = {
        "messages": messages,
        "max_tokens": int(params.get("max_tokens", MARS_MAX_TOKENS)),
        "temperature": float(params.get("temperature", MARS_TEMPERATURE)),
        "frequency_penalty": float(params.get("frequency_penalty", MARS_FREQUENCY_PENALTY)),
        "presence_penalty":  float(params.get("presence_penalty",  MARS_PRESENCE_PENALTY)),
    }
    rep = params.get("repetition_penalty", MARS_REPETITION_PENALTY)
    if rep != 1.0:
        payload["repetition_penalty"] = float(rep)
    model = params.get("model", MARS_MODEL)
    if model:
        payload["model"] = model

    # top-p/k, min_tokens, stops (se suportados)
    if MARS_TOP_P != "":
        try:
            payload["top_p"] = float(MARS_TOP_P)
        except Exception:
            pass
    if MARS_TOP_K and MARS_TOP_K > 0:
        payload["top_k"] = int(MARS_TOP_K)
    if MARS_MIN_TOKENS and MARS_MIN_TOKENS > 0:
        payload["min_tokens"] = int(MARS_MIN_TOKENS)
        payload["min_length"] = int(MARS_MIN_TOKENS)
    if MARS_STOP:
        payload["stop"] = MARS_STOP

    try:
        headers = {
            "Authorization": f"Bearer {MARS_API_KEY}",
            "X-API-Key": MARS_API_KEY,                  # alguns proxies exigem
            "User-Agent": "MarsBridge/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        r = requests.post(url, headers=headers, json=payload, timeout=MARS_TIMEOUT)
        ct = (r.headers.get("content-type") or "").lower()

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

        preview = (r.text or "")[:400]
        return None, f"upstream_{r.status_code}:{preview or 'no-body'}"

    except Exception as e:
        return None, f"exception:{type(e).__name__}"

# ───────────────────────────── ROTAS ─────────────────────────────

@app.get("/")
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "sl-flask-bridge", "time": int(time.time())})

@app.get("/diag")
def diag():
    mem_stats = {
        "sessions": len(_history),
        "examples": {k: len(v) for k, v in list(_history.items())[:3]}
    }
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
            "PROFILE_KEYS": list(PROFILE_MAP.keys())[:24],
            "MARS_MAX_TOKENS": MARS_MAX_TOKENS,
            "MARS_TEMPERATURE": MARS_TEMPERATURE,
            "MARS_FREQ_PENALTY": MARS_FREQUENCY_PENALTY,
            "MARS_PRES_PENALTY": MARS_PRESENCE_PENALTY,
            "MARS_REP_PENALTY": MARS_REPETITION_PENALTY,
            "MARS_TOP_P": MARS_TOP_P or None,
            "MARS_TOP_K": MARS_TOP_K,
            "MARS_MIN_TOKENS": MARS_MIN_TOKENS,
            "MARS_STOP_set": bool(MARS_STOP),
            "MEMORY_TURNS_DEFAULT": MEMORY_TURNS_DEFAULT,
            "MEMORY_MAX_CHARS": MEMORY_MAX_CHARS,
            "PERSIST_ENABLED": PERSIST_ENABLED,
            "MEMORY_DB_PATH": MEMORY_DB_PATH if PERSIST_ENABLED else None,
        },
        "chat_url": (MARS_API_URL.rstrip("/") + MARS_CHAT_PATH) if MARS_API_URL else None,
        "ready_for_mars": _mars_ready(),
        "memory": mem_stats,
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

    # Campos principais
    message    = str(payload.get("message", "")).strip()
    speaker    = str(payload.get("speaker", ""))[:64]
    session_id = _get_session_id(payload)
    if not message:
        return _error("'message' is required")

    # Perfil e overrides
    system_text, overrides = _resolve_profile(payload)

    # Memória (enable/disable + turns override)
    use_memory = payload.get("use_memory", True)
    if isinstance(use_memory, str):
        use_memory = use_memory.lower() not in ("0","false","off","no")
    memory_turns = int(overrides.get("memory_turns", payload.get("memory_turns", MEMORY_TURNS_DEFAULT)))

    # Monta mensagens: [system?] + [histórico?] + [user]
    msgs = [{"role": "user", "content": message}]
    if system_text:
        msgs = [{"role": "system", "content": system_text}] + msgs
    msgs = _apply_memory(msgs, session_id, memory_turns, use_memory)

    # Parâmetros efetivos
    eff_params = {
        "temperature": overrides.get("temperature", MARS_TEMPERATURE),
        "frequency_penalty": overrides.get("frequency_penalty", MARS_FREQUENCY_PENALTY),
        "presence_penalty": overrides.get("presence_penalty", MARS_PRESENCE_PENALTY),
        "repetition_penalty": overrides.get("repetition_penalty", MARS_REPETITION_PENALTY),
        "max_tokens": overrides.get("max_tokens", MARS_MAX_TOKENS),
        "model": MARS_MODEL,
    }

    # IA → fallback ECHO
    reply, err = mars_chat(msgs, eff_params)
    mode = "mars"
    if reply is None:
        reply = f"[ECHO] {speaker+': ' if speaker else ''}{message}"
        mode = "echo"

    # sanitiza antes de clipar e salvar na memória
    reply = _clean_artifacts(reply)
    reply = _clip(reply, 900)

    # Atualiza memória
    if mode == "mars":
        _push_history(session_id, message, reply, memory_turns)

    return jsonify({
        "ok": True,
        "reply": reply,
        "meta": {
            "mode": mode,
            "session_id": session_id,
            "mars_error": err,
            "profile_used": (payload.get("profile") or DEFAULT_PROFILE or ("MARS_SYSTEM" if MARS_SYSTEM else "")),
            "memory_turns": memory_turns,
            "used_memory": bool(use_memory),
        },
    })

# ── manutenção de memória ────────────────────────────────────────────

@app.post("/memory/clear")
def mem_clear():
    payload = request.get_json(silent=True) or {}
    token = _read_token(payload)
    if AUTH_TOKEN and token != AUTH_TOKEN:
        return _error("unauthorized", 401)
    sid = (payload.get("session_id") or "").strip()
    if not sid:
        return _error("'session_id' is required")
    removed = _db_clear_session(sid)
    _history.pop(sid, None)  # limpa cache RAM
    return jsonify({"ok": True, "removed": int(removed)})

@app.get("/memory/export")
def mem_export():
    token = request.args.get("token") or ""
    if AUTH_TOKEN and token != AUTH_TOKEN:
        return _error("unauthorized", 401)
    sid = (request.args.get("session_id") or "").strip()
    if not sid:
        return _error("'session_id' is required")
    turns = int(request.args.get("turns") or MEMORY_TURNS_DEFAULT)
    if PERSIST_ENABLED:
        hist = _db_fetch_recent(sid, turns, MEMORY_MAX_CHARS)
    else:
        hist = list(_get_history(sid))
    return jsonify({"ok": True, "session_id": sid, "history": hist})

# ───────────────────────────── LOCAL RUN ─────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
