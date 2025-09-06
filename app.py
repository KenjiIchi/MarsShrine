import os
import json
import time
import re
import sqlite3
from collections import defaultdict, deque
from typing import List, Dict

import requests
from flask import Flask, request, jsonify

# =========================
# App & JSON (não escapar Unicode)
# =========================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # evita \uXXXX no JSON

# =========================
# Config (ENV)
# =========================
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "kenji-2025-bridge")

# Allowlist (quem pode usar o bridge). Por padrão inclui seu UUID.
MASTER_UUID = os.getenv("MASTER_UUID", "926f0717-f528-4ec2-817a-6690a605e0e6")
ALLOWLIST = set([x.strip() for x in os.getenv("ALLOWLIST", MASTER_UUID).split(",") if x.strip()])

# Mars/Soji upstream
MARS_API_URL = os.getenv("MARS_API_URL", "https://mars.chub.ai/chub/soji/v1")
MARS_CHAT_PATH = os.getenv("MARS_CHAT_PATH", "/chat/completions")
MARS_API_KEY = os.getenv("MARS_API_KEY", "")
MARS_MODEL = os.getenv("MARS_MODEL", "soji")

# Geração
MARS_TEMPERATURE = float(os.getenv("MARS_TEMPERATURE", "1.0"))
MARS_TOP_P = float(os.getenv("MARS_TOP_P", "0.9"))
MARS_TOP_K = int(os.getenv("MARS_TOP_K", "40"))
MARS_FREQUENCY_PENALTY = float(os.getenv("MARS_FREQUENCY_PENALTY", "0.6"))
MARS_PRESENCE_PENALTY = float(os.getenv("MARS_PRESENCE_PENALTY", "0.4"))
MARS_REPETITION_PENALTY = float(os.getenv("MARS_REPETITION_PENALTY", "1.08"))
MARS_MAX_TOKENS = int(os.getenv("MARS_MAX_TOKENS", "220"))
MARS_MIN_TOKENS = int(os.getenv("MARS_MIN_TOKENS", "0"))
MARS_STOP = os.getenv("MARS_STOP", "")  # JSON string opcional: ["###","User:"]

# Perfis (opcional)
DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "soji_gm")
PROFILES_JSON = os.getenv("PROFILES_JSON", "{}")

# Memória
PERSIST_ENABLED = os.getenv("PERSIST_ENABLED", "false").lower() == "true"
DEFAULT_DB = "/var/data/memory.sqlite"
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", DEFAULT_DB)
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "8"))
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "3500"))

# Kill-switch
ENABLED = True

# In-memory fallback
_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MEMORY_TURNS * 2))
_sticky: Dict[str, Dict[str, str]] = defaultdict(dict)

# =========================
# Helpers: Unicode uXXXX / \uXXXX → char (loop)
# =========================
_u_hex_plain = re.compile(r'u([0-9a-fA-F]{4})')   # u2661, uff40, u03b5...
_u_hex_esc   = re.compile(r'\\u([0-9a-fA-F]{4})') # \u2661

def _decode_u_sequences(s: str) -> str:
    """Converte sequências uXXXX e \\uXXXX para os caracteres reais (aplica em loop)."""
    if not s:
        return s
    prev = None
    while prev != s:
        prev = s
        s = _u_hex_esc.sub(lambda m: chr(int(m.group(1), 16)), s)
        s = _u_hex_plain.sub(lambda m: chr(int(m.group(1), 16)), s)
    return s

# =========================
# DB helpers
# =========================
def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _db_conn(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def _init_db():
    """Tenta inicializar persistência em sequência: env -> /var/data -> /tmp -> %TEMP% (Windows)."""
    global MEMORY_DB_PATH, PERSIST_ENABLED
    if not PERSIST_ENABLED:
        print("[memory] persistence disabled")
        return
    candidates = [MEMORY_DB_PATH, DEFAULT_DB, "/tmp/memory.sqlite"]
    try:
        win_tmp = os.path.join(os.getenv("TEMP", ""), "memory.sqlite")
        if win_tmp and win_tmp not in candidates:
            candidates.append(win_tmp)
    except Exception:
        pass

    for path in candidates:
        try:
            if not path:
                continue
            _ensure_parent_dir(path)
            conn = _db_conn(path)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS history (
                session_id TEXT,
                role TEXT,
                content TEXT,
                ts REAL
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS sticky (
                session_id TEXT,
                key TEXT,
                val TEXT,
                PRIMARY KEY (session_id, key)
            )""")
            conn.commit()
            conn.close()
            MEMORY_DB_PATH = path
            print(f"[memory] using sqlite at {path}")
            return
        except Exception as e:
            print(f"[memory] failed at {path}: {e}")

    print("[memory] all sqlite locations failed; falling back to in-memory only")
    PERSIST_ENABLED = False

_init_db()

def _save_history(session_id: str, role: str, content: str):
    _history[session_id].append({"role": role, "content": content})
    if PERSIST_ENABLED:
        try:
            conn = _db_conn(MEMORY_DB_PATH)
            conn.execute(
                "INSERT INTO history (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[memory] save_history error: {e}")

def _load_history(session_id: str):
    if not PERSIST_ENABLED:
        return list(_history[session_id])
    try:
        conn = _db_conn(MEMORY_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM history WHERE session_id=? ORDER BY ts ASC", (session_id,))
        rows = cur.fetchall()
        conn.close()
        out = [{"role": r, "content": c} for (r, c) in rows]
        _history[session_id].clear()
        for msg in out[-(MEMORY_TURNS * 2):]:
            _history[session_id].append(msg)
        return out
    except Exception as e:
        print(f"[memory] load_history error: {e}")
        return list(_history[session_id])

def _set_sticky(session_id: str, key: str, val: str):
    _sticky[session_id][key] = val
    if PERSIST_ENABLED:
        try:
            conn = _db_conn(MEMORY_DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO sticky (session_id, key, val) VALUES (?, ?, ?)",
                (session_id, key, val)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[memory] set_sticky error: {e}")

def _get_sticky(session_id: str) -> Dict[str, str]:
    if PERSIST_ENABLED and session_id not in _sticky:
        try:
            conn = _db_conn(MEMORY_DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT key, val FROM sticky WHERE session_id=?", (session_id,))
            rows = cur.fetchall()
            conn.close()
            _sticky[session_id] = {k: v for (k, v) in rows}
        except Exception as e:
            print(f"[memory] get_sticky error: {e}")
    return _sticky[session_id]

# =========================
# Sticky/state helpers  [[remember:key=value]] / [[state:key=value]]
# =========================
sticky_pattern = re.compile(r"\[\[(remember|state)\s*:\s*([a-zA-Z0-9_]+)\s*=\s*(.+?)\]\]")

def parse_and_update_sticky(text: str, session_id: str) -> str:
    if not text:
        return text
    for m in sticky_pattern.finditer(text):
        key, val = m.group(2), m.group(3).strip()
        _set_sticky(session_id, key, val)
    clean = sticky_pattern.sub("", text).strip()
    return clean

def sticky_header(session_id: str) -> str:
    s = _get_sticky(session_id)
    if not s:
        return ""
    pairs = "; ".join(f"{k}={v}" for k, v in s.items())
    return "STATE: " + pairs

# =========================
# Profiles & messages
# =========================
def _pick_profile_text() -> str:
    try:
        profiles = json.loads(PROFILES_JSON) if PROFILES_JSON else {}
    except Exception:
        profiles = {}
    if isinstance(profiles, dict):
        prof = profiles.get(DEFAULT_PROFILE, {})
        return prof.get("system", "")
    return ""

def _full_mars_url() -> str:
    if MARS_CHAT_PATH and MARS_CHAT_PATH.startswith("/"):
        return MARS_API_URL.rstrip("/") + MARS_CHAT_PATH
    return MARS_API_URL.rstrip("/")

def _auth_ok(token: str) -> bool:
    return (token or "") == AUTH_TOKEN

def _allowed(session_id: str) -> bool:
    return (session_id in ALLOWLIST)

def _build_messages(system_text: str, session_id: str, user_text: str) -> List[Dict[str, str]]:
    _load_history(session_id)
    msgs: List[Dict[str, str]] = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})

    sh = sticky_header(session_id)
    if sh:
        msgs.append({"role": "user", "content": sh})

    msgs.extend(list(_history[session_id]))
    msgs.append({"role": "user", "content": user_text})

    # janela deslizante por caracteres
    def total_chars(mm): return sum(len(m["content"]) for m in mm)
    while total_chars(msgs) > max(MEMORY_MAX_CHARS, 2000) and len(msgs) > 3:
        del msgs[2]  # remove a mais antiga pós-state
    return msgs

# =========================
# Upstream call (Mars/Soji)
# =========================
def _mars_call(messages: List[Dict[str, str]], conversation_id: str) -> str:
    url = _full_mars_url()
    headers = {
        "Authorization": f"Bearer {MARS_API_KEY}",
        "X-API-Key": MARS_API_KEY,
        "Content-Type": "application/json",
        # Headers "de navegador" — ajudam a evitar WAF/CF bloqueando server-to-server
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json",
        "Origin": "https://chub.ai",
        "Referer": "https://chub.ai/"
    }
    payload = {
        "model": MARS_MODEL,
        "conversation_id": conversation_id,
        "messages": messages,
        "temperature": MARS_TEMPERATURE,
        "top_p": MARS_TOP_P,
        "repetition_penalty": MARS_REPETITION_PENALTY,
        "max_tokens": MARS_MAX_TOKENS
    }
    if MARS_MIN_TOKENS > 0:
        payload["min_tokens"] = MARS_MIN_TOKENS
    if MARS_TOP_K > 0:
        payload["top_k"] = MARS_TOP_K
    if MARS_FREQUENCY_PENALTY:
        payload["frequency_penalty"] = MARS_FREQUENCY_PENALTY
    if MARS_PRESENCE_PENALTY:
        payload["presence_penalty"] = MARS_PRESENCE_PENALTY
    if MARS_STOP:
        try:
            payload["stop"] = json.loads(MARS_STOP)
        except Exception:
            pass

    r = requests.post(url, json=payload, headers=headers, timeout=int(os.getenv("MARS_TIMEOUT", "25")))
    if r.status_code >= 400:
        # Repassa texto do provedor pra facilitar debug (403/CF, etc.)
        raise RuntimeError(f"{r.status_code} {r.text}")

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # fallback: retorna JSON bruto para ver formato
        return json.dumps(data)

# =========================
# Routes
# =========================
@app.route("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "service": "sl-flask-bridge",
        "time": int(time.time()),
        "persist_enabled": PERSIST_ENABLED,
        "db_path": MEMORY_DB_PATH if PERSIST_ENABLED else None,
        "mars_url": _full_mars_url(),
        "model": MARS_MODEL
    })

@app.route("/health")
def health_alias():
    return healthz()

@app.route("/admin/toggle", methods=["POST"])
def admin_toggle():
    global ENABLED
    payload = request.get_json(force=True, silent=True) or {}
    token = payload.get("auth_token") or request.headers.get("x-auth-token")
    if not _auth_ok(token):
        return jsonify({"error": "unauthorized", "ok": False}), 401
    enabled = payload.get("enabled")
    if isinstance(enabled, bool):
        ENABLED = enabled
    return jsonify({"enabled": ENABLED, "ok": True})

@app.route("/admin/allowlist", methods=["POST"])
def admin_allowlist():
    payload = request.get_json(force=True, silent=True) or {}
    token = payload.get("auth_token") or request.headers.get("x-auth-token")
    if not _auth_ok(token):
        return jsonify({"error": "unauthorized", "ok": False}), 401
    add = payload.get("add", [])
    remove = payload.get("remove", [])
    for x in add:
        ALLOWLIST.add(str(x).strip())
    for x in remove:
        ALLOWLIST.discard(str(x).strip())
    return jsonify({"allowlist": list(ALLOWLIST), "ok": True})

@app.route("/chat", methods=["POST"])
def chat():
    global ENABLED
    if not ENABLED:
        return jsonify({"error": "bridge_disabled"}), 403

    payload = request.get_json(force=True, silent=True) or {}
    token = payload.get("auth_token") or request.headers.get("x-auth-token")
    if not _auth_ok(token):
        return jsonify({"error": "unauthorized", "ok": False}), 401

    # session_id: aceita body, header x-session-id, ou remote_addr
    session_id = str(
        (payload.get("session_id") if isinstance(payload, dict) else None)
        or request.headers.get("x-session-id")
        or request.remote_addr
        or "default"
    ).strip()

    if not _allowed(session_id):
        return jsonify({"error": "forbidden", "reason": "session not allowlisted"}), 403

    raw_text = str(payload.get("message", "")).strip()
    if not raw_text:
        return jsonify({"error": "empty_message"}), 400

    user_text = parse_and_update_sticky(raw_text, session_id)
    system_text = _pick_profile_text()
    messages = _build_messages(system_text, session_id, user_text)

    _save_history(session_id, "user", user_text)

    try:
        reply = _mars_call(messages, session_id)
        reply = _decode_u_sequences(reply)  # <-- converte uXXXX/\uXXXX
    except Exception as e:
        return jsonify({"error": "upstream_failed", "detail": str(e)}), 502

    _save_history(session_id, "assistant", reply)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
