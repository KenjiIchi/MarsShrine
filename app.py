
import os
import json
import time
import re
import sqlite3
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple

import requests
from flask import Flask, request, jsonify, abort

# -----------------------------
# Config & ENV
# -----------------------------
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "kenji-2025-bridge")

MARS_API_URL = os.getenv("MARS_API_URL", "https://mars.chub.ai/chub/soji/v1")
MARS_CHAT_PATH = os.getenv("MARS_CHAT_PATH", "/chat/completions")
MARS_API_KEY = os.getenv("MARS_API_KEY", "")
MARS_MODEL = os.getenv("MARS_MODEL", "soji")

DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "soji_gm")
PROFILES_JSON = os.getenv("PROFILES_JSON", "{}")

MARS_TEMPERATURE = float(os.getenv("MARS_TEMPERATURE", "1.0"))
MARS_TOP_P = float(os.getenv("MARS_TOP_P", "0.9"))
MARS_TOP_K = int(os.getenv("MARS_TOP_K", "40"))
MARS_FREQUENCY_PENALTY = float(os.getenv("MARS_FREQUENCY_PENALTY", "0.6"))
MARS_PRESENCE_PENALTY = float(os.getenv("MARS_PRESENCE_PENALTY", "0.4"))
MARS_REPETITION_PENALTY = float(os.getenv("MARS_REPETITION_PENALTY", "1.08"))
MARS_MAX_TOKENS = int(os.getenv("MARS_MAX_TOKENS", "220"))
MARS_MIN_TOKENS = int(os.getenv("MARS_MIN_TOKENS", "0"))
MARS_STOP = os.getenv("MARS_STOP", "")

PERSIST_ENABLED = os.getenv("PERSIST_ENABLED", "false").lower() == "true"
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "/data/memory.sqlite")
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "8"))
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "3500"))

MASTER_UUID = os.getenv("MASTER_UUID", "926f0717-f528-4ec2-817a-6690a605e0e6")
ALLOWLIST = set([x.strip() for x in os.getenv("ALLOWLIST", MASTER_UUID).split(",") if x.strip()])

# -----------------------------
# App & Storage
# -----------------------------
app = Flask(__name__)

# In-memory history fallback
_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MEMORY_TURNS * 2))
# Sticky state per session (key/value you "remember")
_sticky: Dict[str, Dict[str, str]] = defaultdict(dict)

# Global toggle (server-side kill-switch)
ENABLED = True

# -----------------------------
# DB helpers (optional persistence)
# -----------------------------
def _db_conn():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def _init_db():
    if not PERSIST_ENABLED:
        return
    os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
    conn = _db_conn()
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

_init_db()

def _save_history(session_id: str, role: str, content: str):
    _history[session_id].append({"role": role, "content": content})
    if PERSIST_ENABLED:
        conn = _db_conn()
        conn.execute("INSERT INTO history (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                     (session_id, role, content, time.time()))
        conn.commit()
        conn.close()

def _load_history(session_id: str) -> List[Dict[str, str]]:
    if not PERSIST_ENABLED:
        return list(_history[session_id])
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM history WHERE session_id=? ORDER BY ts ASC", (session_id,))
    rows = cur.fetchall()
    conn.close()
    out = [{"role": r, "content": c} for (r, c) in rows]
    # also update in-memory deque window
    _history[session_id].clear()
    for msg in out[-(MEMORY_TURNS*2):]:
        _history[session_id].append(msg)
    return out

def _set_sticky(session_id: str, key: str, val: str):
    _sticky[session_id][key] = val
    if PERSIST_ENABLED:
        conn = _db_conn()
        conn.execute("INSERT OR REPLACE INTO sticky (session_id, key, val) VALUES (?, ?, ?)",
                     (session_id, key, val))
        conn.commit()
        conn.close()

def _get_sticky(session_id: str) -> Dict[str, str]:
    # Load from DB if needed
    if PERSIST_ENABLED and session_id not in _sticky:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("SELECT key, val FROM sticky WHERE session_id=?", (session_id,))
        rows = cur.fetchall()
        conn.close()
        _sticky[session_id] = {k: v for (k, v) in rows}
    return _sticky[session_id]

# -----------------------------
# Sticky parser & header
# -----------------------------
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

# -----------------------------
# Profiles & system text
# -----------------------------
def _pick_profile_text() -> str:
    try:
        profiles = json.loads(PROFILES_JSON) if PROFILES_JSON else {}
    except Exception:
        profiles = {}
    if isinstance(profiles, dict):
        prof = profiles.get(DEFAULT_PROFILE, {})
        # Prefer only "system" to save tokens; concatenate extras if you want.
        system_text = prof.get("system") or ""
        # If you want to also inject style/rules/backstory, uncomment:
        # extras = []
        # for k in ("backstory", "style", "rules", "memory_hint"):
        #     if prof.get(k):
        #         extras.append(f"{k.upper()}: {prof[k]}")
        # if extras:
        #     system_text = (system_text + "\n" + "\n".join(extras)).strip()
        return system_text
    return ""

# -----------------------------
# Helpers
# -----------------------------
def _full_mars_url() -> str:
    if MARS_CHAT_PATH and MARS_CHAT_PATH.startswith("/"):
        return MARS_API_URL.rstrip("/") + MARS_CHAT_PATH
    return MARS_API_URL.rstrip("/")

def _auth_ok(token: str) -> bool:
    return (token or "") == AUTH_TOKEN

def _allowed(session_id: str) -> bool:
    # Only master or allowlisted sessions can use the bridge
    return (session_id in ALLOWLIST)

def _build_messages(system_text: str, session_id: str, user_text: str) -> List[Dict[str, str]]:
    # Load rolling history
    _load_history(session_id)  # ensures _history deque is warmed

    msgs: List[Dict[str, str]] = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})

    # Sticky header (always injected)
    sh = sticky_header(session_id)
    if sh:
        msgs.append({"role": "user", "content": sh})

    # Recent history window
    msgs.extend(list(_history[session_id]))

    # Current user text (after absorbing [[remember:...]])
    msgs.append({"role": "user", "content": user_text})

    # Trim by chars if needed (basic safeguard)
    def total_chars(mm): return sum(len(m["content"]) for m in mm)
    while total_chars(msgs) > max(MEMORY_MAX_CHARS, 2000) and len(msgs) > 3:
        # drop the oldest user/assistant pair after system + sticky
        # ensure we keep system and sticky (index 0..1)
        if len(msgs) > 3:
            del msgs[2]  # remove one at a time
        else:
            break
    return msgs

def _mars_call(messages: List[Dict[str, str]], conversation_id: str) -> str:
    url = _full_mars_url()
    headers = {
        "Authorization": f"Bearer {MARS_API_KEY}",
        "Content-Type": "application/json"
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
    # Optional knobs
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
            stop_list = json.loads(MARS_STOP)
            payload["stop"] = stop_list
        except Exception:
            pass

    r = requests.post(url, json=payload, headers=headers, timeout=int(os.getenv("MARS_TIMEOUT", "25")))
    r.raise_for_status()
    data = r.json()

    # Adapt to Mars-like schema: choices[0].message.content
    try:
        reply = data["choices"][0]["message"]["content"]
    except Exception:
        reply = json.dumps(data)  # fallback raw
    return reply

# -----------------------------
# Routes
# -----------------------------
@app.route("/healthz")
def healthz():
    return "ok"

@app.route("/admin/toggle", methods=["POST"])
def admin_toggle():
    global ENABLED
    payload = request.get_json(force=True, silent=True) or {}
    token = payload.get("auth_token") or request.headers.get("x-auth-token")
    if not _auth_ok(token):
        return jsonify({"error": "unauthorized"}), 401
    enabled = payload.get("enabled")
    if isinstance(enabled, bool):
        ENABLED = enabled
    return jsonify({"enabled": ENABLED})

@app.route("/admin/allowlist", methods=["POST"])
def admin_allowlist():
    payload = request.get_json(force=True, silent=True) or {}
    token = payload.get("auth_token") or request.headers.get("x-auth-token")
    if not _auth_ok(token):
        return jsonify({"error": "unauthorized"}), 401
    add = payload.get("add", [])
    remove = payload.get("remove", [])
    for x in add:
        ALLOWLIST.add(str(x).strip())
    for x in remove:
        ALLOWLIST.discard(str(x).strip())
    return jsonify({"allowlist": list(ALLOWLIST)})

@app.route("/chat", methods=["POST"])
def chat():
    if not ENABLED:
        return jsonify({"error": "bridge_disabled"}), 403

    payload = request.get_json(force=True, silent=True) or {}
    token = payload.get("auth_token") or request.headers.get("x-auth-token")
    if not _auth_ok(token):
        return jsonify({"error": "unauthorized"}), 401

    session_id = str(payload.get("session_id") or request.remote_addr or "default").strip()
    # Enforce allowlist
    if not (session_id in ALLOWLIST):
        return jsonify({"error": "forbidden", "reason": "session not allowlisted"}), 403

    raw_text = str(payload.get("message", "")).strip()
    if not raw_text:
        return jsonify({"error": "empty_message"}), 400

    # Capture [[remember: key=val]] / [[state: key=val]]
    user_text = parse_and_update_sticky(raw_text, session_id)

    # System text from profile
    system_text = _pick_profile_text()

    # Build messages
    messages = _build_messages(system_text, session_id, user_text)

    # Save user to history
    _save_history(session_id, "user", user_text)

    # Call Mars
    try:
        reply = _mars_call(messages, session_id)
    except Exception as e:
        return jsonify({"error": "upstream_failed", "detail": str(e)}), 502

    # Save assistant to history
    _save_history(session_id, "assistant", reply)

    return jsonify({"reply": reply})

if __name__ == "__main__":
    # For local debug
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
