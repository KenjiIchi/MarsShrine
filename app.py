import os
import json
import time
import re
import unicodedata
import sqlite3
import logging
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

# Debug flag
DEBUG_BRIDGE = os.getenv("DEBUG_BRIDGE", "false").lower() == "true"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_BRIDGE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

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

# (DB helpers, profiles, mensagens e routes continuam iguais ao que já te passei antes)
# ...
# =========================
# Routes
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    global ENABLED
    if not ENABLED:
        return jsonify({"error": "bridge_disabled"}), 403

    payload = request.get_json(force=True, silent=True) or {}
    token = payload.get("auth_token") or request.headers.get("x-auth-token")
    if not _auth_ok(token):
        return jsonify({"error": "unauthorized", "ok": False}), 401

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
        reply_raw = _mars_call(messages, session_id)
        reply_decoded = _decode_u_sequences(reply_raw)
        reply_decoded = unicodedata.normalize("NFC", reply_decoded)
        reply_final = reply_decoded.encode("utf-8").decode("utf-8")
    except Exception as e:
        logging.exception("upstream_failed")
        return jsonify({"error": "upstream_failed", "detail": str(e)}), 502

    _save_history(session_id, "assistant", reply_final)
    return jsonify({"reply": reply_final})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
