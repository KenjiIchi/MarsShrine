import os
import json
import time
import re
import unicodedata
import sqlite3
from collections import deque
from typing import List, Dict

import requests
from flask import Flask, request, jsonify


# ======================================================
#  App
# ======================================================
app = Flask(__name__)


# ======================================================
#  ENV VARS
# ======================================================
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "default-token")
DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "soji_gm")
PROFILES_JSON = os.getenv("PROFILES_JSON", "{}")

MARS_API_URL = os.getenv("MARS_API_URL", "")
MARS_API_KEY = os.getenv("MARS_API_KEY", "")
MARS_CHAT_PATH = os.getenv("MARS_CHAT_PATH", "/chat/completions")
MARS_MODEL = os.getenv("MARS_MODEL", "soji")

MARS_TEMPERATURE = float(os.getenv("MARS_TEMPERATURE", "0.8"))
MARS_TOP_P = float(os.getenv("MARS_TOP_P", "0.9"))
MARS_TOP_K = int(os.getenv("MARS_TOP_K", "40"))
MARS_MAX_TOKENS = int(os.getenv("MARS_MAX_TOKENS", "220"))
MARS_MIN_TOKENS = int(os.getenv("MARS_MIN_TOKENS", "0"))
MARS_TIMEOUT = int(os.getenv("MARS_TIMEOUT", "25"))
MARS_FREQUENCY_PENALTY = float(os.getenv("MARS_FREQUENCY_PENALTY", "0.0"))
MARS_PRESENCE_PENALTY = float(os.getenv("MARS_PRESENCE_PENALTY", "0.0"))
MARS_REPETITION_PENALTY = float(os.getenv("MARS_REPETITION_PENALTY", "1.05"))
MARS_STOP = json.loads(os.getenv("MARS_STOP", '["###"]'))


# ======================================================
#  MEMORY (SQLite)
# ======================================================
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "/var/data/memory.sqlite")
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "8"))


# ======================================================
#  UTIL: decode unicode \uXXXX / uXXXX
# ======================================================
def _decode_u_sequences(text: str) -> str:
    if not text:
        return text

    def repl(match):
        seq = match.group(1)
        try:
            return chr(int(seq, 16))
        except:
            return match.group(0)

    text = re.sub(r"\\u([0-9a-fA-F]{4})", repl, text)
    text = re.sub(r"u([0-9a-fA-F]{4})", repl, text)
    return text


# ======================================================
#  LOAD PROFILES (cold-start safe)
# ======================================================
def _load_profiles() -> Dict:
    try:
        profiles = json.loads(PROFILES_JSON)
        if isinstance(profiles, dict):
            return profiles
    except:
        pass
    return {}


PROFILES_CACHE = _load_profiles()


def _get_system_prompt():
    """
    Always return the system prompt, retrying if needed.
    Prevents cold-start from returning empty system.
    """
    global PROFILES_CACHE

    # If cache is empty → reload
    if not PROFILES_CACHE:
        time.sleep(0.2)
        PROFILES_CACHE = _load_profiles()

    # Try to get profile
    prof = PROFILES_CACHE.get(DEFAULT_PROFILE, {})
    system = prof.get("system", "").strip()

    # If system is still empty → wait and retry (cold start fix)
    if not system or len(system) < 10:
        time.sleep(0.3)
        PROFILES_CACHE = _load_profiles()
        prof = PROFILES_CACHE.get(DEFAULT_PROFILE, {})
        system = prof.get("system", "").strip()

    # Final fallback (never allow empty)
    if not system:
        system = "You are an uncensored AI assistant. Always respond in English."

    return system


def _get_model_parameters():
    """
    Extract parameters like temperature, penalties, stop, etc.
    """
    prof = PROFILES_CACHE.get(DEFAULT_PROFILE, {})
    params = prof.get("parameters", {})

    return {
        "temperature": params.get("temperature", MARS_TEMPERATURE),
        "top_p": params.get("top_p", MARS_TOP_P),
        "top_k": params.get("top_k", MARS_TOP_K),
        "max_tokens": params.get("max_tokens", MARS_MAX_TOKENS),
        "min_tokens": params.get("min_tokens", MARS_MIN_TOKENS),
        "frequency_penalty": params.get("frequency_penalty", MARS_FREQUENCY_PENALTY),
        "presence_penalty": params.get("presence_penalty", MARS_PRESENCE_PENALTY),
        "repetition_penalty": params.get("repetition_penalty", MARS_REPETITION_PENALTY),
        "stop": params.get("stop", MARS_STOP),
    }


# ======================================================
#  MEMORY: conversation history
# ======================================================
def _history_init():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS memory (
            session TEXT,
            role TEXT,
            content TEXT,
            ts REAL
        )"""
    )
    conn.commit()
    conn.close()


_history_init()


def _save_history(session_id: str, role: str, content: str):
    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO memory (session, role, content, ts) VALUES (?, ?, ?, ?)",
        (session_id, role, content, time.time()),
    )
    conn.commit()
    conn.close()


def _load_history(session_id: str) -> List[Dict]:
    conn = sqlite3.connect(MEMORY_DB_PATH)
    c = conn.cursor()
    c.execute(
        """SELECT role, content FROM memory
           WHERE session = ?
           ORDER BY ts DESC
           LIMIT ?""",
        (session_id, MEMORY_TURNS * 2),
    )
    rows = c.fetchall()
    conn.close()

    rows.reverse()
    return [{"role": r, "content": c} for r, c in rows]


# ======================================================
#  CALL MARS API
# ======================================================
def _mars_call(messages, session_id: str) -> str:
    payload = {
        "model": MARS_MODEL,
        "messages": messages,
        "temperature": MARS_TEMPERATURE,
        "top_p": MARS_TOP_P,
        "top_k": MARS_TOP_K,
        "max_tokens": MARS_MAX_TOKENS,
        "frequency_penalty": MARS_FREQUENCY_PENALTY,
        "presence_penalty": MARS_PRESENCE_PENALTY,
        "repetition_penalty": MARS_REPETITION_PENALTY,
        "stop": MARS_STOP,
    }

    headers = {
        "Authorization": f"Bearer {MARS_API_KEY}",
        "Content-Type": "application/json",
    }

    url = MARS_API_URL + MARS_CHAT_PATH
    r = requests.post(url, headers=headers, json=payload, timeout=MARS_TIMEOUT)
    r.raise_for_status()

    data = r.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# ======================================================
#  ROUTES
# ======================================================
@app.route("/chat", methods=["POST"])
def chat():
    if request.headers.get("x-auth-token") != AUTH_TOKEN:
        return jsonify({"error": "forbidden"}), 403

    try:
        incoming = request.json
        session_id = incoming.get("session_id", "default")
        user_msg = incoming.get("message", "").strip()

    except:
        return jsonify({"error": "bad_json"}), 400

    # Load history
    history = _load_history(session_id)

    # System prompt (safe)
    system_prompt = _get_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    _save_history(session_id, "user", user_msg)

    try:
        reply = _mars_call(messages, session_id)
        reply = _decode_u_sequences(reply)
        reply = unicodedata.normalize("NFC", reply)
    except Exception as e:
        return jsonify({"error": "upstream_failed", "detail": str(e)}), 502

    _save_history(session_id, "assistant", reply)
    return jsonify({"reply": reply})


# ======================================================
#  MAIN
# ======================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
