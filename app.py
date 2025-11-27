import os
import json
import sqlite3
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# ======================================
# CONFIG
# ======================================
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
MARS_API_KEY = os.getenv("MARS_API_KEY", "")
MARS_MODEL = os.getenv("MARS_MODEL", "mixtral-mars")  # modelo padrão
DB_PATH = os.getenv("MEMORY_DB_PATH", "/var/data/memory.sqlite")
PROFILES_JSON = os.getenv("PROFILES_JSON", "{}")

def load_profile():
    try:
        return json.loads(PROFILE_JSON)
    except:
        print("[ERROR] Could not load PROFILE_JSON")
        return {}

PROFILE = load_profile()
DEFAULT_PROFILE = PROFILE.get("soji_gm", {})

# ======================================
# MEMORY SYSTEM
# ======================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            user TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def load_memory(user):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT content FROM memory WHERE user=?", (user,))
        rows = c.fetchall()
        conn.close()

        return [r[0] for r in rows]
    except Exception as e:
        print("[ERROR] load_memory:", e)
        return []

def save_memory(user, msg):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO memory (user, content) VALUES (?,?)", (user, msg))
        conn.commit()
        conn.close()
    except Exception as e:
        print("[ERROR] save_memory:", e)

# ======================================
# HEALTH CHECK
# ======================================
@app.get("/healthz")
def healthz():
    return "OK", 200

# ======================================
# MAIN CHAT ENDPOINT
# ======================================
@app.post("/chat")
def chat():
    # tenta ler JSON
    data = {}
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        data = {}

    # ❤️ VOLTAMOS AO ANTIGO COMPORTAMENTO (JSON OU HEADER)
    auth = data.get("auth") or request.headers.get("x-auth-token")

    if auth != AUTH_TOKEN:
        return jsonify({"error": "Invalid auth token"}), 403

    user = (
        data.get("user")
        or data.get("session_id")
        or request.remote_addr
        or "unknown"
    )

    msg = data.get("msg") or data.get("message") or ""
    if not msg:
        return jsonify({"error": "No message"}), 400

    # memory
    history = load_memory(user)
    save_memory(user, msg)

    # monta payload para Mars
    sys_prompt = DEFAULT_PROFILE.get("system", "")
    messages = [{"role": "system", "content": sys_prompt}]

    for h in history[-8:]:
        messages.append({"role": "user", "content": h})

    messages.append({"role": "user", "content": msg})

    payload = {
        "messages": messages,
        "options": {"model": MARS_MODEL}
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": MARS_API_KEY
    }

    # envia para Mars API
    r = requests.post(
        "https://api.mars.guru/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if r.status_code != 200:
        return jsonify({
            "error": "MARS API error",
            "status": r.status_code,
            "body": r.text
        }), 200

    try:
        reply = r.json()["choices"][0]["message"]["content"]
    except:
        reply = "Malformed reply"

    save_memory(user, reply)

    return jsonify({"reply": reply}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

