import os
import json
import sqlite3
import traceback
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# ---------------------------------------------------------
# ENV VARS
# ---------------------------------------------------------
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
MARS_API_URL = os.getenv("MARS_API_URL", "")
MARS_CHAT_PATH = os.getenv("MARS_CHAT_PATH", "/chat/completions")
MARS_API_KEY = os.getenv("MARS_API_KEY", "")
DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "/var/data/memory.sqlite")
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "3500"))
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "8"))

# ---------------------------------------------------------
# LOAD PROFILE JSON
# ---------------------------------------------------------
def load_profiles():
    try:
        raw = os.getenv("PROFILES_JSON", "{}")
        print("[DEBUG] Loading PROFILES_JSON...")
        profiles = json.loads(raw)
        print("[DEBUG] Profiles loaded successfully.")
        return profiles
    except Exception as e:
        print("[ERROR] Failed loading PROFILES_JSON:", e)
        traceback.print_exc()
        return {}

PROFILES = load_profiles()

def get_profile():
    if DEFAULT_PROFILE and DEFAULT_PROFILE in PROFILES:
        print(f"[DEBUG] Using DEFAULT_PROFILE: {DEFAULT_PROFILE}")
        return PROFILES[DEFAULT_PROFILE]
    print("[DEBUG] Using fallback empty profile.")
    return {}

# ---------------------------------------------------------
# MEMORY SYSTEM
# ---------------------------------------------------------

def init_memory_db():
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                user TEXT,
                content TEXT
            );
        """)
        conn.commit()
        conn.close()
        print("[DEBUG] Memory DB initialized.")
    except Exception as e:
        print("[ERROR] Initializing memory DB:", e)
        traceback.print_exc()

init_memory_db()

def load_memory(user_id):
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        c = conn.cursor()
        c.execute("SELECT content FROM memory WHERE user=?", (user_id,))
        rows = c.fetchall()
        conn.close()
        mem = [r[0] for r in rows][-MEMORY_TURNS:]
        print(f"[DEBUG] Loaded memory for {user_id}: {len(mem)} turns")
        return mem
    except Exception as e:
        print("[ERROR] load_memory:", e)
        traceback.print_exc()
        return []

def save_memory(user_id, content):
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        c = conn.cursor()

        c.execute("SELECT SUM(LENGTH(content)) FROM memory WHERE user=?", (user_id,))
        total = c.fetchone()[0]
        if total is None:
            total = 0

        if total + len(content) > MEMORY_MAX_CHARS:
            c.execute("DELETE FROM memory WHERE user=? ORDER BY ROWID ASC LIMIT 1", (user_id,))

        c.execute("INSERT INTO memory (user, content) VALUES (?, ?)", (user_id, content))
        conn.commit()
        conn.close()
        print(f"[DEBUG] Saved memory entry ({len(content)} chars).")
    except Exception as e:
        print("[ERROR] save_memory:", e)
        traceback.print_exc()

# ---------------------------------------------------------
# MARS SOJI API CALL
# ---------------------------------------------------------
def call_mars(messages):
    try:
        url = MARS_API_URL.strip() + MARS_CHAT_PATH
        print("[DEBUG] MARS URL:", url)

        headers = {
            "Authorization": f"Bearer {MARS_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "options": {
                "model": "soji"
            }
        }

        print("[DEBUG] Sending payload to MARS:")
        print(json.dumps(payload, indent=2))

        response = requests.post(url, json=payload, headers=headers, timeout=40)

        print("[DEBUG] Status Code:", response.status_code)

        if response.status_code != 200:
            print("[ERROR] MARS API returned non-200:")
            print(response.text)
            return f"[API ERROR {response.status_code}] {response.text}"

        data = response.json()
        print("[DEBUG] MARS raw response:", json.dumps(data, indent=2))

        try:
            # Soji returns OpenAI-style choices
            text = data["choices"][0]["message"]["content"]
            return text
        except Exception:
            print("[ERROR] Unexpected response format", data)
            return "[ERROR] Invalid response format from Soji."

    except Exception as e:
        print("[CRASH] Exception in call_mars():", e)
        traceback.print_exc()
        return "[ERROR] Exception contacting Soji."

# ---------------------------------------------------------
# MAIN CHAT ENDPOINT
# ---------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        print("[DEBUG] Incoming request:", data)

        user_token = data.get("auth", "")
        if user_token != AUTH_TOKEN:
            print("[WARN] Unauthorized request.")
            return jsonify({"error": "unauthorized"}), 401

        user_id = data.get("user", "sl-user")
        user_msg = data.get("msg", "")

        profile = get_profile()
        system = profile.get("system", "")
        memory = load_memory(user_id)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        for m in memory:
            messages.append({"role": "assistant", "content": m})

        messages.append({"role": "user", "content": user_msg})

        reply = call_mars(messages)
        save_memory(user_id, reply)

        return jsonify({"reply": reply})

    except Exception as e:
        print("[CRASH] Exception in /chat:", e)
        traceback.print_exc()
        return jsonify({"reply": "[ERROR] Internal server crash."}), 500

# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------
@app.route("/healthz")
def health():
    return "ok", 200
