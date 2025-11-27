import os
import json
import sqlite3
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# ==========================================================
# CONFIG
# ==========================================================
MARS_API = "https://api.chub.ai/api/v1/chat"   # Mixtral (Mars Default)
PROFILE_JSON = os.getenv("PROFILE_JSON", "{}")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")  # x-auth-token for your SL bridge
DB_PATH = os.getenv("MEMORY_DB_PATH", "/var/data/memory.sqlite")

# ==========================================================
# ACCESS CONTROL (MASTER + SUB + EXTRA UUID LIST)
# ==========================================================
MASTER_UUID = os.getenv("MASTER_UUID", "").strip()
SUB_UUID = os.getenv("SUB_UUID", "").strip()
raw_env_allow = os.getenv("ALLOWLIST", "")

parsed_allowlist = [
    x.strip()
    for x in raw_env_allow.split(",")
    if x.strip()
]

ALLOWLIST = set(
    [uuid for uuid in [MASTER_UUID, SUB_UUID] if uuid] +
    parsed_allowlist
)

# ==========================================================
# MEMORY SYSTEM
# ==========================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            user TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def load_memory(user_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT content FROM memory WHERE user=?", (user_id,))
        rows = c.fetchall()
        conn.close()

        if not rows:
            return []

        return [json.loads(r[0]) for r in rows]
    except Exception as e:
        print("[ERROR] load_memory:", e)
        return []

def save_memory(user_id, role, msg):
    try:
        item = json.dumps({"role": role, "content": msg})
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO memory(user, content) VALUES (?, ?)", (user_id, item))
        conn.commit()
        conn.close()
    except Exception as e:
        print("[ERROR] save_memory:", e)

# ==========================================================
# LOAD JSON PROFILE
# ==========================================================
try:
    PROFILE_DATA = json.loads(PROFILE_JSON)
    print("[PROFILE] Loaded successfully.")
except Exception as e:
    PROFILE_DATA = {}
    print("[ERROR] Could not load PROFILE_JSON:", e)

DEFAULT_PROFILE = "soji_gm"

def build_messages(user_id, user_msg):
    profile = PROFILE_DATA.get(DEFAULT_PROFILE, {})
    memory = load_memory(user_id)

    messages = []

    if "system" in profile:
        messages.append({"role": "system", "content": profile["system"]})

    messages.extend(memory)
    messages.append({"role": "user", "content": user_msg})

    return messages

# ==========================================================
# CALL MARS MODEL
# ==========================================================
def call_mars(messages):
    try:
        payload = {
            "messages": messages,
            "model": "mixtral-mars-default",   # Model name for Chub.ai Mixtral
            "temperature": 0.9,
        }

        headers = {"Content-Type": "application/json"}

        r = requests.post(MARS_API, headers=headers, json=payload)

        if r.status_code != 200:
            print("[ERROR] MARS returned:", r.status_code, r.text)
            return "[API ERROR {}] {}".format(r.status_code, r.text)

        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        return reply

    except Exception as e:
        print("[ERROR] call_mars:", e)
        return "[ERROR] Could not reach MARS API"

# ==========================================================
# ROUTES
# ==========================================================
@app.route("/healthz")
def health():
    return "OK", 200

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
    except:
        return jsonify({"error": "Invalid JSON"}), 400

    auth = data.get("auth", "")
    user = data.get("user", "").strip()
    msg = data.get("msg", "").strip()

    # AUTH FOR LSL
    if auth != AUTH_TOKEN:
        return jsonify({"reply": "[403] Invalid auth token"}), 200

    # UUID ALLOWLIST
    if user not in ALLOWLIST:
        print("[DENY] User not in allowlist:", user)
        return jsonify({"reply": "[403] Access denied"}), 200

    print(f"[REQUEST] From {user}: {msg}")

    messages = build_messages(user, msg)
    reply = call_mars(messages)

    save_memory(user, "user", msg)
    save_memory(user, "assistant", reply)

    return jsonify({"reply": reply}), 200

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
