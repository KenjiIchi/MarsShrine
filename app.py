# ==========================================================
# Mars/Mercury Bridge – Stable Backend for Second Life
# Totalmente refeito para Kenji – Novembro 2025
# ==========================================================

import os
import json
import sqlite3
from flask import Flask, request, jsonify
import requests

# ----------------------------------------------------------
# CONFIG FROM ENVIRONMENT
# ----------------------------------------------------------
AUTH_TOKEN      = os.getenv("AUTH_TOKEN", "")
MARS_API_KEY    = os.getenv("MARS_API_KEY", "")
MARS_API_URL    = os.getenv("MARS_API_URL", "")
MARS_CHAT_PATH  = os.getenv("MARS_CHAT_PATH", "/chat/completions")
DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "soji_gm")
MEMORY_DB_PATH  = os.getenv("MEMORY_DB_PATH", "/var/data/memory.sqlite")

# URL final (ex: https://mercury.chub.ai/mistral/v1/chat/completions)
FULL_ENDPOINT = f"{MARS_API_URL}{MARS_CHAT_PATH}"

app = Flask(__name__)

# ----------------------------------------------------------
# MEMORY: SQLite (session_id -> last turns)
# ----------------------------------------------------------
def init_db():
    conn = sqlite3.connect(MEMORY_DB_PATH)
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

def load_memory(user_id):
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        c = conn.cursor()
        c.execute("SELECT content FROM memory WHERE user=?", (user_id,))
        rows = c.fetchall()
        conn.close()

        return [json.loads(r[0]) for r in rows]

    except Exception as e:
        print("[ERROR] load_memory:", e)
        return []


def save_memory(user_id, messages, max_tokens=2000):
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        c = conn.cursor()

        # Limpa memória antiga se ultrapassar limite
        saved_size = sum(len(m["content"]) for m in messages)
        if saved_size > max_tokens:
            messages = messages[-10:]  # mantém só últimas 10

        c.execute("DELETE FROM memory WHERE user=?", (user_id,))
        for m in messages:
            c.execute(
                "INSERT INTO memory (user, content) VALUES (?, ?)",
                (user_id, json.dumps(m))
            )

        conn.commit()
        conn.close()

    except Exception as e:
        print("[ERROR] save_memory:", e)


# ----------------------------------------------------------
# LOAD PROFILE (system prompt, style, etc.)
# ----------------------------------------------------------
def load_profile():
    try:
        with open("mars.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(DEFAULT_PROFILE, {})
    except:
        return {}


PROFILE = load_profile()

# ----------------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------------
@app.route("/healthz")
def health():
    return "OK", 200

# ----------------------------------------------------------
# CHAT ENDPOINT
# ----------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if data.get("auth") != AUTH_TOKEN:
            return jsonify({"error": "invalid token"}), 401

        user_id = data.get("user")
        msg     = data.get("msg", "")

        # Carrega memória
        previous = load_memory(user_id)

        # Construir payload
        messages = []

        system_prompt = PROFILE.get("system", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # histórico
        messages.extend(previous)

        # nova mensagem
        messages.append({"role": "user", "content": msg})

        payload = {
            "messages": messages,
            "options": {
                "model": "mistral"   # sempre MISTRAL (Mercury)
            }
        }

        headers = {
            "Authorization": f"Bearer {MARS_API_KEY}",
            "Content-Type": "application/json"
        }

        resp = requests.post(FULL_ENDPOINT, headers=headers, json=payload, timeout=30)

        if resp.status_code != 200:
            return jsonify({"reply": f"[API ERROR {resp.status_code}] {resp.text}"}), 200

        data = resp.json()

        reply = ""
        try:
            reply = data["choices"][0]["message"]["content"]
        except:
            reply = str(data)

        # salva memória (input + output)
        previous.append({"role": "user", "content": msg})
        previous.append({"role": "assistant", "content": reply})
        save_memory(user_id, previous)

        return jsonify({"reply": reply}), 200

    except Exception as e:
        return jsonify({"reply": f"[SERVER ERROR] {e}"}), 200


# ----------------------------------------------------------
# RUN (Render uses gunicorn, ignore)
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
