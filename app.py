@app.post("/chat")
def chat():
    # 1) Rate limit simples
    if not _rate_limit(request.remote_addr or "?"):
        return _error("too_many_requests", 429)

    # 2) Pegar payload (se houver)
    payload = request.get_json(silent=True) or {}

    # 3) Token: aceita header OU body OU query
    token = request.headers.get("X-Auth-Token") or request.headers.get("Authorization")
    if not token:
        token = payload.get("token") or request.args.get("token")

    if AUTH_TOKEN and token != AUTH_TOKEN:
        return _error("unauthorized", 401)

    # 4) Ler dados
    message = str(payload.get("message", "")).strip()
    speaker = str(payload.get("speaker", ""))[:64]
    session_id = str(payload.get("session_id", ""))[:128]

    if not message:
        return _error("'message' is required")

    # 5) ECHO (por enquanto)
    reply = f"[ECHO] {speaker+': ' if speaker else ''}{message}"

    return jsonify({
        "ok": True,
        "reply": reply,
        "meta": {"mode": "echo", "session_id": session_id},
    })
