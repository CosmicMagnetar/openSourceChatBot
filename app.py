import os
import uuid
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("⚠️ Warning: OPENROUTER_API_KEY not set!")

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# System prompt
system_message = {
    "role": "system",
    "content": (
        "You are OpenSourceBuddy 🤖, a friendly assistant who helps newcomers get started with open-source, "
        "explains contribution steps simply, and shares competitions and opportunities. "
        "Give short, structured answers (2–5 lines). Be encouraging and mentor-like. "
        "If asked something unrelated to open source, politely say: "
        "'I'm here to help you with open-source and opportunities. I can’t help with that.'"
    )
}

# Store chat history in-memory (for production, use a database)
CHAT_DB_FILE = "chat_history.json"
if os.path.exists(CHAT_DB_FILE):
    with open(CHAT_DB_FILE, "r") as f:
        CHAT_HISTORY_DB = json.load(f)
else:
    CHAT_HISTORY_DB = {}

def fetch_chat_history(user_id):
    if user_id in CHAT_HISTORY_DB:
        return CHAT_HISTORY_DB[user_id]
    else:
        CHAT_HISTORY_DB[user_id] = [system_message]
        return CHAT_HISTORY_DB[user_id]

def save_message(user_id, role, content):
    if user_id not in CHAT_HISTORY_DB:
        CHAT_HISTORY_DB[user_id] = [system_message]
    CHAT_HISTORY_DB[user_id].append({"role": role, "content": content})
    # persist to disk
    with open(CHAT_DB_FILE, "w") as f:
        json.dump(CHAT_HISTORY_DB, f, indent=2)

# Serve frontend
@app.route("/")
def index():
    try:
        return send_from_directory(os.path.dirname(__file__), "index.html")
    except Exception:
        traceback.print_exc()
        return "index.html not found", 500

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_input = data["message"].strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        # Get or generate user_id
        user_id = data.get("user_id") or str(uuid.uuid4())

        # End session politely
        if user_input.lower() in ["exit", "quit", "bye"]:
            return jsonify({"response": "It was great helping you! Keep contributing! 👋", "user_id": user_id})

        # Fetch chat history
        chat_history = fetch_chat_history(user_id)
        chat_history.append({"role": "user", "content": user_input})

        # Call OpenRouter
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70b-instruct",  # replace with a valid model
            messages=chat_history,
            max_tokens=400,
            temperature=0.7
        )

        bot_response = completion.choices[0].message.content.strip()

        # Save messages
        save_message(user_id, "user", user_input)
        save_message(user_id, "assistant", bot_response)

        return jsonify({"response": bot_response, "user_id": user_id})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 OpenSourceBuddy running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
