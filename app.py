from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import traceback
from datetime import timedelta

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Secret key for session
app.secret_key = os.getenv("SESSION_SECRET_KEY", "supersecret_dev_key")
app.permanent_session_lifetime = timedelta(hours=2)

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("⚠️ Warning: OPENROUTER_API_KEY not set!")

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# System message
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
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        if user_input.lower() in ["exit", "quit", "bye"]:
            session.clear()
            return jsonify({"response": "It was great helping you! Keep contributing! 👋"})

        # Initialize session chat history
        if "chat_history" not in session:
            session["chat_history"] = [system_message]

        # Add user input
        session["chat_history"].append({"role": "user", "content": user_input})

        # Call OpenRouter API
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70b-instruct",
            messages=session["chat_history"],
            max_tokens=400,
            temperature=0.7
        )

        bot_response = completion.choices[0].message.content.strip()

        # Save bot response
        session["chat_history"].append({"role": "assistant", "content": bot_response})
        session.modified = True

        return jsonify({"response": bot_response})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 OpenSourceBuddy running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
