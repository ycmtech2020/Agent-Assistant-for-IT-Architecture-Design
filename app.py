import uuid
from flask import Flask, render_template, request, jsonify

from architecture_agent import run_architecture_agent

app = Flask(__name__)

@app.route("/")
def index():
    """
    Render the main UI.
    """
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    conversation_id = data.get("conversation_id") or str(uuid.uuid4())
    history = data.get("history", "")

    # If 'history' is a list, join it into a string, otherwise cast to str
    if isinstance(history, list):
        history_str = "\n".join(str(item) for item in history)
    else:
        history_str = str(history)

    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty user message."}), 400

    # Combine full requirements text
    full_requirements_text = (history_str + "\n" + user_message).strip()

    # Single agent call: internally handles architecture + diagram + conditional NFR
    result_payload = run_architecture_agent(
        full_requirements_text,
        thread_id=conversation_id,
    )

    # result_payload already has:
    # summary, pattern_id, components, connections, image_url, dot, nfr_report
    return jsonify(result_payload)


if __name__ == "__main__":
    app.run(debug=True)
