from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Data Acquisition Agent is running"
    })


@app.route("/invoke", methods=["POST"])
def invoke():
    payload = request.get_json(silent=True) or {}

    return jsonify({
        "agent_id": 2,
        "current_stage": "data_acquisition",
        "status": "test_success",
        "received_input": payload
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
