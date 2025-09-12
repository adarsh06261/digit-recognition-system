from flask import Flask, render_template, request, jsonify
from src.utils import dataurl_to_pil
from src.predict_rf import predict_from_pil_rf
from src.predict_cnn import predict_from_pil_cnn

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    data_url = data.get("image")
    model_name = (data.get("model") or "cnn").lower()
    if not data_url:
        return jsonify({"error": "Missing image dataURL"}), 400

    pil_img = dataurl_to_pil(data_url)
    if model_name == "rf":
        pred, probs = predict_from_pil_rf(pil_img)
    else:
        pred, probs = predict_from_pil_cnn(pil_img)

    return jsonify({"model": model_name, "prediction": pred, "probabilities": probs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
