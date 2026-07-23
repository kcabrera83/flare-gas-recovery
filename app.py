import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_DIR = os.path.join("outputs", "models")
FEATURE_COLS = [
    "flare_gas_rate_mcf",
    "methane_pct",
    "ethane_pct",
    "propane_pct",
    "butane_pct",
    "ambient_temp_c",
    "wind_speed_ms",
    "steam_injection_rate",
    "flare_efficiency_pct",
]

# Load models at startup
models = {}


def load_models():
    global models
    try:
        with open(os.path.join(MODEL_DIR, "recovery_optimizer.pkl"), "rb") as f:
            models["recovery_optimizer"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "savings_optimizer.pkl"), "rb") as f:
            models["savings_optimizer"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "emission_predictor.pkl"), "rb") as f:
            models["emission_predictor"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler_recovery.pkl"), "rb") as f:
            models["scaler_recovery"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler_savings.pkl"), "rb") as f:
            models["scaler_savings"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler_emission.pkl"), "rb") as f:
            models["scaler_emission"] = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "training_summary.json"), "r") as f:
            models["summary"] = json.load(f)
        print(f"Loaded {len([k for k in models if 'optimizer' in k or 'predictor' in k])} models.")
    except Exception as e:
        print(f"WARNING: Could not load models: {e}")


def extract_features(data):
    return np.array(
        [[data.get(col, 0) for col in FEATURE_COLS]]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    loaded = len([k for k in models if k not in ("summary",)])
    return jsonify({"status": "ok", "models_loaded": loaded})


@app.route("/api/models")
def list_models():
    info = {}
    if "summary" in models:
        info = models["summary"]
    else:
        info = {"message": "No models loaded. Run train.py first."}
    return jsonify(info)


@app.route("/api/optimize", methods=["POST"])
def optimize():
    try:
        data = request.get_json(force=True)
        X = extract_features(data)

        X_rec = models["scaler_recovery"].transform(X)
        X_sav = models["scaler_savings"].transform(X)

        recovery_rate = float(models["recovery_optimizer"].predict(X_rec)[0])
        economic_savings = float(models["savings_optimizer"].predict(X_sav)[0])

        recovery_rate = max(0, recovery_rate)
        economic_savings = max(0, economic_savings)

        return jsonify(
            {
                "recovery_rate_mcf": round(recovery_rate, 2),
                "economic_savings_usd": round(economic_savings, 2),
                "input": {col: data.get(col, 0) for col in FEATURE_COLS},
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/emissions", methods=["POST"])
def emissions():
    try:
        data = request.get_json(force=True)
        X = extract_features(data)
        X_scaled = models["scaler_emission"].transform(X)

        co2 = float(models["emission_predictor"].predict(X_scaled)[0])
        co2 = max(0, co2)

        return jsonify(
            {
                "co2_emissions_tons": round(co2, 2),
                "input": {col: data.get(col, 0) for col in FEATURE_COLS},
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "Flare Gas Recovery", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/optimize": {"post": {"summary": "Optimize recovery rate and economic savings"}},
            "/api/emissions": {"post": {"summary": "Predict CO2 emissions in tons"}},
        }
    })


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5011, debug=False)
