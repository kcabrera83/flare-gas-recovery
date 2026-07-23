"""FastAPI for flare gas recovery optimization."""

import json
import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Flare Gas Recovery",
    description="Flare gas recovery optimization and CO2 emission prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join("outputs", "models")
FEATURE_COLS = [
    "flare_gas_rate_mcf", "methane_pct", "ethane_pct", "propane_pct",
    "butane_pct", "ambient_temp_c", "wind_speed_ms",
    "steam_injection_rate", "flare_efficiency_pct",
]

models = {}


@app.on_event("startup")
async def load_models():
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
    except Exception as e:
        print(f"WARNING: Could not load models: {e}")


def extract_features(data):
    return np.array([[data.get(col, 0) for col in FEATURE_COLS]])


class GasRequest(BaseModel):
    flare_gas_rate_mcf: float = 0
    methane_pct: float = 0
    ethane_pct: float = 0
    propane_pct: float = 0
    butane_pct: float = 0
    ambient_temp_c: float = 0
    wind_speed_ms: float = 0
    steam_injection_rate: float = 0
    flare_efficiency_pct: float = 0


class OptimizeResponse(BaseModel):
    recovery_rate_mcf: float
    economic_savings_usd: float
    input: dict


class EmissionsResponse(BaseModel):
    co2_emissions_tons: float
    input: dict


@app.get("/api/health")
async def health():
    loaded = len([k for k in models if k not in ("summary",)])
    return {"status": "ok", "models_loaded": loaded}


@app.get("/api/models")
async def models_info():
    if "summary" in models:
        return models["summary"]
    return {"message": "No models loaded. Run train.py first."}


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(request: GasRequest):
    try:
        data = request.model_dump()
        X = extract_features(data)
        X_rec = models["scaler_recovery"].transform(X)
        X_sav = models["scaler_savings"].transform(X)
        recovery_rate = max(0, float(models["recovery_optimizer"].predict(X_rec)[0]))
        economic_savings = max(0, float(models["savings_optimizer"].predict(X_sav)[0]))
        return OptimizeResponse(
            recovery_rate_mcf=round(recovery_rate, 2),
            economic_savings_usd=round(economic_savings, 2),
            input={col: data.get(col, 0) for col in FEATURE_COLS},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/emissions", response_model=EmissionsResponse)
async def emissions(request: GasRequest):
    try:
        data = request.model_dump()
        X = extract_features(data)
        X_scaled = models["scaler_emission"].transform(X)
        co2 = max(0, float(models["emission_predictor"].predict(X_scaled)[0]))
        return EmissionsResponse(
            co2_emissions_tons=round(co2, 2),
            input={col: data.get(col, 0) for col in FEATURE_COLS},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
