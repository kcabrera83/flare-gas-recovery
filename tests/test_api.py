import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app, load_models

load_models()
client = app.test_client()

SAMPLE_INPUT = {
    "flare_gas_rate_mcf": 500.0,
    "methane_pct": 80.0,
    "ethane_pct": 8.0,
    "propane_pct": 5.0,
    "butane_pct": 3.0,
    "ambient_temp_c": 25.0,
    "wind_speed_ms": 5.0,
    "steam_injection_rate": 10.0,
    "flare_efficiency_pct": 95.0,
}


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["models_loaded"] > 0


def test_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "recovery_rate_mcf" in data or "message" in data


def test_api_docs():
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"


def test_optimize_valid():
    response = client.post("/api/optimize", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.get_json()
    assert "recovery_rate_mcf" in data
    assert "economic_savings_usd" in data
    assert data["recovery_rate_mcf"] >= 0
    assert data["economic_savings_usd"] >= 0


def test_optimize_defaults():
    response = client.post("/api/optimize", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert "recovery_rate_mcf" in data


def test_emissions_valid():
    response = client.post("/api/emissions", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.get_json()
    assert "co2_emissions_tons" in data
    assert data["co2_emissions_tons"] >= 0


def test_emissions_defaults():
    response = client.post("/api/emissions", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert "co2_emissions_tons" in data


def test_optimize_empty_body():
    response = client.post("/api/optimize", content_type="application/json")
    assert response.status_code in [200, 400]
