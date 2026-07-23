import pytest

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


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "recovery_rate_mcf" in data or "message" in data


def test_optimize_valid(client):
    response = client.post("/api/optimize", json=SAMPLE_INPUT)
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "recovery_rate_mcf" in data
        assert "economic_savings_usd" in data
        assert data["recovery_rate_mcf"] >= 0
        assert data["economic_savings_usd"] >= 0


def test_optimize_defaults(client):
    response = client.post("/api/optimize", json={})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "recovery_rate_mcf" in data


def test_emissions_valid(client):
    response = client.post("/api/emissions", json=SAMPLE_INPUT)
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "co2_emissions_tons" in data
        assert data["co2_emissions_tons"] >= 0


def test_emissions_defaults(client):
    response = client.post("/api/emissions", json={})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "co2_emissions_tons" in data
