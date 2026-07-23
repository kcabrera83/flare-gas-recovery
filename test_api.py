"""API integration tests for Flare Gas Recovery FastAPI app."""

import sys
from fastapi.testclient import TestClient

sys.path.insert(0, ".")
from app import app

client = TestClient(app)

payload = {
    "flare_gas_rate_mcf": 2500,
    "methane_pct": 75,
    "ethane_pct": 12,
    "propane_pct": 6,
    "butane_pct": 3,
    "ambient_temp_c": 25,
    "wind_speed_ms": 5,
    "steam_injection_rate": 1.0,
    "flare_efficiency_pct": 85,
}


def test_health():
    r = client.get("/api/health")
    data = r.json()
    assert data.get("status") == "ok", f"Unexpected health status: {data}"
    print("[PASS] /api/health")


def test_models():
    r = client.get("/api/models")
    data = r.json()
    assert "recovery_rate_mcf" in data or "message" in data, "Unexpected response"
    print("[PASS] /api/models")


def test_optimize():
    r = client.post("/api/optimize", json=payload)
    data = r.json()
    assert r.status_code == 200
    assert "recovery_rate_mcf" in data
    assert "economic_savings_usd" in data
    print(f"[PASS] /api/optimize -> recovery={data['recovery_rate_mcf']:.1f} MCF, savings=${data['economic_savings_usd']:.0f}")


def test_emissions():
    r = client.post("/api/emissions", json=payload)
    data = r.json()
    assert r.status_code == 200
    assert "co2_emissions_tons" in data
    print(f"[PASS] /api/emissions -> CO2={data['co2_emissions_tons']:.2f} tons")


def main():
    print("=" * 50)
    print("  FLARE GAS RECOVERY - API TESTS")
    print("=" * 50)

    results = []
    for name, fn in [("health", test_health), ("models", test_models),
                     ("optimize", test_optimize), ("emissions", test_emissions)]:
        try:
            fn()
            results.append(True)
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"  RESULTS: {passed}/{total} tests passed")
    print(f"{'=' * 50}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
