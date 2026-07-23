import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:5011"


def test_health():
    try:
        resp = urllib.request.urlopen(f"{BASE}/api/health", timeout=5)
        data = json.loads(resp.read())
        assert data.get("status") == "ok", f"Unexpected health status: {data}"
        print("[PASS] /api/health")
        return True
    except Exception as e:
        print(f"[FAIL] /api/health: {e}")
        return False


def test_models():
    try:
        resp = urllib.request.urlopen(f"{BASE}/api/models", timeout=5)
        data = json.loads(resp.read())
        assert "recovery_rate_mcf" in data or "message" in data, "Unexpected response"
        print("[PASS] /api/models")
        return True
    except Exception as e:
        print(f"[FAIL] /api/models: {e}")
        return False


def test_optimize():
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
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{BASE}/api/optimize",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        assert "recovery_rate_mcf" in data, "Missing recovery_rate_mcf"
        assert "economic_savings_usd" in data, "Missing economic_savings_usd"
        print(f"[PASS] /api/optimize -> recovery={data['recovery_rate_mcf']:.1f} MCF, savings=${data['economic_savings_usd']:.0f}")
        return True
    except Exception as e:
        print(f"[FAIL] /api/optimize: {e}")
        return False


def test_emissions():
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
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{BASE}/api/emissions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        assert "co2_emissions_tons" in data, "Missing co2_emissions_tons"
        print(f"[PASS] /api/emissions -> CO2={data['co2_emissions_tons']:.2f} tons")
        return True
    except Exception as e:
        print(f"[FAIL] /api/emissions: {e}")
        return False


def test_dashboard():
    try:
        resp = urllib.request.urlopen(f"{BASE}/", timeout=5)
        html = resp.read().decode()
        assert "Flare Gas Recovery" in html, "Dashboard content missing"
        print("[PASS] / (dashboard)")
        return True
    except Exception as e:
        print(f"[FAIL] / (dashboard): {e}")
        return False


def main():
    print("=" * 50)
    print("  FLARE GAS RECOVERY - API TESTS")
    print("=" * 50)

    # Wait for server
    for i in range(10):
        try:
            urllib.request.urlopen(f"{BASE}/api/health", timeout=2)
            break
        except Exception:
            if i == 9:
                print("ERROR: Server not reachable at", BASE)
                sys.exit(1)
            time.sleep(1)

    results = [
        test_health(),
        test_models(),
        test_optimize(),
        test_emissions(),
        test_dashboard(),
    ]

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"  RESULTS: {passed}/{total} tests passed")
    print(f"{'=' * 50}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
