# User Guide - Flare Gas Recovery

## Overview

The Flare Gas Recovery system uses machine learning to optimize flare gas recovery rates, predict CO2 emissions, and estimate economic savings from gas recovery systems in industrial operations.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd flare-gas-recovery
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

This generates 5,000 synthetic samples and trains:
- Recovery Optimizer (GradientBoosting) - Predicts recovery rate (MCF) and savings (USD)
- Emission Predictor (RandomForest) - Predicts CO2 emissions (tons)

### Run the Server

```bash
python app.py
```

Open `http://localhost:5011` in your browser.

## Dashboard Features

- **Recovery Optimization Panel** - Input flare gas conditions and get recovery rate + economic savings
- **Emissions Prediction Panel** - Predict CO2 emissions from flare operations
- **Model Metrics** - View training metrics and feature importances
- **Dark Theme UI** - Modern dark-themed dashboard

## API Usage

### Optimize Recovery (curl)

```bash
curl -X POST http://localhost:5011/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "flare_gas_rate_mcf": 2500,
    "methane_pct": 75,
    "ethane_pct": 12,
    "propane_pct": 6,
    "butane_pct": 3,
    "ambient_temp_c": 25,
    "wind_speed_ms": 5,
    "steam_injection_rate": 1.0,
    "flare_efficiency_pct": 85
  }'
```

### Optimize Recovery (Python)

```python
import requests

response = requests.post("http://localhost:5011/api/optimize", json={
    "flare_gas_rate_mcf": 2500,
    "methane_pct": 75,
    "ethane_pct": 12,
    "propane_pct": 6,
    "butane_pct": 3,
    "ambient_temp_c": 25,
    "wind_speed_ms": 5,
    "steam_injection_rate": 1.0,
    "flare_efficiency_pct": 85
})
result = response.json()
print(f"Recovery rate: {result['recovery_rate_mcf']} MCF")
print(f"Economic savings: ${result['economic_savings_usd']}")
```

### Predict Emissions (curl)

```bash
curl -X POST http://localhost:5011/api/emissions \
  -H "Content-Type: application/json" \
  -d '{
    "flare_gas_rate_mcf": 2500,
    "methane_pct": 75,
    "ethane_pct": 12,
    "propane_pct": 6,
    "butane_pct": 3,
    "ambient_temp_c": 25,
    "wind_speed_ms": 5,
    "steam_injection_rate": 1.0,
    "flare_efficiency_pct": 85
  }'
```

### Predict Emissions (Python)

```python
import requests

response = requests.post("http://localhost:5011/api/emissions", json={
    "flare_gas_rate_mcf": 2500,
    "methane_pct": 75,
    "ethane_pct": 12,
    "propane_pct": 6,
    "butane_pct": 3,
    "ambient_temp_c": 25,
    "wind_speed_ms": 5,
    "steam_injection_rate": 1.0,
    "flare_efficiency_pct": 85
})
result = response.json()
print(f"CO2 emissions: {result['co2_emissions_tons']} tons")
```

### Check Health

```bash
curl http://localhost:5011/api/health
```

### Get Model Info

```bash
curl http://localhost:5011/api/models
```

## Typical Workflow

1. Enter flare gas composition and operating conditions
2. Use `/api/optimize` to get optimal recovery rate and savings
3. Use `/api/emissions` to estimate environmental impact
4. Compare scenarios by varying input parameters

## Running Tests

```bash
python test_api.py
```

## Troubleshooting

- **Models not loaded**: Run `python train.py` first
- **Missing fields**: All 9 feature fields are required
- **Port in use**: Change port in `app.py`

---

*Elaborado por Ing. Kelvin Cabrera*
