# Flare Gas Recovery

ML-based optimization system for flare gas recovery in industrial operations using distributed computing and high-performance data processing.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Distributed Computing | **Dask** - parallel data processing |
| High-Performance Data | **Polars** - fast DataFrame operations |
| Data Processing | pandas, numpy, joblib |
| Web Server | **FastAPI** + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |
| Visualization | matplotlib, seaborn |

### Key Libraries
- Dask - Parallel and out-of-core computing
- Polars - High-performance DataFrame library
- FastAPI - Modern async web framework
- pandas / numpy - Data processing

## Overview

This project uses machine learning models to optimize flare gas recovery rates,
predict CO2 emissions, and estimate economic savings from gas recovery systems.

### Models

- **Recovery Optimizer** (Dask + Polars pipeline) - Predicts optimal recovery rate (MCF)
  and economic savings (USD).
- **Emission Predictor** (RandomForest) - Predicts CO2 emissions in tons.

## Quick Start

```bash
pip install -r requirements.txt
python train.py
python app.py
```

The FastAPI server starts on port 5011.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/api/optimize` | POST | Optimize recovery rate and savings |
| `/api/emissions` | POST | Predict CO2 emissions |
| `/api/models` | GET | List loaded models and metrics |
| `/api/health` | GET | Health check |

### Example Request

```bash
curl -X POST http://localhost:5011/api/optimize -H "Content-Type: application/json" -d '{
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

## Project Structure

```
flare-gas-recovery/
  flare_gas_recovery/
    __init__.py
    data_generator.py
    models/
      __init__.py
      recovery_optimizer.py
      emission_predictor.py
    utils/
      __init__.py
      preprocessor.py
  templates/
    index.html
  outputs/
    models/
  app.py
  train.py
  test_api.py
  requirements.txt
  setup.py
```

## Running Tests

```bash
python test_api.py
```

---

Elaborado por Ing. Kelvin Cabrera
