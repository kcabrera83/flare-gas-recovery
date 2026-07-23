# API Documentation - Flare Gas Recovery

## Base URL

```
http://localhost:5011
```

## Endpoints

### GET /

Serve the main web dashboard UI.

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": 3
}
```

---

### GET /api/models

Return training summary and model metrics.

**Response:**
```json
{
  "recovery_rate_mcf": {
    "mse": 123.45,
    "rmse": 11.11,
    "mae": 8.50,
    "r2": 0.92
  },
  "economic_savings_usd": {
    "mse": 50000.0,
    "rmse": 223.61,
    "mae": 180.00,
    "r2": 0.88
  },
  "co2_emissions_tons": {
    "mse": 50.25,
    "rmse": 7.09,
    "mae": 5.50,
    "r2": 0.90
  },
  "feature_importances_recovery": {
    "flare_gas_rate_mcf": 0.35,
    "methane_pct": 0.20,
    "..."
  },
  "feature_importances_emission": {
    "flare_gas_rate_mcf": 0.40,
    "methane_pct": 0.15,
    "..."
  },
  "feature_columns": ["flare_gas_rate_mcf", "methane_pct", "..."],
  "n_samples": 5000
}
```

---

### POST /api/optimize

Optimize flare gas recovery rate and predict economic savings.

**Request:**
```json
{
  "flare_gas_rate_mcf": 2500,
  "methane_pct": 75,
  "ethane_pct": 12,
  "propane_pct": 6,
  "butane_pct": 3,
  "ambient_temp_c": 25,
  "wind_speed_ms": 5,
  "steam_injection_rate": 1.0,
  "flare_efficiency_pct": 85
}
```

**Required Fields:**

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| flare_gas_rate_mcf | float | MCF | Flare gas flow rate |
| methane_pct | float | % | Methane composition |
| ethane_pct | float | % | Ethane composition |
| propane_pct | float | % | Propane composition |
| butane_pct | float | % | Butane composition |
| ambient_temp_c | float | C | Ambient temperature |
| wind_speed_ms | float | m/s | Wind speed |
| steam_injection_rate | float | - | Steam injection rate |
| flare_efficiency_pct | float | % | Flare combustion efficiency |

**Response:**
```json
{
  "recovery_rate_mcf": 1875.50,
  "economic_savings_usd": 45230.75,
  "input": {
    "flare_gas_rate_mcf": 2500,
    "methane_pct": 75,
    "ethane_pct": 12,
    "propane_pct": 6,
    "butane_pct": 3,
    "ambient_temp_c": 25,
    "wind_speed_ms": 5,
    "steam_injection_rate": 1.0,
    "flare_efficiency_pct": 85
  }
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | Prediction error | `{"error": "<details>"}` |

---

### POST /api/emissions

Predict CO2 emissions from flare gas operations.

**Request:**
```json
{
  "flare_gas_rate_mcf": 2500,
  "methane_pct": 75,
  "ethane_pct": 12,
  "propane_pct": 6,
  "butane_pct": 3,
  "ambient_temp_c": 25,
  "wind_speed_ms": 5,
  "steam_injection_rate": 1.0,
  "flare_efficiency_pct": 85
}
```

**Response:**
```json
{
  "co2_emissions_tons": 125.30,
  "input": {
    "flare_gas_rate_mcf": 2500,
    "methane_pct": 75,
    "ethane_pct": 12,
    "propane_pct": 6,
    "butane_pct": 3,
    "ambient_temp_c": 25,
    "wind_speed_ms": 5,
    "steam_injection_rate": 1.0,
    "flare_efficiency_pct": 85
  }
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | Prediction error | `{"error": "<details>"}` |

---

### GET /api/docs

Return OpenAPI 3.0 specification.

---

## Feature Reference

| Feature | Unit | Description |
|---------|------|-------------|
| flare_gas_rate_mcf | MCF | Flare gas flow rate |
| methane_pct | % | Methane composition in gas |
| ethane_pct | % | Ethane composition in gas |
| propane_pct | % | Propane composition in gas |
| butane_pct | % | Butane composition in gas |
| ambient_temp_c | C | Ambient temperature |
| wind_speed_ms | m/s | Wind speed |
| steam_injection_rate | - | Steam injection rate for combustion |
| flare_efficiency_pct | % | Flare combustion efficiency |

## Error Codes

- **200**: Success
- **400**: Bad request (missing or invalid parameters)
- **500**: Internal server error

---

*Elaborado por Ing. Kelvin Cabrera*
