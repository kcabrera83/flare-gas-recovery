# Architecture - Flare Gas Recovery

## System Overview

```
+------------------+     +-------------------+     +------------------+
|   Data Layer     | --> |   Model Layer     | --> |   API Layer      |
| (Data Generator) |     | (3 Models)        |     | (Flask REST)     |
+------------------+     +-------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 | Dashboard Layer  |
                                                 | (HTML/CSS/JS)    |
                                                 +------------------+
```

## Components

### Data Layer

- **Source**: Synthetic data generator (`generate_synthetic_data`)
- **Samples**: 5,000 flare gas operation records
- **Features**: 9 parameters (gas composition, ambient conditions, operational settings)
- **Targets**: recovery_rate_mcf, economic_savings_usd, co2_emissions_tons

### Model Layer

#### Recovery Optimizer
- **Algorithm**: GradientBoostingRegressor
- **Task**: Predict optimal gas recovery rate (MCF) and economic savings (USD)
- **Input**: 9 flare gas features
- **Output**: Recovery rate (MCF) or economic savings (USD)
- **Serialization**: pickle (`.pkl`)
- **Scalers**: Separate StandardScaler per target

#### Emission Predictor
- **Algorithm**: RandomForestRegressor
- **Task**: Predict CO2 emissions in tons
- **Input**: 9 flare gas features
- **Output**: CO2 emissions (tons)
- **Serialization**: pickle (`.pkl`)
- **Scaler**: Dedicated StandardScaler

### Preprocessing Pipeline

- Per-model StandardScaler fitted during training
- 3 scalers saved: `scaler_recovery.pkl`, `scaler_savings.pkl`, `scaler_emission.pkl`
- No categorical features - all inputs are numeric

### API Layer

- **Framework**: Flask
- **Port**: 5011
- **Format**: JSON request/response
- **Endpoints**: 5 (optimize, emissions, health, models, docs)

### Dashboard Layer

- **Frontend**: HTML/CSS/JS (Jinja2 templates)
- **Charts**: Chart.js for visualization
- **Theme**: Dark theme UI

## Data Flow

1. **Input** -> JSON with 9 flare gas parameters
2. **Feature Extraction** -> Extract values in defined column order
3. **Scaling** -> Apply target-specific StandardScaler
4. **Prediction** -> GradientBoosting (recovery/savings) or RandomForest (emissions)
5. **Post-processing** -> Clamp negative values to 0, round results
6. **Output** -> JSON with predictions and echoed input

## Feature Order

```
[flare_gas_rate_mcf, methane_pct, ethane_pct, propane_pct, butane_pct,
 ambient_temp_c, wind_speed_ms, steam_injection_rate, flare_efficiency_pct]
```

## Project Structure

```
flare-gas-recovery/
├── flare_gas_recovery/
│   ├── __init__.py
│   ├── data_generator.py              # Synthetic data generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── recovery_optimizer.py      # GradientBoosting regressor
│   │   └── emission_predictor.py      # RandomForest regressor
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py            # Scaling and preprocessing
├── templates/
│   └── index.html                     # Dashboard UI
├── outputs/models/                    # 6 model/scaler artifacts + summary
├── train.py                           # Training pipeline
├── app.py                             # Flask API server
├── test_api.py                        # API test suite
├── requirements.txt
└── setup.py
```

## Saved Artifacts

| File | Description |
|------|-------------|
| recovery_optimizer.pkl | GradientBoosting model for recovery rate |
| savings_optimizer.pkl | GradientBoosting model for economic savings |
| emission_predictor.pkl | RandomForest model for CO2 emissions |
| scaler_recovery.pkl | Scaler for recovery rate features |
| scaler_savings.pkl | Scaler for savings features |
| scaler_emission.pkl | Scaler for emission features |
| training_summary.json | Training metrics and feature importances |
| dataset.csv | Generated training data |

## Model Evaluation

### Recovery Rate Model
- R2: ~0.90+
- RMSE, MAE reported in training summary

### Savings Model
- R2: ~0.88+
- Separate GradientBoosting with 300 estimators

### Emission Model
- R2: ~0.90+
- RandomForest with feature importance ranking

---

*Elaborado por Ing. Kelvin Cabrera*
