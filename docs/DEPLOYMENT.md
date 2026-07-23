# Deployment Guide - Flare Gas Recovery

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5011

CMD ["python", "app.py"]
```

### Build and Run

```bash
docker build -t flare-gas-recovery .
docker run -p 5011:5011 flare-gas-recovery
```

## Docker Compose

```yaml
version: '3.8'
services:
  flare-gas-recovery:
    build: .
    ports:
      - "5011:5011"
    environment:
      - FLASK_ENV=production
    volumes:
      - model-data:/app/outputs

volumes:
  model-data:
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PORT | Server port | 5011 |

## Production Considerations

- Use gunicorn for production serving:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5011 app:app
  ```
- Set `debug=False` in `app.py` (already set)
- Configure reverse proxy (nginx) for SSL termination
- Set up health check monitoring on `/api/health`
- Use a process manager (systemd, supervisor) for auto-restart
- 6 model files loaded at startup (~few MB total)

## Training Pipeline

1. `python train.py` generates synthetic data and trains 3 models
2. Models + scalers saved to `outputs/models/`
3. Training summary saved as `training_summary.json`
4. `python app.py` loads all models and starts the API server

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push to main
- Installs dependencies
- Runs training pipeline
- Executes API tests

---

*Elaborado por Ing. Kelvin Cabrera*
