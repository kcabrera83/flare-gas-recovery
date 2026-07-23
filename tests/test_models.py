import pytest
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(MODELS_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


def test_recovery_model_loads():
    import pickle
    path = os.path.join(MODELS_DIR, "recovery_optimizer.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_emission_model_loads():
    import pickle
    path = os.path.join(MODELS_DIR, "emission_predictor.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_recovery_prediction():
    import pickle
    import numpy as np
    with open(os.path.join(MODELS_DIR, "recovery_optimizer.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler_recovery.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X = np.array([[500.0, 80.0, 8.0, 5.0, 3.0, 25.0, 5.0, 10.0, 95.0]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    assert pred is not None
    assert len(pred) == 1
    assert pred[0] >= 0


def test_emission_prediction():
    import pickle
    import numpy as np
    with open(os.path.join(MODELS_DIR, "emission_predictor.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler_emission.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X = np.array([[500.0, 80.0, 8.0, 5.0, 3.0, 25.0, 5.0, 10.0, 95.0]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    assert pred is not None
    assert len(pred) == 1
    assert pred[0] >= 0
