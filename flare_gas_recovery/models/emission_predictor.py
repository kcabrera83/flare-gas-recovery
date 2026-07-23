import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EmissionPredictor:
    """RandomForest model for CO2 emission prediction."""

    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 10),
            min_samples_split=kwargs.get("min_samples_split", 5),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = {
            "r2": round(r2_score(y_test, y_pred), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "target": "co2_emissions_tons",
        }
        return metrics

    def feature_importances(self, feature_names):
        importances = self.model.feature_importances_
        return {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, importances)
        }
