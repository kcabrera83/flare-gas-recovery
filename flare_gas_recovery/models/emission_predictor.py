import numpy as np
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import dask.array as da


class EmissionPredictor:
    """Dask-ML LinearRegression model for CO2 emission prediction."""

    def __init__(self, **kwargs):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train, y_train):
        X_da = da.from_array(np.asarray(X_train), chunks=-1)
        y_da = da.from_array(np.asarray(y_train), chunks=-1)
        X_scaled = self.scaler.fit_transform(X_da)
        self.model.fit(X_scaled, y_da)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        X_da = da.from_array(np.asarray(X), chunks=-1)
        X_scaled = self.scaler.transform(X_da)
        preds = self.model.predict(X_scaled)
        if hasattr(preds, "compute"):
            return preds.compute()
        return preds

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
        coef = self.model.coef_
        if hasattr(coef, "compute"):
            coef = coef.compute()
        importances = np.abs(coef)
        total = importances.sum()
        if total > 0:
            importances = importances / total
        return {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, importances)
        }
