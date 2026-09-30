import numpy as np
import pandas as pd

class DemographicEnsembleForecaster:
    """Comparative Ensemble Model for Longitudinal Demographic Projections."""
    def __init__(self):
        self.trained = False
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
            self.gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.use_sklearn = True
        except ImportError:
            self.use_sklearn = False
            self.weights = None

    def prepare_features(self, df: pd.DataFrame):
        df_sorted = df.sort_values(["country", "year"]).copy()
        df_sorted["pop_lag_1"] = df_sorted.groupby("country")["population"].shift(1)
        df_sorted["pop_lag_2"] = df_sorted.groupby("country")["population"].shift(2)
        df_clean = df_sorted.dropna().copy()
        
        feature_cols = ["year", "fertility_rate", "crude_death_rate", "urbanization_pct", "pop_lag_1", "pop_lag_2"]
        X = df_clean[feature_cols]
        y = df_clean["population"]
        return X, y, df_clean

    def train(self, X_train, y_train):
        if self.use_sklearn:
            self.rf.fit(X_train, y_train)
            self.gbr.fit(X_train, y_train)
        else:
            # High-performance closed-form Ridge Regression fallback
            X_mat = np.column_stack([np.ones(len(X_train)), X_train.values])
            y_vec = y_train.values
            lambda_reg = 1e-3
            self.weights = np.linalg.pinv(X_mat.T @ X_mat + lambda_reg * np.eye(X_mat.shape[1])) @ (X_mat.T @ y_vec)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Models must be trained before predicting.")
        if self.use_sklearn:
            rf_pred = self.rf.predict(X)
            gbr_pred = self.gbr.predict(X)
            return 0.5 * rf_pred + 0.5 * gbr_pred
        else:
            X_mat = np.column_stack([np.ones(len(X)), X.values])
            return X_mat @ self.weights

    def evaluate(self, y_true, y_pred) -> dict:
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        percentage_bias = np.mean((y_pred - y_true) / y_true) * 100.0
        return {
            "RMSE": round(float(rmse), 2),
            "MAE": round(float(mae), 2),
            "Percentage_Bias": round(float(percentage_bias), 4)
        }
