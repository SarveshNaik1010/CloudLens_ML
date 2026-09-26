"""
CloudLens - shared model wrapper classes.
Kept in their own module (rather than inside the training scripts) so that
pickle can resolve them consistently regardless of which script loads the
.pkl file later (training, recommend.py, predict.py, an API server, etc).
"""
import pandas as pd
import numpy as np


class CloudLensForecaster:
    """Wraps the cost-forecasting XGBRegressor with the category vocab it was
    trained on, so a fresh batch of (provider, service, region) values is
    encoded the same way at inference time."""

    def __init__(self, model, categories: dict, feature_order: list):
        self.model = model
        self.categories = categories
        self.feature_order = feature_order

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, cats in self.categories.items():
            df[col] = pd.Categorical(df[col], categories=cats)
        return df[self.feature_order]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(self._prep(df)), 0, None)


class CloudLensOptimizer:
    """Wraps the resource right-sizing XGBClassifier with its category vocab
    and decision threshold."""

    def __init__(self, model, categories: dict, feature_order: list, threshold: float = 0.5):
        self.model = model
        self.categories = categories
        self.feature_order = feature_order
        self.threshold = threshold

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, cats in self.categories.items():
            df[col] = pd.Categorical(df[col], categories=cats)
        return df[self.feature_order]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prep(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)
