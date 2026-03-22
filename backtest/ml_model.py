"""
Trade filter classifier.

Predicts the probability that a VWAP signal will be a profitable trade.
Tries backends in order: LightGBM → XGBoost → sklearn RandomForestClassifier.

Usage
-----
    model = TradeFilterModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)  # shape (n,)
    model.save("NSE_INFY")
    model = TradeFilterModel.load("NSE_INFY")
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest_results")
RANDOM_SEED = 42


class TradeFilterModel:
    """
    Binary classifier: 1 = profitable trade, 0 = unprofitable.
    `predict_proba` returns the probability of class 1.
    """

    def __init__(self, n_estimators: int = 200):
        self.n_estimators   = n_estimators
        self.model          = None
        self.feature_names: list[str] = []
        self._backend:      str       = "none"
        self._col_means:    np.ndarray | None = None  # for NaN imputation at inference

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TradeFilterModel":
        """
        Train the classifier.

        Parameters
        ----------
        X : feature DataFrame (n_samples × n_features)
        y : binary Series  (1 = profitable, 0 = not)
        """
        self.feature_names = list(X.columns)
        X_arr = X.values.astype(float)
        y_arr = y.values.astype(int)

        # Drop rows where *any* feature is NaN
        mask      = ~np.isnan(X_arr).any(axis=1)
        X_arr     = X_arr[mask]
        y_arr     = y_arr[mask]

        if len(X_arr) < 10:
            raise ValueError(
                f"Only {len(X_arr)} clean training samples — need at least 10."
            )

        # Store column means for inference-time NaN imputation
        self._col_means = np.nanmean(X_arr, axis=0)

        self.model = self._build_model()
        # Fit with a named DataFrame so LightGBM/XGBoost track feature names
        X_named = pd.DataFrame(X_arr, columns=self.feature_names)
        self.model.fit(X_named, y_arr)
        return self

    def _build_model(self):
        """Return an unfitted estimator, trying backends in order."""
        try:
            import lightgbm as lgb
            self._backend = "lightgbm"
            return lgb.LGBMClassifier(
                n_estimators  = self.n_estimators,
                learning_rate = 0.05,
                num_leaves    = 31,
                min_child_samples = 10,
                random_state  = RANDOM_SEED,
                verbose       = -1,
                n_jobs        = 1,
            )
        except ImportError:
            pass

        try:
            import xgboost as xgb
            self._backend = "xgboost"
            return xgb.XGBClassifier(
                n_estimators  = self.n_estimators,
                learning_rate = 0.05,
                max_depth     = 4,
                random_state  = RANDOM_SEED,
                eval_metric   = "logloss",
                verbosity     = 0,
                n_jobs        = 1,
            )
        except ImportError:
            pass

        from sklearn.ensemble import RandomForestClassifier
        self._backend = "randomforest"
        return RandomForestClassifier(
            n_estimators = self.n_estimators,
            max_depth    = 6,
            random_state = RANDOM_SEED,
            n_jobs       = 1,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability of class 1 (profitable trade) for each row.
        Falls back to 0.5 if the model has not been trained yet.
        """
        if self.model is None:
            return np.full(len(X), 0.5)

        # Align to training feature order and impute NaNs
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.0).copy()
        if self._col_means is not None:
            for i, col in enumerate(self.feature_names):
                mask = X_aligned[col].isna()
                if mask.any():
                    X_aligned.loc[mask, col] = float(self._col_means[i])

        # Predict with a named DataFrame (consistent with how the model was fitted)
        proba = self.model.predict_proba(X_aligned.astype(float))
        return proba[:, 1]

    def predict_proba_single(self, feature_dict: dict) -> float:
        """Convenience: predict for a single row given as a dict."""
        row = pd.DataFrame([feature_dict])
        return float(self.predict_proba(row)[0])

    # ── Threshold optimisation ─────────────────────────────────────────────────

    def find_optimal_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidates: list[float] | None = None,
        min_trades: int = 5,
    ) -> float:
        """
        Find the probability threshold that maximises win-rate on the
        supplied (training) data, subject to a minimum trade-count floor.

        Sweeps `candidates` thresholds.  For each threshold t:
          - Keep only rows where predicted_prob >= t
          - Compute win_rate on those rows
          - Apply a soft penalty when n_kept < min_trades

        Returns the threshold with the highest penalised score.
        Falls back to 0.55 if the model is untrained or no threshold wins.

        Parameters
        ----------
        X          : feature DataFrame (same rows as y)
        y          : binary labels (1=profitable, 0=not)
        candidates : thresholds to sweep; defaults to 0.50–0.85 in 0.05 steps
        min_trades : minimum kept trades before penalty kicks in
        """
        if self.model is None or X.empty:
            return 0.55

        if candidates is None:
            candidates = [round(t, 2) for t in
                          [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]]

        probs = self.predict_proba(X)
        y_arr = y.values

        best_score  = -1.0
        best_thresh = 0.55

        for t in candidates:
            mask = probs >= t
            n    = mask.sum()
            if n == 0:
                continue
            win_rate = float(y_arr[mask].mean())
            # Soft penalty: below min_trades, linearly reduce score
            penalty  = max(0.0, (min_trades - n) / min_trades) * 0.3
            score    = win_rate - penalty
            if score > best_score:
                best_score  = score
                best_thresh = t

        return best_thresh

    def is_trained(self) -> bool:
        return self.model is not None

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self) -> dict:
        """
        Return {feature_name: importance_score} sorted descending.
        Normalised to sum to 1 for comparability across backends.
        """
        if self.model is None:
            return {}
        try:
            imp = self.model.feature_importances_
        except AttributeError:
            return {}

        total = imp.sum()
        if total > 0:
            imp = imp / total

        pairs = sorted(zip(self.feature_names, imp), key=lambda x: x[1], reverse=True)
        return {k: round(float(v), 6) for k, v in pairs}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, tag: str = "UNIVERSE") -> str:
        """Serialise to <RESULTS_DIR>/<tag>_ml_model.pkl. Returns path."""
        import joblib
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"{tag}_ml_model.pkl")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(tag: str = "UNIVERSE") -> "TradeFilterModel":
        """Load a previously saved model."""
        import joblib
        path = os.path.join(MODELS_DIR, f"{tag}_ml_model.pkl")
        return joblib.load(path)

    def __repr__(self) -> str:
        status = f"trained ({self._backend})" if self.model else "untrained"
        return f"TradeFilterModel({status}, features={self.feature_names})"
