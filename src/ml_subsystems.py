"""
Core ML detection engine for Valkyrie-AML.

Provides Isolation Forest anomaly detection with custom feature engineering
and SHAP-based explainability, adapted for the SAML-D transaction schema.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CASH_PAYMENT_TYPES = {"Cash Deposit", "Cash Withdrawal"}
CROSS_BORDER_TYPES = {"Cross-border"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer ML features from SAML-D transaction data.

    Data-driven features based on analysis of what actually discriminates
    suspicious from normal transactions in the SAML-D dataset:

      - Cross-currency and cross-border signals (3-5x suspicious ratio)
      - Cash payment types (4-5x suspicious ratio)
      - Amount magnitude (5x suspicious ratio)
      - Account velocity patterns (inverted: suspicious has fewer txns)
      - Receiver multi-sender ratio (1.8x suspicious ratio)

    Produces 10 features, all with correct directionality.
    """
    df = df.copy()

    # --- 1. Cross-currency flag (3.5x suspicious ratio) ---
    df["is_cross_currency"] = (
        df["Payment_currency"] != df["Received_currency"]
    ).astype(int)

    # --- 2. Cross-border location flag (3.3x suspicious ratio) ---
    df["is_cross_border"] = (
        df["Sender_bank_location"] != df["Receiver_bank_location"]
    ).astype(int)

    # --- 3. Cash payment flag (5x suspicious ratio) ---
    df["is_cash_payment"] = df["Payment_type"].isin(CASH_PAYMENT_TYPES).astype(int)

    # --- 4. Cross-border payment type flag (2.8x suspicious ratio) ---
    df["is_cross_border_type"] = df["Payment_type"].isin(CROSS_BORDER_TYPES).astype(int)

    # --- 5. Log-transformed amount (captures 5x mean difference) ---
    df["amount_log"] = np.log1p(df["Amount"])

    # --- 6. Large amount flag (5.4x suspicious ratio) ---
    df["is_large_amount"] = (df["Amount"] >= 50_000).astype(int)

    # --- 7. Receiver multi-sender ratio (1.8x suspicious ratio) ---
    df["receiver_senders_ratio"] = df.groupby("Receiver_account")[
        "Sender_account"
    ].transform("nunique").astype(float)

    # --- 8. Sender transaction count (inverted: suspicious senders have fewer txns) ---
    df["sender_tx_count"] = df.groupby("Sender_account")["Amount"].transform("count").astype(float)

    # --- 9. Receiver transaction count (inverted: suspicious receivers have fewer txns) ---
    df["receiver_tx_count"] = df.groupby("Receiver_account")["Amount"].transform("count").astype(float)

    # --- 10. Amount z-score vs sender average (captures unusual amounts per account) ---
    sender_avg = df.groupby("Sender_account")["Amount"].transform("mean")
    sender_std = df.groupby("Sender_account")["Amount"].transform("std").fillna(1.0)
    df["amount_zscore"] = ((df["Amount"] - sender_avg) / (sender_std + 1e-8)).clip(-5, 10)

    return df


# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "is_cross_currency",
    "is_cross_border",
    "is_cash_payment",
    "is_cross_border_type",
    "amount_log",
    "is_large_amount",
    "receiver_senders_ratio",
    "sender_tx_count",
    "receiver_tx_count",
    "amount_zscore",
]


class AnomalyDetector:
    """Isolation Forest anomaly detector with SAML-D feature engineering.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies.  Default 0.001 matches the
        ~0.1 % suspicious rate in SAML-D.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.001,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler = StandardScaler()
        self._feature_names: list[str] = FEATURE_COLS.copy()

    def fit(self, df: pd.DataFrame) -> AnomalyDetector:
        """Engineer features, scale, and fit the Isolation Forest.

        Parameters
        ----------
        df : pd.DataFrame
            Raw SAML-D transactions.

        Returns
        -------
        self
        """
        print("[AnomalyDetector] Engineering features ...")
        featured = engineer_features(df)
        X = featured[self._feature_names].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        print(f"[AnomalyDetector] Fitting IsolationForest on {X.shape[0]} rows ...")
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        print("[AnomalyDetector] Fit complete.")
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Score transactions and return binary flags.

        Returns
        -------
        scores : np.ndarray
            Anomaly scores in [0, 1].  Higher = more suspicious.
        flags : np.ndarray
            Binary flags: 1 = flagged, 0 = clean.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        featured = engineer_features(df)
        X = featured[self._feature_names].fillna(0).values
        X_scaled = self.scaler.transform(X)

        raw_scores = self.model.decision_function(X_scaled)
        self.min_s, self.max_s = float(raw_scores.min()), float(raw_scores.max())
        scores = 1.0 - (raw_scores - self.min_s) / (self.max_s - self.min_s + 1e-12)

        flags = (self.model.predict(X_scaled) == -1).astype(int)
        return scores, flags

    def get_feature_names(self) -> list[str]:
        """Return the feature column names."""
        return self._feature_names.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike | None = None) -> str:
        """Serialize the fitted detector to disk.

        Parameters
        ----------
        path : str or None
            Destination path.  Defaults to ``data/models/anomaly_detector.joblib``.

        Returns
        -------
        str
            The absolute path the model was written to.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() before saving.")

        if path is None:
            _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = _DEFAULT_MODEL_DIR / "anomaly_detector.joblib"

        path = os.fspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self._feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }, path)
        print(f"[AnomalyDetector] Saved to {path}")
        return str(path)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> AnomalyDetector:
        """Load a previously saved detector from disk.

        Parameters
        ----------
        path : str or None
            Path to the joblib dump.  Defaults to ``data/models/anomaly_detector.joblib``.

        Returns
        -------
        AnomalyDetector
            A fully restored (fitted) instance.
        """
        if path is None:
            path = _DEFAULT_MODEL_DIR / "anomaly_detector.joblib"

        data = joblib.load(os.fspath(path))
        det = cls(
            contamination=data["contamination"],
            n_estimators=data["n_estimators"],
            random_state=data["random_state"],
        )
        det.model = data["model"]
        det.scaler = data["scaler"]
        det._feature_names = data["feature_names"]
        print(f"[AnomalyDetector] Loaded from {path}")
        return det


# ---------------------------------------------------------------------------
# Supervised classifier (uses labels for training — for demo evaluation)
# ---------------------------------------------------------------------------


class SupervisedDetector:
    """Random Forest classifier trained on ground-truth labels.

    This is used *after* the unsupervised approach to produce
    presentation-ready metrics for the demo.  It uses the same
    engineered features as the IsolationForest but trains a
    supervised model, then outputs anomaly scores = probability
    of being suspicious.

    In a real AML deployment, you'd use the unsupervised approach
    (since labels are rare).  This exists for demo purposes.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 20,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None
        self._feature_names: list[str] = FEATURE_COLS.copy()

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str = "Is_laundering",
    ) -> SupervisedDetector:
        """Engineer features and train the classifier.

        Uses class_weight="balanced" to handle extreme class imbalance
        (0.12% suspicious) without naive upsampling.

        Parameters
        ----------
        df : pd.DataFrame
            Raw SAML-D transactions with ground-truth labels.
        label_col : str
            Column name of the ground-truth label.

        Returns self.
        """
        print("[SupervisedDetector] Engineering features ...")
        featured = engineer_features(df)
        X = featured[self._feature_names].fillna(0).values
        y = df[label_col].values.astype(int)

        n_susp = y.sum()
        n_norm = len(y) - n_susp
        print(f"[SupervisedDetector] Class distribution: {n_norm:,} normal, {n_susp:,} suspicious ({n_susp/len(y)*100:.2f}%)")

        print(f"[SupervisedDetector] Training RandomForest (n_estimators={self.n_estimators}, max_depth={self.max_depth}) ...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        print("[SupervisedDetector] Fit complete.")
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return probability scores and binary flags.

        Scores are in [0, 1]; higher = more suspicious.
        Binary flags use a default 0.5 threshold.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        featured = engineer_features(df)
        X = featured[self._feature_names].fillna(0).values
        probs = self.model.predict_proba(X)[:, 1]
        scores = np.clip(probs, 0, 1)
        flags = (scores >= 0.5).astype(int)
        return scores, flags

    def get_feature_names(self) -> list[str]:
        return self._feature_names.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike | None = None) -> str:
        """Serialize the fitted detector to disk.

        Parameters
        ----------
        path : str or None
            Destination path.  Defaults to ``data/models/supervised_detector.joblib``.

        Returns
        -------
        str
            The absolute path the model was written to.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() before saving.")

        if path is None:
            _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = _DEFAULT_MODEL_DIR / "supervised_detector.joblib"

        path = os.fspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump({
            "model": self.model,
            "feature_names": self._feature_names,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }, path)
        print(f"[SupervisedDetector] Saved to {path}")
        return str(path)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> SupervisedDetector:
        """Load a previously saved detector from disk.

        Parameters
        ----------
        path : str or None
            Path to the joblib dump.  Defaults to ``data/models/supervised_detector.joblib``.

        Returns
        -------
        SupervisedDetector
            A fully restored (fitted) instance.
        """
        if path is None:
            path = _DEFAULT_MODEL_DIR / "supervised_detector.joblib"

        data = joblib.load(os.fspath(path))
        det = cls(
            n_estimators=data["n_estimators"],
            max_depth=data["max_depth"],
            min_samples_leaf=data["min_samples_leaf"],
            random_state=data["random_state"],
        )
        det.model = data["model"]
        det._feature_names = data["feature_names"]
        print(f"[SupervisedDetector] Loaded from {path}")
        return det


# ---------------------------------------------------------------------------
# Explainability engine
# ---------------------------------------------------------------------------

# Human-readable descriptions for each feature
_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "is_cross_currency": "transaction uses different currencies for payment and receipt",
    "is_cross_border": "sender and receiver are in different countries",
    "is_cash_payment": "payment type is Cash Deposit or Cash Withdrawal",
    "is_cross_border_type": "payment type is Cross-border transfer",
    "amount_log": "log-transformed transaction amount",
    "is_large_amount": "transaction amount is at least $50,000",
    "receiver_senders_ratio": "number of unique senders sending to this receiver",
    "sender_tx_count": "total number of transactions from this sender",
    "receiver_tx_count": "total number of transactions to this receiver",
    "amount_zscore": "how unusual this amount is relative to the sender's average",
}


class ExplainabilityEngine:
    """SHAP-based explainability for individual anomaly predictions.

    Parameters
    ----------
    detector : AnomalyDetector or SupervisedDetector
        A fitted detector instance (must have ``.model`` and
        ``get_feature_names()``).
    """

    def __init__(self, detector: Any) -> None:
        if detector.model is None:
            raise RuntimeError("Detector must be fitted before creating explainer.")
        self.detector = detector
        self._explainer = shap.TreeExplainer(detector.model)
        self._has_scaler = hasattr(detector, "scaler")
        self._is_supervised = hasattr(detector, "balance") or type(detector).__name__ == "SupervisedDetector"

    def explain(self, df: pd.DataFrame, transaction_idx: int) -> dict[str, Any]:
        """Explain why a single transaction was scored the way it was.

        Parameters
        ----------
        df : pd.DataFrame
            The full SAML-D DataFrame (used to recompute features).
        transaction_idx : int
            Row index in *df* to explain.

        Returns
        -------
        dict with keys:
            ``transaction_idx``, ``anomaly_score``, ``top_features``
            (list of dicts with ``name``, ``value``, ``shap_value``,
            ``direction``, ``plain_language``), and
            ``overall_explanation`` (str).
        """
        featured = engineer_features(df)
        X = featured[self.detector.get_feature_names()].fillna(0).values

        # Scale only if detector has a scaler (AnomalyDetector)
        if self._has_scaler:
            X_input = self.detector.scaler.transform(X)
        else:
            X_input = X

        # SHAP values
        shap_values = self._explainer.shap_values(X_input[transaction_idx : transaction_idx + 1])
        if isinstance(shap_values, list):
            sv = shap_values[0][0]
        else:
            sv = np.asarray(shap_values).flatten()

        # Anomaly score
        if self._is_supervised:
            # Supervised model: use predict_proba[1] as score
            proba = self.detector.model.predict_proba(X_input[transaction_idx : transaction_idx + 1])[0]
            anomaly_score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Unsupervised IsolationForest: normalize decision_function using cached bounds
            raw_score = self.detector.model.decision_function(
                X_input[transaction_idx : transaction_idx + 1]
            )[0]
            min_s = getattr(self.detector, "min_s", -0.5)
            max_s = getattr(self.detector, "max_s", 0.5)
            anomaly_score = 1.0 - (raw_score - min_s) / (max_s - min_s + 1e-12)
            anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        feature_names = self.detector.get_feature_names()
        feature_values = X[transaction_idx]

        # Rank by absolute SHAP value
        ranked = sorted(
            zip(feature_names, feature_values, sv),
            key=lambda t: abs(t[2]),
            reverse=True,
        )

        top_features = []
        for name, val, shap_val in ranked[:3]:
            direction = "increases" if shap_val > 0 else "decreases"
            desc = _FEATURE_DESCRIPTIONS.get(name, name)
            lang = self._shap_to_plain(name, shap_val, val)
            top_features.append({
                "name": name,
                "value": float(val),
                "shap_value": float(shap_val),
                "direction": direction,
                "plain_language": lang,
            })

        overall = self._build_overall(anomaly_score, top_features)
        return {
            "transaction_idx": int(transaction_idx),
            "anomaly_score": float(anomaly_score),
            "top_features": top_features,
            "overall_explanation": overall,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _shap_to_plain(name: str, shap_val: float, feat_val: float) -> str:
        """Convert a single SHAP contribution into plain English."""
        if name == "is_cross_currency":
            if feat_val > 0:
                return "This transaction uses different currencies for payment and receipt -- a common laundering technique to obscure fund origins."
            return "Same currency used for payment and receipt."
        if name == "is_cross_border":
            if feat_val > 0:
                return "Sender and receiver are in different countries -- cross-border transfers are a layering red flag."
            return "Domestic transaction within the same country."
        if name == "is_cash_payment":
            if feat_val > 0:
                return "Cash payment (deposit or withdrawal) -- cash is harder to trace and a common laundering vehicle."
            return "Non-cash electronic payment."
        if name == "is_cross_border_type":
            if feat_val > 0:
                return "Cross-border payment type -- international transfers are frequently used in layering schemes."
            return "Domestic payment type."
        if name == "amount_log":
            return f"Transaction amount: ${np.expm1(feat_val):,.2f} (log-transformed)."
        if name == "is_large_amount":
            if feat_val > 0:
                return "This transaction is for $50,000 or more -- large transfers warrant enhanced scrutiny."
            return "Standard transaction amount."
        if name == "receiver_senders_ratio":
            return (
                f"This receiver has {int(feat_val)} unique senders -- "
                f"{'high fan-in suggests smurfing or funnel account activity' if feat_val > 3 else 'typical receiving pattern'}."
            )
        if name == "sender_tx_count":
            return (
                f"This sender has {int(feat_val)} total transactions -- "
                f"{'low activity may indicate a shell account' if feat_val < 3 else 'active account'}."
            )
        if name == "receiver_tx_count":
            return (
                f"This receiver has {int(feat_val)} total transactions -- "
                f"{'low volume may indicate a pass-through account' if feat_val < 3 else 'established account'}."
            )
        if name == "amount_zscore":
            direction = "unusually high" if shap_val > 0 else "unusually low"
            return f"This amount is {abs(feat_val):.1f} standard deviations {direction} compared to this sender's typical transactions."
        return f"{name} = {feat_val:.2f}"

    @staticmethod
    def _build_overall(score: float, features: list[dict]) -> str:
        """Build a one-sentence overall explanation."""
        top = features[0]["plain_language"] if features else ""
        if score > 0.9:
            severity = "Very high"
        elif score > 0.7:
            severity = "High"
        elif score > 0.5:
            severity = "Moderate"
        else:
            severity = "Low"
        return f"{severity} anomaly risk (score {score:.3f}). Key factor: {top}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "SAML-D.csv"
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    print(f"Loading SAML-D from {data_path} (sampling 50 000 rows) ...")
    df_full = pd.read_csv(data_path, nrows=50_000)
    print(f"Loaded {len(df_full)} rows.")

    detector = AnomalyDetector(contamination=0.005)
    detector.fit(df_full)

    scores, flags = detector.predict(df_full)
    print(f"\nAnomaly scores -- min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}")
    print(f"Flagged transactions: {flags.sum()} / {len(flags)}")

    # Top 5 most suspicious
    top_idx = np.argsort(scores)[-5:][::-1]
    print("\nTop 5 most suspicious transactions:")
    for idx in top_idx:
        row = df_full.iloc[idx]
        print(
            f"  idx={idx} score={scores[idx]:.4f} "
            f"sender={row['Sender_account']} receiver={row['Receiver_account']} "
            f"amount={row['Amount']:.2f} type={row['Laundering_type']}"
        )

    # SHAP explanation for the top flagged
    print("\n--- SHAP Explanation for top flagged transaction ---")
    explainer = ExplainabilityEngine(detector)
    explanation = explainer.explain(df_full, int(top_idx[0]))
    print(json.dumps(explanation, indent=2))
