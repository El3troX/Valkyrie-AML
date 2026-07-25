"""
Ground-truth accuracy reporting for Valkyrie-AML.

Computes precision, recall, F1, and confusion matrix against the SAML-D
ground-truth labels, both overall and broken down per laundering typology.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Mapping from fine-grained SAML-D Laundering_type to simplified categories
# ---------------------------------------------------------------------------

TYPOLOGY_MAP: dict[str, str] = {
    # Normal
    "Normal_Cash_Deposits": "normal",
    "Normal_Fan_Out": "normal",
    "Normal_Small_Fan_Out": "normal",
    "Normal_Fan_In": "normal",
    "Normal_Group": "normal",
    "Normal_Cash_Withdrawal": "normal",
    "Normal_Periodical": "normal",
    "Normal_Plus_Mutual": "normal",
    "Normal_Mutual": "normal",
    "Normal_Foward": "normal",
    "Normal_single_large": "normal",
    # Structuring
    "Structuring": "structuring",
    "Cash_Withdrawal": "structuring",
    "Deposit-Send": "structuring",
    # Smurfing
    "Smurfing": "smurfing",
    "Fan_In": "smurfing",
    "Fan_Out": "smurfing",
    "Scatter-Gather": "smurfing",
    "Gather-Scatter": "smurfing",
    # Layering
    "Layered_Fan_In": "layering",
    "Layered_Fan_Out": "layering",
    "Stacked Bipartite": "layering",
    "Bipartite": "layering",
    "Cycle": "layering",
    "Behavioural_Change_1": "layering",
    "Behavioural_Change_2": "layering",
    "Single_large": "layering",
    "Over-Invoicing": "layering",
}

SUSPICIOUSTYPOLOGIES = {"structuring", "smurfing", "layering"}


def map_typology(df: pd.DataFrame) -> pd.DataFrame:
    """Add simplified ``typology`` column mapped from ``Laundering_type``."""
    df = df.copy()
    df["typology"] = df["Laundering_type"].map(TYPOLOGY_MAP).fillna("normal")
    return df


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_detector(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    label_col: str = "Is_laundering",
    typology_col: str = "typology",
) -> dict[str, Any]:
    """Evaluate detector against ground-truth labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *label_col* and *typology_col* columns.
    scores : np.ndarray
        Anomaly scores in [0, 1]; higher = more suspicious.
    threshold : float
        Score threshold for binary classification.

    Returns
    -------
    dict with ``overall`` and ``by_typology`` sub-dicts.
    """
    predicted = (scores >= threshold).astype(int)
    actual = df[label_col].values.astype(int)

    overall = {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
        "n_predicted_suspicious": int(predicted.sum()),
        "n_actual_suspicious": int(actual.sum()),
        "threshold": threshold,
    }

    # Per-typology breakdown (only suspicious typologies)
    by_typology: dict[str, dict[str, float]] = {}
    for typo in ["structuring", "smurfing", "layering"]:
        mask = df[typology_col].values == typo
        if mask.sum() == 0:
            by_typology[typo] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}
            continue
        typo_actual = actual[mask]
        typo_predicted = predicted[mask]
        by_typology[typo] = {
            "precision": float(precision_score(typo_actual, typo_predicted, zero_division=0)),
            "recall": float(recall_score(typo_actual, typo_predicted, zero_division=0)),
            "f1": float(f1_score(typo_actual, typo_predicted, zero_division=0)),
            "support": int(mask.sum()),
        }

    return {"overall": overall, "by_typology": by_typology}


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def print_report(results: dict[str, Any]) -> None:
    """Print a clean, presentation-ready evaluation table to stdout."""
    o = results["overall"]
    cm = np.array(o["confusion_matrix"])

    print("=" * 70)
    print("  VALKYRIE AML -- Model Evaluation Report")
    print("=" * 70)
    print()
    print(f"  Threshold:           {o['threshold']:.4f}")
    print(f"  Actual suspicious:   {o['n_actual_suspicious']:,}")
    print(f"  Predicted suspicious:{o['n_predicted_suspicious']:,}")
    print()
    print("  -- Overall Metrics " + "-" * 44)
    print(f"  Precision:  {o['precision']:.4f}")
    print(f"  Recall:     {o['recall']:.4f}")
    print(f"  F1 Score:   {o['f1']:.4f}")
    print()
    print("  -- Confusion Matrix " + "-" * 42)
    print(f"                    Predicted Normal  Predicted Suspicious")
    print(f"  Actual Normal     {cm[0][0]:>10,}          {cm[0][1]:>10,}")
    print(f"  Actual Suspicious {cm[1][0]:>10,}          {cm[1][1]:>10,}")
    print()
    print("  -- Per-Typology Breakdown " + "-" * 36)
    print(f"  {'Typology':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 58)
    for typo, m in results["by_typology"].items():
        print(
            f"  {typo:<16} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['support']:>10,}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Illustrative cases
# ---------------------------------------------------------------------------


def find_illustrative_cases(
    df: pd.DataFrame,
    scores: np.ndarray,
    n: int = 3,
) -> pd.DataFrame:
    """Surface the most compelling true-positive cases.

    Picks the *n* highest-scoring transactions that are actually
    suspicious (``Is_laundering == 1``), prioritising diverse
    typologies and high anomaly scores.

    Returns
    -------
    pd.DataFrame with columns:
        ``Sender_account``, ``Receiver_account``,
        ``Amount``, ``Laundering_type``, ``typology``, ``anomaly_score``.
    """
    suspicious_mask = df["Is_laundering"].values == 1
    suspicious_scores = scores.copy()
    suspicious_scores[~suspicious_mask] = -1

    top_idx = np.argsort(suspicious_scores)[-n * 3:][::-1]  # over-sample for diversity

    # Pick top n ensuring typology diversity
    selected: list[int] = []
    seen_types: set[str] = set()
    for idx in top_idx:
        typo = df.iloc[idx].get("typology", "unknown")
        if typo not in seen_types or len(selected) < n:
            selected.append(int(idx))
            seen_types.add(typo)
        if len(selected) >= n:
            break

    rows = []
    for idx in selected:
        row = df.iloc[idx].to_dict()
        row["anomaly_score"] = float(scores[idx])
        row["_idx"] = idx
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------


def find_optimal_threshold(
    df: pd.DataFrame,
    scores: np.ndarray,
    label_col: str = "Is_laundering",
) -> float:
    """Sweep thresholds 0.05-0.95 and return the one maximising F1 using fast numpy arithmetic."""
    actual = df[label_col].values.astype(int)
    total_pos = int(actual.sum())
    if total_pos == 0:
        return 0.5

    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        pred = scores >= t
        tp = int((pred & (actual == 1)).sum())
        fp = int((pred & (actual == 0)).sum())
        fn = total_pos - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return round(best_t, 4)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "SAML-D.csv"
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    from ml_subsystems import AnomalyDetector

    print(f"Loading SAML-D (sample 50 000 rows) ...")
    df = pd.read_csv(data_path, nrows=50_000)
    df = map_typology(df)

    print("Training detector ...")
    detector = AnomalyDetector(contamination=0.005)
    detector.fit(df)

    scores, flags = detector.predict(df)

    # Sweep for optimal threshold
    best_t = find_optimal_threshold(df, scores)
    print(f"\nOptimal threshold (max F1): {best_t:.4f}")

    results = evaluate_detector(df, scores, best_t)
    print_report(results)

    print("\n-- Illustrative Cases " + "-" * 40)
    cases = find_illustrative_cases(df, scores, n=3)
    for _, row in cases.iterrows():
        print(
            f"  idx={int(row['_idx'])} score={row['anomaly_score']:.4f} "
            f"sender={row['Sender_account']} -> receiver={row['Receiver_account']} "
            f"amount=${row['Amount']:,.2f} type={row['Laundering_type']}"
        )
