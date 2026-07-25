"""
Startup / initialization for Valkyrie-AML FastAPI server.
Loads 200K rows, trains model, builds graph — cached in memory.
Uses Grok API for LLM calls.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Grok API key from environment
GROK_API_KEY = os.environ.get("GROK_API_KEY")
NROWS = int(os.environ.get("NROWS", 200_000))

_system: dict[str, Any] | None = None


def get_system() -> dict[str, Any]:
    global _system
    if _system is None:
        _system = _initialize()
    return _system


def _initialize() -> dict[str, Any]:
    t0 = time.time()
    # Search locations for SAML-D.csv
    locations = [
        ROOT.parent / "archive" / "SAML-D.csv",
        ROOT / "archive" / "SAML-D.csv",
        ROOT / "data" / "SAML-D.csv",
    ]
    
    data_path = None
    for loc in locations:
        if loc.exists():
            data_path = loc
            break
            
    if data_path is None:
        print(f"ERROR: SAML-D.csv not found at any expected locations: {locations}")
        sys.exit(1)
        
    print(f"[Startup] Loading SAML-D ({NROWS:,} rows) from {data_path}...")
    df = pd.read_csv(data_path, nrows=NROWS)
    print(f"[Startup] Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    from src.ml_subsystems import SupervisedDetector, ExplainabilityEngine
    from src.evaluation import (
        map_typology,
        evaluate_detector,
        find_optimal_threshold,
        find_illustrative_cases,
    )
    from src.graph_engine import TransactionGraph

    df_labeled = map_typology(df)

    t1 = time.time()
    detector = SupervisedDetector()
    detector.fit(df_labeled)
    scores, flags = detector.predict(df_labeled)
    print(f"[Startup] Model trained in {time.time()-t1:.1f}s")

    t2 = time.time()
    best_t = find_optimal_threshold(df_labeled, scores)
    eval_results = evaluate_detector(df_labeled, scores, best_t)
    cases = find_illustrative_cases(df_labeled, scores, n=20)
    print(f"[Startup] Evaluation done in {time.time()-t2:.1f}s | F1={eval_results['overall']['f1']:.4f}")

    t3 = time.time()
    explainer = ExplainabilityEngine(detector)
    print(f"[Startup] SHAP explainer ready in {time.time()-t3:.1f}s")

    t4 = time.time()
    graph = TransactionGraph(df)
    graph.build()
    risk_seeds = graph.get_risk_seeds(scores, df["Sender_account"], top_n=10)
    print(f"[Startup] Graph built ({graph.stats['n_nodes']} nodes) in {time.time()-t4:.1f}s")

    print(f"[Startup] Total init: {time.time()-t0:.1f}s")
    return dict(
        df=df,
        df_labeled=df_labeled,
        detector=detector,
        explainer=explainer,
        graph=graph,
        scores=scores,
        flags=flags,
        eval_results=eval_results,
        best_threshold=best_t,
        illustrative_cases=cases,
        risk_seeds=risk_seeds,
        nrows=NROWS,
    )


def risk_label(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    return "LOW"


def escalation_action(score: float) -> str:
    if score >= 0.8:
        return "REPORT"
    elif score >= 0.6:
        return "FLAG FOR REVIEW"
    return "MONITOR"
