"""
Valkyrie-AML — Compliance Investigation Agent entry point.

Usage:
    python run.py --full-pipeline    # Generate data, train, evaluate
    python run.py --evaluate         # Run evaluation and print metrics
    python run.py --dashboard        # Launch Streamlit UI
    python run.py --query "..."      # Run a natural-language query
    python run.py --tune             # Find optimal detection threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_data(nrows: int | None = None) -> pd.DataFrame:
    """Load SAML-D dataset from the data directory."""
    data_path = ROOT / "data" / "SAML-D.csv"
    if not data_path.exists():
        print(f"ERROR: SAML-D.csv not found at {data_path}")
        print("Download it from: https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml")
        sys.exit(1)
    print(f"Loading SAML-D from {data_path} ...")
    kwargs = {"nrows": nrows} if nrows else {}
    df = pd.read_csv(data_path, **kwargs)
    print(f"Loaded {len(df):,} rows, {df.columns.size} columns.")
    return df


def do_full_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline: train ML model, evaluate, save artifacts."""
    nrows = args.nrows
    df = load_data(nrows)

    from src.ml_subsystems import SupervisedDetector
    from src.evaluation import (
        map_typology,
        evaluate_detector,
        find_optimal_threshold,
        print_report,
        find_illustrative_cases,
    )

    print("\n--- Step 1: Train Supervised Detector ---")
    detector = SupervisedDetector()
    detector.fit(df)
    detector.save()  # Persist to disk
    scores, flags = detector.predict(df)

    print("\n--- Step 3: Evaluate ---")
    df_labeled = map_typology(df)
    best_t = find_optimal_threshold(df_labeled, scores)
    print(f"Optimal threshold (max F1): {best_t:.4f}")
    results = evaluate_detector(df_labeled, scores, best_t)
    print_report(results)

    print("\n--- Step 4: Illustrative Cases ---")
    cases = find_illustrative_cases(df_labeled, scores, n=3)
    for _, row in cases.iterrows():
        print(
            f"  score={row['anomaly_score']:.4f} "
            f"sender={row['Sender_account']} -> receiver={row['Receiver_account']} "
            f"amount=${row['Amount']:,.2f} type={row['Laundering_type']}"
        )

    # Save scores for reuse
    out_path = ROOT / "data" / "scores.npz"
    np.savez_compressed(out_path, scores=scores, flags=flags)
    print(f"Scores saved to {out_path}")

    # Top 5 flagged
    print("\n--- Top 5 Most Suspicious Transactions ---")
    top_idx = np.argsort(scores)[-5:][::-1]
    for idx in top_idx:
        row = df.iloc[idx]
        print(
            f"  idx={idx} score={scores[idx]:.4f} "
            f"sender={row['Sender_account']} receiver={row['Receiver_account']} "
            f"amount=${row['Amount']:,.2f} type={row['Laundering_type']}"
        )


def do_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation and print the report."""
    df = load_data(args.nrows)
    from src.evaluation import map_typology, evaluate_detector, find_optimal_threshold, print_report

    from src.ml_subsystems import SupervisedDetector
    detector = SupervisedDetector()
    detector.fit(df)
    scores, flags = detector.predict(df)

    df_labeled = map_typology(df)
    best_t = find_optimal_threshold(df_labeled, scores)
    results = evaluate_detector(df_labeled, scores, best_t)
    print_report(results)


def do_tune(args: argparse.Namespace) -> None:
    """Find the optimal threshold by sweeping values."""
    df = load_data(args.nrows)
    from src.ml_subsystems import AnomalyDetector
    from src.evaluation import map_typology, evaluate_detector, find_optimal_threshold

    detector = AnomalyDetector(contamination=0.001)
    detector.fit(df)
    scores, flags = detector.predict(df)
    df_labeled = map_typology(df)
    best_t = find_optimal_threshold(df_labeled, scores)

    print(f"\nOptimal threshold: {best_t:.4f}\n")
    print("Metrics at various thresholds:")
    for t in np.arange(0.1, 1.0, 0.1):
        results = evaluate_detector(df_labeled, scores, round(t, 2))
        o = results["overall"]
        print(
            f"  t={t:.1f}  precision={o['precision']:.4f}  recall={o['recall']:.4f}  "
            f"f1={o['f1']:.4f}  flagged={o['n_predicted_suspicious']:>6,}"
        )


def do_query(args: argparse.Namespace) -> None:
    """Run a natural-language query through the orchestrator."""
    df = load_data(args.nrows)
    from src.ml_subsystems import SupervisedDetector, ExplainabilityEngine
    from src.graph_engine import TransactionGraph

    print("Initializing subsystems ...")
    saved_path = ROOT / "data" / "models" / "supervised_detector.joblib"
    if saved_path.exists():
        print(f"Loading saved model from {saved_path} ...")
        detector = SupervisedDetector.load(saved_path)
    else:
        print("Training model (this may take a minute)...")
        detector = SupervisedDetector()
        detector.fit(df)
        detector.save()
    scores, flags = detector.predict(df)

    print("Building SHAP explainer ...")
    explainer = ExplainabilityEngine(detector)

    print("Building transaction graph ...")
    graph = TransactionGraph(df)
    graph.build()

    from src.evaluation import map_typology
    df_labeled = map_typology(df)

    from src.orchestrator import ValkyrieOrchestrator, _keyword_fallback
    orchestrator = ValkyrieOrchestrator(
        df=df_labeled,
        detector=detector,
        explainer=explainer,
        graph=graph,
        anomaly_scores=scores,
        binary_flags=flags,
    )

    # Use keyword fallback for speed (skip slow Ollama calls)
    plan = _keyword_fallback(args.query)
    results = {}
    for entry in plan.get("tools", []):
        name, params = entry.get("name", ""), entry.get("params", {})
        print(f"[Tool] Running {name}...")
        results[name] = orchestrator.executor.dispatch(name, params)
    result = {
        "plan": plan,
        "results": results,
        "summary": "Investigation complete. See tool results above.",
        "execution_log": [],
    }

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    for tool_name, tool_result in result["results"].items():
        print(f"\n--- {tool_name} ---")
        if isinstance(tool_result, dict):
            if "error" in tool_result:
                print(f"  ERROR: {tool_result['error']}")
            elif "top_anomalies" in tool_result:
                for item in tool_result["top_anomalies"][:5]:
                    print(f"  score={item['score']:.4f}  sender={item['sender']} -> receiver={item['receiver']}  ${item['amount']:,.2f}  type={item['type']}")
            elif "n_transactions" in tool_result:
                print(f"  Found {tool_result['n_transactions']} transactions totaling ${tool_result['total_amount']:,.2f}")
            else:
                for k, v in list(tool_result.items())[:5]:
                    print(f"  {k}: {v}")
        else:
            print(f"  {tool_result}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(result["summary"])


def do_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_script = str(ROOT / "src" / "dashboard.py")
    print(f"Launching Streamlit dashboard from {dashboard_script} ...")
    subprocess.run(["streamlit", "run", dashboard_script], check=True)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valkyrie-AML Compliance Investigation Agent",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Full pipeline: train -> evaluate -> report metrics",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation and print metrics report",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Find optimal detection threshold",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a natural-language investigation query",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit web dashboard",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=1_000_000,
        help="Number of SAML-D rows to load (default: 1,000,000)",
    )

    args = parser.parse_args()

    if args.full_pipeline:
        do_full_pipeline(args)
    elif args.evaluate:
        do_evaluate(args)
    elif args.tune:
        do_tune(args)
    elif args.query:
        do_query(args)
    elif args.dashboard:
        do_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()