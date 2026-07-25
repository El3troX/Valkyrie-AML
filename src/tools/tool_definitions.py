import pandas as pd
from report_compiler import generate_sar_narrative, compile_pdf
from graph_engine import TransactionGraph
from ml_subsystems import AnomalyDetector, ExplainabilityEngine

# ---------------------------------------------------------------
# Tool definitions – each entry maps a tool name to a callable that
# performs the requested action. These functions are thin wrappers
# around the existing subsystem APIs. They are deliberately simple
# because the heavy‑lifting is already implemented elsewhere.
# ---------------------------------------------------------------

def get_anomaly_scores(top_n: int = 10, account_id: str = None) -> pd.DataFrame:
    """Return a DataFrame of anomaly scores.
    If *account_id* is provided, filter to that account; otherwise return the
    top *top_n* accounts sorted by score descending.
    """
    # Load the dataset (this is a placeholder – in the real system the
    # detector would already have the data loaded).
    df = pd.read_csv('data/SAML-D.csv')
    detector = AnomalyDetector(contamination=0.005)
    detector.fit(df)
    scores, _ = detector.predict(df)
    df['_score'] = scores
    if account_id:
        df = df[df['Sender_account'].astype(str) == account_id]
        return df
    return df.sort_values('_score', ascending=False).head(top_n)


def search_transactions(account: str, limit: int = 20) -> pd.DataFrame:
    """Return the most recent *limit* transactions for *account*.
    This function assumes the CSV has a column named 'Sender_account'.
    """
    df = pd.read_csv('data/SAML-D.csv')
    mask = df['Sender_account'].astype(str) == account
    return df[mask].head(limit)


def compute_network_risk(seed_accounts: list, max_hops: int = 2) -> dict:
    """Run the transaction graph and return top connected accounts.
    Returns a dict with a key ``top_connections`` containing a list of tuples
    ``(account_id, score)``.
    """
    df = pd.read_csv('data/SAML-D.csv')
    graph = TransactionGraph(df)
    graph.build()
    ppr = graph.personalized_pagerank(seed_accounts)
    top = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:max_hops]
    return {"top_connections": top}


def generate_sar(account_id: str) -> str:
    """Generate a SAR narrative and PDF for *account_id* and return the PDF path.
    The function re‑uses the existing SAR generation utilities.
    """
    # Minimal anomaly data – in a real system this would be computed
    df = pd.read_csv('data/SAML-D.csv')
    mask = df['Sender_account'].astype(str) == account_id
    anomaly_data = {
        "total_amount": float(df.loc[mask, 'Amount'].sum()),
        "n_transactions": int(mask.sum()),
        "max_score": float(df.loc[mask, '_score'].max() if '_score' in df.columns else 0.0),
    }
    graph_data = compute_network_risk([account_id])
    shap_explanation = {"top_features": []}
    narrative = generate_sar_narrative(account_id, anomaly_data, graph_data, shap_explanation)
    txns = df[mask].head(10)
    pdf_path = compile_pdf(narrative, account_id, txns)
    return pdf_path

# Export a dict that the orchestrator can import.
TOOL_DEFINITIONS = {
    "get_anomaly_scores": get_anomaly_scores,
    "search_transactions": search_transactions,
    "compute_network_risk": compute_network_risk,
    "generate_sar": generate_sar,
}
