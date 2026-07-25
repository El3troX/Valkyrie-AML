"""Valkyrie AML Streamlit Dashboard

Provides a quick UI to:
- Load a configurable sample of the SAML‑D dataset
- Generate a SAR narrative for any account
- Trace multi‑hop fund‑flow chains
- Show the top anomalous transactions
- Visualise the transaction network around an account (via PyVis)

The script can be launched with `streamlit run src/app_streamlit.py` from the project root.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Local imports – the project root is on PYTHONPATH when run via streamlit
from ml_subsystems import AnomalyDetector, ExplainabilityEngine
from graph_engine import TransactionGraph
from orchestrator import ValkyrieOrchestrator

# ---------------------------------------------------------------------------
# Page configuration & title
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Valkyrie AML Dashboard", layout="wide")
st.title("🕵️ Valkyrie AML Investigation Dashboard")

# ---------------------------------------------------------------------------
# Sidebar – data loading options
# ---------------------------------------------------------------------------
st.sidebar.header("📊 Data Settings")
rows_option = st.sidebar.selectbox(
    "Rows to load",
    options=[10_000, 50_000, 100_000],
    index=0,
    help="Load a subset of the SAML‑D CSV for faster iteration.",
)

# Load data (cached to avoid re‑reading on every UI interaction)
@st.cache_data
def load_data(n_rows: int) -> pd.DataFrame:
    data_path = "data/SAML-D.csv"
    return pd.read_csv(data_path, nrows=n_rows)

df = load_data(rows_option)

# ---------------------------------------------------------------------------
# Initialise ML components (cached for performance)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_ml(df_: pd.DataFrame):
    detector = AnomalyDetector(contamination=0.005)
    detector.fit(df_)
    scores, flags = detector.predict(df_)
    explainer = ExplainabilityEngine(detector)
    graph = TransactionGraph(df_)
    graph.build()
    orchestrator = ValkyrieOrchestrator(df_, detector, explainer, graph, scores, flags)
    return detector, scores, flags, explainer, graph, orchestrator

detector, scores, flags, explainer, graph, orchestrator = init_ml(df)

# ---------------------------------------------------------------------------
# Sidebar – investigation parameters
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Investigation")
account_id = st.sidebar.text_input("Account ID", value="", help="Enter the account you wish to investigate.")
max_hops = st.sidebar.number_input("Max hops for fund‑flow trace", min_value=1, max_value=6, value=4)

# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------
if st.sidebar.button("Generate SAR Narrative"):
    if account_id:
        result = orchestrator.executor.dispatch("generate_sar", {"account_id": account_id})
        narrative = result.get("narrative", "[No narrative generated]")
        st.subheader("📄 SAR Narrative")
        st.text_area("", narrative, height=400)
    else:
        st.warning("Please provide an Account ID.")

if st.sidebar.button("Show Fund‑Flow Chains"):
    if account_id:
        flow = orchestrator.executor.dispatch(
            "trace_fund_flows", {"source_account": account_id, "max_hops": int(max_hops)}
        )
        n = flow.get("n_chains", 0)
        st.subheader(f"🔗 Found {n} fund‑flow chain(s)")
        for i, chain in enumerate(flow.get("chains", [])[:10], 1):
            # Build a readable representation
            path_parts = []
            for step in chain:
                path_parts.append(f"{step['sender']} → {step['receiver']} (${step['amount']:.2f})")
            path = " ➔ ".join(path_parts)
            st.markdown(f"**Chain #{i}** (length {len(chain)}): {path}")
    else:
        st.warning("Please provide an Account ID.")

if st.sidebar.button("Top Anomalies (10)"):
    top = orchestrator.executor.dispatch("get_anomaly_scores", {"top_n": 10})
    df_top = pd.DataFrame(top.get("top_anomalies", []))
    st.subheader("🔥 Top 10 Anomalous Transactions")
    st.dataframe(df_top)

if st.sidebar.button("Visualise Network Around Account"):
    if account_id:
        try:
            from pyvis.network import Network
        except Exception as e:
            st.error(f"PyVis library not available: {e}")
        else:
            vis_data = graph.export_for_pyvis(
                risk_scores={account_id: 1.0},
                focus_account=account_id,
                depth=2,
                max_nodes=200,
            )
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            for node in vis_data["nodes"]:
                net.add_node(
                    node["id"],
                    label=node["label"],
                    title=node["title"],
                    color=node["color"],
                    size=node["size"],
                )
            for edge in vis_data["edges"]:
                net.add_edge(
                    edge["from"],
                    edge["to"],
                    title=edge["title"],
                    label=edge["label"],
                )
            html_path = "network.html"
            net.show(html_path)
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=600, scrolling=True)
    else:
        st.warning("Please provide an Account ID.")

# ---------------------------------------------------------------------------
# Footer / credits
# ---------------------------------------------------------------------------
st.caption("Powered by Valkyrie‑AML – a LangGraph & Ollama driven AML investigation engine.")
