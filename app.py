import streamlit as st
import pandas as pd
from pathlib import Path
import json
import networkx as nx
from pyvis.network import Network

# Project root
ROOT = Path(r"c:/Users/thund/OneDrive/Desktop/SocieteGenerale/Valkyrie-AML")
DATA_PATH = ROOT / "data" / "SAML-D.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def build_graph(df):
    from src.graph_engine import TransactionGraph
    graph = TransactionGraph(df)
    graph.build()
    return graph

st.title("Valkyrie AML Fund‑Flow Explorer")

df = load_data()
graph = build_graph(df)

# Sidebar inputs
source_account = st.sidebar.text_input("Source Account", value="207936746")
max_hops = st.sidebar.slider("Maximum hops", min_value=1, max_value=6, value=4)
min_amount = st.sidebar.number_input("Minimum transaction amount", min_value=0.0, value=0.0, step=0.01)

if st.sidebar.button("Trace Chains"):
    chains = graph.trace_chains(source_account, max_hops=max_hops, min_amount=min_amount)
    st.success(f"Found {len(chains)} chain(s) for account {source_account}")
    # Prepare table data
    table_data = []
    for i, chain in enumerate(chains, start=1):
        path_accounts = [chain[0]["sender"]] + [e["receiver"] for e in chain]
        amounts = [e["amount"] for e in chain]
        total_amount = sum(amounts)
        path_str = " → ".join(map(str, path_accounts))
        table_data.append({"Index": i, "Path": path_str, "Total Amount": total_amount, "Hops": len(chain)})
    st.dataframe(pd.DataFrame(table_data))
    # Download CSV
    csv = pd.DataFrame(table_data).to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name=f"fund_flows_{source_account}.csv", mime="text/csv")
    # Visualise network (first 20 chains for performance)
    if st.checkbox("Show network visualisation (first 20 chains)"):
        net = Network(height="600px", width="100%", directed=True)
        added_nodes = set()
        for chain in chains[:20]:
            for edge in chain:
                src = str(edge["sender"])
                dst = str(edge["receiver"])
                if src not in added_nodes:
                    net.add_node(src, label=src)
                    added_nodes.add(src)
                if dst not in added_nodes:
                    net.add_node(dst, label=dst)
                    added_nodes.add(dst)
                net.add_edge(src, dst, title=f"${edge['amount']:,}")
        net.show("network.html")
        HtmlFile = open("network.html", "r", encoding='utf-8')
        components = st.components.v1.html(HtmlFile.read(), height=600, scrolling=True)
else:
    st.info("Enter parameters and click 'Trace Chains' to begin.")
