"""
Valkyrie-AML — Compliance investigation dashboard.

Neubrutalism-inspired dark theme. Provides:
- Natural-language query input with live state persistence
- Interactive PyVis transaction network graph with Typology Inference Guide & Chain Tracer
- Dynamic Model Performance panel with threshold slider integration & auto-tune button
- SAR PDF generation and download
"""
from __future__ import annotations

import json, os, sys, tempfile, time as _time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Valkyrie AML",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Neubrutalism CSS injection
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Neubrutalism design system — hard borders, offset shadows, bold contrast */
    .stApp { background-color: #020617; }

    section[data-testid="stSidebar"] { background-color: #0F172A; border-right: 3px solid #000000; }

    div[data-testid="stMetric"] {
        background: #1E293B;
        border: 3px solid #000000;
        box-shadow: 4px 4px 0 #000000;
        padding: 12px 16px;
        border-radius: 0;
    }

    div[data-testid="stMetric"] label p { font-weight: 700 !important; color: #94A3B8 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-weight: 800 !important; color: #22C55E !important; }

    div.stButton > button {
        border: 3px solid #000000 !important;
        box-shadow: 4px 4px 0 #000000 !important;
        font-weight: 700 !important;
        border-radius: 0 !important;
        transition: all 0.1s ease !important;
    }
    div.stButton > button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0 #000000 !important;
    }
    div.stButton > button[kind="primary"] {
        background: #22C55E !important;
        color: #000000 !important;
    }

    div[data-testid="stTabs"] button {
        border: 2px solid transparent !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom: 3px solid #22C55E !important;
        color: #22C55E !important;
    }

    div.stAlert { border: 3px solid #000000; box-shadow: 3px 3px 0 #000000; border-radius: 0; }
    div.st-bb { border: none !important; }

    div[data-testid="stDataFrame"] { border: 2px solid #334155; }

    .stStatusWidget { border: 3px solid #000000; box-shadow: 4px 4px 0 #000000; border-radius: 0; }

    h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.02em; }
    .st-cb { color: #F8FAFC !important; }
    hr { border-color: #334155 !important; }

    /* Risk badge colors */
    .risk-critical { color: #EF4444; font-weight: 700; }
    .risk-high { color: #F97316; font-weight: 700; }
    .risk-medium { color: #EAB308; font-weight: 700; }
    .risk-low { color: #22C55E; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# System initialisation (cached once)
# ---------------------------------------------------------------------------


@st.cache_resource
def init_system(nrows: int = 1_000_000) -> dict[str, Any]:
    """Load data, detector, explainer, graph — run once."""
    import time as _t
    t0 = _t.time()

    data_path = ROOT / "data" / "SAML-D.csv"
    print(f"[Init] Loading SAML-D ({data_path}, nrows={nrows:,}) ...")
    df = pd.read_csv(data_path, nrows=nrows)
    print(f"[Init] Loaded {len(df):,} rows in {_t.time()-t0:.1f}s")

    from ml_subsystems import SupervisedDetector, ExplainabilityEngine
    from evaluation import map_typology, evaluate_detector, find_optimal_threshold, find_illustrative_cases
    from graph_engine import TransactionGraph

    df_labeled = map_typology(df)

    t1 = _t.time()
    # Train detector specifically on loaded dataframe to ensure exact feature distribution scaling
    detector = SupervisedDetector()
    detector.fit(df_labeled)
    detector.save()

    scores, flags = detector.predict(df_labeled)
    best_t = find_optimal_threshold(df_labeled, scores)
    print(f"[Init] Model ready in {_t.time()-t1:.1f}s. Optimal F1 threshold: {best_t:.4f}")

    t2 = _t.time()
    explainer = ExplainabilityEngine(detector)
    print(f"[Init] SHAP ready in {_t.time()-t2:.1f}s")

    t3 = _t.time()
    graph = TransactionGraph(df); graph.build()
    print(f"[Init] Graph ready ({graph.stats['n_nodes']} nodes) in {_t.time()-t3:.1f}s")

    t4 = _t.time()
    eval_results = evaluate_detector(df_labeled, scores, best_t)
    cases = find_illustrative_cases(df_labeled, scores, n=5)
    risk_seeds = graph.get_risk_seeds(scores, df["Sender_account"], top_n=10)
    print(f"[Init] Eval ready in {_t.time()-t4:.1f}s")

    print(f"[Init] All done in {_t.time()-t0:.1f}s total")
    return dict(df=df, df_labeled=df_labeled, detector=detector, explainer=explainer,
                graph=graph, scores=scores, flags=flags, eval_results=eval_results,
                best_threshold=best_t, illustrative_cases=cases, risk_seeds=risk_seeds, ppr_scores=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_label(s: float) -> str: return "CRITICAL" if s >= 0.8 else "HIGH" if s >= 0.6 else "MEDIUM" if s >= 0.4 else "LOW"
def _risk_class(s: float) -> str: return "risk-critical" if s >= 0.8 else "risk-high" if s >= 0.6 else "risk-medium" if s >= 0.4 else "risk-low"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(best_threshold: float = 0.65) -> dict:
    with st.sidebar:
        st.markdown("# :material/security: Valkyrie")
        st.markdown("**AML compliance agent** · :material/link: Backend connected")
        st.markdown("---")

        nrows_options = {"500K": 500_000, "1M": 1_000_000, "2M": 2_000_000}
        nrows_label = st.selectbox("Dataset Size", list(nrows_options.keys()), index=1,
                                   help="Number of SAML-D rows to load. Changing this will re-initialize the engine.")
        nrows = nrows_options[nrows_label]
        st.markdown("---")

        q = st.text_area("Query", placeholder="e.g. Show me top 10 suspicious transactions",
                         height=100, label_visibility="collapsed", key="q_input")
        go = st.button(":material/play_arrow: Investigate", type="primary", use_container_width=True)
        if go and q.strip():
            st.session_state["active_query"] = q.strip()
            st.session_state["trigger_investigation"] = True

        st.markdown("**Filters**")
        default_th = st.session_state.get("threshold_slider", float(best_threshold))
        th = st.slider("Risk threshold", 0.0, 1.0, default_th, 0.05, key="threshold_slider")
        aid = st.text_input("Focus account", placeholder="e.g. 207936746", key="aid_input")
        st.markdown("---")
        st.caption(f":material/database: SAML-D {nrows_label} rows")
        st.caption(":material/model_training: RandomForest + SHAP")

    return dict(query=q.strip() if q else "", threshold=th, account_id=aid, nrows=nrows)


# ---------------------------------------------------------------------------
# Network graph
# ---------------------------------------------------------------------------


def render_network(system: dict, focus: str | None = None) -> None:
    """Render PyVis network graph with Typology Inference Guide."""
    st.markdown("### :material/hub: Transaction Network Graph & Risk Topology")

    with st.expander(":material/lightbulb: How to Read & Infer Laundering Patterns from this Graph", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### :material/palette: Node Colors (Risk Severity)")
            st.markdown("- :red[**Red ($\ge 0.8$)**]: High anomaly score (critical suspicious flag)")
            st.markdown("- :orange[**Orange ($\ge 0.6$)**]: Elevated velocity or amount z-score")
            st.markdown("- :yellow[**Yellow ($\ge 0.4$)**]: Moderate network risk exposure")
            st.markdown("- :green[**Green ($< 0.4$)**]: Baseline normal transaction account")
        with c2:
            st.markdown("#### :material/straighten: Node Sizes & Edges")
            st.markdown("- **Node Size**: Proportional to **Personalized PageRank (PPR)** — larger nodes are central hubs spreading risk across the graph.")
            st.markdown("- **Edge Arrows**: Money transfer direction ($A \\rightarrow B$).")
            st.markdown("- **Edge Labels**: Aggregated transfer amount ($).")
        with c3:
            st.markdown("#### :material/search: Visual Patterns to Infer")
            st.markdown("- **Smurfing (Funneling)**: Multiple senders transferring into one central receiver.")
            st.markdown("- **Layering (Chains)**: Sequential transfers $A \\rightarrow B \\rightarrow C \\rightarrow D$ within 72 hours.")
            st.markdown("- **Structuring**: High volume of transfers just under $\$10,000$.")

    try:
        from pyvis.network import Network
    except ImportError:
        st.warning(":material/error: pyvis not installed — run `pip install pyvis`"); return

    g, sc, df = system["graph"], system["scores"], system["df"]
    if focus:
        focus = focus.strip() or None

    # Vectorized risk map creation for both Senders and Receivers
    df_accs = pd.concat([
        pd.DataFrame({"account": df["Sender_account"].astype(str), "score": sc}),
        pd.DataFrame({"account": df["Receiver_account"].astype(str), "score": sc}),
    ])
    risk = df_accs.groupby("account")["score"].max().to_dict()

    # Compute PPR lazily
    ppr = system.get("ppr_scores")
    if ppr is None:
        with st.spinner(":material/hub: Computing PageRank (may take a moment) ..."):
            risk_seeds = system.get("risk_seeds", g.get_risk_seeds(sc, df["Sender_account"], top_n=10))
            ppr = g.personalized_pagerank(risk_seeds)
            system["ppr_scores"] = ppr

    export = g.export_for_pyvis(risk_scores=risk, pagerank_scores=ppr,
                                 focus_account=focus, depth=2, max_nodes=100)

    net = Network(height="600px", width="100%", directed=True,
                  bgcolor="#0F172A", font_color="#F8FAFC")
    net.set_options(json.dumps({
        "physics": {"solver": "forceAtlas2Based", "forceAtlas2Based": {"springLength": 120, "springConstant": 0.01}},
        "interaction": {"hover": True, "navigationButtons": True},
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}},
    }))
    for n in export["nodes"]:
        net.add_node(n["id"], label=n["label"], color=n["color"], size=n["size"], title=n["title"])
    for e in export["edges"]:
        net.add_edge(e["from"], e["to"], label=e.get("label", ""), title=e.get("title", ""))

    path = os.path.join(tempfile.gettempdir(), "valkyrie_graph.html")
    net.save_graph(path)

    with open(path, encoding="utf-8") as fh:
        st.components.v1.html(fh.read(), height=620)

    st.markdown("---")
    st.markdown("### :material/link: Multi-Hop Layering Chain Tracer")
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        seed_accs = system.get("risk_seeds", [])
        default_seed = focus if focus else (seed_accs[0] if seed_accs else "")
        source_account = st.text_input("Source Account ID to Trace Chains From", value=default_seed, placeholder="e.g. 207936746")
    with col_btn:
        st.write("")
        st.write("")
        trace_btn = st.button(":material/search: Trace Layering Chains", type="primary")

    if source_account and (trace_btn or focus):
        chains = system["graph"].trace_chains(source_account, max_hops=4, time_window_hours=72)
        if chains:
            st.success(f"Discovered **{len(chains)}** multi-hop layering transfer chain(s) from Account `{source_account}`")
            for idx, chain in enumerate(chains[:5]):
                st.markdown(f"**Chain #{idx+1} ({len(chain)} Hops):**")
                chain_str = " ➔ ".join([f"`{e['sender']}` (${e['amount']:,.2f})" for e in chain]) + f" ➔ `{chain[-1]['receiver']}`"
                st.markdown(f"> {chain_str}")
        else:
            st.info(f"No multi-hop transfer chains found from account `{source_account}` within 72-hour window.")


# ---------------------------------------------------------------------------
# Investigation panel
# ---------------------------------------------------------------------------


def render_investigation(system: dict, query: str, threshold: float) -> None:
    """Render LLM Orchestrator Investigation state."""
    active_q = st.session_state.get("active_query", query).strip()

    if not active_q:
        with st.container(border=True):
            st.markdown("### :material/search: Valkyrie Compliance Investigation Agent")
            st.markdown("Type an investigation query in the sidebar and click **Investigate**.")
            st.markdown("#### Sample Queries to Try:")
            st.markdown("- `Show me top 10 suspicious transactions`")
            st.markdown("- `Trace multi-hop fund flows from account 207936746`")
            st.markdown("- `Evaluate model performance and precision/recall metrics`")
            st.markdown("- `Generate SAR report for account 1291265097`")
        return

    # Trigger investigation if active query changed or explicit trigger set
    should_run = st.session_state.get("trigger_investigation", False) or ("last_executed_query" not in st.session_state) or (st.session_state.get("last_executed_query") != active_q)

    if should_run:
        st.session_state["trigger_investigation"] = False
        st.session_state["last_executed_query"] = active_q

        from orchestrator import ValkyrieOrchestrator
        orch = ValkyrieOrchestrator(df=system["df_labeled"], detector=system["detector"],
                                     explainer=system["explainer"], graph=system["graph"],
                                     anomaly_scores=system["scores"], binary_flags=system["flags"])

        with st.status(":material/account_tree: LangGraph Pipeline Executing ...", expanded=True) as s:
            t0 = _time.time()
            try:
                s.update(label=":material/schedule: LangGraph: plan_query ➔ execute_plan ➔ summarize ...")
                inv_res = orch.investigate(active_q)
                elapsed = _time.time() - t0

                s.update(label=f":material/check_circle: LangGraph Pipeline Complete ({elapsed:.1f}s)", state="complete")
                st.session_state["investigation_plan"] = inv_res.get("plan")
                st.session_state["investigation_results"] = inv_res.get("results", {})
                st.session_state["investigation_summary"] = inv_res.get("summary", "")
                st.session_state["investigation_elapsed"] = elapsed

            except Exception as e:
                s.update(label=f":material/error: LangGraph Pipeline Failed", state="error")
                st.error(str(e))
                return

    # Display persisted investigation results
    plan = st.session_state.get("investigation_plan", {})
    results = st.session_state.get("investigation_results", {})
    summary = st.session_state.get("investigation_summary", "")
    elapsed = st.session_state.get("investigation_elapsed", 0.0)

    tools = plan.get("tools", []) if plan else []

    with st.container(border=True):
        st.markdown(f"### :material/search: Investigation Query: *\"{active_q}\"*")
        if tools:
            tool_names = ", ".join(f"`{t.get('name', '')}`" for t in tools)
            st.caption(f":material/account_tree: **LangGraph Agent Plan**: {len(tools)} tool(s) — {tool_names} · Executed in **{elapsed:.1f}s**")

        if summary:
            st.info(f":material/auto_awesome: **Valkyrie Compliance Summary**:\n{summary}")

        for tn, tr in results.items():
            with st.expander(f":material/analytics: Subsystem Tool Execution: `{tn}`", expanded=True):
                if isinstance(tr, dict) and "top_anomalies" in tr:
                    dfr = pd.DataFrame(tr["top_anomalies"])
                    if "score" in dfr.columns:
                        dfr["risk"] = dfr["score"].apply(_risk_label)
                    st.dataframe(dfr, column_config={
                        "idx": st.column_config.NumberColumn("Row Index"),
                        "sender": st.column_config.TextColumn("Sender Account"),
                        "receiver": st.column_config.TextColumn("Receiver Account"),
                        "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                        "score": st.column_config.NumberColumn("Anomaly Score", format="%.4f"),
                        "type": st.column_config.TextColumn("Laundering Typology"),
                        "risk": st.column_config.TextColumn("Risk Severity"),
                    }, hide_index=True)
                elif isinstance(tr, dict) and tr.get("overall"):
                    o = tr["overall"]
                    st.success(f"**Precision**: {o['precision']:.1%} | **Recall**: {o['recall']:.1%} | **F1 Score**: {o['f1']:.1%} | **False Positives**: {o['confusion_matrix'][0][1]}")
                else:
                    st.json(tr)

    top = results.get("get_anomaly_scores", {}).get("top_anomalies", [])
    if top:
        with st.container(border=True):
            st.success(f"**{len(top)}** suspicious transactions identified · Peak anomaly score: **{max(a['score'] for a in top):.4f}**")


# ---------------------------------------------------------------------------
# Performance panel
# ---------------------------------------------------------------------------


def render_performance(system: dict, active_threshold: float) -> None:
    from evaluation import evaluate_detector, find_optimal_threshold

    # Compute dynamic metrics based on active slider threshold!
    ev = evaluate_detector(system["df_labeled"], system["scores"], active_threshold)
    o = ev["overall"]
    best_t = system.get("best_threshold", find_optimal_threshold(system["df_labeled"], system["scores"]))

    col_btn, col_blank = st.columns([1, 2])
    with col_btn:
        if st.button(f":material/target: Apply Optimal Threshold ({best_t:.2f})", type="primary", use_container_width=True):
            # Store the optimal threshold in a separate session_state key to avoid mutating the slider widget after creation
            st.session_state["active_threshold"] = float(best_t)
            st.rerun()

    with st.container(horizontal=True):
        st.metric("Precision", f"{o['precision']:.1%}", border=True)
        st.metric("Recall", f"{o['recall']:.1%}", border=True)
        st.metric("F1 Score", f"{o['f1']:.1%}", border=True)
        st.metric("False Positives", f"{o['confusion_matrix'][0][1]:,}", border=True)

    st.caption(f":material/tune: Active Threshold: **{o['threshold']:.4f}** (Optimal F1 threshold is **{best_t:.4f}**) · :material/warning: Actual Suspicious: **{o['n_actual_suspicious']:,}** · :material/flag: Predicted Suspicious: **{o['n_predicted_suspicious']:,}**")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Confusion Matrix**")
            st.dataframe(pd.DataFrame(o["confusion_matrix"], index=["Actual normal", "Actual suspicious"],
                                       columns=["Predicted normal", "Predicted suspicious"]), hide_index=False)
    with c2:
        with st.container(border=True):
            st.markdown("**Per-Typology Breakdown**")
            rows = [{"Typology": t.capitalize(), "Precision": f"{m['precision']:.1%}",
                     "Recall": f"{m['recall']:.1%}", "F1": f"{m['f1']:.1%}",
                     "Support": m["support"]} for t, m in ev["by_typology"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    with st.container(border=True):
        st.markdown("**Top Illustrative Detection Cases**")
        c = system["illustrative_cases"]
        if not c.empty:
            cols = [x for x in ["Sender_account", "Receiver_account", "Amount", "Laundering_type", "anomaly_score"] if x in c.columns]
            st.dataframe(c[cols], column_config={
                "Sender_account": st.column_config.TextColumn("Sender Account"),
                "Receiver_account": st.column_config.TextColumn("Receiver Account"),
                "Amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "anomaly_score": st.column_config.NumberColumn("Anomaly Score", format="%.4f"),
                "Laundering_type": st.column_config.TextColumn("Typology"),
            }, hide_index=True)


# ---------------------------------------------------------------------------
# SAR panel
# ---------------------------------------------------------------------------


def render_sar(system: dict, account_id: str | None) -> None:
    if not account_id or not account_id.strip():
        st.info(":material/description: Enter an account ID in the sidebar to generate a SAR.")
        return
    aid = account_id.strip()
    mask = system["df"]["Sender_account"].astype(str) == aid
    if mask.sum() == 0:
        st.warning(f"No transactions for account {aid}"); return

    if st.button(f":material/description: Generate SAR for {aid}", type="primary"):
        with st.spinner("Generating SAR ..."):
            from report_compiler import generate_sar_narrative, compile_pdf
            txns = system["df"][mask]
            sc = system["scores"][mask.values]
            ad = dict(account_id=aid, n_transactions=int(mask.sum()),
                       total_amount=float(txns["Amount"].sum()),
                       max_score=float(sc.max()), mean_score=float(sc.mean()))
            ppr = system["graph"].personalized_pagerank([aid])
            gd = {"top_connections": sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:5]}
            idx = np.where(mask.values)[0]
            ti = int(idx[np.argmax(sc)])
            sh = system["explainer"].explain(system["df"], ti)

            narr = generate_sar_narrative(aid, ad, gd, sh)
            st.subheader("Narrative")
            st.write(narr)

            pdf = compile_pdf(narr, aid, txns.head(20), sc.tolist())
            with open(pdf, "rb") as f:
                st.download_button(":material/download: Download PDF", f, f"SAR_{aid}.pdf", "application/pdf", type="primary")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    f = render_sidebar(best_threshold=0.65)

    with st.status(":material/settings: Initializing Valkyrie Engine ...", expanded=False) as s:
        s.update(label=f":material/database: Loading {f['nrows']:,} rows & training model ...")
        sys_data = init_system(f["nrows"])
        s.update(label=":material/check_circle: System ready", state="complete")

    f["threshold"] = st.session_state.get("active_threshold", f["threshold"])

    ti, tn, tp, ts = st.tabs([":material/search: Investigation", ":material/hub: Network Graph",
                                ":material/bar_chart: Model Performance", ":material/description: SAR Report"])
    with ti:
        render_investigation(sys_data, f["query"], f["threshold"])
    with tn:
        render_network(sys_data, focus=f["account_id"])
    with tp:
        render_performance(sys_data, active_threshold=f["threshold"])
    with ts:
        render_sar(sys_data, f["account_id"])


if __name__ == "__main__":
    main()
