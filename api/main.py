"""
Valkyrie-AML — FastAPI backend server.
Serves all ML/agent endpoints + SSE streaming for real-time investigation.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.startup import get_system, risk_label, escalation_action, GROK_API_KEY
from api.models import InvestigateRequest, SARRequest

app = FastAPI(
    title="Valkyrie-AML API",
    description="Anti-Money Laundering Compliance Investigation Agent",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Country coordinates lookup for globe
# -------------------------------------------------------------------------
COUNTRY_COORDS = {
    "US": (38.9, -77.0), "GB": (51.5, -0.1), "DE": (52.5, 13.4), "FR": (48.9, 2.3),
    "JP": (35.7, 139.7), "CN": (39.9, 116.4), "SG": (1.35, 103.8), "CH": (46.9, 7.4),
    "HK": (22.3, 114.2), "AU": (35.3, 149.1), "CA": (45.4, -75.7), "IN": (28.6, 77.2),
    "BR": (-15.8, -47.9), "MX": (19.4, -99.1), "RU": (55.8, 37.6), "ZA": (-25.7, 28.2),
    "AE": (24.5, 54.4), "NG": (9.1, 7.5), "KE": (-1.3, 36.8), "TH": (13.8, 100.5),
    "MY": (3.1, 101.7), "PH": (14.6, 121.0), "ID": (-6.2, 106.8), "VN": (21.0, 105.8),
    "KR": (37.6, 127.0), "AR": (-34.6, -58.4), "CL": (-33.5, -70.7), "CO": (4.7, -74.1),
    "PK": (33.7, 73.1), "EG": (30.1, 31.2), "SA": (24.7, 46.7), "IL": (31.8, 35.2),
    "NL": (52.4, 4.9), "IT": (41.9, 12.5), "ES": (40.4, -3.7), "PT": (38.7, -9.1),
    "SE": (59.3, 18.1), "NO": (59.9, 10.7), "DK": (55.7, 12.6), "FI": (60.2, 25.0),
    "PL": (52.2, 21.0), "CZ": (50.1, 14.4), "HU": (47.5, 19.1), "AT": (48.2, 16.4),
    "BE": (50.8, 4.4), "TR": (39.9, 32.9), "GR": (38.0, 23.7), "RO": (44.4, 26.1),
    "UA": (50.4, 30.5), "NZ": (-41.3, 174.8), "TW": (25.0, 121.6), "BD": (23.7, 90.4),
    "LK": (6.9, 79.9), "MM": (16.9, 96.2), "KH": (11.6, 104.9), "LA": (17.9, 102.6),
    "BN": (4.9, 114.9), "QA": (25.3, 51.5), "KW": (29.4, 48.0), "BH": (26.2, 50.6),
    "JO": (31.9, 35.9), "LB": (33.9, 35.5), "IQ": (33.3, 44.4), "MA": (34.0, -6.9),
    "TN": (36.8, 10.2), "DZ": (36.7, 3.2), "ET": (9.0, 38.7), "GH": (5.6, -0.2),
    "TZ": (-6.8, 39.3), "MZ": (-25.9, 32.6), "UG": (0.3, 32.6), "CM": (3.9, 11.5),
    "CI": (5.4, -4.0), "SN": (14.7, -17.4), "ZM": (-15.4, 28.3), "ZW": (-17.8, 31.1),
    "MU": (-20.2, 57.5), "NA": (-22.6, 17.1), "BW": (-24.7, 25.9), "AO": (-8.8, 13.2),
    "KZ": (51.2, 71.4), "UZ": (41.3, 69.3), "TM": (37.9, 58.4), "GE": (41.7, 44.8),
    "AM": (40.2, 44.5), "AZ": (40.4, 49.9), "MD": (47.0, 28.9), "BY": (53.9, 27.6),
    "LT": (54.7, 25.3), "LV": (57.0, 24.1), "EE": (59.4, 24.7), "SK": (48.1, 17.1),
    "SI": (46.1, 14.5), "HR": (45.8, 16.0), "RS": (44.8, 20.5), "BA": (43.9, 17.7),
    "MK": (42.0, 21.4), "AL": (41.3, 19.8), "BG": (42.7, 23.3), "CY": (35.2, 33.4),
    "MT": (35.9, 14.5), "IS": (64.1, -21.9), "LU": (49.6, 6.1), "MC": (43.7, 7.4),
    "LI": (47.1, 9.5), "SM": (43.9, 12.5), "AD": (42.5, 1.5), "VA": (41.9, 12.5),
    "MV": (4.2, 73.5), "BT": (27.5, 90.4), "NP": (27.7, 85.3), "AF": (34.5, 69.2),
    "IR": (35.7, 51.4), "SY": (33.5, 36.3), "YE": (15.4, 44.2), "OM": (23.6, 58.6),
}

def _get_country_coords(country_code: str):
    return COUNTRY_COORDS.get(country_code, (0.0, 0.0))


# -------------------------------------------------------------------------
# Background initialization
# -------------------------------------------------------------------------
_initialized = False

@app.on_event("startup")
async def startup_event():
    global _initialized
    print("[API] Starting Valkyrie-AML system initialization...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_system)
    _initialized = True
    print("[API] System ready.")


# -------------------------------------------------------------------------
# Health
# -------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    sys_data = get_system()
    o = sys_data["eval_results"]["overall"]
    return {
        "status": "ok",
        "initialized": _initialized,
        "nrows": sys_data["nrows"],
        "f1_score": round(o["f1"], 4),
        "optimal_threshold": round(sys_data["best_threshold"], 4),
        "graph_nodes": sys_data["graph"].stats["n_nodes"],
        "graph_edges": sys_data["graph"].stats["n_edges"],
    }


# -------------------------------------------------------------------------
# Dashboard stats
# -------------------------------------------------------------------------

@app.get("/api/dashboard-stats")
async def dashboard_stats():
    sys_data = get_system()
    df = sys_data["df"]
    scores = sys_data["scores"]
    flags = sys_data["flags"]
    o = sys_data["eval_results"]["overall"]
    best_t = sys_data["best_threshold"]

    flagged = int((scores >= best_t).sum())

    # Laundering typology distribution
    typology_counts = {}
    if "Laundering_type" in df.columns:
        vc = df[df["Is_laundering"] == 1]["Laundering_type"].value_counts() if "Is_laundering" in df.columns else df["Laundering_type"].value_counts()
        typology_counts = {k: int(v) for k, v in vc.head(10).items()}

    # Top risk countries (by sender bank location)
    if "Sender_bank_location" in df.columns:
        df_tmp = df.copy()
        df_tmp["_score"] = scores
        country_risk = df_tmp.groupby("Sender_bank_location")["_score"].agg(["max", "mean", "count"]).reset_index()
        country_risk.columns = ["country", "max_risk", "avg_risk", "count"]
        country_risk = country_risk.sort_values("max_risk", ascending=False).head(20)
        top_countries = country_risk.to_dict(orient="records")
    else:
        top_countries = []

    return {
        "total_transactions": int(len(df)),
        "flagged_transactions": flagged,
        "f1_score": round(o["f1"], 4),
        "precision": round(o["precision"], 4),
        "recall": round(o["recall"], 4),
        "false_positives": int(o["confusion_matrix"][0][1]),
        "optimal_threshold": round(best_t, 4),
        "laundering_typologies": typology_counts,
        "avg_anomaly_score": round(float(scores.mean()), 4),
        "top_risk_countries": top_countries,
        "suspicious_count": int(df["Is_laundering"].sum()) if "Is_laundering" in df.columns else 0,
    }


# -------------------------------------------------------------------------
# Top anomalies
# -------------------------------------------------------------------------

@app.get("/api/top-anomalies")
async def top_anomalies(n: int = 20, threshold: float = 0.0):
    sys_data = get_system()
    df = sys_data["df"]
    scores = sys_data["scores"]

    top_idx = np.argsort(scores)[-n:][::-1]
    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        score = float(scores[idx])
        if score < threshold:
            continue
        results.append({
            "idx": int(idx),
            "score": round(score, 4),
            "risk_level": risk_label(score),
            "escalation": escalation_action(score),
            "sender": str(row["Sender_account"]),
            "receiver": str(row["Receiver_account"]),
            "amount": round(float(row["Amount"]), 2),
            "payment_type": str(row.get("Payment_type", "")),
            "laundering_type": str(row.get("Laundering_type", "Unknown")),
            "date": str(row.get("Date", "")),
            "sender_location": str(row.get("Sender_bank_location", "")),
            "receiver_location": str(row.get("Receiver_bank_location", "")),
            "payment_currency": str(row.get("Payment_currency", "")),
            "received_currency": str(row.get("Received_currency", "")),
            "is_cross_currency": str(row.get("Payment_currency", "")) != str(row.get("Received_currency", "")),
        })

    return {"anomalies": results, "total": len(results)}


# -------------------------------------------------------------------------
# Account profile
# -------------------------------------------------------------------------

@app.get("/api/account/{account_id}")
async def account_profile(account_id: str):
    sys_data = get_system()
    df = sys_data["df"]
    scores = sys_data["scores"]
    explainer = sys_data["explainer"]

    mask = (df["Sender_account"].astype(str) == account_id) | (df["Receiver_account"].astype(str) == account_id)
    if mask.sum() == 0:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    acc_scores = scores[mask.values]
    acc_df = df[mask]

    max_score = float(acc_scores.max())
    mean_score = float(acc_scores.mean())

    # SHAP explanation for highest-scoring transaction
    sender_mask = df["Sender_account"].astype(str) == account_id
    sender_indices = np.where(sender_mask.values)[0]
    explanation = {}
    if len(sender_indices) > 0:
        top_txn_idx = int(sender_indices[np.argmax(scores[sender_indices])])
        try:
            explanation = explainer.explain(df, top_txn_idx)
        except Exception:
            pass

    # PPR risk
    try:
        ppr = sys_data["graph"].personalized_pagerank([account_id])
        ppr_score = float(ppr.get(account_id, 0.0))
        top_connections = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        ppr_score = 0.0
        top_connections = []

    return {
        "account_id": account_id,
        "n_transactions": int(mask.sum()),
        "total_amount": round(float(acc_df["Amount"].sum()), 2),
        "max_score": round(max_score, 4),
        "mean_score": round(mean_score, 4),
        "risk_level": risk_label(max_score),
        "escalation": escalation_action(max_score),
        "ppr_score": round(ppr_score, 6),
        "flagged_transactions": int((acc_scores >= 0.5).sum()),
        "explanation": explanation,
        "top_connections": [{"account": a, "ppr_score": round(s, 6)} for a, s in top_connections],
        "recent_transactions": acc_df.head(20).to_dict(orient="records"),
    }


# -------------------------------------------------------------------------
# Network data for Three.js
# -------------------------------------------------------------------------

@app.get("/api/network-data")
async def network_data(max_nodes: int = 60):
    sys_data = get_system()
    df = sys_data["df"]
    scores = sys_data["scores"]
    graph = sys_data["graph"]

    # Build risk score per account (sender + receiver)
    df_tmp = df.copy()
    df_tmp["_score"] = scores
    sender_risk = df_tmp.groupby("Sender_account")["_score"].max().to_dict()
    receiver_risk = df_tmp.groupby("Receiver_account")["_score"].max().to_dict()
    all_risk: dict = {}
    for acc, s in sender_risk.items():
        all_risk[str(acc)] = max(all_risk.get(str(acc), 0.0), float(s))
    for acc, s in receiver_risk.items():
        all_risk[str(acc)] = max(all_risk.get(str(acc), 0.0), float(s))

    # PPR scores from top seeds
    try:
        risk_seeds = sys_data.get("risk_seeds", {})
        seed_list = list(risk_seeds.keys())[:10] if isinstance(risk_seeds, dict) else list(risk_seeds)[:10]
        ppr = graph.personalized_pagerank(seed_list)
    except Exception:
        ppr = {}

    # Pick top accounts by risk score
    capped = min(max_nodes, 60)  # hard cap at 60 nodes for readability
    sorted_accs = sorted(all_risk.items(), key=lambda x: x[1], reverse=True)[:capped]

    # --- Build edges FIRST to find connected accounts ---
    all_accounts = set(a for a, _ in sorted_accs)
    edges_raw: dict = {}
    for _, row in df.iterrows():
        s = str(row["Sender_account"])
        r = str(row["Receiver_account"])
        if s in all_accounts and r in all_accounts and s != r:
            key = f"{s}|{r}"
            if key not in edges_raw:
                edges_raw[key] = {"source": s, "target": r, "amount": 0.0, "count": 0}
            edges_raw[key]["amount"] += float(row["Amount"])
            edges_raw[key]["count"] += 1

    # Keep only top-80 edges by amount so graph isn't tangled
    sorted_edges = sorted(edges_raw.values(), key=lambda e: e["amount"], reverse=True)[:80]

    # Only include nodes that appear in at least one edge
    connected = set()
    for e in sorted_edges:
        connected.add(e["source"])
        connected.add(e["target"])

    def _color(score: float) -> str:
        if score >= 0.8: return "#E63946"
        elif score >= 0.6: return "#F97316"
        elif score >= 0.4: return "#EAB308"
        return "#2EC04A"

    max_ppr = max(ppr.values()) if ppr else 1.0
    nodes = []
    for acc, score in sorted_accs:
        if acc not in connected:
            continue  # skip isolated nodes
        ppr_val = float(ppr.get(acc, 0.0))
        nodes.append({
            "id": acc,
            "risk_score": round(score, 4),
            "pagerank": round(ppr_val / max_ppr if max_ppr > 0 else 0, 4),
            "color": _color(score),
            "size": 4 + 16 * (ppr_val / max_ppr if max_ppr > 0 else 0),
            "label": acc,
        })

    edges = [{
        "source": e["source"],
        "target": e["target"],
        "amount": round(e["amount"], 2),
        "count": e["count"],
    } for e in sorted_edges]

    return {"nodes": nodes, "edges": edges}


# -------------------------------------------------------------------------
# Geo data for Three.js globe
# -------------------------------------------------------------------------

@app.get("/api/geo-data")
async def geo_data():
    sys_data = get_system()
    df = sys_data["df"]
    scores = sys_data["scores"]

    if "Sender_bank_location" not in df.columns or "Receiver_bank_location" not in df.columns:
        return {"arcs": []}

    df_tmp = df.copy()
    df_tmp["_score"] = scores

    # Aggregate by sender/receiver country pair
    group = df_tmp.groupby(["Sender_bank_location", "Receiver_bank_location"]).agg(
        total_amount=("Amount", "sum"),
        count=("Amount", "count"),
        max_risk=("_score", "max"),
    ).reset_index()

    group = group[group["Sender_bank_location"] != group["Receiver_bank_location"]]
    group = group.sort_values("max_risk", ascending=False).head(100)

    def _arc_color(score: float) -> str:
        if score >= 0.8:
            return "#E63946"
        elif score >= 0.6:
            return "#F97316"
        elif score >= 0.4:
            return "#EAB308"
        return "#5BC0EB"

    arcs = []
    for _, row in group.iterrows():
        from_country = str(row["Sender_bank_location"])
        to_country = str(row["Receiver_bank_location"])
        from_lat, from_lng = _get_country_coords(from_country)
        to_lat, to_lng = _get_country_coords(to_country)
        if from_lat == 0 and to_lat == 0:
            continue
        arcs.append({
            "from_country": from_country,
            "to_country": to_country,
            "from_lat": from_lat,
            "from_lng": from_lng,
            "to_lat": to_lat,
            "to_lng": to_lng,
            "total_amount": round(float(row["total_amount"]), 2),
            "count": int(row["count"]),
            "max_risk": round(float(row["max_risk"]), 4),
            "color": _arc_color(float(row["max_risk"])),
        })

    return {"arcs": arcs}


# -------------------------------------------------------------------------
# Model performance
# -------------------------------------------------------------------------

@app.get("/api/model-performance")
async def model_performance(threshold: float = -1.0):
    sys_data = get_system()
    from src.evaluation import evaluate_detector

    active_threshold = threshold if threshold >= 0 else sys_data["best_threshold"]
    ev = evaluate_detector(sys_data["df_labeled"], sys_data["scores"], active_threshold)
    o = ev["overall"]

    return {
        "threshold": active_threshold,
        "optimal_threshold": sys_data["best_threshold"],
        "precision": round(o["precision"], 4),
        "recall": round(o["recall"], 4),
        "f1": round(o["f1"], 4),
        "false_positives": int(o["confusion_matrix"][0][1]),
        "false_negatives": int(o["confusion_matrix"][1][0]),
        "true_positives": int(o["confusion_matrix"][1][1]),
        "true_negatives": int(o["confusion_matrix"][0][0]),
        "n_actual_suspicious": int(o["n_actual_suspicious"]),
        "n_predicted_suspicious": int(o["n_predicted_suspicious"]),
        "confusion_matrix": o["confusion_matrix"],
        "by_typology": {
            k: {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1": round(v["f1"], 4),
                "support": v["support"],
            }
            for k, v in ev.get("by_typology", {}).items()
        },
        "illustrative_cases": sys_data["illustrative_cases"].head(10).to_dict(orient="records"),
    }


# -------------------------------------------------------------------------
# SSE: Investigation streaming
# -------------------------------------------------------------------------

async def _stream_investigation(query: str, threshold: float, account_id: str | None) -> AsyncGenerator[str, None]:
    """Stream LangGraph investigation events as SSE."""

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield _sse("start", {"message": "Valkyrie investigation started", "query": query})
    await asyncio.sleep(0.1)

    sys_data = get_system()

    # --- Step 1: Parse intent ---
    yield _sse("intent", {"message": "Parsing query intent...", "step": 1})
    await asyncio.sleep(0.1)

    intent = _detect_intent(query)
    yield _sse("intent_detected", {
        "intent": intent["type"],
        "filters": intent["filters"],
        "pattern": intent["pattern"],
        "tools": intent["tools"],
        "step": 1,
    })
    await asyncio.sleep(0.2)

    # --- Step 2: Build execution plan via Grok ---
    yield _sse("planning", {"message": "Building dynamic execution plan...", "step": 2})
    await asyncio.sleep(0.1)

    plan = await asyncio.get_event_loop().run_in_executor(None, _build_plan_with_grok, query)
    yield _sse("plan_ready", {"plan": plan, "step": 2})
    await asyncio.sleep(0.2)

    # --- Step 3: Execute tools ---
    from src.orchestrator import ValkyrieToolExecutor
    executor = ValkyrieToolExecutor(
        df=sys_data["df_labeled"],
        detector=sys_data["detector"],
        explainer=sys_data["explainer"],
        graph=sys_data["graph"],
        anomaly_scores=sys_data["scores"],
        binary_flags=sys_data["flags"],
    )

    results = {}
    tools_to_run = plan.get("tools", [])
    total_tools = len(tools_to_run)

    for i, tool_entry in enumerate(tools_to_run):
        tool_name = tool_entry.get("name", "")
        params = tool_entry.get("params", {})

        yield _sse("tool_start", {
            "tool": tool_name,
            "params": params,
            "progress": i,
            "total": total_tools,
            "step": 3,
        })
        await asyncio.sleep(0.05)

        result = await asyncio.get_event_loop().run_in_executor(
            None, executor.dispatch, tool_name, params
        )
        results[tool_name] = result

        # Serialize result (make sure it's JSON-safe)
        safe_result = _make_json_safe(result)
        yield _sse("tool_done", {
            "tool": tool_name,
            "result": safe_result,
            "progress": i + 1,
            "total": total_tools,
            "step": 3,
        })
        await asyncio.sleep(0.1)

    # --- Step 4: Classify and explain ---
    yield _sse("classifying", {"message": "Classifying risk levels...", "step": 4})
    await asyncio.sleep(0.1)

    # Build classified results
    classified = _classify_results(results, sys_data)
    yield _sse("classified", {"classified": classified, "step": 4})
    await asyncio.sleep(0.1)

    # --- Step 5: Generate summary via Grok ---
    yield _sse("summarizing", {"message": "Generating compliance summary...", "step": 5})
    await asyncio.sleep(0.1)

    summary = await asyncio.get_event_loop().run_in_executor(
        None, _generate_summary_grok, query, results, classified
    )

    yield _sse("complete", {
        "summary": summary,
        "plan": plan,
        "results": _make_json_safe(results),
        "classified": classified,
        "query": query,
        "step": 5,
    })


def _detect_intent(query: str) -> dict:
    """Fast local intent detection for routing."""
    q = query.lower()

    # Filters
    filters = {}
    import re
    if "30 day" in q or "last month" in q:
        filters["date_range"] = "last_30_days"
    if "7 day" in q or "last week" in q:
        filters["date_range"] = "last_7_days"

    # Amount filters
    m = re.search(r"\$?(\d[\d,]*)\s*(?:or more|and above|\+|>)", q)
    if m:
        filters["min_amount"] = int(m.group(1).replace(",", ""))

    m2 = re.search(r"under\s+\$?(\d[\d,]*)", q)
    if m2:
        filters["max_amount"] = int(m2.group(1).replace(",", ""))

    # Account ID
    m3 = re.search(r"(?:account|customer|id)\s+#?(\d{6,})", q)
    if m3:
        filters["account_id"] = m3.group(1)

    # Top-N
    m4 = re.search(r"top\s+(\d+)", q)
    if m4:
        filters["top_n"] = int(m4.group(1))

    # Count filter
    m5 = re.search(r"(\d+)\+?\s+transactions", q)
    if m5:
        filters["min_transactions"] = int(m5.group(1))

    # Pattern
    pattern = "general"
    if any(w in q for w in ["structur", "smurfing", "smurf"]):
        pattern = "structuring_smurfing"
    elif any(w in q for w in ["layer", "chain", "flow"]):
        pattern = "layering"
    elif any(w in q for w in ["network", "graph", "connect"]):
        pattern = "network_risk"
    elif any(w in q for w in ["sar", "report", "suspicious activity"]):
        pattern = "sar_generation"
    elif any(w in q for w in ["model", "performance", "precision", "recall", "f1", "metric"]):
        pattern = "evaluation"
    elif re.search(r"account|customer|id", q) and re.search(r"\d{6,}", q):
        pattern = "single_entity"

    # Tool selection based on pattern
    tools_map = {
        "structuring_smurfing": ["search_transactions", "get_anomaly_scores", "get_shap_explanation"],
        "layering": ["trace_fund_flows", "compute_network_risk", "get_anomaly_scores"],
        "network_risk": ["compute_network_risk", "trace_fund_flows"],
        "sar_generation": ["generate_sar", "get_anomaly_scores"],
        "evaluation": ["evaluate_model"],
        "single_entity": ["get_anomaly_scores", "get_shap_explanation"],
        "general": ["get_anomaly_scores", "get_shap_explanation"],
    }
    tools = tools_map.get(pattern, tools_map["general"])

    # Determine type
    if pattern == "single_entity":
        intent_type = "single_entity_lookup"
    elif pattern == "evaluation":
        intent_type = "model_evaluation"
    elif pattern in ("structuring_smurfing",):
        intent_type = "pattern_detection"
    else:
        intent_type = "investigation"

    return {
        "type": intent_type,
        "pattern": pattern,
        "filters": filters,
        "tools": tools,
    }


def _build_plan_with_grok(query: str) -> dict:
    """Call Grok API to build execution plan. Falls back to local intent detection."""
    from src.orchestrator import TOOLS, _system_prompt
    import urllib.request

    api_key = GROK_API_KEY
    if not api_key:
        return _fallback_plan(query)

    tools_json = json.dumps(TOOLS, indent=2)
    user_message = (
        f"User query: {query}\n\nAvailable tools:\n{tools_json}\n\n"
        'Respond with ONLY a JSON object: {"tools": [{"name": "...", "params": {...}}]}'
    )
    full_prompt = f"{_system_prompt}\n\nUser Message:\n{user_message}"

    try:
        payload = {
            "model": "grok-3-mini",
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0,
            "max_tokens": 1024,
        }
        req = urllib.request.Request(
            "https://api.x.ai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()

            import re
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
            parsed = json.loads(cleaned)
            if "tools" in parsed:
                return parsed
    except Exception as e:
        print(f"[Grok] Plan generation failed: {e}")

    return _fallback_plan(query)


def _fallback_plan(query: str) -> dict:
    """Keyword-based fallback plan generation."""
    intent = _detect_intent(query)
    filters = intent["filters"]

    tools = []
    for tool_name in intent["tools"]:
        params: dict = {}
        if tool_name == "search_transactions":
            if "account_id" in filters:
                params["account"] = filters["account_id"]
            if "date_range" in filters:
                import datetime
                if filters["date_range"] == "last_30_days":
                    d = datetime.date.today() - datetime.timedelta(days=30)
                    params["date_start"] = d.isoformat()
                elif filters["date_range"] == "last_7_days":
                    d = datetime.date.today() - datetime.timedelta(days=7)
                    params["date_start"] = d.isoformat()
            if "max_amount" in filters:
                params["max_amount"] = filters["max_amount"]
        elif tool_name == "get_anomaly_scores":
            if "account_id" in filters:
                params["account_id"] = filters["account_id"]
            params["top_n"] = filters.get("top_n", 10)
        elif tool_name == "get_shap_explanation":
            pass
        elif tool_name == "trace_fund_flows":
            if "account_id" in filters:
                params["source_account"] = filters["account_id"]
            params["max_hops"] = 4
        elif tool_name == "compute_network_risk":
            pass
        elif tool_name == "evaluate_model":
            pass
        elif tool_name == "generate_sar":
            if "account_id" in filters:
                params["account_id"] = filters["account_id"]
        tools.append({"name": tool_name, "params": params})

    return {"tools": tools}


def _classify_results(results: dict, sys_data: dict) -> dict:
    """Classify and enrich tool results with risk levels and escalation actions."""
    classified = {}

    for tool_name, result in results.items():
        if not isinstance(result, dict):
            classified[tool_name] = result
            continue

        if "top_anomalies" in result:
            enriched = []
            for item in result["top_anomalies"]:
                score = item.get("score", 0)
                item["risk_level"] = risk_label(score)
                item["escalation"] = escalation_action(score)
                item["explanation"] = _quick_explanation(score, item)
                enriched.append(item)
            classified[tool_name] = {**result, "top_anomalies": enriched}

        elif "account_id" in result and "max_score" in result:
            score = result.get("max_score", 0)
            classified[tool_name] = {
                **result,
                "risk_level": risk_label(score),
                "escalation": escalation_action(score),
            }
        else:
            classified[tool_name] = result

    return classified


def _quick_explanation(score: float, item: dict) -> str:
    reasons = []
    if item.get("is_cross_currency"):
        reasons.append("cross-currency transaction")
    if item.get("amount", 0) >= 50_000:
        reasons.append(f"large amount (${item['amount']:,.0f})")
    ltype = item.get("type", "") or item.get("laundering_type", "")
    if ltype and ltype not in ("Unknown", "nan", "None"):
        reasons.append(f"{ltype} pattern detected")
    if not reasons:
        reasons.append("anomalous transaction profile")
    return f"Risk score {score:.3f}: " + ", ".join(reasons)


def _generate_summary_grok(query: str, results: dict, classified: dict) -> str:
    """Generate compliance summary via Grok or fallback."""
    import urllib.request

    api_key = GROK_API_KEY
    summary_prompt = (
        "You are Valkyrie, an AML compliance agent. "
        "Summarize the following investigation results for a compliance officer. "
        "Highlight key findings, risk levels, and recommended next steps. "
        "Be concise (2-4 sentences). Format as plain text."
    )

    # Build context
    parts = []
    for tool_name, result in results.items():
        if isinstance(result, dict):
            if "top_anomalies" in result:
                top = result["top_anomalies"]
                total_amt = result.get("total_amount", sum(t.get("amount", 0) for t in top))
                if top:
                    parts.append(
                        f"Detected {len(top)} high-risk transactions, "
                        f"highest score: {top[0].get('score', 0):.3f}, "
                        f"total flagged amount: ${total_amt:,.0f}"
                    )
            elif "account_id" in result and "n_transactions" in result:
                total_amt = result.get("total_amount", 0)
                parts.append(
                    f"Account {result['account_id']}: {result['n_transactions']} transactions "
                    f"totaling ${total_amt:,.0f}, max risk score: {result.get('max_score', 0):.3f}, "
                    f"{result.get('flagged', 0)} flagged above threshold"
                )
            elif "n_transactions" in result:
                total_amt = result.get("total_amount", 0)
                parts.append(
                    f"Found {result['n_transactions']} matching transactions totaling ${total_amt:,.0f}"
                )
            elif "n_chains" in result:
                parts.append(f"Found {result['n_chains']} layering chains from source account")
            elif "n_scored" in result:
                parts.append(f"Network risk scored {result['n_scored']} accounts via PageRank")
            elif "narrative" in result:
                parts.append(f"SAR generated for account {result.get('account_id', '')}")

    context = "; ".join(parts) if parts else "No significant findings."

    if not api_key:
        return f"Investigation complete. Query: '{query}'. Findings: {context}. Review flagged transactions and take appropriate action."

    try:
        payload = {
            "model": "grok-3-mini",
            "messages": [{
                "role": "user",
                "content": f"{summary_prompt}\n\nQuery: {query}\nFindings: {context}\n\nSummary:"
            }],
            "temperature": 0.3,
            "max_tokens": 256,
        }
        req = urllib.request.Request(
            "https://api.x.ai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Grok] Summary generation failed: {e}")
        return f"Investigation complete. {context}. Recommend reviewing flagged transactions for compliance action."


def _make_json_safe(obj):
    """Recursively make object JSON serializable."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


@app.post("/api/investigate")
async def investigate(request: InvestigateRequest):
    """SSE streaming investigation endpoint."""
    return StreamingResponse(
        _stream_investigation(request.query, request.threshold, request.account_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# -------------------------------------------------------------------------
# SAR generation
# -------------------------------------------------------------------------

@app.post("/api/generate-sar")
async def generate_sar(request: SARRequest):
    sys_data = get_system()
    df = sys_data["df"]
    scores = sys_data["scores"]
    account_id = request.account_id

    mask = df["Sender_account"].astype(str) == account_id
    if mask.sum() == 0:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    txns = df[mask]
    acc_scores = scores[mask.values]

    anomaly_data = {
        "account_id": account_id,
        "n_transactions": int(mask.sum()),
        "total_amount": round(float(txns["Amount"].sum()), 2),
        "mean_score": round(float(acc_scores.mean()), 4),
        "max_score": round(float(acc_scores.max()), 4),
    }

    try:
        ppr = sys_data["graph"].personalized_pagerank([account_id])
        graph_data = {
            "top_connections": sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    except Exception:
        graph_data = {"top_connections": []}

    # SHAP explanation
    indices = np.where(mask.values)[0]
    shap_explanation = {}
    if len(indices) > 0:
        top_txn_idx = int(indices[np.argmax(acc_scores)])
        try:
            shap_explanation = sys_data["explainer"].explain(df, top_txn_idx)
        except Exception:
            pass

    try:
        from src.report_compiler import generate_sar_narrative
        narrative = generate_sar_narrative(account_id, anomaly_data, graph_data, shap_explanation)
    except Exception as e:
        narrative = f"SAR for account {account_id}. {anomaly_data['n_transactions']} transactions totaling ${anomaly_data['total_amount']:,.2f}. Max anomaly score: {anomaly_data['max_score']:.4f}. Risk level: {risk_label(anomaly_data['max_score'])}."

    return {
        "account_id": account_id,
        "narrative": narrative,
        "anomaly_data": anomaly_data,
        "risk_level": risk_label(anomaly_data["max_score"]),
        "escalation": escalation_action(anomaly_data["max_score"]),
        "shap_explanation": _make_json_safe(shap_explanation),
        "top_connections": [{"account": a, "ppr_score": round(s, 6)} for a, s in graph_data["top_connections"]],
        "transactions": txns.head(20).to_dict(orient="records"),
    }


# -------------------------------------------------------------------------
# Structuring detection (special endpoint)
# -------------------------------------------------------------------------

@app.get("/api/structuring")
async def detect_structuring(
    max_amount: float = 10000.0,
    min_transactions: int = 3,
    days: int = 30,
    top_n: int = 20,
):
    """Detect structuring: multiple sub-threshold transactions from same sender."""
    sys_data = get_system()
    df = sys_data["df"]
    scores = sys_data["scores"]

    df_tmp = df.copy()
    df_tmp["_score"] = scores

    # Filter under threshold amount
    under = df_tmp[df_tmp["Amount"] < max_amount]

    # Count per sender
    sender_counts = under.groupby("Sender_account").agg(
        count=("Amount", "count"),
        total=("Amount", "sum"),
        max_score=("_score", "max"),
    ).reset_index()
    sender_counts = sender_counts[sender_counts["count"] >= min_transactions]
    sender_counts = sender_counts.sort_values("max_score", ascending=False).head(top_n)

    results = []
    for _, row in sender_counts.iterrows():
        results.append({
            "sender": str(row["Sender_account"]),
            "transaction_count": int(row["count"]),
            "total_amount": round(float(row["total"]), 2),
            "max_score": round(float(row["max_score"]), 4),
            "risk_level": risk_label(float(row["max_score"])),
            "escalation": escalation_action(float(row["max_score"])),
            "pattern": "structuring",
        })

    return {"structuring_suspects": results, "threshold": max_amount, "min_transactions": min_transactions}
