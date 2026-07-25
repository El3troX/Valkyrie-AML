"""
LLM-driven investigation orchestrator for Valkyrie-AML.

Uses LangGraph to coordinate an LLM-based planning loop with local tool
execution. All LLM calls go through Ollama (Gemma4:e4b) — no external API
keys required.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# LangGraph + LLM imports
# ---------------------------------------------------------------------------

try:
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, StateGraph
    from langgraph.graph.state import CompiledStateGraph
    from typing_extensions import TypedDict
except ImportError:
    ChatOllama = None  # type: ignore
    StateGraph = None
    END = "END"
    from typing import TypedDict

# ---------------------------------------------------------------------------
# Tool definitions exposed to the LLM
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_transactions",
        "description": "Search transactions by account, date range, or amount range.",
        "parameters": {
            "type": "object",
            "properties": {
                "account": {"type": "string", "description": "Account ID to search for (sender or receiver)."},
                "date_start": {"type": "string", "description": "Start date YYYY-MM-DD."},
                "date_end": {"type": "string", "description": "End date YYYY-MM-DD."},
                "min_amount": {"type": "number", "description": "Minimum transaction amount."},
                "max_amount": {"type": "number", "description": "Maximum transaction amount."},
            },
        },
    },
    {
        "name": "get_anomaly_scores",
        "description": "Get ML anomaly scores. Without account_id, returns top-N most anomalous.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Optional account ID for specific scoring."},
                "top_n": {"type": "integer", "description": "Number of top anomalies to return."},
            },
        },
    },
    {
        "name": "get_shap_explanation",
        "description": "Explain why a specific transaction was flagged as anomalous.",
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_idx": {"type": "integer", "description": "Row index of the transaction in the dataset."},
            },
        },
    },
    {
        "name": "trace_fund_flows",
        "description": "Trace multi-hop fund flows from an account (for layering detection).",
        "parameters": {
            "type": "object",
            "properties": {
                "source_account": {"type": "string", "description": "Account ID to trace from."},
                "max_hops": {"type": "integer", "description": "Maximum number of hops."},
            },
        },
    },
    {
        "name": "compute_network_risk",
        "description": "Compute Personalised PageRank risk propagation from seed accounts.",
        "parameters": {
            "type": "object",
            "properties": {
                "seed_accounts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of high-risk seed account IDs.",
                },
            },
        },
    },
    {
        "name": "evaluate_model",
        "description": "Get model evaluation metrics (precision, recall, F1).",
        "parameters": {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "description": "Score threshold for binary classification."},
            },
        },
    },
    {
        "name": "generate_sar",
        "description": "Generate a Suspicious Activity Report for an account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Account ID to generate SAR for."},
            },
        },
    },
]

# ---------------------------------------------------------------------------
# LangGraph state definition
# ---------------------------------------------------------------------------


class ValkyrieState(TypedDict):
    """State passed between LangGraph nodes."""

    user_query: str
    plan: Optional[dict]
    results: dict[str, Any]
    summary: str
    error: Optional[str]
    execution_log: list[dict]


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.2"

_system_prompt = (
    "You are Valkyrie, an AML (Anti-Money Laundering) compliance investigation agent. "
    "Given a user query and the available investigation tools, produce a structured "
    "execution plan as a JSON object with a 'tools' array. Each tool entry has 'name' "
    "and 'params' matching the tool's parameters. Do not include any text outside the "
    "pure JSON object."
)

_summary_prompt = (
    "You are Valkyrie, an AML compliance agent. Summarize the following investigation "
    "results for a compliance officer. Highlight key findings, risk levels, and "
    "recommended next steps. Keep it concise (2-4 sentences)."
)


try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path.home() / ".env")
except ImportError:
    pass


def _get_grok_api_key() -> str | None:
    return os.environ.get("GROK_API_KEY")


def _get_gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _call_llm(
    system: str,
    user_message: str,
    model: str = "grok-3-mini",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout_secs: int = 15,
) -> str:
    """Call Grok API with Gemini and Ollama fallbacks."""
    grok_key = _get_grok_api_key()
    if grok_key:
        try:
            import json, urllib.request
            url = "https://api.x.ai/v1/chat/completions"
            payload = {
                "model": "grok-3-mini",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message}
                ],
                "temperature": temperature,
                "max_tokens": 1024 if "JSON" in user_message else 2048,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {grok_key}"
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"].strip()
                if text:
                    print(f"[LLM] Grok API response received ({len(text)} chars)")
                    return text
        except Exception as e:
            print(f"[LLM] Grok API call failed: {e}. Trying Gemini...")

    api_key = _get_gemini_api_key()
    if api_key:
        # Attempt 1: google-genai SDK (import google.genai as genai)
        try:
            import google.genai as genai
            from google.genai import types
            print(f"[LLM] Calling Gemini API (gemini-2.5-flash) via SDK...")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system}\n\nUser Message:\n{user_message}",
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            text = response.text.strip() if response.text else ""
            if text:
                print(f"[LLM] Gemini SDK response received ({len(text)} chars)")
                return text
        except Exception as e:
            print(f"[LLM] Gemini SDK import/call failed ({e}), using REST fallback...")

        # Attempt 2: Direct zero-dependency HTTP REST request
        for m_name in ["gemini-2.5-flash", "gemini-1.5-flash"]:
            try:
                import json, urllib.request
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": f"{system}\n\nUser Message:\n{user_message}"}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "").strip()
                            if text:
                                print(f"[LLM] Gemini REST API response received ({m_name}, {len(text)} chars)")
                                return text
            except Exception as e:
                print(f"[LLM] Gemini REST call failed ({m_name}): {e}")

    if ChatOllama is None:
        return ""

    def _do_ollama_call() -> str:
        llm = ChatOllama(
            model=DEFAULT_MODEL,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=timeout_secs,
        )
        messages = [("system", system), ("human", user_message)]
        print(f"[LLM] >>> Sending request to Ollama ...")
        response = llm.invoke(messages)
        return response.content.strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_do_ollama_call)
        try:
            return fut.result(timeout=timeout_secs)
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# LangGraph node factories (closures capture the executor)
# ---------------------------------------------------------------------------


def _make_plan_query_node():
    """Create a plan_query node that calls Ollama (or keyword fallback)."""

    def _plan_query_node(state: ValkyrieState) -> dict:
        """Given a user query, produce an execution plan via Ollama."""
        query = state["user_query"]
        print(f"\n[DEBUG] === plan_query node ===")
        print(f"[DEBUG]     User query: \"{query}\"")
        tools_json = json.dumps(TOOLS, indent=2)
        user_message = (
            f"User query: {query}\n\nAvailable tools:\n{tools_json}\n\n"
            "Respond with ONLY a JSON object: {\"tools\": [{\"name\": \"...\", \"params\": {...}}]}"
        )

        print("[DEBUG] Calling Ollama for plan generation...")
        content = _call_llm(_system_prompt, user_message)
        plan = None
        if content:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "tools" in parsed:
                    plan = parsed
                    print(f"[DEBUG] LLM generated plan with {len(plan['tools'])} tool(s)")
            except (json.JSONDecodeError, ValueError):
                print("[DEBUG] LLM response was not valid JSON")

        if plan is None:
            print("[DEBUG] LLM failed to produce a valid plan – returning empty plan.")
            plan = {"tools": []}

        print(f"[DEBUG] Final plan: {json.dumps(plan)}")
        return {"plan": plan, "execution_log": [{"node": "plan_query", "plan": plan}]}

    return _plan_query_node


def _make_execute_plan_node(executor: "ValkyrieToolExecutor"):
    """Create an execute_plan node that dispatches tools via the executor."""

    def _execute_plan_node(state: ValkyrieState) -> dict:
        """Execute each tool in the plan, collecting results."""
        plan = state.get("plan", {"tools": []})
        tools = plan.get("tools", [])

        print(f"\n[DEBUG] === execute_plan node ===")
        print(f"[DEBUG] Executing {len(tools)} tool(s) in plan")

        results: dict[str, Any] = {}
        log_entries = list(state.get("execution_log", []))

        for i, entry in enumerate(tools):
            name = entry.get("name", "")
            params = entry.get("params", {})
            param_str = ", ".join(f"{k}={v}" for k, v in params.items() if v)
            print(f"[DEBUG]   [{i+1}/{len(tools)}] Dispatching: {name}({param_str})")
            result = executor.dispatch(name, params)
            if isinstance(result, dict):
                if "error" in result:
                    print(f"[DEBUG]     -> Error: {result['error']}")
                else:
                    print(f"[DEBUG]     -> OK ({len(list(result.keys()))} keys)")
            elif isinstance(result, list):
                print(f"[DEBUG]     -> {len(result)} items returned")
            else:
                print(f"[DEBUG]     -> {type(result).__name__}")
            results[name] = result
            log_entries.append({"node": "execute", "tool": name, "params": params})

        print(f"[DEBUG] All {len(tools)} tool(s) executed")
        return {"results": results, "execution_log": log_entries}

    return _execute_plan_node


def _make_summarize_node(model: str = DEFAULT_MODEL):
    """Create a summarize node that calls Ollama for a natural-language summary."""

    def _summarize_node(state: ValkyrieState) -> dict:
        """Call Ollama to produce a natural-language summary of results."""
        results = state.get("results", {})
        print(f"\n[DEBUG] === summarize node ===")
        if not results:
            print("[DEBUG] No results to summarize")
            return {"summary": "No results to summarize.", "execution_log": state.get("execution_log", []) + [{"node": "summarize"}]}

        # Build a compact representation of results
        parts = []
        for tool_name, result in results.items():
            if isinstance(result, dict):
                if "error" in result:
                    parts.append(f"[{tool_name}] Error: {result['error']}")
                elif "n_transactions" in result:
                    parts.append(f"[{tool_name}] Found {result['n_transactions']} transactions")
                elif "top_anomalies" in result:
                    parts.append(f"[{tool_name}] Top {len(result['top_anomalies'])} anomalies")
                elif "narrative" in result:
                    parts.append(f"[SAR] Generated for {result['account_id']}")
                elif "n_chains" in result:
                    parts.append(f"[Trace] Found {result['n_chains']} chains from {result.get('source_account', '?')}")
                elif "n_scored" in result:
                    parts.append(f"[Risk] Scored {result['n_scored']} accounts via PPR")
                else:
                    parts.append(f"[{tool_name}] {json.dumps(result)[:200]}")
            elif isinstance(result, list):
                parts.append(f"[{tool_name}] {len(result)} entries returned")
            else:
                parts.append(f"[{tool_name}] {result}")

        summary_text = "; ".join(parts)
        print(f"[DEBUG] Built summary context ({len(summary_text)} chars)")
        print("[DEBUG] Calling Ollama for summary generation...")
        user_message = f"Investigation results:\n{summary_text}\n\nProvide a concise summary."
        summary = _call_llm(_summary_prompt, user_message, model=model, temperature=0.3, max_tokens=512)

        if not summary:
            print("[DEBUG] Ollama summary failed, using text-based summary")
            summary = f"Investigation complete. Key findings: {summary_text}"
        else:
            print(f"[DEBUG] Ollama summary received ({len(summary)} chars)")

        return {"summary": summary, "execution_log": state.get("execution_log", []) + [{"node": "summarize"}]}

    return _summarize_node


def _should_continue(state: ValkyrieState) -> Literal["execute_plan", "summarize", END]:
    """Edge router: after plan_query, go to execute; after execute, go to summarize."""
    if state.get("plan") and "tools" in state["plan"]:
        return "execute_plan"
    return END


# ---------------------------------------------------------------------------
# Keyword fallback (plan generation when LLM fails)
# ---------------------------------------------------------------------------

# Keyword fallback removed – pure LLM planning only


# ---------------------------------------------------------------------------
# Tool executor (injected into the graph at compile time)
# ---------------------------------------------------------------------------


class ValkyrieToolExecutor:
    """Dispatches tool calls to the appropriate subsystem.

    This is injected into the compiled LangGraph so that execute_plan_node
    can call it without needing all subsystems passed through state.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        detector: Any,
        explainer: Any,
        graph: Any,
        anomaly_scores: np.ndarray,
        binary_flags: np.ndarray,
    ) -> None:
        self.df = df
        self.detector = detector
        self.explainer = explainer
        self.graph = graph
        self.anomaly_scores = anomaly_scores
        self.binary_flags = binary_flags

        # Build dispatch table
        self._dispatch_map = {
            "search_transactions": self._tool_search_transactions,
            "get_anomaly_scores": self._tool_anomaly_scores,
            "get_shap_explanation": self._tool_shap_explanation,
            "trace_fund_flows": self._tool_trace_fund_flows,
            "compute_network_risk": self._tool_network_risk,
            "evaluate_model": self._tool_evaluate,
            "generate_sar": self._tool_generate_sar,
            "get_illustrative_cases": self._tool_illustrative_cases,
        }

    def dispatch(self, name: str, params: dict) -> Any:
        handler = self._dispatch_map.get(name)
        if handler is None:
            print(f"[Executor] Unknown tool: {name}")
            return {"error": f"Unknown tool: {name}"}
        try:
            print(f"[Executor] Running {name}...")
            result = handler(params)
            print(f"[Executor] {name} completed")
            return result
        except Exception as e:
            print(f"[Executor] {name} failed: {e}")
            return {"error": str(e)}

    # ---- Tool implementations ----

    def _tool_search_transactions(self, params: dict) -> dict:
        print(f"  [Tool:search_transactions] params={params}")
        df = self.df.copy()
        account = params.get("account")
        if account:
            acc = str(account)
            mask = (df["Sender_account"].astype(str) == acc) | (df["Receiver_account"].astype(str) == acc)
            df = df[mask]
        ds = params.get("date_start")
        if ds:
            df = df[df["Date"] >= ds]
        de = params.get("date_end")
        if de:
            df = df[df["Date"] <= de]
        ma_min = params.get("min_amount")
        if ma_min is not None:
            df = df[df["Amount"] >= ma_min]
        ma_max = params.get("max_amount")
        if ma_max is not None:
            df = df[df["Amount"] <= ma_max]
        # Cap at 1000 for speed
        total = len(df)
        if total > 1000:
            df = df.head(1000)
        n = len(df)
        print(f"[Tool:search_transactions] Found {total} matching transactions (showing {n})")
        sample_cols = [c for c in ["Date", "Sender_account", "Receiver_account", "Amount", "Laundering_type", "Payment_type"] if c in df.columns]
        return {
            "n_transactions": total,
            "total_amount": float(df["Amount"].sum()),
            "sample": df.head(10)[sample_cols].to_dict(orient="records") if not df.empty else [],
        }

    def _tool_anomaly_scores(self, params: dict) -> dict:
        print(f"  [Tool:get_anomaly_scores] params={params}")
        top_n = params.get("top_n", 10)
        account_id = params.get("account_id", None)
        if account_id:
            acc = str(account_id)
            # Match both sender AND receiver rows for a full picture
            sender_mask = self.df["Sender_account"].astype(str) == acc
            receiver_mask = self.df["Receiver_account"].astype(str) == acc
            full_mask = sender_mask | receiver_mask
            indices = np.where(full_mask.values)[0]
            if len(indices) == 0:
                print(f"[Tool:get_anomaly_scores] Account {account_id} not found")
                return {"error": f"Account {account_id} not found", "score": None}
            scores = self.anomaly_scores[indices]
            acc_df = self.df[full_mask]
            total_amount = float(acc_df["Amount"].sum())
            max_amount = float(acc_df["Amount"].max())
            flagged_count = int((scores >= 0.5).sum())
            print(f"[Tool:get_anomaly_scores] Account {account_id}: {len(indices)} txns, total=${total_amount:,.0f}, max_score={scores.max():.4f}")
            return {
                "account_id": account_id,
                "n_transactions": len(indices),
                "total_amount": round(total_amount, 2),
                "max_transaction": round(max_amount, 2),
                "max_score": round(float(scores.max()), 4),
                "mean_score": round(float(scores.mean()), 4),
                "flagged": flagged_count,
            }
        top_idx = np.argsort(self.anomaly_scores)[-top_n:][::-1]
        results = []
        total_flagged_amount = 0.0
        for idx in top_idx:
            row = self.df.iloc[idx]
            amt = float(row["Amount"])
            total_flagged_amount += amt
            results.append({
                "idx": int(idx),
                "score": float(self.anomaly_scores[idx]),
                "sender": str(row["Sender_account"]),
                "receiver": str(row["Receiver_account"]),
                "amount": round(amt, 2),
                "type": str(row.get("Laundering_type", "Unknown")),
            })
        print(f"[Tool:get_anomaly_scores] Top {len(results)} by score (range {results[-1]['score']:.4f} - {results[0]['score']:.4f}), total=${total_flagged_amount:,.0f}")
        return {"top_anomalies": results, "total_amount": round(total_flagged_amount, 2)}

    def _tool_shap_explanation(self, params: dict) -> dict:
        print(f"  [Tool:get_shap_explanation] params={params}")
        idx = params.get("transaction_idx")
        if idx is None:
            # Default to the most anomalous transaction
            idx = int(np.argmax(self.anomaly_scores))
            print(f"[Tool:get_shap_explanation] No index given, using top anomaly at row {idx}")
        idx = int(idx)
        print(f"[Tool:get_shap_explanation] Explaining transaction at row index {idx}")
        result = self.explainer.explain(self.df, idx)
        top = result.get('top_features', [])
        print(f"[Tool:get_shap_explanation] Top feature: {top[0]['plain_language'][:60] if top else 'N/A'}")
        return result

    def _tool_trace_fund_flows(self, params: dict) -> dict:
        print(f"  [Tool:trace_fund_flows] params={params}")
        source = params.get("source_account")
        if not source:
            # Use top anomaly sender by default
            top_idx = int(np.argmax(self.anomaly_scores))
            source = str(self.df.iloc[top_idx]["Sender_account"])
            print(f"[Tool:trace_fund_flows] No source given, using top anomaly sender: {source}")
        max_hops = params.get("max_hops", 4)
        print(f"[Tool:trace_fund_flows] Tracing chains from {source} (max {max_hops} hops)")
        chains = self.graph.trace_chains(source, max_hops=max_hops)
        print(f"[Tool:trace_fund_flows] Found {len(chains)} chain(s)")
        return {
            "source_account": source,
            "n_chains": len(chains),
            "max_chain_length": len(chains[0]) if chains else 0,
            "chains": chains,
        }

    def _tool_network_risk(self, params: dict) -> dict:
        print(f"  [Tool:compute_network_risk] params={params}")
        seeds = params.get("seed_accounts", [])
        if not seeds:
            # Use risk seeds from the graph (top anomalous accounts)
            risk_seeds = self.graph.get_risk_seeds(self.anomaly_scores, self.df["Sender_account"], top_n=10)
            seeds = list(risk_seeds.keys())
            print(f"[Tool:compute_network_risk] No seeds given, using top {len(seeds)} anomalous accounts")
        print(f"[Tool:compute_network_risk] Running PPR with {len(seeds)} seed(s)")
        ppr = self.graph.personalized_pagerank(seeds)
        top = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:20]
        print(f"[Tool:compute_network_risk] Scored {len(ppr)} accounts, top PPR={top[0][1]:.6f if top else 0}")
        return {
            "seed_accounts": seeds[:5],
            "n_scored": len(ppr),
            "top_risk_accounts": [{"account": a, "ppr_score": round(s, 6)} for a, s in top],
        }

    def _tool_evaluate(self, params: dict) -> dict:
        print(f"  [Tool:evaluate_model] params={params}")
        threshold = params.get("threshold", 0.5)
        try:
            from evaluation import evaluate_detector, map_typology
        except ModuleNotFoundError:
            from src.evaluation import evaluate_detector, map_typology
        df_labeled = map_typology(self.df)
        result = evaluate_detector(df_labeled, self.anomaly_scores, threshold)
        print(f"[Tool:evaluate_model] F1={result['overall']['f1']:.4f} at threshold={threshold}")
        return result

    def _tool_generate_sar(self, params: dict) -> dict:
        print(f"  [Tool:generate_sar] params={params}")
        account_id = params.get("account_id")
        if not account_id:
            # Default to the most anomalous sender
            top_idx = int(np.argmax(self.anomaly_scores))
            account_id = str(self.df.iloc[top_idx]["Sender_account"])
            print(f"[Tool:generate_sar] No account given, using top anomaly sender: {account_id}")
        try:
            from report_compiler import generate_sar_narrative
        except ModuleNotFoundError:
            from src.report_compiler import generate_sar_narrative
        mask = self.df["Sender_account"].astype(str) == account_id
        txns = self.df[mask]
        scores = self.anomaly_scores[mask.values]
        anomaly_data = {
            "account_id": account_id,
            "n_transactions": len(txns),
            "total_amount": float(txns["Amount"].sum()),
            "mean_score": float(scores.mean()) if len(scores) > 0 else 0,
            "max_score": float(scores.max()) if len(scores) > 0 else 0,
        }
        ppr = self.graph.personalized_pagerank([account_id])
        graph_data = {"top_connections": sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:5]}
        shap_explanation = {}
        indices = np.where(mask.values)[0]
        if len(indices) > 0:
            top_txn = int(indices[np.argmax(self.anomaly_scores[indices])])
            print(f"[Tool:generate_sar] Getting SHAP explanation for top txn idx={top_txn}")
            shap_explanation = self.explainer.explain(self.df, top_txn)
        print(f"[Tool:generate_sar] Generating narrative for account {account_id}...")
        narrative = generate_sar_narrative(account_id, anomaly_data, graph_data, shap_explanation)
        print(f"[Tool:generate_sar] Narrative generated ({len(narrative)} chars)")
        return {"account_id": account_id, "narrative": narrative, "anomaly_data": anomaly_data}

    def _tool_illustrative_cases(self, params: dict) -> dict:
        print(f"  [Tool:get_illustrative_cases] params={params}")
        n = params.get("n", 3)
        try:
            from evaluation import find_illustrative_cases, map_typology
        except ModuleNotFoundError:
            from src.evaluation import find_illustrative_cases, map_typology
        df_labeled = map_typology(self.df)
        cases = find_illustrative_cases(df_labeled, self.anomaly_scores, n=n)
        print(f"[Tool:get_illustrative_cases] Returning {len(cases)} case(s)")
        return cases.to_dict(orient="records")


# ---------------------------------------------------------------------------
# ValkyrieOrchestrator — public API
# ---------------------------------------------------------------------------


class ValkyrieOrchestrator:
    """LangGraph-based investigation orchestrator using Ollama.

    Parameters
    ----------
    df : pd.DataFrame
        SAML-D transaction data.
    detector : AnomalyDetector or SupervisedDetector
        Fitted detector instance.
    explainer : ExplainabilityEngine
        SHAP explainer wrapping *detector*.
    graph : TransactionGraph
        Built transaction graph.
    anomaly_scores : np.ndarray
        Per-row anomaly scores in [0, 1].
    binary_flags : np.ndarray
        Per-row binary anomaly flags.
    model : str
        Ollama model name (default: gemma4:e4b).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        detector: Any,
        explainer: Any,
        graph: Any,
        anomaly_scores: np.ndarray,
        binary_flags: np.ndarray,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.executor = ValkyrieToolExecutor(df, detector, explainer, graph, anomaly_scores, binary_flags)
        self.model = model
        self._graph_app: CompiledStateGraph | None = None

    def _build_graph(self) -> CompiledStateGraph:
        """Build and return the LangGraph state graph.

        Uses closure-based node factories so that the executor and model
        are captured at graph-build time rather than threaded through state.
        """
        if StateGraph is None:
            raise RuntimeError("langgraph is not installed.")

        builder = StateGraph(ValkyrieState)

        # Add closure-captured nodes
        builder.add_node("plan_query", _make_plan_query_node())
        builder.add_node("execute_plan", _make_execute_plan_node(self.executor))
        builder.add_node("summarize", _make_summarize_node(self.model))

        # Add edges
        builder.set_entry_point("plan_query")
        builder.add_conditional_edges("plan_query", _should_continue)
        builder.add_edge("execute_plan", "summarize")
        builder.add_edge("summarize", END)

        app = builder.compile()
        return app

    def investigate(self, user_query: str) -> dict:
        """Run a full investigation: plan -> execute -> summarize.

        Parameters
        ----------
        user_query : str
            Natural-language investigation request.

        Returns
        -------
        dict with keys:
            ``plan``, ``results``, ``summary``, ``execution_log``.
        """
        print(f"\n{'='*60}")
        print(f"  INVESTIGATION START")
        print(f"  Query: \"{user_query}\"")
        print(f"  LLM: Ollama ({self.model})")
        print(f"{'='*60}")

        state: ValkyrieState = {
            "user_query": user_query,
            "plan": None,
            "results": {},
            "summary": "",
            "error": None,
            "execution_log": [],
        }

        try:
            print("\n[DEBUG] Building LangGraph...")
            graph = self._build_graph()
            print("[DEBUG] LangGraph built, invoking pipeline...")
            print("[DEBUG]   Node sequence: plan_query -> execute_plan -> summarize")
            final = graph.invoke(state)
            print("\n[DEBUG] LangGraph pipeline completed successfully")
        except Exception as e:
            print(f"\n[DEBUG] !!! LangGraph pipeline failed: {e}")
            print("[DEBUG] Falling back to direct keyword + execute...")
            plan = _keyword_fallback(user_query)
            results = {}
            for entry in plan.get("tools", []):
                name, params = entry.get("name", ""), entry.get("params", {})
                print(f"[DEBUG] Fallback dispatch: {name}")
                results[name] = self.executor.dispatch(name, params)
            summary = f"Fallback investigation. Results: {dict((k, list(v.keys())[:3] if isinstance(v, dict) else 'OK') for k, v in results.items())}"
            final = {
                "plan": plan,
                "results": results,
                "summary": summary,
                "execution_log": [{"note": "Fallback after LangGraph error"}],
            }

        n_tools = len(final.get("results", {}))
        print(f"\n{'='*60}")
        print(f"  INVESTIGATION COMPLETE")
        print(f"  Tools executed: {n_tools}")
        print(f"  Summary: {final.get('summary', '')[:80]}...")
        print(f"{'='*60}")
        return {
            "plan": final.get("plan"),
            "results": final.get("results", {}),
            "summary": final.get("summary", ""),
            "execution_log": final.get("execution_log", []),
        }

    # Convenience methods for backward compatibility
    def plan_query(self, user_query: str) -> dict:
        """Return just the plan (single step, no graph)."""
        return _keyword_fallback(user_query)  # simplified

    def execute_plan(self, plan: dict) -> dict:
        """Execute a plan directly."""
        results = {}
        for entry in plan.get("tools", []):
            name, params = entry.get("name", ""), entry.get("params", {})
            results[name] = self.executor.dispatch(name, params)
        return {"plan": plan, "results": results, "summary": "Direct execution complete."}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Valkyrie-AML Orchestrator (LangGraph + Ollama)")
    print("=" * 60)
    print("\n  To test, run through the dashboard or use:")
    print("    python run.py --query 'show me anomalies'")
    print(f"\n  LLM backend: Ollama ({DEFAULT_MODEL})")
    print("=" * 60)