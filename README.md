# 🛡️ Valkyrie AML — AI-Powered Anti-Money Laundering Agent

> **Hackathon Submission** — Intelligent, agentic AML compliance system that autonomously investigates financial crime using ML anomaly detection, graph analysis, and LLM-powered reasoning.

---

## 🌟 Overview

Valkyrie is a fully agentic AML investigation platform built for financial compliance teams. Given a natural-language query from a compliance officer, Valkyrie:

1. **Detects intent** — classifies the query into investigation typologies (structuring, layering, network risk, SAR generation, model evaluation, account ranking)
2. **Plans a tool sequence** — uses Grok-3 (xAI) to generate an ordered execution plan
3. **Executes multi-step tools** — runs anomaly detection, fund flow tracing, graph centrality scoring, SHAP explanations in sequence
4. **Classifies risk** — enriches every result with CRITICAL / HIGH / MEDIUM / LOW labels and escalation actions
5. **Summarizes findings** — Grok-3 compiles a compliance-grade natural language summary

The agent never gives a one-shot LLM response. Every answer is backed by real data from 200,000 transactions.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Next.js Frontend                       │
│  Dashboard · Network Graph · Investigation Terminal      │
│  Risk Ticker · Performance Panel · SAR Generator        │
└────────────────────┬───────────────────────────────────-┘
                     │ SSE streaming (real-time events)
┌────────────────────▼────────────────────────────────────┐
│                FastAPI Backend (api/main.py)             │
│                                                          │
│  /api/investigate  ──► SSE stream with 5 agentic steps   │
│  /api/stats        ──► Dashboard KPI metrics             │
│  /api/anomalies    ──► Top suspicious transactions       │
│  /api/network-data ──► Graph nodes + edges               │
│  /api/generate-sar ──► FinCEN SAR narrative              │
│  /api/structuring  ──► Structuring detection             │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Agentic Orchestrator (src/)                 │
│                                                          │
│  ValkyrieOrchestrator  (LangGraph state machine)        │
│  ValkyrieToolExecutor  (8 pluggable tools)              │
│                                                          │
│  Tools:                                                  │
│  ├─ search_transactions          (filter by account/date/amount)  │
│  ├─ get_anomaly_scores           (per-row ML scores)    │
│  ├─ rank_accounts_by_suspicion   (account aggregation)  │
│  ├─ get_shap_explanation         (feature explainability) │
│  ├─ trace_fund_flows             (multi-hop layering)   │
│  ├─ compute_network_risk         (Personalised PageRank) │
│  ├─ evaluate_model               (precision/recall/F1)  │
│  └─ generate_sar                 (FinCEN narrative)     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                ML Subsystems (src/)                      │
│                                                          │
│  SupervisedDetector   — XGBoost on 200K SAML-D rows     │
│  ExplainabilityEngine — SHAP values for each prediction │
│  TransactionGraph     — NetworkX + Personalised PageRank│
│  ReportCompiler       — FinCEN/FATF SAR narratives      │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Agentic Pipeline** | Multi-step LangGraph orchestration with Grok-3 planning |
| **Real Data** | 200,000 transactions from SAML-D dataset loaded at startup |
| **ML Detection** | XGBoost supervised detector with per-row anomaly scores |
| **SHAP Explainability** | Every flagged transaction explained in plain English |
| **Graph Analysis** | Personalised PageRank risk propagation across sender/receiver network |
| **Fund Flow Tracing** | Multi-hop layering chain detection (up to N hops) |
| **SAR Generation** | FinCEN-compliant Suspicious Activity Report narratives via Grok-3 |
| **Structuring Detection** | Identify smurfing patterns (sub-threshold multi-transaction accounts) |
| **Live Streaming** | Investigation results streamed via Server-Sent Events (SSE) |
| **Neubrutalist UI** | Bold, high-contrast dashboard with real-time network graph |

---

## 🤖 Agentic Criteria

This implementation satisfies all hackathon agentic requirements:

- ✅ **Multi-step automatic execution** — after receiving a query, the agent automatically runs 2–4 tool calls in sequence without any user input
- ✅ **Query-driven tool selection** — the orchestrator (backed by Grok-3) decides which tools to run and in what order based on the intent of the query
- ✅ **Not a one-shot LLM response** — the LLM plans a structured tool sequence; each tool hits real data and real ML models
- ✅ **Deterministic agentic pipeline** — clear query-driven pipeline stages: intent → plan → execute → classify → summarize

**Example: "Which account had the most suspicious transactions?"**
1. Intent detected → `most_suspicious_account` pattern
2. Tools selected → `rank_accounts_by_suspicion`, then `trace_fund_flows`
3. `rank_accounts_by_suspicion` aggregates 200K rows, counts flagged transactions per account
4. Returns the **single definitive answer**: Account #XXXXXXXX — N flagged, $Y total sent
5. `trace_fund_flows` then traces layering chains from that account
6. Grok-3 compiles a compliance-grade summary answering the exact question

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Grok API key from [x.ai](https://x.ai)

### 1. Clone and install backend
```bash
git clone https://github.com/El3troX/Valkyrie-AML.git
cd Valkyrie-AML

pip install -r requirements.txt
pip install -r api/requirements-api.txt
```

### 2. Set your API key
```bash
# Windows
set GROK_API_KEY=your_grok_api_key_here

# Linux/Mac
export GROK_API_KEY=your_grok_api_key_here
```

### 3. Place the dataset
Download **SAML-D.csv** and place it at:
```
Valkyrie-AML/archive/SAML-D.csv
```

### 4. Start the backend
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
> First startup takes ~60–120 seconds to load 200K rows, train the XGBoost model, build the graph, and cache everything in memory.

### 5. Start the frontend
```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:3000**

---

## 🔬 Investigation Examples

Try these queries in the terminal:

| Query | What the agent does |
|---|---|
| `Which account had the most suspicious transactions?` | Aggregates by account, ranks by flagged count → single definitive answer |
| `Is customer 207936746 suspicious?` | Looks up all transactions, computes risk scores, runs SHAP explanation |
| `Find structuring patterns in the last 30 days` | Filters by date, detects sub-threshold multi-transaction senders |
| `Trace layering chains from account 123456789` | Multi-hop fund flow tracing + PageRank risk propagation |
| `Generate a SAR for the highest risk account` | Full FinCEN SAR narrative compiled by Grok-3 |
| `Evaluate model performance and precision/recall` | Returns precision, recall, F1, AUC at optimal threshold |

---

## 🗂️ Project Structure

```
Valkyrie-AML/
├── api/
│   ├── main.py          # FastAPI server, SSE endpoints, agentic pipeline
│   ├── startup.py       # One-time model training + graph building (cached)
│   └── models.py        # Pydantic request/response models
├── src/
│   ├── orchestrator.py  # ValkyrieOrchestrator + ValkyrieToolExecutor (LangGraph)
│   ├── ml_subsystems.py # SupervisedDetector (XGBoost) + ExplainabilityEngine (SHAP)
│   ├── graph_engine.py  # TransactionGraph (NetworkX + Personalised PageRank)
│   ├── report_compiler.py # SAR narrative generation
│   └── evaluation.py    # Model evaluation metrics + illustrative cases
├── frontend/
│   ├── src/app/dashboard/page.tsx      # Main dashboard
│   ├── src/components/network/         # Canvas 2D network graph
│   ├── src/components/agent/           # Investigation terminal (SSE)
│   └── src/components/dashboard/       # Performance panel, risk ticker
└── requirements.txt
```

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.10+, asyncio SSE |
| LLM | Grok-3-mini (xAI) with Gemini fallback |
| ML | XGBoost, SHAP |
| Graph | NetworkX, Personalised PageRank |
| Orchestration | LangGraph state machine |
| Dataset | SAML-D (200,000 synthetic AML transactions) |

---

## 📄 License

MIT — Built for hackathon purposes.