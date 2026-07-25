# 🛡️ Valkyrie-AML — Compliance Investigation Agent

Valkyrie-AML is a compliance investigation agent for detecting money laundering in financial transaction networks. Built for the SAML-D dataset, it combines **unsupervised anomaly detection (Isolation Forest)**, **graph-based risk propagation (Personalized PageRank)**, **SHAP explainability**, and an **LLM-driven orchestrator (Ollama Gemma4:e4b)** to surface suspicious activity, trace fund flows, and generate Suspicious Activity Reports (SARs).

## Dataset

This project uses the **SAML-D (Synthetic Anti-Money Laundering) Dataset** available on Kaggle:

- **Source**: [berkanoztas/synthetic-transaction-monitoring-dataset-aml](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
- **Rows**: ~9.5M
- **Schema**: `Time`, `Date`, `Sender_account`, `Receiver_account`, `Amount`, `Payment_currency`, `Received_currency`, `Sender_bank_location`, `Receiver_bank_location`, `Payment_type`, `Is_laundering`, `Laundering_type`
- **Laundering typologies**: Structuring, Smurfing, Layering, plus normal transaction patterns

### Setup

1. **Place the dataset** at `data/SAML-D.csv` (download from Kaggle and copy to the `data/` directory)
2. The system will load the CSV directly — no preprocessing needed.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (train + evaluate + report)
python run.py --full-pipeline --nrows 200000

# Launch the interactive dashboard
python run.py --dashboard
```

## Usage

### Command-Line Interface

| Command | Description |
|---------|-------------|
| `python run.py --full-pipeline --nrows 200000` | Train, evaluate, and report metrics on 200K rows |
| `python run.py --evaluate --nrows 200000` | Run evaluation and print metrics only |
| `python run.py --tune --nrows 200000` | Sweep thresholds to find optimal F1 |
| `python run.py --dashboard` | Launch the Streamlit web dashboard |
| `python run.py --query "Show me the most suspicious accounts"` | Run a natural-language query |

### Streamlit Dashboard

```bash
python run.py --dashboard
```

The dashboard features four tabs:

1. **Investigation** — Natural-language query input; sends queries through the LLM orchestrator to plan and execute detection tasks.
2. **Network Graph** — Interactive PyVis network visualization with nodes colored by risk score (green → yellow → red) and sized by PageRank.
3. **Model Performance** — Precision, recall, F1, confusion matrix, and per-typology breakdown.
4. **SAR Report** — Generate and download a Suspicious Activity Report PDF for any account.

### Running with Python Modules

Each module can also be run independently:

```bash
# ML detection
python -m src.ml_subsystems          # Quick test on 50K rows

# Evaluation (requires src/ in PYTHONPATH)
python -m src.evaluation             # Run evaluation on 50K sample

# Graph engine
python -m src.graph_engine           # Build graph and test PPR

# Orchestrator (runs locally via Ollama)
python -m src.orchestrator           # LLM-based query planning

# SAR report (runs locally via Ollama for narrative generation)
python -m src.report_compiler        # Generate sample SAR PDF
```

## Project Structure

```
valkyrie-compliance-agent/
├── src/
│   ├── __init__.py
│   ├── ml_subsystems.py      # Isolation Forest + SHAP explainability
│   ├── evaluation.py         # Precision/recall/F1, confusion matrix, illustrative cases
│   ├── graph_engine.py       # NetworkX directed graph, Personalized PageRank, chain tracing
│   ├── orchestrator.py       # LangGraph + Ollama query planner → tool dispatch
│   ├── report_compiler.py    # Ollama SAR narrative + ReportLab PDF generation
│   └── dashboard.py          # Streamlit + PyVis interactive UI
├── data/
│   └── SAML-D.csv            # Kaggle SAML-D dataset (not tracked in git)
├── .streamlit/
│   └── config.toml           # Dark theme config
├── requirements.txt
├── run.py                    # CLI entry point
└── README.md
```

## Architecture

1. **ml_subsystems.py** — Engineers 6 features per transaction: sender transaction count, total volume, average amount, amount deviation (z-score), cash ratio, and near-threshold flag. Fits an IsolationForest (unsupervised) and returns anomaly scores in [0, 1]. The `ExplainabilityEngine` wraps SHAP TreeExplainer to translate model decisions into plain language.

2. **evaluation.py** — Computes precision, recall, F1, and confusion matrix against SAML-D ground-truth labels. Supports per-typology breakdown (structuring, smurfing, layering). Finds optimal binary classification threshold by sweeping.

3. **graph_engine.py** — Builds a directed NetworkX graph from the transaction DataFrame. Implements Personalized PageRank for network risk propagation from known high-risk accounts. `trace_chains()` discovers multi-hop layering paths within configurable time windows.

4. **orchestrator.py** — Takes a natural-language query and routes it through a LangGraph pipeline (plan_query → execute_plan → summarize) backed by Ollama Gemma4:e4b. Returns a structured JSON execution plan. Dispatches each tool call to the appropriate subsystem. Falls back to keyword routing if the LLM is unavailable.

5. **report_compiler.py** — Generates SAR narratives via Ollama Gemma4:e4b (or template fallback) and compiles professional PDFs with ReportLab, including transaction tables with proper cell wrapping.

6. **dashboard.py** — Streamlit app with dark theme, PyVis network graph, model performance panel, and SAR download.

## Evaluation Results (200K rows)

| Metric | Overall | Structuring | Smurfing | Layering |
|--------|---------|-------------|----------|----------|
| Precision | 0.6971 | 1.0000 | 1.0000 | 1.0000 |
| Recall | 0.6144 | 0.8182 | 0.6286 | 0.3000 |
| F1 Score | 0.6532 | 0.9000 | 0.7719 | 0.4615 |
| Support | 236 | 121 | 35 | 80 |

**Top detection cases** (highest-scoring true positives):
- `sender=207936746` → `Single_large` ($161K, score=1.000)  
- `sender=3979625953` → `Stacked Bipartite` ($9.3K, score=0.9999)  
- `sender=6381030298` → `Cash_Withdrawal` ($148, score=0.9993)  

## License

This project is for hackathon / educational purposes only. The SAML-D dataset is licensed under CC-BY-NC-SA-4.0.