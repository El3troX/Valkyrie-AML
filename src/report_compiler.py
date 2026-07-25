"""
SAR (Suspicious Activity Report) generation for Valkyrie-AML.

Uses Ollama (llama3.2) to draft professional FATF/FinCEN-style narratives
from structured investigation data, then produces a formatted PDF via
ReportLab.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# LLM imports
# ---------------------------------------------------------------------------

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore

# ---------------------------------------------------------------------------
# LLM-generated SAR narrative
# ---------------------------------------------------------------------------

DEFAULT_SAR_MODEL = "llama3.2"


def generate_sar_narrative(
    account_id: str,
    anomaly_data: dict[str, Any],
    graph_data: dict[str, Any],
    shap_explanation: dict[str, Any],
    model: str = "grok-3-mini",
) -> str:
    """Draft a professional FinCEN/FATF SAR narrative using Grok API (with Gemini/Ollama fallback)."""
    print(f"[SAR] Building SAR narrative for account {account_id} ...")
    prompt = _build_sar_prompt(account_id, anomaly_data, graph_data, shap_explanation)

    system_inst = (
        "You are an expert FinCEN Compliance Officer drafting a Suspicious Activity Report (SAR). "
        "Write in a formal, professional, objective tone. "
        "CRITICAL RULE: Do NOT output any bracketed placeholder text like [Date] or [Account Holder Name]. "
        "Do NOT include raw math like '(SHAP: -0.04)'. Use ONLY plain English compliance descriptions."
    )

    grok_key = os.environ.get("GROK_API_KEY")
    if grok_key:
        try:
            import json, urllib.request
            url = "https://api.x.ai/v1/chat/completions"
            payload = {
                "model": "grok-3-mini",
                "messages": [
                    {"role": "system", "content": system_inst},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 2048,
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
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                narrative = data["choices"][0]["message"]["content"].strip()
                if narrative and "[" not in narrative and "]" not in narrative:
                    print(f"[SAR] Grok SAR narrative received ({len(narrative)} chars)")
                    return narrative
        except Exception as e:
            print(f"[SAR] Grok API call failed: {e}. Trying Gemini...")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            load_dotenv(Path.home() / ".env")
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        except ImportError:
            pass

    if api_key:
        # Attempt 1: google-genai SDK (import google.genai as genai)
        try:
            import google.genai as genai
            from google.genai import types
            print(f"[SAR] Calling Gemini API (gemini-2.5-flash) via SDK...")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_inst}\n\n{prompt}",
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
            )
            narrative = response.text.strip() if response.text else ""
            if narrative and "[" not in narrative and "]" not in narrative:
                print(f"[SAR] Gemini SAR narrative received ({len(narrative)} chars)")
                return narrative
        except Exception as e:
            print(f"[SAR] Gemini SDK import/call failed ({e}), using REST fallback...")

        # Attempt 2: Direct HTTP REST fallback
        for m_name in ["gemini-2.5-flash", "gemini-1.5-flash"]:
            try:
                import json, urllib.request
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": f"{system_inst}\n\n{prompt}"}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            narrative = parts[0].get("text", "").strip()
                            if narrative and "[" not in narrative and "]" not in narrative:
                                print(f"[SAR] Gemini REST narrative received ({m_name}, {len(narrative)} chars)")
                                return narrative
            except Exception as e:
                print(f"[SAR] Gemini REST call failed ({m_name}): {e}")

    if ChatOllama is not None:
        try:
            print(f"[SAR] Calling Ollama for narrative generation...")
            llm = ChatOllama(model="llama3.2", temperature=0.2, num_predict=600, timeout=3)
            messages = [
                (
                    "system",
                    (
                        "You are an expert FinCEN Compliance Officer drafting a Suspicious Activity Report (SAR). "
                        "Write in a formal, professional, objective tone. "
                        "CRITICAL RULE: Do NOT output any bracketed placeholder text like [Date] or [Account Holder Name]. "
                        "Do NOT include raw math like '(SHAP: -0.04)'. Use ONLY plain English compliance descriptions."
                    ),
                ),
                ("human", prompt),
            ]
            response = llm.invoke(messages)
            narrative = response.content.strip()
            if narrative and "[" not in narrative and "]" not in narrative:
                print(f"[SAR] Ollama narrative received ({len(narrative)} chars)")
                return narrative
        except Exception as e:
            print(f"[SAR] Ollama call skipped ({e}), using template.")

    return _template_narrative(account_id, anomaly_data, graph_data, shap_explanation)


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------


def _template_narrative(
    account_id: str,
    anomaly_data: dict,
    graph_data: dict | None = None,
    shap_explanation: dict | None = None,
) -> str:
    """Generate a clean, fully-populated professional FinCEN-style SAR narrative."""
    amount = anomaly_data.get("total_amount", 0)
    txns = anomaly_data.get("n_transactions", 0)
    score = anomaly_data.get("max_score", 0.0)

    connections = graph_data.get("top_connections", []) if graph_data else []
    conn_list = [str(a) for a, s in connections if str(a) != str(account_id)]
    conn_str = ", ".join(conn_list[:5]) if conn_list else "None identified within 2-hop radius"

    red_flags = []
    if shap_explanation and "top_features" in shap_explanation:
        for f in shap_explanation["top_features"][:4]:
            red_flags.append(f"• {f['plain_language']}")
    if not red_flags:
        red_flags = [
            "• Transaction amounts deviate significantly from account historical baseline",
            "• High-velocity transfer frequency within condensed timeframe",
            "• Exposure to high-risk counterparty networks",
        ]
    red_flags_text = "\n".join(red_flags)

    severity = "CRITICAL" if score >= 0.8 else "HIGH" if score >= 0.6 else "MODERATE"

    return (
        f"FINCEN SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE\n"
        f"Subject Account ID: {account_id} | Risk Rating: {severity} (Anomaly Score: {score:.3f})\n"
        f"--------------------------------------------------------------------------------\n\n"
        f"1. EXECUTIVE SUMMARY\n"
        f"An investigation was conducted on Subject Account {account_id} following automated "
        f"flagging by Valkyrie-AML detection subsystems. During the observation window, account "
        f"{account_id} conducted a total of {txns:,} transactions with a cumulative volume of "
        f"${amount:,.2f}. The peak anomaly score reached {score:.3f}, meeting the threshold "
        f"for mandatory SAR documentation.\n\n"
        f"2. TYPOLOGY & RED FLAG ANALYSIS\n"
        f"Quantitative feature analysis and SHAP explainability models identified the following "
        f"primary compliance red flags:\n"
        f"{red_flags_text}\n\n"
        f"These indicators are strongly characteristic of money laundering typologies, specifically "
        f"rapid layering, smurfing, and structuring designed to evade reporting thresholds.\n\n"
        f"3. NETWORK RISK & COUNTERPARTY EXPOSURE\n"
        f"Graph analytics and Personalised PageRank (PPR) risk propagation identified key network "
        f"connections between Account {account_id} and the following counterparty accounts: "
        f"{conn_str}. Risk scores propagated through the transaction network indicate potential "
        f"coordinated multi-account laundering schemes.\n\n"
        f"4. RECOMMENDATIONS & NEXT STEPS\n"
        f"Based on the empirical evidence, the compliance officer recommends:\n"
        f"• Place Subject Account {account_id} under enhanced monitoring and temporary restriction.\n"
        f"• Initiate full Know-Your-Customer (KYC) / Enhanced Due Diligence (EDD) review.\n"
        f"• Escalate this report to the Financial Intelligence Unit (FIU) and regulatory authorities."
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_sar_prompt(
    account_id: str,
    anomaly_data: dict,
    graph_data: dict,
    shap_explanation: dict,
) -> str:
    txns = anomaly_data.get("n_transactions", 0)
    total = anomaly_data.get("total_amount", 0)
    max_score = anomaly_data.get("max_score", 0)

    connections = graph_data.get("top_connections", [])
    conn_list = [str(a) for a, s in connections if str(a) != str(account_id)]
    conn_str = ", ".join(conn_list[:5]) if conn_list else "None"

    shap_top = shap_explanation.get("top_features", [])
    red_flags = []
    for f in shap_top[:4]:
        red_flags.append(f"- {f['plain_language']}")
    red_flags_str = "\n".join(red_flags) if red_flags else "- Statistically high velocity and amount deviation"

    return (
        f"Write a formal Suspicious Activity Report for account {account_id}.\n\n"
        f"EXACT DATA (Do NOT use placeholders like [Date] or [Name]):\n"
        f"- Subject Account: {account_id}\n"
        f"- Total Transactions: {txns:,}\n"
        f"- Total Amount: ${total:,.2f}\n"
        f"- Peak Anomaly Score: {max_score:.3f}\n"
        f"- Connected Risk Accounts: {conn_str}\n\n"
        f"RED FLAGS:\n{red_flags_str}\n\n"
        f"Write 4 structured sections: 1. Executive Summary, 2. Typology & Red Flag Analysis, 3. Network Risk Exposure, 4. Recommendations."
    )


# ---------------------------------------------------------------------------
# PDF compilation
# ---------------------------------------------------------------------------


def _md_to_reportlab_html(text: str) -> str:
    """Convert Markdown formatting (**bold**, * bullet) into ReportLab-supported HTML tags."""
    import re
    # Replace HTML entities safely
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Convert bold **text** to <b>text</b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    # Convert italic *text* or _text_ to <i>text</i>
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    # Convert bullet points * or - at start of line
    text = re.sub(r"^\s*[\*\-]\s*", r"&bull; ", text, flags=re.MULTILINE)
    return text


def compile_pdf(
    narrative: str,
    account_id: str,
    transaction_table: pd.DataFrame,
    anomaly_scores: list[float] | None = None,
    output_path: str = "data/sar_report.pdf",
) -> str:
    """Generate a formatted SAR PDF using ReportLab.

    Parameters
    ----------
    narrative : str
        SAR narrative text.
    account_id : str
        Subject account ID.
    transaction_table : pd.DataFrame
        Transactions to include in the report.
    anomaly_scores : list[float] or None
        Per-transaction anomaly scores (same length as table rows).
    output_path : str
        Destination path for the PDF.

    Returns
    -------
    str
        The absolute path to the generated PDF.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    print(f"[SAR:pdf] Compiling PDF for account {account_id}...")
    print(f"[SAR:pdf]   Transaction table: {len(transaction_table)} rows")
    print(f"[SAR:pdf]   Narrative length: {len(narrative)} chars")
    print(f"[SAR:pdf]   Output: {output_path}")

    # Prepare document
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SARTitle", parent=styles["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#0F172A"), alignment=0
    )
    heading_style = ParagraphStyle(
        "SARHeading", parent=styles["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#1E293B"), spaceBefore=10, spaceAfter=6
    )
    normal_style = ParagraphStyle(
        "SARBody", parent=styles["Normal"], spaceAfter=6, fontSize=9, leading=13, textColor=colors.HexColor("#334155")
    )
    cell_style = ParagraphStyle(
        "Cell", parent=styles["Normal"], fontSize=8, leading=10, spaceAfter=0, textColor=colors.HexColor("#0F172A")
    )

    story: list = []

    # Title
    story.append(Paragraph("SUSPICIOUS ACTIVITY REPORT (SAR)", title_style))
    story.append(Spacer(1, 4))

    # Header info
    story.append(
        Paragraph(
            f"<b>Subject Account ID:</b> {account_id} &nbsp;|&nbsp; "
            f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            heading_style,
        )
    )
    story.append(Spacer(1, 10))

    # Narrative section
    story.append(Paragraph("Compliance Investigation Narrative", heading_style))
    for para in narrative.strip().split("\n\n"):
        para = para.strip()
        if para:
            formatted_para = _md_to_reportlab_html(para).replace("\n", "<br/>")
            story.append(Paragraph(formatted_para, normal_style))
    story.append(Spacer(1, 10))

    # Transaction table
    story.append(Paragraph("Supporting Suspicious Transactions", heading_style))
    story.append(Spacer(1, 6))

    if not transaction_table.empty:
        cols = ["Date", "Sender_account", "Receiver_account", "Amount", "Laundering_type"]
        available_cols = [c for c in cols if c in transaction_table.columns]

        col_display_names = {
            "Date": "Date",
            "Sender_account": "Sender Account",
            "Receiver_account": "Receiver Account",
            "Amount": "Amount ($)",
            "Laundering_type": "Typology / Category",
        }

        header = [Paragraph(f"<b>{col_display_names.get(c, c)}</b>", cell_style) for c in available_cols]

        data_rows = [header]
        for i, (_, row) in enumerate(transaction_table.head(20).iterrows()):
            score = anomaly_scores[i] if anomaly_scores and i < len(anomaly_scores) else None
            vals = []
            for c in available_cols:
                v = row[c]
                if c == "Amount":
                    v = f"${v:,.2f}"
                else:
                    v = str(v)
                vals.append(Paragraph(v, cell_style))

            if score is not None and available_cols:
                typ_val = str(row[available_cols[-1]])
                vals[-1] = Paragraph(f"{typ_val}<br/><b>(Score: {score:.3f})</b>", cell_style)

            data_rows.append(vals)

        # Well-proportioned column widths (Total = 7.3 inches printable area)
        col_widths = [0.85 * inch, 1.25 * inch, 1.25 * inch, 1.0 * inch, 2.35 * inch]

        table = Table(data_rows, colWidths=col_widths[: len(available_cols)], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)
    else:
        story.append(Paragraph("No transaction data available.", normal_style))

    story.append(Spacer(1, 20))

    # Footer
    story.append(
        Paragraph(
            f"<i>Report generated by Valkyrie-AML Compliance Agent on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}. "
            f"This report is confidential and intended for compliance use only.</i>",
            ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
        )
    )

    doc.build(story)
    output_abs = str(out_path.resolve())
    print(f"[SAR:pdf] PDF generated at: {output_abs}")
    return output_abs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing SAR report generation ...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    data_path = Path(__file__).resolve().parent.parent / "data" / "SAML-D.csv"
    print(f"Loading SAML-D (10K sample) ...")
    df = pd.read_csv(data_path, nrows=10_000)

    # Quick anomaly scores
    from ml_subsystems import AnomalyDetector

    detector = AnomalyDetector(contamination=0.005)
    detector.fit(df)
    scores, flags = detector.predict(df)

    # Pick top account
    top_idx = int(scores.argmax())
    account = str(df.iloc[top_idx]["Sender_account"])

    # Anomaly data
    mask = df["Sender_account"].astype(str) == account
    anomaly_data = {
        "account_id": account,
        "n_transactions": int(mask.sum()),
        "total_amount": float(df.loc[mask, "Amount"].sum()),
        "max_score": float(scores[mask.values].max()),
        "mean_score": float(scores[mask.values].mean()),
    }

    # Graph
    from graph_engine import TransactionGraph
    graph = TransactionGraph(df)
    graph.build()
    ppr = graph.personalized_pagerank([account])
    graph_data = {"top_connections": sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:5]}

    # SHAP
    from ml_subsystems import ExplainabilityEngine
    explainer = ExplainabilityEngine(detector)
    shap_explanation = explainer.explain(df, int(top_idx))

    print(f"Generating SAR for account {account} ...")
    narrative = generate_sar_narrative(account, anomaly_data, graph_data, shap_explanation)
    print(f"\nNarrative:\n{narrative}\n")

    txns = df[mask].head(10)
    out = compile_pdf(narrative, account, txns, scores[mask.values].tolist())
    print(f"PDF generated: {out}")
