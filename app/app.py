"""Kauvery CAPEX Phase 2 — Procurement Review & Comparison Engine (Databricks App).

Upload a vendor quotation (or fill the form), score it against Kauvery's own purchase
history via the registered ML model, and see verdict + gaps + citations.
"""
import json
import os

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient

st.set_page_config(page_title="CAPEX Quote Review", page_icon="🏥", layout="wide")

ENDPOINT = os.getenv("SERVING_ENDPOINT", "capex-worth-it")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")  # optional: enables PDF parsing


@st.cache_resource
def get_client() -> WorkspaceClient:
    return WorkspaceClient()


def score_quote(records: list[dict]) -> list[dict]:
    resp = get_client().serving_endpoints.query(name=ENDPOINT, dataframe_records=records)
    return resp.predictions


def verdict_color(v: str) -> str:
    return {"Accept": "#1B5E20", "Negotiate": "#E65100", "Reject": "#B71C1C"}.get(v, "#333")


# ---------------------------------------------------------------- header
st.title("🏥 CAPEX Procurement Review & Comparison Engine")
st.caption(
    "Phase 2 — score a vendor quotation against Kauvery's own purchase history. "
    f"Model endpoint: `{ENDPOINT}`. Grounded in historical benchmarks, not a model's general knowledge."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Describe the quoted line item (or upload the PDF).\n"
        "2. The custom ML model benchmarks it against Kauvery history.\n"
        "3. You get a **worth-it score (0–100)**, a verdict, and a **gap checklist with citations**.\n\n"
        "Weighting: Price 30 · Warranty 15 · AMC/CAMC 15 · Delivery 10 · FOC 10 · Frequency 10 · Payment 10."
    )
    st.divider()
    st.caption("Data privacy: document parsing uses Databricks-hosted Foundation Model APIs; "
               "data stays inside Databricks and is not used to train the models.")

# ---------------------------------------------------------------- input form
st.subheader("Quotation line item")
with st.form("quote"):
    c1, c2, c3 = st.columns(3)
    with c1:
        item_description = st.text_input("Item description", "Philips IntelliVue MX450 Patient Monitor")
        make_brand = st.text_input("Brand", "Philips")
        model_no = st.text_input("Model no.", "IntelliVue MX450")
    with c2:
        qty = st.number_input("Quantity", min_value=1, value=10)
        unit_rate = st.number_input("Quoted unit rate (INR)", min_value=0.0, value=415000.0, step=1000.0)
        warranty_months = st.number_input("Warranty (months)", min_value=0, value=12)
    with c3:
        delivery_lead_days = st.number_input("Delivery lead time (days)", min_value=0, value=95)
        payment_terms = st.selectbox(
            "Payment terms",
            ["100% advance", "50% advance, 50% against delivery", "30% advance, 70% on installation",
             "100% against delivery", "Net 30 days", "Net 45 days"],
        )
    c4, c5, c6, c7 = st.columns(4)
    amc_present = c4.checkbox("AMC/CMC included", value=False)
    foc_present = c5.checkbox("FOC accessories", value=False)
    has_training = c6.checkbox("Training package", value=False)
    has_installation = c7.checkbox("Installation & commissioning", value=True)
    submitted = st.form_submit_button("Score quotation", type="primary")

# ---------------------------------------------------------------- results
if submitted:
    record = {
        "item_description": item_description, "make_brand": make_brand, "model_no": model_no,
        "qty": int(qty), "unit_rate": float(unit_rate), "warranty_months": int(warranty_months),
        "amc_present": bool(amc_present), "camc_present": False, "foc_present": bool(foc_present),
        "delivery_lead_days": int(delivery_lead_days), "payment_terms": payment_terms,
        "has_training": bool(has_training), "has_installation": bool(has_installation),
    }
    try:
        preds = score_quote([record])
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not reach the serving endpoint `{ENDPOINT}`: {e}")
        st.stop()

    r = preds[0]
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Worth-it score", f"{r['worth_score']} / 100")
    m2.markdown(f"### <span style='color:{verdict_color(r['verdict'])}'>{r['verdict']}</span>",
                unsafe_allow_html=True)
    m3.metric("Match level", r["match_level"])
    pv = r.get("price_variance_pct")
    m4.metric("Price vs benchmark", f"{pv:+.1f}%" if pv is not None else "n/a",
              help="Quoted unit rate vs most-recent comparable historical purchase.")

    bench = r.get("benchmark_unit_rate")
    if bench:
        st.caption(
            f"Benchmark: most-recent comparable purchase was INR {bench:,.0f} "
            f"(PO {r.get('benchmark_po')}, {r.get('benchmark_po_date')}). "
            f"Quoted: INR {r['quoted_unit_rate']:,.0f}."
        )

    gaps = json.loads(r["gaps_json"]) if isinstance(r.get("gaps_json"), str) else (r.get("gaps") or [])
    st.subheader(f"Gap checklist — {len(gaps)} flag(s)")
    if not gaps:
        st.success("No gaps detected against the Reference BOM for this category.")
    for g in gaps:
        icon = "🔴" if g.get("severity") == "High" else "🟠"
        with st.container(border=True):
            st.markdown(f"{icon} **{g['component']}** ({g.get('severity','')}) — {g['message']}")
            st.caption(f"📎 {g['citation']}")
