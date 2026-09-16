"""Kauvery CAPEX — Conversational Quotation Gap-Detection app (Databricks App).

Genie-style chat. The app orchestrates a Databricks-hosted Llama foundation model with tool-calling:
the FM handles conversation + multi-turn; the tools do the deterministic work —
  - price_fairness  -> the custom ML model endpoint (capex-worth-it)
  - cross_unit_history / recommend_vendor -> UC SQL functions over Kauvery history.
Everything stays inside Databricks (no external egress). Only the final decision is persisted.
"""
import io
import json
import os
import uuid

import streamlit as st
from databricks.sdk import WorkspaceClient

st.set_page_config(page_title="CAPEX Quote Assistant", page_icon="🏥", layout="wide")

CHAT_LLM = os.getenv("CHAT_LLM", "databricks-llama-4-maverick")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "capex-worth-it")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
CATALOG = os.getenv("CATALOG", "kauvey_poc")
SCHEMA = os.getenv("SCHEMA", "gold")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/landing"
DECISIONS_TABLE = f"{CATALOG}.{SCHEMA}.purchase_decisions"

SYNTH_PROMPT = (
    "You are the CAPEX Procurement Intelligence assistant for Kauvery Hospital. You are given a "
    "quotation and pre-computed EVIDENCE (JSON) from three deterministic tools: price_fairness (the ML "
    "model: worth_score 0-100, verdict, price_variance_pct vs the most-recent comparable purchase, gaps "
    "with citations), cross_unit_history (purchases across all units, low-to-high), and recommend_vendor "
    "(vendors ranked by value; bundled FOC+AMC at low price ranks highest).\n\n"
    "Answer CONVERSATIONALLY, per line item:\n"
    "  1. Item -> quoted price.  2. Gaps found (with their citations).  3. Price fairness: state the "
    "verdict and the % vs the most-recent comparable purchase (never say 'overcharging').  4. A markdown "
    "cross-unit table (unit, vendor, date, unit_rate, warranty, AMC, FOC) low-to-high across ALL units.  "
    "5. Recommended vendor (prefer bundled FOC+AMC at a low price).  6. An overall actionable "
    "recommendation the capex team can take into negotiation.\n"
    "Use ONLY the evidence; never invent prices, vendors, dates, or specs.")

FOLLOWUP_PROMPT = (
    "You are the CAPEX Procurement Intelligence assistant for Kauvery Hospital. Answer the user's "
    "follow-up using the conversation and the EVIDENCE already gathered (JSON below). Be concise and "
    "management-oriented. Use only the evidence; do not invent data. If the user asks to draft a vendor "
    "letter, write a short professional letter citing the specific missing items.")

EXTRACT_PROMPT = (
    "Extract quotation line items from the user's message as JSON with key 'line_items' = array of "
    "{item_description, make_brand, model_no, qty (int), unit_rate (number), warranty_months (int), "
    "amc_present (bool), foc_present (bool), delivery_lead_days (int), payment_terms (string), "
    "has_training (bool), has_installation (bool)}. Booleans reflect whether the quote includes AMC/CMC, "
    "FOC, training, installation. If the message is a general question (not a quotation), return "
    "{\"line_items\": []}.")


@st.cache_resource
def wc() -> WorkspaceClient:
    return WorkspaceClient()


@st.cache_resource
def llm():
    return wc().serving_endpoints.get_open_ai_client()


def run_sql(statement: str):
    r = wc().statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout="50s")
    if r.status and r.status.state.value != "SUCCEEDED":
        raise RuntimeError(getattr(r.status.error, "message", r.status.state.value))
    return [row for row in (r.result.data_array or [])] if r.result else []


# ---- tool executors ----
def exec_price_fairness(**kw) -> str:
    rec = {"item_description": kw.get("item_description"), "make_brand": kw.get("make_brand"),
           "model_no": kw.get("model_no"), "qty": int(kw.get("qty", 1)),
           "unit_rate": float(kw.get("unit_rate", 0)), "warranty_months": int(kw.get("warranty_months", 0)),
           "amc_present": bool(kw.get("amc_present", False)), "camc_present": False,
           "foc_present": bool(kw.get("foc_present", False)),
           "delivery_lead_days": int(kw.get("delivery_lead_days", 60)),
           "payment_terms": kw.get("payment_terms", ""), "has_training": bool(kw.get("has_training", False)),
           "has_installation": bool(kw.get("has_installation", True))}
    resp = wc().serving_endpoints.query(name=MODEL_ENDPOINT, dataframe_records=[rec])
    return json.dumps(resp.predictions[0] if resp.predictions else {})


def exec_sql_fn(fn: str, search: str) -> str:
    safe = search.replace("'", "")
    rows = run_sql(f"SELECT {CATALOG}.{SCHEMA}.{fn}('{safe}') AS r")
    return rows[0][0] if rows and rows[0] else "[]"


def _chat(messages, max_tokens=2000, json_mode=False):
    kw = {"model": CHAT_LLM, "messages": messages, "max_tokens": max_tokens}
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    return llm().chat.completions.create(**kw).choices[0].message.content or ""


def extract_line_items(text: str) -> list[dict]:
    """Use the FM to pull structured line items from free text; [] if it's a general question."""
    try:
        out = _chat([{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": text}],
                    max_tokens=800, json_mode=True)
        return json.loads(out).get("line_items", [])
    except Exception:  # noqa: BLE001
        return []


def gather_evidence(lines: list[dict]) -> list[dict]:
    """Deterministically run all three tools per line item (no reliance on LLM tool-sequencing)."""
    ev = []
    for ln in lines:
        search = ln.get("model_no") or ln.get("item_description") or ""
        try:
            pf = json.loads(exec_price_fairness(**ln))
        except Exception as e:  # noqa: BLE001
            pf = {"error": str(e)}
        ev.append({"line": ln, "price_fairness": pf,
                   "cross_unit_history": json.loads(exec_sql_fn("cross_unit_history", search) or "[]"),
                   "recommend_vendor": json.loads(exec_sql_fn("recommend_vendor", search) or "[]")})
    return ev


def handle_turn(user_text: str, pending_lines) -> str:
    """One conversational turn. If a quotation is present, gather evidence + synthesize; else follow-up."""
    lines = pending_lines if pending_lines is not None else extract_line_items(user_text)
    hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]
    if lines:
        evidence = gather_evidence(lines)
        st.session_state.evidence = evidence
        msgs = [{"role": "system", "content": SYNTH_PROMPT}] + hist + [
            {"role": "user", "content": f"{user_text}\n\nEVIDENCE:\n{json.dumps(evidence, default=str)}"}]
        return _chat(msgs)
    # follow-up: answer from conversation + previously gathered evidence
    ev = st.session_state.get("evidence", [])
    msgs = [{"role": "system", "content": FOLLOWUP_PROMPT}] + hist + [
        {"role": "user", "content": f"{user_text}\n\nEVIDENCE:\n{json.dumps(ev, default=str)}"}]
    return _chat(msgs)


def parse_pdf(file_bytes: bytes) -> list[dict]:
    path = f"{VOLUME_PATH}/{uuid.uuid4().hex}.pdf"
    wc().files.upload(path, io.BytesIO(file_bytes), overwrite=True)
    schema = ("{item_description, make_brand, model_no, qty (int), unit_rate (number), "
              "warranty_months (int), amc_present (bool), foc_present (bool), delivery_lead_days (int), "
              "payment_terms (string), has_training (bool), has_installation (bool)}")
    stmt = f"""
      WITH raw AS (SELECT content FROM read_files('{path}', format => 'binaryFile')),
      parsed AS (SELECT concat_ws('\\n', transform(
                          cast(ai_parse_document(content):document:elements AS ARRAY<VARIANT>),
                          e -> e:content::string)) AS txt FROM raw)
      SELECT ai_query('{CHAT_LLM}',
        concat('Extract every quotation line item as JSON with key "line_items" = array of {schema}. ',
               'Booleans reflect whether the quote includes AMC/CMC, FOC, training, installation. Text:\\n', txt),
        responseFormat => '{{"type":"json_object"}}') AS extracted FROM parsed"""
    rows = run_sql(stmt)
    return json.loads(rows[0][0]).get("line_items", []) if rows and rows[0] else []


def save_decision(item, vendor, price, rationale):
    run_sql(f"""CREATE TABLE IF NOT EXISTS {DECISIONS_TABLE}
        (decided_at TIMESTAMP, item_description STRING, chosen_vendor STRING,
         agreed_unit_rate DOUBLE, rationale STRING, decided_by STRING)""")
    who = (wc().current_user.me().user_name or "app").replace("'", "")
    run_sql(f"""INSERT INTO {DECISIONS_TABLE} VALUES (current_timestamp(),
        '{item.replace("'","")}','{vendor.replace("'","")}',{float(price)},
        '{rationale.replace("'","")}','{who}')""")


# ---------------------------------------------------------------- UI
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🏥 CAPEX Quotation Assistant")
st.caption(f"Conversational gap-detection over Kauvery's history · reasoning: `{CHAT_LLM}` · "
           f"scoring: `{MODEL_ENDPOINT}` (custom ML) · all processing stays inside Databricks.")

with st.sidebar:
    st.header("📄 Upload a quotation")
    up = st.file_uploader("Vendor quotation PDF", type=["pdf"])
    if up is not None and st.button("Analyze quotation", type="primary", use_container_width=True):
        with st.spinner("Parsing PDF with Databricks AI functions…"):
            try:
                lines = parse_pdf(up.getvalue())
            except Exception as e:  # noqa: BLE001
                lines = []
                st.warning(f"PDF parse unavailable ({e}). Describe the item in the chat instead.")
        if lines:
            names = ", ".join(f"{l.get('qty','')}x {l.get('item_description','item')}" for l in lines)
            st.session_state.pending_lines = lines
            st.session_state.pending_text = f"Please review this uploaded quotation: {names}."
            st.success(f"Extracted {len(lines)} line item(s).")

    st.divider()
    st.header("✅ Record final decision")
    with st.form("decision"):
        d_item = st.text_input("Item"); d_vendor = st.text_input("Chosen vendor")
        d_price = st.number_input("Agreed unit rate (INR)", min_value=0.0, step=1000.0)
        d_note = st.text_area("Rationale", height=70)
        if st.form_submit_button("Save decision") and d_item and d_vendor:
            try:
                save_decision(d_item, d_vendor, d_price, d_note)
                st.success("Saved (only the final decision is stored — not the quotation).")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not save: {e}")

    st.divider()
    if st.button("Draft a vendor query letter", use_container_width=True):
        st.session_state.pending_text = ("Draft a short professional letter to the vendor asking them to "
                                         "clarify or include the missing items you identified.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about a quotation, a vendor, or cross-unit pricing…")
pending_lines = None
if not prompt and st.session_state.get("pending_text"):
    prompt = st.session_state.pop("pending_text")
    pending_lines = st.session_state.pop("pending_lines", None)

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            try:
                answer = handle_turn(prompt, pending_lines)
            except Exception as e:  # noqa: BLE001
                answer = f"Sorry — I hit an error: {e}"
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
