"""Kauvery CAPEX — Conversational Quotation Gap-Detection app (Databricks App).

A Genie-style chat over a custom agent. Upload a quotation PDF; the app extracts line items
with Databricks AI functions, then talks to the CAPEX agent endpoint (which wraps the ML model +
cross-unit history + vendor recommendation) for a conversational, multi-turn review.

No raw model calls here — the app only talks to the AGENT endpoint. Nothing leaves Databricks.
"""
import io
import json
import os
import uuid

import streamlit as st
from databricks.sdk import WorkspaceClient

st.set_page_config(page_title="CAPEX Quote Assistant", page_icon="🏥", layout="wide")

# ---- config (injected via app.yaml valueFrom / env) ----
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", "capex-agent")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
PARSE_LLM = os.getenv("PARSE_LLM", "databricks-llama-4-maverick")
CATALOG = os.getenv("CATALOG", "kauvey_poc")
SCHEMA = os.getenv("SCHEMA", "gold")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/landing"
DECISIONS_TABLE = f"{CATALOG}.{SCHEMA}.purchase_decisions"

LINE_ITEM_SCHEMA = ("{item_description, make_brand, model_no, qty (int), unit_rate (number, INR), "
                    "warranty_months (int), amc_present (bool), foc_present (bool), "
                    "delivery_lead_days (int), payment_terms (string), has_training (bool), "
                    "has_installation (bool)}")


@st.cache_resource
def wc() -> WorkspaceClient:
    return WorkspaceClient()


def run_sql(statement: str):
    """Run a SQL statement on the app's SQL warehouse; return list-of-dict rows."""
    r = wc().statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout="50s")
    if r.status and r.status.state.value not in ("SUCCEEDED",):
        raise RuntimeError(getattr(r.status.error, "message", r.status.state.value))
    cols = [c.name for c in r.manifest.schema.columns] if r.manifest and r.manifest.schema else []
    data = r.result.data_array if (r.result and r.result.data_array) else []
    return [dict(zip(cols, row)) for row in data]


def parse_pdf(file_bytes: bytes) -> list[dict]:
    """Upload the PDF to the UC Volume, parse with ai_parse_document, extract line items via ai_query."""
    path = f"{VOLUME_PATH}/{uuid.uuid4().hex}.pdf"
    wc().files.upload(path, io.BytesIO(file_bytes), overwrite=True)
    stmt = f"""
      WITH raw AS (SELECT content FROM read_files('{path}', format => 'binaryFile')),
      parsed AS (
        SELECT concat_ws('\\n', transform(ai_parse_document(content):document:elements,
                                           e -> e:content::string)) AS txt
        FROM raw)
      SELECT ai_query('{PARSE_LLM}',
        concat('Extract every quotation line item as a JSON object with key "line_items" = array of ',
               '{LINE_ITEM_SCHEMA}. Booleans reflect whether the quote includes AMC/CMC, FOC accessories, ',
               'training, installation. Text:\\n', txt),
        responseFormat => '{{"type":"json_object"}}') AS extracted
      FROM parsed
    """
    rows = run_sql(stmt)
    if not rows:
        return []
    obj = json.loads(rows[0]["extracted"])
    return obj.get("line_items", [])


def ask_agent(messages: list[dict]) -> str:
    """Send the full multi-turn conversation to the agent endpoint (OpenAI-compatible)."""
    client = wc().serving_endpoints.get_open_ai_client()
    resp = client.chat.completions.create(model=AGENT_ENDPOINT, messages=messages, max_tokens=2000)
    return resp.choices[0].message.content


def save_decision(item: str, vendor: str, price: float, rationale: str):
    run_sql(f"""CREATE TABLE IF NOT EXISTS {DECISIONS_TABLE}
        (decided_at TIMESTAMP, item_description STRING, chosen_vendor STRING,
         agreed_unit_rate DOUBLE, rationale STRING, decided_by STRING)""")
    who = (wc().current_user.me().user_name or "app").replace("'", "")
    run_sql(f"""INSERT INTO {DECISIONS_TABLE} VALUES (current_timestamp(),
        '{item.replace("'","")}', '{vendor.replace("'","")}', {float(price)},
        '{rationale.replace("'","")}', '{who}')""")


def quote_to_message(lines: list[dict]) -> str:
    return ("Here is a vendor quotation to review. For each line item, assess price fairness, list "
            "gaps, show cross-unit history, recommend a vendor, and give an overall recommendation.\n\n"
            f"Line items (JSON):\n{json.dumps(lines, indent=2)}")


# ---------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content}]

st.title("🏥 CAPEX Quotation Assistant")
st.caption(f"Conversational gap-detection over Kauvery's purchase history · agent: `{AGENT_ENDPOINT}` · "
           "all processing stays inside Databricks (no external egress).")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📄 Upload a quotation")
    up = st.file_uploader("Vendor quotation PDF", type=["pdf"])
    if up is not None and st.button("Analyze quotation", type="primary", use_container_width=True):
        with st.spinner("Parsing PDF with Databricks AI functions…"):
            try:
                lines = parse_pdf(up.getvalue())
            except Exception as e:  # noqa: BLE001
                lines = []
                st.warning(f"PDF parse unavailable ({e}). You can describe the item in the chat instead.")
        if lines:
            st.session_state.pending_quote = lines
            st.success(f"Extracted {len(lines)} line item(s). Sending to the assistant…")

    st.divider()
    st.header("✅ Record final decision")
    with st.form("decision"):
        d_item = st.text_input("Item")
        d_vendor = st.text_input("Chosen vendor")
        d_price = st.number_input("Agreed unit rate (INR)", min_value=0.0, step=1000.0)
        d_note = st.text_area("Rationale", height=80)
        if st.form_submit_button("Save decision") and d_item and d_vendor:
            try:
                save_decision(d_item, d_vendor, d_price, d_note)
                st.success(f"Saved to {DECISIONS_TABLE}. (Only the final decision is stored — not the quotation.)")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not save: {e}")

    st.divider()
    if st.button("Draft a vendor query letter", use_container_width=True):
        st.session_state.pending_prompt = ("Draft a short, professional letter to the vendor asking them "
                                            "to clarify or include the missing items you identified.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------------------------------------------------------- turn handling
prompt = st.chat_input("Ask about a quotation, a vendor, or cross-unit pricing…")

# a queued quote upload or quick-action becomes the next user turn
if not prompt and st.session_state.get("pending_quote"):
    prompt = quote_to_message(st.session_state.pop("pending_quote"))
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = ask_agent(st.session_state.messages)
            except Exception as e:  # noqa: BLE001
                answer = f"Sorry — I couldn't reach the agent endpoint `{AGENT_ENDPOINT}`: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
