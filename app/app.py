"""CAPEX Quotation Assistant — conversational quotation review for procurement teams.

A polished chat app: upload a vendor quotation, get a fair-price verdict, missing-item gaps,
cross-site pricing, and a vendor recommendation, then ask follow-ups. Implementation details
(models, endpoints, platform) are intentionally abstracted away from the UI.
"""
import io
import json
import re
import uuid

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

st.set_page_config(page_title="CAPEX Quotation Assistant", page_icon="📋", layout="centered",
                   initial_sidebar_state="collapsed")

# ---- runtime config (kept out of the UI) ----
import os
CHAT_LLM = os.getenv("CHAT_LLM", "databricks-llama-4-maverick")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "capex-worth-it")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
CATALOG = os.getenv("CATALOG", "kauvey_poc")
SCHEMA = os.getenv("SCHEMA", "gold")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/landing"
DECISIONS_TABLE = f"{CATALOG}.{SCHEMA}.purchase_decisions"

WELCOME = (
    "👋 <b>Welcome.</b> Upload a vendor quotation on the left — or just describe it in the chat — and "
    "I'll tell you whether the price is fair, flag anything missing, compare it against your past "
    "purchases across sites, and suggest how to negotiate. Your quotation data stays private and in "
    "your control.")

CHIPS = ["Is this price fair?", "Which vendor should we pick?",
         "What's missing from this quote?", "Show cross-site prices"]

CSS = """
<style>
/* Genie-style: restrained neutral surfaces, one calm blue accent, readable everywhere. */
:root{
  --bg:#0E1117; --surface:#161A21; --surface2:#1C222B; --border:#2A313C;
  --text:#E6E8EB; --muted:#8B95A3; --accent:#4C8DFF; --accent-weak:rgba(76,141,255,.12);
  --radius:12px;
}
[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,sans-serif;}
.block-container{padding-top:2.4rem;padding-bottom:7rem;max-width:840px;}
h1{font-size:1.55rem;font-weight:650;letter-spacing:-.2px;color:var(--text);margin-bottom:.2rem;}
h2,h3{color:var(--text);font-weight:600;}
p{color:var(--muted);line-height:1.6;}
footer,[data-testid="stToolbar"]{display:none;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border);}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{
  font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);font-weight:600;}

/* file uploader */
[data-testid="stFileUploaderDropzone"]{background:var(--surface2);border:1px dashed var(--border);
  border-radius:var(--radius);transition:all .15s ease;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);background:var(--accent-weak);}

/* chat — clean thread, no loud bubbles */
[data-testid="stChatMessage"]{background:transparent;border:none;border-bottom:1px solid var(--border);
  border-radius:0;padding:1rem .25rem;margin:0;box-shadow:none;}
[data-testid="stChatMessage"] .stMarkdown{color:var(--text);line-height:1.65;}
[data-testid="stChatMessage"] table{width:100%;border-collapse:collapse;font-size:.85rem;margin:.5rem 0;}
[data-testid="stChatMessage"] th,[data-testid="stChatMessage"] td{
  border:1px solid var(--border);padding:.4rem .6rem;text-align:left;}
[data-testid="stChatMessage"] th{background:var(--surface2);color:var(--text);font-weight:600;}

/* chat input — clean rounded bar */
[data-testid="stChatInput"]{background:var(--surface2);border:1px solid var(--border);
  border-radius:14px;box-shadow:0 4px 16px rgba(0,0,0,.3);}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent);
  box-shadow:0 0 0 2px var(--accent-weak);}
[data-testid="stChatInput"] textarea::placeholder{color:var(--muted);}

/* primary button (Analyze) — solid accent, white text */
button[kind="primary"],button[kind="primaryFormSubmit"]{background:var(--accent);color:#fff;
  border:none;border-radius:10px;font-weight:600;box-shadow:none;}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover{background:#3B7BEE;color:#fff;}

/* default buttons (chips, sidebar actions) — subtle outlined pills, readable */
.stButton>button{background:var(--surface2);color:var(--text);border:1px solid var(--border);
  border-radius:10px;font-weight:500;transition:all .15s ease;box-shadow:none;padding:.5rem 1rem;}
.stButton>button:hover{background:var(--accent-weak);border-color:var(--accent);color:var(--text);}
[data-testid="stSidebar"] .stButton>button{width:100%;}

/* inputs */
input,textarea{background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;}
input:focus,textarea:focus{border-color:var(--accent)!important;box-shadow:0 0 0 2px var(--accent-weak)!important;}

/* welcome card */
.welcome{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:1.1rem 1.25rem;color:var(--muted);line-height:1.6;margin:.5rem 0 1rem;}
.welcome b{color:var(--text);}

::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-thumb{background:rgba(139,149,163,.3);border-radius:5px;}
::-webkit-scrollbar-thumb:hover{background:rgba(139,149,163,.55);}
</style>
"""

SYNTH_PROMPT = (
    "You are a procurement-intelligence assistant for a hospital group's capex team. You are given a "
    "quotation and pre-computed EVIDENCE (JSON): a fairness assessment (worth_score 0-100, a verdict of "
    "Accept/Negotiate/Reject, price_variance_pct vs the most-recent comparable purchase, and gaps = "
    "missing inclusions with source references), the item's purchase history across all sites "
    "(low-to-high), and a vendor value ranking (vendors bundling free-of-cost items + maintenance at a "
    "low price rank highest).\n\n"
    "Answer CONVERSATIONALLY, per line item:\n"
    "  1. Item -> quoted price.  2. Gaps found (with their source references).  3. Price fairness: the "
    "verdict and the % vs the most-recent comparable purchase (never say a vendor is 'overcharging').  "
    "4. A markdown table of purchases across sites (site, vendor, date, unit price, warranty, "
    "maintenance, free-of-cost) low-to-high.  5. Recommended vendor (prefer bundled free-of-cost + "
    "maintenance at a low price).  6. An overall actionable recommendation for negotiation.\n"
    "Use ONLY the evidence; never invent prices, vendors, dates, or specs. Do not mention models, "
    "endpoints, or the underlying platform.")

FOLLOWUP_PROMPT = (
    "You are a procurement-intelligence assistant for a hospital group's capex team. Answer the user's "
    "follow-up using the conversation and the EVIDENCE already gathered (JSON below). Be concise and "
    "management-oriented; use only the evidence. If no quotation has been analysed yet, invite the user "
    "to upload one. If asked to draft a vendor letter, write a short professional letter citing the "
    "specific missing items. Never mention models, endpoints, or the underlying platform.")

ROUTE_PROMPT = (
    "You route messages for a hospital procurement assistant. Return ONLY a JSON object with two keys:\n"
    " - line_items: array of {item_description, make_brand, model_no, qty (int), unit_rate (number), "
    "warranty_months (int), amc_present (bool), foc_present (bool), delivery_lead_days (int), "
    "payment_terms (string), has_training (bool), has_installation (bool)} — fill ONLY when the user is "
    "giving a specific quotation to evaluate (it has a price). Booleans reflect whether the quote "
    "includes maintenance (AMC/CMC), free-of-cost items, training, installation.\n"
    " - search_terms: array of short item/brand/model keywords to look up when the user asks a general "
    "question about pricing, vendors, or trends (e.g. [\"patient monitor\"] or [\"CT scanner\"]).\n"
    "A quotation -> fill line_items, empty search_terms. A data question -> fill search_terms, empty "
    "line_items. A general follow-up (draft a letter, thanks, etc.) -> both empty. Return ONLY JSON.")

DATAQ_PROMPT = (
    "You are a procurement-intelligence assistant for a hospital group's capex team. Answer the user's "
    "question using the EVIDENCE (JSON) below — purchase history across sites (prices, warranty, "
    "maintenance, free-of-cost) and vendor value rankings for the relevant items. Be concise and "
    "management-oriented: surface price ranges/trends, cheapest vs most-recent, and the best-value "
    "vendor; use a small markdown table or bullets. Use ONLY the evidence; never invent data, and never "
    "mention models, endpoints, or the underlying platform.")

_ROLE = {"system": ChatMessageRole.SYSTEM, "user": ChatMessageRole.USER,
         "assistant": ChatMessageRole.ASSISTANT}


@st.cache_resource
def wc() -> WorkspaceClient:
    return WorkspaceClient()


def _chat(messages, max_tokens=2000) -> str:
    msgs = [ChatMessage(role=_ROLE.get(m["role"], ChatMessageRole.USER), content=m["content"])
            for m in messages]
    r = wc().serving_endpoints.query(name=CHAT_LLM, messages=msgs, max_tokens=max_tokens)
    return (r.choices[0].message.content or "") if r.choices else ""


def _parse_json(txt: str) -> dict:
    txt = re.sub(r"^```[a-zA-Z]*|```$", "", txt.strip()).strip()
    s, e = txt.find("{"), txt.rfind("}")
    try:
        return json.loads(txt[s:e + 1]) if s >= 0 and e > s else {}
    except Exception:  # noqa: BLE001
        return {}


def run_sql(statement: str):
    r = wc().statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout="50s")
    if r.status and r.status.state.value != "SUCCEEDED":
        raise RuntimeError(getattr(r.status.error, "message", r.status.state.value))
    return [row for row in (r.result.data_array or [])] if r.result else []


def exec_price_fairness(**kw) -> dict:
    rec = {"item_description": kw.get("item_description"), "make_brand": kw.get("make_brand"),
           "model_no": kw.get("model_no"), "qty": int(kw.get("qty", 1) or 1),
           "unit_rate": float(kw.get("unit_rate", 0) or 0),
           "warranty_months": int(kw.get("warranty_months", 0) or 0),
           "amc_present": bool(kw.get("amc_present", False)), "camc_present": False,
           "foc_present": bool(kw.get("foc_present", False)),
           "delivery_lead_days": int(kw.get("delivery_lead_days", 60) or 60),
           "payment_terms": kw.get("payment_terms", ""), "has_training": bool(kw.get("has_training", False)),
           "has_installation": bool(kw.get("has_installation", True))}
    resp = wc().serving_endpoints.query(name=MODEL_ENDPOINT, dataframe_records=[rec])
    return resp.predictions[0] if resp.predictions else {}


def exec_sql_fn(fn: str, search: str):
    safe = (search or "").replace("'", "")
    rows = run_sql(f"SELECT {CATALOG}.{SCHEMA}.{fn}('{safe}') AS r")
    try:
        return json.loads(rows[0][0]) if rows and rows[0] and rows[0][0] else []
    except Exception:  # noqa: BLE001
        return []


def gather_evidence(lines: list[dict]) -> list[dict]:
    ev = []
    for ln in lines:
        search = ln.get("model_no") or ln.get("item_description") or ""
        try:
            pf = exec_price_fairness(**ln)
        except Exception as e:  # noqa: BLE001
            pf = {"error": str(e)}
        ev.append({"line": ln, "fairness": pf,
                   "cross_site_history": exec_sql_fn("cross_unit_history", search),
                   "vendor_ranking": exec_sql_fn("recommend_vendor", search)})
    return ev


def route(text: str):
    """Classify a message -> (line_items to evaluate, search_terms for a data question)."""
    try:
        j = _parse_json(_chat([{"role": "system", "content": ROUTE_PROMPT},
                               {"role": "user", "content": text}], max_tokens=700))
        return (j.get("line_items") or []), (j.get("search_terms") or [])
    except Exception:  # noqa: BLE001
        return [], []


def handle_turn(user_text: str, pending_lines) -> str:
    hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]
    lines, terms = (pending_lines, []) if pending_lines is not None else route(user_text)

    if lines:  # a quotation to evaluate
        evidence = gather_evidence(lines)
        st.session_state.evidence = evidence
        msgs = [{"role": "system", "content": SYNTH_PROMPT}] + hist + [
            {"role": "user", "content": f"{user_text}\n\nEVIDENCE:\n{json.dumps(evidence, default=str)}"}]
        return _chat(msgs)

    if terms:  # a general data question about an item / vendor / trend
        ev = [{"search": t, "cross_site_history": exec_sql_fn("cross_unit_history", t),
               "vendor_ranking": exec_sql_fn("recommend_vendor", t)} for t in terms[:3]]
        st.session_state.evidence = ev
        msgs = [{"role": "system", "content": DATAQ_PROMPT}] + hist + [
            {"role": "user", "content": f"{user_text}\n\nEVIDENCE:\n{json.dumps(ev, default=str)}"}]
        return _chat(msgs)

    # general follow-up — use whatever evidence is already on the table
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
               'Booleans reflect whether the quote includes maintenance (AMC/CMC), free-of-cost items, ',
               'training, installation. Text:\\n', txt),
        responseFormat => '{{"type":"json_object"}}') AS extracted FROM parsed"""
    rows = run_sql(stmt)
    return _parse_json(rows[0][0]).get("line_items", []) if rows and rows[0] else []


def save_decision(item, vendor, price, rationale):
    run_sql(f"""CREATE TABLE IF NOT EXISTS {DECISIONS_TABLE}
        (decided_at TIMESTAMP, item_description STRING, chosen_vendor STRING,
         agreed_unit_rate DOUBLE, rationale STRING, decided_by STRING)""")
    who = (wc().current_user.me().user_name or "app").replace("'", "")
    run_sql(f"""INSERT INTO {DECISIONS_TABLE} VALUES (current_timestamp(),
        '{item.replace("'","")}','{vendor.replace("'","")}',{float(price)},
        '{rationale.replace("'","")}','{who}')""")


# ---------------------------------------------------------------- UI
st.markdown(CSS, unsafe_allow_html=True)
if "history" not in st.session_state:
    st.session_state.history = []

# Slim, collapsed-by-default sidebar: new conversation + a tucked-away decision recorder.
with st.sidebar:
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.pop("evidence", None)
        st.rerun()
    with st.expander("Record final decision"):
        with st.form("decision"):
            d_item = st.text_input("Item"); d_vendor = st.text_input("Chosen vendor")
            d_price = st.number_input("Agreed unit price (INR)", min_value=0.0, step=1000.0)
            d_note = st.text_area("Rationale", height=70)
            if st.form_submit_button("Save decision") and d_item and d_vendor:
                try:
                    save_decision(d_item, d_vendor, d_price, d_note)
                    st.success("Saved — only the final decision is kept, not the quotation.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not save: {e}")

st.title("📋 CAPEX Quotation Assistant")
st.markdown("Ask about an instrument, a vendor, or a price trend — or attach a quotation to compare it "
            "against your purchase history.")

# empty state: quiet welcome + suggested-question pills
if not st.session_state.history:
    st.markdown(f'<div class="welcome">{WELCOME}</div>', unsafe_allow_html=True)
    st.caption("Try asking")
    cols = st.columns(len(CHIPS))
    for c, chip in zip(cols, CHIPS):
        if c.button(chip, use_container_width=True):
            st.session_state.pending_text = chip
            st.rerun()

for m in st.session_state.history:
    with st.chat_message(m["role"], avatar="📋" if m["role"] == "assistant" else "🧑‍⚕️"):
        st.markdown(m["content"])

# Single Genie-style input: type a question, or use the paperclip to attach a quotation PDF.
ci = st.chat_input("Message the assistant — or attach a quotation PDF", accept_file=True, file_type=["pdf"])

prompt, files, pending_lines = None, [], None
if ci is not None:
    if isinstance(ci, str):                      # older Streamlit: plain text only
        prompt = ci.strip()
    else:                                        # ChatInputValue: .text + .files
        prompt = (getattr(ci, "text", "") or "").strip()
        files = list(getattr(ci, "files", []) or [])
elif st.session_state.get("pending_text"):       # a suggestion chip was clicked
    prompt = st.session_state.pop("pending_text")
    pending_lines = st.session_state.pop("pending_lines", None)

# An attached PDF becomes a quotation to review.
attach_note = ""
if files:
    with st.spinner("Reading your quotation…"):
        try:
            lines = parse_pdf(files[0].getvalue())
        except Exception:  # noqa: BLE001
            lines = []
    if lines:
        pending_lines = lines
        names = ", ".join(f"{l.get('qty','')}× {l.get('item_description','item')}" for l in lines)
        attach_note = f"📎 {files[0].name}"
        prompt = prompt or f"Please review this quotation: {names}."
    else:
        prompt = prompt or "I attached a quotation but it couldn't be read automatically."

if prompt:
    shown = f"{attach_note}\n\n{prompt}" if attach_note else prompt
    st.session_state.history.append({"role": "user", "content": shown})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(shown)
    with st.chat_message("assistant", avatar="📋"):
        with st.spinner("Comparing against your purchase history…"):
            try:
                answer = handle_turn(prompt, pending_lines)
            except Exception as e:  # noqa: BLE001
                answer = f"Sorry — something went wrong: {e}"
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
