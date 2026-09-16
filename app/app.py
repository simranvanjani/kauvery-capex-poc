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

CHIPS = ["📈 Patient monitor price trend", "🏆 Best vendor for ventilators",
         "📊 Cross-site CT scanner prices", "💡 What can you do?"]

CSS = """
<style>
/* Modelled on Databricks Genie One: near-black canvas, soft glow, centered hero, icon pills. */
:root{
  --bg:#0B0C0E; --surface:#161719; --surface2:#1C1E22; --border:#2A2D33;
  --text:#ECEDEE; --muted:#8A9099; --accent:#5B8DEF; --accent-weak:rgba(91,141,239,.14);
  --radius:14px;
}
[data-testid="stAppViewContainer"]{
  background:radial-gradient(900px 420px at 50% -60px, rgba(124,92,255,.10), transparent 70%), var(--bg);
  color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,sans-serif;}
.block-container{padding-top:3rem;padding-bottom:7rem;max-width:820px;}
h1,h2,h3{color:var(--text);font-weight:600;letter-spacing:-.2px;}
p{color:var(--muted);line-height:1.6;}
footer,[data-testid="stToolbar"],[data-testid="stHeader"]{display:none;}

/* sidebar */
[data-testid="stSidebar"]{background:#0E0F11;border-right:1px solid var(--border);}
[data-testid="stSidebar"] .stExpander{border:1px solid var(--border);border-radius:10px;background:var(--surface);}

/* ---- centered hero (empty state) ---- */
.hero{text-align:center;margin:7vh auto 1.6rem;}
.hero .tile{width:60px;height:60px;border-radius:18px;margin:0 auto 1.4rem;display:flex;
  align-items:center;justify-content:center;font-size:28px;
  background:linear-gradient(150deg,#2A2340,#17181C);border:1px solid var(--border);
  box-shadow:0 10px 40px rgba(124,92,255,.28);}
.hero .h{font-size:2.1rem;font-weight:600;color:#F4F5F6;margin:0;}
.hero .sub{color:var(--muted);margin:.6rem auto 0;max-width:560px;font-size:.98rem;}

/* chat — clean thread */
[data-testid="stChatMessage"]{background:transparent;border:none;border-bottom:1px solid var(--border);
  border-radius:0;padding:1.1rem .25rem;margin:0;box-shadow:none;}
[data-testid="stChatMessage"] .stMarkdown{color:var(--text);line-height:1.65;}
[data-testid="stChatMessage"] table{width:100%;border-collapse:collapse;font-size:.85rem;margin:.5rem 0;}
[data-testid="stChatMessage"] th,[data-testid="stChatMessage"] td{
  border:1px solid var(--border);padding:.42rem .6rem;text-align:left;}
[data-testid="stChatMessage"] th{background:var(--surface2);color:var(--text);font-weight:600;}

/* chat input — large elevated rounded card, like Genie's ask box */
[data-testid="stChatInput"]{background:var(--surface);border:1px solid var(--border);
  border-radius:16px;box-shadow:0 10px 34px rgba(0,0,0,.45);padding:.35rem .5rem;}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-weak),0 10px 34px rgba(0,0,0,.45);}
[data-testid="stChatInput"] textarea{font-size:1rem;}
[data-testid="stChatInput"] textarea::placeholder{color:var(--muted);}

/* primary button */
button[kind="primary"],button[kind="primaryFormSubmit"]{background:var(--accent);color:#fff;border:none;
  border-radius:10px;font-weight:600;box-shadow:none;}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover{background:#4A7CE0;color:#fff;}

/* action pills (suggested prompts) — dark rounded pills with icon, readable */
.stButton>button{background:var(--surface);color:var(--text);border:1px solid var(--border);
  border-radius:11px;font-weight:500;transition:all .15s ease;box-shadow:none;padding:.6rem .9rem;
  font-size:.9rem;}
.stButton>button:hover{background:var(--surface2);border-color:#3A3E46;color:#fff;}
[data-testid="stSidebar"] .stButton>button{width:100%;}

/* inputs */
input,textarea{background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;}
input:focus,textarea:focus{border-color:var(--accent)!important;box-shadow:0 0 0 2px var(--accent-weak)!important;}

::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-thumb{background:rgba(138,144,153,.3);border-radius:5px;}
::-webkit-scrollbar-thumb:hover{background:rgba(138,144,153,.55);}
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
    "4. Purchases across sites — render as a STANDALONE markdown table with a blank line before and "
    "after it (never indent it inside the numbered list), columns Site | Vendor | Date | Unit Price | "
    "Warranty | Maintenance | Free-of-Cost, sorted low-to-high. If the cross-site history is empty, "
    "write one short line that no comparable cross-site purchases were found and DO NOT print an empty "
    "table.  5. Recommended vendor (prefer bundled free-of-cost + maintenance at a low price).  "
    "6. An overall actionable recommendation for negotiation.\n"
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
    " - search_terms: EQUIPMENT names only (item / brand / model), e.g. [\"patient monitor\"], "
    "[\"ventilator\"], [\"GE CT\"]. Use this ONLY when the user names a NEW piece of equipment to look "
    "up. NEVER put attributes here (foc, free-of-cost, amc, maintenance, warranty, price, training, "
    "delivery, payment, vendor, unit/site).\n"
    "Rules: a quotation -> line_items only. A question naming a NEW equipment type -> search_terms only. "
    "Anything about the quotation/items already discussed, or about an attribute of them (e.g. 'which "
    "site has the best FOC', 'compare their warranties', 'draft a letter') -> BOTH arrays empty (it's a "
    "follow-up answered from existing data). Return ONLY the JSON object.")

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
        model_search = ln.get("model_no") or ln.get("item_description") or ""
        try:
            pf = exec_price_fairness(**ln)
        except Exception as e:  # noqa: BLE001
            pf = {"error": str(e)}
        # category (from the model output) so vendor comparison spans ALL vendors of this
        # equipment type, not just the ones selling this exact model.
        category = (pf.get("category") if isinstance(pf, dict) else None) or model_search
        # history for the exact model; if that model string isn't in our records, widen to the category
        history = exec_sql_fn("cross_unit_history", model_search)
        if not history and category and category != model_search:
            history = exec_sql_fn("cross_unit_history", category)
        ev.append({"line": ln, "category": category, "fairness": pf,
                   "cross_site_history": history,
                   "category_vendor_ranking": exec_sql_fn("recommend_vendor", category)})
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

    if terms:  # a data question naming an equipment type
        ev = [{"search": t, "cross_site_history": exec_sql_fn("cross_unit_history", t),
               "vendor_ranking": exec_sql_fn("recommend_vendor", t)} for t in terms[:3]]
        if any(r["cross_site_history"] or r["vendor_ranking"] for r in ev):
            st.session_state.evidence = ev
            msgs = [{"role": "system", "content": DATAQ_PROMPT}] + hist + [
                {"role": "user", "content": f"{user_text}\n\nEVIDENCE:\n{json.dumps(ev, default=str)}"}]
            return _chat(msgs)
        # nothing matched (e.g. mis-routed attribute) -> answer from existing evidence below

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
    # don't retain the quotation: remove the uploaded PDF from the volume once parsed
    try:
        wc().files.delete(path)
    except Exception:  # noqa: BLE001
        pass
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

# empty-state hero (Genie-style: icon tile → heading → suggestion pills)
if not st.session_state.history:
    st.markdown(
        '<div class="hero"><div class="tile">✦</div>'
        '<div class="h">How can I help you?</div>'
        '<div class="sub">Ask about an instrument, a vendor, or a price trend — or attach a vendor '
        "quotation and I'll compare it against your purchase history.</div></div>",
        unsafe_allow_html=True)
    cols = st.columns(len(CHIPS))
    for c, chip in zip(cols, CHIPS):
        if c.button(chip, use_container_width=True):
            st.session_state.history.append({"role": "user", "content": chip})
            st.session_state.pending = {"text": chip, "file": None, "name": None}
            st.rerun()

# render the conversation so far
for m in st.session_state.history:
    with st.chat_message(m["role"], avatar="📋" if m["role"] == "assistant" else "🧑‍⚕️"):
        st.markdown(m["content"])

# process a pending user turn (the user's message is already rendered above)
if st.session_state.get("pending"):
    p = st.session_state.pop("pending")
    with st.chat_message("assistant", avatar="📋"):
        with st.spinner("Working on it…"):
            text, pending_lines = (p.get("text") or ""), None
            if p.get("file"):
                try:
                    pending_lines = parse_pdf(p["file"]) or None
                except Exception:  # noqa: BLE001
                    pending_lines = None
                if pending_lines:
                    names = ", ".join(f"{l.get('qty','')}× {l.get('item_description','item')}"
                                      for l in pending_lines)
                    text = text or f"Please review this quotation: {names}."
                elif not text:
                    text = "I attached a quotation but it couldn't be read automatically."
            try:
                answer = handle_turn(text, pending_lines)
            except Exception as e:  # noqa: BLE001
                answer = f"Sorry — something went wrong: {e}"
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()

# input: type a question, or use the ＋ paperclip to attach a quotation PDF
ci = st.chat_input("Message the assistant — or attach a quotation PDF", accept_file=True, file_type=["pdf"])
if ci is not None:
    if isinstance(ci, str):                       # older Streamlit fallback: text only
        text, fbytes, fname = ci.strip(), None, None
    else:                                         # ChatInputValue: .text + .files
        text = (getattr(ci, "text", "") or "").strip()
        fs = list(getattr(ci, "files", []) or [])
        fbytes = fs[0].getvalue() if fs else None
        fname = fs[0].name if fs else None
    if text or fbytes:
        shown = (f"📎 {fname}" + (f"\n\n{text}" if text else "")) if fname else text
        st.session_state.history.append({"role": "user", "content": shown})
        st.session_state.pending = {"text": text, "file": fbytes, "name": fname}
        st.rerun()
