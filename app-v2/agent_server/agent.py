import json
import logging
import os
from contextlib import AsyncExitStack
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.history import normalize_history_items
from agent_server.utils import get_session_id, process_agent_stream_events

logger = logging.getLogger(__name__)

# NOTE: this will work for all databricks models OTHER than GPT-OSS, which uses a slightly different API
set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

# ---- CAPEX config -----------------------------------------------------------------
# CHAT_MODEL is the conversational FM. Swap it to any Databricks-hosted model
# (databricks-llama-4-maverick / a Claude / a GPT endpoint) — the code is model-agnostic.
CHAT_MODEL = os.getenv("CHAT_MODEL", "databricks-llama-4-maverick")
# PDF extraction runs ai_query with responseFormat=json_object, which Claude does NOT support
# ("responseFormat is invalid or unsupported by the model"). Keep a Llama/GPT model here even
# when the agent's CHAT_MODEL is Claude.
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "databricks-llama-4-maverick")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "capex-worth-it")  # custom scikit-learn scorer
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
CATALOG = os.getenv("CATALOG", "kauvey_poc")
SCHEMA = os.getenv("SCHEMA", "gold")

_WC: WorkspaceClient | None = None


def wc() -> WorkspaceClient:
    """App service-principal client. Tool calls (FM parse, model endpoint, UC functions) run as the
    SP so they don't depend on per-user grants; per-user isolation of history/feedback is handled by
    the frontend/Node server + Lakebase auth."""
    global _WC
    if _WC is None:
        _WC = WorkspaceClient()
    return _WC


def _run_sql(statement: str) -> list:
    r = wc().statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout="50s"
    )
    if r.status and r.status.state.value != "SUCCEEDED":
        raise RuntimeError(getattr(r.status.error, "message", r.status.state.value))
    return list(r.result.data_array or []) if r.result else []


def _uc_fn(fn: str, search: str):
    safe = (search or "").replace("'", "")
    try:
        rows = _run_sql(f"SELECT {CATALOG}.{SCHEMA}.{fn}('{safe}') AS r")
    except Exception as e:  # DB / warehouse unreachable or query failed — surface it, don't hide
        return {"error": f"could not reach the history database ({CATALOG}.{SCHEMA}): {e}"}
    try:
        return json.loads(rows[0][0]) if rows and rows[0] and rows[0][0] else []
    except Exception:  # noqa: BLE001
        return []


# ---- CAPEX tools ------------------------------------------------------------------
@function_tool
def parse_quotation_pdf(volume_path: str) -> str:
    """Parse a vendor quotation PDF of ANY format into structured line items.

    `volume_path` is a UC Volume path to the uploaded PDF
    (e.g. /Volumes/<catalog>/<schema>/landing/<file>.pdf). Vendor layouts differ wildly; this uses
    ai_parse_document + an LLM extraction so the SAME details are captured regardless of format.
    Returns a JSON object {"line_items": [ {item_description, make_brand, model_no, qty, unit_rate,
    warranty_months, amc_present, foc_present, delivery_lead_days, payment_terms, has_training,
    has_installation}, ... ]}."""
    item_schema = (
        "{item_description, make_brand, model_no, qty (int), unit_rate (number), "
        "warranty_months (int), amc_present (bool), foc_present (bool), delivery_lead_days (int), "
        "payment_terms (string), has_training (bool), has_installation (bool)}"
    )
    stmt = f"""
      WITH raw AS (SELECT content FROM read_files('{volume_path}', format => 'binaryFile')),
      parsed AS (SELECT concat_ws('\\n', transform(
                          cast(ai_parse_document(content):document:elements AS ARRAY<VARIANT>),
                          e -> e:content::string)) AS txt FROM raw)
      SELECT ai_query('{EXTRACT_MODEL}',
        concat('Extract EVERY quotation line item as JSON with key "line_items" = array of {item_schema}. ',
               'The PDF layout is arbitrary and vendor-specific — find the same fields regardless of format. ',
               'Booleans reflect whether the quote includes maintenance (AMC/CMC), free-of-cost items, ',
               'training, installation. Text:\\n', txt),
        responseFormat => '{{"type":"json_object"}}') AS extracted FROM parsed"""
    rows = _run_sql(stmt)
    return rows[0][0] if rows and rows[0] else '{"line_items": []}'


@function_tool
def score_line_item(
    item_description: str,
    make_brand: str,
    model_no: str,
    qty: int,
    unit_rate: float,
    warranty_months: int,
    amc_present: bool,
    foc_present: bool,
    delivery_lead_days: int,
    payment_terms: str,
    has_training: bool = False,
    has_installation: bool = True,
) -> str:
    """Score ONE quotation line item with the custom CAPEX ML model (capex-worth-it endpoint).

    Returns JSON with worth_score (0-100), verdict (Accept/Negotiate/Reject), price_variance_pct vs the
    most-recent comparable purchase, match_level, benchmark_unit_rate, and gaps (missing inclusions with
    citations). Call once PER line item."""
    rec = {
        "item_description": item_description, "make_brand": make_brand, "model_no": model_no,
        "qty": int(qty), "unit_rate": float(unit_rate), "warranty_months": int(warranty_months),
        "amc_present": bool(amc_present), "camc_present": False, "foc_present": bool(foc_present),
        "delivery_lead_days": int(delivery_lead_days), "payment_terms": payment_terms,
        "has_training": bool(has_training), "has_installation": bool(has_installation),
    }
    try:
        resp = wc().serving_endpoints.query(name=MODEL_ENDPOINT, dataframe_records=[rec])
    except Exception as e:  # scoring endpoint unreachable — surface it, don't fabricate a score
        return json.dumps({"error": f"could not reach the scoring model '{MODEL_ENDPOINT}': {e}"})
    return json.dumps(resp.predictions[0] if resp.predictions else {})


@function_tool
def cross_unit_history(search: str) -> str:
    """Historical purchases of a matching item across ALL Kauvery units, cheapest first, as JSON.
    Use the item / model / brand as `search`."""
    return json.dumps(_uc_fn("cross_unit_history", search))


@function_tool
def recommend_vendor(search: str) -> str:
    """Vendors ranked by value-for-money for a matching item (bundled FOC + AMC at a low price rank
    highest), as JSON. Use the item / model / brand as `search`."""
    return json.dumps(_uc_fn("recommend_vendor", search))


CAPEX_INSTRUCTIONS = (
    "You are the Kauvery Hospital CAPEX Procurement Intelligence assistant. You review vendor "
    "quotations against Kauvery's own purchase history and help the capex team negotiate better deals.\n\n"
    "AUTO-REVIEW — IMPORTANT: as soon as the user provides a quotation — whether an uploaded PDF (you will "
    "be given its UC Volume path) OR typed / pasted line items OR free text describing the quote — "
    "IMMEDIATELY review it. Do NOT ask clarifying questions first; extract what you can and run the review.\n\n"
    "Workflow per quotation:\n"
    "1. If given a PDF volume path, call parse_quotation_pdf to extract line items (any vendor format).\n"
    "2. For EACH line item, call score_line_item, then cross_unit_history, then recommend_vendor. When "
    "calling cross_unit_history / recommend_vendor pass a SHORT, general search — the equipment type or "
    "make/brand (e.g. 'ventilator', 'syringe pump', 'CT scanner', 'Draeger'), NOT the full line-item "
    "description: the history is matched as a substring, so a broad term is what finds comparable purchases.\n"
    "3. Present, per line item: item -> quoted price; gaps found (each with its citation); price fairness "
    "(verdict + % vs the most-recent comparable purchase — say 'X% above/below the most-recent comparable "
    "purchase', never 'overcharging'); a markdown cross-site history table (unit, vendor, date, unit price, "
    "warranty, maintenance, FOC, cheapest first); the recommended vendor; and an actionable negotiation ask.\n\n"
    "NEW ITEM (no Kauvery history): if score_line_item returns match_level 'No historical purchase found' "
    "or cross_unit_history returns an empty list, show a clear warning — **⚠️ New product: no Kauvery "
    "purchase history for this item, so there is no historical price benchmark** — and review only what the "
    "quotation itself provides (inclusions/gaps, warranty, AMC/CMC, FOC, payment/delivery terms). Do NOT "
    "provide a market-price estimate or any price benchmark you cannot ground in Kauvery's own history.\n\n"
    "DATA ERRORS: if a tool returns an object with an \"error\" key (e.g. the history database or scoring "
    "model is unreachable), STOP and tell the user plainly that the historical benchmark / score could NOT "
    "be retrieved because of that error, and show the error message. Do NOT fabricate history or a benchmark, "
    "and do NOT silently skip it or treat it as 'no history'.\n\n"
    "Ground every claim in tool output; never invent or estimate prices, vendors, dates, or specs — if you "
    "cannot ground a figure in Kauvery's history, say so rather than guessing."
)


def create_agent() -> Agent:
    return Agent(
        name="Kauvery CAPEX Assistant",
        instructions=CAPEX_INSTRUCTIONS,
        model=CHAT_MODEL,
        tools=[parse_quotation_pdf, score_line_item, cross_unit_history, recommend_vendor],
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    async with AsyncExitStack():
        agent = create_agent()
        messages = normalize_history_items([i.model_dump() for i in request.input])
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    async with AsyncExitStack():
        agent = create_agent()
        messages = normalize_history_items([i.model_dump() for i in request.input])
        result = Runner.run_streamed(agent, input=messages)
        async for event in process_agent_stream_events(result.stream_events()):
            yield event
