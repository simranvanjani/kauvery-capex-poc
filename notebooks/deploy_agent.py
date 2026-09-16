# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy the CAPEX conversational agent
# MAGIC
# MAGIC Wraps the registered ML model (via the `price_fairness` UC function) plus `cross_unit_history`
# MAGIC and `recommend_vendor` in a LangGraph **ResponsesAgent**, using a **Databricks-hosted Llama**
# MAGIC foundation model for reasoning + multi-turn conversation. Registers the agent to Unity Catalog
# MAGIC and deploys it as **its own serving endpoint** (`capex-agent`). The app calls this agent — never
# MAGIC the raw model.
# MAGIC
# MAGIC Run this AFTER `capex_phase2_demo` (which creates the model, endpoint, and UC functions).

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain langgraph==0.3.4 databricks-agents mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "kauvey_poc", "Catalog")
dbutils.widgets.text("schema", "gold", "Schema")
dbutils.widgets.text("llm_endpoint", "databricks-llama-4-maverick", "Reasoning LLM (Databricks-hosted)")
dbutils.widgets.text("agent_model", "capex_agent", "Agent UC model name")
dbutils.widgets.text("agent_endpoint", "capex-agent", "Agent serving endpoint name")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint").strip()
AGENT_MODEL = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('agent_model').strip()}"
AGENT_ENDPOINT = dbutils.widgets.get("agent_endpoint").strip()
MODEL_ENDPOINT = "capex-worth-it"
# price_fairness is called directly against the model endpoint (a Python tool) so the agent's
# downscoped passthrough token authorizes it as a declared serving-endpoint resource. The two
# pure-SQL functions stay as UC-function tools.
FUNCTIONS = [f"{CATALOG}.{SCHEMA}.cross_unit_history",
             f"{CATALOG}.{SCHEMA}.recommend_vendor"]
print("LLM:", LLM_ENDPOINT, "| agent:", AGENT_MODEL, "-> endpoint", AGENT_ENDPOINT)

# COMMAND ----------

import os, sys
SRC_DIR = "/tmp/capex_agent_src"
os.makedirs(SRC_DIR, exist_ok=True)

AGENT_CODE = r'''
"""CAPEX conversational gap-detection agent (LangGraph ResponsesAgent).

Reasoning LLM: a Databricks-hosted Llama foundation model (in-region, no external egress).
Tools (Unity Catalog functions): price_fairness (wraps the ML model), cross_unit_history,
recommend_vendor. Deterministic scoring lives in the model/functions; this agent adds the
multi-turn conversation and orchestration on top.
"""
import os
import json as _json
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent,
    output_to_responses_items_stream, to_chat_completions_input,
)
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from typing import Annotated, Generator, Sequence, TypedDict

# Config baked in at log time (env vars do NOT survive to the serving process).
LLM_ENDPOINT = "__LLM_ENDPOINT__"
MODEL_ENDPOINT = "__MODEL_ENDPOINT__"
UC_FUNCTIONS = __UC_FUNCTIONS__


@tool
def price_fairness(item_description: str, make_brand: str, model_no: str, qty: int, unit_rate: float,
                   warranty_months: int, amc_present: bool, foc_present: bool, delivery_lead_days: int,
                   payment_terms: str, has_training: bool = False, has_installation: bool = True) -> str:
    """Score ONE quotation line item with the CAPEX ML model. Returns JSON with worth_score (0-100),
    verdict (Accept/Negotiate/Reject), price_variance_pct vs the most-recent comparable purchase,
    match_level, benchmark_unit_rate, and gaps (missing inclusions with citations). Call once per line."""
    rec = {"item_description": item_description, "make_brand": make_brand, "model_no": model_no,
           "qty": int(qty), "unit_rate": float(unit_rate), "warranty_months": int(warranty_months),
           "amc_present": bool(amc_present), "camc_present": False, "foc_present": bool(foc_present),
           "delivery_lead_days": int(delivery_lead_days), "payment_terms": payment_terms,
           "has_training": bool(has_training), "has_installation": bool(has_installation)}
    resp = WorkspaceClient().serving_endpoints.query(name=MODEL_ENDPOINT, dataframe_records=[rec])
    return _json.dumps(resp.predictions[0] if resp.predictions else {})

SYSTEM_PROMPT = (
    "You are the CAPEX Procurement Intelligence assistant for Kauvery Hospital. You help the capex "
    "team review vendor quotations against Kauvery's own purchase history and negotiate better deals. "
    "You have three tools:\n"
    "  - price_fairness(...): scores ONE quotation line item with the ML model. Returns worth_score "
    "(0-100), verdict (Accept/Negotiate/Reject), price_variance_pct vs the most-recent comparable "
    "purchase, and gaps (missing inclusions with citations). Call it once PER line item, filling every "
    "argument from the quotation the user provides.\n"
    "  - cross_unit_history(search): historical purchases of the same item across ALL Kauvery units, "
    "sorted price low-to-high. Use the item/model name as `search`.\n"
    "  - recommend_vendor(search): vendors ranked by value-for-money (low price + bundled FOC + AMC). "
    "Vendors bundling FOC+AMC into a low price rank highest.\n\n"
    "When a user shares a quotation, for each line item: call price_fairness, then cross_unit_history, "
    "then recommend_vendor. Then answer CONVERSATIONALLY in this structure:\n"
    "  1. Item -> quoted price.\n"
    "  2. Gaps found (missing inclusions), each with its citation.\n"
    "  3. Price fairness: the verdict and % vs the most-recent comparable purchase. Never say a vendor "
    "is 'overcharging' - say 'X% above/below the most-recent comparable purchase'.\n"
    "  4. A cross-unit history table (markdown), low-to-high, showing unit, vendor, date, price, "
    "warranty, AMC, FOC - across all units, not one.\n"
    "  5. Recommended vendor, preferring one that bundles FOC + AMC into a low price (saves separate "
    "service costs).\n"
    "  6. An overall actionable recommendation the capex team can take into negotiation.\n\n"
    "Be concise and management-oriented. Ground every claim in tool output; never invent prices, "
    "vendors, dates, or specs. For follow-up questions, use the retained conversation context."
)

class State(TypedDict):
    messages: Annotated[Sequence, add_messages]

class CapexAgent(ResponsesAgent):
    def __init__(self):
        self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)
        self.tools = [price_fairness] + list(UCFunctionToolkit(function_names=UC_FUNCTIONS).tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _graph(self):
        def call_model(state):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
            return {"messages": [self.llm_with_tools.invoke(msgs)]}
        def should_continue(state):
            last = state["messages"][-1]
            return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"
        g = StateGraph(State)
        g.add_node("agent", RunnableLambda(call_model))
        g.add_node("tools", ToolNode(self.tools))
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        g.add_edge("tools", "agent")
        return g.compile()

    def predict_stream(self, req: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        msgs = to_chat_completions_input([m.model_dump() for m in req.input])
        for kind, payload in self._graph().stream({"messages": msgs}, stream_mode=["updates"]):
            if kind != "updates":
                continue
            for node in payload.values():
                if node.get("messages"):
                    yield from output_to_responses_items_stream(node["messages"])

    def predict(self, req: ResponsesAgentRequest) -> ResponsesAgentResponse:
        items = [ev.item for ev in self.predict_stream(req)
                 if ev.type == "response.output_item.done"]
        return ResponsesAgentResponse(output=items)

mlflow.langchain.autolog()
mlflow.models.set_model(CapexAgent())
'''
import json as _json
AGENT_CODE = (AGENT_CODE
              .replace("__LLM_ENDPOINT__", LLM_ENDPOINT)
              .replace("__MODEL_ENDPOINT__", MODEL_ENDPOINT)
              .replace("__UC_FUNCTIONS__", _json.dumps(FUNCTIONS)))
with open(f"{SRC_DIR}/agent.py", "w") as fh:
    fh.write(AGENT_CODE)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
print("agent.py written; tools = price_fairness +", FUNCTIONS)

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksFunction
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient

mlflow.set_registry_uri("databricks-uc")
me = WorkspaceClient().current_user.me().user_name
exp_dir = f"/Users/{me}/kauvery_capex_phase2"
WorkspaceClient().workspace.mkdirs(exp_dir)
mlflow.set_experiment(f"{exp_dir}/capex_agent")

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=MODEL_ENDPOINT),   # price_fairness calls this via ai_query
    *[DatabricksFunction(function_name=f) for f in FUNCTIONS],
]
input_example = {"input": [{"role": "user", "content":
    "Quotation: 10x Philips IntelliVue MX450 Patient Monitor (model IntelliVue MX450), "
    "unit rate INR 415000, warranty 12 months, no AMC, no FOC, delivery 95 days, 100% advance. "
    "Is this worth it and what is missing?"}]}

with mlflow.start_run(run_name="capex_agent"):
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=f"{SRC_DIR}/agent.py",
        resources=resources,
        input_example=input_example,
        pip_requirements=["mlflow", "databricks-langchain", "langgraph==0.3.4",
                          "databricks-agents", "pydantic>=2"],
        registered_model_name=AGENT_MODEL,
    )

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(AGENT_MODEL, "prod", info.registered_model_version)
print(f"Registered {AGENT_MODEL} v{info.registered_model_version}")

# COMMAND ----------

from databricks import agents
import json
deployment = agents.deploy(AGENT_MODEL, info.registered_model_version,
                           endpoint_name=AGENT_ENDPOINT,
                           tags={"project": "kauvery-capex-phase2"})
dbutils.notebook.exit(json.dumps({
    "agent_model": AGENT_MODEL, "version": info.registered_model_version,
    "endpoint_name": deployment.endpoint_name, "query_endpoint": deployment.query_endpoint,
}))
