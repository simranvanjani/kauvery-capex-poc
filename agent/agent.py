"""CAPEX conversational agent — readable copy of the LangGraph ResponsesAgent.

Source of truth is notebooks/deploy_agent.py (which writes + logs + deploys this). This copy is
for reading the tool/agent structure. Tools: price_fairness (direct call to the ML model
endpoint), cross_unit_history + recommend_vendor (UC functions). LLM: Databricks-hosted Llama.
"""
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
LLM_ENDPOINT = "databricks-llama-4-maverick"
MODEL_ENDPOINT = "capex-worth-it"
UC_FUNCTIONS = ["kauvey_poc.gold.cross_unit_history", "kauvey_poc.gold.recommend_vendor"]


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
