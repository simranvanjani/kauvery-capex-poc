import os
from pathlib import Path

from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

# Load env vars from .env before importing the agent for proper auth
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# Need to import the agent to register the functions with the server
import agent_server.agent  # noqa: E402

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=False)
# Define the app as a module level variable to enable multiple workers
app = agent_server.app  # noqa: F841

# Git-based version tracking is a local dev/eval feature: it reads the repo's .git to tag traces
# with the current commit. The deployed app container has no .git (DABs uploads source, not git
# metadata), so this only runs locally when both a repo and an experiment are present. Running it
# unconditionally crashed the app at startup on deploy.
if Path(__file__).parents[1].joinpath(".git").exists() and os.getenv("MLFLOW_EXPERIMENT_ID"):
    setup_mlflow_git_based_version_tracking()


# ============ Custom CAPEX Copilot frontend + API ============
import io  # noqa: E402
import re  # noqa: E402
import uuid  # noqa: E402

from agents import Runner  # noqa: E402
from fastapi import Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from agent_server.agent import create_agent  # noqa: E402
from agent_server.history import normalize_history_items  # noqa: E402

_STATIC = Path(__file__).parents[1] / "static"
CATALOG = os.getenv("CATALOG", "kauvey_poc")
SCHEMA = os.getenv("SCHEMA", "gold")
VOLUME = os.getenv("VOLUME", "landing")  # UC Volume uploaded quotation PDFs land in

# Conversation history + feedback persistence on the capex-v2 Lakebase. Best-effort: if the DB is
# unreachable, chat/review still work and only history + feedback degrade.
from agent_server import store  # noqa: E402

try:
    store.init_schema()
except Exception as _e:  # noqa: BLE001
    import logging
    logging.getLogger(__name__).warning("Lakebase init_schema failed (history/feedback off): %s", _e)


def _user_email(request: Request) -> str:
    h = request.headers
    return h.get("x-forwarded-email") or h.get("x-forwarded-preferred-username") or "local@dev"


def _user_token(request: Request) -> str | None:
    """OBO token forwarded by the Databricks Apps proxy for the signed-in user."""
    return request.headers.get("x-forwarded-access-token")


_wc = None


def _w(token: str | None = None):
    """User-scoped client when an OBO token is given (uploads land as the signed-in user); the SP
    otherwise (local dev / identity lookup with no forwarded token)."""
    from databricks.sdk import WorkspaceClient
    if token:
        return WorkspaceClient(token=token, auth_type="pat")
    global _wc
    if _wc is None:
        _wc = WorkspaceClient()
    return _wc


@app.get("/")
def _index():
    return FileResponse(_STATIC / "index.html")


def _clean_output(text: str) -> str:
    """Guard: strip any leaked tool-call syntax (e.g. Llama's `<|python_tag|>[tool(...)]ipython`)
    so raw function-call text never reaches the UI. With a tool-calling model (Claude) this is a
    no-op; it only fires if a model role-plays the tool in plain text."""
    text = text.replace("<|python_tag|>", "")
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if s == "ipython":
            continue
        if re.match(r"^\[[A-Za-z_][A-Za-z0-9_]*\(.*\)\]$", s):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


@app.post("/api/chat")
async def _chat(request: Request):
    payload = await request.json()
    messages = payload.get("messages") or []
    if not messages and payload.get("message"):
        messages = [{"role": "user", "content": payload["message"]}]
    session_id = payload.get("session_id") or uuid.uuid4().hex
    email = _user_email(request)
    # OBO: run the agent's tools (warehouse SQL, scoring endpoint, PDF parse) as the signed-in user.
    from agent_server.agent import set_user_token
    set_user_token(_user_token(request))
    user_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    msgs = normalize_history_items(messages)
    try:
        result = await Runner.run(create_agent(), msgs)
        text = result.final_output
        if not isinstance(text, str):
            text = str(text)
        text = _clean_output(text)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=200)
    message_id = None
    try:  # best-effort persistence — never fail the chat on a DB hiccup
        store.create_session(session_id, email, title=(user_text or "New review")[:60])
        store.add_message(session_id, "user", user_text)
        message_id = store.add_message(session_id, "assistant", text)
    except Exception as pe:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("history persist failed: %s", pe)
    return {"text": text, "message_id": message_id, "session_id": session_id}


@app.post("/api/upload")
async def _upload(request: Request):
    data = await request.body()
    fname = request.headers.get("x-filename", "quotation.pdf")
    ext = ("." + fname.rsplit(".", 1)[-1]) if "." in fname else ".pdf"
    path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{uuid.uuid4().hex}{ext}"
    try:
        # Pass raw bytes (not BytesIO): the SDK/base client sets Content-Length via len(contents),
        # which raises "object of type '_io.BytesIO' has no len()" for a stream.
        _w(_user_token(request)).files.upload(path, data, overwrite=True)
        return {"volume_path": path}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=200)


@app.get("/api/me")
def _me(request: Request):
    h = request.headers
    email = h.get("x-forwarded-email") or h.get("x-forwarded-preferred-username") or ""
    name = h.get("x-forwarded-preferred-username") or email
    if not email:
        try:
            email = _w(_user_token(request)).current_user.me().user_name or ""
            name = email
        except Exception:  # noqa: BLE001
            pass
    disp = (name or email or "User").split("@")[0].replace(".", " ").title()
    initials = "".join(p[0] for p in disp.split()[:2]).upper() or "U"
    return {"email": email, "name": disp, "initials": initials}


@app.get("/api/history")
def _history(request: Request):
    try:
        return {"sessions": store.list_sessions(_user_email(request))}
    except Exception:  # noqa: BLE001
        return {"sessions": []}


@app.get("/api/session/{session_id}")
def _session(session_id: str):
    try:
        return {"messages": store.get_session_messages(session_id)}
    except Exception:  # noqa: BLE001
        return {"messages": []}


@app.post("/api/feedback")
async def _feedback(request: Request):
    body = await request.json()
    try:
        store.add_feedback(
            message_id=body.get("message_id"),
            session_id=body.get("session_id"),
            user_email=_user_email(request),
            rating=body.get("rating"),
            comment=body.get("comment"),
        )
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=200)
    return {"ok": True}


if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
