"""Conversation-history + feedback persistence on the capex-v2 Lakebase (Autoscaling Postgres).

Self-contained so it doesn't collide with route edits in start_server.py. Connects to the
capex-v2 project's production/primary endpoint, mints a short-lived OAuth DB credential via the
Databricks REST API (POST /api/2.0/postgres/credentials), and exposes small CRUD helpers.

Scope: sessions + messages (history) and feedback. No Monitor.

Connection identity:
- Locally you connect as your user email; in the deployed app the Postgres role is the app's
  service principal. The SP can connect + own its schema only if the `postgres` app resource is
  attached with CAN_CONNECT_AND_CREATE (see INTEGRATION.md). init_schema() must run at startup so
  the SP creates and owns the `capex` schema.
"""

import os
import threading
import time

import psycopg2
from databricks.sdk import WorkspaceClient

# capex-v2 Autoscaling Lakebase — production branch / primary endpoint. Overridable via env.
ENDPOINT = os.getenv(
    "LAKEBASE_ENDPOINT", "projects/capex-v2/branches/production/endpoints/primary"
)
PGHOST = os.getenv(
    "PGHOST", "ep-winter-flower-e7lbinuq.database.centralindia.azuredatabricks.net"
)
PGDATABASE = os.getenv("PGDATABASE", "databricks_postgres")
SCHEMA = os.getenv("LAKEBASE_SCHEMA", "capex")
# Postgres role to connect as. Leave PGUSER unset in-app to use the SP identity from the SDK.
_PGUSER_ENV = os.getenv("PGUSER")

_lock = threading.Lock()
_conn = None
_token = None
_token_exp = 0.0
_wc = None
_user = None


def _client() -> WorkspaceClient:
    global _wc
    if _wc is None:
        _wc = WorkspaceClient()
    return _wc


def _role() -> str:
    global _user
    if _user is None:
        _user = _PGUSER_ENV or _client().current_user.me().user_name
    return _user


def _mint_token() -> str:
    global _token, _token_exp
    resp = _client().api_client.do(
        "POST", "/api/2.0/postgres/credentials", body={"endpoint": ENDPOINT}
    )
    _token = resp["token"]
    _token_exp = time.time() + 50 * 60  # tokens last ~1h; refresh early
    return _token


def _connect():
    tok = _token if (_token and time.time() < _token_exp) else _mint_token()
    c = psycopg2.connect(
        host=PGHOST, user=_role(), password=tok, dbname=PGDATABASE,
        sslmode="require", connect_timeout=15,
    )
    c.autocommit = True
    return c


def _healthy(c) -> bool:
    try:
        with c.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def _get_conn():
    global _conn
    with _lock:
        if _conn is None or _conn.closed or not _healthy(_conn):
            _conn = _connect()
        return _conn


def _exec(sql, params=None, fetch=None):
    """Execute with one reconnect retry (handles token expiry / scale-to-zero wakeups)."""
    global _conn, _token, _token_exp
    for attempt in (1, 2):
        try:
            c = _get_conn()
            with c.cursor() as cur:
                cur.execute(sql, params or ())
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                return None
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            with _lock:
                _conn = None
                _token = None
                _token_exp = 0.0
            if attempt == 2:
                raise


def init_schema():
    """Idempotent. Run once at app startup (the SP becomes owner of the schema)."""
    _exec(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    _exec(
        f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.sessions(
            session_id text PRIMARY KEY,
            user_email text,
            title text,
            created_ts timestamptz DEFAULT now())"""
    )
    _exec(
        f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.messages(
            id bigserial PRIMARY KEY,
            session_id text,
            role text,
            content text,
            ts timestamptz DEFAULT now())"""
    )
    _exec(
        f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.feedback(
            id bigserial PRIMARY KEY,
            message_id bigint,
            session_id text,
            user_email text,
            rating text,
            comment text,
            ts timestamptz DEFAULT now())"""
    )
    _exec(f"CREATE INDEX IF NOT EXISTS ix_sessions_user ON {SCHEMA}.sessions(user_email, created_ts DESC)")
    _exec(f"CREATE INDEX IF NOT EXISTS ix_messages_session ON {SCHEMA}.messages(session_id, ts)")


def create_session(session_id: str, user_email: str, title: str | None = None) -> None:
    _exec(
        f"""INSERT INTO {SCHEMA}.sessions(session_id, user_email, title) VALUES(%s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET title = COALESCE(EXCLUDED.title, {SCHEMA}.sessions.title)""",
        (session_id, user_email, title),
    )


def add_message(session_id: str, role: str, content: str) -> int | None:
    row = _exec(
        f"INSERT INTO {SCHEMA}.messages(session_id, role, content) VALUES(%s, %s, %s) RETURNING id",
        (session_id, role, content),
        fetch="one",
    )
    return row[0] if row else None


def list_sessions(user_email: str, days: int = 7) -> list[dict]:
    rows = (
        _exec(
            f"""SELECT session_id, title, created_ts FROM {SCHEMA}.sessions
                WHERE user_email = %s AND created_ts > now() - make_interval(days => %s)
                ORDER BY created_ts DESC LIMIT 100""",
            (user_email, days),
            fetch="all",
        )
        or []
    )
    return [
        {"session_id": r[0], "title": r[1], "created_ts": r[2].isoformat() if r[2] else None}
        for r in rows
    ]


def get_session_messages(session_id: str) -> list[dict]:
    rows = (
        _exec(
            f"SELECT id, role, content, ts FROM {SCHEMA}.messages WHERE session_id = %s ORDER BY ts, id",
            (session_id,),
            fetch="all",
        )
        or []
    )
    return [
        {"id": r[0], "role": r[1], "content": r[2], "ts": r[3].isoformat() if r[3] else None}
        for r in rows
    ]


def add_feedback(
    message_id: int | None, session_id: str, user_email: str, rating: str, comment: str | None = None
) -> None:
    # One feedback row per message: update the existing row when the rating/comment changes rather
    # than appending duplicates (repeated clicks of the same rating are no-ops on the frontend too).
    if message_id is not None:
        updated = _exec(
            f"""UPDATE {SCHEMA}.feedback SET rating = %s, comment = %s, user_email = %s, ts = now()
                WHERE message_id = %s RETURNING id""",
            (rating, comment, user_email, message_id),
            fetch="one",
        )
        if updated:
            return
    _exec(
        f"""INSERT INTO {SCHEMA}.feedback(message_id, session_id, user_email, rating, comment)
            VALUES(%s, %s, %s, %s, %s)""",
        (message_id, session_id, user_email, rating, comment),
    )


def list_feedback(limit: int = 200) -> list[dict]:
    """All feedback rows, newest first — for the admin Monitor page."""
    rows = (
        _exec(
            f"""SELECT id, message_id, session_id, user_email, rating, comment, ts
                FROM {SCHEMA}.feedback ORDER BY ts DESC LIMIT %s""",
            (limit,),
            fetch="all",
        )
        or []
    )
    return [
        {"id": r[0], "message_id": r[1], "session_id": r[2], "user_email": r[3],
         "rating": r[4], "comment": r[5], "ts": r[6].isoformat() if r[6] else None}
        for r in rows
    ]
