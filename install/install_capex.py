# Databricks notebook source
# MAGIC %md
# MAGIC # CAPEX Copilot — Installer
# MAGIC
# MAGIC **Import this notebook and click *Run all*.** It stands up (or *updates*, if it already exists)
# MAGIC everything the CAPEX app needs, in **your** workspace, against **your** already-loaded history.
# MAGIC No local setup, no CLI, no profile — the notebook runs as you, on your cluster.
# MAGIC
# MAGIC **Set your values once in §1 (Configuration) — that single cell is the only thing you edit.**
# MAGIC Every step below, *and the app*, reads exactly those values, so the backend and the app can't drift.
# MAGIC
# MAGIC | Step | What it installs | Behaviour |
# MAGIC |---|---|---|
# MAGIC | 1 | **Configuration** | the one place you set catalog / schema / model / endpoint / warehouse / app |
# MAGIC | 2 | Catalog · schema · landing volume | create if missing (never overwritten) |
# MAGIC | 3 | 3 Unity Catalog functions (`cross_unit_history`, `recommend_vendor`, `price_fairness`) | `CREATE OR REPLACE` — refreshed every run |
# MAGIC | 4 | ML model (scikit-learn + MLflow, alias `@prod`) | train if missing; **retrain when `UPDATE_ASSETS = True`** |
# MAGIC | 5 | Model serving endpoint | create, or update to the latest `@prod` |
# MAGIC | 6 | Lakebase project (conversation history + feedback, 7-day) | create if missing |
# MAGIC | 7 | The CAPEX app | prints the exact deploy commands, pre-filled from §1 |
# MAGIC
# MAGIC Every step is **idempotent** — an asset that exists is updated in place, never duplicated.
# MAGIC You load historical POs directly into `<catalog>.<schema>.extracted_pdf_datas` (e.g. from Oracle);
# MAGIC this installer never parses PDFs for historical data and never overwrites that table.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "databricks-sdk>=0.81.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 1 · Configuration

# COMMAND ----------

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — the ONLY cell you edit. Set these for your environment, Run All.
#  Every step below AND the app (step 7) use exactly these values.
# ══════════════════════════════════════════════════════════════════════════════

# --- Where everything lives (Unity Catalog) ------------------------------------
CATALOG          = "kauvey_poc"       # your Unity Catalog
SCHEMA           = "gold"             # schema holding ALL CAPEX assets: history table, functions, model
VOLUME           = "landing"          # UC Volume where uploaded quotation PDFs land

# --- ML asset names ------------------------------------------------------------
MODEL_NAME       = "capex_worth_it"   # registered model (Unity Catalog)
ENDPOINT         = "capex-worth-it"   # model serving endpoint (lowercase + hyphens, <= 63 chars)

# --- Conversation history + feedback ------------------------------------------
LAKEBASE_PROJECT = "capex-v2"         # Lakebase project_id (lowercase + hyphens — NOT the display name)

# --- Foundation models the app calls ------------------------------------------
CHAT_MODEL       = "databricks-claude-sonnet-4-5"   # agent chat model — MUST support tool-calling (Claude)
EXTRACT_MODEL    = "databricks-llama-4-maverick"    # PDF field extraction — MUST support ai_query json_object (Llama/GPT; Claude does NOT)

# --- App + compute -------------------------------------------------------------
APP_NAME         = "capex-quote-review"   # Databricks App name (lowercase + hyphens, <= 26 chars)
WAREHOUSE_ID     = ""                     # SQL warehouse id (Compute → SQL Warehouses → your warehouse → copy ID)

# --- Re-install / upgrade ------------------------------------------------------
UPDATE_ASSETS    = False   # True  → retrain & re-register an EXISTING model (v1 → v2 upgrade)
                           # False → create-if-missing (safe first install). Functions + endpoint refresh either way.

# ── nothing below this line needs editing ─────────────────────────────────────
FULL_MODEL      = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
HIST_TABLE      = f"{CATALOG}.{SCHEMA}.extracted_pdf_datas"
LAKEBASE_SCHEMA = "capex_app"   # Postgres schema the app's service principal creates & owns

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

def _brief(err):  # one-line reason only — never dump a full JVM/SDK stacktrace to the customer
    return str(err).strip().splitlines()[0][:300]

print("Running as:", w.current_user.me().user_name)
print("Target     :", f"{CATALOG}.{SCHEMA}  · model {MODEL_NAME}  · endpoint {ENDPOINT}  · app {APP_NAME}")

# COMMAND ----------

# MAGIC %md ## 2 · Catalog / schema / volume  (create if missing)

# COMMAND ----------

# CREATE CATALOG can fail harmlessly when the catalog already exists or the account uses Default
# Storage — the schema + volume steps below are the real check, so we don't surface that here.
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception:
    pass
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"catalog / schema / volume ready: {CATALOG}.{SCHEMA}.{VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Unity Catalog functions  (`CREATE OR REPLACE` — idempotent)
# MAGIC `cross_unit_history` + `recommend_vendor` read the tabular history directly; `price_fairness`
# MAGIC wraps the model endpoint. (DABs can't deploy UC functions, so they're created here as SQL.)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.cross_unit_history(search STRING)
RETURNS STRING
COMMENT 'Historical purchases of a matching item across all units, cheapest first, as a JSON array.'
RETURN (
  SELECT to_json(collect_list(rec)) FROM (
    SELECT named_struct('unit', unit_name, 'vendor', vendor_name, 'date', po_date,
      'unit_price', unit_rate, 'warranty_months', warranty_months,
      'maintenance', coalesce(amc_value, camc_value), 'foc', foc_details) AS rec
    FROM {CATALOG}.{SCHEMA}.extracted_pdf_datas
    WHERE unit_rate IS NOT NULL AND unit_rate > 0
      AND ( lower(item_description) LIKE concat('%', lower(search), '%')
         OR lower(model_no) LIKE concat('%', lower(search), '%')
         OR lower(make_brand) LIKE concat('%', lower(search), '%') )
    ORDER BY unit_rate ASC LIMIT 50 ) )
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.recommend_vendor(search STRING)
RETURNS STRING
COMMENT 'Vendors ranked by value-for-money; bundled FOC+AMC at low price rank highest. JSON array.'
RETURN (
  SELECT to_json(collect_list(rec)) FROM (
    SELECT named_struct('vendor', vendor_name, 'purchases', cnt, 'min_price', min_price,
      'avg_price', avg_price, 'foc_rate', foc_rate, 'amc_rate', amc_rate, 'value_score', value_score) AS rec
    FROM (
      SELECT vendor_name, count(*) AS cnt, round(min(unit_rate),2) AS min_price,
             round(avg(unit_rate),2) AS avg_price,
             round(avg(CASE WHEN foc_details IS NOT NULL THEN 1 ELSE 0 END),2) AS foc_rate,
             round(avg(CASE WHEN amc_value IS NOT NULL OR camc_value IS NOT NULL THEN 1 ELSE 0 END),2) AS amc_rate,
             round(avg(CASE WHEN foc_details IS NOT NULL THEN 1 ELSE 0 END)*0.5
                 + avg(CASE WHEN amc_value IS NOT NULL OR camc_value IS NOT NULL THEN 1 ELSE 0 END)*0.5,3) AS value_score
      FROM {CATALOG}.{SCHEMA}.extracted_pdf_datas
      WHERE unit_rate IS NOT NULL AND unit_rate > 0
        AND ( lower(item_description) LIKE concat('%', lower(search), '%')
           OR lower(model_no) LIKE concat('%', lower(search), '%')
           OR lower(make_brand) LIKE concat('%', lower(search), '%') )
      GROUP BY vendor_name )
    ORDER BY value_score DESC, min_price ASC LIMIT 20 ) )
""")

# price_fairness wraps the serving endpoint via ai_query, and Databricks validates that endpoint at
# CREATE time. On a first install the endpoint doesn't exist yet (it's built in §5), so this would
# fail. It's OPTIONAL — the app scores via the endpoint directly, not this function — so create it
# best-effort; re-run this cell after §5 (or on a later install) once the endpoint is READY.
try:
    spark.sql(f"""
    CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.price_fairness(
      item_description STRING, make_brand STRING, model_no STRING, qty INT, unit_rate DOUBLE,
      warranty_months INT, amc_present BOOLEAN, foc_present BOOLEAN, delivery_lead_days INT,
      payment_terms STRING, has_training BOOLEAN, has_installation BOOLEAN)
    RETURNS STRING
    COMMENT 'Scores one quotation line item via the {ENDPOINT} model serving endpoint.'
    RETURN ai_query('{ENDPOINT}', named_struct(
      'item_description', item_description, 'make_brand', make_brand, 'model_no', model_no,
      'qty', qty, 'unit_rate', unit_rate, 'warranty_months', warranty_months,
      'amc_present', amc_present, 'camc_present', false, 'foc_present', foc_present,
      'delivery_lead_days', delivery_lead_days, 'payment_terms', payment_terms,
      'has_training', has_training, 'has_installation', has_installation))
    """)
    print("functions ready: cross_unit_history, recommend_vendor, price_fairness")
except Exception as e:
    print("functions ready: cross_unit_history, recommend_vendor")
    print(f"[skipped] price_fairness needs the '{ENDPOINT}' serving endpoint to exist first "
          f"({_brief(e)}). It's optional — the app scores via the endpoint directly. Re-run this "
          f"cell after §5 creates the endpoint if you want the SQL/Genie scorer function.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · ML model  (train if missing, or retrain when *Update existing assets* = yes)
# MAGIC If `capex_worth_it` isn't registered yet, this runs the training notebook against your real data
# MAGIC (`data_source=real`). If it **already exists** (a prior v1 install) it's left as-is — **unless** you set
# MAGIC **`UPDATE_ASSETS = True`** in §1, which retrains and re-registers `@prod` (use this to pick up
# MAGIC model/logic changes, e.g. the removal of equipment categories). The training notebook lives at
# MAGIC `../notebooks/capex_phase2_demo` — resolves when the repo is imported as a **Git folder**. If you imported
# MAGIC notebooks individually, set `TRAIN_NOTEBOOK_PATH` below to its actual path.

# COMMAND ----------

# Relative to THIS notebook's folder (install/). The training notebook is in ../notebooks/.
# Override if you imported the notebooks individually.
TRAIN_NOTEBOOK_PATH = "../notebooks/capex_phase2_demo"

model_exists = False
try:
    w.registered_models.get(FULL_MODEL)
    model_exists = True
except Exception:
    model_exists = False

if model_exists and not UPDATE_ASSETS:
    print(f"{FULL_MODEL} already registered — leaving it (set UPDATE_ASSETS = True in §1 to retrain).")
else:
    reason = "retraining (update mode)" if model_exists else "not found — training"
    print(f"{FULL_MODEL} {reason} via {TRAIN_NOTEBOOK_PATH} …")
    try:
        dbutils.notebook.run(TRAIN_NOTEBOOK_PATH, 3600, {
            "data_source": "real", "catalog": CATALOG, "schema": SCHEMA, "model_name": MODEL_NAME})
        model_exists = True
        print("training complete — new version registered and @prod moved to it (endpoint updates in step 5).")
    except Exception as e:
        print(f"[action needed] couldn't auto-run the training notebook: {_brief(e)}")
        print(f"  Fix: set TRAIN_NOTEBOOK_PATH above to the real path of capex_phase2_demo, OR")
        print(f"  open capex_phase2_demo, set data_source=real + catalog={CATALOG} + schema={SCHEMA}, "
              f"Run All, then re-run this cell.")

# COMMAND ----------

# MAGIC %md ## 5 · Model serving endpoint  (create, or update to the latest `@prod`)

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.tracking import MlflowClient

try:
    # ServedEntityInput takes a concrete entity_version, not an alias — resolve @prod to a version.
    prod_version = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(FULL_MODEL, "prod").version
    entity = ServedEntityInput(entity_name=FULL_MODEL, entity_version=prod_version,
                               scale_to_zero_enabled=True, workload_size="Small")

    endpoint_exists = True
    try:
        w.serving_endpoints.get(ENDPOINT)
    except Exception:
        endpoint_exists = False

    if endpoint_exists:
        print(f"{ENDPOINT} exists — updating to {FULL_MODEL} v{prod_version} (@prod)")
        w.serving_endpoints.update_config(name=ENDPOINT, served_entities=[entity])
    else:
        print(f"creating {ENDPOINT} -> {FULL_MODEL} v{prod_version} (@prod)")
        w.serving_endpoints.create(name=ENDPOINT, config=EndpointCoreConfigInput(served_entities=[entity]))
except Exception as e:
    print(f"[action needed] endpoint step failed: {_brief(e)}. If the model isn't trained yet or has no "
          f"@prod alias, finish step 4 then re-run this cell.")

# COMMAND ----------

# MAGIC %md ## 6 · Lakebase project for conversation history  (create if missing, 7-day retention)

# COMMAND ----------

# List via the REST API (works on any SDK version; w.postgres may be absent on older SDKs).
try:
    _projects = w.api_client.do("GET", "/api/2.0/postgres/projects").get("projects") or []
except Exception as e:
    _projects = []
    print(f"[note] couldn't list Lakebase projects: {_brief(e)}")

# Match on project_id OR display_name (forgiving if §1 holds the display name), then normalise
# LAKEBASE_PROJECT to the real project_id — §7's Lakebase resource path needs the id, not the name.
_match = next((p for p in _projects
               if LAKEBASE_PROJECT in (p.get("project_id"), p.get("status", {}).get("display_name"))), None)
if _match:
    LAKEBASE_PROJECT = _match["project_id"]
    print(f"Lakebase project '{LAKEBASE_PROJECT}' ready (reusing existing).")
else:
    print(f"[create once] Lakebase project '{LAKEBASE_PROJECT}' not found — create it, then re-run this cell:")
    print("  databricks postgres create-project <project_id> "
          "--json '{\"spec\": {\"display_name\": \"CAPEX conversation history + feedback\"}}'")
    print("  (or Compute → Lakebase → New project). Set LAKEBASE_PROJECT in §1 to the project_id (lowercase-hyphens), not the display name.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Deploy the CAPEX app  (built / reused **and deployed** from this notebook)
# MAGIC The cell below writes `app-v2/app.yaml` from your §1 values, then **creates the app (or reuses it if it
# MAGIC already exists), attaches the service-principal grants, and deploys the code** via the Apps API — no
# MAGIC terminal needed. If the auto-deploy can't complete in your workspace it prints the equivalent CLI commands
# MAGIC to run instead. Re-runs redeploy the latest code and keep the same URL.

# COMMAND ----------

import json, os, time

# app-v2/app.yaml, generated from §1 so the app always matches the backend
_env = [
    ("MLFLOW_TRACKING_URI", "databricks"), ("MLFLOW_REGISTRY_URI", "databricks-uc"),
    ("CHAT_PROXY_TIMEOUT_SECONDS", "300"),
    ("CATALOG", CATALOG), ("SCHEMA", SCHEMA), ("MODEL_ENDPOINT", ENDPOINT),
    ("CHAT_MODEL", CHAT_MODEL), ("EXTRACT_MODEL", EXTRACT_MODEL), ("LAKEBASE_SCHEMA", LAKEBASE_SCHEMA),
]
app_yaml = 'command: ["uv", "run", "start-server"]\n'
app_yaml += "# Generated from install_capex §1 — edit §1 and re-run; do not hand-edit.\n\nenv:\n"
for k, v in _env:
    app_yaml += f'  - name: {k}\n    value: "{v}"\n'
app_yaml += '  - name: DATABRICKS_WAREHOUSE_ID\n    valueFrom: "sql-warehouse"\n'

# Resolve this Git folder's app-v2/app.yaml from the notebook's own path, and write it.
WS_APP_PATH = None
try:
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    _root = os.path.dirname(os.path.dirname(_nb))          # .../install/install_capex -> repo root
    WS_APP_PATH = f"/Workspace{_root}/app-v2"
    with open(f"{WS_APP_PATH}/app.yaml", "w") as fh:
        fh.write(app_yaml)
    print(f"✅ wrote {WS_APP_PATH}/app.yaml from §1\n")
except Exception as e:
    print(f"[note] couldn't auto-write app.yaml: {_brief(e)} — paste this into app-v2/app.yaml:\n\n{app_yaml}")
    WS_APP_PATH = "/Workspace/Users/<you>/kauvery-capex-poc/app-v2"

# --- 2. service-principal grants (endpoints + warehouse + Lakebase) ---
res_list = [
    {"name": "chat-llm",       "serving_endpoint": {"name": CHAT_MODEL,    "permission": "CAN_QUERY"}},
    {"name": "extract-llm",    "serving_endpoint": {"name": EXTRACT_MODEL, "permission": "CAN_QUERY"}},
    {"name": "model-endpoint", "serving_endpoint": {"name": ENDPOINT,      "permission": "CAN_QUERY"}},
    {"name": "sql-warehouse",  "sql_warehouse":    {"id": WAREHOUSE_ID,    "permission": "CAN_USE"}},
    {"name": "postgres",       "postgres": {
        "branch":   f"projects/{LAKEBASE_PROJECT}/branches/production",
        "database": f"projects/{LAKEBASE_PROJECT}/branches/production/databases/databricks-postgres",
        "permission": "CAN_CONNECT_AND_CREATE"}},
]

def _print_cli():
    print("\n── or deploy from a terminal (source is already in this Git folder) ──")
    print(f"databricks apps create {APP_NAME}")
    print(f"databricks apps deploy {APP_NAME} --source-code-path {WS_APP_PATH}")
    print("cat > resources.json <<'JSON'")
    print(json.dumps({"update_mask": "resources", "app": {"resources": res_list}}, indent=2))
    print("JSON")
    print(f"databricks apps create-update {APP_NAME} --json @resources.json")
    print(f"databricks apps deploy {APP_NAME} --source-code-path {WS_APP_PATH}")

# --- 3. build-or-reuse the app, attach grants, deploy the code (Apps API; CLI fallback on any error) ---
if not WAREHOUSE_ID:
    print("\n[action needed] WAREHOUSE_ID is empty in §1 — set it, then re-run. Skipping auto-deploy.")
    _print_cli()
elif not WS_APP_PATH or WS_APP_PATH.endswith("<you>/kauvery-capex-poc/app-v2"):
    print("\n[note] couldn't resolve this Git folder's path — deploy with the CLI below.")
    _print_cli()
else:
    def _api(method, path, body=None, query=None):
        return w.api_client.do(method, path, body=body, query=query)
    try:
        try:
            _api("GET", f"/api/2.0/apps/{APP_NAME}")
            print(f"reusing existing app '{APP_NAME}'")
        except Exception:
            print(f"creating app '{APP_NAME}' … (compute provisioning, ~1–2 min)")
            _api("POST", "/api/2.0/apps", body={"name": APP_NAME, "description": "Kauvery CAPEX Copilot"})
            for _ in range(60):
                if _api("GET", f"/api/2.0/apps/{APP_NAME}").get("compute_status", {}).get("state") in ("ACTIVE", "ERROR"):
                    break
                time.sleep(5)
        # attach grants BEFORE deploy so the app's SP owns the Lakebase schema when it starts
        _api("PATCH", f"/api/2.0/apps/{APP_NAME}", body={"name": APP_NAME, "resources": res_list},
             query={"update_mask": "resources"})
        print("attached resource grants (endpoints + warehouse + Lakebase)")
        print("deploying code …")
        dep = _api("POST", f"/api/2.0/apps/{APP_NAME}/deployments",
                   body={"source_code_path": WS_APP_PATH, "mode": "SNAPSHOT"})
        dep_id = dep.get("deployment_id")
        state = "PENDING"
        for _ in range(120):
            state = _api("GET", f"/api/2.0/apps/{APP_NAME}/deployments/{dep_id}").get("status", {}).get("state", "")
            if state in ("SUCCEEDED", "FAILED", "STOPPED"):
                break
            time.sleep(5)
        app = _api("GET", f"/api/2.0/apps/{APP_NAME}")
        print(f"\n✅ deploy {state} · app state {app.get('app_status', {}).get('state')}")
        print(f"   URL: {app.get('url')}")
        if state != "SUCCEEDED":
            _print_cli()
    except Exception as e:
        print(f"\n[auto-deploy didn't complete: {_brief(e)}]")
        _print_cli()

print(f"\n✅ Backend install complete for {CATALOG}.{SCHEMA}.")
