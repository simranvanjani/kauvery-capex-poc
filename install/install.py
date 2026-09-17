#!/usr/bin/env python3
"""
Kauvery CAPEX — one-command installer.

Stands up (or UPDATES, if it already exists) everything the CAPEX app needs, in the
CUSTOMER's own workspace, against their already-loaded historical data:

  1. catalog / schema / landing volume            (CREATE ... IF NOT EXISTS)
  2. the 3 Unity Catalog functions                (CREATE OR REPLACE — always refreshed)
  3. the custom ML model  `capex_worth_it`         (train+register via the notebook job if missing)
  4. the model serving endpoint  `capex-worth-it`  (create, or update to the latest @prod version)
  5. a Lakebase project for conversation history   (create if missing)
  6. the Databricks App  (V2, app-v2/)             (DAB deploy — create-or-update)

Every step is idempotent: an asset that already exists is updated in place, never duplicated.
The customer loads their historical POs directly into `<catalog>.<schema>.extracted_pdf_datas`
(from Oracle) — this installer never parses PDFs for historical data and never overwrites that table.

Usage:
    python install/install.py --profile <PROFILE> \
        --catalog purchase_capex_catalog --schema gold \
        --warehouse-id <SQL_WAREHOUSE_ID>

    # skip individual steps if you manage them yourself:
    #   --skip-model  --skip-endpoint  --skip-lakebase  --skip-app

Requires: databricks-sdk (pip install databricks-sdk), the Databricks CLI (for the app bundle),
and a profile in ~/.databrickscfg. Run `databricks auth login --profile <PROFILE>` first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )
except ImportError:
    sys.exit("databricks-sdk is required:  pip install databricks-sdk")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
APP_DIR = REPO / "app-v2"
NOTEBOOK = REPO / "notebooks" / "capex_phase2_demo.py"
FUNCTIONS_SQL = HERE / "uc_functions.sql"


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


# --------------------------------------------------------------------------- SQL helper
class Sql:
    def __init__(self, w: WorkspaceClient, warehouse_id: str):
        self.w = w
        self.warehouse_id = warehouse_id

    def exec(self, statement: str):
        r = self.w.statement_execution.execute_statement(
            warehouse_id=self.warehouse_id, statement=statement, wait_timeout="50s"
        )
        state = r.status.state.value if r.status and r.status.state else "UNKNOWN"
        if state != "SUCCEEDED":
            err = getattr(getattr(r.status, "error", None), "message", state)
            raise RuntimeError(f"SQL failed ({state}): {err}\n  statement: {statement[:120]}...")
        return r


def resolve_warehouse(w: WorkspaceClient, given: str | None) -> str:
    if given:
        return given
    # Fall back to the first available warehouse, preferring a running one. Print what we picked.
    whs = list(w.warehouses.list())
    if not whs:
        sys.exit("No SQL warehouse found. Pass --warehouse-id explicitly.")
    running = [x for x in whs if getattr(x.state, "value", "") == "RUNNING"]
    chosen = (running or whs)[0]
    log("warehouse", f"no --warehouse-id given; using '{chosen.name}' ({chosen.id})")
    return chosen.id


# --------------------------------------------------------------------------- steps
def step_catalog(sql: Sql, catalog: str, schema: str, volume: str) -> None:
    log("catalog", f"ensuring {catalog}.{schema} + volume '{volume}'")
    # Catalog may already exist (customer-owned); only create schema/volume if missing.
    try:
        sql.exec(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    except Exception as e:  # noqa: BLE001 — Default-Storage metastores reject CREATE CATALOG; that's fine if it exists
        log("catalog", f"note: could not create catalog (ok if it already exists): {e}")
    sql.exec(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    sql.exec(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")
    log("catalog", "ready")


def step_functions(sql: Sql, catalog: str, schema: str) -> None:
    log("functions", "creating/replacing UC functions from uc_functions.sql")
    ddl = FUNCTIONS_SQL.read_text().replace("{{CATALOG}}", catalog).replace("{{SCHEMA}}", schema)
    # split on the trailing ');' of each CREATE OR REPLACE FUNCTION statement
    statements = [s.strip() for s in ddl.split(");\n") ]
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt or stmt.startswith("--") and "FUNCTION" not in stmt:
            continue
        if "CREATE OR REPLACE FUNCTION" not in stmt:
            continue
        if not stmt.endswith(")"):
            stmt += ")"
        sql.exec(stmt + ";" if not stmt.endswith(";") else stmt)
    log("functions", "cross_unit_history, recommend_vendor, price_fairness ready")


def step_model(w: WorkspaceClient, catalog: str, schema: str, model_name: str,
               profile: str, skip: bool) -> None:
    full = f"{catalog}.{schema}.{model_name}"
    if skip:
        log("model", f"--skip-model set; assuming {full} exists")
        return
    try:
        w.registered_models.get(full)
        log("model", f"{full} already registered — leaving it (re-run the notebook to retrain)")
        return
    except Exception:  # noqa: BLE001 — not found -> train it
        pass
    log("model", f"{full} not found — submitting the training notebook (data_source=real) as a job")
    ws_path = f"/Workspace/Users/{w.current_user.me().user_name}/kauvery-capex-install/capex_phase2_demo"
    try:
        w.workspace.mkdirs(ws_path.rsplit("/", 1)[0])
        with open(NOTEBOOK, "rb") as fh:
            import base64
            from databricks.sdk.service.workspace import ImportFormat, Language
            w.workspace.import_(
                path=ws_path, content=base64.b64encode(fh.read()).decode(),
                format=ImportFormat.SOURCE, language=Language.PYTHON, overwrite=True,
            )
        log("model", f"uploaded notebook -> {ws_path}; submitting serverless run")
        subprocess.run(
            ["databricks", "jobs", "submit", "--no-wait", "--profile", profile, "--json",
             f'{{"run_name":"capex-install-train","tasks":[{{"task_key":"train",'
             f'"notebook_task":{{"notebook_path":"{ws_path}",'
             f'"base_parameters":{{"data_source":"real","catalog":"{catalog}","schema":"{schema}",'
             f'"model_name":"{model_name}"}}}},"environment_key":"e"}}],'
             f'"environments":[{{"environment_key":"e","spec":{{"client":"4",'
             f'"dependencies":["scikit-learn","joblib","mlflow","fpdf2"]}}}}]}}'],
            check=True,
        )
        log("model", "training job submitted (runs in background; check the Jobs UI)")
    except Exception as e:  # noqa: BLE001
        log("model", f"could not auto-train ({e}).")
        log("model", f"MANUAL FALLBACK: open {NOTEBOOK.name} in the workspace, set data_source=real, "
                     f"catalog={catalog}, schema={schema}, and Run All.")


def step_endpoint(w: WorkspaceClient, catalog: str, schema: str, model_name: str,
                  endpoint: str, skip: bool) -> None:
    if skip:
        log("endpoint", f"--skip-endpoint set; skipping {endpoint}")
        return
    full = f"{catalog}.{schema}.{model_name}"
    entity = ServedEntityInput(
        entity_name=full, entity_version=None, entity_alias="prod",
        scale_to_zero_enabled=True, workload_size="Small",
    )
    config = EndpointCoreConfigInput(served_entities=[entity])
    try:
        w.serving_endpoints.get(endpoint)
        log("endpoint", f"{endpoint} exists — updating to {full}@prod")
        w.serving_endpoints.update_config(name=endpoint, served_entities=[entity])
    except Exception:  # noqa: BLE001 — not found -> create
        log("endpoint", f"creating {endpoint} -> {full}@prod")
        try:
            w.serving_endpoints.create(name=endpoint, config=config)
        except Exception as e:  # noqa: BLE001
            log("endpoint", f"create failed ({e}). If the model isn't trained yet, re-run install "
                            f"after the training job finishes.")


def step_lakebase(w: WorkspaceClient, project: str, profile: str, skip: bool) -> None:
    if skip:
        log("lakebase", f"--skip-lakebase set; skipping {project}")
        return
    existing = subprocess.run(
        ["databricks", "postgres", "list-projects", "--profile", profile, "--output", "json"],
        capture_output=True, text=True,
    ).stdout
    if f'"{project}"' in existing:
        log("lakebase", f"project '{project}' already exists — leaving it (7-day retention)")
        return
    log("lakebase", f"creating Lakebase project '{project}' (scale-to-zero, 7-day retention)")
    subprocess.run(
        ["databricks", "postgres", "create-project", project, "--profile", profile,
         "--json", f'{{"spec": {{"display_name": "Kauvery CAPEX — conversation history"}}}}'],
        check=True,
    )


def step_app(profile: str, target: str, skip: bool) -> None:
    if skip:
        log("app", "--skip-app set; skipping app deploy")
        return
    if not (APP_DIR / "databricks.yml").exists():
        log("app", f"no databricks.yml in {APP_DIR}; skipping app deploy")
        return
    log("app", f"deploying the app bundle (create-or-update) from {APP_DIR}")
    subprocess.run(["databricks", "bundle", "deploy", "-t", target, "--profile", profile],
                   cwd=APP_DIR, check=True)
    subprocess.run(["databricks", "bundle", "run", "capex_copilot", "-t", target, "--profile", profile],
                   cwd=APP_DIR, check=True)
    log("app", "app deployed / updated")


def main() -> None:
    p = argparse.ArgumentParser(description="Install/update the Kauvery CAPEX app + backend.")
    p.add_argument("--profile", required=True, help="Databricks CLI profile")
    p.add_argument("--catalog", default="purchase_capex_catalog")
    p.add_argument("--schema", default="gold")
    p.add_argument("--warehouse-id", default=None, help="SQL warehouse for DDL (auto-picked if omitted)")
    p.add_argument("--volume", default="landing")
    p.add_argument("--model-name", default="capex_worth_it")
    p.add_argument("--endpoint", default="capex-worth-it")
    p.add_argument("--lakebase-project", default="capex-v2")
    p.add_argument("--target", default="dev", help="DAB target for the app bundle")
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--skip-endpoint", action="store_true")
    p.add_argument("--skip-lakebase", action="store_true")
    p.add_argument("--skip-app", action="store_true")
    args = p.parse_args()

    w = WorkspaceClient(profile=args.profile)
    log("auth", f"connected as {w.current_user.me().user_name}")
    sql = Sql(w, resolve_warehouse(w, args.warehouse_id))

    step_catalog(sql, args.catalog, args.schema, args.volume)
    step_functions(sql, args.catalog, args.schema)
    step_model(w, args.catalog, args.schema, args.model_name, args.profile, args.skip_model)
    step_endpoint(w, args.catalog, args.schema, args.model_name, args.endpoint, args.skip_endpoint)
    step_lakebase(w, args.lakebase_project, args.profile, args.skip_lakebase)
    step_app(args.profile, args.target, args.skip_app)

    print("\n✅ Install complete. Open the app from the Databricks Apps page.")
    print("   Re-run this script any time — existing assets are updated, not duplicated.")


if __name__ == "__main__":
    main()
