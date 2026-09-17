# Databricks notebook source
# MAGIC %md
# MAGIC # CAPEX Copilot — Installer
# MAGIC
# MAGIC **Import this notebook and click *Run all*.** It stands up (or *updates*, if it already exists)
# MAGIC everything the CAPEX app needs, in **your** workspace, against **your** already-loaded history.
# MAGIC No local setup, no CLI, no profile — the notebook runs as you, on your cluster.
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | 1 | catalog / schema / landing volume (create if missing) |
# MAGIC | 2 | the 3 Unity Catalog functions (`CREATE OR REPLACE` — always refreshed) |
# MAGIC | 3 | the ML model `capex_worth_it` (train via the notebook if missing) |
# MAGIC | 4 | the serving endpoint `capex-worth-it` (create, or update to the latest `@prod`) |
# MAGIC | 5 | a Lakebase project for conversation history (create if missing) |
# MAGIC | 6 | the CAPEX app (guided — see the last cell) |
# MAGIC
# MAGIC Every step is **idempotent**: an asset that exists is updated in place, never duplicated.
# MAGIC You load historical POs directly into `<catalog>.<schema>.extracted_pdf_datas` (from Oracle) —
# MAGIC this installer never parses PDFs for historical data and never overwrites that table.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "databricks-sdk>=0.81.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 1 · Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "purchase_capex_catalog", "Catalog")
dbutils.widgets.text("schema", "gold", "Schema")
dbutils.widgets.text("volume", "landing", "Landing volume")
dbutils.widgets.text("model_name", "capex_worth_it", "Registered model name")
dbutils.widgets.text("endpoint", "capex-worth-it", "Model serving endpoint")
dbutils.widgets.text("lakebase_project", "capex-v2", "Lakebase project")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
VOLUME = dbutils.widgets.get("volume").strip()
MODEL_NAME = dbutils.widgets.get("model_name").strip()
ENDPOINT = dbutils.widgets.get("endpoint").strip()
LAKEBASE_PROJECT = dbutils.widgets.get("lakebase_project").strip()

FULL_MODEL = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
HIST_TABLE = f"{CATALOG}.{SCHEMA}.extracted_pdf_datas"

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
print("Running as:", w.current_user.me().user_name)
print("Target     :", f"{CATALOG}.{SCHEMA}  · model {MODEL_NAME}  · endpoint {ENDPOINT}")

# COMMAND ----------

# MAGIC %md ## 2 · Catalog / schema / volume  (create if missing)

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception as e:
    print(f"[note] could not create catalog (fine if it already exists): {e}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print("catalog / schema / volume ready")

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · ML model  (train via `capex_phase2_demo` if missing)
# MAGIC If `capex_worth_it` isn't registered yet, this runs the training notebook against your real data
# MAGIC (`data_source=real`). The training notebook lives at `../notebooks/capex_phase2_demo` — this resolves
# MAGIC when the whole repo is imported as a **Git folder** (keeping the `install/` + `notebooks/` layout).
# MAGIC If you imported notebooks individually, set `TRAIN_NOTEBOOK_PATH` below to its actual path.

# COMMAND ----------

# Relative to THIS notebook's folder (install/). The training notebook is in ../notebooks/.
# Override if you imported the notebooks individually.
TRAIN_NOTEBOOK_PATH = "../notebooks/capex_phase2_demo"

model_exists = False
try:
    w.registered_models.get(FULL_MODEL)
    model_exists = True
    print(f"{FULL_MODEL} already registered — leaving it (re-run capex_phase2_demo to retrain).")
except Exception:
    print(f"{FULL_MODEL} not found — training now via {TRAIN_NOTEBOOK_PATH} …")
    try:
        dbutils.notebook.run(TRAIN_NOTEBOOK_PATH, 3600, {
            "data_source": "real", "catalog": CATALOG, "schema": SCHEMA, "model_name": MODEL_NAME})
        model_exists = True
        print("training complete.")
    except Exception as e:
        print(f"[action needed] couldn't auto-run the training notebook ({e}).")
        print(f"  Fix: set TRAIN_NOTEBOOK_PATH above to the real path of capex_phase2_demo, OR")
        print(f"  open capex_phase2_demo, set data_source=real + catalog={CATALOG} + schema={SCHEMA}, "
              f"Run All, then re-run this cell.")

# COMMAND ----------

# MAGIC %md ## 5 · Model serving endpoint  (create, or update to the latest `@prod`)

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

entity = ServedEntityInput(entity_name=FULL_MODEL, entity_version=None, entity_alias="prod",
                           scale_to_zero_enabled=True, workload_size="Small")
try:
    w.serving_endpoints.get(ENDPOINT)
    print(f"{ENDPOINT} exists — updating to {FULL_MODEL}@prod")
    w.serving_endpoints.update_config(name=ENDPOINT, served_entities=[entity])
except Exception:
    print(f"creating {ENDPOINT} -> {FULL_MODEL}@prod")
    try:
        w.serving_endpoints.create(name=ENDPOINT, config=EndpointCoreConfigInput(served_entities=[entity]))
    except Exception as e:
        print(f"[action needed] endpoint create failed ({e}). If the model isn't trained yet, "
              f"finish step 4 then re-run this cell.")

# COMMAND ----------

# MAGIC %md ## 6 · Lakebase project for conversation history  (create if missing, 7-day retention)

# COMMAND ----------

try:
    projects = [p.project_id for p in w.postgres.list_projects()]
    if LAKEBASE_PROJECT in projects:
        print(f"Lakebase project '{LAKEBASE_PROJECT}' already exists — leaving it.")
    else:
        print(f"creating Lakebase project '{LAKEBASE_PROJECT}' …")
        w.postgres.create_project(
            project_id=LAKEBASE_PROJECT,
            spec={"display_name": "Kauvery CAPEX — conversation history + feedback"})
        print("created (scale-to-zero, 7-day retention).")
except Exception as e:
    print(f"[action needed] Lakebase step needs attention ({e}).")
    print("  Create it once in Compute → Lakebase, or via CLI: "
          f"databricks postgres create-project {LAKEBASE_PROJECT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Deploy the app
# MAGIC The app is deployed from source (a build step), so it's the one piece done from the CLI, once:
# MAGIC
# MAGIC ```bash
# MAGIC cd app-v2 && databricks bundle deploy -t prod && databricks bundle run capex_copilot -t prod
# MAGIC ```
# MAGIC Set `catalog`, `schema`, `warehouse_id`, and the Lakebase project in `app-v2/databricks.yml` first.
# MAGIC After it's created once, redeploys keep the same URL.

# COMMAND ----------

print("✅ Backend install complete for", f"{CATALOG}.{SCHEMA}")
print("   Re-run this notebook any time — existing assets are updated, not duplicated.")
print("   Last step: deploy the app (cell 7).")
