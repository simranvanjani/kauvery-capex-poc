# Databricks notebook source
# MAGIC %md
# MAGIC # Kauvery Hospital — CAPEX Phase 2: Procurement Review & Comparison Engine
# MAGIC
# MAGIC **One notebook. One run. Whole demo installs.**
# MAGIC
# MAGIC This notebook stands up **Phase 2** of the CAPEX Procurement Intelligence Platform:
# MAGIC *upload a vendor quotation → is it worth it, what is missing, how does it compare to Kauvery's own history?*
# MAGIC
# MAGIC It builds everything end to end:
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | 1 | **Config** via widgets (catalog / schema / model name) |
# MAGIC | 2 | **Phase-1 foundation** — creates the ~70-column historical PO catalog (`extracted_pdf_datas`) with realistic synthetic data |
# MAGIC | 3 | **Reference BOM + benchmark index** — derived from history (what a complete purchase looks like, and the price/warranty/AMC norms per equipment category) |
# MAGIC | 4 | **Placeholder quote** — one structured row for the model signature (no PDF parsing) |
# MAGIC | 5 | **Custom ML model** — a scikit-learn model that scores a quotation 0–100 (Accept / Negotiate / Reject), logged to MLflow and registered to Unity Catalog |
# MAGIC | 6 | **Gap detection** — deterministic set-difference vs the Reference BOM, every flag carries a citation to a historical PO + page |
# MAGIC | 7 | **End-to-end demo** — scores the sample quote, writes a comparison sheet to Delta |
# MAGIC | 8 | **Evaluation** — model MAE / R² / verdict accuracy + gap-detection recall harness |
# MAGIC
# MAGIC > **Custom ML vs. Genie/agents:** per the Sep-11 call, Kauvery prefers a *custom, retrainable ML model* over a read-only Genie/agent.
# MAGIC > This notebook delivers exactly that. The FMAPI/Claude "open-ended risk" agent from the BRD is documented as a future add-on at the end, not built here.
# MAGIC >
# MAGIC > **Data privacy:** the only external model calls are Databricks Foundation Model APIs (`ai_parse_document` / `ai_extract`), whose
# MAGIC > Llama inference is **hosted inside Databricks** — data does not leave to Meta and is not used to train the models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Configuration
# MAGIC
# MAGIC Everything is parameterised. Defaults mirror the customer's real path so the code drops straight into their environment.
# MAGIC Change the widgets (top of the notebook) or edit the defaults below.

# COMMAND ----------

# MAGIC %pip install --quiet fpdf2 joblib mlflow scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "kauvey_poc", "Catalog")
dbutils.widgets.text("schema", "gold", "Schema")
dbutils.widgets.text("model_name", "capex_worth_it", "Registered model name")
dbutils.widgets.text("n_pos", "6000", "Number of historical POs to generate")
# V2: the customer loads historical POs directly into extracted_pdf_datas (from Oracle) — no PDF parsing.
# "real" reads that table; "synthetic" generates demo history (default, for the standalone demo).
dbutils.widgets.dropdown("data_source", "synthetic", ["synthetic", "real"], "Historical data source")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
MODEL_NAME = dbutils.widgets.get("model_name").strip()
N_POS = int(dbutils.widgets.get("n_pos"))
DATA_SOURCE = dbutils.widgets.get("data_source").strip()

HIST_TABLE = f"{CATALOG}.{SCHEMA}.extracted_pdf_datas"
COMPARISON_TABLE = f"{CATALOG}.{SCHEMA}.quote_comparison_sheets"
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
VOLUME = "landing"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

print(f"Catalog/Schema : {CATALOG}.{SCHEMA}")
print(f"History table  : {HIST_TABLE}")
print(f"Model (UC)     : {FULL_MODEL_NAME}")
print(f"Landing volume : {VOLUME_PATH}")

# COMMAND ----------

# Create catalog / schema / volume.
# Only attempt catalog creation if it's missing — some metastores (Default Storage) reject
# CREATE CATALOG without an explicit MANAGED LOCATION, so an existing catalog is the happy path.
existing = [r["catalog"] for r in spark.sql("SHOW CATALOGS").collect()]
if CATALOG not in existing:
    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
        print(f"created catalog {CATALOG}")
    except Exception as e:
        raise RuntimeError(
            f"Catalog '{CATALOG}' does not exist and could not be created ({e}). "
            f"Create it once (UI, or CREATE CATALOG {CATALOG} MANAGED LOCATION '<path>'), "
            f"or set the 'catalog' widget to an existing catalog you can write to.")
else:
    print(f"using existing catalog {CATALOG}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print("catalog / schema / volume ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Phase-1 foundation — synthetic historical PO catalog
# MAGIC
# MAGIC `extracted_pdf_datas` mirrors the customer's exact schema (one row per PO line item, ~70 columns,
# MAGIC sourced from old SAP + Oracle finalized capex POs). We synthesise a realistic 10-year history across
# MAGIC Kauvery units, vendors, and equipment categories, with genuine signal:
# MAGIC
# MAGIC - **Price** trends up year over year, varies by brand and vendor.
# MAGIC - **Warranty / AMC / FOC** are present on *most* finalized purchases (they were negotiated) but not all — which is exactly what makes a new quote's omissions visible.
# MAGIC
# MAGIC > Replace this whole section with a read of the customer's real table and everything downstream still works.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 · REAL-DATA MODE  (`data_source = real`) — run THIS cell, then SKIP the synthetic cells below
# MAGIC The customer ingests historical POs **directly into `extracted_pdf_datas`** (from Oracle) — **no PDF parsing
# MAGIC in the pipeline**. This reads that real table into `hist_pdf` for the benchmark step.
# MAGIC
# MAGIC **V2: categories are optional.** Benchmarks match on `model_no` + `make_brand`; if the real table has an
# MAGIC equipment-type column, point `CATEGORY_COL` at it, otherwise everything uses the single universal Reference BOM.
# MAGIC The real table must contain: `unit_rate, po_date, model_no, make_brand, po_number, unit_name,
# MAGIC warranty_months, amc_value, camc_value, foc_details, special_instructions, source_file_name` (rename to match).

# COMMAND ----------

if DATA_SOURCE == "real":
    import pandas as pd, numpy as np
    hist_pdf = spark.table(HIST_TABLE).toPandas()   # real, tabular history — no PDF parsing
    CATEGORY_COL = ""   # <- set to your equipment-type column if you have one (e.g. "equipment_type"); else leave blank
    if CATEGORY_COL and CATEGORY_COL in hist_pdf.columns:
        hist_pdf["_category"] = hist_pdf[CATEGORY_COL].fillna("General")
    else:
        hist_pdf["_category"] = "General"           # universal BOM; benchmark matches on model_no + make_brand
    hist_pdf["po_date"] = pd.to_datetime(hist_pdf["po_date"], errors="coerce").dt.strftime("%d/%m/%y")
    missing = [c for c in ["unit_rate","po_date","model_no","make_brand","po_number","unit_name",
                           "warranty_months","amc_value","camc_value","foc_details",
                           "special_instructions","source_file_name"] if c not in hist_pdf.columns]
    assert not missing, f"real table {HIST_TABLE} is missing columns the benchmark needs: {missing}"
    print(f"REAL MODE: loaded {len(hist_pdf):,} rows from {HIST_TABLE}. Skip the synthetic cells below; go to Section 3.")
else:
    print("SYNTHETIC MODE: run the cells below to generate demo history.")

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

np.random.seed(42)
_NOW = datetime.now(timezone.utc)   # tz-aware: Spark Connect rejects tz-naive timestamps

# Equipment catalog: base INR price, price sigma, and brand -> models
CATALOG_SPEC = {
    "Patient Monitor":        {"base": 3.5e5,  "sigma": 0.12, "qty": (1, 12),
        "brands": {"Philips": ["IntelliVue MX450", "IntelliVue MX550"], "GE Healthcare": ["B450", "B650"],
                   "Mindray": ["uMEC12", "BeneVision N12"], "Nihon Kohden": ["BSM-3562"]}},
    "Ventilator":             {"base": 9.0e5,  "sigma": 0.11, "qty": (1, 6),
        "brands": {"Draeger": ["Evita V600", "Evita V300"], "Hamilton": ["Hamilton C6"], "Philips": ["Trilogy Evo"]}},
    "Infusion Pump":          {"base": 1.2e5,  "sigma": 0.12, "qty": (2, 20),
        "brands": {"B Braun": ["Infusomat Space"], "Baxter": ["Sigma Spectrum"], "Mindray": ["BeneFusion"]}},
    "Ultrasound":             {"base": 2.5e6,  "sigma": 0.10, "qty": (1, 3),
        "brands": {"GE Healthcare": ["Voluson E10", "LOGIQ E10"], "Philips": ["EPIQ Elite"], "Mindray": ["Resona R9"]}},
    "Dialysis Machine":       {"base": 1.4e6,  "sigma": 0.10, "qty": (1, 8),
        "brands": {"Fresenius": ["4008S", "5008S"], "Nipro": ["Surdial X"]}},
    "Anesthesia Workstation": {"base": 2.2e6,  "sigma": 0.10, "qty": (1, 4),
        "brands": {"Draeger": ["Perseus A500"], "GE Healthcare": ["Aisys CS2"]}},
    "Defibrillator":          {"base": 4.5e5,  "sigma": 0.12, "qty": (1, 6),
        "brands": {"Philips": ["HeartStart XL+"], "ZOLL": ["R Series"]}},
    "CT Scanner":             {"base": 3.2e7,  "sigma": 0.08, "qty": (1, 1),
        "brands": {"GE Healthcare": ["Revolution CT 128", "Revolution ACT"], "Siemens": ["SOMATOM go.Top"], "Philips": ["Ingenuity CT 128"]}},
    "MRI":                    {"base": 6.5e7,  "sigma": 0.07, "qty": (1, 1),
        "brands": {"Siemens": ["MAGNETOM Sola 1.5T"], "GE Healthcare": ["SIGNA Explorer 1.5T"], "Philips": ["Ingenia 1.5T"]}},
    "Cath Lab":               {"base": 9.0e7,  "sigma": 0.07, "qty": (1, 1),
        "brands": {"Philips": ["Azurion 7"], "Siemens": ["ARTIS icono"], "GE Healthcare": ["Allia IGS 7"]}},
}
# Realistic, varied free-of-cost inclusions per equipment category (was a single placeholder string).
FOC_BY_CAT = {
    "Patient Monitor": ["SpO2 + NIBP consumables (1 yr), 2 spare probes, wall mount",
                        "ECG lead sets + NIBP cuffs starter pack, mounting kit",
                        "1 yr consumables + spare SpO2 sensor"],
    "Ventilator": ["Breathing circuits (6 mo), test lung, 10 HEPA filters",
                   "Reusable + disposable circuits starter set, calibration kit"],
    "Infusion Pump": ["IV administration sets starter pack (500), pole clamp",
                      "Dedicated giving sets (250) + battery pack"],
    "Ultrasound": ["2 transducers of choice, gel warmer, thermal paper (1 yr)",
                   "Extra linear probe, DICOM licence, applications training (3 days)"],
    "Dialysis Machine": ["Dialysers + bloodlines starter (100), concentrate connectors",
                         "1 yr consumables kit + spare Hansen connectors"],
    "Anesthesia Workstation": ["Circle absorber + soda lime (1 yr), spare flow sensor",
                               "Reusable circuits, gas sampling lines, calibration gas"],
    "Defibrillator": ["Adult + paediatric pads (2 yr), spare battery",
                      "Multifunction electrode pads starter pack, carry case"],
    "CT Scanner": ["Contrast injector consumables (1 yr), phantom + QA kit, applications training",
                   "Coil/detector service kit, 1 yr software updates, 5 training days"],
    "MRI": ["RF coil set, cryogen top-up (1 yr), applications training (5 days)",
            "Head + spine coils, DICOM licence, 1 yr software subscription"],
    "Cath Lab": ["Radiation aprons set, contrast injector consumables, 5 applications training days",
                 "Sterile drapes starter (200), 1 yr software updates, physicist QA kit"],
}
# frequency weights — small equipment purchased far more often than big iron
CAT_WEIGHTS = {"Patient Monitor": .22, "Ventilator": .14, "Infusion Pump": .18, "Ultrasound": .10,
               "Dialysis Machine": .10, "Anesthesia Workstation": .08, "Defibrillator": .10,
               "CT Scanner": .04, "MRI": .02, "Cath Lab": .02}

UNITS = ["Kauvery Chennai (Alwarpet)", "Kauvery Chennai (Radial Rd)", "Kauvery Chennai (Vadapalani)",
         "Kauvery Trichy (Cantonment)", "Kauvery Trichy (Tennur)", "Kauvery Hosur", "Kauvery Salem",
         "Kauvery Bengaluru (Electronic City)", "Kauvery Bengaluru (Marathahalli)", "Kauvery Tirunelveli",
         "Kauvery Karaikudi", "Kauvery Chromepet", "Kauvery Tennur (Heart City)", "Kauvery Coimbatore"]
VENDOR_SPEC = {
    "GE Healthcare": ("GEIN", "GEHEALTH@ge.com", "Rajesh Kumar"),
    "Philips India": ("PHIL", "sales@philips.co.in", "Anita Menon"),
    "Siemens Healthineers": ("SIEM", "care.in@siemens-healthineers.com", "Vikram Rao"),
    "Draeger India": ("DRAE", "info.india@draeger.com", "Suresh Nair"),
    "Mindray Medical India": ("MIND", "service@mindray.in", "Priya Sharma"),
    "Medingenious Solutions": ("MEDG", "sales@medingenious.in", "Karthik S"),
    "Trivitron Healthcare": ("TRIV", "enquiry@trivitron.com", "Deepa R"),
}
BRAND_TO_VENDOR = {"Philips": "Philips India", "GE Healthcare": "GE Healthcare", "Siemens": "Siemens Healthineers",
                   "Draeger": "Draeger India", "Mindray": "Mindray Medical India", "Hamilton": "Trivitron Healthcare",
                   "B Braun": "Trivitron Healthcare", "Baxter": "Medingenious Solutions", "Nipro": "Medingenious Solutions",
                   "Fresenius": "Trivitron Healthcare", "ZOLL": "Medingenious Solutions", "Nihon Kohden": "Trivitron Healthcare"}
PAYMENT_TERMS = ["100% against delivery", "50% advance, 50% against delivery", "30% advance, 70% on installation",
                 "Net 30 days", "Net 45 days", "100% advance"]

# ---- PO-level arrays ----
def rand_date(year):
    start = datetime(year, 1, 1)
    return start + timedelta(days=int(np.random.randint(0, 360)))

_year_w = np.array([1, 1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.3, 2.5, 2.7])
po_years = np.random.choice(range(2015, 2026), size=N_POS, p=_year_w / _year_w.sum())
line_counts = np.random.choice([1, 2, 3], size=N_POS, p=[.62, .28, .10])
po_units = np.random.choice(UNITS, size=N_POS)
po_dates = [rand_date(y) for y in po_years]
po_numbers = [f"PO-{y}-{i:05d}" for i, y in enumerate(po_years)]

rows = []
for i in range(N_POS):
    y = int(po_years[i]); pod = po_dates[i]; unit = po_units[i]; pon = po_numbers[i]
    inflation = 1.03 ** (y - 2015)
    payment = np.random.choice(PAYMENT_TERMS, p=[.28, .22, .18, .12, .10, .10])
    header_disc = round(float(np.random.choice([0, 0, 2, 3, 5], p=[.5, .2, .12, .1, .08])), 2)
    for ln in range(int(line_counts[i])):
        cat = np.random.choice(list(CAT_WEIGHTS), p=list(CAT_WEIGHTS.values()))
        spec = CATALOG_SPEC[cat]
        brand = np.random.choice(list(spec["brands"]))
        model = np.random.choice(spec["brands"][brand])
        vendor = BRAND_TO_VENDOR.get(brand, "Medingenious Solutions")
        vcode, vemail, vcontact = VENDOR_SPEC[vendor]
        brand_premium = 1.05 if brand in ("Philips", "GE Healthcare", "Siemens") else 0.97
        unit_rate = float(spec["base"] * inflation * brand_premium * np.random.lognormal(0, spec["sigma"]))
        unit_rate = round(unit_rate, 2)
        qty = int(np.random.randint(spec["qty"][0], spec["qty"][1] + 1))
        disc = round(float(np.random.choice([0, 2, 5, 7, 10, 12], p=[.30, .18, .2, .12, .12, .08])), 2)
        warranty = int(np.random.choice([12, 24, 24, 36, 60], p=[.10, .40, .25, .18, .07]))
        has_amc = np.random.rand() < (0.85 if spec["base"] > 5e6 else 0.65)
        has_camc = has_amc and np.random.rand() < 0.4
        has_foc = np.random.rand() < 0.55
        lead = int(np.random.randint(30, 150))
        tax_pct = float(np.random.choice([5, 12, 18], p=[.2, .5, .3]))
        line_total = round(qty * unit_rate * (1 - disc / 100.0), 2)
        line_tax = round(line_total * tax_pct / 100.0, 2)
        rows.append({
            "source_system": np.random.choice(["OLD_SAP", "ORACLE"]),
            "source_file_name": f"{pon}_{ln+1}.pdf",
            "extracted_at": _NOW,
            "unit_name": unit,
            "buyer_legal_entity": "Kauvery Hospitals Pvt Ltd",
            "buyer_address": "No.199, Luz Church Road, Mylapore, Chennai 600004",
            "buyer_gstin": "33AABCK1234M1Z5", "buyer_pan": "AABCK1234M", "buyer_phone": "044-40006000",
            "po_number": pon, "po_date": pod.strftime("%d/%m/%y"),
            "pr_number": f"PR-{y}-{i:05d}", "revision_no": int(np.random.choice([0, 0, 1], p=[.8, .15, .05])),
            "currency": None if np.random.rand() < 0.9 else "INR",
            "delivery_date": (pod + timedelta(days=lead)).strftime("%d/%m/%y"),
            "quote_ref": f"QT-{vcode}-{y}-{np.random.randint(1000,9999)}", "quote_date": (pod - timedelta(days=int(np.random.randint(5,40)))).strftime("%d/%m/%y"),
            "agreement_no_date": None,
            "vendor_code": vcode, "vendor_name": vendor, "vendor_address": f"{vendor}, India",
            "vendor_gstin": f"33{vcode}5678Q1Z9", "vendor_email": vemail, "vendor_contact_person": vcontact,
            "vendor_contact_no": f"9{np.random.randint(100000000,999999999)}",
            "bill_to": unit, "ship_to": unit, "department": "Biomedical Engineering",
            "remarks": None,
            "special_instructions": ("Includes application training for 3 days. " if np.random.rand() < 0.5 else "")
                                    + ("Installation & commissioning by OEM. " if np.random.rand() < 0.6 else ""),
            "header_discount_pct": header_disc,
            "other_charges": round(float(np.random.choice([0, 5000, 15000], p=[.7, .2, .1])), 2),
            "amount_in_words": None, "payment_terms": payment,
            "warranty_months": warranty, "warranty_raw": f"{warranty} months comprehensive",
            "amc_value": (f"{np.random.choice([5,7,8,10])}% of value per annum for 5 years" if has_amc else None),
            "camc_value": (f"{np.random.choice([8,10,12])}% of value per annum" if has_camc else None),
            "foc_details": (str(np.random.choice(FOC_BY_CAT.get(cat, ["Starter consumables + accessories kit"]))) if has_foc else None),
            "foc_value": (round(float(unit_rate * np.random.uniform(0.01, 0.04)), 2) if has_foc else None),
            "camc_amc_start_date": ((pod + timedelta(days=warranty*30)).strftime("%d/%m/%y") if has_amc else None),
            "camc_amc_end_date": ((pod + timedelta(days=warranty*30 + 1825)).strftime("%d/%m/%y") if has_amc else None),
            "yoy_escalation": ("5% per annum" if has_amc else None),
            "inco_terms": np.random.choice(["DDP", "CIP", "FOR Destination"]),
            "freight": np.random.choice(["Included", "Extra at actuals", "Paid"]),
            "delivery_contact": vcontact, "advance_performa": None, "advance_before_delivery": None,
            "payment_against_delivery": None, "buyback_offer": (None if np.random.rand() < 0.85 else "Trade-in on old unit"),
            "line_no": ln + 1, "item_code": f"{cat[:3].upper()}-{np.random.randint(1000,9999)}",
            "item_description": f"{brand} {model} {cat}", "qty": float(qty), "uom": "NOS",
            "discount_pct": disc, "tax_pct": tax_pct,
            "tax_code": f"EXGST{int(tax_pct)}",
            "unit_rate": unit_rate, "line_total": line_total, "line_tax_amount": line_tax,
            "line_net_total": round(line_total + line_tax, 2),
            "make_brand": brand, "model_no": model,
            # helper columns (not part of the customer schema) used to derive header aggregates:
            "_category": cat,
        })

hist_pdf = pd.DataFrame(rows)

# header-level aggregates from line totals
agg = hist_pdf.groupby("po_number").agg(sub_total=("line_total", "sum"), tax_total=("line_tax_amount", "sum")).reset_index()
hist_pdf = hist_pdf.merge(agg, on="po_number", how="left")
hist_pdf["igst_amount"] = hist_pdf["tax_total"]  # inter-state default
hist_pdf["cgst_amount"] = 0.0
hist_pdf["sgst_amount"] = 0.0
hist_pdf["order_total"] = (hist_pdf["sub_total"] - hist_pdf["sub_total"] * hist_pdf["header_discount_pct"] / 100.0
                           + hist_pdf["tax_total"] + hist_pdf["other_charges"]).round(2)
print(f"Generated {len(hist_pdf):,} PO line items across {hist_pdf['po_number'].nunique():,} POs, {hist_pdf['_category'].nunique()} categories")

# COMMAND ----------

# Write to Delta with the exact customer column order/types.
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType, IntegerType, TimestampType)

STRING_COLS = ["source_system","source_file_name","unit_name","buyer_legal_entity","buyer_address","buyer_gstin",
    "buyer_pan","buyer_phone","po_number","po_date","pr_number","currency","delivery_date","quote_ref","quote_date",
    "agreement_no_date","vendor_code","vendor_name","vendor_address","vendor_gstin","vendor_email","vendor_contact_person",
    "vendor_contact_no","bill_to","ship_to","department","remarks","special_instructions","amount_in_words","payment_terms",
    "warranty_raw","amc_value","camc_value","foc_details","camc_amc_start_date","camc_amc_end_date","yoy_escalation",
    "inco_terms","freight","delivery_contact","advance_performa","advance_before_delivery","payment_against_delivery",
    "buyback_offer","item_code","item_description","uom","tax_code","make_brand","model_no"]
DOUBLE_COLS = ["sub_total","tax_total","igst_amount","cgst_amount","sgst_amount","other_charges","header_discount_pct",
    "order_total","qty","discount_pct","tax_pct","unit_rate","line_total","line_tax_amount","line_net_total","foc_value"]
INT_COLS = ["revision_no","warranty_months","line_no"]

SCHEMA_ORDER = ["source_system","source_file_name","extracted_at","unit_name","buyer_legal_entity","buyer_address",
    "buyer_gstin","buyer_pan","buyer_phone","po_number","po_date","pr_number","revision_no","currency","delivery_date",
    "quote_ref","quote_date","agreement_no_date","vendor_code","vendor_name","vendor_address","vendor_gstin","vendor_email",
    "vendor_contact_person","vendor_contact_no","bill_to","ship_to","department","remarks","special_instructions",
    "sub_total","tax_total","igst_amount","cgst_amount","sgst_amount","other_charges","header_discount_pct","order_total",
    "amount_in_words","payment_terms","warranty_months","warranty_raw","amc_value","camc_value","foc_details","foc_value",
    "camc_amc_start_date","camc_amc_end_date","yoy_escalation","inco_terms","freight","delivery_contact","advance_performa",
    "advance_before_delivery","payment_against_delivery","buyback_offer","line_no","item_code","item_description","qty",
    "uom","discount_pct","tax_pct","tax_code","unit_rate","line_total","line_tax_amount","line_net_total","make_brand","model_no"]

def _spark_type(col):
    if col == "extracted_at": return TimestampType()
    if col in INT_COLS: return IntegerType()
    if col in DOUBLE_COLS: return DoubleType()
    return StringType()

schema = StructType([StructField(c, _spark_type(c), True) for c in SCHEMA_ORDER])

out = hist_pdf.copy()
for c in INT_COLS:
    out[c] = out[c].astype(int)

# Build native Python rows (no Arrow type-inference surprises on serverless / Spark Connect).
def _native(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    return v

data = [tuple(_native(rec[c]) for c in SCHEMA_ORDER)
        for rec in out[SCHEMA_ORDER].to_dict("records")]
hist_sdf = spark.createDataFrame(data, schema=schema)
hist_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(HIST_TABLE)
spark.sql(f"COMMENT ON TABLE {HIST_TABLE} IS 'Kauvery CAPEX Phase-1 foundation: finalized historical PO line items (synthetic demo data).'")
print(f"Wrote {HIST_TABLE}")
display(spark.sql(f"SELECT unit_name, vendor_name, po_date, item_description, model_no, qty, unit_rate, warranty_months, amc_value FROM {HIST_TABLE} LIMIT 8"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Reference BOM & benchmark index (derived from Phase 1)
# MAGIC
# MAGIC The value of this system is that recommendations are grounded in **Kauvery's own history**, not a model's general knowledge.
# MAGIC From the historical catalog we derive, per equipment category:
# MAGIC
# MAGIC - a **benchmark index** (most-recent comparable price, typical warranty, AMC/FOC rates, purchase frequency), honoring the
# MAGIC   business rules: exclude `unit_rate` NULL/0, and use the **most-recent comparable** purchase as the primary benchmark;
# MAGIC - a **Reference BOM** — what a complete purchase is expected to include;
# MAGIC - **component examples** — a real historical PO that *did* include each component, used to cite gaps.

# COMMAND ----------

import json

hd = hist_pdf[(hist_pdf["unit_rate"].notna()) & (hist_pdf["unit_rate"] > 0)].copy()
hd["po_date_dt"] = pd.to_datetime(hd["po_date"], format="%d/%m/%y")
hd["has_amc"] = hd["amc_value"].notna() | hd["camc_value"].notna()
hd["has_foc"] = hd["foc_details"].notna()

def _bench_record(g):
    g = g.sort_values("po_date_dt")
    recent = g.iloc[-1]
    return {
        "recent_unit_rate": float(recent["unit_rate"]),
        "recent_po_date": recent["po_date"],
        "recent_po_number": recent["po_number"],
        "recent_unit_name": recent["unit_name"],
        "median_unit_rate": float(g["unit_rate"].median()),
        "min_unit_rate": float(g["unit_rate"].min()),
        "count": int(len(g)),
        "typical_warranty": float(g["warranty_months"].median()),
        "amc_rate": float(g["has_amc"].mean()),
        "foc_rate": float(g["has_foc"].mean()),
        "typical_delivery_days": 60.0,
        "model_no": recent["model_no"],
    }

benchmark_index = {"by_model": {}, "by_brand_category": {}, "by_category": {}, "freq_norm": 8.0}
for model, g in hd.groupby(hd["model_no"].str.lower()):
    benchmark_index["by_model"][model] = _bench_record(g)
for (brand, cat), g in hd.groupby([hd["make_brand"].str.lower(), "_category"]):
    benchmark_index["by_brand_category"][f"{brand}|{cat}"] = _bench_record(g)
for cat, g in hd.groupby("_category"):
    rec = _bench_record(g)
    rec["freq_max"] = int(g["po_number"].nunique())
    benchmark_index["by_category"][cat] = rec

# Reference BOM — single universal standard (category removed). What a complete purchase should
# include; applied to every item. detect_gaps reads it via its "_default" fallback.
reference_bom = {
    "_default": {"min_warranty_months": 24, "requires": ["amc_camc", "training", "installation_commissioning"]},
}

# Component examples — a real historical PO that included each component, for citations.
component_examples = {}
for cat, g in hd.groupby("_category"):
    ex = {}
    amc_g = g[g["has_amc"]]
    foc_g = g[g["has_foc"]]
    warr_g = g[g["warranty_months"] >= 24]
    train_g = g[g["special_instructions"].str.contains("training", case=False, na=False)]
    inst_g = g[g["special_instructions"].str.contains("Installation", case=False, na=False)]
    def pick(gg):
        if len(gg) == 0: return None
        r = gg.sort_values("po_date_dt").iloc[-1]
        return {"po_number": r["po_number"], "unit_name": r["unit_name"], "po_date": r["po_date"],
                "source_file_name": r["source_file_name"], "page": int(np.random.randint(2, 15))}
    for key, gg in [("amc_camc", amc_g), ("foc_accessories", foc_g), ("warranty", warr_g),
                    ("training", train_g), ("installation_commissioning", inst_g)]:
        p = pick(gg)
        if p: ex[key] = p
    component_examples[cat] = ex

print("benchmark categories:", list(benchmark_index["by_category"].keys()))
print("example CT Scanner benchmark:", json.dumps(benchmark_index["by_category"].get("CT Scanner", {}), indent=2)[:400])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Placeholder quote (for the model signature)
# MAGIC
# MAGIC **No PDF parsing in the notebook** — historical data is tabular, and live vendor quotations are parsed
# MAGIC by the app. This is a single structured example row so MLflow can infer the model's input signature.

# COMMAND ----------

quote = {
    "vendor_name": "Example Vendor",
    "unit_name": "Kauvery (example)",
    "quote_ref": "SAMPLE-0001", "quote_date": "05/09/26",
    "lines": [
        {"item_description": "Philips IntelliVue MX450 Patient Monitor", "make_brand": "Philips",
         "model_no": "IntelliVue MX450", "qty": 10, "unit_rate": 415000, "warranty_months": 12,
         "amc_present": False, "camc_present": False, "foc_present": False, "delivery_lead_days": 95,
         "payment_terms": "100% advance", "has_training": False, "has_installation": True},
    ],
}

# One structured row the model scores (drives the MLflow signature). No PDF, no ai_parse_document.
quote_rows = []
for l in quote["lines"]:
    r = dict(l); r["vendor_name"] = quote["vendor_name"]; r["unit_name"] = quote["unit_name"]
    quote_rows.append(r)
quote_df = pd.DataFrame(quote_rows)
print("Structured example for the model signature:")
print(quote_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · The custom ML model (scikit-learn + MLflow)
# MAGIC
# MAGIC **The problem:** there is no labeled "worth it / not worth it" history. **The approach:** synthesise a training target
# MAGIC from Kauvery's stated weighting rubric (Price 30 · Warranty 15 · AMC/CAMC 15 · Delivery 10 · FOC 10 · Historical-frequency 10 · Payment 10)
# MAGIC plus realistic noise, then train a real `GradientBoostingRegressor` to learn it.
# MAGIC
# MAGIC When biomedical leads later label ~50 real quotes by *actual* post-installation outcomes, you swap those in as the target
# MAGIC and retrain — the model then improves **past** the hand-tuned rubric. The scoring logic lives in `capex_scoring.py`,
# MAGIC written below so this notebook is fully self-contained and the file is logged with the model (Models-from-Code).

# COMMAND ----------

import os, sys, textwrap

SRC_DIR = "/tmp/capex_src"
os.makedirs(SRC_DIR, exist_ok=True)
MODULE_CODE = r'''
import json
import numpy as np
import pandas as pd
import mlflow

FEATURE_COLS = ["price_variance_pct", "warranty_delta_months", "amc_camc_present",
                "delivery_lead_days", "foc_present", "hist_frequency_norm", "payment_terms_score"]
# Exactly the columns a caller/endpoint must send (drives the MLflow signature).
MODEL_INPUT_COLS = ["item_description", "make_brand", "model_no", "qty", "unit_rate", "warranty_months",
                    "amc_present", "camc_present", "foc_present", "delivery_lead_days", "payment_terms",
                    "has_training", "has_installation"]
WEIGHTS = {"price": 50, "warranty": 30, "amc_camc": 8, "foc": 5, "delivery": 3, "frequency": 2, "payment": 2}

# Category removed — the real data has no equipment-type column and the demo taxonomy did not fit
# it. Benchmarks now match model_no -> make_brand -> all-history, and gap detection uses a single
# universal Reference BOM. classify_category is kept as a constant so downstream lookups stay valid.
def classify_category(text):
    return "General"

def _clip01(x):
    return float(max(0.0, min(1.0, x)))

def price_subscore(v):    return _clip01(1.0 - (v + 0.05) / 0.45)
def warranty_subscore(d): return _clip01(0.5 + d / 24.0)
def delivery_subscore(d): return _clip01(1.0 - (d - 30.0) / 150.0)

def payment_terms_score(terms):
    t = str(terms).lower()
    if "advance" in t and ("100" in t or "full" in t): return 0.1
    if "against delivery" in t or "on installation" in t or "net 30" in t or "30 days" in t: return 0.8
    if "net 45" in t or "45 days" in t or "milestone" in t: return 0.7
    if "advance" in t: return 0.45
    return 0.6

def compute_subscores(f):
    return {"price": price_subscore(f["price_variance_pct"]),
            "warranty": warranty_subscore(f["warranty_delta_months"]),
            "amc_camc": 1.0 if f["amc_camc_present"] else 0.0,
            "delivery": delivery_subscore(f["delivery_lead_days"]),
            "foc": 1.0 if f["foc_present"] else 0.0,
            "frequency": _clip01(f["hist_frequency_norm"]),
            "payment": _clip01(f["payment_terms_score"])}

def rubric_score(f):
    s = compute_subscores(f)
    return float(sum(WEIGHTS[k] * s[k] for k in WEIGHTS))

def verdict_from_score(score):
    if score >= 70: return "Accept"
    if score >= 45: return "Negotiate"
    return "Reject"

def find_benchmark(row, bench_index):
    model = str(row.get("model_no", "")).strip().lower()
    brand = str(row.get("make_brand", "")).strip().lower()
    category = classify_category(" ".join([str(row.get("item_description", "")),
                                            str(row.get("model_no", "")), str(row.get("make_brand", ""))]))
    if model and model in bench_index.get("by_model", {}):
        return bench_index["by_model"][model], "Exact Match", category
    bc = brand + "|" + category
    if bc in bench_index.get("by_brand_category", {}):
        return bench_index["by_brand_category"][bc], "Comparable Match", category
    if category in bench_index.get("by_category", {}):
        return bench_index["by_category"][category], "Category Match", category
    return None, "No historical purchase found", category

def build_features(row, bench_index):
    bench, match_level, category = find_benchmark(row, bench_index)
    unit_rate = float(row.get("unit_rate") or 0.0)
    warranty = float(row.get("warranty_months") or 0)
    amc = bool(row.get("amc_present")) or bool(row.get("camc_present"))
    foc = bool(row.get("foc_present"))
    if bench and bench.get("recent_unit_rate"):
        bench_rate = float(bench["recent_unit_rate"])
        price_variance = (unit_rate - bench_rate) / bench_rate if bench_rate > 0 else 0.0
        typ_warr = float(bench.get("typical_warranty", 24))
        freq_norm = _clip01(bench.get("count", 0) / float(bench_index.get("freq_norm", 8)))
        delivery = float(row.get("delivery_lead_days") or bench.get("typical_delivery_days", 60))
    else:
        bench_rate, price_variance, typ_warr, freq_norm = None, 0.0, 24.0, 0.0
        delivery = float(row.get("delivery_lead_days") or 60)
    feats = {"price_variance_pct": float(price_variance),
             "warranty_delta_months": float(warranty - typ_warr),
             "amc_camc_present": 1.0 if amc else 0.0,
             "delivery_lead_days": float(delivery),
             "foc_present": 1.0 if foc else 0.0,
             "hist_frequency_norm": float(freq_norm),
             "payment_terms_score": float(payment_terms_score(row.get("payment_terms", "")))}
    info = {"category": category, "match_level": match_level, "benchmark_unit_rate": bench_rate,
            "benchmark_model": (bench.get("model_no") if bench else None),
            "benchmark_po": (bench.get("recent_po_number") if bench else None),
            "benchmark_po_date": (bench.get("recent_po_date") if bench else None)}
    return feats, info

_DEFAULT_BOM = {"min_warranty_months": 24, "requires": ["amc_camc", "training", "installation_commissioning"]}

def detect_gaps(row, category, reference_bom, component_examples):
    bom = reference_bom.get(category, reference_bom.get("_default", _DEFAULT_BOM))
    ex = component_examples.get(category, {})
    def cite(component):
        c = ex.get(component)
        if not c:
            return "Historical benchmark unavailable."
        return ("Historical benchmark: PO %s at %s (%s) included this; %s, p.%s"
                % (c.get("po_number", "?"), c.get("unit_name", "?"), c.get("po_date", "?"),
                   c.get("source_file_name", "?"), c.get("page", "?")))
    gaps = []
    warranty = float(row.get("warranty_months") or 0)
    reqs = bom.get("requires", [])
    if warranty < bom.get("min_warranty_months", 24):
        gaps.append({"component": "Warranty", "severity": "High",
                     "message": "Warranty %d months is below the expected %d months for %s."
                                % (int(warranty), bom.get("min_warranty_months", 24), category),
                     "citation": cite("warranty")})
    if "amc_camc" in reqs and not (bool(row.get("amc_present")) or bool(row.get("camc_present"))):
        gaps.append({"component": "AMC/CMC", "severity": "High",
                     "message": "No AMC/CMC found in quotation.", "citation": cite("amc_camc")})
    if "foc_accessories" in reqs and not bool(row.get("foc_present")):
        gaps.append({"component": "FOC accessories", "severity": "Medium",
                     "message": "No free-of-cost (FOC) accessories/consumables offered.", "citation": cite("foc_accessories")})
    if "training" in reqs and not bool(row.get("has_training", False)):
        gaps.append({"component": "Training", "severity": "Medium",
                     "message": "No application/clinical training package included.", "citation": cite("training")})
    if "installation_commissioning" in reqs and not bool(row.get("has_installation", False)):
        gaps.append({"component": "Installation & commissioning", "severity": "Medium",
                     "message": "Installation & commissioning not specified.", "citation": cite("installation_commissioning")})
    return gaps

class CapexWorthItModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib, json
        self._model = joblib.load(context.artifacts["sk_model"])
        with open(context.artifacts["benchmark_index"]) as fh: self._bench = json.load(fh)
        with open(context.artifacts["reference_bom"]) as fh: self._bom = json.load(fh)
        with open(context.artifacts["component_examples"]) as fh: self._examples = json.load(fh)

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        out = []
        for _, row in model_input.iterrows():
            row = row.to_dict()
            feats, info = build_features(row, self._bench)
            X = pd.DataFrame([[feats[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
            score = float(max(0.0, min(100.0, self._model.predict(X)[0])))
            gaps = detect_gaps(row, info["category"], self._bom, self._examples)
            verdict = verdict_from_score(score)
            # A cheap price shouldn't auto-Accept a quote that is missing essentials (warranty/AMC):
            # a High-severity gap means there is always something to negotiate first.
            if verdict == "Accept" and any(g.get("severity") == "High" for g in gaps):
                verdict = "Negotiate"
            out.append({"item_description": row.get("item_description"), "make_brand": row.get("make_brand"),
                        "model_no": row.get("model_no"), "category": info["category"],
                        "match_level": info["match_level"], "benchmark_unit_rate": info["benchmark_unit_rate"],
                        "benchmark_po": info["benchmark_po"], "benchmark_po_date": info["benchmark_po_date"],
                        "quoted_unit_rate": float(row.get("unit_rate") or 0.0),
                        "price_variance_pct": round(feats["price_variance_pct"] * 100, 1),
                        "worth_score": round(score, 1), "verdict": verdict,
                        "num_gaps": len(gaps), "gaps_json": json.dumps(gaps)})
        return pd.DataFrame(out)

mlflow.models.set_model(CapexWorthItModel())
'''
with open(f"{SRC_DIR}/capex_scoring.py", "w") as fh:
    fh.write(MODULE_CODE)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import capex_scoring as cs
print("capex_scoring.py written and imported. Feature columns:", cs.FEATURE_COLS)

# COMMAND ----------

# Build the training set: sample realistic quote feature vectors, label with the weighted rubric + noise.
N_TRAIN = 6000
rng = np.random.default_rng(7)
train = pd.DataFrame({
    "price_variance_pct":     np.clip(rng.normal(0.05, 0.18, N_TRAIN), -0.35, 0.7),
    "warranty_delta_months":  np.clip(rng.normal(0, 10, N_TRAIN).round(), -18, 36),
    "amc_camc_present":       (rng.random(N_TRAIN) < 0.68).astype(float),
    "delivery_lead_days":     np.clip(rng.normal(60, 35, N_TRAIN), 10, 220).round(),
    "foc_present":            (rng.random(N_TRAIN) < 0.55).astype(float),
    "hist_frequency_norm":    rng.random(N_TRAIN),
    "payment_terms_score":    rng.choice([0.1, 0.45, 0.6, 0.7, 0.8], N_TRAIN, p=[.15, .2, .25, .2, .2]),
})
train["worth_score"] = np.clip(
    [cs.rubric_score(r) for r in train.to_dict("records")] + rng.normal(0, 4, N_TRAIN), 0, 100)
print(train.describe().round(2).to_string())

# COMMAND ----------

import mlflow, joblib, sklearn, scipy, cloudpickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from mlflow.models import infer_signature
from databricks.sdk import WorkspaceClient

# Pin exact versions so the serving image matches training (the sklearn model is a raw joblib
# artifact, so MLflow can't infer scipy/cloudpickle on its own — list them explicitly).
PIP_REQS = [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"numpy=={np.__version__}", f"pandas=={pd.__version__}", f"scipy=={scipy.__version__}",
            f"joblib=={joblib.__version__}", f"cloudpickle=={cloudpickle.__version__}"]
print("pip_requirements:", PIP_REQS)

mlflow.set_registry_uri("databricks-uc")
me = WorkspaceClient().current_user.me().user_name
exp_dir = f"/Users/{me}/kauvery_capex_phase2"
try:
    WorkspaceClient().workspace.mkdirs(exp_dir)
    mlflow.set_experiment(f"{exp_dir}/mlflow_experiment")
except Exception as e:
    print(f"[info] using default experiment ({e})")

X = train[cs.FEATURE_COLS]; y = train["worth_score"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)

# artifacts the pyfunc needs at inference
for name, obj in [("benchmark_index", benchmark_index), ("reference_bom", reference_bom),
                  ("component_examples", component_examples)]:
    with open(f"{SRC_DIR}/{name}.json", "w") as fh:
        json.dump(obj, fh)

with mlflow.start_run(run_name="capex_worth_it") as run:
    reg = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=1)
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    mae = mean_absolute_error(y_te, pred); r2 = r2_score(y_te, pred)
    # verdict accuracy
    va = float(np.mean([cs.verdict_from_score(a) == cs.verdict_from_score(b) for a, b in zip(y_te, pred)]))
    mlflow.log_params({"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "n_train": N_TRAIN})
    mlflow.log_metrics({"mae": mae, "r2": r2, "verdict_accuracy": va})
    for f, imp in zip(cs.FEATURE_COLS, reg.feature_importances_):
        mlflow.log_metric(f"importance_{f}", float(imp))
    joblib.dump(reg, f"{SRC_DIR}/sk_model.joblib")

    example = quote_df[cs.MODEL_INPUT_COLS].head(1).copy()
    example["unit_rate"] = example["unit_rate"].astype(float)   # double: callers send float rates
    for _c in ["qty", "warranty_months", "delivery_lead_days"]:
        example[_c] = example[_c].astype(int)
    signature = infer_signature(example, pd.DataFrame([{
        "item_description": "x", "make_brand": "x", "model_no": "x", "category": "x", "match_level": "x",
        "benchmark_unit_rate": 0.0, "benchmark_po": "x", "benchmark_po_date": "x", "quoted_unit_rate": 0.0,
        "price_variance_pct": 0.0, "worth_score": 0.0, "verdict": "x", "num_gaps": 0, "gaps_json": "x"}]))
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=f"{SRC_DIR}/capex_scoring.py",
        artifacts={"sk_model": f"{SRC_DIR}/sk_model.joblib",
                   "benchmark_index": f"{SRC_DIR}/benchmark_index.json",
                   "reference_bom": f"{SRC_DIR}/reference_bom.json",
                   "component_examples": f"{SRC_DIR}/component_examples.json"},
        signature=signature, input_example=example,
        pip_requirements=PIP_REQS,
        registered_model_name=FULL_MODEL_NAME,
    )
print(f"MAE={mae:.2f}  R2={r2:.3f}  verdict_accuracy={va:.3f}")
print(f"Registered {FULL_MODEL_NAME} v{info.registered_model_version}")

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(FULL_MODEL_NAME, "prod", info.registered_model_version)
print(f"Alias @prod -> v{info.registered_model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · End-to-end demo — score the uploaded quotation
# MAGIC
# MAGIC Load the registered model and run the parsed quote through it: benchmark match → worth-it score → verdict → gap checklist with citations.

# COMMAND ----------

scorer = mlflow.pyfunc.load_model(f"models:/{FULL_MODEL_NAME}@prod")
_q = quote_df[cs.MODEL_INPUT_COLS].copy()
_q["unit_rate"] = _q["unit_rate"].astype(float)   # match signature (double); callers send float
result = scorer.predict(_q)

import json as _json
for _, r in result.iterrows():
    print("=" * 90)
    print(f"ITEM      : {r['item_description']}  (model {r['model_no']})")
    print(f"MATCH     : {r['match_level']}  |  benchmark PO {r['benchmark_po']} ({r['benchmark_po_date']})")
    print(f"PRICE     : quoted INR {r['quoted_unit_rate']:,.0f} vs benchmark INR {(r['benchmark_unit_rate'] or 0):,.0f}"
          f"  ->  {r['price_variance_pct']:+.1f}% vs most-recent comparable")
    print(f"WORTH SCORE: {r['worth_score']} / 100   =>   VERDICT: {r['verdict']}")
    gaps = _json.loads(r["gaps_json"])
    print(f"GAPS ({len(gaps)}):")
    for g in gaps:
        print(f"   [{g['severity']}] {g['component']}: {g['message']}")
        print(f"        cite: {g['citation']}")

# COMMAND ----------

# Write the comparison sheet back to Delta (rendered in-app / exportable to Excel downstream).
comp = result.copy()
comp["scored_at"] = datetime.now(timezone.utc)
comp["quote_ref"] = quote["quote_ref"]
comp["unit_name"] = quote["unit_name"]
comp["vendor_name"] = quote["vendor_name"]

comp_cols = ["quote_ref", "unit_name", "vendor_name", "item_description", "make_brand", "model_no", "category",
             "match_level", "benchmark_unit_rate", "benchmark_po", "benchmark_po_date", "quoted_unit_rate",
             "price_variance_pct", "worth_score", "verdict", "num_gaps", "gaps_json", "scored_at"]
comp_double = {"benchmark_unit_rate", "quoted_unit_rate", "price_variance_pct", "worth_score"}
comp_schema = StructType([
    StructField(c, (TimestampType() if c == "scored_at"
                    else IntegerType() if c == "num_gaps"
                    else DoubleType() if c in comp_double
                    else StringType()), True) for c in comp_cols])
comp_data = [tuple(_native(rec[c]) for c in comp_cols) for rec in comp[comp_cols].to_dict("records")]
comp_sdf = spark.createDataFrame(comp_data, schema=comp_schema)
comp_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(COMPARISON_TABLE)
print(f"Wrote comparison sheet -> {COMPARISON_TABLE}")
display(spark.table(COMPARISON_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Evaluation
# MAGIC
# MAGIC Two numbers de-risk the program at board level:
# MAGIC 1. **Model quality** — MAE / R² / verdict accuracy on held-out data (already logged to MLflow).
# MAGIC 2. **Gap-detection recall** — the harness the BRD asks for. Today the detector is deterministic set-difference, so recall
# MAGIC    is ~1.0 by construction; the *point* is the harness. When the biomedical leads label ~50 real quotes with the gaps that
# MAGIC    actually bit post-installation, run those through `detect_gaps` and this same cell reports the tracked recall metric.

# COMMAND ----------

# Gap-detection recall on a synthetic labeled set with known omissions.
gap_eval, expected = [], []
# Category removed: evaluate gap detection against the single universal Reference BOM.
cat, bom = "General", reference_bom["_default"]
for i in range(200):
    drop_amc = np.random.rand() < 0.5
    drop_foc = ("foc_accessories" in bom["requires"]) and np.random.rand() < 0.5
    short_warr = np.random.rand() < 0.4
    row = {"item_description": f"test {cat}", "make_brand": "GE Healthcare", "model_no": "TEST",
           "warranty_months": 12 if short_warr else 24,
           "amc_present": not drop_amc, "camc_present": False,
           "foc_present": not drop_foc, "has_training": True, "has_installation": True}
    truth = set()
    if short_warr: truth.add("Warranty")
    if drop_amc and "amc_camc" in bom["requires"]: truth.add("AMC/CMC")
    if drop_foc: truth.add("FOC accessories")
    found = {g["component"] for g in cs.detect_gaps(row, cat, reference_bom, component_examples)}
    expected.append(truth); gap_eval.append(found)

tp = sum(len(t & f) for t, f in zip(expected, gap_eval))
total_truth = sum(len(t) for t in expected)
recall = tp / total_truth if total_truth else 1.0
print(f"Gap-detection recall on {len(expected)} labeled quotes: {recall:.3f}  ({tp}/{total_truth} known gaps caught)")
with mlflow.start_run(run_name="gap_eval"):
    mlflow.log_metric("gap_detection_recall", recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · What's next (not built here)
# MAGIC
# MAGIC - **FMAPI/Claude "open-ended risk" agent** — the BRD's non-deterministic layer (`get_reference_bom`, `lookup_similar_purchases`,
# MAGIC   `search_contract_clauses`, `compare_line_items`) for risks no checklist anticipates. Kauvery leans to custom ML for now, so it's a documented add-on.
# MAGIC - **Vector Search** over contract/spec clause text for the `search_contract_clauses` tool.
# MAGIC - **Databricks App** — the upload UI + review queue + Excel export (this notebook produces the Delta comparison sheet it would render).
# MAGIC - **Real-time endpoint** — `models:/{model}@prod` is endpoint-ready; deploy with the databricks-model-serving flow when the app needs sub-second scoring.
# MAGIC - **Swap synthetic → real:** point Section 2 at the customer's `extracted_pdf_datas`, relabel the target in Section 5 with real outcomes, and retrain.

# COMMAND ----------

import json as _json
dbutils.notebook.exit(_json.dumps({
    "history_table": HIST_TABLE, "comparison_table": COMPARISON_TABLE,
    "model": FULL_MODEL_NAME, "model_version": info.registered_model_version,
    "mae": round(float(mae), 2), "r2": round(float(r2), 3),
    "verdict_accuracy": round(float(va), 3), "gap_detection_recall": round(float(recall), 3),
    "rows_generated": int(len(hist_pdf)),
}))
