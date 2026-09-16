"""Kauvery CAPEX Phase 2 — core scoring & gap-detection logic.

Exact module the notebook writes at runtime and logs WITH the MLflow model, so it is what runs
in the serving endpoint. Weights follow the customer rubric (price 50 / warranty 30 / rest);
a High-severity gap prevents an automatic Accept.
"""
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

CATEGORY_KEYWORDS = {
    "CT Scanner": ["ct scanner", "somatom", "revolution ct", "ingenuity ct", "128 slice", "128-slice"],
    "MRI": ["mri", "magnetom", "signa", "achieva", "ingenia", "tesla", "1.5t", "3t"],
    "Cath Lab": ["cath lab", "azurion", "artis", "allia", "angiography", "cardiac cath"],
    "Ventilator": ["ventilator", "evita", "hamilton", "trilogy"],
    "Ultrasound": ["ultrasound", "voluson", "epiq", "resona", "logiq", "sonography"],
    "Patient Monitor": ["patient monitor", "intellivue", "bedside monitor", "b450", "b650", "umec", "benevision", "bsm-"],
    "Defibrillator": ["defibrillator", "heartstart", "zoll", "aed"],
    "Anesthesia Workstation": ["anesthesia", "anaesthesia", "perseus", "aisys"],
    "Infusion Pump": ["infusion pump", "syringe pump", "infusomat", "benefusion", "sigma spectrum"],
    "Dialysis Machine": ["dialysis", "hemodialysis", "fresenius", "surdial", "4008", "5008"],
}

def classify_category(text):
    t = str(text).lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return cat
    return "Other"

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
