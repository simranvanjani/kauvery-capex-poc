-- Kauvery CAPEX — Unity Catalog functions the app depends on.
-- CREATE OR REPLACE is idempotent: re-running updates the function in place.
-- Placeholders {{CATALOG}} / {{SCHEMA}} are substituted by install.py.
-- Reads the tabular history table extracted_pdf_datas (customer-loaded from Oracle; no PDF parsing).

-- cross_unit_history(search) -> JSON array of matching purchases across all units, cheapest first.
CREATE OR REPLACE FUNCTION {{CATALOG}}.{{SCHEMA}}.cross_unit_history(search STRING)
RETURNS STRING
COMMENT 'Historical purchases of a matching item across all Kauvery units, cheapest first, as a JSON array.'
RETURN (
  SELECT to_json(collect_list(rec)) FROM (
    SELECT named_struct(
      'unit',            unit_name,
      'vendor',          vendor_name,
      'date',            po_date,
      'unit_price',      unit_rate,
      'warranty_months', warranty_months,
      'maintenance',     coalesce(amc_value, camc_value),
      'foc',             foc_details
    ) AS rec
    FROM {{CATALOG}}.{{SCHEMA}}.extracted_pdf_datas
    WHERE unit_rate IS NOT NULL AND unit_rate > 0
      AND ( lower(item_description) LIKE concat('%', lower(search), '%')
         OR lower(model_no)         LIKE concat('%', lower(search), '%')
         OR lower(make_brand)       LIKE concat('%', lower(search), '%') )
    ORDER BY unit_rate ASC
    LIMIT 50
  )
);

-- recommend_vendor(search) -> JSON array of vendors ranked by value (bundled FOC + AMC at low price rank highest).
CREATE OR REPLACE FUNCTION {{CATALOG}}.{{SCHEMA}}.recommend_vendor(search STRING)
RETURNS STRING
COMMENT 'Vendors ranked by value-for-money for a matching item; bundled FOC+AMC at low price rank highest. JSON array.'
RETURN (
  SELECT to_json(collect_list(rec)) FROM (
    SELECT named_struct(
      'vendor',      vendor_name,
      'purchases',   cnt,
      'min_price',   min_price,
      'avg_price',   avg_price,
      'foc_rate',    foc_rate,
      'amc_rate',    amc_rate,
      'value_score', value_score
    ) AS rec
    FROM (
      SELECT vendor_name,
             count(*)                                                              AS cnt,
             round(min(unit_rate), 2)                                              AS min_price,
             round(avg(unit_rate), 2)                                              AS avg_price,
             round(avg(CASE WHEN foc_details IS NOT NULL THEN 1 ELSE 0 END), 2)    AS foc_rate,
             round(avg(CASE WHEN amc_value IS NOT NULL OR camc_value IS NOT NULL
                            THEN 1 ELSE 0 END), 2)                                 AS amc_rate,
             round(avg(CASE WHEN foc_details IS NOT NULL THEN 1 ELSE 0 END) * 0.5
                 + avg(CASE WHEN amc_value IS NOT NULL OR camc_value IS NOT NULL
                            THEN 1 ELSE 0 END) * 0.5, 3)                           AS value_score
      FROM {{CATALOG}}.{{SCHEMA}}.extracted_pdf_datas
      WHERE unit_rate IS NOT NULL AND unit_rate > 0
        AND ( lower(item_description) LIKE concat('%', lower(search), '%')
           OR lower(model_no)         LIKE concat('%', lower(search), '%')
           OR lower(make_brand)       LIKE concat('%', lower(search), '%') )
      GROUP BY vendor_name
    )
    ORDER BY value_score DESC, min_price ASC
    LIMIT 20
  )
);

-- price_fairness(...) -> wraps the capex-worth-it serving endpoint. Optional: the app calls the
-- endpoint directly; this exists so the agent's UC-function toolkit can score too.
CREATE OR REPLACE FUNCTION {{CATALOG}}.{{SCHEMA}}.price_fairness(
  item_description STRING, make_brand STRING, model_no STRING, qty INT, unit_rate DOUBLE,
  warranty_months INT, amc_present BOOLEAN, foc_present BOOLEAN, delivery_lead_days INT,
  payment_terms STRING, has_training BOOLEAN, has_installation BOOLEAN)
RETURNS STRING
COMMENT 'Scores one quotation line item via the capex-worth-it model serving endpoint.'
RETURN ai_query(
  'capex-worth-it',
  named_struct(
    'item_description', item_description, 'make_brand', make_brand, 'model_no', model_no,
    'qty', qty, 'unit_rate', unit_rate, 'warranty_months', warranty_months,
    'amc_present', amc_present, 'camc_present', false, 'foc_present', foc_present,
    'delivery_lead_days', delivery_lead_days, 'payment_terms', payment_terms,
    'has_training', has_training, 'has_installation', has_installation)
);
