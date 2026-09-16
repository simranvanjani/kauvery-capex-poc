# Data residency, model hosting & retention — one-pager

*For Kauvery Hospital. Answers the questions raised on the Sep-11 call: where does our data go, is
the model self-hosted by Databricks, is data used for training, and how long is anything retained.*

> Bottom line: with the design below, **your quotation and purchase data stays inside your Databricks
> workspace, in your chosen cloud region, and is never sent to a model vendor's servers or used to
> train their models.** Confirm the exact abuse-logging retention / opt-out for your workspace with
> your Databricks account team before go-live (we will bring this in writing).

## 1. What runs where

| Component | Where it runs | Does data leave Databricks? |
|---|---|---|
| Historical catalog + benchmarks (`extracted_pdf_datas`) | Unity Catalog / Delta, your workspace, your region | No |
| The custom scoring model (`capex_worth_it`) | Your Model Serving endpoint, your workspace | No |
| Cross-site history & vendor ranking (UC functions) | SQL in your workspace | No |
| Foundation model (the conversational layer) | **Databricks-hosted** Foundation Model API, in-region | **No — served by Databricks, not the model vendor's servers** |
| The app | Databricks Apps, your workspace | No |
| Uploaded quotation PDF | UC Volume, **deleted immediately after parsing** | No |

## 2. Foundation model hosting & retention (the key concern)

- **Self-hosted by Databricks.** The Llama foundation model used for conversation is served on
  **Databricks-managed infrastructure inside your cloud region** — requests do **not** go to Meta's
  (or any external vendor's) servers.
- **Not used for training.** Your inputs and outputs are **not** used to train the foundation models.
- **Retention.** Inputs/outputs may be retained **up to ~30 days** for safety/abuse monitoring, then
  deleted. Zero-retention / abuse-logging opt-out is available for qualifying workspaces — we will
  confirm the exact policy and opt-out for your account in writing (Databricks action item).
- **Partner (external) models** (e.g. GPT, Gemini) are a *different* path and would leave the
  Databricks boundary — we are **not** using those in this design.

## 3. Custom model path (maximum control)

If you prefer to avoid foundation models entirely for scoring, the scoring model here is a **custom
model you own**: trained with open-source libraries (scikit-learn + MLflow), registered in your Unity
Catalog, served from your workspace. Its data never leaves your environment and there is no external
inference at all. The conversational layer is the only part that uses a foundation model, and that is
Databricks-hosted and in-region as above.

## 4. Governance & PII

- **Region pinning** — deploy in your required region so data residency is enforced.
- **Unity Catalog** — row-level security by unit, column masking on commercial terms, full lineage
  from any answer back to the source record, and audit logs.
- **No credentials in the shared code** — the app and model use Databricks-managed service-principal
  auth; API tokens are left blank in anything shared with you.
- **We persist only the final purchase decision** — not the quotations, and no accept/reject training
  tables (per your Sep-11 direction).

## 5. Model recommendation

You asked for a model with **strong logical reasoning (for ranking/re-ranking) and good multi-turn
follow-up handling**, kept in-region with no external egress.

- **Recommended: a Databricks-hosted Llama foundation model** (open-weights, in-region, no vendor
  egress) for the conversational layer — this is what the current build uses. It satisfies the
  privacy constraint and handles the multi-turn Q&A and re-ranking narration well.
- **If partner models become acceptable** after your review, a Databricks-hosted Claude model offers
  stronger reasoning; the model is a **one-line swap** in the code. We will bring a specific
  recommendation (and the AI SME's input) once residency is signed off.
- The **judgment (price fairness, gaps, vendor value) is done by your deterministic custom model +
  rules**, not the foundation model — so the numbers are explainable and auditable regardless of
  which conversational model you choose.

## 6. Open item we will close for you
Exact abuse-logging retention window and opt-out for your workspace/region — to be confirmed in
writing with the Databricks account/security team, plus an optional call with a Databricks AI SME for
any deeper security questions.
