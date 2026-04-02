# Technical Report: Fake Listing Detection Case Study

## 1. Problem Framing

The task is to classify marketplace transactions as genuine or fake using a mix of structured, categorical, and text features. This is a practical fraud-detection problem, not a pure benchmarking exercise: the model must be efficient, maintainable, and robust enough for a team that may train locally on CPU or a modest GPU.

The business goal is not simply high accuracy. The useful operating point is a model that can flag suspicious listings early, with controlled false positives, so the platform can route borderline cases to manual review and reduce customer harm.

## 2. Data Profile

The dataset has 11,358 rows and a 10.7% positive rate for `Fake`. It is a small-to-medium tabular dataset with meaningful text, repeated entities, and several missingness patterns that carry signal.

Key data characteristics:

- `top_review` is missing in 19.1% of rows.
- `Brand` is missing in 9.7% of rows.
- `DeliveryCategory` is missing in 3.9% of rows.
- There are 522 exact duplicate rows and 20 conflicting duplicate-like rows.
- `ProductName` repeats heavily, which makes random row splits unreliable.
- `PurchDate` is parseable as `%Y-%d-%m`, and the dataset spans mainly 2019 to 2020 with a few smaller edge-year cohorts.

A few signals stand out immediately:

- `top_review` contains explicit star-rating language, and `1/5 stars` rows are overwhelmingly fake-heavy.
- Missing `top_review` is itself informative rather than random noise.
- Lower price references and older platform vintage appear more associated with fake listings.
- `DeliveryCategory` looks suspiciously class-skewed and may reflect process artifacts as much as product quality.

## 3. Technical Recommendation

For the minimum viable model, the repo should use **TabPFN v2 as the core classifier**.

Why this is a strong choice here:

- It gives a high-quality tabular baseline without a long training loop.
- It is ideal for a take-home setting where the focus is practical delivery rather than GPU-heavy optimization.
- It is easier to run on mixed hardware environments than a custom deep model training stack.
- It lets us move quickly while still preserving a strong modeling story.

The important limitation is that TabPFN expects scalar tabular inputs, not raw text. That means the practical MVP is:

- numeric marketplace features
- categorical IDs encoded as integers
- missingness flags
- lightweight text-derived scalar features such as review star value, text lengths, and keyword indicators

The repo now includes that path in `src/tabpfn_pipeline.py`, using `TabPFNClassifier.create_default_for_version(ModelVersion.V2)` as the main entrypoint.

The existing PyTorch 2 scaffolding remains valuable, but as a **Phase 2 custom-model extension**, not as the first model we should reach for.

## 4. Feature Strategy

The recommended feature design is deliberately conservative and practical.

### Structured features

Use these as numeric inputs:

- `MarketDate`
- `ProductAge`
- `AveragePrice`
- `TransactionPrice`
- `ConfirmedPrice`

Add simple derived features:

- purchase year, month, day, weekday from `PurchDate`
- `TransactionPrice - AveragePrice`
- `TransactionPrice - ConfirmedPrice`
- `TransactionPrice / AveragePrice`
- `TransactionPrice / ConfirmedPrice`
- missingness flags for `AveragePrice`, `ConfirmedPrice`, `Brand`, `DeliveryCategory`, and `top_review`

### Categorical features

Treat these as embeddings or target-safe categorical IDs:

- `VNZIP1`
- `VNST`
- `Category`
- `Brand`
- `FullfillmentType`
- `DeliveryCategory`

### Text features

Because TabPFN expects scalar inputs, the MVP does **not** pass raw text directly into the model. Instead, it extracts lightweight scalar text signals from:

- `ProductName`
- `product_description`
- `top_review`

Examples:

- product name word count
- description word count
- review word count
- review-star value if present
- review missingness flag
- a small set of keyword indicators from the review text

If a future custom PyTorch model is introduced, these same text fields can later be upgraded to hashed or frozen-embedding features.

## 5. Split And Evaluation Strategy

A naive random row split would likely overestimate performance because the same products repeat many times. The safer evaluation design is:

- Primary split: temporal holdout on `PurchDate`.
- Optional robustness check: grouped split by `ProductName` or `Brand + ProductName`.
- Remove exact duplicates before modeling.
- Log conflicting duplicate-like rows and keep them out of training/evaluation.

Recommended split logic:

- Train: rows up to 2020-07-31
- Validation: 2020-08-01 to 2020-09-30
- Test: 2020-10-01 onward

Recommended metrics:

- PR-AUC as the primary selection metric
- precision, recall, F1, and F2 at the operating threshold
- confusion matrix and flag rate
- calibration metrics such as Brier score and ECE if probabilities are exposed downstream

The relevant design principle is to optimize for ranking and review workflow, not only overall accuracy.

## 6. Shortcut And Leakage Risks

This dataset contains several shortcut risks that could produce misleadingly strong offline scores.

Main risks:

- `top_review` star patterns are highly predictive and may not generalize if review text is incomplete or manipulated.
- `DeliveryCategory` is unusually class-skewed and may encode operational artifacts.
- Repeated `ProductName` values make product-level leakage likely unless the split is controlled.
- Exact duplicates and conflicting duplicate-like rows can inflate apparent performance if not filtered.
- `RefId` should never be used as a feature.

Interpretation rule:

- Signals that are predictive but unstable should be treated as evidence of a workflow issue, not as a reason to declare the model ready for production.

## 7. Deployment Considerations

The production target should be a model that can run on commodity hardware.

Practical deployment constraints:

- Training must work without a GPU.
- Inference should not require large embedding tables or large cached corpora.
- The feature pipeline should tolerate missing text fields and unseen categories.
- Probabilities should be calibrated before any automated blocking use case.
- The system should support two thresholds: one for aggressive auto-blocking and one for manual review routing.

Recommended deployment shape:

- Offline preprocessing for scalar feature construction and categorical vocab construction
- TabPFN v2 classifier as the core scoring engine
- Thresholded decision layer external to the model
- Optional later upgrade path to a custom PyTorch fusion model if needed

## 8. Concrete Next Steps

If we continue beyond Phase 1, the next implementation steps should be:

1. Run the TabPFN v2 pipeline end to end on the target hardware.
2. Compare full-data versus smaller CPU smoke-test runs.
3. Evaluate under the temporal split and compare against a grouped split.
4. Remove review-star patterns and delivery metadata in ablations to measure shortcut reliance.
5. Add calibration and threshold selection for a real review workflow.
6. Move to the custom PyTorch hybrid only if TabPFN leaves meaningful performance headroom.

## 9. Current Repo Status

The current repository already contains the building blocks for this plan:

- `src/data.py` handles validation, cleaning, and duplicate detection.
- `src/eda.py` generates summary tables and plots.
- `src/report.py` turns the analysis artifacts into a narrative summary.
- `src/features.py` prepares reusable feature rows.
- `src/tabpfn_pipeline.py` provides the minimum viable TabPFN v2 inference workflow.
- `src/modeling_stub.py` provides CPU/GPU-aware PyTorch scaffolding.

That means the codebase now has both a practical MVP model path and a clear upgrade path for a later custom modeling phase.
