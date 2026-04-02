# Executive Summary

## Situation

Amazing needs a practical way to identify likely fake listings on a marketplace platform. This Phase 1 submission focuses on understanding the data landscape, surfacing business-relevant risks, and preparing a clean foundation for a later modeling phase without requiring GPU training today.

## Key Findings

- The dataset contains **11,358 transactions** with a **10.7% fake rate**, so the task is moderately imbalanced rather than extremely rare-event.
- There are **522 exact duplicates** and **20 conflicting duplicate-like rows**, which is manageable but worth cleaning before any future modeling.
- `top_review` is missing in **19.1%** of rows, and the missing/no-star bucket itself carries a **15.4% fake rate**.
- Explicit `1/5 stars` language is a major shortcut signal: those rows show an **89.1% fake rate**, far above the base rate.
- Product repetition is extreme: **11,177 rows** belong to repeated product names, so naive random row splits would likely overstate generalization.
- Fake listings skew older and cheaper on median: `MarketDate` median shifts from **2016** to **2015**, while `TransactionPrice` median moves from **67.70** to **61.35**.

## Business Interpretation

The most important operational takeaway is that the dataset contains both **meaningful product-risk signals** and **suspicious shortcut signals**. For example, delivery metadata and explicit review-star language appear unusually predictive. That can be useful for triage, but it also means a model could look strong offline while learning artifacts that may not hold up in production.

The highest-risk delivery bucket in the current sample is **Missing value** with a fake rate of **63.5%** across **446 rows**. Among large geographies, **Oslo** shows the highest fake rate in the current slice, and **Pet Supplies Birds** is the most exposed broad category with enough support to matter operationally.

## Data Quality And Evaluation Risks

- Duplicate management matters: 522 exact duplicates and 20 conflicting duplicate-like rows should be handled consistently before training.
- Product repetition is the main evaluation hazard: 11,177 rows belong to repeated `Brand + ProductName` pairs.
- Review availability is non-random: missing or unparseable review-star information carries a 15.4% fake rate.
- Numeric signals move in the expected direction for counterfeits: fake rows have older platform vintages and lower median transaction/reference prices.

## Signal Inventory

| signal | observation | why_it_matters |
| --- | --- | --- |
| DeliveryCategory | Missing value has the highest fake rate among large buckets | Strongly predictive, but may encode process artifacts rather than intrinsic product quality. |
| Review stars | 1-star reviews are disproportionately fake-heavy | Useful for triage, but risky if reviews are synthetic or unavailable at listing time. |
| Repeated products | 522 product names repeat across the dataset | Entity leakage is a real evaluation risk and needs split discipline. |
| Price references | ConfirmedPrice median is 73.64 for real vs 64.16 for fake | Relative price gaps look promising for a compact tabular model later on. |

### Text Clues Worth Monitoring

| token | fake_rows | real_rows |
| --- | --- | --- |
| product | 578 | 3272 |
| get | 427 | 711 |
| best | 413 | 688 |
| buy | 317 | 98 |
| dont | 269 | 213 |
| amazing | 258 | 323 |

### Temporal View

| purchase_year | count | fake_count | fake_rate |
| --- | --- | --- | --- |
| 2018 | 28 | 4 | 14.2857 |
| 2019 | 5460 | 541 | 9.9084 |
| 2020 | 5831 | 667 | 11.4389 |
| 2021 | 39 | 5 | 12.8205 |

![Purchase year fake rate](../artifacts/plots/purchase_year_fake_rate.svg)

![Delivery category fake rate](../artifacts/plots/delivery_category_fake_rate.svg)

## Recommended Phase 2 Modeling Direction

A sensible next step is a compact hybrid model rather than a giant pretrained system. The data is small enough that a lightweight tabular tower plus a modest text representation should be easier to train, easier to explain, and far easier for downstream users to run on commodity hardware.

Recommended order of operations:

1. Start with a tabular baseline using numeric, categorical, and missingness features.
2. Add a cheap text component such as hashed text embeddings or cached small-sentence embeddings.
3. Evaluate with time-aware or entity-aware splits rather than naive random rows.
4. Run ablations that remove review stars, delivery metadata, and other shortcut-heavy fields.

## What We Would Do Next

- Implement a CPU/GPU-ready PyTorch 2 training path using the existing stubs in this repo.
- Compare tabular-only, text-only, and fused models under a temporal split.
- Calibrate probabilities and define separate thresholds for auto-blocking versus manual review.
- Stress-test the model without review stars and without delivery metadata to measure reliance on possible artifacts.
