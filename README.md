# Fakes Case Study

This repository contains a Phase 1 submission for the fake-listing detection case study.

The focus of this phase is:

- structured exploratory analysis of the provided marketplace dataset
- a polished executive summary for business and technical stakeholders
- CPU/GPU-ready PyTorch 2 scaffolding for a later modeling phase

## Included Deliverables

- `01A_fakes.pdf`: original assignment brief
- `src/data.py`: data loading, schema validation, cleaning, duplicate detection
- `src/eda.py`: exploratory analysis pipeline that writes CSV tables and SVG plots
- `src/report.py`: executive summary generator
- `src/features.py`: reusable feature engineering helpers for later modeling
- `src/modeling_stub.py`: device-aware PyTorch 2 model scaffolds for Phase 2
- `report/executive_summary.md`: generated executive summary report

## Data Expectations

The raw CSV is expected locally at:

- `./data_fakes.csv`

The dataset is intentionally ignored by git to keep the repository lightweight.

## Quickstart

Run the analysis pipeline:

```bash
python -m src.eda
```

Generate the executive summary report:

```bash
python -m src.report
```

The analysis step writes outputs to:

- `artifacts/tables/`
- `artifacts/plots/`

The report step writes:

- `report/executive_summary.md`

## Current Findings Snapshot

The current analysis highlights a few immediate modeling risks and opportunities:

- fake listings make up about `10.7%` of the dataset
- `top_review` contains very strong shortcut-like signals, especially explicit star ratings
- missing `DeliveryCategory` is highly associated with fake outcomes
- repeated product identities are pervasive, so naive random row splits would likely overestimate model quality
- fake rows tend to be slightly older and slightly cheaper on the main price references

## Phase 2 Direction

A sensible next step is a compact hybrid model:

1. start with a tabular baseline over numeric, categorical, and missingness features
2. add lightweight text handling with hashed embeddings or cached small-text embeddings
3. evaluate with temporal or entity-aware splits instead of naive random rows
4. calibrate decision thresholds for auto-blocking versus manual review
