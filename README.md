# Fakes Case Study

This repository contains a Phase 1 submission for the fake-listing detection case study.

The current repo now ships two concrete deliverables:

- a polished EDA and reporting workflow for understanding the problem deeply
- a minimum viable modeling path built around **TabPFN v2** for low-friction tabular inference

## Included Deliverables

- `01A_fakes.pdf`: original assignment brief
- `src/data.py`: data loading, schema validation, cleaning, duplicate detection
- `src/eda.py`: exploratory analysis pipeline that writes CSV tables and SVG plots
- `src/report.py`: executive summary generator
- `src/tabpfn_pipeline.py`: runnable TabPFN v2 inference pipeline for this dataset
- `src/features.py`: reusable feature engineering helpers for later modeling
- `src/modeling_stub.py`: device-aware PyTorch 2 model scaffolds for a later custom model phase
- `report/executive_summary.md`: polished stakeholder-facing summary
- `report/technical_report.md`: modeling, evaluation, and deployment notes

## Data Expectations

The raw CSV is expected locally at:

- `./data_fakes.csv`

The dataset is intentionally ignored by git to keep the repository lightweight.

## Quickstart: EDA And Reports

Run the analysis pipeline:

```bash
python -m src.eda
```

Generate the executive summary report:

```bash
python -m src.report
```

Outputs:

- `artifacts/tables/`
- `artifacts/plots/`
- `report/executive_summary.md`
- `report/assets/*.svg`

## Quickstart: TabPFN v2 Inference

### 1. Install dependencies

```bash
python -m pip install -r requirements-tabpfn.txt
```

### 2. Choose how to run

TabPFN's official documentation recommends GPU for best performance and notes that CPU is generally practical only for smaller datasets (roughly `<=1000` rows). This dataset is larger, so there are two sensible ways to run it:

GPU or high-memory machine:

```bash
python -m src.tabpfn_pipeline --device auto
```

CPU smoke test on a smaller training slice:

```bash
python -m src.tabpfn_pipeline --device cpu --max-train-rows 1000
```

CPU full-dataset run if you explicitly want to force it:

```bash
python -m src.tabpfn_pipeline --device cpu --allow-cpu-large-dataset
```

### 3. Generated outputs

The pipeline writes:

- `artifacts/tabpfn/validation_predictions.csv`
- `artifacts/tabpfn/test_predictions.csv`
- `artifacts/tabpfn/run_summary.json`

## Modeling Notes

This MVP uses **TabPFN v2** as the core model via:

```python
TabPFNClassifier.create_default_for_version(ModelVersion.V2)
```

Why this is a good fit here:

- it gives a strong tabular baseline without a long training loop
- it is attractive for a take-home because the setup is compact and the workflow is easy to explain
- it works well when we convert mixed marketplace data into scalar features, including categorical IDs and lightweight text-derived signals

Current TabPFN feature blocks include:

- raw numeric marketplace features
- date-derived features from `PurchDate`
- price gaps and price ratios
- missingness indicators
- ordinal-encoded categoricals
- lightweight text-derived scalar signals from names, descriptions, and reviews

## Current Findings Snapshot

The current analysis highlights a few immediate modeling risks and opportunities:

- fake listings make up about `10.7%` of the dataset
- `top_review` contains very strong shortcut-like signals, especially explicit star ratings
- missing `DeliveryCategory` is highly associated with fake outcomes
- `BUSINESS + missing DeliveryCategory` is an especially risky interaction slice
- repeated product identities are pervasive, so naive random row splits would likely overestimate model quality
- fake rows tend to be slightly older and slightly cheaper on the main price references
- listings priced well above confirmed reference prices also become more fake-prone

## Suggested Next Steps

1. Run the TabPFN pipeline on the target hardware and inspect `artifacts/tabpfn/run_summary.json`.
2. Compare full-data versus `--max-train-rows 1000` behavior on CPU.
3. Use the PyTorch scaffolding later if you want to move from the MVP foundation-model baseline to a custom hybrid model.
