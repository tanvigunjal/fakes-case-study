# Fakes Case Study

Phase 1 of the fake-listing detection case study focuses on three things:

- exploratory analysis of the provided marketplace dataset
- a polished executive summary report for business and technical stakeholders
- CPU/GPU-ready PyTorch 2 scaffolding for a later modeling phase

## Repository Layout

- `01A_fakes.pdf`: original assignment brief
- `src/`: data loading, EDA, reporting, and future-model scaffolding
- `report/`: human-readable summary outputs
- `artifacts/tables/`: generated CSV summaries
- `artifacts/plots/`: generated SVG plots

## Data Expectations

The raw dataset is expected locally at:

- `./data_fakes.csv`

The CSV is intentionally ignored by git to keep the repository lightweight.

## Planned Workflow

1. run the analysis pipeline to create tables and plots
2. generate the executive summary report
3. reuse the shared feature utilities and modeling stubs in Phase 2

## Expected Commands

```bash
python -m src.eda
python -m src.report
```

Additional commands will be documented as the implementation lands.
