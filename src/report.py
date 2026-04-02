from __future__ import annotations

import json
from pathlib import Path

from src.data import project_root


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def report_path() -> Path:
    return project_root() / "report" / "executive_summary.md"


def load_summary() -> dict[str, object]:
    path = artifacts_dir() / "tables" / "analysis_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            "Expected analysis artifacts at artifacts/tables/analysis_summary.json. Run `python -m src.eda` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def humanize_label(value: str) -> str:
    if value == "<MISSING>":
        return "Missing value"
    return value


def top_group(summary: dict[str, object], column: str) -> dict[str, object]:
    rows = summary["grouped_fake_rates"].get(column, [])
    return rows[0] if rows else {"label": "n/a", "fake_rate": 0.0, "count": 0}


def render_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def build_report(summary: dict[str, object]) -> str:
    overview = summary["overview"]
    repeated = summary["repeated_entities"]
    review = summary["review_signals"]
    tokens = summary["token_signals"]
    numeric = summary["numeric_comparison"]
    temporal = summary["temporal_summary"]

    delivery_top = top_group(summary, "DeliveryCategory")
    city_top = top_group(summary, "VNST")
    category_top = top_group(summary, "Category")

    market_date_row = next(row for row in numeric if row["column"] == "MarketDate")
    transaction_row = next(row for row in numeric if row["column"] == "TransactionPrice")
    confirmed_row = next(row for row in numeric if row["column"] == "ConfirmedPrice")
    one_star_row = next(row for row in review["stars"] if row["bucket"] == "1")
    missing_star_row = next(row for row in review["stars"] if row["bucket"] == "missing_or_no_star")

    key_findings = [
        f"The dataset contains **{overview['row_count']:,} transactions** with a **{fmt_pct(overview['fake_rate'])} fake rate**, so the task is moderately imbalanced rather than extremely rare-event.",
        f"There are **{overview['exact_duplicates']} exact duplicates** and **{overview['conflicting_duplicate_rows']} conflicting duplicate-like rows**, which is manageable but worth cleaning before any future modeling.",
        f"`top_review` is missing in **{overview['top_missing'][0]['missing_pct']:.1f}%** of rows, and the missing/no-star bucket itself carries a **{fmt_pct(missing_star_row['fake_rate'])} fake rate**.",
        f"Explicit `1/5 stars` language is a major shortcut signal: those rows show an **{fmt_pct(one_star_row['fake_rate'])} fake rate**, far above the base rate.",
        f"Product repetition is extreme: **{repeated['rows_in_repeated_product_names']:,} rows** belong to repeated product names, so naive random row splits would likely overstate generalization.",
        f"Fake listings skew older and cheaper on median: `MarketDate` median shifts from **{market_date_row['real_p50']:.0f}** to **{market_date_row['fake_p50']:.0f}**, while `TransactionPrice` median moves from **{transaction_row['real_p50']:.2f}** to **{transaction_row['fake_p50']:.2f}**.",
    ]

    stakeholder_table = [
        {"signal": "DeliveryCategory", "observation": f"{humanize_label(delivery_top['label'])} has the highest fake rate among large buckets", "why_it_matters": "Strongly predictive, but may encode process artifacts rather than intrinsic product quality."},
        {"signal": "Review stars", "observation": "1-star reviews are disproportionately fake-heavy", "why_it_matters": "Useful for triage, but risky if reviews are synthetic or unavailable at listing time."},
        {"signal": "Repeated products", "observation": f"{repeated['repeated_product_names']} product names repeat across the dataset", "why_it_matters": "Entity leakage is a real evaluation risk and needs split discipline."},
        {"signal": "Price references", "observation": f"ConfirmedPrice median is {confirmed_row['real_p50']:.2f} for real vs {confirmed_row['fake_p50']:.2f} for fake", "why_it_matters": "Relative price gaps look promising for a compact tabular model later on."},
    ]

    token_table = tokens["fake_salient"][:6]
    temporal_table = temporal

    lines = [
        "# Executive Summary",
        "",
        "## Situation",
        "",
        "Amazing needs a practical way to identify likely fake listings on a marketplace platform. This Phase 1 submission focuses on understanding the data landscape, surfacing business-relevant risks, and preparing a clean foundation for a later modeling phase without requiring GPU training today.",
        "",
        "## Key Findings",
        "",
        *[f"- {finding}" for finding in key_findings],
        "",
        "## Business Interpretation",
        "",
        f"The most important operational takeaway is that the dataset contains both **meaningful product-risk signals** and **suspicious shortcut signals**. For example, delivery metadata and explicit review-star language appear unusually predictive. That can be useful for triage, but it also means a model could look strong offline while learning artifacts that may not hold up in production.",
        "",
        f"The highest-risk delivery bucket in the current sample is **{humanize_label(delivery_top['label'])}** with a fake rate of **{fmt_pct(delivery_top['fake_rate'])}** across **{delivery_top['count']} rows**. Among large geographies, **{humanize_label(city_top['label'])}** shows the highest fake rate in the current slice, and **{humanize_label(category_top['label'])}** is the most exposed broad category with enough support to matter operationally.",
        "",
        "## Data Quality And Evaluation Risks",
        "",
        f"- Duplicate management matters: {overview['exact_duplicates']} exact duplicates and {overview['conflicting_duplicate_rows']} conflicting duplicate-like rows should be handled consistently before training.",
        f"- Product repetition is the main evaluation hazard: {repeated['rows_in_repeated_brand_product_pairs']:,} rows belong to repeated `Brand + ProductName` pairs.",
        f"- Review availability is non-random: missing or unparseable review-star information carries a {fmt_pct(missing_star_row['fake_rate'])} fake rate.",
        f"- Numeric signals move in the expected direction for counterfeits: fake rows have older platform vintages and lower median transaction/reference prices.",
        "",
        "## Signal Inventory",
        "",
        render_table(stakeholder_table, ["signal", "observation", "why_it_matters"]),
        "",
        "### Text Clues Worth Monitoring",
        "",
        render_table(token_table, ["token", "fake_rows", "real_rows"]),
        "",
        "### Temporal View",
        "",
        render_table(temporal_table, ["purchase_year", "count", "fake_count", "fake_rate"]),
        "",
        "![Purchase year fake rate](../artifacts/plots/purchase_year_fake_rate.svg)",
        "",
        "![Delivery category fake rate](../artifacts/plots/delivery_category_fake_rate.svg)",
        "",
        "## Recommended Phase 2 Modeling Direction",
        "",
        "A sensible next step is a compact hybrid model rather than a giant pretrained system. The data is small enough that a lightweight tabular tower plus a modest text representation should be easier to train, easier to explain, and far easier for downstream users to run on commodity hardware.",
        "",
        "Recommended order of operations:",
        "",
        "1. Start with a tabular baseline using numeric, categorical, and missingness features.",
        "2. Add a cheap text component such as hashed text embeddings or cached small-sentence embeddings.",
        "3. Evaluate with time-aware or entity-aware splits rather than naive random rows.",
        "4. Run ablations that remove review stars, delivery metadata, and other shortcut-heavy fields.",
        "",
        "## What We Would Do Next",
        "",
        "- Implement a CPU/GPU-ready PyTorch 2 training path using the existing stubs in this repo.",
        "- Compare tabular-only, text-only, and fused models under a temporal split.",
        "- Calibrate probabilities and define separate thresholds for auto-blocking versus manual review.",
        "- Stress-test the model without review stars and without delivery metadata to measure reliance on possible artifacts.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = load_summary()
    output = build_report(summary)
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
    print(f"Wrote report to {path}")


if __name__ == "__main__":
    main()
