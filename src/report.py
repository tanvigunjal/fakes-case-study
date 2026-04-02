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
    return value.replace("<MISSING>", "Missing value")


def top_group(summary: dict[str, object], column: str) -> dict[str, object]:
    rows = summary["grouped_fake_rates"].get(column, [])
    return rows[0] if rows else {"label": "n/a", "fake_rate": 0.0, "count": 0}


def render_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    def render_cell(value: object) -> str:
        text = humanize_label(str(value))
        return text.replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(render_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def build_report(summary: dict[str, object]) -> str:
    overview = summary["overview"]
    repeated = summary["repeated_entities"]
    review = summary["review_signals"]
    tokens = summary["token_signals"]
    numeric = summary["numeric_comparison"]
    temporal = summary["temporal_summary"]
    market_vintage = summary["market_vintage_summary"]
    missingness = summary["missingness_signals"]
    interaction = summary["interaction_fake_rates"]["FullfillmentType__DeliveryCategory"]
    brand_risk = summary["brand_risk"]
    price_ratio = summary["price_ratio_summary"]

    delivery_top = top_group(summary, "DeliveryCategory")
    city_top = top_group(summary, "VNST")
    category_top = top_group(summary, "Category")

    market_date_row = next(row for row in numeric if row["column"] == "MarketDate")
    transaction_row = next(row for row in numeric if row["column"] == "TransactionPrice")
    confirmed_row = next(row for row in numeric if row["column"] == "ConfirmedPrice")
    one_star_row = next(row for row in review["stars"] if row["bucket"] == "1")
    missing_star_row = next(row for row in review["stars"] if row["bucket"] == "missing_or_no_star")
    top_interaction = interaction[0]
    delivery_missing_row = next(row for row in missingness if row["signal"] == "delivery_missing")
    top_brand = brand_risk[0]
    high_ratio_row = next(row for row in price_ratio if row["price_ratio_bucket"] == ">=1.2x")
    low_ratio_row = next(row for row in price_ratio if row["price_ratio_bucket"] == "0.8-1.0x")

    key_findings = [
        f"The dataset contains **{overview['row_count']:,} transactions** with a **{fmt_pct(overview['fake_rate'])} fake rate**, so the task is moderately imbalanced rather than extremely rare-event.",
        f"There are **{overview['exact_duplicates']} exact duplicates** and **{overview['conflicting_duplicate_rows']} conflicting duplicate-like rows**, which is manageable but worth cleaning before any future modeling.",
        f"`top_review` is missing in **{overview['top_missing'][0]['missing_pct']:.1f}%** of rows, and the missing/no-star bucket itself carries a **{fmt_pct(missing_star_row['fake_rate'])} fake rate**.",
        f"Explicit `1/5 stars` language is a major shortcut signal: those rows show an **{fmt_pct(one_star_row['fake_rate'])} fake rate**, far above the base rate.",
        f"The sharpest interaction is **{humanize_label(top_interaction['label'])}** at **{fmt_pct(top_interaction['fake_rate'])}** fake rate, which suggests operations metadata is interacting with listing quality rather than acting independently.",
        f"Product repetition is extreme: **{repeated['rows_in_repeated_product_names']:,} rows** belong to repeated product names, so naive random row splits would likely overstate generalization.",
        f"Fake listings skew older and cheaper on median: `MarketDate` median shifts from **{market_date_row['real_p50']:.0f}** to **{market_date_row['fake_p50']:.0f}**, while `TransactionPrice` median moves from **{transaction_row['real_p50']:.2f}** to **{transaction_row['fake_p50']:.2f}**.",
        f"Pricing above confirmed references is also risky: the fake rate rises from **{fmt_pct(low_ratio_row['fake_rate'])}** in the `0.8-1.0x` bucket to **{fmt_pct(high_ratio_row['fake_rate'])}** once listings exceed `1.2x` confirmed price.",
    ]

    stakeholder_table = [
        {"signal": "DeliveryCategory", "observation": f"{humanize_label(delivery_top['label'])} has the highest fake rate among large buckets", "why_it_matters": "Strongly predictive, but may encode process artifacts rather than intrinsic product quality."},
        {"signal": "Fulfillment x delivery", "observation": f"{humanize_label(top_interaction['label'])} reaches {fmt_pct(top_interaction['fake_rate'])}", "why_it_matters": "The risk is strongest in interaction form, not just in the single-column marginal view."},
        {"signal": "Review stars", "observation": "1-star reviews are disproportionately fake-heavy", "why_it_matters": "Useful for triage, but risky if reviews are synthetic or unavailable at listing time."},
        {"signal": "Repeated products", "observation": f"{repeated['repeated_product_names']} product names repeat across the dataset", "why_it_matters": "Entity leakage is a real evaluation risk and needs split discipline."},
        {"signal": "Price references", "observation": f"ConfirmedPrice median is {confirmed_row['real_p50']:.2f} for real vs {confirmed_row['fake_p50']:.2f} for fake", "why_it_matters": "Relative price gaps look promising for a compact tabular model later on."},
        {"signal": "Brand concentration", "observation": f"{humanize_label(top_brand['brand'])} is the highest-risk brand bucket with enough support", "why_it_matters": "Some brand families may need targeted policy review or SKU-level monitoring."},
    ]

    token_table = tokens["fake_salient"][:6]
    temporal_table = temporal
    interaction_table = interaction[:6]
    vintage_table = market_vintage
    price_ratio_table = price_ratio

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
        f"The most important refinement from the deeper slice analysis is that **missing delivery metadata behaves very differently across fulfillment flows**. When delivery data is missing, the fake rate jumps to **{fmt_pct(delivery_missing_row['missing_fake_rate'])}** overall, but the interaction table shows that BUSINESS and CENTRAL flows are especially exposed. That is the sort of pattern a production model would exploit immediately, and it is also the sort of pattern that needs policy review before being trusted blindly.",
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
        "### Interaction View",
        "",
        render_table(interaction_table, ["label", "count", "fake_count", "fake_rate"]),
        "",
        "### Pricing View",
        "",
        render_table(price_ratio_table, ["price_ratio_bucket", "count", "fake_count", "fake_rate"]),
        "",
        "### Product Vintage View",
        "",
        render_table(vintage_table, ["market_year", "count", "fake_count", "fake_rate"]),
        "",
        "![Purchase year fake rate](assets/purchase_year_fake_rate.svg)",
        "",
        "![Product vintage fake rate](assets/market_vintage_fake_rate.svg)",
        "",
        "![Delivery category fake rate](assets/delivery_category_fake_rate.svg)",
        "",
        "![Fulfillment x delivery interaction](assets/fulfillment_delivery_interaction.svg)",
        "",
        "![Fulfillment x delivery heatmap](assets/fulfillment_delivery_heatmap.svg)",
        "",
        "![Review star fake rate](assets/review_star_fake_rate.svg)",
        "",
        "![Price ratio fake rate](assets/price_ratio_fake_rate.svg)",
        "",
        "![Numeric median comparison](assets/numeric_median_comparison.svg)",
        "",
        "## Recommended Modeling Direction",
        "",
        "For the minimum viable modeling path, TabPFN v2 is a strong core choice because it provides a powerful tabular foundation model without a long training cycle. For this dataset, the practical move is to convert mixed marketplace signals into scalar features, run TabPFN v2 as the first production-grade baseline, and keep the PyTorch fusion model as the next custom upgrade only if we need more flexibility.",
        "",
        "Recommended order of operations:",
        "",
        "1. Start with TabPFN v2 on numeric, categorical, missingness, and lightweight text-derived scalar features.",
        "2. Evaluate with time-aware or entity-aware splits rather than naive random rows.",
        "3. Run ablations that remove review stars, delivery metadata, and other shortcut-heavy fields.",
        "4. Move to the custom PyTorch hybrid only if the TabPFN baseline leaves meaningful headroom.",
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
