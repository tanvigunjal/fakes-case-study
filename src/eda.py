from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.data import (
    NUMERIC_COLUMNS,
    DatasetBundle,
    Record,
    load_dataset,
    project_root,
)

STAR_PATTERN = re.compile(r"(\d)/5\s*stars", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "are",
    "was",
    "have",
    "has",
    "you",
    "your",
    "its",
    "our",
    "but",
    "not",
    "all",
    "can",
    "will",
    "they",
    "their",
    "them",
    "his",
    "her",
    "she",
    "he",
    "it",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "is",
    "as",
    "at",
    "by",
    "or",
    "be",
    "if",
    "no",
    "my",
    "im",
    "we",
    "i",
}


def artifacts_root() -> Path:
    root = project_root() / "artifacts"
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    return root


def report_assets_root() -> Path:
    root = project_root() / "report" / "assets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * probability))
    return ordered[index]


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator * 100.0 / denominator, 4)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_bar_chart(
    path: Path,
    rows: list[tuple[str, float]],
    title: str,
    subtitle: str,
    color: str = "#c2410c",
) -> None:
    width = 900
    height = 420
    margin_left = 200
    margin_top = 70
    margin_bottom = 70
    inner_width = width - margin_left - 60
    inner_height = height - margin_top - margin_bottom
    max_value = max((value for _, value in rows), default=1.0) or 1.0
    bar_gap = 10
    bar_height = max(14, (inner_height - bar_gap * max(len(rows) - 1, 0)) // max(len(rows), 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f7f4"/>',
        f'<text x="40" y="36" font-size="24" font-family="Georgia, serif" fill="#1f2937">{title}</text>',
        f'<text x="40" y="58" font-size="13" font-family="Arial, sans-serif" fill="#6b7280">{subtitle}</text>',
    ]
    for idx, (label, value) in enumerate(rows):
        y = margin_top + idx * (bar_height + bar_gap)
        bar_width = 0 if max_value == 0 else int(inner_width * value / max_value)
        parts.append(f'<text x="{margin_left - 12}" y="{y + bar_height - 4}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#374151">{label}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{margin_left + bar_width + 8}" y="{y + bar_height - 4}" font-size="12" font-family="Arial, sans-serif" fill="#111827">{value:.1f}%</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_plot_pair(
    filename: str,
    rows: list[tuple[str, float]],
    title: str,
    subtitle: str,
    plots_root: Path,
    assets_root: Path,
    color: str = "#c2410c",
) -> None:
    for destination in (plots_root / filename, assets_root / filename):
        render_bar_chart(destination, rows, title, subtitle, color=color)


def render_heatmap(
    path: Path,
    matrix: dict[str, dict[str, float]],
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    subtitle: str,
) -> None:
    cell_width = 160
    cell_height = 56
    margin_left = 210
    margin_top = 110
    width = margin_left + cell_width * len(column_labels) + 60
    height = margin_top + cell_height * len(row_labels) + 70
    values = [matrix.get(row, {}).get(column, 0.0) for row in row_labels for column in column_labels]
    max_value = max(values, default=1.0) or 1.0

    def fill(value: float) -> str:
        intensity = value / max_value if max_value else 0.0
        red = 248 - int(80 * intensity)
        green = 235 - int(150 * intensity)
        blue = 226 - int(190 * intensity)
        return f"rgb({red},{green},{blue})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f7f4"/>',
        f'<text x="40" y="38" font-size="24" font-family="Georgia, serif" fill="#1f2937">{title}</text>',
        f'<text x="40" y="60" font-size="13" font-family="Arial, sans-serif" fill="#6b7280">{subtitle}</text>',
    ]
    for column_index, column_label in enumerate(column_labels):
        x = margin_left + column_index * cell_width + cell_width / 2
        parts.append(
            f'<text x="{x}" y="{margin_top - 18}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#374151">{column_label}</text>'
        )
    for row_index, row_label in enumerate(row_labels):
        y = margin_top + row_index * cell_height
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 34}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#374151">{row_label}</text>'
        )
        for column_index, column_label in enumerate(column_labels):
            x = margin_left + column_index * cell_width
            value = matrix.get(row_label, {}).get(column_label, 0.0)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_width - 8}" height="{cell_height - 8}" rx="8" fill="{fill(value)}"/>')
            parts.append(
                f'<text x="{x + (cell_width - 8) / 2}" y="{y + 30}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#111827">{value:.1f}%</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def render_dumbbell_chart(path: Path, rows: list[dict[str, object]], title: str, subtitle: str) -> None:
    width = 960
    height = 380
    margin_left = 240
    margin_top = 80
    inner_width = width - margin_left - 70
    row_gap = 48
    max_value = max(max(float(row["real_p50"]), float(row["fake_p50"])) for row in rows) if rows else 1.0
    max_value = max_value or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f7f4"/>',
        f'<text x="40" y="38" font-size="24" font-family="Georgia, serif" fill="#1f2937">{title}</text>',
        f'<text x="40" y="60" font-size="13" font-family="Arial, sans-serif" fill="#6b7280">{subtitle}</text>',
    ]
    for index, row in enumerate(rows):
        y = margin_top + index * row_gap
        real_x = margin_left + (float(row["real_p50"]) / max_value) * inner_width
        fake_x = margin_left + (float(row["fake_p50"]) / max_value) * inner_width
        parts.append(f'<text x="{margin_left - 12}" y="{y + 5}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#374151">{row["column"]}</text>')
        parts.append(f'<line x1="{real_x}" y1="{y}" x2="{fake_x}" y2="{y}" stroke="#9ca3af" stroke-width="2"/>')
        parts.append(f'<circle cx="{real_x}" cy="{y}" r="7" fill="#2563eb"/>')
        parts.append(f'<circle cx="{fake_x}" cy="{y}" r="7" fill="#dc2626"/>')
        parts.append(f'<text x="{real_x - 12}" y="{y - 12}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#2563eb">{float(row["real_p50"]):.2f}</text>')
        parts.append(f'<text x="{fake_x + 12}" y="{y - 12}" font-size="11" font-family="Arial, sans-serif" fill="#dc2626">{float(row["fake_p50"]):.2f}</text>')
    parts.append('<text x="40" y="350" font-size="11" font-family="Arial, sans-serif" fill="#2563eb">Blue = median real listing</text>')
    parts.append('<text x="250" y="350" font-size="11" font-family="Arial, sans-serif" fill="#dc2626">Red = median fake listing</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def summarize_overview(bundle: DatasetBundle) -> dict[str, object]:
    clean_rows = bundle.clean_rows
    fake_count = sum(record.fake for record in bundle.rows)
    top_missing = sorted(
        (
            {
                "column": column,
                "missing_count": bundle.missing_counts[column],
                "missing_pct": pct(bundle.missing_counts[column], len(bundle.rows)),
                "unique_count": bundle.unique_counts[column],
            }
            for column in bundle.unique_counts
        ),
        key=lambda row: row["missing_pct"],
        reverse=True,
    )
    return {
        "row_count": len(bundle.rows),
        "clean_row_count": len(clean_rows),
        "fake_count": fake_count,
        "fake_rate": pct(fake_count, len(bundle.rows)),
        "exact_duplicates": len(bundle.duplicate_indices),
        "conflicting_duplicate_rows": len(bundle.conflicting_duplicate_indices),
        "top_missing": top_missing[:8],
    }


def grouped_fake_rates(rows: list[Record], column: str, min_support: int = 50, top_n: int = 12) -> list[dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in rows:
        label = str(record.values[column]) or "<MISSING>"
        grouped[label][0] += 1
        grouped[label][1] += record.fake

    result = []
    for label, (count, fake_count) in grouped.items():
        if count < min_support:
            continue
        result.append(
            {
                "column": column,
                "label": label,
                "count": count,
                "fake_count": fake_count,
                "fake_rate": pct(fake_count, count),
            }
        )
    result.sort(key=lambda row: (-row["fake_rate"], -row["count"], row["label"]))
    return result[:top_n]


def numeric_comparison(rows: list[Record]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    by_label: dict[int, dict[str, list[float]]] = {
        0: {column: [] for column in NUMERIC_COLUMNS},
        1: {column: [] for column in NUMERIC_COLUMNS},
    }
    for record in rows:
        for column in NUMERIC_COLUMNS:
            value = record.values[column]
            if value is not None:
                by_label[record.fake][column].append(float(value))

    for column in NUMERIC_COLUMNS:
        real_values = by_label[0][column]
        fake_values = by_label[1][column]
        output.append(
            {
                "column": column,
                "real_p10": quantile(real_values, 0.1),
                "real_p50": quantile(real_values, 0.5),
                "real_p90": quantile(real_values, 0.9),
                "fake_p10": quantile(fake_values, 0.1),
                "fake_p50": quantile(fake_values, 0.5),
                "fake_p90": quantile(fake_values, 0.9),
            }
        )
    return output


def repeated_entity_summary(rows: list[Record]) -> dict[str, object]:
    product_counter = Counter()
    brand_product_counter = Counter()
    for record in rows:
        product = str(record.values["ProductName"])
        brand = str(record.values["Brand"])
        product_counter[product] += 1
        brand_product_counter[(brand, product)] += 1

    repeated_product_rows = sum(count for count in product_counter.values() if count > 1)
    repeated_brand_product_rows = sum(count for count in brand_product_counter.values() if count > 1)
    return {
        "unique_product_names": len(product_counter),
        "repeated_product_names": sum(1 for count in product_counter.values() if count > 1),
        "rows_in_repeated_product_names": repeated_product_rows,
        "unique_brand_product_pairs": len(brand_product_counter),
        "repeated_brand_product_pairs": sum(1 for count in brand_product_counter.values() if count > 1),
        "rows_in_repeated_brand_product_pairs": repeated_brand_product_rows,
        "top_repeated_products": [
            {"product_name": name, "count": count}
            for name, count in product_counter.most_common(10)
        ],
    }


def review_signal_summary(rows: list[Record]) -> dict[str, object]:
    star_counts = Counter()
    review_lengths = {0: [], 1: []}
    for record in rows:
        review_text = str(record.values["top_review"])
        review_lengths[record.fake].append(len(review_text.split()))
        match = STAR_PATTERN.search(review_text)
        bucket = match.group(1) if match else "missing_or_no_star"
        star_counts[(bucket, record.fake)] += 1

    stars = []
    for bucket in ["1", "2", "3", "4", "5", "missing_or_no_star"]:
        total = star_counts[(bucket, 0)] + star_counts[(bucket, 1)]
        stars.append(
            {
                "bucket": bucket,
                "real_count": star_counts[(bucket, 0)],
                "fake_count": star_counts[(bucket, 1)],
                "fake_rate": pct(star_counts[(bucket, 1)], total),
            }
        )

    return {
        "review_length": {
            "real_p50": quantile(review_lengths[0], 0.5),
            "real_p90": quantile(review_lengths[0], 0.9),
            "fake_p50": quantile(review_lengths[1], 0.5),
            "fake_p90": quantile(review_lengths[1], 0.9),
        },
        "stars": stars,
    }


def token_signal_summary(rows: list[Record], limit: int = 12) -> dict[str, list[dict[str, object]]]:
    token_counts = {0: Counter(), 1: Counter()}
    for record in rows:
        text = " ".join(
            [
                str(record.values["ProductName"]),
                str(record.values["product_description"]),
                str(record.values["top_review"]),
            ]
        ).lower()
        tokens = {
            token
            for token in TOKEN_PATTERN.findall(text)
            if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
        }
        token_counts[record.fake].update(tokens)

    fake_salient = []
    for token, count in token_counts[1].most_common(100):
        if token_counts[0][token] < count * 8:
            fake_salient.append({"token": token, "fake_rows": count, "real_rows": token_counts[0][token]})
        if len(fake_salient) >= limit:
            break

    real_salient = []
    for token, count in token_counts[0].most_common(100):
        if token_counts[1][token] < count * 0.2:
            real_salient.append({"token": token, "real_rows": count, "fake_rows": token_counts[1][token]})
        if len(real_salient) >= limit:
            break

    return {"fake_salient": fake_salient, "real_salient": real_salient}


def temporal_summary(rows: list[Record]) -> list[dict[str, object]]:
    grouped: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for record in rows:
        year = record.purch_date.year
        grouped[year][0] += 1
        grouped[year][1] += record.fake

    output = []
    for year in sorted(grouped):
        count, fake_count = grouped[year]
        output.append(
            {
                "purchase_year": year,
                "count": count,
                "fake_count": fake_count,
                "fake_rate": pct(fake_count, count),
            }
        )
    return output


def market_vintage_summary(rows: list[Record]) -> list[dict[str, object]]:
    grouped: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for record in rows:
        vintage = int(float(record.values["MarketDate"]))
        grouped[vintage][0] += 1
        grouped[vintage][1] += record.fake

    output = []
    for vintage in sorted(grouped):
        count, fake_count = grouped[vintage]
        output.append(
            {
                "market_year": vintage,
                "count": count,
                "fake_count": fake_count,
                "fake_rate": pct(fake_count, count),
            }
        )
    return output


def combined_fake_rates(
    rows: list[Record],
    first: str,
    second: str,
    min_support: int = 50,
    top_n: int = 10,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for record in rows:
        first_value = str(record.values[first]) or "<MISSING>"
        second_value = str(record.values[second]) or "<MISSING>"
        grouped[(first_value, second_value)][0] += 1
        grouped[(first_value, second_value)][1] += record.fake

    output = []
    for (first_value, second_value), (count, fake_count) in grouped.items():
        if count < min_support:
            continue
        output.append(
            {
                "first": first,
                "second": second,
                "first_value": first_value,
                "second_value": second_value,
                "label": f"{first_value} | {second_value}",
                "count": count,
                "fake_count": fake_count,
                "fake_rate": pct(fake_count, count),
            }
        )
    output.sort(key=lambda row: (-row["fake_rate"], -row["count"], row["label"]))
    return output[:top_n]


def missingness_signal_summary(rows: list[Record]) -> list[dict[str, object]]:
    checks = {
        "top_review_missing": lambda record: not record.values["top_review"],
        "brand_missing": lambda record: not record.values["Brand"],
        "delivery_missing": lambda record: not record.values["DeliveryCategory"],
    }
    output = []
    for label, predicate in checks.items():
        missing_count = 0
        missing_fake = 0
        present_count = 0
        present_fake = 0
        for record in rows:
            if predicate(record):
                missing_count += 1
                missing_fake += record.fake
            else:
                present_count += 1
                present_fake += record.fake
        output.append(
            {
                "signal": label,
                "missing_count": missing_count,
                "missing_fake_rate": pct(missing_fake, missing_count),
                "present_count": present_count,
                "present_fake_rate": pct(present_fake, present_count),
            }
        )
    return output


def brand_risk_summary(rows: list[Record], min_support: int = 80, top_n: int = 12) -> list[dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in rows:
        brand = str(record.values["Brand"]) or "<MISSING>"
        grouped[brand][0] += 1
        grouped[brand][1] += record.fake

    output = []
    for brand, (count, fake_count) in grouped.items():
        if count < min_support:
            continue
        output.append(
            {
                "brand": brand,
                "count": count,
                "fake_count": fake_count,
                "fake_rate": pct(fake_count, count),
            }
        )
    output.sort(key=lambda row: (-row["fake_rate"], -row["count"], row["brand"]))
    return output[:top_n]


def price_ratio_summary(rows: list[Record]) -> list[dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    order = ["<0.8x", "0.8-1.0x", "1.0-1.2x", ">=1.2x", "missing"]

    for record in rows:
        transaction_price = record.values["TransactionPrice"]
        confirmed_price = record.values["ConfirmedPrice"]
        if transaction_price is None or confirmed_price in (None, 0.0):
            bucket = "missing"
        else:
            ratio = float(transaction_price) / float(confirmed_price)
            if ratio < 0.8:
                bucket = "<0.8x"
            elif ratio < 1.0:
                bucket = "0.8-1.0x"
            elif ratio < 1.2:
                bucket = "1.0-1.2x"
            else:
                bucket = ">=1.2x"
        grouped[bucket][0] += 1
        grouped[bucket][1] += record.fake

    return [
        {
            "price_ratio_bucket": bucket,
            "count": grouped[bucket][0],
            "fake_count": grouped[bucket][1],
            "fake_rate": pct(grouped[bucket][1], grouped[bucket][0]),
        }
        for bucket in order
        if grouped[bucket][0] > 0
    ]


def build_summary_payload(bundle: DatasetBundle) -> dict[str, object]:
    rows = bundle.rows
    overview = summarize_overview(bundle)
    grouped_tables = {}
    for column in ("FullfillmentType", "DeliveryCategory", "Category", "VNST", "Brand"):
        grouped_tables[column] = grouped_fake_rates(rows, column)

    return {
        "overview": overview,
        "numeric_comparison": numeric_comparison(rows),
        "temporal_summary": temporal_summary(rows),
        "market_vintage_summary": market_vintage_summary(rows),
        "grouped_fake_rates": grouped_tables,
        "interaction_fake_rates": {
            "FullfillmentType__DeliveryCategory": combined_fake_rates(
                rows,
                "FullfillmentType",
                "DeliveryCategory",
            ),
        },
        "missingness_signals": missingness_signal_summary(rows),
        "brand_risk": brand_risk_summary(rows),
        "price_ratio_summary": price_ratio_summary(rows),
        "repeated_entities": repeated_entity_summary(rows),
        "review_signals": review_signal_summary(rows),
        "token_signals": token_signal_summary(rows),
    }


def write_artifacts(summary: dict[str, object], root: Path) -> None:
    tables = root / "tables"
    plots = root / "plots"
    assets = report_assets_root()

    write_json(tables / "analysis_summary.json", summary)
    write_csv(tables / "dataset_overview.csv", [summary["overview"]], list(summary["overview"].keys()))
    write_csv(tables / "numeric_comparison.csv", summary["numeric_comparison"], list(summary["numeric_comparison"][0].keys()))
    write_csv(tables / "temporal_summary.csv", summary["temporal_summary"], list(summary["temporal_summary"][0].keys()))
    write_csv(tables / "market_vintage_summary.csv", summary["market_vintage_summary"], list(summary["market_vintage_summary"][0].keys()))
    write_csv(tables / "review_stars.csv", summary["review_signals"]["stars"], list(summary["review_signals"]["stars"][0].keys()))
    write_csv(tables / "missingness_signals.csv", summary["missingness_signals"], list(summary["missingness_signals"][0].keys()))
    write_csv(tables / "brand_risk.csv", summary["brand_risk"], list(summary["brand_risk"][0].keys()))
    write_csv(tables / "price_ratio_summary.csv", summary["price_ratio_summary"], list(summary["price_ratio_summary"][0].keys()))
    write_csv(tables / "fake_salient_tokens.csv", summary["token_signals"]["fake_salient"], ["token", "fake_rows", "real_rows"])
    write_csv(tables / "real_salient_tokens.csv", summary["token_signals"]["real_salient"], ["token", "real_rows", "fake_rows"])

    for column, rows in summary["grouped_fake_rates"].items():
        if rows:
            write_csv(tables / f"grouped_fake_rates_{column}.csv", rows, list(rows[0].keys()))

    for name, rows in summary["interaction_fake_rates"].items():
        if rows:
            write_csv(tables / f"interaction_{name}.csv", rows, list(rows[0].keys()))

    missing_rows = summary["overview"]["top_missing"]
    write_csv(tables / "top_missing.csv", missing_rows, list(missing_rows[0].keys()))

    delivery_rows = summary["grouped_fake_rates"].get("DeliveryCategory", [])
    if delivery_rows:
        write_plot_pair(
            "delivery_category_fake_rate.svg",
            [(row["label"], float(row["fake_rate"])) for row in delivery_rows],
            "Fake Rate by Delivery Category",
            "Missing delivery information is a meaningful signal in this dataset.",
            plots,
            assets,
            color="#c2410c",
        )

    temporal_rows = summary["temporal_summary"]
    write_plot_pair(
        "purchase_year_fake_rate.svg",
        [(str(row["purchase_year"]), float(row["fake_rate"])) for row in temporal_rows],
        "Fake Rate by Purchase Year",
        "Older cohorts appear more fake-prone than more recent purchases.",
        plots,
        assets,
        color="#2563eb",
    )

    review_rows = summary["review_signals"]["stars"]
    write_plot_pair(
        "review_star_fake_rate.svg",
        [(row["bucket"], float(row["fake_rate"])) for row in review_rows],
        "Fake Rate by Review Star Bucket",
        "Explicit review-star language is one of the strongest shortcut signals in the dataset.",
        plots,
        assets,
        color="#b91c1c",
    )

    interaction_rows = summary["interaction_fake_rates"]["FullfillmentType__DeliveryCategory"]
    write_plot_pair(
        "fulfillment_delivery_interaction.svg",
        [(row["label"], float(row["fake_rate"])) for row in interaction_rows],
        "Fake Rate by Fulfillment x Delivery",
        "Missing delivery metadata becomes especially risky inside BUSINESS and CENTRAL flows.",
        plots,
        assets,
        color="#7c3aed",
    )

    vintage_rows = summary["market_vintage_summary"]
    write_plot_pair(
        "market_vintage_fake_rate.svg",
        [(str(row["market_year"]), float(row["fake_rate"])) for row in vintage_rows],
        "Fake Rate by Product Vintage",
        "Older product vintages are materially more counterfeit-prone than recent cohorts.",
        plots,
        assets,
        color="#0f766e",
    )

    ratio_rows = summary["price_ratio_summary"]
    write_plot_pair(
        "price_ratio_fake_rate.svg",
        [(row["price_ratio_bucket"], float(row["fake_rate"])) for row in ratio_rows],
        "Fake Rate by Transaction-to-Confirmed Price Ratio",
        "Listings priced well above confirmed references are materially riskier.",
        plots,
        assets,
        color="#1d4ed8",
    )

    interaction_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    row_labels = ["BUSINESS", "CENTRAL", "MARKETPLACE"]
    column_labels = ["<MISSING>", "DIRECT", "PRIME"]
    for row in interaction_rows:
        interaction_matrix[row["first_value"]][row["second_value"]] = float(row["fake_rate"])
    for destination in (plots / "fulfillment_delivery_heatmap.svg", assets / "fulfillment_delivery_heatmap.svg"):
        render_heatmap(
            destination,
            interaction_matrix,
            row_labels,
            column_labels,
            "Heatmap: Fulfillment x Delivery Risk",
            "The missing-delivery shortcut is especially severe inside BUSINESS and CENTRAL flows.",
        )

    for destination in (plots / "numeric_median_comparison.svg", assets / "numeric_median_comparison.svg"):
        render_dumbbell_chart(
            destination,
            summary["numeric_comparison"],
            "Median Numeric Features: Real vs Fake Listings",
            "Fake listings skew older and cheaper across the main price signals.",
        )


def main() -> None:
    bundle = load_dataset()
    summary = build_summary_payload(bundle)
    root = artifacts_root()
    write_artifacts(summary, root)
    print(f"Wrote analysis artifacts to {root}")


if __name__ == "__main__":
    main()
