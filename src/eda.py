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


def render_bar_chart(path: Path, rows: list[tuple[str, float]], title: str, subtitle: str) -> None:
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
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" rx="5" fill="#c2410c"/>')
        parts.append(f'<text x="{margin_left + bar_width + 8}" y="{y + bar_height - 4}" font-size="12" font-family="Arial, sans-serif" fill="#111827">{value:.1f}%</text>')
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
        "grouped_fake_rates": grouped_tables,
        "repeated_entities": repeated_entity_summary(rows),
        "review_signals": review_signal_summary(rows),
        "token_signals": token_signal_summary(rows),
    }


def write_artifacts(summary: dict[str, object], root: Path) -> None:
    tables = root / "tables"
    plots = root / "plots"

    write_json(tables / "analysis_summary.json", summary)
    write_csv(tables / "dataset_overview.csv", [summary["overview"]], list(summary["overview"].keys()))
    write_csv(tables / "numeric_comparison.csv", summary["numeric_comparison"], list(summary["numeric_comparison"][0].keys()))
    write_csv(tables / "temporal_summary.csv", summary["temporal_summary"], list(summary["temporal_summary"][0].keys()))
    write_csv(tables / "review_stars.csv", summary["review_signals"]["stars"], list(summary["review_signals"]["stars"][0].keys()))
    write_csv(tables / "fake_salient_tokens.csv", summary["token_signals"]["fake_salient"], ["token", "fake_rows", "real_rows"])
    write_csv(tables / "real_salient_tokens.csv", summary["token_signals"]["real_salient"], ["token", "real_rows", "fake_rows"])

    for column, rows in summary["grouped_fake_rates"].items():
        if rows:
            write_csv(tables / f"grouped_fake_rates_{column}.csv", rows, list(rows[0].keys()))

    missing_rows = summary["overview"]["top_missing"]
    write_csv(tables / "top_missing.csv", missing_rows, list(missing_rows[0].keys()))

    delivery_rows = summary["grouped_fake_rates"].get("DeliveryCategory", [])
    if delivery_rows:
        render_bar_chart(
            plots / "delivery_category_fake_rate.svg",
            [(row["label"], float(row["fake_rate"])) for row in delivery_rows],
            "Fake Rate by Delivery Category",
            "Missing delivery information is a meaningful signal in this dataset.",
        )

    temporal_rows = summary["temporal_summary"]
    render_bar_chart(
        plots / "purchase_year_fake_rate.svg",
        [(str(row["purchase_year"]), float(row["fake_rate"])) for row in temporal_rows],
        "Fake Rate by Purchase Year",
        "Older cohorts appear more fake-prone than more recent purchases.",
    )


def main() -> None:
    bundle = load_dataset()
    summary = build_summary_payload(bundle)
    root = artifacts_root()
    write_artifacts(summary, root)
    print(f"Wrote analysis artifacts to {root}")


if __name__ == "__main__":
    main()
