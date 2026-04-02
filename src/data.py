from __future__ import annotations

import csv
import html
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

EXPECTED_COLUMNS = [
    "RefId",
    "PurchDate",
    "VNZIP1",
    "VNST",
    "MarketDate",
    "ProductAge",
    "AveragePrice",
    "TransactionPrice",
    "ConfirmedPrice",
    "ProductName",
    "Category",
    "Brand",
    "FullfillmentType",
    "DeliveryCategory",
    "Fake",
    "product_description",
    "top_review",
]

TEXT_COLUMNS = ("ProductName", "product_description", "top_review")
NUMERIC_COLUMNS = (
    "MarketDate",
    "ProductAge",
    "AveragePrice",
    "TransactionPrice",
    "ConfirmedPrice",
)
CATEGORICAL_COLUMNS = (
    "VNZIP1",
    "VNST",
    "Category",
    "Brand",
    "FullfillmentType",
    "DeliveryCategory",
)


@dataclass(frozen=True)
class Record:
    ref_id: str
    purch_date_raw: str
    purch_date: date
    fake: int
    values: dict[str, object]


@dataclass(frozen=True)
class DatasetBundle:
    rows: list[Record]
    duplicate_indices: list[int]
    conflicting_duplicate_indices: list[int]
    missing_counts: Counter[str]
    unique_counts: dict[str, int]

    @property
    def clean_rows(self) -> list[Record]:
        blocked = set(self.duplicate_indices) | set(self.conflicting_duplicate_indices)
        return [row for idx, row in enumerate(self.rows) if idx not in blocked]


class SchemaError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_path() -> Path:
    return project_root() / "data_fakes.csv"


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(html.unescape(value).replace("\xa0", " ").split())


def parse_purchase_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%d-%m").date()


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def validate_columns(fieldnames: Iterable[str] | None) -> list[str]:
    actual = list(fieldnames or [])
    if actual != EXPECTED_COLUMNS:
        raise SchemaError(f"Unexpected columns. Expected {EXPECTED_COLUMNS}, got {actual}")
    return actual


def clean_row(raw: dict[str, str]) -> Record:
    cleaned: dict[str, object] = {}
    for key, value in raw.items():
        if key in TEXT_COLUMNS or key in CATEGORICAL_COLUMNS:
            cleaned[key] = normalize_text(value)
        elif key == "PurchDate":
            cleaned[key] = parse_purchase_date(value)
        elif key == "Fake":
            cleaned[key] = parse_int(value)
        elif key == "RefId":
            cleaned[key] = normalize_text(value)
        else:
            cleaned[key] = parse_float(value)

    fake = cleaned["Fake"]
    if fake not in (0, 1):
        raise SchemaError(f"Unexpected target value: {fake!r}")

    return Record(
        ref_id=str(cleaned["RefId"]),
        purch_date_raw=str(raw["PurchDate"]),
        purch_date=cleaned["PurchDate"],
        fake=fake,
        values=cleaned,
    )


def fingerprint_record(record: Record, include_target: bool = False) -> tuple[object, ...]:
    keys = [column for column in EXPECTED_COLUMNS if column != "RefId"]
    items: list[object] = []
    for key in keys:
        if key == "Fake" and not include_target:
            continue
        value = record.values[key]
        if isinstance(value, date):
            items.append(value.isoformat())
        else:
            items.append(value)
    return tuple(items)


def load_dataset(csv_path: Path | None = None) -> DatasetBundle:
    path = csv_path or default_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    rows: list[Record] = []
    missing_counts: Counter[str] = Counter()
    unique_values: dict[str, set[object]] = {column: set() for column in EXPECTED_COLUMNS}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)
        for raw_row in reader:
            record = clean_row(raw_row)
            rows.append(record)
            for column in EXPECTED_COLUMNS:
                value = record.values[column]
                if value in (None, ""):
                    missing_counts[column] += 1
                unique_values[column].add(value)

    duplicate_indices, conflicting_duplicate_indices = detect_duplicates(rows)
    unique_counts = {column: len(values) for column, values in unique_values.items()}
    return DatasetBundle(
        rows=rows,
        duplicate_indices=duplicate_indices,
        conflicting_duplicate_indices=conflicting_duplicate_indices,
        missing_counts=missing_counts,
        unique_counts=unique_counts,
    )


def detect_duplicates(rows: list[Record]) -> tuple[list[int], list[int]]:
    exact_seen: dict[tuple[object, ...], int] = {}
    feature_to_targets: dict[tuple[object, ...], set[int]] = {}
    feature_to_indices: dict[tuple[object, ...], list[int]] = {}
    duplicate_indices: list[int] = []

    for idx, record in enumerate(rows):
        exact_key = fingerprint_record(record, include_target=True)
        if exact_key in exact_seen:
            duplicate_indices.append(idx)
        else:
            exact_seen[exact_key] = idx

        feature_key = fingerprint_record(record, include_target=False)
        feature_to_targets.setdefault(feature_key, set()).add(record.fake)
        feature_to_indices.setdefault(feature_key, []).append(idx)

    conflicting_duplicate_indices: list[int] = []
    for feature_key, targets in feature_to_targets.items():
        if len(targets) > 1:
            conflicting_duplicate_indices.extend(feature_to_indices[feature_key])

    return sorted(set(duplicate_indices)), sorted(set(conflicting_duplicate_indices))


def label_balance(rows: list[Record]) -> Counter[int]:
    return Counter(record.fake for record in rows)


def missing_percentages(bundle: DatasetBundle) -> dict[str, float]:
    total = len(bundle.rows) or 1
    return {
        column: round(bundle.missing_counts[column] * 100.0 / total, 4)
        for column in EXPECTED_COLUMNS
    }
