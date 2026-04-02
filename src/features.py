from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, Record


@dataclass(frozen=True)
class FeatureRow:
    ref_id: str
    label: int
    numeric: dict[str, float]
    categorical: dict[str, str]
    text: str


DERIVED_NUMERIC_COLUMNS = (
    "purchase_year",
    "purchase_month",
    "purchase_day",
    "purchase_weekday",
    "price_gap_vs_average",
    "price_gap_vs_confirmed",
    "ratio_vs_average",
    "ratio_vs_confirmed",
    "average_price_missing",
    "confirmed_price_missing",
    "brand_missing",
    "delivery_missing",
    "review_missing",
)


def safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator in (None, 0.0):
        return 0.0
    return numerator / denominator


def safe_gap(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 0.0
    return left - right


def combined_text(record: Record) -> str:
    parts = [
        str(record.values["ProductName"]),
        str(record.values["product_description"]),
        str(record.values["top_review"]),
    ]
    return " [SEP] ".join(part for part in parts if part)


def build_feature_row(record: Record) -> FeatureRow:
    average_price = record.values["AveragePrice"]
    confirmed_price = record.values["ConfirmedPrice"]
    transaction_price = record.values["TransactionPrice"]

    numeric = {
        column: float(record.values[column] or 0.0)
        for column in NUMERIC_COLUMNS
    }
    numeric.update(
        {
            "purchase_year": float(record.purch_date.year),
            "purchase_month": float(record.purch_date.month),
            "purchase_day": float(record.purch_date.day),
            "purchase_weekday": float(record.purch_date.weekday()),
            "price_gap_vs_average": safe_gap(transaction_price, average_price),
            "price_gap_vs_confirmed": safe_gap(transaction_price, confirmed_price),
            "ratio_vs_average": safe_ratio(transaction_price, average_price),
            "ratio_vs_confirmed": safe_ratio(transaction_price, confirmed_price),
            "average_price_missing": float(average_price is None),
            "confirmed_price_missing": float(confirmed_price is None),
            "brand_missing": float(not record.values["Brand"]),
            "delivery_missing": float(not record.values["DeliveryCategory"]),
            "review_missing": float(not record.values["top_review"]),
        }
    )

    categorical = {
        column: str(record.values[column]) if record.values[column] else "<MISSING>"
        for column in CATEGORICAL_COLUMNS
    }

    return FeatureRow(
        ref_id=record.ref_id,
        label=record.fake,
        numeric=numeric,
        categorical=categorical,
        text=combined_text(record),
    )


def build_feature_rows(records: Iterable[Record]) -> list[FeatureRow]:
    return [build_feature_row(record) for record in records]


def build_vocab(rows: Iterable[FeatureRow], column: str, min_count: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.categorical[column]
        counts[value] = counts.get(value, 0) + 1

    vocab = {"<UNK>": 0, "<MISSING>": 1}
    next_index = len(vocab)
    for value in sorted(counts):
        if counts[value] < min_count or value in vocab:
            continue
        vocab[value] = next_index
        next_index += 1
    return vocab


def temporal_split(rows: Iterable[FeatureRow]) -> dict[str, list[FeatureRow]]:
    train: list[FeatureRow] = []
    validation: list[FeatureRow] = []
    test: list[FeatureRow] = []

    for row in rows:
        year = int(row.numeric["purchase_year"])
        month = int(row.numeric["purchase_month"])
        day = int(row.numeric["purchase_day"])
        stamp = (year, month, day)
        if stamp <= (2020, 7, 31):
            train.append(row)
        elif stamp <= (2020, 9, 30):
            validation.append(row)
        else:
            test.append(row)

    return {"train": train, "validation": validation, "test": test}
