from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, Record, default_data_path, load_dataset, project_root
from src.modeling_stub import resolve_device

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    TabPFNClassifier = None
    ModelVersion = None

STAR_PATTERN = re.compile(r"(\d)/5\s*stars", re.IGNORECASE)
TOKEN_FLAGS = {
    "token_buy": "buy",
    "token_amazing": "amazing",
    "token_received": "received",
    "token_best": "best",
}


@dataclass(frozen=True)
class PreparedSplit:
    records: list[Record]
    features: np.ndarray
    labels: np.ndarray


class TabPFNUnavailableError(RuntimeError):
    pass


class TabPFNFeatureEncoder:
    def __init__(self) -> None:
        self.numeric_columns = list(NUMERIC_COLUMNS) + [
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
            "product_name_words",
            "description_words",
            "review_words",
            "review_has_star",
            "review_star_value",
            "review_exclamation_count",
            *TOKEN_FLAGS.keys(),
        ]
        self.category_vocabs: dict[str, dict[str, int]] = {}

    def fit(self, rows: Iterable[Record]) -> "TabPFNFeatureEncoder":
        counts_by_column: dict[str, dict[str, int]] = {column: {} for column in CATEGORICAL_COLUMNS}
        for record in rows:
            for column in CATEGORICAL_COLUMNS:
                value = str(record.values[column]) if record.values[column] else "<MISSING>"
                counts_by_column[column][value] = counts_by_column[column].get(value, 0) + 1

        for column, counts in counts_by_column.items():
            vocab = {"<UNK>": 0, "<MISSING>": 1}
            next_index = 2
            for value in sorted(counts):
                if value in vocab:
                    continue
                vocab[value] = next_index
                next_index += 1
            self.category_vocabs[column] = vocab
        return self

    def transform(self, rows: Iterable[Record]) -> np.ndarray:
        matrix = []
        for record in rows:
            numeric_values = self._numeric_features(record)
            categorical_values = [
                float(self.category_vocabs[column].get(str(record.values[column]) if record.values[column] else "<MISSING>", 0))
                for column in CATEGORICAL_COLUMNS
            ]
            matrix.append(numeric_values + categorical_values)
        return np.asarray(matrix, dtype=np.float32)

    def feature_names(self) -> list[str]:
        categorical_names = [f"cat_{column}" for column in CATEGORICAL_COLUMNS]
        return [*self.numeric_columns, *categorical_names]

    def _numeric_features(self, record: Record) -> list[float]:
        average_price = self._as_float(record.values["AveragePrice"])
        confirmed_price = self._as_float(record.values["ConfirmedPrice"])
        transaction_price = self._as_float(record.values["TransactionPrice"])
        product_name = str(record.values["ProductName"])
        description = str(record.values["product_description"])
        review = str(record.values["top_review"])
        star_match = STAR_PATTERN.search(review)
        star_value = float(star_match.group(1)) if star_match else 0.0
        review_lower = review.lower()

        base = [self._as_float(record.values[column]) for column in NUMERIC_COLUMNS]
        base.extend(
            [
                float(record.purch_date.year),
                float(record.purch_date.month),
                float(record.purch_date.day),
                float(record.purch_date.weekday()),
                self._safe_gap(transaction_price, average_price),
                self._safe_gap(transaction_price, confirmed_price),
                self._safe_ratio(transaction_price, average_price),
                self._safe_ratio(transaction_price, confirmed_price),
                float(average_price == 0.0),
                float(confirmed_price == 0.0),
                float(not record.values["Brand"]),
                float(not record.values["DeliveryCategory"]),
                float(not record.values["top_review"]),
                float(len(product_name.split())),
                float(len(description.split())),
                float(len(review.split())),
                float(bool(star_match)),
                star_value,
                float(review.count("!")),
            ]
        )
        for token in TOKEN_FLAGS.values():
            base.append(float(token in review_lower))
        return base

    @staticmethod
    def _as_float(value: object) -> float:
        if value in (None, ""):
            return 0.0
        return float(value)

    @staticmethod
    def _safe_ratio(left: float, right: float) -> float:
        if right == 0.0:
            return 0.0
        return left / right

    @staticmethod
    def _safe_gap(left: float, right: float) -> float:
        return left - right


def split_records(rows: list[Record]) -> dict[str, list[Record]]:
    train: list[Record] = []
    validation: list[Record] = []
    test: list[Record] = []
    for record in rows:
        stamp = (record.purch_date.year, record.purch_date.month, record.purch_date.day)
        if stamp <= (2020, 7, 31):
            train.append(record)
        elif stamp <= (2020, 9, 30):
            validation.append(record)
        else:
            test.append(record)
    return {"train": train, "validation": validation, "test": test}


def stratified_sample(rows: list[Record], max_rows: int, seed: int) -> list[Record]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    grouped = {0: [], 1: []}
    for record in rows:
        grouped[record.fake].append(record)
    rng = random.Random(seed)
    positives = round(max_rows * len(grouped[1]) / len(rows))
    positives = max(1, min(len(grouped[1]), positives))
    negatives = max_rows - positives
    negatives = max(1, min(len(grouped[0]), negatives))
    sampled = rng.sample(grouped[1], positives) + rng.sample(grouped[0], negatives)
    rng.shuffle(sampled)
    return sampled


def prepare_splits(rows: list[Record], max_train_rows: int | None, seed: int) -> tuple[TabPFNFeatureEncoder, dict[str, PreparedSplit]]:
    split_rows = split_records(rows)
    train_rows = stratified_sample(split_rows["train"], max_train_rows, seed) if max_train_rows else split_rows["train"]
    encoder = TabPFNFeatureEncoder().fit(train_rows)
    prepared = {}
    for name, split in (("train", train_rows), ("validation", split_rows["validation"]), ("test", split_rows["test"])):
        prepared[name] = PreparedSplit(
            records=split,
            features=encoder.transform(split),
            labels=np.asarray([record.fake for record in split], dtype=np.int64),
        )
    return encoder, prepared


def instantiate_classifier(device: str):
    if TabPFNClassifier is None or ModelVersion is None:
        raise TabPFNUnavailableError(
            "TabPFN is not installed. Install it with `pip install tabpfn` before running this pipeline."
        )
    try:
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device)
    except TypeError:
        classifier = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        if hasattr(classifier, "set_params"):
            try:
                classifier.set_params(device=device)
            except ValueError:
                pass
        return classifier


def positive_class_scores(model, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    return probabilities[:, 1]


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "positive_rate": round(float(np.mean(predictions)), 6),
        "average_precision": round(average_precision(labels, probabilities), 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp_cumsum = np.cumsum(sorted_labels == 1)
    precision = tp_cumsum / (np.arange(len(sorted_labels)) + 1)
    return float(np.sum(precision[sorted_labels == 1]) / positives)


def write_predictions(path: Path, records: list[Record], probabilities: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["RefId", "PurchDate", "Fake", "tabpfn_score", "tabpfn_prediction"],
        )
        writer.writeheader()
        for record, score in zip(records, probabilities, strict=True):
            writer.writerow(
                {
                    "RefId": record.ref_id,
                    "PurchDate": record.purch_date.isoformat(),
                    "Fake": record.fake,
                    "tabpfn_score": round(float(score), 8),
                    "tabpfn_prediction": int(score >= 0.5),
                }
            )


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TabPFN v2 inference on the fake-listing dataset.")
    parser.add_argument("--data-path", type=Path, default=default_data_path())
    parser.add_argument("--output-dir", type=Path, default=project_root() / "artifacts" / "tabpfn")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional cap for a quicker smoke test, useful on CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu-large-dataset",
        action="store_true",
        help="Sets TABPFN_ALLOW_CPU_LARGE_DATASET=true for local CPU runs on datasets larger than 1000 rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_cpu_large_dataset:
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"

    device = resolve_device() if args.device == "auto" else args.device
    bundle = load_dataset(args.data_path)
    rows = bundle.clean_rows
    encoder, prepared = prepare_splits(rows, args.max_train_rows, args.seed)

    if device == "cpu" and len(prepared["train"].records) > 1000 and not args.allow_cpu_large_dataset:
        raise RuntimeError(
            "TabPFN's official guidance recommends CPU only for small datasets (roughly <=1000 rows). "
            "Re-run with --allow-cpu-large-dataset or use --max-train-rows 1000 for a smoke test."
        )

    classifier = instantiate_classifier(device)
    classifier.fit(prepared["train"].features, prepared["train"].labels)

    validation_scores = positive_class_scores(classifier, prepared["validation"].features)
    test_scores = positive_class_scores(classifier, prepared["test"].features)

    output_dir: Path = args.output_dir
    write_predictions(output_dir / "validation_predictions.csv", prepared["validation"].records, validation_scores)
    write_predictions(output_dir / "test_predictions.csv", prepared["test"].records, test_scores)
    write_summary(
        output_dir / "run_summary.json",
        {
            "model": "TabPFN v2 classifier",
            "device": device,
            "feature_count": len(encoder.feature_names()),
            "feature_names": encoder.feature_names(),
            "train_rows": len(prepared["train"].records),
            "validation_rows": len(prepared["validation"].records),
            "test_rows": len(prepared["test"].records),
            "validation_metrics": classification_metrics(prepared["validation"].labels, validation_scores),
            "test_metrics": classification_metrics(prepared["test"].labels, test_scores),
        },
    )
    print(f"Wrote TabPFN artifacts to {output_dir}")


if __name__ == "__main__":
    main()
