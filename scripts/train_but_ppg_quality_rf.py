#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import AppConfig
from app.processing.filters import PPGNoiseFilter
from app.processing.quality import FEATURE_NAMES, extract_morphology, extract_quality_features


DEFAULT_DATASET = "brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0"
DEFAULT_OUTPUT = "models/ppg_quality_rf.joblib"
LABELS = ["low", "high"]


@dataclass(frozen=True)
class FeatureDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    record_ids: list[str]
    skipped: dict[str, int]


def load_quality_labels(dataset: Path) -> dict[str, str]:
    labels_path = dataset / "quality-hr-ann.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"quality annotation file not found: {labels_path}")

    labels: dict[str, str] = {}
    with labels_path.open(encoding="utf-8-sig", newline="") as labels_file:
        for row in csv.DictReader(labels_file):
            record_id = str(row.get("ID", "")).strip()
            quality = str(row.get("Quality", "")).strip()
            if not record_id:
                continue
            if quality == "0":
                labels[record_id] = "low"
            elif quality == "1":
                labels[record_id] = "high"
            else:
                raise ValueError(f"unsupported Quality value for {record_id}: {quality!r}")
    return labels


def build_feature_dataset(
    dataset: Path,
    *,
    config: AppConfig | None = None,
    limit: int | None = None,
) -> FeatureDataset:
    config = config or AppConfig()
    ppg_filter = PPGNoiseFilter(config)
    labels = load_quality_labels(dataset)

    rows: list[list[float]] = []
    targets: list[str] = []
    groups: list[str] = []
    record_ids: list[str] = []
    skipped: Counter[str] = Counter()

    for record_id, label in sorted(labels.items()):
        loaded = _load_but_ppg_channels(dataset, record_id)
        if isinstance(loaded, str):
            skipped[loaded] += 1
            continue

        raw_primary, raw_secondary, fs = loaded
        try:
            filtered_primary = ppg_filter.filter_channel(raw_primary, fs)
            morphology = extract_morphology(filtered_ir=filtered_primary, fs=fs, config=config)
            feature_values = extract_quality_features(
                raw_ir=raw_primary,
                raw_red=raw_secondary,
                filtered_ir=filtered_primary,
                fs=fs,
                config=config,
                morphology=morphology,
            )
        except Exception as exc:
            skipped[f"feature_error:{type(exc).__name__}"] += 1
            continue

        rows.append([float(feature_values.get(name) or 0.0) for name in FEATURE_NAMES])
        targets.append(label)
        groups.append(_subject_id(record_id))
        record_ids.append(record_id)
        if limit is not None and len(rows) >= limit:
            break

    if not rows:
        raise ValueError(f"no usable BUT-PPG records found in {dataset}")

    return FeatureDataset(
        features=np.asarray(rows, dtype=float),
        labels=np.asarray(targets, dtype=object),
        groups=np.asarray(groups, dtype=object),
        record_ids=record_ids,
        skipped=dict(skipped),
    )


def train_random_forest(
    feature_dataset: FeatureDataset,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    random_state: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(feature_dataset.features, feature_dataset.labels)
    return model


def evaluate_random_forest(
    feature_dataset: FeatureDataset,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    random_state: int,
    n_splits: int,
    test_size: float,
) -> dict[str, Any]:
    if len(set(feature_dataset.labels.tolist())) < 2:
        raise ValueError("at least two quality classes are required for evaluation")
    if len(set(feature_dataset.groups.tolist())) < 2:
        raise ValueError("at least two subject groups are required for grouped evaluation")

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    splits: list[dict[str, Any]] = []
    balanced_scores: list[float] = []

    for split_index, (train_index, test_index) in enumerate(
        splitter.split(feature_dataset.features, feature_dataset.labels, feature_dataset.groups),
        start=1,
    ):
        split_dataset = FeatureDataset(
            features=feature_dataset.features[train_index],
            labels=feature_dataset.labels[train_index],
            groups=feature_dataset.groups[train_index],
            record_ids=[feature_dataset.record_ids[index] for index in train_index.tolist()],
            skipped={},
        )
        model = train_random_forest(
            split_dataset,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        expected = feature_dataset.labels[test_index]
        predicted = model.predict(feature_dataset.features[test_index])
        balanced_accuracy = float(balanced_accuracy_score(expected, predicted))
        balanced_scores.append(balanced_accuracy)
        splits.append(
            {
                "split": split_index,
                "train_records": int(train_index.size),
                "test_records": int(test_index.size),
                "train_groups": int(len(set(feature_dataset.groups[train_index].tolist()))),
                "test_groups": int(len(set(feature_dataset.groups[test_index].tolist()))),
                "balanced_accuracy": round(balanced_accuracy, 4),
                "confusion_matrix": confusion_matrix(expected, predicted, labels=LABELS).tolist(),
                "classification_report": classification_report(
                    expected,
                    predicted,
                    labels=LABELS,
                    output_dict=True,
                    zero_division=0,
                ),
            }
        )

    return {
        "n_splits": n_splits,
        "test_size": test_size,
        "balanced_accuracy": {
            "mean": round(float(np.mean(balanced_scores)), 4),
            "std": round(float(np.std(balanced_scores)), 4),
            "values": [round(score, 4) for score in balanced_scores],
        },
        "splits": splits,
    }


def write_model_and_metrics(
    *,
    model: RandomForestClassifier,
    output_path: Path,
    metrics_path: Path,
    feature_dataset: FeatureDataset,
    evaluation: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to save the trained model") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    label_counts = Counter(str(label) for label in feature_dataset.labels.tolist())
    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "output": str(output_path),
        "records": {
            "usable": int(feature_dataset.labels.size),
            "skipped": feature_dataset.skipped,
            "labels": dict(sorted(label_counts.items())),
            "groups": int(len(set(feature_dataset.groups.tolist()))),
        },
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": args.n_estimators,
            "class_weight": "balanced",
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
            "classes": [str(label) for label in model.classes_.tolist()],
        },
        "feature_names": FEATURE_NAMES,
        "feature_importances": {
            name: round(float(importance), 6)
            for name, importance in sorted(
                zip(FEATURE_NAMES, model.feature_importances_, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "evaluation": evaluation,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest PPG quality model on BUT-PPG.")
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET), help="Path to the BUT-PPG dataset root.")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Output .joblib model path.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Output JSON metrics path. Defaults to <output>.metrics.json.",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for quick local checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_output or args.output.with_suffix(".metrics.json")

    feature_dataset = build_feature_dataset(args.dataset, limit=args.limit)
    evaluation = evaluate_random_forest(
        feature_dataset,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_splits=args.splits,
        test_size=args.test_size,
    )
    model = train_random_forest(
        feature_dataset,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    write_model_and_metrics(
        model=model,
        output_path=args.output,
        metrics_path=metrics_path,
        feature_dataset=feature_dataset,
        evaluation=evaluation,
        args=args,
    )

    print(f"usable records: {feature_dataset.labels.size}")
    print(f"skipped records: {feature_dataset.skipped}")
    print(f"balanced accuracy mean: {evaluation['balanced_accuracy']['mean']}")
    print(f"model saved: {args.output}")
    print(f"metrics saved: {metrics_path}")


def _load_but_ppg_channels(dataset: Path, record_id: str) -> tuple[np.ndarray, np.ndarray, float] | str:
    header_path = dataset / record_id / f"{record_id}_PPG.hea"
    data_path = dataset / record_id / f"{record_id}_PPG.dat"
    if not header_path.exists() or not data_path.exists():
        return "missing_ppg_files"

    try:
        nsig, fs, nsamp = _read_wfdb_header_shape(header_path)
    except ValueError:
        return "invalid_header"

    data = np.fromfile(data_path, dtype="<i2").astype(float)
    if nsig == 3 and nsamp == 300 and abs(fs - 30.0) > 1e-9:
        return f"unsupported_sample_rate:{fs:g}"
    if nsig == 3 and nsamp == 300 and data.size == 900:
        samples = data.reshape(nsamp, nsig)
        return samples[:, 1], samples[:, 0], fs
    if nsig == 300 and nsamp == 1 and data.size == 300:
        return "legacy_transposed"
    return f"unsupported_shape:{nsig}x{nsamp}:{data.size}"


def _read_wfdb_header_shape(header_path: Path) -> tuple[int, float, int]:
    first_line = header_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    parts = first_line.split()
    if len(parts) < 4:
        raise ValueError(f"invalid WFDB header: {header_path}")
    return int(parts[1]), float(parts[2]), int(parts[3])


def _subject_id(record_id: str) -> str:
    return record_id[:3]


if __name__ == "__main__":
    main()
