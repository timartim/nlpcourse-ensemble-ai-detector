from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm

from cli import classification_metrics, normalize_label, write_table
from model_service import BertAIDetector, DetectorConfig


DEFAULT_DATASET = "iitolstykh/LLMTrace_classification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the project detector on LLMTrace classification split.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--lang", default="ru")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-summary", default="data/output_data/llmtrace_test_summary.json")
    parser.add_argument("--output-predictions", default="data/output_data/llmtrace_test_predictions.jsonl")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--detector-type", choices=["ensemble", "hf"], default="ensemble")
    parser.add_argument("--model-path", default="trained_models/ensemble_models_2_3000/manifest.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ai-label-index", type=int, default=1)
    return parser.parse_args()


def resolve_device(device: str | None) -> str | None:
    if not device or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps was requested, but torch.backends.mps.is_available() is False.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but torch.cuda.is_available() is False.")
    return device


def load_llmtrace(args: argparse.Namespace) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load LLMTrace: pip install datasets") from exc

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.lang:
        dataset = dataset.filter(lambda row: row["lang"] == args.lang)
    df = dataset.to_pandas()
    if args.max_samples:
        df = df.head(args.max_samples).copy()
    required = {"text", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LLMTrace dataset is missing required columns: {missing}. Available columns: {list(df.columns)}")
    df = df[df["text"].fillna("").astype(str).str.strip().astype(bool)].reset_index(drop=True)
    df = df[df["label"].astype(str).map(normalize_label).isin({"ai", "not_ai"})].reset_index(drop=True)
    return df


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    df = load_llmtrace(args)
    detector = BertAIDetector(
        DetectorConfig(
            detector_type=args.detector_type,
            model_path=args.model_path,
            threshold=args.threshold,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=resolve_device(args.device),
            ai_label_index=args.ai_label_index,
        )
    )

    results = []
    starts = range(0, len(df), args.batch_size)
    for start in tqdm(starts, desc=f"LLMTrace eval ({args.split}, {args.lang})", disable=args.no_progress):
        texts = df["text"].iloc[start : start + args.batch_size].astype(str).tolist()
        results.extend(detector.score_batch(texts))

    y_true = df["label"].astype(str).map(normalize_label).tolist()
    y_pred = [normalize_label(result.label) for result in results]
    probabilities = [float(result.probability_ai) for result in results]

    labels = ["not_ai", "ai"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    p_ai, r_ai, f_ai, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["ai"],
        average=None,
        zero_division=0,
    )
    p_human, r_human, f_human, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["not_ai"],
        average=None,
        zero_division=0,
    )

    predictions = df.copy()
    predictions["probability_ai"] = probabilities
    predictions["predicted_label"] = y_pred

    summary: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "lang": args.lang,
        "n_samples": len(df),
        "detector": {
            "detector_type": args.detector_type,
            "model_path": args.model_path,
            "threshold": args.threshold,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "device": detector.device,
        },
        "label_distribution": {
            "true": series_counts_json_safe(y_true),
            "predicted": series_counts_json_safe(y_pred),
        },
        "metrics": {
            **classification_metrics(y_true, y_pred),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f_macro),
            "precision_ai": float(p_ai[0]),
            "recall_ai": float(r_ai[0]),
            "f1_ai": float(f_ai[0]),
            "precision_human": float(p_human[0]),
            "recall_human": float(r_human[0]),
            "f1_human": float(f_human[0]),
            "mean_probability_ai": float(np.mean(probabilities)) if probabilities else 0.0,
            "median_probability_ai": float(np.median(probabilities)) if probabilities else 0.0,
        },
        "confusion_matrix_rows_true_cols_pred_labels_not_ai_ai": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=["not_ai(human)", "AI(ai)"],
            digits=4,
            zero_division=0,
        ),
    }

    if "model" in predictions.columns:
        summary["by_model"] = grouped_metrics(predictions, group_col="model")
    return summary, predictions


def grouped_metrics(predictions: pd.DataFrame, *, group_col: str) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for group_value, group_df in predictions.groupby(group_col, dropna=False):
        y_true = group_df["label"].astype(str).map(normalize_label).tolist()
        y_pred = group_df["predicted_label"].astype(str).map(normalize_label).tolist()
        grouped[str(group_value)] = {
            "n_samples": len(group_df),
            "metrics": classification_metrics(y_true, y_pred),
            "mean_probability_ai": float(group_df["probability_ai"].mean()),
        }
    return grouped


def series_counts_json_safe(values: list[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in pd.Series(values).value_counts().sort_index().items()}


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    summary, predictions = evaluate(args)

    if args.output_predictions:
        Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
        write_table(predictions, args.output_predictions)

    if args.output_summary:
        summary_path = Path(args.output_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(to_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(to_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
