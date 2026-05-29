from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from model_service import BertAIDetector, BertAIObfuscator, DetectorConfig, ObfuscatorConfig


def read_texts(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]
    if args.input_file:
        return [Path(args.input_file).read_text(encoding="utf-8")]
    raise ValueError("Provide --text or --input-file.")


def build_config(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(
        detector_type=args.detector_type,
        model_path=args.model_path,
        threshold=args.threshold,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )


def cmd_score(args: argparse.Namespace) -> None:
    detector = BertAIDetector(build_config(args))
    results = [asdict(result) for result in detector.score_batch(read_texts(args))]
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_obfuscate(args: argparse.Namespace) -> None:
    detector = BertAIDetector(build_config(args))
    obfuscator = BertAIObfuscator(detector, build_obfuscator_config(args))

    if args.dataset:
        dataset = load_table(args.dataset)
        df_out, df_log = obfuscator.process_dataframe(
            dataset,
            text_col=args.text_col,
            out_col=args.output_col,
        )
        if not args.output_path:
            raise ValueError("Provide --output-path when using --dataset.")
        write_table(df_out, args.output_path)
        if args.log_path:
            write_table(df_log, args.log_path)
        print(
            json.dumps(
                {
                    "rows": len(df_out),
                    "rewrites": len(df_log),
                    "output_path": args.output_path,
                    "log_path": args.log_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    result = obfuscator.obfuscate(read_texts(args)[0])
    payload = asdict(result)
    if args.output_path:
        Path(args.output_path).write_text(result.obfuscated_text, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_test(args: argparse.Namespace) -> None:
    dataset = load_table(args.dataset)
    texts = dataset[args.text_col].astype(str).tolist()
    labels = dataset[args.label_col].astype(str).tolist()

    detector = BertAIDetector(build_config(args))
    results = detector.score_batch(texts)
    predicted = [result.label for result in results]
    probabilities = [result.probability_ai for result in results]

    metrics = classification_metrics(labels, predicted)
    output = {
        "metrics": metrics,
        "n_samples": len(texts),
        "predictions": [
            {"label": pred, "probability_ai": prob}
            for pred, prob in zip(predicted, probabilities)
        ],
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "n_samples": len(texts)}, ensure_ascii=False, indent=2))


def cmd_heatmap(args: argparse.Namespace) -> None:
    from heatmap.analysis import run_heatmap_analysis
    from heatmap.run_scientific_heatmap import load_articles

    df_articles = load_articles(
        args.input_path,
        text_col=args.text_col,
        article_id_col=args.article_id_col,
    )

    detector = BertAIDetector(build_config(args))
    result = run_heatmap_analysis(
        detector=detector,
        df_articles=df_articles,
        out_dir=args.output_dir,
        text_col=args.text_col,
        article_id_col=args.article_id_col,
        title_col=args.title_col,
        batch_size=args.batch_size,
        window_size=args.window_size,
        min_sentences=args.min_sentences,
        stride=args.stride,
        threshold=args.threshold,
        max_articles_in_heatmap=args.max_articles_in_heatmap,
        show_progress=not args.no_progress,
    )

    summary_json = Path(args.output_dir) / "global_stats.json"
    summary_json.write_text(json.dumps(result["global_stats"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== GLOBAL STATS ===")
    for key, value in result["global_stats"].items():
        print(f"{key}: {value}")

    print("\n=== TOP ARTICLES BY mean_p_ai ===")
    print(result["article_summary"].head(10).to_string(index=False))

    print("\n=== TOP FEATURE DIFFERENCES ===")
    print(result["feature_comparison"].head(12).to_string(index=False))

    print("\n=== SAVED FILES ===")
    for name, path in result["artifacts"].items():
        print(f"{name}: {path}")
    print(f"global_stats_json: {summary_json}")


def cmd_train(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError("Training requires datasets, scikit-learn and transformers.") from exc

    df = load_table(args.dataset)
    df = df[[args.text_col, args.label_col]].dropna().copy()
    labels = sorted(df[args.label_col].astype(str).unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    df["label"] = df[args.label_col].astype(str).map(label_to_id)

    train_df, eval_df = train_test_split(
        df[[args.text_col, "label"]],
        test_size=args.eval_size,
        random_state=args.seed,
        stratify=df["label"] if len(labels) > 1 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch[args.text_col], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
    eval_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label={idx: label for label, idx in label_to_id.items()},
        label2id=label_to_id,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "labels.json").write_text(json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8")


def load_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError("Supported dataset formats: csv, parquet, json, jsonl.")


def write_table(df: pd.DataFrame, path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        return
    if suffix == ".json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
        return
    raise ValueError("Supported output formats: csv, parquet, json, jsonl.")


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    total = len(y_true)
    correct = sum(1 for actual, pred in zip(y_true, y_pred) if normalize_label(actual) == normalize_label(pred))
    return {"accuracy": correct / total if total else 0.0}


def normalize_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized in {"1", "ai", "true", "generated"}:
        return "ai"
    return "not_ai"


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector-type", choices=["ensemble", "hf"], default="ensemble")
    parser.add_argument("--model-path", default="trained_models/ensemble_models_2_3000/manifest.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)


def add_obfuscator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rewrite-threshold", type=float, default=0.8)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--anchor", choices=["first", "center", "last"], default="center")
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--sleep-between-calls", type=float, default=0.0)


def build_obfuscator_config(args: argparse.Namespace) -> ObfuscatorConfig:
    return ObfuscatorConfig(
        window_size=args.window_size,
        stride=args.stride,
        anchor=args.anchor,
        aggregation=args.aggregation,
        rewrite_threshold=args.rewrite_threshold,
        sleep_between_calls=args.sleep_between_calls,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BERT AI Detector CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score one text or a text file.")
    add_model_args(score_parser)
    score_parser.add_argument("--text")
    score_parser.add_argument("--input-file")
    score_parser.set_defaults(func=cmd_score)

    obfuscate_parser = subparsers.add_parser("obfuscate", help="Rewrite AI-looking sentences in text or dataset.")
    add_model_args(obfuscate_parser)
    add_obfuscator_args(obfuscate_parser)
    obfuscate_parser.add_argument("--text")
    obfuscate_parser.add_argument("--input-file")
    obfuscate_parser.add_argument("--dataset")
    obfuscate_parser.add_argument("--text-col", default="text")
    obfuscate_parser.add_argument("--output-col", default="text_rewritten")
    obfuscate_parser.add_argument("--output-path")
    obfuscate_parser.add_argument("--log-path")
    obfuscate_parser.set_defaults(func=cmd_obfuscate)

    test_parser = subparsers.add_parser("test", help="Evaluate a detector on a labeled dataset.")
    add_model_args(test_parser)
    test_parser.add_argument("--dataset", required=True)
    test_parser.add_argument("--text-col", default="text")
    test_parser.add_argument("--label-col", default="label")
    test_parser.add_argument("--output-json")
    test_parser.set_defaults(func=cmd_test)

    heatmap_parser = subparsers.add_parser("heatmap", help="Build heatmap analysis for scored article chunks.")
    add_model_args(heatmap_parser)
    heatmap_parser.add_argument("--input-path", required=True)
    heatmap_parser.add_argument("--output-dir", required=True)
    heatmap_parser.add_argument("--text-col", default="text")
    heatmap_parser.add_argument("--article-id-col", default="article_id")
    heatmap_parser.add_argument("--title-col", default=None)
    heatmap_parser.add_argument("--window-size", type=int, default=4)
    heatmap_parser.add_argument("--min-sentences", type=int, default=3)
    heatmap_parser.add_argument("--stride", type=int, default=None)
    heatmap_parser.add_argument("--max-articles-in-heatmap", type=int, default=50)
    heatmap_parser.add_argument("--no-progress", action="store_true")
    heatmap_parser.set_defaults(func=cmd_heatmap)

    train_parser = subparsers.add_parser("train", help="Fine-tune a single HF sequence classifier.")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--base-model", default="DeepPavlov/rubert-base-cased")
    train_parser.add_argument("--output-dir", default="trained_models/custom_hf_model")
    train_parser.add_argument("--text-col", default="text")
    train_parser.add_argument("--label-col", default="label")
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--epochs", type=float, default=3.0)
    train_parser.add_argument("--train-batch-size", type=int, default=8)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--eval-size", type=float, default=0.15)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=cmd_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
