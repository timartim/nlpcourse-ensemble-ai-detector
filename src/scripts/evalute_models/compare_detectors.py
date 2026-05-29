from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from cli import classification_metrics, load_table, normalize_label, write_table
from model_service import BertAIDetector, DetectorConfig


DEFAULT_BASELINE_MODEL = "orzhan/ruroberta-ruatd-binary"
DEFAULT_BASELINE_TOKENIZER = "sberbank-ai/ruRoberta-large"
DEFAULT_GIGACHECK_MODEL = "iitolstykh/GigaCheck-Classifier-Multi"
GENERATOR_ALIASES = {
    "chatgpt5": ["openai:gpt-5"],
    "gpt5": ["openai:gpt-5"],
    "openai:gpt5": ["openai:gpt-5"],
    "chatgpt5mini": ["openai:gpt-5-mini"],
    "gpt5mini": ["openai:gpt-5-mini"],
    "deepseek32": ["deepseek:deepseek-chat", "deepseek:deepseek-reasoner"],
    "deepseek3_2": ["deepseek:deepseek-chat", "deepseek:deepseek-reasoner"],
    "deepseek3.2": ["deepseek:deepseek-chat", "deepseek:deepseek-reasoner"],
    "deepseekv32": ["deepseek:deepseek-chat", "deepseek:deepseek-reasoner"],
    "deepseekchat": ["deepseek:deepseek-chat"],
    "deepseekreasoner": ["deepseek:deepseek-reasoner"],
}


@dataclass(frozen=True)
class DetectorRun:
    name: str
    model_path: str
    probability_ai: list[float]
    labels: list[str]


class HuggingFaceDetector:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: str | None = None,
        ai_label_index: int = 1,
        ai_label: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
        show_progress: bool = True,
        progress_desc: str = "HF baseline",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = resolve_torch_device(device)
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.ai_label_index = self._resolve_ai_label_index(ai_label, ai_label_index)

    @torch.no_grad()
    def predict_proba_ai(self, texts: list[str]) -> np.ndarray:
        probabilities: list[np.ndarray] = []
        starts = range(0, len(texts), self.batch_size)
        for start in tqdm(starts, desc=self.progress_desc, disable=not self.show_progress):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, self.ai_label_index]
            probabilities.append(batch_probs.detach().cpu().numpy().astype(np.float32))
        if not probabilities:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(probabilities)

    def _resolve_ai_label_index(self, ai_label: str | None, fallback_index: int) -> int:
        id2label = self.model.config.id2label or {}
        if ai_label:
            wanted = ai_label.strip().lower()
            for index, label in id2label.items():
                if str(label).strip().lower() == wanted:
                    return int(index)
            raise ValueError(f"Label {ai_label!r} was not found in model id2label={id2label!r}.")
        return fallback_index


class GigaCheckDetector:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        ai_label_index: int = 0,
        batch_size: int = 1,
        device: str | None = None,
        torch_dtype: str = "auto",
        show_progress: bool = True,
        progress_desc: str = "GigaCheck",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.ai_label_index = ai_label_index
        self.batch_size = batch_size
        self.device = resolve_torch_device(device)
        self.show_progress = show_progress
        self.progress_desc = progress_desc

        dtype = resolve_torch_dtype(torch_dtype, self.device)
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.device == "cuda":
            kwargs["device_map"] = "cuda:0"

        try:
            self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        except ImportError as exc:
            raise RuntimeError(
                "GigaCheck requires the gigacheck package. Install it with: "
                "pip install git+https://github.com/ai-forever/gigacheck"
            ) from exc

        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba_ai(self, texts: list[str]) -> np.ndarray:
        probabilities: list[np.ndarray] = []
        starts = range(0, len(texts), self.batch_size)
        for start in tqdm(starts, desc=self.progress_desc, disable=not self.show_progress):
            batch = [str(text).replace("\n", " ") for text in texts[start : start + self.batch_size]]
            output = self.model(batch)
            probs = output.classification_head_probs[:, self.ai_label_index]
            probabilities.append(probs.detach().cpu().numpy().astype(np.float32))
        if not probabilities:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(probabilities)


def run_own_detector(args: argparse.Namespace, texts: list[str]) -> DetectorRun:
    detector = BertAIDetector(
        DetectorConfig(
            detector_type=args.own_detector_type,
            model_path=args.own_model_path,
            threshold=args.threshold,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=resolve_torch_device(args.device),
            ai_label_index=args.own_ai_label_index,
        )
    )
    results = []
    starts = range(0, len(texts), args.batch_size)
    for start in tqdm(starts, desc=args.own_name, disable=args.no_progress):
        results.extend(detector.score_batch(texts[start : start + args.batch_size]))
    return DetectorRun(
        name=args.own_name,
        model_path=args.own_model_path,
        probability_ai=[result.probability_ai for result in results],
        labels=[result.label for result in results],
    )


def run_baseline_detector(args: argparse.Namespace, texts: list[str]) -> DetectorRun:
    if args.baseline_kind == "gigacheck":
        return run_gigacheck_detector(args, texts)

    ai_label_index = 1 if args.baseline_ai_label_index is None else args.baseline_ai_label_index
    detector = HuggingFaceDetector(
        args.baseline_model,
        tokenizer_name_or_path=args.baseline_tokenizer,
        ai_label_index=ai_label_index,
        ai_label=args.baseline_ai_label,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        show_progress=not args.no_progress,
        progress_desc=args.baseline_name,
    )
    probabilities = detector.predict_proba_ai(texts)
    labels = ["AI" if probability >= args.threshold else "not_ai" for probability in probabilities.tolist()]
    return DetectorRun(
        name=args.baseline_name,
        model_path=f"{args.baseline_model} tokenizer={args.baseline_tokenizer}",
        probability_ai=[float(probability) for probability in probabilities.tolist()],
        labels=labels,
    )


def run_gigacheck_detector(args: argparse.Namespace, texts: list[str]) -> DetectorRun:
    ai_label_index = 0 if args.baseline_ai_label_index is None else args.baseline_ai_label_index
    detector = GigaCheckDetector(
        args.baseline_model,
        ai_label_index=ai_label_index,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.baseline_torch_dtype,
        show_progress=not args.no_progress,
        progress_desc=args.baseline_name,
    )
    probabilities = detector.predict_proba_ai(texts)
    labels = ["AI" if probability >= args.threshold else "not_ai" for probability in probabilities.tolist()]
    return DetectorRun(
        name=args.baseline_name,
        model_path=args.baseline_model,
        probability_ai=[float(probability) for probability in probabilities.tolist()],
        labels=labels,
    )


def build_predictions_table(df: pd.DataFrame, text_col: str, runs: list[DetectorRun]) -> pd.DataFrame:
    output = df.copy()
    for run in runs:
        prefix = normalize_column_prefix(run.name)
        output[f"{prefix}_probability_ai"] = run.probability_ai
        output[f"{prefix}_label"] = run.labels
    return output


def save_predictions(df: pd.DataFrame, runs: list[DetectorRun], output_path: str | None) -> None:
    if not output_path:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_table(build_predictions_table(df, "text", runs), output_path)


def save_summary(summary: dict[str, Any], output_path: str | None) -> None:
    if not output_path:
        return
    summary_path = Path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(to_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")


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


def prepare_eval_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    df = load_table(args.dataset)
    if args.generated_text_col:
        df = build_generated_eval_dataframe(df, args)
        args.text_col = "text"
        args.label_col = args.label_col or "label"
        args.generator_col = "generator"
    if args.generators:
        df = filter_by_generators(
            df,
            args.generators,
            generator_col=args.generator_col,
            keep_human=args.include_human,
        )
    if args.max_samples:
        df = df.head(args.max_samples).copy()
    if args.max_samples_per_generator:
        if args.generator_col not in df.columns:
            raise ValueError(
                f"Cannot use --max-samples-per-generator: column {args.generator_col!r} was not found."
            )
        sampled_groups = [
            group.sample(min(len(group), args.max_samples_per_generator), random_state=args.seed)
            for _, group in df.groupby(args.generator_col, dropna=False)
        ]
        df = pd.concat(sampled_groups, ignore_index=True)
    return df.reset_index(drop=True)


def build_generated_eval_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = [args.text_col, args.generated_text_col, args.generator_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available columns: {list(df.columns)}")

    generated = pd.DataFrame(
        {
            "text": df[args.generated_text_col].fillna("").astype(str),
            "label": "AI",
            "generator": df[args.generator_col].fillna("unknown").astype(str),
            "sample_type": "generated",
        }
    )
    generated = generated[generated["text"].str.strip().astype(bool)].copy()

    if not args.include_human:
        return generated

    human = pd.DataFrame(
        {
            "text": df[args.text_col].fillna("").astype(str),
            "label": "human",
            "generator": "human",
            "sample_type": "human",
        }
    )
    human = human[human["text"].str.strip().astype(bool)].copy()
    return pd.concat([generated, human], ignore_index=True)


def filter_by_generators(
    df: pd.DataFrame,
    requested: list[str],
    *,
    generator_col: str,
    keep_human: bool = False,
) -> pd.DataFrame:
    if generator_col not in df.columns:
        raise ValueError(f"Generator column {generator_col!r} was not found. Available columns: {list(df.columns)}")

    available = sorted(df[generator_col].dropna().astype(str).unique().tolist())
    requested_values = expand_generator_aliases(requested)
    normalized_requested = {normalize_generator_name(value) for value in requested_values}
    mask = df[generator_col].astype(str).map(normalize_generator_name).isin(normalized_requested)
    if keep_human:
        mask = mask | (df[generator_col].astype(str).map(normalize_generator_name) == "human")
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(
            "No rows matched requested generators "
            f"{requested!r}. Expanded values: {requested_values!r}. Available generators: {available!r}"
        )
    return filtered


def expand_generator_aliases(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        key = normalize_generator_name(value)
        expanded.extend(GENERATOR_ALIASES.get(key, [value]))
    return expanded


def normalize_generator_name(value: str) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def build_summary(
    *,
    df: pd.DataFrame,
    text_col: str,
    label_col: str | None,
    runs: list[DetectorRun],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_samples": len(df),
        "text_col": text_col,
        "label_col": label_col,
        "detectors": [asdict(run) | {"probability_ai": None, "labels": None} for run in runs],
        "metrics": {},
    }

    for run in runs:
        probs = np.asarray(run.probability_ai, dtype=np.float32)
        run_metrics: dict[str, Any] = {
            "mean_probability_ai": float(probs.mean()) if len(probs) else 0.0,
            "median_probability_ai": float(np.median(probs)) if len(probs) else 0.0,
            "ai_share": float(np.mean([normalize_label(label) == "ai" for label in run.labels])) if run.labels else 0.0,
        }
        if label_col:
            y_true = df[label_col].astype(str).tolist()
            run_metrics.update(classification_metrics(y_true, run.labels))
            run_metrics.update(binary_prf(y_true, run.labels))
        summary["metrics"][run.name] = run_metrics

    if len(runs) == 2:
        own, baseline = runs
        own_probs = np.asarray(own.probability_ai, dtype=np.float32)
        baseline_probs = np.asarray(baseline.probability_ai, dtype=np.float32)
        summary["comparison"] = {
            "mean_probability_delta_own_minus_baseline": float((own_probs - baseline_probs).mean()),
            "same_labels_share": float(np.mean([a == b for a, b in zip(own.labels, baseline.labels)])),
            "own_more_confident_ai_share": float(np.mean(own_probs > baseline_probs)),
        }

    if "generator" in df.columns:
        summary["by_generator"] = build_grouped_metrics(df, runs, group_col="generator", label_col=label_col)
        if label_col:
            summary["binary_by_generator_vs_human"] = build_binary_generator_metrics(
                df,
                runs,
                group_col="generator",
                label_col=label_col,
            )

    return summary


def build_binary_generator_metrics(
    df: pd.DataFrame,
    runs: list[DetectorRun],
    *,
    group_col: str,
    label_col: str,
) -> dict[str, Any]:
    df = df.reset_index(drop=True)
    human_mask = df[label_col].astype(str).map(normalize_label) == "not_ai"
    human_indices = set(df[human_mask].index.tolist())
    grouped: dict[str, Any] = {}

    for group_value, group_df in df[~human_mask].groupby(group_col, dropna=False):
        generated_indices = set(group_df.index.tolist())
        indices = sorted(human_indices | generated_indices)
        subset = df.loc[indices]
        grouped[str(group_value)] = {
            "n_samples": len(indices),
            "n_ai": len(generated_indices),
            "n_human": len(human_indices),
            "metrics": {},
        }
        for run in runs:
            labels = [run.labels[index] for index in indices]
            probs = np.asarray([run.probability_ai[index] for index in indices], dtype=np.float32)
            y_true = subset[label_col].astype(str).tolist()
            metrics: dict[str, Any] = {
                "mean_probability_ai": float(probs.mean()) if len(probs) else 0.0,
                "ai_share": float(np.mean([normalize_label(label) == "ai" for label in labels])) if labels else 0.0,
            }
            metrics.update(classification_metrics(y_true, labels))
            metrics.update(binary_prf(y_true, labels))
            metrics.update(binary_confusion(y_true, labels))
            grouped[str(group_value)]["metrics"][run.name] = metrics

    return grouped


def build_grouped_metrics(
    df: pd.DataFrame,
    runs: list[DetectorRun],
    *,
    group_col: str,
    label_col: str | None,
) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for group_value, group_df in df.reset_index(drop=True).groupby(group_col, dropna=False):
        indices = [int(index) for index in group_df.index.tolist()]
        grouped[str(group_value)] = {"n_samples": len(indices), "metrics": {}}
        for run in runs:
            labels = [run.labels[index] for index in indices]
            probs = np.asarray([run.probability_ai[index] for index in indices], dtype=np.float32)
            metrics: dict[str, Any] = {
                "mean_probability_ai": float(probs.mean()) if len(probs) else 0.0,
                "ai_share": float(np.mean([normalize_label(label) == "ai" for label in labels])) if labels else 0.0,
            }
            if label_col:
                y_true = group_df[label_col].astype(str).tolist()
                metrics.update(classification_metrics(y_true, labels))
                metrics.update(binary_prf(y_true, labels))
            grouped[str(group_value)]["metrics"][run.name] = metrics
    return grouped


def binary_prf(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    actual = [normalize_label(label) for label in y_true]
    predicted = [normalize_label(label) for label in y_pred]
    tp = sum(1 for a, p in zip(actual, predicted) if a == "ai" and p == "ai")
    fp = sum(1 for a, p in zip(actual, predicted) if a != "ai" and p == "ai")
    fn = sum(1 for a, p in zip(actual, predicted) if a == "ai" and p != "ai")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision_ai": precision, "recall_ai": recall, "f1_ai": f1}


def binary_confusion(y_true: list[str], y_pred: list[str]) -> dict[str, int]:
    actual = [normalize_label(label) for label in y_true]
    predicted = [normalize_label(label) for label in y_pred]
    tp = sum(1 for a, p in zip(actual, predicted) if a == "ai" and p == "ai")
    tn = sum(1 for a, p in zip(actual, predicted) if a != "ai" and p != "ai")
    fp = sum(1 for a, p in zip(actual, predicted) if a != "ai" and p == "ai")
    fn = sum(1 for a, p in zip(actual, predicted) if a == "ai" and p != "ai")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def normalize_column_prefix(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the project detector with a pretrained HuggingFace AI-text detector."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col")
    parser.add_argument("--generated-text-col")
    parser.add_argument("--generator-col", default="generator")
    parser.add_argument("--generators", nargs="+")
    parser.add_argument("--include-human", action="store_true")
    parser.add_argument("--list-generators", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-samples-per-generator", type=int)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--output-predictions")
    parser.add_argument("--output-summary", default="data/output_data/detector_comparison_summary.json")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--own-name", default="ours")
    parser.add_argument("--skip-own", action="store_true")
    parser.add_argument("--own-detector-type", choices=["ensemble", "hf"], default="ensemble")
    parser.add_argument("--own-model-path", default="trained_models/ensemble_models_2_3000/manifest.json")
    parser.add_argument("--own-ai-label-index", type=int, default=1)

    parser.add_argument("--baseline-name", default="ruatd_ruroberta")
    parser.add_argument("--baseline-kind", choices=["hf", "gigacheck"], default="hf")
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--baseline-tokenizer", default=DEFAULT_BASELINE_TOKENIZER)
    parser.add_argument("--baseline-ai-label-index", type=int)
    parser.add_argument("--baseline-ai-label")
    parser.add_argument("--baseline-torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.baseline_kind == "gigacheck":
        if args.baseline_model == DEFAULT_BASELINE_MODEL:
            args.baseline_model = DEFAULT_GIGACHECK_MODEL
        if args.baseline_name == "ruatd_ruroberta":
            args.baseline_name = "gigacheck_classifier_multi"
    return args


def resolve_torch_device(device: str | None) -> str:
    if device and device != "auto":
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device=mps was requested, but torch.backends.mps.is_available() is False.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested, but torch.cuda.is_available() is False.")
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(torch_dtype: str, device: str) -> torch.dtype | None:
    if torch_dtype == "float32":
        return torch.float32
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    df = prepare_eval_dataframe(args)
    if args.list_generators:
        if args.generator_col not in df.columns:
            raise ValueError(f"Generator column {args.generator_col!r} was not found. Available columns: {list(df.columns)}")
        print(df[args.generator_col].astype(str).value_counts().to_string())
        return
    if args.text_col not in df.columns:
        raise ValueError(f"Text column {args.text_col!r} was not found. Available columns: {list(df.columns)}")
    if args.label_col and args.label_col not in df.columns:
        raise ValueError(f"Label column {args.label_col!r} was not found. Available columns: {list(df.columns)}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    runs = []

    if not args.skip_own:
        own_run = run_own_detector(args, texts)
        runs.append(own_run)
        save_predictions(df, runs, args.output_predictions)

    baseline_run = run_baseline_detector(args, texts)
    runs.append(baseline_run)
    save_predictions(df, runs, args.output_predictions)

    summary = build_summary(df=df, text_col=args.text_col, label_col=args.label_col, runs=runs)

    save_summary(summary, args.output_summary)

    print(json.dumps(to_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
# ensemble_detector
#
