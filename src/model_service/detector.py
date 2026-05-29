from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from data_models import EnsembleAIModel

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as exc:
    raise ImportError("Install torch and transformers to use BertAIDetector.") from exc


DetectorType = Literal["ensemble", "hf"]


@dataclass(frozen=True)
class DetectorConfig:
    detector_type: DetectorType
    model_path: str
    threshold: float = 0.5
    max_length: int = 256
    batch_size: int = 32
    device: str | None = None
    ai_label_index: int = 1


@dataclass(frozen=True)
class ScoreResult:
    text: str
    label: str
    probability_ai: float
    threshold: float
    detector_type: DetectorType
    member_probabilities: list[float] | None = None
    best_member: dict[str, Any] | None = None
    generator_group: dict[str, Any] | None = None


class BertAIDetector:
    """Loads either one HuggingFace classifier or an ensemble manifest and scores texts."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(config.model_path)
        self._single_model: dict[str, Any] | None = None
        self._ensemble_model: EnsembleAIModel | None = None

        if config.detector_type == "ensemble":
            self._ensemble_model = EnsembleAIModel(
                str(self.model_path),
                threshold=config.threshold,
                max_length=config.max_length,
                device=self.device,
            )
        elif config.detector_type == "hf":
            self._single_model = self._load_hf_model(self.model_path)
        else:
            raise ValueError("detector_type must be 'ensemble' or 'hf'.")

    def score(self, text: str) -> ScoreResult:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[ScoreResult]:
        texts = [str(text) for text in texts]
        if self.config.detector_type == "ensemble":
            return [
                ScoreResult(
                    text=item["text"],
                    label=item["label"],
                    probability_ai=item["probability_ai"],
                    threshold=item["threshold"],
                    detector_type=self.config.detector_type,
                    member_probabilities=item["member_probabilities"],
                    best_member=item["best_member"],
                    generator_group=item["generator_group"],
                )
                for item in self._require_ensemble().predict_detailed_batch(
                    texts,
                    batch_size=self.config.batch_size,
                    threshold=self.config.threshold,
                )
            ]

        probs = self._predict_single(texts)
        return [self._build_result(text=text, probability=float(prob)) for text, prob in zip(texts, probs)]

    def predict(self, text: str) -> str:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str], *, batch_size: int | None = None) -> list[str]:
        texts = [str(text) for text in texts]
        if self.config.detector_type == "ensemble":
            return self._require_ensemble().predict_batch(texts, batch_size=batch_size or self.config.batch_size)

        probs = self._predict_single(texts, batch_size=batch_size)
        return ["AI" if prob >= self.config.threshold else "not_ai" for prob in probs]

    def predict_proba_ai(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        texts = [str(text) for text in texts]
        if self.config.detector_type == "ensemble":
            return self.predict_proba_or(texts, batch_size=batch_size)
        return self._predict_single(texts, batch_size=batch_size)

    def predict_proba_or(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        texts = [str(text) for text in texts]
        if self.config.detector_type != "ensemble":
            return self._predict_single(texts, batch_size=batch_size)
        return self._require_ensemble().predict_proba_or(texts, batch_size=batch_size or self.config.batch_size)

    def predict_member_matrix(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        if self.config.detector_type != "ensemble":
            raise RuntimeError("predict_member_matrix is available only for ensemble detector.")

        texts = [str(text) for text in texts]
        return self._require_ensemble().predict_member_matrix(texts, batch_size=batch_size or self.config.batch_size)

    def predict_generator_group_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.config.detector_type != "ensemble":
            raise RuntimeError("predict_generator_group_batch is available only for ensemble detector.")

        texts = [str(text) for text in texts]
        return self._require_ensemble().predict_generator_group_batch(
            texts,
            batch_size=batch_size or self.config.batch_size,
            threshold=threshold,
        )

    def predict_generator_group(self, text: str, *, threshold: float | None = None) -> dict[str, Any]:
        return self._require_ensemble().predict_generator_group(text, threshold=threshold)

    def _load_hf_model(self, model_dir: Path) -> dict[str, Any]:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.to(self.device)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}

    @torch.no_grad()
    def _predict_single(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        if self._single_model is None:
            raise RuntimeError("Single HF model is not loaded.")
        return self._predict_with_loaded_model(self._single_model, texts, batch_size=batch_size)

    @torch.no_grad()
    def _predict_with_loaded_model(
        self,
        loaded: dict[str, Any],
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        all_probs: list[np.ndarray] = []
        effective_batch_size = batch_size or self.config.batch_size

        for start in range(0, len(texts), effective_batch_size):
            batch = texts[start : start + effective_batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, self.config.ai_label_index]
            all_probs.append(probs.detach().cpu().numpy().astype(np.float32))

        if not all_probs:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(all_probs, axis=0)

    def _build_result(
        self,
        *,
        text: str,
        probability: float,
        member_probs: list[float] | None = None,
        best_member: dict[str, Any] | None = None,
        generator_group: dict[str, Any] | None = None,
    ) -> ScoreResult:
        label = "AI" if probability >= self.config.threshold else "not_ai"
        return ScoreResult(
            text=text,
            label=label,
            probability_ai=probability,
            threshold=self.config.threshold,
            detector_type=self.config.detector_type,
            member_probabilities=member_probs,
            best_member=best_member,
            generator_group=generator_group,
        )

    def _require_ensemble(self) -> EnsembleAIModel:
        if self._ensemble_model is None:
            raise RuntimeError("Ensemble model is not loaded.")
        return self._ensemble_model
