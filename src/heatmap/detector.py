from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as exc:
    raise ImportError(
        "This module requires torch and transformers. Install them first."
    ) from exc


class HFSequenceClassifierDetector:

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: Optional[str] = None,
        max_length: int = 256,
        ai_label_index: int = 1,
    ) -> None:
        self.model_dir = str(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.ai_label_index = int(ai_label_index)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba_ai(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        texts = [str(text) for text in texts]
        all_probs: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, self.ai_label_index]
            all_probs.append(probs.detach().cpu().numpy().astype(np.float32))

        if not all_probs:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(all_probs, axis=0)


class EnsembleAIModel:
    """
    Rewritten version of your ensemble detector.
    It loads each member from manifest.json and combines probabilities via OR-rule:
        P(AI) = 1 - Π(1 - p_i)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        threshold: float = 0.5,
        max_length: int = 256,
        device: Optional[str] = None,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.members: list[dict[str, Any]] = []
        for entry in manifest:
            model_dir = entry["model_dir"]
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.to(self.device)
            model.eval()
            self.members.append(
                {
                    "tokenizer": tokenizer,
                    "model": model,
                    "meta": entry,
                }
            )

    @torch.no_grad()
    def _member_probs_ai(self, member: Dict[str, Any], texts: List[str], batch_size: int = 64) -> np.ndarray:
        tokenizer = member["tokenizer"]
        model = member["model"]
        probs: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            logits = model(**enc).logits
            p_ai = torch.softmax(logits, dim=-1)[:, 1]
            probs.append(p_ai.detach().cpu().numpy().astype(np.float32))

        if not probs:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(probs, axis=0)

    def predict_member_matrix(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        texts = [str(text) for text in texts]
        if not texts:
            return np.zeros((0, len(self.members)), dtype=np.float32)

        columns = [self._member_probs_ai(member, texts, batch_size=batch_size) for member in self.members]
        return np.vstack(columns).T

    def predict_proba_ai(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        texts = [str(text) for text in texts]
        if not texts:
            return np.zeros((0,), dtype=np.float32)

        p_not = np.ones(len(texts), dtype=np.float64)
        for member in self.members:
            p_ai = self._member_probs_ai(member, texts, batch_size=batch_size).astype(np.float64)
            p_not *= (1.0 - p_ai)
        return (1.0 - p_not).astype(np.float32)

    def predict_labels(self, texts: List[str], batch_size: int = 64, threshold: Optional[float] = None) -> List[str]:
        threshold = self.threshold if threshold is None else float(threshold)
        probs = self.predict_proba_ai(texts, batch_size=batch_size)
        return ["AI" if p >= threshold else "not_ai" for p in probs]


def build_detector(
    *,
    detector_type: str,
    model_path: str,
    device: Optional[str] = None,
    max_length: int = 256,
    threshold: float = 0.5,
) -> Any:
    detector_type = detector_type.lower().strip()

    if detector_type == "ensemble":
        return EnsembleAIModel(
            model_path,
            threshold=threshold,
            max_length=max_length,
            device=device,
        )

    if detector_type in {"hf", "huggingface", "sequence_classifier"}:
        return HFSequenceClassifierDetector(
            model_path,
            device=device,
            max_length=max_length,
        )

    raise ValueError(
        f"Unknown detector_type={detector_type!r}. Use 'ensemble' or 'hf'."
    )
