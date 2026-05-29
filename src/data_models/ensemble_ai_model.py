import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EnsembleAIDetector:
    def __init__(
        self,
        manifest_path: str,
        *,
        threshold: float = 0.5,
        max_length: int = 256,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = float(threshold)
        self.max_length = int(max_length)

        self.manifest_path = Path(manifest_path)

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.members = []
        for m in manifest:
            model_dir = self._resolve_model_dir(m["model_dir"])
            tok = AutoTokenizer.from_pretrained(model_dir)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
            mdl.to(self.device)
            mdl.eval()

            self.members.append({
                "tokenizer": tok,
                "model": mdl,
                "meta": m,
                "model_dir": model_dir,
            })

    def _resolve_model_dir(self, model_dir: str) -> str:
        normalized = Path(str(model_dir).replace("\\", "/"))
        if normalized.is_absolute() and normalized.exists():
            return str(normalized)
        if normalized.exists():
            return str(normalized)

        manifest_parent = self.manifest_path.parent
        if normalized.parts and normalized.parts[0] == manifest_parent.name:
            candidate = manifest_parent.parent / normalized
        else:
            candidate = manifest_parent / normalized

        return str(candidate)

    @torch.no_grad()
    def _member_probs_ai(self, member, texts: List[str], batch_size: int = 64) -> np.ndarray:
        tok = member["tokenizer"]
        mdl = member["model"]

        probs = []
        for i in range(0, len(texts), batch_size):
            bt = texts[i:i + batch_size]
            enc = tok(
                bt,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            logits = mdl(**enc).logits
            p_ai = torch.softmax(logits, dim=-1)[:, 1]
            probs.append(p_ai.detach().cpu().numpy())

        return np.concatenate(probs)

    def predict_batch(self, texts: List[str], *, batch_size: int = 64) -> List[str]:
        texts = [str(t) for t in texts]
        any_ai = np.zeros(len(texts), dtype=bool)

        for member in self.members:
            p_ai = self._member_probs_ai(member, texts, batch_size=batch_size)
            any_ai |= (p_ai >= self.threshold)

        return ["AI" if x else "not_ai" for x in any_ai]

    def predict(self, text: str) -> str:
        return self.predict_batch([text])[0]

    def predict_proba_or(self, texts: List[str], *, batch_size: int = 64) -> np.ndarray:
        texts = [str(t) for t in texts]
        member_matrix = self.predict_member_matrix(texts, batch_size=batch_size)
        return self._predict_proba_or_from_matrix(member_matrix)


    def predict_member_matrix(self, texts: List[str], *, batch_size: int = 64) -> np.ndarray:
        texts = [str(t) for t in texts]
        all_member_probs = []

        for member in self.members:
            p_ai = self._member_probs_ai(member, texts, batch_size=batch_size)
            all_member_probs.append(p_ai)


        return np.vstack(all_member_probs).T


    def predict_generator_group_batch(
        self,
        texts: List[str],
        *,
        batch_size: int = 64,
        threshold: float = None,
    ) -> List[Dict[str, Any]]:
        if threshold is None:
            threshold = self.threshold

        texts = [str(t) for t in texts]
        member_matrix = self.predict_member_matrix(texts, batch_size=batch_size)

        return [
            self._build_generator_group(text, member_matrix[i], threshold=threshold)
            for i, text in enumerate(texts)
        ]

    def predict_generator_group(
        self,
        text: str,
        *,
        threshold: float = None,
    ) -> Dict[str, Any]:
        return self.predict_generator_group_batch([text], threshold=threshold, batch_size=1)[0]

    def predict_detailed_batch(
        self,
        texts: List[str],
        *,
        batch_size: int = 64,
        threshold: float = None,
    ) -> List[Dict[str, Any]]:
        if threshold is None:
            threshold = self.threshold

        texts = [str(t) for t in texts]
        member_matrix = self.predict_member_matrix(texts, batch_size=batch_size)
        probabilities = self._predict_proba_or_from_matrix(member_matrix)

        results = []
        for i, text in enumerate(texts):
            generator_group = self._build_generator_group(text, member_matrix[i], threshold=threshold)
            results.append({
                "text": text,
                "label": "AI" if probabilities[i] >= threshold else "not_ai",
                "probability_ai": float(probabilities[i]),
                "threshold": float(threshold),
                "member_probabilities": member_matrix[i].tolist(),
                "best_member": self._best_member(member_matrix[i]),
                "generator_group": generator_group,
            })

        return results

    def _predict_proba_or_from_matrix(self, member_matrix: np.ndarray) -> np.ndarray:
        if member_matrix.size == 0:
            return np.zeros((member_matrix.shape[0],), dtype=np.float32)
        p_not = np.prod(1.0 - member_matrix.astype(np.float64), axis=1)
        return (1.0 - p_not).astype(np.float32)

    def _best_member(self, probs: np.ndarray) -> Dict[str, Any]:
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        best_member = self.members[best_idx]
        member_meta = best_member["meta"]
        return {
            "index": best_idx,
            "member_id": member_meta.get("member_id"),
            "generators": member_meta.get("generators", []),
            "probability_ai": best_prob,
            "model_dir": best_member.get("model_dir"),
        }

    def _build_generator_group(self, text: str, probs: np.ndarray, *, threshold: float) -> Dict[str, Any]:
        best_member = self._best_member(probs)
        return {
            "text": text,
            "is_ai_by_best_member": int(best_member["probability_ai"] >= threshold),
            "best_member_idx": best_member["index"],
            "best_member_id": best_member["member_id"],
            "best_member_prob_ai": best_member["probability_ai"],
            "predicted_generators": best_member["generators"],
            "all_member_probs": probs.tolist(),
        }
