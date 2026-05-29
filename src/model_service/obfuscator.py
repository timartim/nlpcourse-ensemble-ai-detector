from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .detector import BertAIDetector


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])(?=\s+[\"'«(А-ЯA-Z0-9])")


@dataclass(frozen=True)
class ObfuscatorConfig:
    window_size: int = 4
    stride: int = 1
    anchor: str = "center"
    aggregation: str = "max"
    rewrite_threshold: float = 0.8
    sleep_between_calls: float = 0.0
    rewrite_prompt_template: str = (
        "Перепиши фрагмент по-русски для ясности и естественного стиля. "
        "Сохрани смысл и факты, не добавляй новых утверждений. "
        "Не используй канцелярит. Текст должен быть кратким и понятным.\n\n"
        "Фрагмент:\n{fragment}"
    )


@dataclass(frozen=True)
class ObfuscationLogItem:
    sentence_id: int
    score: float
    old: str
    new: str
    detector_scores: dict[str, float]


@dataclass(frozen=True)
class ObfuscationResult:
    original_text: str
    obfuscated_text: str
    changed: bool
    rewrites: list[ObfuscationLogItem]
    sentence_scores: list[float]
    threshold: float


@dataclass(frozen=True)
class Window:
    sentence_ids: list[int]
    text: str
    anchor_id: int


class ProbabilityDetector(Protocol):
    name: str

    def predict_proba_ai(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        ...


class RewriteClient:
    """Base rewrite client with optional JSON cache."""

    def __init__(self, *, cache_path: str | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def rewrite(self, text: str, *, prompt: str) -> str:
        key = self._key(text)
        if key in self.cache:
            return self.cache[key]

        rewritten = self._rewrite_uncached(text, prompt=prompt)
        self.cache[key] = rewritten
        self._save_cache()
        return rewritten

    def _rewrite_uncached(self, text: str, *, prompt: str) -> str:
        raise NotImplementedError

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")


class SimpleRewriteClient(RewriteClient):

    _REPLACEMENTS: tuple[tuple[str, str], ...] = (
        ("Данная", "Эта"),
        ("данная", "эта"),
        ("Данный", "Этот"),
        ("данный", "этот"),
        ("Данное", "Это"),
        ("данное", "это"),
        ("является", "служит"),
        ("представляет собой", "это"),
        ("осуществляется", "происходит"),
        ("посредством", "с помощью"),
        ("в целях", "чтобы"),
        ("необходимо отметить, что", "важно, что"),
        ("следует отметить, что", "важно, что"),
        ("таким образом", "поэтому"),
        ("Кроме того", "Также"),
        ("позволяет осуществлять", "помогает"),
        ("имеет возможность", "может"),
    )

    def _rewrite_uncached(self, text: str, *, prompt: str) -> str:
        rewritten = text.strip()
        for old, new in self._REPLACEMENTS:
            rewritten = re.sub(rf"\b{re.escape(old)}\b", new, rewritten)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        return rewritten or text


class BertAIObfuscator:
    """Scores text by sentence windows and rewrites sentences that look AI-generated."""

    def __init__(
        self,
        detector: ProbabilityDetector | BertAIDetector | list[ProbabilityDetector | BertAIDetector],
        config: ObfuscatorConfig | None = None,
        rewrite_client: RewriteClient | None = None,
    ) -> None:
        self.detectors = detector if isinstance(detector, list) else [detector]
        if not self.detectors:
            raise ValueError("At least one detector is required.")
        self.config = config or ObfuscatorConfig()
        self.rewrite_client = rewrite_client or SimpleRewriteClient()

    def obfuscate(self, text: str) -> ObfuscationResult:
        return self.obfuscate_batch([text])[0]

    def obfuscate_batch(self, texts: list[str]) -> list[ObfuscationResult]:
        return [self._obfuscate_one(str(text)) for text in texts]

    def process_dataframe(
        self,
        df: pd.DataFrame,
        *,
        text_col: str,
        out_col: str = "text_rewritten",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_out = df.copy()
        logs: list[dict[str, Any]] = []
        rewritten_texts: list[str] = []

        for row_idx, row in df_out.iterrows():
            result = self.obfuscate(str(row.get(text_col, "") or ""))
            rewritten_texts.append(result.obfuscated_text)
            for item in result.rewrites:
                logs.append(
                    {
                        "row_idx": int(row_idx),
                        "sent_id": item.sentence_id,
                        "score_combined": item.score,
                        "old": item.old,
                        "new": item.new,
                        "detector_scores": item.detector_scores,
                    }
                )

        df_out[out_col] = rewritten_texts
        return df_out, pd.DataFrame(logs)

    def _obfuscate_one(self, text: str) -> ObfuscationResult:
        if not text.strip():
            return self._empty_result(text)

        sentence_spans = split_sentences_with_offsets(text)
        sentences = [sentence for sentence, _, _ in sentence_spans]
        if not sentences:
            return self._empty_result(text)

        windows = make_windows(
            sentences,
            window_size=self.config.window_size,
            stride=self.config.stride,
            anchor=self.config.anchor,
        )
        if not windows:
            return self._empty_result(text)

        sent_scores_by_detector = self._score_sentences(windows, len(sentences))
        combined_scores = np.zeros(len(sentences), dtype=np.float32)
        for scores in sent_scores_by_detector.values():
            combined_scores = np.maximum(combined_scores, scores)

        new_sentences = list(sentences)
        rewrites: list[ObfuscationLogItem] = []
        for sentence_id, score in enumerate(combined_scores.tolist()):
            if score < self.config.rewrite_threshold:
                continue

            old = sentences[sentence_id].strip()
            if not old:
                continue

            prompt = self.config.rewrite_prompt_template.format(fragment=old)
            new = self.rewrite_client.rewrite(old, prompt=prompt).strip()
            if new and new != old:
                new_sentences[sentence_id] = new
                rewrites.append(
                    ObfuscationLogItem(
                        sentence_id=sentence_id,
                        score=float(score),
                        old=old,
                        new=new,
                        detector_scores={
                            name: float(scores[sentence_id])
                            for name, scores in sent_scores_by_detector.items()
                        },
                    )
                )

            if self.config.sleep_between_calls > 0:
                time.sleep(self.config.sleep_between_calls)

        obfuscated_text = " ".join(sentence.strip() for sentence in new_sentences).strip()
        return ObfuscationResult(
            original_text=text,
            obfuscated_text=obfuscated_text,
            changed=obfuscated_text != text,
            rewrites=rewrites,
            sentence_scores=[float(score) for score in combined_scores.tolist()],
            threshold=self.config.rewrite_threshold,
        )

    def _score_sentences(self, windows: list[Window], n_sentences: int) -> dict[str, np.ndarray]:
        window_texts = [window.text for window in windows]
        scores_by_detector: dict[str, np.ndarray] = {}
        for index, detector in enumerate(self.detectors):
            detector_name = getattr(detector, "name", detector.__class__.__name__) or f"detector_{index}"
            p_ai_windows = detector.predict_proba_ai(window_texts, batch_size=None)
            if len(p_ai_windows) != len(windows):
                raise RuntimeError(f"Detector {detector_name}: wrong output size.")
            scores_by_detector[detector_name] = score_sentences_from_windows(
                windows,
                p_ai_windows,
                n_sentences,
                aggregation=self.config.aggregation,
            )
        return scores_by_detector

    def _empty_result(self, text: str) -> ObfuscationResult:
        return ObfuscationResult(
            original_text=text,
            obfuscated_text=text,
            changed=False,
            rewrites=[],
            sentence_scores=[],
            threshold=self.config.rewrite_threshold,
        )


def split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    if not text:
        return []

    spans: list[tuple[str, int, int]] = []
    parts = _SENT_SPLIT_RE.split(text)
    cursor = 0
    for part in parts:
        if not part:
            continue
        start = text.find(part, cursor)
        if start < 0:
            continue
        end = start + len(part)
        cursor = end
        spans.append((text[start:end], start, end))
    return spans


def make_windows(
    sentences: list[str],
    *,
    window_size: int = 4,
    stride: int = 1,
    anchor: str = "center",
) -> list[Window]:
    if not sentences:
        return []

    windows: list[Window] = []
    if anchor not in {"first", "center", "last"}:
        raise ValueError("anchor must be 'first', 'center' or 'last'.")

    for anchor_id in range(0, len(sentences), stride):
        if anchor == "first":
            start = anchor_id
        elif anchor == "last":
            start = anchor_id - window_size + 1
        else:
            start = anchor_id - window_size // 2

        start = max(0, min(start, max(len(sentences) - window_size, 0)))
        end = min(start + window_size, len(sentences))
        sentence_ids = list(range(start, end))

        window_text = " ".join(sentences[index].strip() for index in sentence_ids).strip()
        if window_text:
            windows.append(Window(sentence_ids=sentence_ids, text=window_text, anchor_id=anchor_id))
    return windows


def score_sentences_from_windows(
    windows: list[Window],
    p_ai_windows: np.ndarray,
    n_sentences: int,
    *,
    aggregation: str = "max",
) -> np.ndarray:
    sent_scores: list[list[float]] = [[] for _ in range(n_sentences)]
    for window, probability in zip(windows, p_ai_windows.tolist()):
        sent_scores[window.anchor_id].append(float(probability))

    output = np.zeros(n_sentences, dtype=np.float32)
    for index, scores in enumerate(sent_scores):
        if not scores:
            output[index] = 0.0
        elif aggregation == "mean":
            output[index] = float(np.mean(scores))
        elif aggregation == "max":
            output[index] = float(np.max(scores))
        else:
            raise ValueError("aggregation must be 'max' or 'mean'.")
    return output
