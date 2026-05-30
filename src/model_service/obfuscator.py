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
from tqdm.auto import tqdm

from .detector import BertAIDetector


EDIT_OPEN = "<EDIT>"
EDIT_CLOSE = "</EDIT>"
_TAG_BLOCK_RE = re.compile(r"<EDIT>(.*?)</EDIT>", re.DOTALL)
_SENT_SEG_RE = re.compile(r".*?(?:[.!?…]+(?:\s+|$)|$)", re.DOTALL)

SYSTEM_TAG_EDITOR = (
    "Ты — опытный редактор русских текстов. "
    "Тебе пришёл фрагмент, где часть помечена тегами <EDIT>...</EDIT>. "
    "Отредактируй ТОЛЬКО текст внутри <EDIT>...</EDIT>, "
    "не добавляй фактов, сохрани смысл. "
    "Текст ВНЕ тегов НЕ МЕНЯЙ (символ в символ). "
    "Верни весь фрагмент целиком, СОХРАНИВ теги <EDIT>...</EDIT>. "
    "Текст внутри тегов должен быть ПЕРЕФРАЗИРОВАН (не допускается идентичный вариант)."
)

USER_TAG_REWRITE_TEMPLATE = """Отредактируй ТОЛЬКО то, что внутри <EDIT>...</EDIT>.
Требования к правке внутри тегов:
- полностью перефразируй (другие слова и конструкции), сохрани смысл и стиль
- не добавляй новых фактов
- избегай дословного совпадения с исходником

ОГРАНИЧЕНИЯ:
- текст ВНЕ тегов <EDIT>...</EDIT> НЕ МЕНЯЙ (символ в символ)
- верни ВЕСЬ фрагмент целиком
- теги <EDIT> и </EDIT> ОБЯЗАТЕЛЬНО должны остаться в ответе

ФРАГМЕНТ:
{fragment}
"""


@dataclass(frozen=True)
class ObfuscatorConfig:
    sent_threshold: float = 0.8
    sent_max_retries: int = 3
    neighbors: int = 1
    detector_batch_size: int = 128
    rewrite_sleep: float = 0.0
    show_progress: bool = True

    @property
    def rewrite_threshold(self) -> float:
        return self.sent_threshold


@dataclass(frozen=True)
class ObfuscationLogItem:
    sentence_id: int
    score: float
    old: str
    new: str
    p_ai_after: float
    retries: int
    history: list[dict[str, Any]]


@dataclass(frozen=True)
class ObfuscationResult:
    original_text: str
    obfuscated_text: str
    changed: bool
    rewrites: list[ObfuscationLogItem]
    sentence_scores: list[float]
    threshold: float
    flagged_sentence_ids: list[int]


class ProbabilityDetector(Protocol):
    def predict_proba_ai(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        ...


class RewriteClient:
    """Tagged-fragment rewrite client with optional JSON cache."""

    def __init__(self, *, cache_path: str | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def rewrite_tagged_fragment(self, fragment_with_tags: str) -> str:
        key = self._key(fragment_with_tags)
        if key in self.cache:
            return self.cache[key]

        rewritten = self._rewrite_tagged_fragment_uncached(fragment_with_tags)
        self.cache[key] = rewritten
        self._save_cache()
        return rewritten

    def rewrite(self, text: str, *, prompt: str | None = None) -> str:
        tagged = f"{EDIT_OPEN}{text}{EDIT_CLOSE}"
        edited = extract_edited_from_tagged(self.rewrite_tagged_fragment(tagged))
        return edited or text

    def _rewrite_tagged_fragment_uncached(self, fragment_with_tags: str) -> str:
        raise NotImplementedError

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")


class SimpleRewriteClient(RewriteClient):
    """Local tagged rewriter for smoke tests when an LLM client is not provided."""

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

    def _rewrite_tagged_fragment_uncached(self, fragment_with_tags: str) -> str:
        edited = extract_edited_from_tagged(fragment_with_tags)
        if edited is None:
            return fragment_with_tags

        rewritten = clean_text(edited)
        for old, new in self._REPLACEMENTS:
            rewritten = re.sub(rf"\b{re.escape(old)}\b", new, rewritten)
        rewritten = re.sub(r"\s+", " ", rewritten).strip() or edited
        return replace_edited_in_tagged(fragment_with_tags, rewritten)


class OpenAICompatibleRewriteClient(RewriteClient):
    """Rewrite client that uses the notebook prompts with OpenAI-compatible APIs."""

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        deepseek_api_key: str | None = None,
        openai_model: str = "gpt-5-mini",
        deepseek_model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_output_tokens: int = 2000,
        cache_path: str | None = None,
    ) -> None:
        super().__init__(cache_path=cache_path)
        self.openai_api_key = openai_api_key
        self.deepseek_api_key = deepseek_api_key
        self.openai_model = openai_model
        self.deepseek_model = deepseek_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _rewrite_tagged_fragment_uncached(self, fragment_with_tags: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenAICompatibleRewriteClient.") from exc

        fragment = clean_text(fragment_with_tags)
        if not fragment:
            return ""

        if self.openai_api_key:
            client = OpenAI(api_key=self.openai_api_key)
            response = client.responses.create(
                model=self.openai_model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_TAG_EDITOR}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": USER_TAG_REWRITE_TEMPLATE.format(fragment=fragment),
                            }
                        ],
                    },
                ],
                max_output_tokens=self.max_output_tokens,
            )
            output = clean_text(extract_text_from_responses_api(response))
            if output and EDIT_OPEN in output and EDIT_CLOSE in output:
                return output

        if self.deepseek_api_key:
            client = OpenAI(api_key=self.deepseek_api_key, base_url="https://api.deepseek.com/v1")
            response = client.chat.completions.create(
                model=self.deepseek_model,
                messages=[
                    {"role": "system", "content": SYSTEM_TAG_EDITOR},
                    {"role": "user", "content": USER_TAG_REWRITE_TEMPLATE.format(fragment=fragment)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            output = clean_text(response.choices[0].message.content or "")
            if output and EDIT_OPEN in output and EDIT_CLOSE in output:
                return output

        return ""


class ModelAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_proba_ai_batched(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        desc: str = "detector inference",
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = ["" if text is None else str(text) for text in texts]
        out: list[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        for start in tqdm(iterator, desc=desc, dynamic_ncols=True, disable=not show_progress):
            batch = texts[start : start + batch_size]
            out.append(self._call_model(batch, batch_size=min(batch_size, len(batch))))
        return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)

    def _call_model(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        if hasattr(self.model, "predict_proba_ai") and callable(getattr(self.model, "predict_proba_ai")):
            try:
                return np.asarray(self.model.predict_proba_ai(texts, batch_size=batch_size), dtype=np.float32)
            except TypeError:
                return np.asarray(self.model.predict_proba_ai(texts), dtype=np.float32)
        if hasattr(self.model, "predict_proba_or") and callable(getattr(self.model, "predict_proba_or")):
            return np.asarray(self.model.predict_proba_or(texts, batch_size=batch_size), dtype=np.float32)
        raise TypeError("model must implement predict_proba_ai(...) or predict_proba_or(...)")


class BertAIObfuscator:
    """Detector-guided iterative sentence obfuscator from the train_bert notebook pipeline."""

    def __init__(
        self,
        detector: ProbabilityDetector | BertAIDetector,
        config: ObfuscatorConfig | None = None,
        rewrite_client: RewriteClient | None = None,
    ) -> None:
        self.detector = detector
        self.adapter = ModelAdapter(detector)
        self.config = config or ObfuscatorConfig()
        self.rewrite_client = rewrite_client or SimpleRewriteClient()

    def obfuscate(self, text: str) -> ObfuscationResult:
        return self._obfuscate_one(str(text))

    def obfuscate_batch(self, texts: list[str]) -> list[ObfuscationResult]:
        return [
            self._obfuscate_one(str(text), row_index=index)
            for index, text in enumerate(
                tqdm(texts, desc="rewrite-by-sent-detector", disable=not self.config.show_progress)
            )
        ]

    def process_dataframe(
        self,
        df: pd.DataFrame,
        *,
        text_col: str,
        out_col: str = "text_rewritten",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if text_col not in df.columns:
            raise ValueError(f"Missing text_col={text_col!r}.")

        df_out = df.copy()
        rewritten_texts: list[str] = []
        logs: list[dict[str, Any]] = []

        iterator = df_out.iterrows()
        for row_idx, row in tqdm(iterator, total=len(df_out), desc="rewrite dataframe", disable=not self.config.show_progress):
            result = self._obfuscate_one(str(row.get(text_col, "") or ""), row_index=int(row_idx))
            rewritten_texts.append(result.obfuscated_text)
            for item in result.rewrites:
                logs.append(
                    {
                        "row_idx": int(row_idx),
                        "sent_id": item.sentence_id,
                        "old": item.old,
                        "new": item.new,
                        "p_ai_before_sent": item.score,
                        "p_ai_after_sent": item.p_ai_after,
                        "retries": item.retries,
                        "history": json.dumps(item.history, ensure_ascii=False),
                    }
                )

        df_out[out_col] = rewritten_texts
        return df_out, pd.DataFrame(logs)

    def _obfuscate_one(self, text: str, *, row_index: int | None = None) -> ObfuscationResult:
        parts = split_sentences_keep_ws(text)
        if not parts:
            return self._empty_result(text)

        cores = [core for core, _ in parts]
        p_sent = score_sentences_p_ai(
            self.adapter,
            cores,
            batch_size=self.config.detector_batch_size,
            desc="sent detector (select)" if row_index is None else f"sent detector row={row_index}",
            show_progress=False,
        )
        flagged = [i for i, probability in enumerate(p_sent.tolist()) if probability >= self.config.sent_threshold]
        clean_cores = list(cores)
        rewrites: list[ObfuscationLogItem] = []

        for sent_id in flagged:
            result = iterative_rewrite_core_until_ok(
                adapter=self.adapter,
                rewrite_client=self.rewrite_client,
                cores=clean_cores,
                sent_id=sent_id,
                neighbors=self.config.neighbors,
                sent_threshold=self.config.sent_threshold,
                sent_max_retries=self.config.sent_max_retries,
                detector_batch_size=self.config.detector_batch_size,
                show_progress=False,
            )
            if result["changed"]:
                clean_cores[sent_id] = result["new"]
                rewrites.append(
                    ObfuscationLogItem(
                        sentence_id=int(sent_id),
                        score=float(result["p_ai_before"]),
                        old=result["old"],
                        new=result["new"],
                        p_ai_after=float(result["p_ai_after"]),
                        retries=int(result["retries"]),
                        history=result["history"],
                    )
                )

            if self.config.rewrite_sleep > 0:
                time.sleep(self.config.rewrite_sleep)

        clean_parts = [(clean_cores[i], whitespace) for i, (_core, whitespace) in enumerate(parts)]
        obfuscated = join_sentences_keep_ws(clean_parts)
        return ObfuscationResult(
            original_text=text,
            obfuscated_text=obfuscated,
            changed=obfuscated.strip() != text.strip(),
            rewrites=rewrites,
            sentence_scores=[float(value) for value in p_sent.tolist()],
            threshold=float(self.config.sent_threshold),
            flagged_sentence_ids=flagged,
        )

    def _empty_result(self, text: str) -> ObfuscationResult:
        return ObfuscationResult(
            original_text=text,
            obfuscated_text=text,
            changed=False,
            rewrites=[],
            sentence_scores=[],
            threshold=float(self.config.sent_threshold),
            flagged_sentence_ids=[],
        )


def iterative_rewrite_core_until_ok(
    *,
    adapter: ModelAdapter,
    rewrite_client: RewriteClient,
    cores: list[str],
    sent_id: int,
    neighbors: int,
    sent_threshold: float,
    sent_max_retries: int,
    detector_batch_size: int,
    show_progress: bool,
) -> dict[str, Any]:
    old = cores[sent_id]
    p0 = float(
        score_sentences_p_ai(
            adapter,
            [old],
            batch_size=detector_batch_size,
            desc="sent detector (single)",
            show_progress=show_progress,
        )[0]
    )
    history: list[dict[str, Any]] = [{"try": 0, "p_ai": p0, "text": old}]
    best_text = old
    best_p = p0

    if p0 < sent_threshold:
        return _rewrite_result(False, sent_id, old, old, p0, p0, 0, history)

    cur = old
    attempts = 0
    for attempt in range(1, int(sent_max_retries) + 1):
        attempts += 1
        tmp = list(cores)
        tmp[sent_id] = cur
        block = build_context_block(tmp, sent_id, neighbors=neighbors)

        out_block = rewrite_client.rewrite_tagged_fragment(block)
        edited = clean_text(extract_edited_from_tagged(out_block) or "")
        if not edited:
            history.append({"try": attempt, "p_ai": None, "text": cur, "note": "empty_or_broken_llm_output"})
            continue
        if edited.strip() == cur.strip():
            history.append({"try": attempt, "p_ai": None, "text": cur, "note": "no_change"})
            continue

        p_new = float(
            score_sentences_p_ai(
                adapter,
                [edited],
                batch_size=detector_batch_size,
                desc="sent detector (single)",
                show_progress=show_progress,
            )[0]
        )
        history.append({"try": attempt, "p_ai": p_new, "text": edited})
        if p_new < best_p:
            best_p = p_new
            best_text = edited

        cur = edited
        if p_new < sent_threshold:
            break

    return _rewrite_result(best_text.strip() != old.strip(), sent_id, old, best_text, p0, best_p, attempts, history)


def _rewrite_result(
    changed: bool,
    sent_id: int,
    old: str,
    new: str,
    p_ai_before: float,
    p_ai_after: float,
    retries: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "changed": bool(changed),
        "sent_id": int(sent_id),
        "old": old,
        "new": new,
        "p_ai_before": float(p_ai_before),
        "p_ai_after": float(p_ai_after),
        "retries": int(retries),
        "history": history,
    }


def score_sentences_p_ai(
    adapter: ModelAdapter,
    sentences: list[str],
    *,
    batch_size: int = 128,
    desc: str = "sent detector",
    show_progress: bool = True,
) -> np.ndarray:
    if not sentences:
        return np.zeros((0,), dtype=np.float32)
    return adapter.predict_proba_ai_batched(sentences, batch_size=batch_size, desc=desc, show_progress=show_progress)


def split_sentences_keep_ws(text: str) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for match in _SENT_SEG_RE.finditer(str(text)):
        segment = match.group(0)
        if not segment or not segment.strip():
            continue
        core = segment.rstrip()
        trailing_ws = segment[len(core) :]
        output.append((core, trailing_ws))
    return output


def join_sentences_keep_ws(parts: list[tuple[str, str]]) -> str:
    return "".join(core + whitespace for core, whitespace in parts)


def build_context_block(cores: list[str], sent_id: int, *, neighbors: int = 1) -> str:
    left = max(0, sent_id - neighbors)
    right = min(len(cores), sent_id + neighbors + 1)
    chunk: list[str] = []
    for index in range(left, right):
        if index == sent_id:
            chunk.append(f"{EDIT_OPEN}{cores[index]}{EDIT_CLOSE}")
        else:
            chunk.append(cores[index])
    return " ".join(item.strip() for item in chunk if item and item.strip()).strip()


def extract_edited_from_tagged(fragment_with_tags: str) -> str | None:
    if not fragment_with_tags:
        return None
    match = _TAG_BLOCK_RE.search(fragment_with_tags)
    if not match:
        return None
    inner = clean_text(match.group(1) or "")
    return inner or None


def replace_edited_in_tagged(fragment_with_tags: str, new_text: str) -> str:
    return _TAG_BLOCK_RE.sub(f"{EDIT_OPEN}{new_text}{EDIT_CLOSE}", fragment_with_tags, count=1)


def extract_text_from_responses_api(response: Any) -> str:
    output = getattr(response, "output_text", None)
    if output:
        return str(output).strip()

    text = ""
    try:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        text += getattr(content, "text", "") or ""
    except Exception:
        pass
    return text.strip()


def clean_text(text: str) -> str:
    return (text or "").replace("\x00", "").strip()
