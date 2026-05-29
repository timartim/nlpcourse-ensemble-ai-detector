from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
CYR_RE = re.compile(r"[А-Яа-яЁё]")
LAT_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
CITATION_BRACKET_RE = re.compile(r"\[[0-9,;\-\s]+\]")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
ET_AL_RE = re.compile(r"et\s+al\.?", re.IGNORECASE)
PARENS_RE = re.compile(r"\([^)]*\)")
MULTISPACE_RE = re.compile(r"\s+")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(str(text).lower())


def _avg(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_text_features(text: str, n_sentences: int | None = None) -> dict:
    text = str(text or "")
    text_clean = MULTISPACE_RE.sub(" ", text).strip()
    words = _tokenize_words(text_clean)
    n_chars = len(text_clean)
    n_words = len(words)
    n_sentences = int(n_sentences) if n_sentences is not None else max(1, text_clean.count(".") + text_clean.count("!") + text_clean.count("?"))

    word_lengths = [len(w) for w in words]
    uniq_words = len(set(words))
    word_counter = Counter(words)
    repeated_words = sum(1 for _, cnt in word_counter.items() if cnt >= 3)

    commas = text_clean.count(",")
    semicolons = text_clean.count(";")
    colons = text_clean.count(":")
    dashes = text_clean.count("-") + text_clean.count("—")
    quotes = text_clean.count('"') + text_clean.count("«") + text_clean.count("»")

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sentences": n_sentences,
        "avg_words_per_sentence": _safe_div(n_words, n_sentences),
        "avg_chars_per_word": _avg(word_lengths),
        "type_token_ratio": _safe_div(uniq_words, n_words),
        "repeated_words_ge3": repeated_words,
        "comma_ratio": _safe_div(commas, n_chars),
        "semicolon_ratio": _safe_div(semicolons, n_chars),
        "colon_ratio": _safe_div(colons, n_chars),
        "dash_ratio": _safe_div(dashes, n_chars),
        "quote_ratio": _safe_div(quotes, n_chars),
        "digit_ratio": _safe_div(len(DIGIT_RE.findall(text_clean)), n_chars),
        "cyr_ratio": _safe_div(len(CYR_RE.findall(text_clean)), max(1, len(CYR_RE.findall(text_clean)) + len(LAT_RE.findall(text_clean)))),
        "citation_brackets": len(CITATION_BRACKET_RE.findall(text_clean)),
        "year_mentions": len(YEAR_RE.findall(text_clean)),
        "et_al_mentions": len(ET_AL_RE.findall(text_clean)),
        "parenthetical_blocks": len(PARENS_RE.findall(text_clean)),
    }


def add_text_features(
    df: pd.DataFrame,
    *,
    text_col: str = "chunk_text",
    sentence_count_col: str = "n_sentences",
) -> pd.DataFrame:
    df_out = df.copy()
    feature_rows = []
    for _, row in df_out.iterrows():
        feature_rows.append(
            compute_text_features(
                row[text_col],
                n_sentences=row[sentence_count_col] if sentence_count_col in row else None,
            )
        )
    features_df = pd.DataFrame(feature_rows)
    return pd.concat([df_out.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
