from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    text = str(text).strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents


def make_sentence_chunks(
    sentences: list[str],
    *,
    window_size: int = 4,
    min_sentences: int = 3,
    stride: Optional[int] = None,
) -> list[dict]:
    if stride is None:
        stride = window_size

    chunks = []
    chunk_index = 0

    for start in range(0, len(sentences), stride):
        chunk_sents = sentences[start:start + window_size]
        if len(chunk_sents) < min_sentences:
            continue

        chunk_text = " ".join(chunk_sents).strip()
        if not chunk_text:
            continue

        chunks.append(
            {
                "chunk_index": chunk_index,
                "sentence_start": start,
                "sentence_end": start + len(chunk_sents) - 1,
                "n_sentences": len(chunk_sents),
                "chunk_text": chunk_text,
            }
        )
        chunk_index += 1

    return chunks


def chunk_articles(
    df_articles: pd.DataFrame,
    *,
    text_col: str = "text",
    article_id_col: str = "article_id",
    title_col: Optional[str] = None,
    window_size: int = 4,
    min_sentences: int = 3,
    stride: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []

    iterator = df_articles.iterrows()
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(df_articles),
            desc="Chunking articles",
        )

    for _, row in iterator:
        article_id = row[article_id_col]
        text = str(row[text_col])
        title = row[title_col] if title_col is not None and title_col in row else None

        sentences = split_sentences(text)
        chunks = make_sentence_chunks(
            sentences,
            window_size=window_size,
            min_sentences=min_sentences,
            stride=stride,
        )

        for chunk in chunks:
            out_row = {
                "article_id": article_id,
                "chunk_id": f"{article_id}__chunk_{chunk['chunk_index']:04d}",
                "chunk_index": chunk["chunk_index"],
                "sentence_start": chunk["sentence_start"],
                "sentence_end": chunk["sentence_end"],
                "n_sentences": chunk["n_sentences"],
                "chunk_text": chunk["chunk_text"],
            }
            if title_col is not None:
                out_row["title"] = title
            rows.append(out_row)

    return pd.DataFrame(rows)
