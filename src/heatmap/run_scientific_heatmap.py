from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".parquet", ".json", ".jsonl", ".txt"}


def load_articles(input_path: str, *, text_col: str, article_id_col: str) -> pd.DataFrame:
    path = Path(input_path)

    if path.is_dir():
        rows = []
        for file_path in sorted(path.glob("*.txt")):
            rows.append(
                {
                    article_id_col: file_path.stem,
                    text_col: file_path.read_text(encoding="utf-8", errors="ignore"),
                }
            )
        return pd.DataFrame(rows)

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}. Use csv/parquet/json/jsonl/txt or a folder with .txt files.")

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
    elif suffix == ".txt":
        df = pd.DataFrame(
            [
                {
                    article_id_col: path.stem,
                    text_col: path.read_text(encoding="utf-8", errors="ignore"),
                }
            ]
        )
    else:
        raise AssertionError("Unreachable branch")

    if article_id_col not in df.columns:
        df = df.copy()
        df[article_id_col] = [f"article_{i:04d}" for i in range(len(df))]

    if text_col not in df.columns:
        raise ValueError(f"Column {text_col!r} not found in input data.")

    return df
