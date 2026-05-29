from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .chunking import chunk_articles
from .features import add_text_features
from tqdm.auto import tqdm

FEATURE_COLUMNS = [
    "n_chars",
    "n_words",
    "n_sentences",
    "avg_words_per_sentence",
    "avg_chars_per_word",
    "type_token_ratio",
    "repeated_words_ge3",
    "comma_ratio",
    "semicolon_ratio",
    "colon_ratio",
    "dash_ratio",
    "quote_ratio",
    "digit_ratio",
    "cyr_ratio",
    "citation_brackets",
    "year_mentions",
    "et_al_mentions",
    "parenthetical_blocks",
]


def score_chunks(
    detector: Any,
    df_chunks: pd.DataFrame,
    *,
    text_col: str = "chunk_text",
    batch_size: int = 64,
    threshold: float = 0.8,
    show_progress: bool = True,
    progress_desc: str = "Scoring chunks",
) -> pd.DataFrame:
    df_out = df_chunks.copy()
    texts = df_out[text_col].astype(str).tolist()

    probs_all = []
    iterator = range(0, len(texts), batch_size)

    if show_progress:
        iterator = tqdm(
            iterator,
            total=(len(texts) + batch_size - 1) // batch_size,
            desc=progress_desc,
        )

    for start in iterator:
        batch_texts = texts[start:start + batch_size]
        batch_probs = detector.predict_proba_ai(batch_texts, batch_size=len(batch_texts))
        probs_all.append(np.asarray(batch_probs))

    probs = np.concatenate(probs_all) if probs_all else np.array([], dtype=float)

    df_out["p_ai"] = probs
    df_out["is_ai_like"] = (df_out["p_ai"] >= float(threshold)).astype(int)
    return df_out


def summarize_articles(df_scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_scored.groupby("article_id", dropna=False)
        .agg(
            n_chunks=("chunk_id", "count"),
            mean_p_ai=("p_ai", "mean"),
            median_p_ai=("p_ai", "median"),
            max_p_ai=("p_ai", "max"),
            share_ai_like=("is_ai_like", "mean"),
            mean_words=("n_words", "mean"),
            mean_words_per_sentence=("avg_words_per_sentence", "mean"),
        )
        .sort_values(["mean_p_ai", "max_p_ai"], ascending=False)
        .reset_index()
    )
    return summary


def build_heatmap_matrix(df_scored: pd.DataFrame, *, max_articles: Optional[int] = 50) -> pd.DataFrame:
    article_order = (
        df_scored.groupby("article_id")["p_ai"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    if max_articles is not None:
        article_order = article_order[: int(max_articles)]

    matrix = (
        df_scored[df_scored["article_id"].isin(article_order)]
        .pivot(index="article_id", columns="chunk_index", values="p_ai")
        .reindex(article_order)
    )
    return matrix


def save_heatmap(matrix: pd.DataFrame, out_path: str | Path) -> None:
    out_path = str(out_path)
    if matrix.empty:
        raise ValueError("Heatmap matrix is empty. Nothing to plot.")

    fig_height = max(4, 0.45 * len(matrix))
    fig_width = max(8, 0.6 * len(matrix.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(matrix.values, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title("AI-likeness heatmap for Russian scientific article chunks")
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Article ID")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.tolist())
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(v) for v in matrix.index.tolist()])

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("P(AI)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compare_feature_groups(
    df_scored: pd.DataFrame,
    *,
    high_quantile: float = 0.90,
    low_quantile: float = 0.10,
) -> pd.DataFrame:
    if df_scored.empty:
        return pd.DataFrame()

    q_high = float(df_scored["p_ai"].quantile(high_quantile))
    q_low = float(df_scored["p_ai"].quantile(low_quantile))

    high = df_scored[df_scored["p_ai"] >= q_high].copy()
    low = df_scored[df_scored["p_ai"] <= q_low].copy()

    rows = []
    for col in FEATURE_COLUMNS:
        high_mean = float(high[col].mean()) if len(high) else np.nan
        low_mean = float(low[col].mean()) if len(low) else np.nan
        delta = high_mean - low_mean if pd.notna(high_mean) and pd.notna(low_mean) else np.nan
        ratio = (high_mean / low_mean) if low_mean not in {0, np.nan} and pd.notna(low_mean) and low_mean != 0 else np.nan
        rows.append(
            {
                "feature": col,
                "high_ai_mean": high_mean,
                "low_ai_mean": low_mean,
                "delta_high_minus_low": delta,
                "ratio_high_over_low": ratio,
            }
        )

    out = pd.DataFrame(rows).sort_values("delta_high_minus_low", ascending=False)
    return out.reset_index(drop=True)


def compare_top_tokens(
    df_scored: pd.DataFrame,
    *,
    text_col: str = "chunk_text",
    high_quantile: float = 0.90,
    low_quantile: float = 0.10,
    min_token_len: int = 3,
    top_k: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import re
    from collections import Counter

    token_re = re.compile(r"[A-Za-zА-Яа-яЁё-]+")

    q_high = float(df_scored["p_ai"].quantile(high_quantile))
    q_low = float(df_scored["p_ai"].quantile(low_quantile))

    high = df_scored[df_scored["p_ai"] >= q_high][text_col].astype(str).tolist()
    low = df_scored[df_scored["p_ai"] <= q_low][text_col].astype(str).tolist()

    def collect(texts: list[str]) -> Counter:
        cnt = Counter()
        for text in texts:
            for token in token_re.findall(text.lower()):
                token = token.strip("-")
                if len(token) >= min_token_len:
                    cnt[token] += 1
        return cnt

    high_cnt = collect(high)
    low_cnt = collect(low)

    high_total = sum(high_cnt.values()) or 1
    low_total = sum(low_cnt.values()) or 1
    vocab = set(high_cnt) | set(low_cnt)

    rows = []
    for token in vocab:
        high_freq = high_cnt[token] / high_total
        low_freq = low_cnt[token] / low_total
        rows.append(
            {
                "token": token,
                "high_ai_count": int(high_cnt[token]),
                "low_ai_count": int(low_cnt[token]),
                "high_ai_freq": float(high_freq),
                "low_ai_freq": float(low_freq),
                "freq_delta": float(high_freq - low_freq),
            }
        )

    df_tokens = pd.DataFrame(rows)
    high_tokens = df_tokens.sort_values("freq_delta", ascending=False).head(top_k).reset_index(drop=True)
    low_tokens = df_tokens.sort_values("freq_delta", ascending=True).head(top_k).reset_index(drop=True)
    return high_tokens, low_tokens


def build_examples_report(
    df_scored: pd.DataFrame,
    *,
    out_path: str | Path,
    top_k: int = 8,
) -> None:
    out_path = str(out_path)
    high = df_scored.sort_values("p_ai", ascending=False).head(top_k)
    low = df_scored.sort_values("p_ai", ascending=True).head(top_k)

    lines: list[str] = []
    lines.append("# Examples from detector analysis\n")

    lines.append("## Most AI-like chunks\n")
    for _, row in high.iterrows():
        lines.append(f"### {row['chunk_id']}")
        lines.append(f"- article_id: {row['article_id']}")
        lines.append(f"- p_ai: {row['p_ai']:.4f}")
        lines.append(f"- n_sentences: {int(row['n_sentences'])}")
        lines.append(f"- avg_words_per_sentence: {row['avg_words_per_sentence']:.2f}")
        lines.append(f"- type_token_ratio: {row['type_token_ratio']:.4f}")
        lines.append(f"- citation_brackets: {int(row['citation_brackets'])}")
        lines.append("")
        lines.append(str(row["chunk_text"]))
        lines.append("")

    lines.append("## Least AI-like chunks\n")
    for _, row in low.iterrows():
        lines.append(f"### {row['chunk_id']}")
        lines.append(f"- article_id: {row['article_id']}")
        lines.append(f"- p_ai: {row['p_ai']:.4f}")
        lines.append(f"- n_sentences: {int(row['n_sentences'])}")
        lines.append(f"- avg_words_per_sentence: {row['avg_words_per_sentence']:.2f}")
        lines.append(f"- type_token_ratio: {row['type_token_ratio']:.4f}")
        lines.append(f"- citation_brackets: {int(row['citation_brackets'])}")
        lines.append("")
        lines.append(str(row["chunk_text"]))
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_heatmap_analysis(
    *,
    detector: Any,
    df_articles: pd.DataFrame,
    out_dir: str | Path,
    text_col: str = "text",
    article_id_col: str = "article_id",
    title_col: Optional[str] = None,
    batch_size: int = 64,
    window_size: int = 4,
    min_sentences: int = 3,
    stride: Optional[int] = None,
    threshold: float = 0.8,
    max_articles_in_heatmap: int = 50,
    show_progress: bool = True,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_chunks = chunk_articles(
        df_articles,
        text_col=text_col,
        article_id_col=article_id_col,
        title_col=title_col,
        window_size=window_size,
        min_sentences=min_sentences,
        stride=stride,
        show_progress=show_progress,
    )
    if df_chunks.empty:
        raise ValueError("No chunks were created. Check your texts and chunking parameters.")

    df_scored = score_chunks(
        detector,
        df_chunks,
        text_col="chunk_text",
        batch_size=batch_size,
        threshold=threshold,
        show_progress=show_progress,
        progress_desc="Detector inference on chunks",
    )

    df_scored = add_text_features(
        df_scored,
        text_col="chunk_text",
        sentence_count_col="n_sentences",
    )

    df_scored = df_scored.loc[:, ~df_scored.columns.duplicated()].copy()

    article_summary = summarize_articles(df_scored)
    heatmap_matrix = build_heatmap_matrix(df_scored, max_articles=max_articles_in_heatmap)
    feature_comparison = compare_feature_groups(df_scored)
    top_tokens_high, top_tokens_low = compare_top_tokens(df_scored)

    chunks_csv = out_dir / "chunk_scores.csv"
    article_csv = out_dir / "article_summary.csv"
    feature_csv = out_dir / "feature_comparison.csv"
    heatmap_png = out_dir / "scientific_articles_heatmap.png"
    examples_md = out_dir / "examples_report.md"
    top_tokens_high_csv = out_dir / "top_tokens_high_ai.csv"
    top_tokens_low_csv = out_dir / "top_tokens_low_ai.csv"

    if show_progress:
        print("Saving artifacts...")

    df_scored.to_csv(chunks_csv, index=False)
    article_summary.to_csv(article_csv, index=False)
    feature_comparison.to_csv(feature_csv, index=False)
    top_tokens_high.to_csv(top_tokens_high_csv, index=False)
    top_tokens_low.to_csv(top_tokens_low_csv, index=False)
    save_heatmap(heatmap_matrix, heatmap_png)
    build_examples_report(df_scored, out_path=examples_md)

    global_stats = {
        "n_articles": int(df_articles[article_id_col].nunique()),
        "n_chunks": int(len(df_scored)),
        "mean_p_ai": float(df_scored["p_ai"].mean()),
        "median_p_ai": float(df_scored["p_ai"].median()),
        "share_ai_like": float(df_scored["is_ai_like"].mean()),
        "threshold": float(threshold),
        "top_article_by_mean_p_ai": article_summary.iloc[0]["article_id"] if not article_summary.empty else None,
    }


    return {
        "df_chunks": df_chunks,
        "df_scored": df_scored,
        "article_summary": article_summary,
        "feature_comparison": feature_comparison,
        "top_tokens_high": top_tokens_high,
        "top_tokens_low": top_tokens_low,
        "heatmap_matrix": heatmap_matrix,
        "global_stats": global_stats,
        "artifacts": {
            "chunk_scores_csv": str(chunks_csv),
            "article_summary_csv": str(article_csv),
            "feature_comparison_csv": str(feature_csv),
            "top_tokens_high_csv": str(top_tokens_high_csv),
            "top_tokens_low_csv": str(top_tokens_low_csv),
            "heatmap_png": str(heatmap_png),
            "examples_md": str(examples_md),
        },
    }
