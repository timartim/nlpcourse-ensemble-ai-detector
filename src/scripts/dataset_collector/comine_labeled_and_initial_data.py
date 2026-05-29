import os
import re
import json
import random
from typing import Dict, Any, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


INPUT_PARQUET = os.path.join("../../../data/train_data", "combined_human.parquet")
LABELED_PARQUET = os.path.join("../../../data/train_data", "labeled_rows.parquet")
OUTPUT_FINAL = os.path.join("../../../data/train_data", "final_dataset.parquet")

SEED = 40
random.seed(SEED)

N_LONG_ADD = 80_000
N_SHORT_ADD = 80_000

READ_FRACTION = 1.0
MAX_CHARS_PER_TEXT = 6000

MIN_TEXT_CHARS = 30
MIN_SHORT_SENTENCES = 1
MAX_SHORT_SENTENCES = 4
LONG_SENTENCES_GT = 4

FLUSH_EVERY = 50_000


_SENT_SPLIT_RE = re.compile(r"[.!?…]+(?:\s+|$)")


def count_sentences(text: str) -> int:
    if not text:
        return 0
    parts = _SENT_SPLIT_RE.split(str(text).strip())
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def clip_text(text: str, max_chars: int = MAX_CHARS_PER_TEXT) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def reservoir_push(reservoir: List[Dict[str, Any]], item: Dict[str, Any], k: int, seen_count: int) -> None:
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = random.randint(1, seen_count)
    if j <= k:
        reservoir[j - 1] = item


def load_labeled_source_indices(labeled_parquet_path: str) -> Set[int]:
    """
    Берём sourceRowIndex из размеченного набора.
    Это то, что поможет исключить уже размеченные строки из исходного датасета.
    """
    if not os.path.exists(labeled_parquet_path):
        raise FileNotFoundError(f"Не найден размеченный parquet: {labeled_parquet_path}")

    tbl = pq.read_table(labeled_parquet_path, columns=["sourceRowIndex"])
    col = tbl.column("sourceRowIndex")

    labeled: Set[int] = set()

    for v in col.to_pylist():
        if v is None:
            continue
        try:
            labeled.add(int(v))
        except Exception:
            continue
    return labeled


def get_total_rows_parquet_fast(parquet_path: str) -> int:
    pf = pq.ParquetFile(parquet_path)
    return int(pf.metadata.num_rows)


def sample_unlabeled_long_short_from_source(
    input_parquet_path: str,
    labeled_src_idx: Set[int],
    read_fraction: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    """
    Возвращает:
      - long_to_add: список из <= N_LONG_ADD записей
      - short_to_add: список из <= N_SHORT_ADD записей
      - limit_rows: сколько строк прочитали (из-за read_fraction)
      - total_rows: сколько всего строк в parquet
    """
    total_rows = get_total_rows_parquet_fast(input_parquet_path)
    rf = float(read_fraction)
    if rf <= 0:
        raise ValueError("READ_FRACTION должен быть > 0.")
    if rf > 1.0:
        rf = 1.0
    limit_rows = max(1, int(total_rows * rf))

    ds_stream = load_dataset(
        "parquet",
        data_files=input_parquet_path,
        split="train",
        streaming=True,
    )

    long_res: List[Dict[str, Any]] = []
    short_res: List[Dict[str, Any]] = []
    seen_long = 0
    seen_short = 0


    from tqdm import tqdm
    pbar = tqdm(total=limit_rows, desc=f"Sampling unlabeled ({rf:.2%})", unit="rows", dynamic_ncols=True)

    for idx, ex in enumerate(ds_stream, start=1):
        if idx > limit_rows:
            break


        if idx in labeled_src_idx:
            pbar.update(1)
            continue

        text = str(ex.get("text", "") or "").strip()
        if not text:
            pbar.update(1)
            continue
        if len(text) < MIN_TEXT_CHARS:
            pbar.update(1)
            continue

        sc = count_sentences(text)
        if sc < 1:
            pbar.update(1)
            continue

        base = {
            "text": clip_text(text, MAX_CHARS_PER_TEXT),
            "label": ex.get("label", "human"),
            "initial_dataset": ex.get("initial_dataset", ""),
            "sentenceCount": sc,
            "sourceRowIndex": idx,
        }


        if sc > LONG_SENTENCES_GT:
            seen_long += 1
            reservoir_push(long_res, base, N_LONG_ADD, seen_long)


        elif MIN_SHORT_SENTENCES <= sc <= MAX_SHORT_SENTENCES:
            seen_short += 1
            reservoir_push(short_res, base, N_SHORT_ADD, seen_short)

        pbar.update(1)

        if idx % 50_000 == 0:
            pbar.set_postfix_str(
                f"long={len(long_res)}/{N_LONG_ADD} short={len(short_res)}/{N_SHORT_ADD}"
            )

    pbar.close()
    return long_res, short_res, limit_rows, total_rows


def ensure_columns(table: pa.Table, all_cols: List[str]) -> pa.Table:
    """
    Добавляет недостающие колонки как null, выравнивает порядок колонок.
    """
    existing = set(table.column_names)
    cols = []
    for c in all_cols:
        if c in existing:
            cols.append(table[c])
        else:
            cols.append(pa.nulls(table.num_rows))
    return pa.Table.from_arrays(cols, names=all_cols)


def rows_to_table(rows: List[Dict[str, Any]], all_cols: List[str], schema: Optional[pa.Schema] = None) -> pa.Table:
    """
    Превращает список dict в таблицу с заданными колонками.
    """
    norm = [{k: r.get(k, None) for k in all_cols} for r in rows]
    t = pa.Table.from_pylist(norm)
    t = ensure_columns(t, all_cols)
    if schema is not None:

        try:
            t = t.cast(schema)
        except Exception:
            pass
    return t


def write_final_dataset(
    labeled_parquet_path: str,
    long_add: List[Dict[str, Any]],
    short_add: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Пишем final_dataset.parquet = labeled + добавленные 80k long + 80k short (unlabeled).
    """


    new_rows: List[Dict[str, Any]] = []

    for i, r in enumerate(long_add):
        rr = dict(r)
        rr.update({
            "rowId": f"orig_long_{i:05d}_src{r.get('sourceRowIndex', 0)}",
            "taskType": "original",
            "classLabel": "human",
            "llmUsed": None,
            "generatedText": None,
        })
        new_rows.append(rr)

    for i, r in enumerate(short_add):
        rr = dict(r)
        rr.update({
            "rowId": f"orig_short_{i:05d}_src{r.get('sourceRowIndex', 0)}",
            "taskType": "original",
            "classLabel": "human",
            "llmUsed": None,
            "generatedText": None,
        })
        new_rows.append(rr)


    pf = pq.ParquetFile(labeled_parquet_path)
    labeled_schema = pf.schema_arrow

    labeled_cols = list(labeled_schema.names)
    new_cols = sorted(set().union(*(r.keys() for r in new_rows))) if new_rows else []
    all_cols = sorted(set(labeled_cols).union(new_cols))


    if os.path.exists(output_path):
        os.remove(output_path)

    writer: Optional[pq.ParquetWriter] = None


    kept_labeled = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        t = ensure_columns(t, all_cols)
        if writer is None:
            writer = pq.ParquetWriter(output_path, t.schema)
        writer.write_table(t)
        kept_labeled += t.num_rows
        if kept_labeled % (FLUSH_EVERY * 2) == 0:
            print(f"written labeled: {kept_labeled}")


    kept_new = 0
    if new_rows:
        for start in range(0, len(new_rows), FLUSH_EVERY):
            batch = new_rows[start:start + FLUSH_EVERY]
            t2 = rows_to_table(batch, all_cols, schema=writer.schema if writer else None)
            if writer is None:
                writer = pq.ParquetWriter(output_path, t2.schema)
            writer.write_table(t2)
            kept_new += t2.num_rows
            if kept_new % (FLUSH_EVERY * 2) == 0:
                print(f"written new: {kept_new}")

    if writer is not None:
        writer.close()

    print("\nDONE")
    print(f"output: {output_path}")
    print(f"labeled_rows_written: {kept_labeled}")
    print(f"new_unlabeled_written: {kept_new}")
    print(f"total_written: {kept_labeled + kept_new}")


def main():
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(f"Не найден INPUT_PARQUET: {INPUT_PARQUET}")
    if not os.path.exists(LABELED_PARQUET):
        raise FileNotFoundError(f"Не найден LABELED_PARQUET: {LABELED_PARQUET}")

    print("Loading labeled sourceRowIndex set...")
    labeled_src = load_labeled_source_indices(LABELED_PARQUET)
    print(f"Labeled sourceRowIndex: {len(labeled_src)}")

    print("\nSampling new unlabeled rows from source...")
    long_add, short_add, limit_rows, total_rows = sample_unlabeled_long_short_from_source(
        INPUT_PARQUET,
        labeled_src_idx=labeled_src,
        read_fraction=READ_FRACTION,
    )
    print(f"\nSource total rows: {total_rows}")
    print(f"Source read limit: {limit_rows} ({READ_FRACTION:.2%})")
    print(f"Sampled to add: long={len(long_add)} / {N_LONG_ADD}, short={len(short_add)} / {N_SHORT_ADD}")

    print("\nWriting final dataset parquet...")
    write_final_dataset(
        labeled_parquet_path=LABELED_PARQUET,
        long_add=long_add,
        short_add=short_add,
        output_path=OUTPUT_FINAL,
    )


if __name__ == "__main__":
    main()
