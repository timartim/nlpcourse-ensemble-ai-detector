import os
import json
from typing import Dict, Any, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


INPUT_DIR = os.path.join("../../../data/train_data", "labeled_rows")
OUTPUT_PARQUET = os.path.join("../../../data/train_data", "labeled_rows.parquet")

INCLUDE_ERROR_ROWS = False
GLOB_SUFFIX = ".json"
FLUSH_EVERY = 2000


def iter_json_files(input_dir: str, suffix: str = ".json"):
    for name in os.listdir(input_dir):
        if name.endswith(suffix):
            yield os.path.join(input_dir, name)


def safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_rows(rows: List[Dict[str, Any]], all_keys: List[str]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append({k: r.get(k, None) for k in all_keys})
    return out


def write_parquet_incremental(rows_iter, output_path: str) -> None:
    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    all_keys: Optional[List[str]] = None

    buffer: List[Dict[str, Any]] = []
    total = 0
    kept = 0
    skipped = 0
    bad = 0


    if os.path.exists(output_path):
        os.remove(output_path)

    def flush(buf: List[Dict[str, Any]]):
        nonlocal writer, schema, all_keys, kept
        if not buf:
            return


        if all_keys is None:
            keys = set()
            for r in buf:
                keys.update(r.keys())
            all_keys = sorted(keys)

        norm = normalize_rows(buf, all_keys)
        table = pa.Table.from_pylist(norm)

        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(output_path, schema)

        writer.write_table(table)
        kept += len(buf)

    for path in rows_iter:
        total += 1
        j = safe_load_json(path)
        if j is None:
            bad += 1
            continue


        if (not INCLUDE_ERROR_ROWS) and j.get("error"):
            skipped += 1
            continue


        if (not INCLUDE_ERROR_ROWS) and (not (j.get("generatedText") or "").strip()):
            skipped += 1
            continue

        buffer.append(j)

        if len(buffer) >= FLUSH_EVERY:
            flush(buffer)
            buffer.clear()

        if total % 5000 == 0:
            print(f"processed={total} kept={kept} skipped={skipped} bad_json={bad}")


    if buffer:
        flush(buffer)
        buffer.clear()

    if writer is not None:
        writer.close()

    print("\nDONE")
    print(f"input_dir:      {INPUT_DIR}")
    print(f"output_parquet: {OUTPUT_PARQUET}")
    print(f"processed:      {total}")
    print(f"kept:           {kept}")
    print(f"skipped:        {skipped} (error/empty rows)")
    print(f"bad_json:       {bad} (failed to parse)")


def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    files = list(iter_json_files(INPUT_DIR, GLOB_SUFFIX))
    print(f"Found {len(files)} json files in: {INPUT_DIR}")

    write_parquet_incremental(files, OUTPUT_PARQUET)


if __name__ == "__main__":
    main()
