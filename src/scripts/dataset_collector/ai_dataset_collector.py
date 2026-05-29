import os
import re
import json
import random
import threading
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

import pyarrow as pa
import pyarrow.parquet as pq

load_dotenv()

INPUT_PARQUET = os.path.join("data", "combined_human.parquet")

OUTPUT_PARQUET = os.path.join("data", "combined_human_labeled.parquet")

OUTPUT_DIR = os.path.join("data", "labeled_rows")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 43
N_LONG = 6124
N_SHORT = 6124

MAX_CHARS_PER_TEXT = 6000
READ_FRACTION = 1.0

MAX_WORKERS = 30
LLM_MAX_CONCURRENCY = 65
OPENAI_MAX_CONCURRENCY = 30
DEEPSEEK_MAX_CONCURRENCY = 30

SKIP_IF_JSON_EXISTS = True
FLUSH_EVERY = 200

MAX_COMPLETION_TOKENS = 1000

MIN_SHORT_SENTENCES = 2
MAX_SHORT_SENTENCES = 8
MIN_TEXT_CHARS = 45

random.seed(SEED)

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


@dataclass(frozen=True)
class LLMChoice:
    provider: str
    model: str


LLM_POOL: List[LLMChoice] = [

    LLMChoice("openai", "gpt-4.1-mini"),
    LLMChoice("openai", "gpt-5-mini"),

    LLMChoice("deepseek", "deepseek-chat"),
    LLMChoice("deepseek", "deepseek-reasoner"),
]


def make_clients() -> Dict[str, OpenAI]:
    clients: Dict[str, OpenAI] = {}

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        clients["openai"] = OpenAI(api_key=openai_key)

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        clients["deepseek"] = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")

    if not clients:
        raise RuntimeError("Не найдены ключи. Установи OPENAI_API_KEY и/или DEEPSEEK_API_KEY.")
    return clients


def pick_llm(clients: Dict[str, OpenAI]) -> LLMChoice:
    available = [c for c in LLM_POOL if c.provider in clients]
    if not available:
        raise RuntimeError("Нет доступных LLM: проверь ключи и LLM_POOL.")
    return random.choice(available)


def pick_deepseek(clients: Dict[str, OpenAI]) -> Optional[LLMChoice]:
    if "deepseek" not in clients:
        return None

    return LLMChoice("deepseek", "deepseek-chat")

SYSTEM_SUMMARY = (
    "Ты — аккуратный редактор русскоязычных текстов. "
    "Строго следуй инструкциям пользователя. "
    "Ничего не выдумывай и не добавляй фактов. "
    "Пиши кратко и по делу."
)

SYSTEM_PARAPHRASE = (
    "Ты — аккуратный перефразировщик русскоязычных текстов. "
    "Строго следуй инструкциям пользователя. "
    "Ничего не выдумывай и не добавляй фактов. "
    "Сохраняй смысл, но меняй формулировки. "
    "Пиши кратко. "
    "Пиши вывод на русском языке."
)

USER_SUMMARY_TEMPLATE_EN = """Summarize the following Russian text in EXACTLY 4–6 sentences.

Rules:
- Output ONLY the final summary (no lists, headings, explanations, prefaces).
- Do NOT add any new facts or details.
- Do NOT use bullets, numbering, or Markdown.
- Write the summary in Russian.

TEXT:
{text}
"""

USER_PARAPHRASE_TEMPLATE_EN = """Rewrite the following Russian text using different wording (paraphrase it).
Then provide a brief summary of the meaning in 2–3 sentences.

Rules:
- Do NOT add any new facts or details.
- Keep the meaning as close as possible to the original.
- Write in Russian.
- Output ONLY the final summary (4–6 sentences). No comments, no lists.

TEXT:
{text}
"""


USER_SUMMARY_TEMPLATE_RU = """Суммаризируй следующий русский текст РОВНО в 4–6 предложениях.

Правила:
- Выведи ТОЛЬКО итоговую суммаризацию (никаких списков, заголовков, пояснений, прелюдий).
- Не добавляй новых фактов и деталей.
- Не используй буллеты, нумерацию или Markdown.

ТЕКСТ:
{text}
"""

USER_PARAPHRASE_TEMPLATE_RU = """Перепиши следующий русский текст другими словами (перефразируй).
Затем дай краткую суммаризацию смысла в 4–6 предложениях.

Правила:
- Не добавляй новых фактов и деталей.
- Сохраняй исходный смысл максимально близко.
- Выводи на русском языке.
- Выведи ТОЛЬКО итоговую суммаризацию (4–6 предложения). Без комментариев и списков.

ТЕКСТ:
{text}
"""


_all_sem = threading.Semaphore(LLM_MAX_CONCURRENCY)
_openai_sem = threading.Semaphore(OPENAI_MAX_CONCURRENCY)
_deepseek_sem = threading.Semaphore(DEEPSEEK_MAX_CONCURRENCY)


def _provider_sem(provider: str) -> threading.Semaphore:
    return _openai_sem if provider == "openai" else _deepseek_sem


class TransientLLMError(Exception):
    pass


class EmptyLLMResponseError(TransientLLMError):
    pass


def _is_transient_error(msg_lower: str) -> bool:
    transient_markers = ["rate", "timeout", "tempor", "503", "502", "connection", "overload", "try again"]
    return any(m in msg_lower for m in transient_markers)


def _clean_text(s: str) -> str:
    return (s or "").replace("\x00", "").strip()


def _extract_text_from_responses_api(resp: Any) -> str:
    out = getattr(resp, "output_text", None)
    if out:
        return str(out).strip()

    text = ""
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        text += getattr(c, "text", "") or ""
    except Exception:
        pass
    return (text or "").strip()


_CYR_RE = re.compile(r"[А-Яа-яЁё]")


def _looks_russian(s: str) -> bool:
    if not s:
        return False
    return len(_CYR_RE.findall(s)) >= 3


def _build_user_prompt(provider: str, mode: str, clipped_text: str) -> str:

    if provider == "openai":
        if mode == "summary":
            return USER_SUMMARY_TEMPLATE_EN.format(text=clipped_text)
        return USER_PARAPHRASE_TEMPLATE_EN.format(text=clipped_text)

    if mode == "summary":
        return USER_SUMMARY_TEMPLATE_RU.format(text=clipped_text)
    return USER_PARAPHRASE_TEMPLATE_RU.format(text=clipped_text)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(TransientLLMError),
)
def call_llm(
    client: OpenAI,
    provider: str,
    model: str,
    system_text: str,
    user_text: str,
    *,
    clients: Optional[Dict[str, OpenAI]] = None,
    mode: Optional[str] = None,
    clipped_source_text: Optional[str] = None,
) -> str:

    with _all_sem:
        with _provider_sem(provider):
            try:
                if provider == "openai":
                    resp = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
                        ],
                        max_output_tokens=MAX_COMPLETION_TOKENS,
                    )
                    out = _clean_text(_extract_text_from_responses_api(resp))


                    if (not out) or (not _looks_russian(out)):
                        if clients:
                            ds = pick_deepseek(clients)
                            if ds:
                                ds_client = clients["deepseek"]
                                if mode and clipped_source_text:
                                    ds_user = _build_user_prompt("deepseek", mode, clipped_source_text)
                                else:
                                    ds_user = user_text

                                ds_resp = ds_client.chat.completions.create(
                                    model=ds.model,
                                    messages=[
                                        {"role": "system", "content": system_text},
                                        {"role": "user", "content": ds_user},
                                    ],
                                    temperature=0.2,
                                    max_tokens=MAX_COMPLETION_TOKENS,
                                )
                                out2 = _clean_text(ds_resp.choices[0].message.content or "")
                                if not out2:
                                    raise EmptyLLMResponseError(
                                        "DeepSeek returned empty response (after OpenAI fallback)."
                                    )
                                return out2

                        raise EmptyLLMResponseError(
                            "OpenAI returned empty/non-RU response and DeepSeek is unavailable."
                        )

                    return out


                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.2,
                    max_tokens=MAX_COMPLETION_TOKENS,
                )
                out = _clean_text(resp.choices[0].message.content or "")
                if not out:
                    raise EmptyLLMResponseError("DeepSeek returned empty response.")
                return out

            except Exception as e:
                msg = str(e)
                msg_lower = msg.lower()
                if isinstance(e, TransientLLMError) or _is_transient_error(msg_lower):
                    raise TransientLLMError(msg)
                raise


def row_json_path(row_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", row_id)
    return os.path.join(OUTPUT_DIR, f"{safe}.json")


def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_already_labeled_ids(output_dir: str) -> Set[str]:

    labeled: Set[str] = set()
    try:
        for name in os.listdir(output_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(output_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                row_id = j.get("rowId")
                if not row_id:
                    continue
                if j.get("error"):
                    continue
                gt = (j.get("generatedText") or "").strip()
                if gt:
                    labeled.add(row_id)
            except Exception:
                continue
    except FileNotFoundError:
        pass
    return labeled


def reservoir_push(reservoir: List[Dict[str, Any]], item: Dict[str, Any], k: int, seen_count: int) -> None:
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = random.randint(1, seen_count)
    if j <= k:
        reservoir[j - 1] = item


def get_total_rows_parquet_fast(parquet_path: str) -> int:
    pf = pq.ParquetFile(parquet_path)
    return int(pf.metadata.num_rows)


def sample_long_short_streaming(
    read_fraction: float,
    *,
    already_labeled_src: Optional[Set[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:

    read_fraction = float(read_fraction)
    if read_fraction <= 0:
        raise ValueError("READ_FRACTION должен быть > 0 (например 0.01, 0.1, 1.0).")
    if read_fraction > 1.0:
        read_fraction = 1.0

    total_rows = get_total_rows_parquet_fast(INPUT_PARQUET)
    limit_rows = max(1, int(total_rows * read_fraction))

    ds_stream = load_dataset(
        "parquet",
        data_files=INPUT_PARQUET,
        split="train",
        streaming=True,
    )

    long_res: List[Dict[str, Any]] = []
    short_res: List[Dict[str, Any]] = []
    seen_long = 0
    seen_short = 0

    pbar = tqdm(
        total=limit_rows,
        desc=f"Streaming + sampling ({read_fraction:.2%})",
        unit="rows",
        dynamic_ncols=True,
    )

    for idx, ex in enumerate(ds_stream, start=1):
        if idx > limit_rows:
            break

        if already_labeled_src is not None and idx in already_labeled_src:
            pbar.update(1)
            continue

        raw_text = ex.get("text", "")
        text = str(raw_text or "").strip()

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
            "text": text,
            "label": ex.get("label", "human"),
            "initial_dataset": ex.get("initial_dataset", ""),
            "sentenceCount": sc,
            "sourceRowIndex": idx,
        }

        if sc > MAX_SHORT_SENTENCES:
            if sc > 4:
                seen_long += 1
                reservoir_push(long_res, base, N_LONG, seen_long)
        else:
            if MIN_SHORT_SENTENCES <= sc <= MAX_SHORT_SENTENCES:
                seen_short += 1
                reservoir_push(short_res, base, N_SHORT, seen_short)

        pbar.update(1)

        if idx % 50_000 == 0 or idx == limit_rows:
            remaining = limit_rows - idx
            pbar.set_postfix_str(f"remaining={remaining} | long_seen={seen_long} short_seen={seen_short}")

    pbar.close()
    return long_res, short_res, limit_rows, total_rows


def process_one(
    base_row: Dict[str, Any],
    row_id: str,
    mode: str,
    clients: Dict[str, OpenAI],
) -> Optional[Dict[str, Any]]:
    out_path = row_json_path(row_id)


    if SKIP_IF_JSON_EXISTS and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            if (j.get("generatedText") or "").strip() and not j.get("error"):
                return j
        except Exception:
            pass

    llm_used = None

    try:
        text_c = clip_text(str(base_row.get("text", "") or ""))

        llm = pick_llm(clients)
        client = clients[llm.provider]
        llm_used = f"{llm.provider}:{llm.model}"

        if mode == "summary":
            system_text = SYSTEM_SUMMARY
            user_prompt = _build_user_prompt(llm.provider, "summary", text_c)
            generated = call_llm(
                client,
                llm.provider,
                llm.model,
                system_text,
                user_prompt,
                clients=clients,
                mode="summary",
                clipped_source_text=text_c,
            )
            class_label = "AI"
            task_type = "summarize_2_3_sent"
        else:
            system_text = SYSTEM_PARAPHRASE
            user_prompt = _build_user_prompt(llm.provider, "paraphrase", text_c)
            generated = call_llm(
                client,
                llm.provider,
                llm.model,
                system_text,
                user_prompt,
                clients=clients,
                mode="paraphrase",
                clipped_source_text=text_c,
            )
            class_label = "human+AI"
            task_type = "paraphrase_and_summarize"

        out = dict(base_row)
        out.update({
            "generatedText": generated,
            "taskType": task_type,
            "classLabel": class_label,
            "llmUsed": llm_used,
            "rowId": row_id,
        })

        atomic_write_json(out_path, out)
        return out

    except Exception as e:
        err = dict(base_row)
        err.update({
            "rowId": row_id,
            "mode": mode,
            "llmUsed": llm_used,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        atomic_write_json(out_path, err)
        return None


def make_parquet_writer(path: str, sample_row: Dict[str, Any]) -> pq.ParquetWriter:
    table = pa.Table.from_pylist([sample_row])
    return pq.ParquetWriter(path, table.schema)


def normalize_rows(rows: List[Dict[str, Any]], all_keys: List[str]) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = {k: r.get(k, None) for k in all_keys}
        out.append(rr)
    return out


def main():
    clients = make_clients()

    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(f"Не найден файл: {INPUT_PARQUET}")

    print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
    print("DEEPSEEK_API_KEY loaded:", bool(os.getenv("DEEPSEEK_API_KEY")))

    labeled_row_ids = load_already_labeled_ids(OUTPUT_DIR)

    labeled_src_idx: Set[int] = set()
    for rid in labeled_row_ids:
        m = re.search(r"_src(\d+)$", rid)
        if m:
            labeled_src_idx.add(int(m.group(1)))

    print(f"Already labeled (by JSON in {OUTPUT_DIR}): {len(labeled_row_ids)} rows")

    long_rows, short_rows, limit_rows, total_rows = sample_long_short_streaming(
        READ_FRACTION,
        already_labeled_src=labeled_src_idx,
    )
    print(f"Total rows in parquet: {total_rows}")
    print(f"Read rows limit: {limit_rows} ({READ_FRACTION:.2%})")
    print(f"Sampled long={len(long_rows)} (>4 sent), short={len(short_rows)} (1..4 sent, >= {MIN_TEXT_CHARS} chars)")

    jobs: List[Tuple[str, Dict[str, Any], str]] = []

    for i, base in enumerate(long_rows):
        row_id = f"long_{i:05d}_src{base.get('sourceRowIndex', 0)}"
        if row_id not in labeled_row_ids:
            jobs.append((row_id, base, "summary"))

    for i, base in enumerate(short_rows):
        row_id = f"short_{i:05d}_src{base.get('sourceRowIndex', 0)}"
        if row_id not in labeled_row_ids:
            jobs.append((row_id, base, "paraphrase"))

    random.shuffle(jobs)
    print(f"Jobs to process (not labeled yet): {len(jobs)}")

    parquet_writer: Optional[pq.ParquetWriter] = None
    parquet_lock = threading.Lock()
    parquet_keys: Optional[List[str]] = None

    if os.path.exists(OUTPUT_PARQUET):
        os.remove(OUTPUT_PARQUET)

    buffer_rows: List[Dict[str, Any]] = []
    ok = 0
    fail = 0
    error_samples: List[str] = []

    def flush_parquet(rows: List[Dict[str, Any]]) -> None:
        nonlocal parquet_writer, parquet_keys
        if not rows:
            return

        with parquet_lock:
            if parquet_keys is None:
                keys = sorted(set().union(*(r.keys() for r in rows)))
                parquet_keys = keys

            norm = normalize_rows(rows, parquet_keys)
            table = pa.Table.from_pylist(norm)

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema)

            parquet_writer.write_table(table)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_one, base_row, row_id, mode, clients): row_id
            for (row_id, base_row, mode) in jobs
        }

        pbar = tqdm(total=len(futures), desc="LLM labeling", dynamic_ncols=True)
        for fut in as_completed(futures):
            row_id = futures[fut]
            try:
                res = fut.result()
                if res is not None:
                    buffer_rows.append(res)
                    ok += 1
                else:
                    fail += 1
                    if len(error_samples) < 5:
                        p = row_json_path(row_id)
                        if os.path.exists(p):
                            try:
                                with open(p, "r", encoding="utf-8") as f:
                                    j = json.load(f)
                                if "error" in j:
                                    error_samples.append(f"{row_id}: {j.get('llmUsed')} | {j.get('error')}")
                            except Exception:
                                pass
            except Exception as e:
                fail += 1
                if len(error_samples) < 5:
                    error_samples.append(f"{row_id}: FUTURE_EXCEPTION {repr(e)}")

            pbar.update(1)

            if (ok + fail) % FLUSH_EVERY == 0 and buffer_rows:
                flush_parquet(buffer_rows)
                buffer_rows.clear()

        pbar.close()

    if buffer_rows:
        flush_parquet(buffer_rows)
        buffer_rows.clear()

    if parquet_writer is not None:
        parquet_writer.close()

    print(f"Saved: {OUTPUT_PARQUET} | ok={ok} | fail={fail}")
    print(f"Per-row JSON saved in: {OUTPUT_DIR}/")

    if error_samples:
        print("\nПримеры ошибок:")
        for s in error_samples:
            print("-", s)


if __name__ == "__main__":
    main()
