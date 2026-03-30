"""Auto-generated from HM_Taha.ipynb cell 5."""

import os
import json
import re
import time
import glob
import asyncio
from typing import List, Dict, Optional

import pandas as pd
from openai import AsyncOpenAI

# Configuration
client = AsyncOpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY",
        ""
    )
)

PRIMARY_MODEL = "o4-mini"
FALLBACK_MODEL = "o3"
PRIMARY_ATTEMPTS = 2

INPUT_CSV = "b_processed_results_LLM.csv"
OUTPUT_CSV = "c_parsed_names.csv"
CHUNK_SIZE = 64
OVERLAP_SIZE = 16
CONCURRENT_REQUESTS = 10

TMP_DIR = "test_c_tmp_parsed_names_batches"
RESUME_FROM_CHECKPOINTS = True


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def chunk_records_with_overlap(records: List[Dict], chunk_size: int, overlap_size: int):
    """
    Create chunks with overlap to ensure name variations appear together in at least one batch.
    The overlap ensures that records near batch boundaries get processed with context from both sides.
    """
    yield records[:chunk_size]

    i = chunk_size
    while i < len(records):
        start_idx = max(0, i - overlap_size)
        end_idx = min(i + chunk_size - overlap_size, len(records))

        chunk = records[start_idx:end_idx]

        chunk_with_metadata = []
        for j, record in enumerate(chunk):
            record_copy = record.copy()
            record_copy['_is_overlap'] = (start_idx + j) < i
            record_copy['_original_index'] = start_idx + j
            chunk_with_metadata.append(record_copy)

        yield chunk_with_metadata

        i += (chunk_size - overlap_size)


def extract_json_array(text: str) -> str:
    """Extract JSON array from potentially malformed LLM response."""
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or start > end:
        raise ValueError("No JSON array delimiters found")
    return text[start:end + 1]


def load_checkpoint(batch_idx: int) -> Optional[List[Dict]]:
    """
    Load an existing checkpoint for the given batch index.
    Returns parsed list if available and valid, else None.
    """
    path = os.path.join(TMP_DIR, f"batch_{batch_idx:05d}.json")
    part = path + ".part"

    for candidate in (path, part):
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    return None


def save_checkpoint(batch_idx: int, data: List[Dict]) -> None:
    """Atomically write the batch results to a checkpoint file."""
    ensure_dir(TMP_DIR)
    path = os.path.join(TMP_DIR, f"batch_{batch_idx:05d}.json")
    tmp_path = path + ".part"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def consolidate_checkpoints_with_dedup() -> pd.DataFrame:
    """
    Read all batch checkpoint files and return a DataFrame of parsed rows.
    Handles deduplication of overlapped records by preferring the most complete version.
    """
    ensure_dir(TMP_DIR)
    files = sorted(glob.glob(os.path.join(TMP_DIR, "batch_*.json")))

    best_records = {}

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list):
                for record in rows:
                    if isinstance(record, dict) and 'RecID' in record:
                        rec_id = record['RecID']

                        if rec_id not in best_records:
                            best_records[rec_id] = record
                        else:
                            current = best_records[rec_id]
                            new_completeness = sum(1 for v in record.values() if v and str(v).strip())
                            current_completeness = sum(1 for v in current.values() if v and str(v).strip())

                            new_last = str(record.get('last_name', ''))
                            current_last = str(current.get('last_name', ''))

                            if new_completeness > current_completeness or \
                               (new_completeness == current_completeness and len(new_last) > len(current_last)):
                                best_records[rec_id] = record
        except Exception as e:
            print(f"Warning: Could not read checkpoint file {fp}: {e}")
            pass

    if not best_records:
        return pd.DataFrame(columns=["RecID", "first_name", "middle_name", "last_name"])

    return pd.DataFrame(list(best_records.values()), dtype=str)


async def parse_chunk_async(chunk: List[Dict], model: str) -> str:
    """Process a chunk of records using the LLM API."""
    clean_chunk = []
    for record in chunk:
        clean_record = {k: v for k, v in record.items() if not k.startswith('_')}
        clean_chunk.append(clean_record)

    system_message = {
        "role": "system",
        "content": (
            "You are an expert in batch name standardization and normalization. Process the entire batch systematically.\n"
            "\n"
            "## STEP-BY-STEP PROCESS:\n"
            "\n"
            "### 1. NORMALIZE\n"
            "• Convert to uppercase\n"
            "• Remove all punctuation except spaces\n"
            "• Replace multiple spaces with single spaces\n"
            "• Trim whitespace\n"
            "\n"
            "### 2. ANALYZE BATCH PATTERNS\n"
            "• Identify all unique tokens across the entire batch\n"
            "• Count frequency of each token in each position (first/middle/last)\n"
            "• Build a comprehensive nickname-to-formal name mapping from the batch data\n"
            "\n"
            "### 3. NICKNAME & ABBREVIATION EXPANSION\n"
            "Apply these rules in order:\n"
            "• **Common nicknames**: MIKE→MITCHELL, MIKE→MICHAEL, BOB→ROBERT, JIM→JAMES, etc.\n"
            "• **Initials**: Single letters (M, J, S) → most frequent full name in that position\n"
            "• **Partial names**: MITCH→MITCHELL, JACKIE→JACQUELINE if the full form exists in batch\n"
            "• **Fuzzy matches**: Levenshtein distance ≤2 with 80%+ similarity\n"
            "\n"
            "### 4. BATCH-BASED INFERENCE\n"
            "For incomplete names, infer missing parts by:\n"
            "• Finding records with same last name and complete information\n"
            "• Using most frequent first/middle name combination for that family\n"
            "• Prioritizing exact matches over partial matches\n"
            "\n"
            "### 5. STANDARDIZE TO CANONICAL FORM\n"
            "• Choose the most complete/formal version found in the batch\n"
            "• MITCHELL is more formal than MIKE, MICHAEL, MITCH\n"
            "• JACQUELINE is more formal than JACKIE, JACKI\n"
            "• Always prefer the longest, most complete form\n"
            "\n"
            "### 6. FINAL PARSING\n"
            "Split standardized names:\n"
            "• 3+ tokens: first_name=token[0], middle_name=token[1], last_name=remaining_tokens\n"
            "• 2 tokens: first_name=token[0], middle_name='', last_name=token[1]\n"
            "• 1 token: first_name='', middle_name='', last_name=token[0]\n"
            "\n"
            "## CRITICAL REQUIREMENTS:\n"
            "• Process ALL records in the batch consistently\n"
            "• Use the batch context to resolve ambiguities\n"
            "• Map ALL variations to the same canonical form\n"
            "• Handle nicknames intelligently (MIKE→MITCHELL if MITCHELL exists in batch)\n"
            "• Output ONLY valid JSON array\n"
            "\n"
            "## EXAMPLE:\n"
            "Input batch contains: 'MIKE S', 'MITCHELL SANDERS', 'M SANDERS', 'MICHAEL S'\n"
            "Analysis: MITCHELL is the most complete form\n"
            "Result: All should become 'MITCHELL' as first_name\n"
            "\n"
            "Output format: [{\"RecID\":\"...\", \"first_name\":\"...\", \"middle_name\":\"...\", \"last_name\":\"...\"}]\n"
        )
    }
    user_message = {
        "role": "user",
        "content": json.dumps(clean_chunk, ensure_ascii=False)
    }
    resp = await client.chat.completions.create(model=model, messages=[system_message, user_message])
    return resp.choices[0].message.content


def safe_parse(raw: str) -> List[Dict]:
    """Safely parse JSON response with fallback extraction."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        snippet = extract_json_array(raw)
        return json.loads(snippet)


async def process_batch_with_fallback(batch_idx: int, chunk: List[Dict], total_batches: int, semaphore: asyncio.Semaphore) -> None:
    """Process a single batch with fallback model support."""
    async with semaphore:
        if RESUME_FROM_CHECKPOINTS:
            existing = load_checkpoint(batch_idx)
            if existing is not None:
                print(f"Batch {batch_idx}/{total_batches} ({len(chunk)} recs) ✓ loaded from checkpoint")
                return

        t0 = time.perf_counter()
        last_error = None

        new_records = sum(1 for r in chunk if not r.get('_is_overlap', False))
        overlap_records = len(chunk) - new_records

        for attempt in range(PRIMARY_ATTEMPTS):
            try:
                raw = await parse_chunk_async(chunk, PRIMARY_MODEL)
                data = safe_parse(raw)

                save_checkpoint(batch_idx, data)
                dt = time.perf_counter() - t0

                print(f"Batch {batch_idx}/{total_batches} ({new_records} new + {overlap_records} overlap) "
                      f"parsed in {dt:.2f}s - {PRIMARY_MODEL}")
                return

            except Exception as e:
                last_error = e
                if attempt < PRIMARY_ATTEMPTS - 1:
                    await asyncio.sleep(1)
                    continue

        print(f"Batch {batch_idx}/{total_batches} - Primary model failed, trying fallback {FALLBACK_MODEL}")
        try:
            raw = await parse_chunk_async(chunk, FALLBACK_MODEL)
            data = safe_parse(raw)

            save_checkpoint(batch_idx, data)
            dt = time.perf_counter() - t0

            print(f"Batch {batch_idx}/{total_batches} ({new_records} new + {overlap_records} overlap) "
                  f"parsed in {dt:.2f}s - {FALLBACK_MODEL}")

        except Exception as e:
            print(f"Batch {batch_idx}/{total_batches} ✗ failed with both models. Last error: {e}")


async def process_batches_async(records: List[Dict]) -> None:
    """Process records in chunks with overlap for better context at boundaries."""
    total = len(records)

    batches = list(chunk_records_with_overlap(records, CHUNK_SIZE, OVERLAP_SIZE))
    total_batches = len(batches)

    print(f"Starting concurrent batch processing: {total} records in {total_batches} batches")
    print(f"Batch size: {CHUNK_SIZE}, Overlap: {OVERLAP_SIZE} records")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    tasks = []
    for idx, batch in enumerate(batches, start=1):
        task = process_batch_with_fallback(idx, batch, total_batches, semaphore)
        tasks.append(task)

    await asyncio.gather(*tasks)


def merge_with_source_and_write(df_source: pd.DataFrame) -> str:
    """
    Consolidate all checkpoints, merge with source dataframe, and write final output CSV.
    Uses deduplication to handle overlapped records.
    """
    parsed_df = consolidate_checkpoints_with_dedup()
    if not parsed_df.empty and "RecID" in parsed_df.columns:
        parsed_df["RecID"] = parsed_df["RecID"].astype(str)
    else:
        parsed_df = pd.DataFrame(columns=["RecID", "first_name", "middle_name", "last_name"])

    out = df_source.merge(parsed_df, on="RecID", how="left")

    tmp_out = OUTPUT_CSV + ".part"
    out.to_csv(tmp_out, index=False)
    os.replace(tmp_out, OUTPUT_CSV)
    return OUTPUT_CSV


async def main():
    """Main execution function for name parsing pipeline."""
    start_total = time.perf_counter()

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    df["RecID"] = df["RecID"].astype(str)
    records = df[["RecID", "Name"]].to_dict(orient="records")

    await process_batches_async(records)

    out_path = merge_with_source_and_write(df)

    total_elapsed = time.perf_counter() - start_total
    print(f"✔ Finalized → {out_path}")
    print(f"Total elapsed time: {total_elapsed:.2f} seconds")
    print(f"(If some batches are missing, re-run to resume. Existing checkpoints will be reused.)")


if __name__ == "__main__":
    asyncio.run(main())
