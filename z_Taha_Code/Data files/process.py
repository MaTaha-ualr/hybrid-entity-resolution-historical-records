#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

ENV_FILE_CANDIDATES = (Path("openai.env"), Path(".env"))

MODEL = "gpt-4.1"
BATCH_SIZE = 32
MAX_CONCURRENCY = 12
RETRIES = 3
BACKOFF_S = 1.5


def _log(message: str) -> None:
    print(message)
    # In independent mode, we just print to console.
    # If DWM10_Parms were available, we would also write to its logFile.


def _load_openai_api_key_from_env_files() -> Optional[Path]:
    for candidate in ENV_FILE_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            with candidate.open('r', encoding='utf-8') as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith('#'):
                        continue
                    key, _, value = line.partition('=')
                    if key.strip() == 'OPENAI_API_KEY':
                        cleaned = value.strip().strip('"').strip("'")
                        if cleaned:
                            os.environ.setdefault('OPENAI_API_KEY', cleaned)
                            return candidate
        except Exception:
            continue
    return None


def _ensure_openai_api_key() -> Optional[Path]:
    existing = os.environ.get('OPENAI_API_KEY', '').strip()
    if existing:
        return None
    return _load_openai_api_key_from_env_files()


def chunkify(lst: List[Dict[str, str]], n: int) -> Iterable[List[Dict[str, str]]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


ZIP_LONG_RUN_RE = re.compile(r"\d{8,}")
NUMERIC_DOTTED_RE = re.compile(r"\s+[0-9]*\.[0-9]+")
HYPHENATED_TOKENS_RE = re.compile(r"\s+\S*-\S*")
POBOX_RE = re.compile(r"(?i)\bPO\s*BOX\b")

ZIP5_RE = re.compile(r"\b(\d{5})(?![A-Za-z])\b")
ZIP9_RE = re.compile(r"\b(\d{9})\b")
NOISY5_A23_RE = re.compile(r"\b(\d{2})[A-Za-z](\d{3})\b")
NOISY5_A32_RE = re.compile(r"\b(\d{3})[A-Za-z](\d{2})\b")


def find_zip_span(addr: str, start_idx: int) -> Optional[Tuple[int, int, str]]:
    tail = addr[start_idx:]

    m = ZIP5_RE.search(tail)
    if m:
        s, e = m.span()
        return (start_idx + s, start_idx + e, m.group(1))

    m = NOISY5_A23_RE.search(tail)
    if m:
        s, e = m.span()
        digits = m.group(1) + m.group(2)
        return (start_idx + s, start_idx + e, digits)

    m = NOISY5_A32_RE.search(tail)
    if m:
        s, e = m.span()
        digits = m.group(1) + m.group(2)
        return (start_idx + s, start_idx + e, digits)

    m = ZIP9_RE.search(tail)
    if m:
        s, e = m.span()
        return (start_idx + s, start_idx + s + 5, m.group(1)[:5])

    runs = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\d+", tail)]
    for i, (rs, re_, txt) in enumerate(runs):
        if len(txt) == 4:
            later = next((r for r in runs[i + 1 :] if len(r[2]) >= 6), None)
            if later:
                return (start_idx + rs, start_idx + rs + 4, txt)
    return None


def rule_based_address_truncate(address: str) -> str:
    addr = collapse_spaces(address)

    m_po = POBOX_RE.search(addr)
    if m_po:
        addr = addr[: m_po.start()].strip()

    digit_runs = list(re.finditer(r"\d+", addr))
    if digit_runs:
        search_from = digit_runs[0].end()
    else:
        search_from = 0

    zip_span = find_zip_span(addr, search_from)
    if zip_span is not None:
        _, zip_end, _ = zip_span
        truncated = addr[:zip_end].strip()
    else:
        truncated = addr

    truncated = re.split(r"\s*\(", truncated)[0].strip()
    truncated = re.split(ZIP_LONG_RUN_RE, truncated)[0].strip()
    truncated = re.split(HYPHENATED_TOKENS_RE, truncated)[0].strip()
    truncated = re.split(NUMERIC_DOTTED_RE, truncated)[0].strip()

    return collapse_spaces(truncated)


def rule_parse_record(rec: Dict[str, str], tag: str = "rule") -> Dict[str, str]:
    recid = str(rec.get("RecID", "")).strip()
    concat = str(rec.get("concatenated", "")).strip()

    if not concat:
        return {"RecID": recid, "Name": "", "Address": "", "ParsedBy": tag}

    m = re.search(r"\d", concat)
    if m:
        name = collapse_spaces(concat[: m.start()])
        addr_candidate = collapse_spaces(concat[m.start() :])
    else:
        return {"RecID": recid, "Name": collapse_spaces(concat), "Address": "", "ParsedBy": tag}

    address = rule_based_address_truncate(addr_candidate)
    return {"RecID": recid, "Name": name, "Address": address, "ParsedBy": tag}


def parse_records_rule_based(records: List[Dict[str, str]], tag: str = "rule") -> List[Dict[str, str]]:
    parsed: List[Dict[str, str]] = []
    for rec in tqdm(records, desc="Rule-based parsing"):
        parsed.append(rule_parse_record(rec, tag=tag))
    return parsed


def extract_results_array(json_text: str) -> List[dict]:
    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict) and isinstance(obj.get("results"), list):
            return obj["results"]
    except Exception:
        pass

    m = re.search(r"\[.*\]", json_text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            return arr if isinstance(arr, list) else []
        except Exception:
            return []
    return []


async def async_llm_parse_batch(
    batch: List[Dict[str, str]],
    model: str,
    client,
    sem: asyncio.Semaphore,
) -> List[Dict[str, str]]:
    if not batch:
        return []

    system_prompt = {
        "role": "system",
        "content": (
            "You are a meticulous data extraction service. For each record, split the 'concatenated' string into 'Name' and 'Address'.\n"
            "Output STRICT JSON with a single top-level key 'results' that maps to an array.\n"
            "Each array item MUST be an object with keys: 'RecID', 'Name', 'Address'.\n\n"
            "Rules to follow exactly:\n"
            "1) Name = all text BEFORE the first digit in 'concatenated'.\n"
            "2) Address = starts at the FIRST digit and extends THROUGH the ZIP code. The ZIP heuristics are:\n"
            "   - Prefer a 5-digit run NOT followed by letters; OR\n"
            "   - A 4-digit run followed later by a run of >=6 digits. Use the 4-digit point.\n"
            "3) If 'PO BOX' appears, drop it and everything after.\n"
            "4) EXCLUDE anything after the ZIP: apartment numbers, phone numbers, SSNs, hyphenated tokens, numeric dotted tokens.\n"
            "5) Preserve dotted tokens with letters (e.g., 'St.').\n"
            "6) Trim extra spaces; return clean strings only."
        ),
    }

    user_prompt = {
        "role": "user",
        "content": (
            'Return ONLY JSON like: {"results":[{"RecID":"...","Name":"...","Address":"..."}, ...]}\n'
            + json.dumps(batch, ensure_ascii=False)
        ),
    }

    for attempt in range(1, RETRIES + 1):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[system_prompt, user_prompt],
                )
            raw = resp.choices[0].message.content or ""
            arr = extract_results_array(raw)

            results: List[Dict[str, str]] = []
            for item in arr:
                results.append(
                    {
                        "RecID": collapse_spaces(str(item.get("RecID", ""))),
                        "Name": collapse_spaces(str(item.get("Name", ""))),
                        "Address": collapse_spaces(str(item.get("Address", ""))),
                        "ParsedBy": "llm",
                    }
                )
            return results
        except Exception as exc:
            if attempt == RETRIES:
                _log(
                    f"LLM parsing failed after {RETRIES} attempts for batch starting RecID={batch[0].get('RecID', '?')}: {exc}"
                )
            else:
                wait = BACKOFF_S * (2 ** (attempt - 1))
                _log(
                    f"LLM error attempt {attempt} for batch starting RecID={batch[0].get('RecID', '?')}: {exc}. Retrying in {wait:.1f}s."
                )
                await asyncio.sleep(wait)
    return []


async def parse_records_llm_async(
    records: List[Dict[str, str]],
    model: str,
    batch_size: int,
) -> List[Dict[str, str]]:
    try:
        from openai import AsyncOpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenAI package not available. Install `openai`.") from exc

    env_source = _ensure_openai_api_key()
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Set it in your environment or add it to openai.env (see comments).")
    if env_source:
        _log(f"OPENAI_API_KEY loaded from {env_source}")

    batches = list(chunkify(records, batch_size))
    if not batches:
        return []

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    _log(f"Submitting {len(batches)} batches to LLM (max concurrency {MAX_CONCURRENCY})...")
    tasks = [async_llm_parse_batch(batch, model, client, sem) for batch in batches]
    all_results = await asyncio.gather(*tasks)
    _log("LLM processing complete. Applying fallbacks as needed...")

    parsed: List[Dict[str, str]] = []
    for original_batch, parsed_batch in zip(batches, all_results):
        parsed_map = {row.get("RecID"): row for row in parsed_batch if row.get("RecID")}
        for rec in original_batch:
            rec_id = rec.get("RecID", "")
            if rec_id in parsed_map:
                parsed.append(parsed_map[rec_id])
            else:
                parsed.append(rule_parse_record(rec, tag="fallback_rule"))
    return parsed


def _load_records(path: Path, delimiter: str, has_header: bool) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"Input file '{path}' not found.")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        if has_header:
            try:
                next(reader)
            except StopIteration:
                return records
        for row in reader:
            if not row:
                continue
            rec_id = str(row[0]).strip()
            if not rec_id:
                continue
            tail = [collapse_spaces(col) for col in row[1:] if col and col.strip()]
            concatenated = collapse_spaces(" ".join(tail))
            records.append({"RecID": rec_id, "concatenated": concatenated})
    return records


def _write_results(
    results: List[Dict[str, str]],
    destination: Path,
    delimiter: str,
    include_header: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if include_header:
            writer.writerow(["RecID", "Name", "Address"])
        for row in results:
            writer.writerow(
                [
                    row.get("RecID", ""),
                    row.get("Name", ""),
                    row.get("Address", ""),
                ]
            )
    return destination


def _load_parsed_records(path: Path, delimiter: str, has_header: bool) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"Input file '{path}' not found.")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        if has_header:
            try:
                header = next(reader)
                recid_idx = header.index('RecID') if 'RecID' in header else 0
                name_idx = header.index('Name') if 'Name' in header else 1
                address_idx = header.index('Address') if 'Address' in header else 2
            except StopIteration:
                return records
            except ValueError: # Header not found, assume fixed positions
                recid_idx, name_idx, address_idx = 0, 1, 2
        else:
            recid_idx, name_idx, address_idx = 0, 1, 2 # Assume fixed positions if no header

        for row in reader:
            if not row:
                continue
            rec_id = str(row[recid_idx]).strip() if len(row) > recid_idx else ""
            name = str(row[name_idx]).strip() if len(row) > name_idx else ""
            address = str(row[address_idx]).strip() if len(row) > address_idx else ""
            if rec_id:
                records.append({"RecID": rec_id, "Name": name, "Address": address})
    return records


def compare_parsed_results(
    truth_records: List[Dict[str, str]],
    rule_records: List[Dict[str, str]],
    llm_records: List[Dict[str, str]],
) -> List[str]:
    results_output: List[str] = []
    truth_map = {rec["RecID"]: rec for rec in truth_records}
    rule_map = {rec["RecID"]: rec for rec in rule_records}
    llm_map = {rec["RecID"]: rec for rec in llm_records}

    results_output.append("--- Comparison Report ---")
    results_output.append(f"Truth Records: {len(truth_records)}")
    results_output.append(f"Rule-based Records: {len(rule_records)}")
    results_output.append(f"LLM-based Records: {len(llm_records)}")
    results_output.append("\nDetailed Comparison by RecID:")

    all_rec_ids = sorted(list(set(truth_map.keys()) | set(rule_map.keys()) | set(llm_map.keys())))

    for recid in all_rec_ids:
        truth = truth_map.get(recid, {"Name": "N/A", "Address": "N/A"})
        rule = rule_map.get(recid, {"Name": "N/A", "Address": "N/A"})
        llm = llm_map.get(recid, {"Name": "N/A", "Address": "N/A"})

        results_output.append(f"\nRecID: {recid}")
        results_output.append(f"  Truth Name:    {truth['Name']}")
        results_output.append(f"  Truth Address: {truth['Address']}")
        results_output.append(f"  Rule Name:     {rule['Name']} {'(MATCH)' if truth['Name'] == rule['Name'] else '(MISMATCH)'}")
        results_output.append(f"  Rule Address:  {rule['Address']} {'(MATCH)' if truth['Address'] == rule['Address'] else '(MISMATCH)'}")
        results_output.append(f"  LLM Name:      {llm['Name']} {'(MATCH)' if truth['Name'] == llm['Name'] else '(MISMATCH)'}")
        results_output.append(f"  LLM Address:   {llm['Address']} {'(MATCH)' if truth['Address'] == llm['Address'] else '(MISMATCH)'}")

    return results_output


import argparse

def main():
    _log("--- Starting Independent Parsing and Comparison ---")

    parser = argparse.ArgumentParser(description="Compare parsing results from truth, rule-based, and LLM-based files.")
    parser.add_argument("--truth", required=True, help="Path to the truth file (e.g., Data&Parms/truthABCgoodDQ.txt)")
    parser.add_argument("--rule", required=True, help="Path to the rule-based parsed file (e.g., Data&Parms/S12PX_R1.txt)")
    parser.add_argument("--llm", required=True, help="Path to the LLM-based parsed file (e.g., Taha_ER_Code/b_processed_results_LLM.csv)")
    parser.add_argument("--output", required=True, help="Desired output file name for results (e.g., comparison_results.txt)")
    parser.add_argument("--delimiter", default=",", help="Delimiter for CSV files (default is ',')")
    parser.add_argument("--no-header", action="store_true", help="Specify if input files do NOT have a header row")

    args = parser.parse_args()

    truth_file_path_str = args.truth
    rule_file_path_str = args.rule
    llm_file_path_str = args.llm
    output_file_path_str = args.output
    delimiter = args.delimiter
    has_header = not args.no_header

    try:
        truth_records = _load_parsed_records(Path(truth_file_path_str), delimiter, has_header)
        rule_records = _load_parsed_records(Path(rule_file_path_str), delimiter, has_header)
        llm_records = _load_parsed_records(Path(llm_file_path_str), delimiter, has_header)
    except FileNotFoundError as exc:
        _log(f"Error: {exc}")
        return
    except Exception as exc:
        _log(f"An error occurred while loading files: {exc}")
        return

    comparison_report = compare_parsed_results(truth_records, rule_records, llm_records)

    try:
        output_path = Path(output_file_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for line in comparison_report:
                f.write(line + "\n")
        _log(f"\nComparison results written to {output_path}")
    except Exception as exc:
        _log(f"Error writing comparison results to file: {exc}")

    _log("--- Independent Parsing and Comparison Finished ---")


if __name__ == "__main__":
    main()
