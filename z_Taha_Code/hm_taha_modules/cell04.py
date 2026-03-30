"""Auto-generated from HM_Taha.ipynb cell 4."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from collections import Counter
from typing import Iterable, List, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Configuration
INPUT_CSV  = "a_concatenated.csv"
MODEL      = "gpt-4.1"
BATCH_SIZE = 32
MAX_CONCURRENCY = 12
RETRIES    = 3
BACKOFF_S  = 1.5
ADD_PARSED_BY_TO_MAIN = False


def chunkify(lst: List[dict], n: int) -> Iterable[List[dict]]:
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def collapse_spaces(s: str) -> str:
    """Normalize whitespace to single spaces."""
    return re.sub(r"\s+", " ", s.strip())


# Rule-Based Parsing Components
ZIP_LONG_RUN_RE      = re.compile(r"\d{8,}")
NUMERIC_DOTTED_RE    = re.compile(r"\s+[0-9]*\.[0-9]+")
HYPHENATED_TOKENS_RE = re.compile(r"\s+\S*-\S*")
POBOX_RE             = re.compile(r"(?i)\bPO\s*BOX\b")

ZIP5_RE              = re.compile(r"\b(\d{5})(?![A-Za-z])\b")
ZIP9_RE              = re.compile(r"\b(\d{9})\b")
NOISY5_A23_RE        = re.compile(r"\b(\d{2})[A-Za-z](\d{3})\b")
NOISY5_A32_RE        = re.compile(r"\b(\d{3})[A-Za-z](\d{2})\b")


def find_zip_span(addr: str, start_idx: int) -> Optional[Tuple[int, int, str]]:
    """
    Find a ZIP-like span in address string starting at given index.

    Returns:
        Tuple of (zip_start, zip_end, zip_text) or None if not found.
        Priority: ZIP5 > NOISY5 > ZIP9 > 4-digit fallback rule
    """
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
            later = next((r for r in runs[i+1:] if len(r[2]) >= 6), None)
            if later:
                return (start_idx + rs, start_idx + rs + 4, txt)

    return None


def rule_based_address_truncate(address: str) -> str:
    """Apply PO BOX removal, ZIP detection, truncation, and cleanup."""
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
    """Parse a single record using rule-based approach."""
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
    """Parse all records using rule-based approach."""
    return [rule_parse_record(rec, tag=tag) for rec in tqdm(records, desc="Rule-based parsing")]


def extract_results_array(json_text: str) -> List[dict]:
    """Parse JSON response from LLM, with fallback to regex extraction."""
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


async def async_llm_parse_batch(batch: List[Dict[str, str]], model: str, client, sem: asyncio.Semaphore) -> List[Dict[str, str]]:
    """Process a single batch of records using OpenAI API with retries and concurrency control."""
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

            results = []
            for r in arr:
                results.append({
                    "RecID": str(r.get("RecID", "")),
                    "Name": collapse_spaces(str(r.get("Name", ""))),
                    "Address": collapse_spaces(str(r.get("Address", ""))),
                    "ParsedBy": "llm",
                })
            return results
        except Exception as e:
            if attempt == RETRIES:
                tqdm.write(f"LLM API error (final) on batch starting RecID={batch[0].get('RecID','?')}: {e}")
            else:
                wait = BACKOFF_S * (2 ** (attempt - 1))
                tqdm.write(f"LLM API error (attempt {attempt}) on batch starting RecID={batch[0].get('RecID','?')}: {e}. Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)

    return []


async def parse_records_llm_async(records: List[Dict[str, str]], model: str, batch_size: int) -> List[Dict[str, str]]:
    """Parse records using LLM with concurrent batch processing."""
    try:
        from openai import AsyncOpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI package not available. Install `openai`.") from e

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set in environment for LLM mode.")

    client = AsyncOpenAI()
    out: List[Dict[str, str]] = []
    batches = list(chunkify(records, batch_size))

    print(f"Submitting {len(batches)} batches to LLM concurrently (cap={MAX_CONCURRENCY})...")
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [async_llm_parse_batch(batch, model, client, sem) for batch in batches]

    all_parsed_results = await asyncio.gather(*tasks)
    print("LLM processing complete. Applying fallbacks and finalizing...")

    for original_batch, parsed_results in tqdm(zip(batches, all_parsed_results), total=len(batches), desc="Processing results"):
        parsed_map = {r.get("RecID"): r for r in parsed_results if r.get("RecID") is not None}
        for rec in original_batch:
            rid = rec["RecID"]
            if rid in parsed_map:
                out.append(parsed_map[rid])
            else:
                out.append(rule_parse_record(rec, tag="fallback_rule"))
    return out


# Sanity Check Patterns
ZIP5_END_RE = re.compile(r"\b\d{5}\b$")
PHONE_RE    = re.compile(r"(\b\d{10}\b|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)")
SSN_RE      = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
APT_RE      = re.compile(r"\b(apt|apartment|unit|#)\b", re.IGNORECASE)


def sanity_check_row(row: Dict[str, str]) -> Dict[str, str]:
    """Perform quality checks on parsed row and generate audit metadata."""
    recid   = str(row.get("RecID",""))
    name    = str(row.get("Name",""))
    address = str(row.get("Address",""))
    parsed  = str(row.get("ParsedBy",""))

    issues = []

    if not name: issues.append("empty_name")
    if not address: issues.append("empty_address")
    if re.search(r"\d", name): issues.append("name_has_digits")
    if not ZIP5_END_RE.search(address): issues.append("zip_missing_or_not_at_end")
    if PHONE_RE.search(address): issues.append("phone_in_address")
    if SSN_RE.search(address): issues.append("ssn_in_address")
    if APT_RE.search(address): issues.append("apt_mention")

    if "empty_address" in issues or "ssn_in_address" in issues:
        quality = "FAIL"
    elif issues:
        quality = "WARN"
    else:
        quality = "OK"

    return {
        "RecID": recid,
        "ParsedBy": parsed or "",
        "NameLen": len(name),
        "AddressLen": len(address),
        "Quality": quality,
        "Issues": ";".join(issues) if issues else "",
        "ZipAtEnd": bool(ZIP5_END_RE.search(address)),
        "NameHasDigits": bool(re.search(r"\d", name)),
        "PhoneLike": bool(PHONE_RE.search(address)),
        "SSNLike": bool(SSN_RE.search(address)),
        "AptMention": bool(APT_RE.search(address)),
    }


async def main():
    """Main execution function supporting both rule-based and LLM parsing modes."""
    choice = input("Choose parsing mode (type 'rule' or 'llm'): ").strip().lower()
    if choice not in {"rule", "llm"}:
        print("Invalid choice. Please run again and type 'rule' or 'llm'.")
        return

    OUTPUT_CSV = "b_processed_results_LLM.csv" if choice == "llm" else "b_processed_results.csv"

    try:
        df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found in the current folder.")
        return

    required = {"RecID", "concatenated"}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: Input is missing required columns: {sorted(missing)}")
        return

    records: List[Dict[str, str]] = df[["RecID", "concatenated"]].to_dict(orient="records")

    if choice == "rule":
        results = parse_records_rule_based(records, tag="rule")
    else:
        results = await parse_records_llm_async(records, model=MODEL, batch_size=BATCH_SIZE)

    out_df = pd.DataFrame(results, dtype=str).fillna("")
    out_df.rename(columns={"name": "Name", "address": "Address"}, inplace=True)

    main_cols = ["RecID", "Name", "Address"]
    if ADD_PARSED_BY_TO_MAIN:
        if "ParsedBy" not in out_df.columns:
            out_df["ParsedBy"] = ""
        main_cols.append("ParsedBy")

    out_df = out_df[main_cols]
    out_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

    AUDIT_CSV = OUTPUT_CSV.replace(".csv", "_audit.csv")
    audit_rows = [sanity_check_row(r) for r in tqdm(results, desc="Sanity checks")]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(AUDIT_CSV, index=False, quoting=csv.QUOTE_ALL)

    print("\n=== Parse Provenance ===")
    by_source = audit_df["ParsedBy"].value_counts(dropna=False).to_dict()
    for k in ("llm", "fallback_rule", "rule", ""):
        if k in by_source and by_source[k] > 0:
            label = k if k else "(unknown)"
            print(f"{label:>14}: {by_source[k]}")

    print("\n=== Quality Summary ===")
    by_quality = audit_df["Quality"].value_counts(dropna=False).to_dict()
    for k in ("OK", "WARN", "FAIL"):
        if k in by_quality:
            print(f"{k:>5}: {by_quality[k]}")

    common_issues = Counter(";".join([i for i in audit_df["Issues"] if i]).split(";"))
    if common_issues:
        print("\nTop issues:")
        for issue, cnt in common_issues.most_common(10):
            if issue:
                print(f" - {issue}: {cnt}")

    print(f"\n✔ Execution complete. Saved: {OUTPUT_CSV}")
    print(f"✔ Audit saved: {AUDIT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())
