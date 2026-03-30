"""Auto-generated from HM_Taha.ipynb cell 6."""

import os
import re
import sys
import json
import time
import glob
import asyncio
import hashlib
from typing import List, Dict, Optional

import pandas as pd
from openai import AsyncOpenAI

# Configuration
INPUT_CSV = "b_processed_results_LLM.csv"
OUTPUT_CSV = "c_parsed_addresses.csv"

CHUNK_SIZE = 64
CONCURRENT_REQUESTS = 10

PRIMARY_MODEL = "o4-mini"
FALLBACK_MODEL = "o3"
PRIMARY_ATTEMPTS = 3

client = AsyncOpenAI()

TMP_DIR = "tmp_parsed_address_batches"
CACHE_DIR = "api_response_cache"
RESUME_FROM_CHECKPOINTS = True

REQUIRED_KEYS = ["RecID", "house_number", "street_name", "city", "state", "zip"]


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def chunk_records(records: List[Dict], size: int):
    """Split records into chunks of specified size."""
    for i in range(0, len(records), size):
        yield records[i:i + size]


def extract_json_array(text: str) -> str:
    """Extract JSON array from potentially malformed response."""
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or start > end:
        raise ValueError("No JSON array delimiters found")
    return text[start:end + 1]


def load_checkpoint(batch_idx: int) -> Optional[List[Dict]]:
    """Load an existing checkpoint for the given batch index if present and valid."""
    path = os.path.join(TMP_DIR, f"batch_{batch_idx:05d}.json")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if all(isinstance(x, dict) for x in data):
                    return data
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_checkpoint(batch_idx: int, data: List[Dict]) -> None:
    """Atomically write the batch results to a checkpoint file."""
    ensure_dir(TMP_DIR)
    path = os.path.join(TMP_DIR, f"batch_{batch_idx:05d}.json")
    tmp_path = path + ".part"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def get_cache_key(chunk: List[Dict], model: str) -> str:
    """Generate cache key for a chunk and model combination."""
    content = json.dumps(chunk, sort_keys=True) + model
    return hashlib.md5(content.encode()).hexdigest()


def load_from_cache(cache_key: str) -> Optional[List[Dict]]:
    """Load cached API response if it exists."""
    ensure_dir(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if isinstance(cached_data, list):
                    return cached_data
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_to_cache(cache_key: str, data: List[Dict]) -> None:
    """Save successful API response to cache."""
    ensure_dir(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    tmp_file = cache_file + ".tmp"

    try:
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, cache_file)
    except Exception as e:
        print(f"Warning: Failed to save cache {cache_key}: {e}", file=sys.stderr)
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def consolidate_checkpoints() -> pd.DataFrame:
    """Read all batch checkpoint files and return a DataFrame."""
    ensure_dir(TMP_DIR)
    files = sorted(glob.glob(os.path.join(TMP_DIR, "batch_*.json")))
    all_rows = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list):
                for item in rows:
                    if isinstance(item, dict):
                        all_rows.append(item)
        except Exception:
            print(f"Warning: Could not read or parse checkpoint file: {fp}", file=sys.stderr)

    if not all_rows:
        return pd.DataFrame(columns=["RecID", "house_number", "street_name", "city", "state", "zip"])
    return pd.DataFrame(all_rows, dtype=str)


SYSTEM_PROMPT = """You are an expert in US address normalization and component extraction for large batches. Your job: STANDARDIZE and SPLIT each address into exactly these fields:

- house_number
- street_name   (street name + type + directionals + unit/building info; for PO Boxes use "PO BOX")
- city
- state         (2-letter USPS)
- zip           (5 digits only)

You must:
1) Analyze the WHOLE BATCH first to find families of near-duplicate addresses (same/near street and same house number) and choose ONE consistent, most-complete canonical form within each family.
2) Fix typos, spacing, fused tokens, and abbreviations CONSISTENTLY across that family using the tables below.
3) DO NOT invent facts: if a component cannot be confidently resolved from batch evidence, leave it as "" (empty string). Never make up a ZIP or city without batch support.
4) Use uppercase for all tokens; trim all extra spaces; remove stray punctuation.

COMPONENT RULES
• house_number:
  - For street addresses: leading digits only (e.g., "621 GARDENIA ST..." → 621).
  - For PO Boxes/APARTADO/etc.: house_number is the box number (e.g., "PO BOX 150706" → 150706).
  - Keep only digits; drop non-digits. If missing or unclear, set "".
• street_name:
  - Include street name + street type (USPS abbreviations) + directionals + unit/building info (APT/BLDG/UNIT/STE/LOT).
  - For PO boxes, set street_name = "PO BOX".
  - For numbered streets, keep ordinals as given (e.g., 118TH).
• city:
  - Use the most frequent correct spelling from the batch family (e.g., RELDDING → REDDING).
• state:
  - 2-letter USPS only. Correct common errors using the table below (TEAS/TEXA/TEXPAS → TX; CALIF./CALI/CALIFORNIA → CA; FLE/FLE./FLORIDYA → FL; ARK./ARKANSAS → AR).
• zip:
  - 5 digits only. If you see a 4-digit or corrupted ZIP, replace it ONLY when the same address family in the batch clearly shows the correct 5-digit ZIP. Otherwise set "".
  - Strip any letter prefixes/suffixes (e.g., H33025 → 33025).
  - Do NOT add ZIP+4.

NORMALIZATION TABLES
• Directionals → abbreviate:
  NORTH→N, SOUTH→S, EAST→E, WEST→W, NORTHEAST→NE, NORTHWEST→NW, SOUTHEAST→SE, SOUTHWEST→SW
• Street types → USPS abbreviations (examples):
  AVENUE/AVEN/AVENU/AVN/AOVE → AVE
  STREET/ST/STR/STRT → ST
  DRIVE/DRIV/DRV/DR → DR
  LANE/LAN/LN → LN
  ROAD/ROD/RD → RD
  BOULEVARD/BLVD → BLVD
  CIRCLE/CIRC/CRCL → CIR
  COURT/CRT/CT → CT
  TERRACE/TERR/TERRACE → TER
  PLACE/PLC/PL → PL
  PARKWAY/PKWAY/PKWY → PKWY
  HIGHWAY/HWY → HWY
  WAY → WAY
• Common PO BOX terms → normalize to "PO BOX":
  APARTADO/APTDO/BUZON/BÚZON/POST OFFICE/POST OFFICE BOX/POST BOX/OFFICE BOX/CALLER → PO BOX
  (e.g., "3127 APARTADO..." → "PO BOX 3127", "150706 CALLER..." → "PO BOX 150706")
• Units/Buildings (append to street_name):
  APT/AP/APTO/APT./UNIT/STE/SUITE/BLDG/BUILDING/LOT/# → normalize to APT/UNIT/STE/BLDG/LOT/# (use APT for AP)
• State cleanup → 2-letter:
  CALI/CALIF./CALIFORNIA → CA
  FLA/FLOR/FLE/FLE./FLORIDA/FLORIDYA → FL
  TEXAS/TEXA/TEAS/TEXPAS → TX
  ARK./ARKANSAS → AR
  CALIF → CA
  C → CA only if clearly used as a truncated state in a family already resolved to CA
• City typos → fix using batch frequency:
  LOS ANGELGES → LOS ANGELES
  MRAMAR → MIRAMAR
  HOUSTEON → HOUSTON
  RELDDING/REDDIN → REDDING
  MCLLEN/MCALLEN → MCALLEN
  WELLINGJTON/WABURWTON/WBURTON → WELLINGTON (use most common family spelling)
  LKEWOOD/YLAKEWOOD → LAKEWOOD
  DESOTO (keep DESOTO; do not split)

FUSED & NOISE TOKENS
• Insert missing spaces between numbers/letters (e.g., 861SOUTH → 861 SOUTH; 10THAVENU → 10TH AVENUE).
• Remove stray single-letter tokens near city/state (e.g., "ELK GROVE B CA" → "ELK GROVE CA").
• Remove leading garbage around ZIP (e.g., H33025 → 33025).
• Drop trailing state words after the 2-letter state (e.g., "... TX TEXAS" → TX).

ZIP CORRECTION POLICY (STRICT)
• If multiple records in the SAME BATCH FAMILY show the same address and a proper 5-digit ZIP, use that ZIP for all members.
• If the zip is corupted by characters, then strip the zip of charcters and output only the digits
• If no clear consensus exists, set zip = "" rather than guessing.

OUTPUT FORMAT (STRICT)
• Return ONLY a JSON object with a single key "records" containing the array of outputs.
• Each object in "records" MUST contain exactly: RecID, house_number, street_name, city, state, zip.
• Uppercase all text fields; zip must be 5 digits or "".

QUALITY CHECKS BEFORE RETURN
1) For any PO BOX: house_number must be digits of the box; street_name must be "PO BOX".
2) Ensure state is 2 letters; if not resolvable, set "".
3) Ensure no hallucinated components (no invented cities/ZIPs not supported by batch evidence)."""

USER_PREFIX = """Process this batch of address records. Use collective batch context to standardize all addresses. Apply the SYSTEM rules and tables strictly.

FOCUSED MICRO-EXAMPLES (covering typical failures):

1) PO BOX variants
Input:
  "3127 APARTADO MISSION TX 78573"
  "52840 BUZON MCALLEN TEXAS 78505"
  "881605 POST OFFICE LOS ANGELES CA 90009"
  "150706 CALLER CAPE CORAL FL 33915"
  "150706 OFFICE BOX CAPE CORAL FL 33915"
Output:
  [{"RecID":"X","house_number":"3127","street_name":"PO BOX","city":"MISSION","state":"TX","zip":"78573"},
   {"RecID":"X","house_number":"52840","street_name":"PO BOX","city":"MCALLEN","state":"TX","zip":"78505"},
   {"RecID":"X","house_number":"881605","street_name":"PO BOX","city":"LOS ANGELES","state":"CA","zip":"90009"},
   {"RecID":"X","house_number":"150706","street_name":"PO BOX","city":"CAPE CORAL","state":"FL","zip":"33915"},
   {"RecID":"X","house_number":"150706","street_name":"PO BOX","city":"CAPE CORAL","state":"FL","zip":"33915"}]

2) State typos and street types
Input:
  "621 GARDENIA ST DESOTO TEAS 75115"
  "21122 JADE BLUFF LANE KATY TEAS 77450"
  "6590 ELMHURST DRV TUJUNGA CALIF. 91042"
Output:
  [{"RecID":"X","house_number":"621","street_name":"GARDENIA ST","city":"DESOTO","state":"TX","zip":"75115"},
   {"RecID":"X","house_number":"21122","street_name":"JADE BLUFF LN","city":"KATY","state":"TX","zip":"77450"},
   {"RecID":"X","house_number":"6590","street_name":"ELMHURST DR","city":"TUJUNGA","state":"CA","zip":"91042"}]

3) Fused tokens and consensus ZIP
Input (same family appears multiple times):
  "861 SOUTH ST REDDING CALIFORNIA 9600"
  "861SOUTH ST REDDING CA 96001"
  "861 SOUTH STRT REDDING CA 96001"
  "861 SOZUTH STRT REDDING CA 96001"
Output (use family consensus ZIP 96001; fix fused and street type):
  [{"RecID":"X","house_number":"861","street_name":"SOUTH ST","city":"REDDING","state":"CA","zip":"96001"},
   {"RecID":"X","house_number":"861","street_name":"SOUTH ST","city":"REDDING","state":"CA","zip":"96001"},
   {"RecID":"X","house_number":"861","street_name":"SOUTH ST","city":"REDDING","state":"CA","zip":"96001"},
   {"RecID":"X","house_number":"861","street_name":"SOUTH ST","city":"REDDING","state":"CA","zip":"96001"}]

4) Units and AP→APT
Input:
  "301 PINE FOREST DR APT 20 MAUMELLE AR 72113"
  "7250 FRANKLIN AVE AP 609 LOS ANGELGES CALIF. 90046"
Output:
  [{"RecID":"X","house_number":"301","street_name":"PINE FOREST DR APT 20","city":"MAUMELLE","state":"AR","zip":"72113"},
   {"RecID":"X","house_number":"7250","street_name":"FRANKLIN AVE APT 609","city":"LOS ANGELES","state":"CA","zip":"90046"}]

5) Street-family canonicalization
Input (same family; correct typos and keep most complete form):
  "12438 JASMINE BROOK LN HOUSTON TX 77089"
  "12438 JASMINE BQROOK LN HOUSTON TX 77089"
  "12438 JASMINE BROOKX LN HOUSTEON TX 77089"
  "12438 JASMINE BROOK LANE HOUSTON TX 77089"
Output (canonical "JASMINE BROOK LN", city from consensus):
  [{"RecID":"X","house_number":"12438","street_name":"JASMINE BROOK LN","city":"HOUSTON","state":"TX","zip":"77089"},
   {"RecID":"X","house_number":"12438","street_name":"JASMINE BROOK LN","city":"HOUSTON","state":"TX","zip":"77089"},
   {"RecID":"X","house_number":"12438","street_name":"JASMINE BROOK LN","city":"HOUSTON","state":"TX","zip":"77089"},
   {"RecID":"X","house_number":"12438","street_name":"JASMINE BROOK LN","city":"HOUSTON","state":"TX","zip":"77089"}]

NOW PROCESS THIS BATCH USING THE SAME RULES:

"""

USER_SUFFIX = """

Return ONLY a JSON object with a single key "records" that maps to the array. Each object must have: RecID, house_number, street_name, city, state, zip. Uppercase all strings; zip is 5 digits or ""."""


async def parse_chunk_async(chunk: List[Dict], model: str, use_streaming: bool = True) -> str:
    """Calls the model with enhanced address normalization prompt asynchronously with optional streaming."""
    cache_key = get_cache_key(chunk, model)
    cached_result = load_from_cache(cache_key)
    if cached_result is not None:
        print(f"  → Cache hit for chunk (model: {model})")
        return json.dumps({"records": cached_result})

    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    user_message = {
        "role": "user",
        "content": USER_PREFIX + json.dumps(chunk, ensure_ascii=False) + USER_SUFFIX,
    }

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "AddressBatch",
            "schema": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "RecID": {"type": "string"},
                                "house_number": {"type": "string"},
                                "street_name": {"type": "string"},
                                "city": {"type": "string"},
                                "state": {"type": "string"},
                                "zip": {"type": "string"},
                            },
                            "required": ["RecID", "house_number", "street_name", "city", "state", "zip"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["records"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    if use_streaming:
        stream = await client.chat.completions.create(
            model=model,
            messages=[system_message, user_message],
            response_format=response_format,
            temperature=1,
            stream=True,
        )

        content_chunks = []
        async for chunk_response in stream:
            if chunk_response.choices[0].delta.content:
                content_chunks.append(chunk_response.choices[0].delta.content)

        result = ''.join(content_chunks)
    else:
        resp = await client.chat.completions.create(
            model=model,
            messages=[system_message, user_message],
            response_format=response_format,
            temperature=1,
        )
        result = resp.choices[0].message.content

    try:
        parsed = safe_parse(result)
        if parsed:
            save_to_cache(cache_key, parsed)
    except:
        pass

    return result


async def smart_retry_on_mismatch(chunk: List[Dict], parsed: List[Dict], model: str, attempt: int) -> List[Dict]:
    """If we got fewer records, try to process missing ones separately."""
    if len(parsed) == len(chunk):
        return parsed

    if len(parsed) < len(chunk):
        parsed_ids = {r.get('RecID', '') for r in parsed if r.get('RecID')}
        chunk_ids = {r['RecID'] for r in chunk}
        missing_ids = chunk_ids - parsed_ids

        if missing_ids:
            missing_records = [r for r in chunk if r['RecID'] in missing_ids]
            print(f"  → Smart retry: Processing {len(missing_records)} missing records separately (attempt {attempt})")

            if len(missing_records) > 10:
                sub_chunks = list(chunk_records(missing_records, 10))
                all_missing_parsed = []

                for sub_chunk in sub_chunks:
                    try:
                        raw = await parse_chunk_async(sub_chunk, model, use_streaming=False)
                        sub_parsed = safe_parse(raw)
                        ok, _ = validate_batch_output(sub_parsed, expected_len=len(sub_chunk))
                        if ok:
                            all_missing_parsed.extend(sub_parsed)
                    except Exception as e:
                        print(f"    → Sub-chunk failed: {e}", file=sys.stderr)

                if all_missing_parsed:
                    return parsed + all_missing_parsed
            else:
                try:
                    raw = await parse_chunk_async(missing_records, model, use_streaming=False)
                    missing_parsed = safe_parse(raw)
                    ok, _ = validate_batch_output(missing_parsed, expected_len=len(missing_records))
                    if ok:
                        return parsed + missing_parsed
                except Exception as e:
                    print(f"  → Smart retry failed: {e}", file=sys.stderr)

    elif len(parsed) > len(chunk):
        chunk_ids = {r['RecID'] for r in chunk}
        filtered = [r for r in parsed if r.get('RecID') in chunk_ids]
        if len(filtered) == len(chunk):
            print(f"  → Filtered extra records: kept {len(filtered)} of {len(parsed)}")
            return filtered

    return parsed


def safe_parse(raw_json_string: str) -> List[Dict]:
    """
    Safely parses a JSON string by trying multiple strategies.
    Does NOT modify values (no normalization here).
    """
    try:
        data = json.loads(raw_json_string)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return value
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(\[.*?\])\s*```", raw_json_string, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            json_array_str = extract_json_array(raw_json_string)
            return json.loads(json_array_str)
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError("Failed to find or parse a valid JSON array in the response.") from e

    raise ValueError("No JSON list found in the parsed object.")


def validate_batch_output(parsed: List[Dict], expected_len: int) -> (bool, str):
    """Minimal structural validation only (no rule-based normalization)."""
    if not isinstance(parsed, list):
        return False, "Model output is not a list"
    if len(parsed) != expected_len:
        return False, f"Length mismatch (got {len(parsed)}, expected {expected_len})"
    for i, o in enumerate(parsed):
        if not isinstance(o, dict):
            return False, f"Item {i} is not an object"
        for k in REQUIRED_KEYS:
            if k not in o:
                return False, f"Item {i} missing key: {k}"
            if not isinstance(o[k], str):
                return False, f"Item {i} key {k} is not a string"
    return True, ""


async def run_with_fallback(chunk: List[Dict], batch_idx: int, total_batches: int) -> List[Dict]:
    """Try primary model with retries, then fallback to secondary model."""
    last_err = ""
    for attempt in range(1, PRIMARY_ATTEMPTS + 1):
        try:
            raw = await parse_chunk_async(chunk, model=PRIMARY_MODEL)
            parsed = safe_parse(raw)

            ok, reason = validate_batch_output(parsed, expected_len=len(chunk))

            if not ok and "Length mismatch" in reason:
                parsed = await smart_retry_on_mismatch(chunk, parsed, PRIMARY_MODEL, attempt)
                ok, reason = validate_batch_output(parsed, expected_len=len(chunk))

            if ok:
                if attempt > 1:
                    print(f"Batch {batch_idx:04d}/{total_batches} ✓ primary recovered on attempt {attempt}")
                return parsed

            last_err = f"validation failed: {reason}"
            print(f"Batch {batch_idx:04d} primary attempt {attempt} failed: {reason}", file=sys.stderr)
        except Exception as e:
            last_err = str(e)
            print(f"Batch {batch_idx:04d} primary attempt {attempt} exception: {e}", file=sys.stderr)
        if attempt < PRIMARY_ATTEMPTS:
            await asyncio.sleep(2 ** (attempt - 1))

    print(f"Batch {batch_idx:04d}/{total_batches} → falling back to {FALLBACK_MODEL} after primary failures ({last_err})")
    raw = await parse_chunk_async(chunk, model=FALLBACK_MODEL, use_streaming=False)
    parsed = safe_parse(raw)

    ok, reason = validate_batch_output(parsed, expected_len=len(chunk))
    if not ok and "Length mismatch" in reason:
        parsed = await smart_retry_on_mismatch(chunk, parsed, FALLBACK_MODEL, 1)
        ok, reason = validate_batch_output(parsed, expected_len=len(chunk))

    if ok:
        print(f"Batch {batch_idx:04d}/{total_batches} ✓ parsed via fallback {FALLBACK_MODEL}")
        return parsed

    raise ValueError(f"Fallback {FALLBACK_MODEL} validation failed: {reason}")


async def process_one_batch(idx: int, chunk: List[Dict], total_batches: int, semaphore: asyncio.Semaphore):
    """Processes a single batch asynchronously, with primary→fallback model logic."""
    async with semaphore:
        if RESUME_FROM_CHECKPOINTS and load_checkpoint(idx) is not None:
            print(f"Batch {idx:04d}/{total_batches} ✓ loaded from checkpoint.")
            return

        t0 = time.perf_counter()
        try:
            parsed_data = await run_with_fallback(chunk, batch_idx=idx, total_batches=total_batches)
            save_checkpoint(idx, parsed_data)
            dt = time.perf_counter() - t0
            print(f"Batch {idx:04d}/{total_batches} ({len(chunk)} recs) ✓ parsed in {dt:.2f}s")
            return
        except Exception as e:
            print(f"Batch {idx:04d}/{total_batches} ✗ failed after primary + fallback: {e}", file=sys.stderr)
            print(f"--- Failing Data for Batch {idx:04d} ---", file=sys.stderr)
            print(json.dumps(chunk, indent=2), file=sys.stderr)
            print(f"--- End of Failing Data ---", file=sys.stderr)


async def process_batches_async(records: List[Dict]):
    """Creates and runs all batch processing tasks concurrently."""
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    batches = list(chunk_records(records, CHUNK_SIZE))
    total_batches = len(batches)
    print(f"Starting async address processing: {len(records)} records in {total_batches} batches.")
    print(f"Cache directory: {CACHE_DIR}")

    tasks = [
        process_one_batch(idx, chunk, total_batches, semaphore)
        for idx, chunk in enumerate(batches, start=1)
    ]
    await asyncio.gather(*tasks)


def merge_with_source_and_write(df_source: pd.DataFrame) -> str:
    """Consolidates checkpoints, merges with the source, and writes the final CSV."""
    parsed_df = consolidate_checkpoints()
    if not parsed_df.empty and "RecID" in parsed_df.columns:
        parsed_df["RecID"] = parsed_df["RecID"].astype(str)
    else:
        parsed_df = pd.DataFrame(columns=["RecID", "house_number", "street_name", "city", "state", "zip"])

    out_df = df_source.merge(parsed_df, on="RecID", how="left")

    for col in ["house_number", "street_name", "city", "state", "zip"]:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df.fillna("", inplace=True)

    tmp_out = OUTPUT_CSV + ".part"
    out_df.to_csv(tmp_out, index=False)
    os.replace(tmp_out, OUTPUT_CSV)
    return OUTPUT_CSV


async def main():
    """Main execution function for address parsing pipeline."""
    start_total = time.perf_counter()

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at '{INPUT_CSV}'", file=sys.stderr)
        return

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    if "RecID" not in df.columns or "Address" not in df.columns:
        print("Error: INPUT_CSV must have columns 'RecID' and 'Address'.", file=sys.stderr)
        return
    df["RecID"] = df["RecID"].astype(str)

    records = df[["RecID", "Address"]].to_dict(orient="records")

    await process_batches_async(records)

    out_path = merge_with_source_and_write(df)

    total_elapsed = time.perf_counter() - start_total
    print("\n--------------------------------------------------")
    print(f"✔ Finalized → {out_path}")
    print(f"Total elapsed time: {total_elapsed:.2f} seconds")
    print("(Re-run the script to resume processing any failed or missing batches.)")


if __name__ == "__main__":
    asyncio.run(main())
