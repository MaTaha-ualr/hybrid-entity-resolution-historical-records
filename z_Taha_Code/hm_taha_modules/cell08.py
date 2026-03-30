"""Auto-generated from HM_Taha.ipynb cell 8."""

#Step 5 : Ask the LLM to choose the best name and best address (Optimized with Ordered Output)
import sys
import os
import time
import csv
import asyncio
from typing import List, Dict, Tuple, Optional
from collections import Counter

import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI

# ──────────────────────────── Configuration ────────────────────────────
# NOTE: for safety, we do not hardcode a default API key.
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    sys.stderr.write("ERROR: OPENAI_API_KEY is not set in environment.\n")
    sys.exit(1)

# Fixed inputs (no resolver / no alternatives)
NAME_CLUSTERS = "d1_name_clusters.csv"
PARSED_NAMES  = "c_parsed_names.csv"

ADDRESS_CLUSTERS = "d1_address_clusters.csv"
PARSED_ADDR      = "c_parsed_addresses.csv"

CSV_SEP = ","
MODEL_NAME = "gpt-4o-mini"

# Outputs
NAMES_OUT_CSV     = "e_cluster_complete_names.csv"
ADDRESSES_OUT_CSV = "e_cluster_complete_addresses.csv"


# ──────────────────────────── Helpers ────────────────────────────
def atomic_write_csv(df: pd.DataFrame, path: str, sep: str = CSV_SEP) -> None:
    tmp = path + ".part"
    df.to_csv(tmp, index=False, sep=sep, quoting=csv.QUOTE_ALL)
    os.replace(tmp, path)


def dict_dedupe(seq: List[str]) -> List[str]:
    """Order-preserving dedupe."""
    return list(dict.fromkeys(seq))


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison (remove extra spaces, standardize case)."""
    return " ".join(text.strip().upper().split())


def get_min_recid(rec_ids_str: str) -> int:
    """Extract the minimum RecID from a semicolon-separated string of RecIDs."""
    rec_ids = [r.strip() for r in str(rec_ids_str).split(";") if r.strip()]
    numeric_ids = []
    for rid in rec_ids:
        try:
            # Handle different RecID formats (e.g., "R123", "123", etc.)
            # Extract numeric portion
            numeric_part = ''.join(filter(str.isdigit, rid))
            if numeric_part:
                numeric_ids.append(int(numeric_part))
        except (ValueError, AttributeError):
            continue
    return min(numeric_ids) if numeric_ids else float('inf')


def sort_and_reassign_cluster_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataframe by minimum RecID in each cluster and reassign ClusterIDs.
    """
    if df.empty:
        return df

    # Calculate minimum RecID for each cluster
    df['MinRecID'] = df['RecIDs'].apply(get_min_recid)

    # Sort by minimum RecID
    df_sorted = df.sort_values('MinRecID').reset_index(drop=True)

    # Reassign ClusterIDs based on sorted order
    df_sorted['ClusterID'] = range(1, len(df_sorted) + 1)
    df_sorted['ClusterID'] = 'C' + df_sorted['ClusterID'].astype(str).str.zfill(6)

    # Drop the temporary MinRecID column
    df_sorted = df_sorted.drop(columns=['MinRecID'])

    return df_sorted


# ──────────────────────────── Smart Selection Logic ────────────────────────────
def smart_select_name(candidates: List[str], raw_names: List[str]) -> Tuple[str, str]:
    """
    Smart name selection with multiple fallback strategies before LLM.
    Returns (chosen_name, selection_method)
    """
    if not candidates:
        if raw_names:
            chosen = max(raw_names, key=len)
            return chosen, "fallback_raw_longest"
        return "", "no_candidates"

    if len(candidates) == 1:
        return candidates[0], "single_candidate"

    # Normalize candidates for comparison
    normalized_candidates = [normalize_for_comparison(c) for c in candidates]

    # Check if all normalized candidates are identical
    if len(set(normalized_candidates)) == 1:
        # Return the longest original version (preserves better formatting)
        chosen = max(candidates, key=len)
        return chosen, "identical_normalized"

    # Check for clear "most complete" name (contains all parts of others)
    complete_candidate = find_most_complete_name(candidates)
    if complete_candidate:
        return complete_candidate, "most_complete"

    # Check for majority consensus (if one name appears multiple times)
    name_counts = Counter(normalized_candidates)
    if name_counts.most_common(1)[0][1] > 1:
        most_common_normalized = name_counts.most_common(1)[0][0]
        # Find original version of most common
        for orig, norm in zip(candidates, normalized_candidates):
            if norm == most_common_normalized:
                return orig, "majority_consensus"

    # Need LLM decision
    return "", "needs_llm"


def find_most_complete_name(candidates: List[str]) -> Optional[str]:
    """
    Find a name that contains all parts of other names (is most complete).
    Returns None if no clear winner.
    """
    def name_parts(name: str) -> set:
        return set(name.strip().upper().split())

    candidates_parts = [(c, name_parts(c)) for c in candidates]

    for candidate, parts in candidates_parts:
        # Check if this candidate contains all parts of all other candidates
        is_most_complete = True
        for other_candidate, other_parts in candidates_parts:
            if candidate != other_candidate and not other_parts.issubset(parts):
                is_most_complete = False
                break

        if is_most_complete and len(parts) > 1:  # Ensure it's not just a single part
            return candidate

    return None


def smart_select_address(candidates: List[str], raw_variants: List[str]) -> Tuple[str, str]:
    """
    Smart address selection with multiple fallback strategies before LLM.
    Returns (chosen_address, selection_method)
    """
    if not candidates:
        if raw_variants:
            chosen = max(raw_variants, key=len).upper()
            return chosen, "fallback_raw_longest"
        return "", "no_candidates"

    if len(candidates) == 1:
        return candidates[0].upper(), "single_candidate"

    # Normalize candidates for comparison
    normalized_candidates = [normalize_for_comparison(c) for c in candidates]

    # Check if all normalized candidates are identical
    if len(set(normalized_candidates)) == 1:
        chosen = max(candidates, key=len).upper()
        return chosen, "identical_normalized"

    # Check for most complete address (contains most components)
    complete_candidate = find_most_complete_address(candidates)
    if complete_candidate:
        return complete_candidate.upper(), "most_complete"

    # Check for majority consensus
    addr_counts = Counter(normalized_candidates)
    if addr_counts.most_common(1)[0][1] > 1:
        most_common_normalized = addr_counts.most_common(1)[0][0]
        for orig, norm in zip(candidates, normalized_candidates):
            if norm == most_common_normalized:
                return orig.upper(), "majority_consensus"

    # Need LLM decision
    return "", "needs_llm"


def find_most_complete_address(candidates: List[str]) -> Optional[str]:
    """
    Find address with most components (house number, street, city, state, zip).
    """
    def count_address_components(addr: str) -> int:
        # Simple heuristic: count likely address components
        parts = addr.strip().split()
        components = 0

        # Check for house number (starts with digit)
        if parts and parts[0][0].isdigit():
            components += 1

        # Check for state (2-letter word, often near end)
        for part in parts:
            if len(part) == 2 and part.isalpha():
                components += 1
                break

        # Check for ZIP (5 or 9 digits, possibly with dash)
        for part in parts:
            if part.replace('-', '').isdigit() and len(part.replace('-', '')) in [5, 9]:
                components += 1
                break

        # Remaining parts likely street name and city
        remaining_parts = len(parts) - components
        if remaining_parts >= 2:  # At least street and city
            components += 2
        elif remaining_parts == 1:
            components += 1

        return components

    if not candidates:
        return None

    # Find candidate with most components
    scored_candidates = [(c, count_address_components(c)) for c in candidates]
    max_score = max(score for _, score in scored_candidates)

    # Check if there's a clear winner
    winners = [c for c, score in scored_candidates if score == max_score]
    if len(winners) == 1 and max_score > 2:  # At least 3 components
        return winners[0]

    return None


# ──────────────────────────── Async LLM Selectors ────────────────────────────
async def llm_select_single(system_prompt: str, user_prompt: str, candidates: List[str]) -> Optional[str]:
    """
    Ask the LLM to pick exactly one candidate. Returns the chosen string or None.
    """
    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_completion_tokens=80,
        )
        choice = (resp.choices[0].message.content or "").strip()
        return choice if choice else None
    except Exception as e:
        print(f"OpenAI API error: {e}", file=sys.stderr)
        return None


# Cache repeated decisions to avoid duplicate LLM calls
decision_cache: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], str] = {}


async def select_representative_name_async(candidates: List[str], raw_names: List[str], rec_ids: List[str]) -> str:
    """
    Async LLM picks the best combined name, with smart pre-filtering.
    """
    # Try smart selection first
    result, method = smart_select_name(candidates, raw_names)
    if method != "needs_llm":
        return result

    # Need LLM decision
    key = ("name", tuple(candidates), tuple(raw_names))
    if key in decision_cache:
        return decision_cache[key]

    system_prompt = (
        "You are a world-class name-normalization expert. "
        "From the given list of properly combined names (First Middle Last), "
        "select exactly one as the correct, most complete form."
    )
    user_prompt = (
        "Candidates (combined):\n" +
        "\n".join(f"- {n}" for n in candidates) +
        "\n\nRaw variants:\n" +
        "\n".join(f"- {r}" for r in raw_names) +
        "\n\nRespond with ONLY the chosen combined name."
    )

    chosen = await llm_select_single(system_prompt, user_prompt, candidates)
    if chosen and chosen in candidates:
        decision_cache[key] = chosen
        return chosen

    # Fallback
    fallback = max(candidates, key=len)
    decision_cache[key] = fallback
    return fallback


async def select_representative_address_async(candidates: List[str], raw_variants: List[str], rec_ids: List[str]) -> str:
    """
    Async LLM picks the best fully-standardized address, with smart pre-filtering.
    """
    # Try smart selection first
    result, method = smart_select_address(candidates, raw_variants)
    if method != "needs_llm":
        return result

    # Need LLM decision
    key = ("address", tuple(candidates), tuple(raw_variants))
    if key in decision_cache:
        return decision_cache[key]

    system_prompt = (
        "You are a world-class address normalization expert. "
        "From the given list of combined address candidates, "
        "choose exactly one correct, fully-standardized address."
    )
    user_prompt = (
        "Candidates (combined):\n" +
        "\n".join(f"- {a}" for a in candidates) +
        "\n\nRaw variants:\n" +
        "\n".join(f"- {v}" for v in raw_variants) +
        "\n\nRespond with ONLY the chosen combined address."
    )

    chosen = await llm_select_single(system_prompt, user_prompt, candidates)
    if chosen:
        for c in candidates:
            if chosen.strip().upper() == c.strip().upper():
                decision_cache[key] = c.strip().upper()
                return decision_cache[key]

    fallback = max(candidates, key=len).upper()
    decision_cache[key] = fallback
    return fallback


# ──────────────────────────── Async Processing Functions ────────────────────────────
async def process_name_cluster(row, parsed_df: pd.DataFrame) -> Dict:
    """Process a single name cluster asynchronously."""
    recs = [r.strip() for r in str(row.RecIDs).split(";") if r.strip()]
    raw_variants = []
    combined_candidates = []

    for r in recs:
        match = parsed_df[parsed_df.RecID == r]
        if match.empty:
            continue

        raw_name = (match.iloc[0].get("Name") or "").strip()
        if raw_name:
            raw_variants.append(raw_name)

        fn = (match.iloc[0].get("first_name") or "").strip()
        mn = (match.iloc[0].get("middle_name") or "").strip()
        ln = (match.iloc[0].get("last_name") or "").strip()
        if fn and ln:
            combined = f"{fn} {mn} {ln}".replace("  ", " ").strip() if mn else f"{fn} {ln}"
            combined_candidates.append(combined)

    if not combined_candidates:
        if raw_variants:
            chosen = max(dict_dedupe(raw_variants), key=len)
            return {
                "ClusterID": row.ClusterID,  # Keep original for now, will be reassigned
                "CompleteName": chosen,
                "RecIDs": ";".join(recs),
                "Names": ";".join(dict_dedupe(raw_variants))
            }
        return None

    unique_raw = dict_dedupe(raw_variants)
    unique_combined = dict_dedupe(combined_candidates)

    chosen = await select_representative_name_async(unique_combined, unique_raw, recs)
    return {
        "ClusterID":   row.ClusterID,  # Keep original for now, will be reassigned
        "CompleteName": chosen,
        "RecIDs":      ";".join(recs),
        "Names":       ";".join(unique_raw)
    }


async def process_address_cluster(row, parsed_addr_df: pd.DataFrame) -> Dict:
    """Process a single address cluster asynchronously."""
    rec_ids = [r.strip() for r in str(row.RecIDs).split(";") if r.strip()]
    raw_variants = []
    combined_candidates = []

    for rid in rec_ids:
        match = parsed_addr_df[parsed_addr_df.RecID == rid]
        if match.empty:
            continue

        raw_addr = (match.iloc[0].get("Address") or "").strip()
        if raw_addr:
            raw_variants.append(raw_addr)

        hn   = (match.iloc[0].get("house_number") or "").strip()
        sn   = (match.iloc[0].get("street_name") or "").strip()
        city = (match.iloc[0].get("city") or "").strip()
        st   = (match.iloc[0].get("state") or "").strip()
        zp   = (match.iloc[0].get("zip") or "").strip()

        parts = [p for p in [hn, sn, city, st, zp] if p]
        if parts:
            combined = " ".join(parts).strip()
            combined_candidates.append(combined)

    if not combined_candidates:
        if raw_variants:
            chosen = max(dict_dedupe(raw_variants), key=len).upper()
            return {
                "ClusterID":       row.ClusterID,  # Keep original for now, will be reassigned
                "CompleteAddress": chosen,
                "RecIDs":          ";".join(rec_ids),
                "Addresses":       ";".join(dict_dedupe(raw_variants))
            }
        return None

    unique_raw = dict_dedupe(raw_variants)
    unique_combined = dict_dedupe(combined_candidates)

    chosen = await select_representative_address_async(unique_combined, unique_raw, rec_ids)
    return {
        "ClusterID":       row.ClusterID,  # Keep original for now, will be reassigned
        "CompleteAddress": chosen,
        "RecIDs":          ";".join(rec_ids),
        "Addresses":       ";".join(unique_raw)
    }


async def run_names_pipeline() -> None:
    """Run the names pipeline with async processing."""
    name_clusters_path = NAME_CLUSTERS
    parsed_names_path  = PARSED_NAMES

    if not os.path.exists(name_clusters_path) or not os.path.exists(parsed_names_path):
        print("Skipping NAME clusters: required files not found.", file=sys.stderr)
        return

    clusters = pd.read_csv(name_clusters_path, dtype=str, sep=CSV_SEP).fillna("")
    parsed_df = pd.read_csv(parsed_names_path, dtype=str, sep=CSV_SEP).fillna("")

    print(f"\nProcessing NAME clusters from '{name_clusters_path}'…")

    # Create tasks for all clusters
    tasks = [process_name_cluster(row, parsed_df) for _, row in clusters.iterrows()]

    # Process with progress bar and concurrency control
    semaphore = asyncio.Semaphore(10)  # Limit concurrent LLM calls

    async def bounded_task(task):
        async with semaphore:
            return await task

    bounded_tasks = [bounded_task(task) for task in tasks]

    # Process all tasks with progress tracking
    results = []
    for coro in tqdm(asyncio.as_completed(bounded_tasks), total=len(bounded_tasks), desc="Name clusters"):
        result = await coro
        if result:
            results.append(result)

    if results:
        out_df = pd.DataFrame(results)
        # Sort by minimum RecID and reassign ClusterIDs
        out_df = sort_and_reassign_cluster_ids(out_df)
        atomic_write_csv(out_df, NAMES_OUT_CSV, sep=CSV_SEP)
        print(f"→ Names result written to {NAMES_OUT_CSV} (sorted by RecID with reassigned ClusterIDs)")
    else:
        print("No NAME clusters produced output.", file=sys.stderr)


async def run_addresses_pipeline() -> None:
    """Run the addresses pipeline with async processing."""
    addr_clusters_path = ADDRESS_CLUSTERS
    parsed_addr_path   = PARSED_ADDR

    if not os.path.exists(addr_clusters_path) or not os.path.exists(parsed_addr_path):
        print("Skipping ADDRESS clusters: required files not found.", file=sys.stderr)
        return

    clusters_df = pd.read_csv(addr_clusters_path, dtype=str, sep=CSV_SEP).fillna("")
    parsed_addr_df = pd.read_csv(parsed_addr_path, dtype=str, sep=CSV_SEP).fillna("")

    print(f"\nProcessing ADDRESS clusters from '{addr_clusters_path}'…")

    # Create tasks for all clusters
    tasks = [process_address_cluster(row, parsed_addr_df) for _, row in clusters_df.iterrows()]

    # Process with progress bar and concurrency control
    semaphore = asyncio.Semaphore(10)  # Limit concurrent LLM calls

    async def bounded_task(task):
        async with semaphore:
            return await task

    bounded_tasks = [bounded_task(task) for task in tasks]

    # Process all tasks with progress tracking
    results = []
    for coro in tqdm(asyncio.as_completed(bounded_tasks), total=len(bounded_tasks), desc="Address clusters"):
        result = await coro
        if result:
            results.append(result)

    if results:
        out_df = pd.DataFrame(results)
        # Sort by minimum RecID and reassign ClusterIDs
        out_df = sort_and_reassign_cluster_ids(out_df)
        atomic_write_csv(out_df, ADDRESSES_OUT_CSV, sep=CSV_SEP)
        print(f"→ Addresses result written to {ADDRESSES_OUT_CSV} (sorted by RecID with reassigned ClusterIDs)")
    else:
        print("No ADDRESS clusters produced output.", file=sys.stderr)


# ──────────────────────────── Main ────────────────────────────
async def main_async():
    """Main async function."""
    print("Starting combined Name+Address cluster summarization…")
    print("Output will be sorted by RecID with reassigned ClusterIDs based on order.")

    await asyncio.gather(
        run_names_pipeline(),
        run_addresses_pipeline()
    )


def main():
    """Main entry point."""
    t0 = time.time()

    # Handle both regular Python and Jupyter notebook environments
    try:
        loop = asyncio.get_running_loop()
        # We're in a Jupyter notebook - create a task
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main_async())
    except RuntimeError:
        # We're in regular Python
        asyncio.run(main_async())
    except ImportError:
        # nest_asyncio not available, try alternative approach
        print("Note: Running in Jupyter without nest_asyncio. Consider installing: pip install nest-asyncio")
        print("For now, running synchronously...")
        # Fall back to sync version if needed
        asyncio.run(main_async())

    # Print optimization statistics
    total_decisions = len(decision_cache)
    print(f"\nOptimization Statistics:")
    print(f"Total LLM calls made: {total_decisions}")
    print(f"Done in {time.time() - t0:.2f}s.")


# Alternative function for Jupyter notebooks
async def main_jupyter():
    """Use this function directly in Jupyter notebooks."""
    t0 = time.time()

    await main_async()

    # Print optimization statistics
    total_decisions = len(decision_cache)
    print(f"\nOptimization Statistics:")
    print(f"Total LLM calls made: {total_decisions}")
    print(f"Done in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
