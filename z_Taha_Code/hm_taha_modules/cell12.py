"""Auto-generated from HM_Taha.ipynb cell 12."""

# Step 8 — refined merge with parallel, batch, and async API calls
import os
import pandas as pd
import json
import asyncio
import aiohttp
from itertools import combinations
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Tuple, Dict, Set

# Ensure your OPENAI_API_KEY is exported as an environment variable
# export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration for API optimization
API_CONFIG = {
    "max_parallel_workers": 10,  # Adjust based on your API rate limits
    "batch_size": 5,  # Number of comparisons per batch
    "use_async": True,  # Toggle between async and thread-based parallel (set to False to avoid async issues)
    "use_batch": True,  # Toggle batch processing (set to False if batching fails)
    "rate_limit_delay": 0.1,  # Delay between batches to respect rate limits
}

# --- SINGLE API CALL (Original) ---
def call_o4_mini_same_person(name1: str, name2: str, shared_addresses: list[str]) -> bool:
    """
    Original single API call - kept for compatibility and fallback.
    """
    prompt = f"""
You are an expert data-deduplication specialist. Your task is to determine if two name clusters refer to the EXACT SAME INDIVIDUAL.

**CRITICAL RULE:** Only merge if you are confident these are the SAME PERSON. When in doubt, DO NOT MERGE.

**NEVER MERGE when:**
- First names are completely different (e.g., GLORIA vs TOMMY, JOHN vs MARY, MICHAEL vs SUSAN)
- Names suggest different genders
- Names have no reasonable connection (not nicknames, not variations)

**ONLY MERGE when:**
- Same first name + shared address (likely last name change due to marriage)
- Same first name + same middle name/initial + shared address
- Clear nickname relationship + other strong evidence (e.g., BOB/ROBERT, MIKE/MICHAEL)

**Examples:**

1) Example #1 (MERGE - Same person, maiden/married name)
    Name A: "GLORIA B MIDDLEBROOK"
    Name B: "GLORIA B NEL"
    Shared address: ["10223 LIPSCOMB DR SAN DIEGO CA"]
    Analysis: Same first name (GLORIA), same middle initial (B), shared address
    → {{ "same_person": true }}

2) Example #2 (DO NOT MERGE - Different people)
    Name A: "GLORIA B MIDDLEBROOK"
    Name B: "TOMMY ALAN NOEL"
    Shared address: ["10223 LIPSCOMB DR SAN DIEGO CA"]
    Analysis: Completely different first names (GLORIA vs TOMMY), different genders
    → {{ "same_person": false }}

3) Example #3 (DO NOT MERGE - Family members)
    Name A: "JOHN SMITH"
    Name B: "MARY SMITH"
    Shared address: ["123 MAIN ST"]
    Analysis: Different first names, likely spouses or family
    → {{ "same_person": false }}

4) Example #4 (MERGE - Same person with nickname)
    Name A: "MICHAEL J JONES"
    Name B: "MIKE J JONES"
    Shared address: ["456 OAK AVE"]
    Analysis: MIKE is common nickname for MICHAEL, same middle initial, shared address
    → {{ "same_person": true }}

**Now evaluate:**
Name A: "{name1}"
Name B: "{name2}"
Shared address(es): {json.dumps(shared_addresses)}

Are these the SAME INDIVIDUAL? Return ONLY: {{ "same_person": true/false }}"""

    response = client.chat.completions.create(
        model="o4-mini",  # Using o4-mini as requested
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=1,  # As requested
        max_completion_tokens=16384  # Maximum tokens allowed
    )
    text = response.choices[0].message.content.strip()
    try:
        json_str = text[text.find('{'):text.rfind('}')+1]
        parsed = json.loads(json_str)
        return bool(parsed.get("same_person", False))
    except (json.JSONDecodeError, AttributeError):
        low = text.lower()
        return '"same_person": true' in low

# --- BATCH API CALL ---
def call_o4_mini_batch(pairs_info: List[Tuple[str, str, List[str]]]) -> List[bool]:
    """
    Process multiple pairs in a single API call for efficiency.
    Returns a list of booleans for each pair.
    """
    if not pairs_info:
        return []

    prompt = """You are an expert data-deduplication specialist. For each pair, determine if they are the SAME INDIVIDUAL.

CRITICAL RULES:
- NEVER merge people with completely different first names (GLORIA vs TOMMY = FALSE)
- NEVER merge obvious family members (JOHN vs MARY at same address = FALSE)
- ONLY merge if confident they are the SAME PERSON
- When in doubt, return FALSE

Valid merges:
- Same first name + shared address (even with different last names)
- Clear nicknames (BOB/ROBERT, MIKE/MICHAEL) + other evidence
- Same first + middle initial + shared address

"""

    for i, (name1, name2, addrs) in enumerate(pairs_info, 1):
        prompt += f"""
Pair {i}:
  Name A: {name1}
  Name B: {name2}
  Shared: {json.dumps(addrs)}
"""

    prompt += "\nReturn ONLY a JSON array of booleans: [true/false, true/false, ...]"

    try:
        response = client.chat.completions.create(
            model="o4-mini",  # Using o4-mini as requested
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=1,  # As requested
            max_completion_tokens=16384  # Maximum tokens allowed
        )
        text = response.choices[0].message.content.strip()

        # Extract JSON array
        json_str = text[text.find('['):text.rfind(']')+1]
        results = json.loads(json_str)

        # Ensure we have the right number of results
        if len(results) != len(pairs_info):
            print(f"Warning: Expected {len(pairs_info)} results, got {len(results)}. Using fallback.")
            return [call_o4_mini_same_person(n1, n2, addrs)
                    for n1, n2, addrs in pairs_info]

        return [bool(r) for r in results]
    except Exception as e:
        print(f"Batch processing failed: {e}. Using fallback.")
        return [call_o4_mini_same_person(n1, n2, addrs)
                for n1, n2, addrs in pairs_info]

# --- ASYNC API CALLS ---
async def call_o4_mini_async(session, name1: str, name2: str, shared_addresses: List[str]) -> Tuple[str, str, bool]:
    """
    Async version of the API call using aiohttp.
    Returns (name1, name2, result) tuple.
    """
    prompt = f"""Determine if these names are the SAME INDIVIDUAL:
Name A: {name1}
Name B: {name2}
Shared addresses: {json.dumps(shared_addresses)}

CRITICAL:
- NEVER merge different first names (GLORIA vs TOMMY = false)
- ONLY merge if same person (e.g., GLORIA B SMITH vs GLORIA B JONES = true)
- When in doubt, return false

Return only: {{"same_person": true/false}}"""

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "o4-mini",  # Using o4-mini as requested
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,  # As requested
        "max_completion_tokens": 16384  # Maximum tokens allowed
    }

    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            result = await response.json()
            text = result['choices'][0]['message']['content'].strip()

            try:
                json_str = text[text.find('{'):text.rfind('}')+1]
                parsed = json.loads(json_str)
                return (name1, name2, bool(parsed.get("same_person", False)))
            except:
                return (name1, name2, '"same_person": true' in text.lower())
    except Exception as e:
        print(f"Async call failed for {name1} vs {name2}: {e}")
        return (name1, name2, False)

async def process_pairs_async(pairs_to_check: List[Tuple[int, int, Set[str]]], cluster_info: Dict) -> List[Tuple[int, int, bool]]:
    """
    Process all pairs using async API calls.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        pair_mapping = {}  # Map (name1, name2) to (cid1, cid2)

        for cid1, cid2, shared_addrs in pairs_to_check:
            name1 = cluster_info[cid1]['name']
            name2 = cluster_info[cid2]['name']
            pair_mapping[(name1, name2)] = (cid1, cid2)

            task = call_o4_mini_async(session, name1, name2, list(shared_addrs))
            tasks.append(task)

            # Add small delay to respect rate limits
            if len(tasks) % 10 == 0:
                await asyncio.sleep(API_CONFIG["rate_limit_delay"])

        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async API Calls"):
            name1, name2, should_merge = await coro
            cid1, cid2 = pair_mapping[(name1, name2)]
            results.append((cid1, cid2, should_merge))

        return results

# --- PARALLEL API CALLS (Thread-based) ---
def process_single_pair(args: Tuple[int, int, Dict, Set[str]]) -> Tuple[int, int, bool]:
    """
    Helper function for parallel processing using threads.
    """
    cid1, cid2, cluster_info, shared_addresses = args
    name1 = cluster_info[cid1]['name']
    name2 = cluster_info[cid2]['name']

    result = call_o4_mini_same_person(name1, name2, list(shared_addresses))
    return (cid1, cid2, result)

def process_pairs_parallel(pairs_to_check: List[Tuple[int, int, Set[str]]], cluster_info: Dict) -> List[Tuple[int, int, bool]]:
    """
    Process pairs using ThreadPoolExecutor for parallel API calls.
    """
    # Prepare arguments for parallel processing
    args_list = [(cid1, cid2, cluster_info, shared_addrs)
                 for cid1, cid2, shared_addrs in pairs_to_check]

    results = []
    with ThreadPoolExecutor(max_workers=API_CONFIG["max_parallel_workers"]) as executor:
        futures = [executor.submit(process_single_pair, args) for args in args_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel API Calls"):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                print(f"Parallel processing error: {e}")

    return results

# --- BATCH PROCESSING WITH PARALLEL/ASYNC ---
def process_pairs_batch(pairs_to_check: List[Tuple[int, int, Set[str]]], cluster_info: Dict) -> List[Tuple[int, int, bool]]:
    """
    Process pairs in batches for efficiency.
    """
    results = []
    batch_size = API_CONFIG["batch_size"]

    # Prepare batch data
    batch_data = []
    cid_mapping = []

    for cid1, cid2, shared_addrs in pairs_to_check:
        name1 = cluster_info[cid1]['name']
        name2 = cluster_info[cid2]['name']
        batch_data.append((name1, name2, list(shared_addrs)))
        cid_mapping.append((cid1, cid2))

    # Process in batches
    for i in tqdm(range(0, len(batch_data), batch_size), desc="Batch Processing"):
        batch = batch_data[i:i + batch_size]
        batch_cids = cid_mapping[i:i + batch_size]

        batch_results = call_o4_mini_batch(batch)

        for (cid1, cid2), should_merge in zip(batch_cids, batch_results):
            results.append((cid1, cid2, should_merge))

        # Rate limit delay between batches
        if i + batch_size < len(batch_data):
            time.sleep(API_CONFIG["rate_limit_delay"])

    return results

# --- HELPERS (unchanged from original) ---

def get_name_parts(name: str) -> tuple[str, str, str]:
    """Parses a name string into (first_name, middle_name, last_name), uppercased."""
    parts = name.strip().upper().split()
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        # Assume a single token is a last name
        return "", "", parts[0]
    first_name = parts[0]
    last_name = parts[-1]
    middle_name = " ".join(parts[1:-1])
    return first_name, middle_name, last_name

def middle_names_match(m1: str, m2: str) -> bool:
    """
    Middle names match if:
    - exact match (ignoring '.'),
    - or one is the initial of the other (e.g., 'J' vs 'JOANNE').
    Requires both to be non-empty.
    """
    if not m1 or not m2:
        return False
    m1 = m1.replace(".", "")
    m2 = m2.replace(".", "")
    if m1 == m2:
        return True
    if len(m1) == 1 and m2.startswith(m1):
        return True
    if len(m2) == 1 and m1.startswith(m2):
        return True
    return False

def names_are_compatible(name1_parts: tuple[str, str, str], name2_parts: tuple[str, str, str]) -> bool:
    """
    Check if two name tuples could represent the same person.
    ENHANCED: More conservative to avoid false merges.
    """
    first1, middle1, last1 = name1_parts
    first2, middle2, last2 = name2_parts

    # CRITICAL: Different first names = NOT the same person
    # (unless it's a known nickname relationship)
    if first1 and first2:
        # Check for exact match
        if first1 == first2:
            # Same first name - check other conditions
            pass
        else:
            # Check for common nicknames only
            nicknames = {
                ('MIKE', 'MICHAEL'), ('MICHAEL', 'MIKE'),
                ('BOB', 'ROBERT'), ('ROBERT', 'BOB'),
                ('BILL', 'WILLIAM'), ('WILLIAM', 'BILL'),
                ('JIM', 'JAMES'), ('JAMES', 'JIM'),
                ('DICK', 'RICHARD'), ('RICHARD', 'DICK'),
                ('TOM', 'THOMAS'), ('THOMAS', 'TOM'),
                ('CHRIS', 'CHRISTOPHER'), ('CHRISTOPHER', 'CHRIS'),
                ('LIZ', 'ELIZABETH'), ('ELIZABETH', 'LIZ'),
                ('BETH', 'ELIZABETH'), ('ELIZABETH', 'BETH'),
            }
            if (first1, first2) not in nicknames:
                return False  # Different first names, not nicknames = different people

    # Last names must match (unless one is empty)
    if last1 and last2 and last1 != last2:
        return False

    # Case 1: One name is "MIDDLE LAST" and the other is "FIRST MIDDLE LAST"
    if not first1 and middle1 and last1:
        if first2 and middle2 and last2:
            return middle1 == middle2 and last1 == last2
    elif not first2 and middle2 and last2:
        if first1 and middle1 and last1:
            return middle1 == middle2 and last1 == last2

    # Case 2: First+Last match, middles compatible / optional
    if first1 and first2 and first1 == first2 and last1 == last2:
        if not middle1 or not middle2:
            return True
        if middle1 == middle2:
            return True
        # One middle initial vs the other's full
        if len(middle1) == 1 and middle2.startswith(middle1):
            return True
        if len(middle2) == 1 and middle1.startswith(middle2):
            return True

    return False

# --- ENHANCED MERGE LOGIC ---

def merge_clusters(df_name: pd.DataFrame) -> dict[int, int]:
    """
    Merges clusters using:
    1) Deterministic rules for shared addresses and name compatibility.
    2) Bridge logic: connects A and B with same (first, middle) but different last names.
    3) AI fallback with parallel/batch/async processing for remaining pairs.
    """
    parent = {cid: cid for cid in df_name["ClusterID"].astype(int)}

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # --- Pre-computation Step ---
    print("Pre-computing name parts and address lookups...")
    cluster_info = {}
    for _, row in df_name.iterrows():
        cid = int(row["ClusterID"])
        name = row["Name"]
        first, middle, last = get_name_parts(name)
        addresses = set(addr.strip() for addr in row["Addresses"].split(";") if addr.strip())
        cluster_info[cid] = {
            "first": first, "middle": middle, "last": last,
            "name": name, "addresses": addresses,
            "name_parts": (first, middle, last)
        }

    # Lookups for bridge logic
    fm_name_map = defaultdict(list)   # (first, middle) -> [cids]
    last_name_map = defaultdict(list) # last -> [cids]
    for cid, info in cluster_info.items():
        if info['first'] and info['middle']:
            fm_name_map[(info['first'], info['middle'])].append(cid)
        if info['last']:
            last_name_map[info['last']].append(cid)

    # --- Step 1: Deterministic merges for pairs with shared addresses ---
    print("\nStep 1: Applying deterministic rules for shared addresses...")
    all_cids = list(parent.keys())
    total_pairs = len(all_cids) * (len(all_cids) - 1) // 2

    for cid1, cid2 in tqdm(combinations(all_cids, 2), total=total_pairs, desc="Name Compatibility"):
        if find(cid1) == find(cid2):
            continue

        shared_addresses = cluster_info[cid1]['addresses'].intersection(cluster_info[cid2]['addresses'])
        if not shared_addresses:
            continue

        # Fast path: identical names
        if cluster_info[cid1]['name'].strip().upper() == cluster_info[cid2]['name'].strip().upper():
            union(cid1, cid2)
            continue

        # Deterministic rule: same FIRST + MIDDLE + shared address
        f1, m1, _ = cluster_info[cid1]['name_parts']
        f2, m2, _ = cluster_info[cid2]['name_parts']
        if f1 and (f1 == f2) and middle_names_match(m1, m2):
            union(cid1, cid2)
            continue

        # Existing deterministic rule: names_are_compatible
        if names_are_compatible(cluster_info[cid1]['name_parts'], cluster_info[cid2]['name_parts']):
            union(cid1, cid2)
            continue

    # --- Step 2: Bridge logic ---
    print("\nStep 2: Applying the 'bridge' logic for name/address connections...")
    for (first, middle), cids in tqdm(fm_name_map.items(), desc="Bridge Logic"):
        if len(cids) < 2:
            continue

        for cid_a, cid_b in combinations(cids, 2):
            if find(cid_a) == find(cid_b):
                continue

            info_a = cluster_info[cid_a]
            info_b = cluster_info[cid_b]

            if info_a['last'] == info_b['last']:
                continue

            # Bridge Type 1
            bridged = False
            for cid_c in last_name_map.get(info_b['last'], []):
                if cid_c == cid_a or cid_c == cid_b:
                    continue
                info_c = cluster_info[cid_c]
                if info_c['addresses'].intersection(info_a['addresses']):
                    union(cid_a, cid_b)
                    bridged = True
                    break
            if bridged:
                continue

            # Bridge Type 2
            for cid_c in last_name_map.get(info_a['last'], []):
                if cid_c == cid_a or cid_c == cid_b:
                    continue
                info_c = cluster_info[cid_c]
                if info_c['addresses'].intersection(info_b['addresses']):
                    union(cid_a, cid_b)
                    break

    # --- Step 3: Enhanced AI fallback with parallel/batch/async ---
    print("\nStep 3: Comparing remaining clusters using optimized AI calls...")
    print(f"Configuration: Batch={API_CONFIG['use_batch']}, Async={API_CONFIG['use_async']}, Workers={API_CONFIG['max_parallel_workers']}")

    # Collect pairs that need AI evaluation
    pairs_to_check = []
    for cid1, cid2 in combinations(all_cids, 2):
        if find(cid1) == find(cid2):
            continue

        shared_addresses = cluster_info[cid1]['addresses'].intersection(cluster_info[cid2]['addresses'])
        if not shared_addresses:
            continue

        name1 = cluster_info[cid1]['name']
        name2 = cluster_info[cid2]['name']

        # Skip deterministic cases
        f1, m1, _ = cluster_info[cid1]['name_parts']
        f2, m2, _ = cluster_info[cid2]['name_parts']
        if (name1.strip().upper() == name2.strip().upper() or
            (f1 and (f1 == f2) and middle_names_match(m1, m2)) or
            names_are_compatible(cluster_info[cid1]['name_parts'], cluster_info[cid2]['name_parts'])):
            continue

        pairs_to_check.append((cid1, cid2, shared_addresses))

    print(f"Found {len(pairs_to_check)} pairs requiring AI evaluation")

    if pairs_to_check:
        # Choose processing method based on configuration
        if API_CONFIG["use_batch"]:
            results = process_pairs_batch(pairs_to_check, cluster_info)
        elif API_CONFIG["use_async"]:
            # Run async processing
            results = asyncio.run(process_pairs_async(pairs_to_check, cluster_info))
        else:
            # Use thread-based parallel processing
            results = process_pairs_parallel(pairs_to_check, cluster_info)

        # Apply the merge results
        for cid1, cid2, should_merge in results:
            if should_merge:
                union(cid1, cid2)

        print(f"Completed {len(results)} AI evaluations")

    return {cid: find(cid) for cid in parent}

# --- OUTPUT BUILDERS (unchanged) ---

def rebuild_merged_clusters(
    df_name: pd.DataFrame,
    merged_map: dict[int, int]
) -> pd.DataFrame:
    """
    Collapse each union-find set into:
      – Names = all distinct original names
      – AllRecIDs = all RecIDs from every merged cluster
    Returns a DataFrame with columns ["RootID", "Names", "AllRecIDs"].
    """
    temp: dict[int, dict[str, set | list]] = defaultdict(lambda: {
        "Names": set(),
        "AllRecIDs": []
    })

    for _, row in df_name.iterrows():
        orig_cid = int(row["ClusterID"])
        root = merged_map[orig_cid]
        temp[root]["Names"].add(row["Name"])
        rec_list = [r.strip() for r in row["RecIDs"].split(";") if r.strip()]
        temp[root]["AllRecIDs"].extend(rec_list)

    merged_rows = []
    for root, info in temp.items():
        merged_rows.append({
            "RootID": root,
            "Names": sorted(list(info["Names"])),
            "AllRecIDs": sorted(list(set(info["AllRecIDs"])))
        })
    return pd.DataFrame(merged_rows)

def group_by_address_for_merged(
    df_addr: pd.DataFrame,
    merged_clusters_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Re-groups the final merged clusters by address to produce the final output format.
    """
    recid_to_addr: dict[str, str] = {}
    for _, row in df_addr.iterrows():
        addr = row["Addresses"]
        recs = [r.strip() for r in row["RecIDs"].split(";") if r.strip()]
        for r in recs:
            recid_to_addr[r] = addr

    output_rows = []
    for _, merged_row in merged_clusters_df.iterrows():
        root = int(merged_row["RootID"])
        names = merged_row["Names"]
        all_recs = merged_row["AllRecIDs"]

        addr_to_recs: dict[str, list[str]] = defaultdict(list)
        for rec in all_recs:
            if rec in recid_to_addr:
                a = recid_to_addr[rec]
                addr_to_recs[a].append(rec)

        sorted_addrs = sorted(addr_to_recs.keys())
        bracketed_lists = []
        direct_links = []
        for a in sorted_addrs:
            recs_at_a = sorted(addr_to_recs[a])
            bracketed_lists.append(f"[{', '.join(recs_at_a)}]")
            n = len(recs_at_a)
            direct_links.append(str(n * (n - 1) // 2))

        output_rows.append({
            "OldClusterID": root,
            "Name": "; ".join(names),
            "RecIDs": "; ".join(bracketed_lists),
            "Addresses": "; ".join(sorted_addrs),
            "DirectLink": "; ".join(direct_links)
        })

    result_df = pd.DataFrame(output_rows)
    if not result_df.empty:
        old_ids = sorted(result_df["OldClusterID"].astype(int).tolist())
        id_map = {old: new for new, old in enumerate(old_ids)}
        result_df["ClusterID"] = result_df["OldClusterID"].map(id_map)
        result_df = result_df[["ClusterID", "Name", "RecIDs", "Addresses", "DirectLink"]]
    else:
        return pd.DataFrame(columns=["ClusterID", "Name", "RecIDs", "Addresses", "DirectLink"])

    return result_df

# --- MAIN ---

def main():
    """Main execution function."""
    try:
        # Display configuration
        print("=" * 60)
        print("STEP 8: Enhanced Merge with Optimized API Calls")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Batch Processing: {'Enabled' if API_CONFIG['use_batch'] else 'Disabled'}")
        print(f"  - Async Processing: {'Enabled' if API_CONFIG['use_async'] else 'Disabled'}")
        print(f"  - Max Parallel Workers: {API_CONFIG['max_parallel_workers']}")
        print(f"  - Batch Size: {API_CONFIG['batch_size']}")
        print(f"  - Rate Limit Delay: {API_CONFIG['rate_limit_delay']}s")
        print("=" * 60)

        # Load input files
        df_name = pd.read_csv("g_grouped_by_name.csv")
        df_addr = pd.read_csv("g_grouped_by_address.csv")

        print(f"\nLoaded {len(df_name)} name clusters and {len(df_addr)} address groups")

        # Process merges
        start_time = time.time()
        merged_map = merge_clusters(df_name)
        merge_time = time.time() - start_time

        # Build output
        merged_clusters_df = rebuild_merged_clusters(df_name, merged_map)
        final_df = group_by_address_for_merged(df_addr, merged_clusters_df)

        # Save results
        final_df.to_csv("h_refined_clusters_new.csv", index=False)

        # Report statistics
        original_clusters = len(df_name)
        final_clusters = len(final_df)
        reduction = original_clusters - final_clusters

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Original clusters: {original_clusters}")
        print(f"Final clusters: {final_clusters}")
        print(f"Clusters merged: {reduction}")
        print(f"Reduction: {reduction/original_clusters*100:.1f}%")
        print(f"Processing time: {merge_time:.2f} seconds")
        print(f"\nOutput written to: h_refined_clusters_new.csv")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'g_grouped_by_name.csv' and 'g_grouped_by_address.csv' are in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import aiohttp
    except ImportError:
        print("Installing aiohttp for async operations...")
        os.system("pip install aiohttp")
        print("Please restart the script after installation.")
        exit(1)

    main()
