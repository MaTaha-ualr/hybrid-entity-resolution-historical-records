"""Auto-generated from HM_Taha.ipynb cell 9."""

#Step 6: the standardized results of LLM are put together
#outouts a major file that is relevant - direct links
import os
import sys
import re
import time
from collections import Counter
import pandas as pd

def debug_read_csv(path, sep=','):
    """Read CSV, print and clean column names."""
    try:
        df = pd.read_csv(path, sep=sep, dtype=str)
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Raw columns in {path}: {df.columns.tolist()}")
    df.columns = df.columns.str.strip().str.replace('"', '')
    print(f"Clean columns in {path}: {df.columns.tolist()}")
    return df

def build_map_from_complete(path, complete_col):
    """
    Build a dict mapping each RecID to its exact CompleteName or CompleteAddress.
    Expects columns: complete_col, RecIDs.
    """
    df = debug_read_csv(path)
    if complete_col not in df.columns or 'RecIDs' not in df.columns:
        raise KeyError(f"{path} must contain '{complete_col}' and 'RecIDs'")
    mapping = {}
    for _, row in df.iterrows():
        val = row[complete_col].strip()
        recids = [
            r.strip() for r in re.split(r'[;,]\s*', row['RecIDs'])
            if r.strip()
        ]
        for rid in recids:
            mapping[rid] = val
    return mapping

def combine_clusters(combined_path, names_map, addrs_map):
    """
    Reads combined_clusters.csv and for each ClusterID & its RecIDs:
      • collects CompleteName values,
      • picks the single most frequent name (ties broken by first occurrence),
      • collects CompleteAddress values (deduped & joined by ';'),
      • calculates Direct links as n*(n-1)/2.
    """
    df = debug_read_csv(combined_path)
    if 'ClusterID' not in df.columns or 'RecIDs' not in df.columns:
        raise KeyError(f"{combined_path} must contain 'ClusterID' and 'RecIDs'")

    best_names = []
    best_addrs = []
    direct_links = []

    for _, row in df.iterrows():
        recids = [
            r.strip() for r in re.split(r'[;,]\s*', row['RecIDs'])
            if r.strip()
        ]
        n = len(recids)
        direct_links.append(n * (n - 1) // 2)

        # Gather all names for these RecIDs
        names = [names_map.get(rid, '') for rid in recids]
        names = [n for n in names if n]  # drop empties

        # Choose the single most frequent name
        if names:
            counts = Counter(names)
            # preserve first-occurrence order
            unique_names = list(dict.fromkeys(names))
            max_count = max(counts.values())
            # pick the first name with highest count
            chosen_name = next(n for n in unique_names if counts[n] == max_count)
        else:
            chosen_name = ''

        best_names.append(chosen_name)

        # Addresses: dedupe preserving order
        addrs = [addrs_map.get(rid, '') for rid in recids]
        unique_addrs = list(dict.fromkeys(a for a in addrs if a))
        best_addrs.append(";".join(unique_addrs))

    df['BestName'] = best_names
    df['BestAddress'] = best_addrs
    df['Direct links'] = direct_links
    return df

def main():
    start = time.time()
    names_file     = 'e_cluster_complete_names.csv'
    addresses_file = 'e_cluster_complete_addresses.csv'
    combined_file  = 'd1_combined_clusters.csv'
    output_file    = 'f_combined_clusters_with_best.csv'

    for f in (names_file, addresses_file, combined_file):
        if not os.path.exists(f):
            print(f"Missing file: {f}", file=sys.stderr)
            sys.exit(1)

    names_map = build_map_from_complete(names_file, 'CompleteName')
    addrs_map = build_map_from_complete(addresses_file, 'CompleteAddress')

    result_df = combine_clusters(combined_file, names_map, addrs_map)
    result_df.to_csv(output_file, index=False)
    elapsed = time.time() - start
    print(f"Saved '{output_file}' ({len(result_df)} rows) in {elapsed:.2f}s")

    # Compute and display sum of all Direct links
    total_direct_links = result_df['Direct links'].sum()
    print(f"Sum of all Direct links: {total_direct_links}")

if __name__ == "__main__":
    main()
