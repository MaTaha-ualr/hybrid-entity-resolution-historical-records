"""Auto-generated from HM_Taha.ipynb cell 19."""

#!/usr/bin/env python3
import pandas as pd
import re

def parse_recids(cell: str) -> list[str]:
    """
    Given a RecIDs cell like "[1.5]; [1.11, 1.4, 1.8]; [1.10]; …",
    returns a flat list of RecID strings in order:
      ['1.5','1.11','1.4','1.8','1.10', …]
    """
    parts = cell.split(';')
    recids = []
    for part in parts:
        # remove brackets and surrounding whitespace
        part = part.strip().lstrip('[').rstrip(']')
        if not part:
            continue
        # split on commas
        for rid in part.split(','):
            rid = rid.strip()
            if rid:
                recids.append(rid)
    return recids

def build_linked_pairs(input_csv: str, output_txt: str):
    # 1) Load the merged clusters file
    df = pd.read_csv(input_csv, dtype=str)

    rows = []
    # 2) For each cluster, parse and link adjacent RecIDs
    for cell in df['RecIDs']:
        recids = parse_recids(cell)
        for a, b in zip(recids, recids[1:]):
            rows.append({'recid1': a, 'recid2': b})

    # 3) Save as a two‐column CSV (or .txt)
    out = pd.DataFrame(rows, columns=['recid1','recid2'])
    out.to_csv(output_txt, index=False)

if __name__ == '__main__':
    build_linked_pairs(
        input_csv='i_refined_clusters_merged.csv',
        output_txt='z_LinkedPair_Taha.txt'
    )
