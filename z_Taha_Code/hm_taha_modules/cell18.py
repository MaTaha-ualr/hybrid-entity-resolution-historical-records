"""Auto-generated from HM_Taha.ipynb cell 18."""

#!/usr/bin/env python3
import pandas as pd

def parse_recids(recids_cell: str) -> list[str]:
    """
    Given a cell like "[1.5]; [1.11, 1.4, 1.8]; [1.10]; ...",
    return a flat list of RecID strings.
    """
    recid_lists = recids_cell.split(';')
    recids = []
    for part in recid_lists:
        # strip brackets and whitespace
        part = part.strip().lstrip('[').rstrip(']')
        if not part:
            continue
        # split on commas
        for rid in part.split(','):
            rid = rid.strip()
            if rid:
                recids.append(rid)
    return recids

def recid_key(r: str) -> tuple[int,int]:
    """
    Convert RecID string 'X.Y' into a tuple (X, Y) of ints for proper numeric sorting.
    """
    major, minor = r.split('.', 1)
    return (int(major), int(minor))

def build_refindex(input_csv: str, output_txt: str):
    # Load the merged clusters file
    df = pd.read_csv(input_csv, dtype=str)

    rows = []
    for _, row in df.iterrows():
        all_recids = parse_recids(row['RecIDs'])
        # find the smallest RecID by numeric (major, minor)
        min_recid = min(all_recids, key=recid_key)
        # map each RecID to that min_recid
        for rid in all_recids:
            rows.append({'RefID': rid, 'ClusterID': min_recid})

    out_df = pd.DataFrame(rows, columns=['RefID','ClusterID'])
    out_df.to_csv(output_txt, index=False)

if __name__ == '__main__':
    build_refindex('i_refined_clusters_merged.csv', 'z_LinkIndex_Taha.txt')
