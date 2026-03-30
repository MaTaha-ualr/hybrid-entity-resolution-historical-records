"""Auto-generated from HM_Taha.ipynb cell 14."""

import pandas as pd
import itertools
import re

def normalize_and_export(
    input_path: str,
    normalized_path: str = "i_refined_clusters_merged.csv",
    households_path: str = "j_households_new.csv",
) -> None:
    df = pd.read_csv(input_path, dtype=str)
    df = df.drop(columns=['First', 'Middle', 'NameKey', 'AddrSet', 'NameTokens', 'LastNames', 'FirstLikelyGender'], errors='ignore')

    overall_sum = 0
    if 'DirectLink' in df.columns:
        def _sum_directlink(s: str) -> int:
            if pd.isna(s): return 0
            return sum(int(n) for n in re.findall(r'-?\d+', str(s)))
        df['DirectLink_Sum'] = df['DirectLink'].apply(_sum_directlink)
        overall_sum = int(df['DirectLink_Sum'].sum())

    df['ClusterID'] = df['ClusterID'].astype(int)
    df = df.sort_values('ClusterID').reset_index(drop=True)
    df['ClusterID'] = df.index
    df.to_csv(normalized_path, index=False)

    cluster_map = {}
    for _, row in df.iterrows():
        addrs = [a.strip() for a in str(row['Addresses']).split(';')]
        recid_groups = []
        for grp in str(row['RecIDs']).split(';'):
            cleaned = grp.strip().strip('[]')
            recid_groups.append([x.strip() for x in cleaned.split(',') if x.strip()])
        cluster_map[row['ClusterID']] = list(zip(addrs[:len(recid_groups)], recid_groups[:len(addrs)]))

    addr_map = {}
    for _, row in df.iterrows():
        cid = row['ClusterID']
        name = row.get('Name', '')
        for addr, recids in cluster_map.get(cid, []):
            addr_map.setdefault(addr, []).append((cid, name, recids))

    households = []
    h_id = 1
    for address, entries in addr_map.items():
        if len(entries) < 2:
            continue
        for (cid1, name1, rec1), (cid2, name2, rec2) in itertools.combinations(entries, 2):
            a, b = ((cid1, name1, rec1), (cid2, name2, rec2)) if len(rec1) <= len(rec2) else ((cid2, name2, rec2), (cid1, name1, rec1))
            households.append({
                'HouseholdId': h_id,
                'ClusterID_a': a[0],
                'Name_a': a[1],
                'RecIDs_a': ';'.join(a[2]),
                'ClusterID_b': b[0],
                'Name_b': b[1],
                'RecIDs_b': ';'.join(b[2]),
                'Address': address
            })
            h_id += 1

    pd.DataFrame(households).to_csv(households_path, index=False)
    print(f"Wrote {normalized_path} and {households_path}. DirectLink overall sum: {overall_sum}")

if __name__ == "__main__":
    normalize_and_export("i_refined_clusters_merged.csv")
