"""Auto-generated from HM_Taha.ipynb cell 16."""

#Step 11
import pandas as pd

# --- Load data
households_df = pd.read_csv('j_households_new.csv', sep=',')

# --- Fill missing so string operations don't blow up
households_df['RecIDs_a'] = households_df['RecIDs_a'].fillna('')
households_df['RecIDs_b'] = households_df['RecIDs_b'].fillna('')
households_df['Address']   = households_df['Address'].fillna('')

# --- Build an unordered pair key so (a, b) and (b, a) collapse into one group
pair_keys = households_df.apply(
    lambda r: tuple(sorted((r['ClusterID_a'], r['ClusterID_b']))),
    axis=1
)
pair_groups = households_df.groupby(pair_keys)

movements = []

for (c1, c2), group in pair_groups:
    # only interested in pairs that share at least 2 addresses
    if len(group) < 2:
        continue

    # figure out each cluster's name from the first row
    first = group.iloc[0]
    name1 = first['Name_a'] if first['ClusterID_a'] == c1 else first['Name_b']
    name2 = first['Name_a'] if first['ClusterID_a'] == c2 else first['Name_b']

    row = {
        'ClusterID_a': c1,
        'Name_a':      name1,
        'ClusterID_b': c2,
        'Name_b':      name2,
        'ClusterID_c': '',
        'Name_c':      ''
    }

    direct_links = []
    total1 = total2 = 0

    # for up to four shared-address records, extract RecIDs and compute links
    for idx in range(4):
        rec_key = f'RecIDs_{idx+1}'
        addr_key = f'Address_{idx+1}'

        if idx < len(group):
            r = group.iloc[idx]
            # get this row's RecIDs for cluster1 and cluster2 in the sorted order
            recs1 = r['RecIDs_a'] if r['ClusterID_a'] == c1 else r['RecIDs_b']
            recs2 = r['RecIDs_a'] if r['ClusterID_a'] == c2 else r['RecIDs_b']

            row[rec_key] = f"[{recs1}];[{recs2}]"
            row[addr_key] = r['Address']

            # count direct links = n*(n−1)/2 for each side
            list1 = [x.strip() for x in str(recs1).split(';') if x.strip()]
            list2 = [x.strip() for x in str(recs2).split(';') if x.strip()]
            n1 = len(list1)
            n2 = len(list2)
            direct_links.append(f"[{n1*(n1-1)//2},{n2*(n2-1)//2}]")
            total1 += n1
            total2 += n2
        else:
            row[rec_key] = ''
            row[addr_key] = ''

    # aggregate direct- and indirect-links
    row['Direct_Links']   = ";".join(direct_links)
    row['Indirect_Links'] = f"[{total1*(total1-1)//2}];[{total2*(total2-1)//2}]"

    movements.append(row)

# --- Build final DataFrame
movements_df = pd.DataFrame(movements)
movements_df.insert(0, 'HouseholdMovementsID', range(len(movements_df)))

# --- Reorder to your specification
cols = [
    'HouseholdMovementsID',
    'ClusterID_a', 'Name_a',
    'ClusterID_b', 'Name_b',
    'ClusterID_c', 'Name_c',
    'RecIDs_1', 'Address_1',
    'RecIDs_2', 'Address_2',
    'RecIDs_3', 'Address_3',
    'RecIDs_4', 'Address_4',
    'Direct_Links',
    'Indirect_Links'
]
movements_df = movements_df[cols]

# --- Write out
movements_df.to_csv('j_household_movements_new.csv', index=False)
print(f"✓ Generated j_household_movements_new.csv with {len(movements_df)} rows")
