"""Auto-generated from HM_Taha.ipynb cell 20."""

#!/usr/bin/env python3
import pandas as pd

def recid_key(r: str) -> tuple[int,int]:
    """
    Convert a RecID like 'X.Y' into a tuple (X, Y) of ints for proper sorting.
    """
    major, minor = r.split('.', 1)
    return (int(major), int(minor))


def build_truthfile(input_csv: str, output_txt: str):
    # 1) Load the concatenated data (assuming comma-separated)
    df = pd.read_csv(input_csv, dtype=str).fillna('')

    # 2) Extract the group key (the part before the first dot)
    # Use named args to avoid positional split error
    df['group'] = df['RecID'].str.split(pat='.', n=1).str[0]

    # 3) Compute the smallest RecID per group using numeric sort
    truth_map = (
        df
        .groupby('group')['RecID']
        .agg(lambda recs: min(recs.tolist(), key=recid_key))
        .to_dict()
    )

    # 4) Map each RecID to its group's IdTruth
    df['IdTruth'] = df['group'].map(truth_map)

    # 5) Write out only RecID and IdTruth
    df[['RecID', 'IdTruth']].to_csv(output_txt, index=False)

if __name__ == '__main__':
    build_truthfile(
        input_csv='a_concatenated.csv',
        output_txt='z_Truthfile_Taha.txt'
    )
