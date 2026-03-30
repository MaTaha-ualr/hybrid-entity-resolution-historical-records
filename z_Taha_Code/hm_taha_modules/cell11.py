"""Auto-generated from HM_Taha.ipynb cell 11."""

#Just to check if there is overlap or not
#explains where the LLM failed
import pandas as pd
import io

# This part simulates your CSV file.
# In your actual use, you would replace this with:
# file_path = 'd_name_clusters.csv'
# df = pd.read_csv(file_path)

csv_data = "f_combined_clusters_with_best.csv"

# Read the data into a pandas DataFrame
df = pd.read_csv(csv_data)

# --- Main Logic ---

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    cluster_id = row['ClusterID']
    rec_ids_str = row['RecIDs']

    # Split the RecIDs string by the semicolon to get a list of individual IDs
    rec_id_list = rec_ids_str.split(';')

    # Extract the leading digit (the part before the '.') from each RecID
    # A set is used to store them, as it automatically handles duplicates.
    leading_digits = set(rec_id.split('.')[0] for rec_id in rec_id_list)

    # Check if the set contains exactly one unique leading digit
    if len(leading_digits) == 1:
        # If yes, the check passes for this cluster
        print(f"✅ ClusterID {cluster_id}: OK. All RecIDs start with '{list(leading_digits)[0]}'.")
    else:
        # If no, the check fails
        print(f"❌ ClusterID {cluster_id}: FAIL. Mismatched leading digits found: {sorted(list(leading_digits))}.")
