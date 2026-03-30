"""Auto-generated from HM_Taha.ipynb cell 17."""

#Step 12
import pandas as pd
import numpy as np
import re
from sklearn.metrics import (
    pair_confusion_matrix,
    adjusted_rand_score,
    rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    normalized_mutual_info_score,
)

# -------- Config --------
INPUT_CSV = "i_refined_clusters_merged.csv"
OUTPUT_CSV = "cluster_eval_metrics.csv"  # Uncomment to save results

# -------- Load & expand labels --------
df_raw = pd.read_csv(INPUT_CSV)

records = []
for _, row in df_raw.iterrows():
    pred = str(row["ClusterID"]).strip()
    recids = re.findall(r"\d+\.\d+", str(row["RecIDs"]))
    for rid in recids:
        true = int(rid.split(".")[0])  # integer part is the true cluster ID
        records.append({"true": true, "pred": pred})

if not records:
    raise ValueError("No records found. Check that 'RecIDs' contains patterns like '123.1; 123.2; ...'")

df = pd.DataFrame(records)

# -------- Pairwise confusion counts --------
tn, fp, fn, tp = pair_confusion_matrix(df["true"], df["pred"]).ravel()

# -------- Pairwise metrics --------
def safe_div(num, den):
    return (num / den) if den else 0.0

pair_precision = safe_div(tp, tp + fp)
pair_recall    = safe_div(tp, tp + fn)
pair_accuracy  = safe_div(tp + tn, tp + tn + fp + fn)
pair_f1        = safe_div(2 * pair_precision * pair_recall, pair_precision + pair_recall)

# -------- Clustering indices --------
ari  = adjusted_rand_score(df["true"], df["pred"])
ri   = rand_score(df["true"], df["pred"])
fmi  = fowlkes_mallows_score(df["true"], df["pred"])
homo = homogeneity_score(df["true"], df["pred"])
compl = completeness_score(df["true"], df["pred"])
vmeasure = v_measure_score(df["true"], df["pred"])
nmi = normalized_mutual_info_score(df["true"], df["pred"])

# Purity: assign each predicted cluster to the most frequent true label
contingency = pd.crosstab(df["true"], df["pred"])
purity = np.sum(np.max(contingency.values, axis=0)) / contingency.values.sum()

# -------- Summarize --------
results = pd.DataFrame({
    "Metric": [
        "Pairwise Precision", "Pairwise Recall", "Pairwise Accuracy", "Pairwise F1",
        "Adjusted Rand Index", "Rand Index", "Fowlkes–Mallows Index",
        "Homogeneity", "Completeness", "V-Measure",
        "Normalized Mutual Information", "Purity"
    ],
    "Value": [
        pair_precision, pair_recall, pair_accuracy, pair_f1,
        ari, ri, fmi,
        homo, compl, vmeasure,
        nmi, purity
    ],
})

print("Pairwise counts (TN, FP, FN, TP):", (tn, fp, fn, tp))
print()
print(results.to_string(index=False))

# Optional: save results
# results.to_csv(OUTPUT_CSV, index=False)
# print(f"\nSaved metrics to {OUTPUT_CSV}")
