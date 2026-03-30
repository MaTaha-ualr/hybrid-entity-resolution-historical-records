"""Auto-generated from HM_Taha.ipynb cell 7."""

import time
import sys
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)


def simple_normalize(text):
    """Normalize text to lowercase and strip whitespace."""
    if pd.isnull(text):
        return ""
    return str(text).lower().strip()


def l2_normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Apply L2 normalization to matrix rows."""
    return normalize(mat, norm="l2", axis=1, copy=False)


def recid_key(s: str):
    """
    Generate numeric key for sorting RecIDs like '88.10' > '88.2' correctly.
    Returns (major, minor) as integers; non-numeric parts become 0.
    """
    s = str(s)
    if "." in s:
        a, b = s.split(".", 1)
    else:
        a, b = s, "0"

    def to_int(x):
        try:
            return int(x)
        except:
            digits = "".join(ch for ch in x if ch.isdigit())
            return int(digits) if digits else 0

    return (to_int(a), to_int(b))


def detect_duplicate_boundary(features: np.ndarray, n_samples: int = 20) -> dict:
    """
    Sample random records and find the gap between duplicates and non-duplicates.
    Returns epsilon values at p85, p90, p95 percentiles.
    """
    np.random.seed(42)
    n_records = features.shape[0]
    sample_size = min(n_samples, n_records)

    nn = NearestNeighbors(n_neighbors=min(11, n_records), metric="cosine")
    nn.fit(features)

    sample_indices = np.random.choice(n_records, sample_size, replace=False)
    close_distances = []

    for idx in sample_indices:
        distances = nn.kneighbors([features[idx]], n_neighbors=11)[0][0][1:]

        gaps = np.diff(distances)
        for i, gap in enumerate(gaps):
            if gap > distances[i] * 0.5 and gap > 0.05:
                close_distances.extend(distances[:i+1])
                break
        else:
            threshold = min(distances[0] * 2, 0.15)
            close_distances.extend(distances[distances <= threshold])

    if not close_distances:
        return None

    close_distances = [d for d in close_distances if d > 1e-9]
    if not close_distances:
        return None

    return {
        85: float(np.percentile(close_distances, 85)),
        90: float(np.percentile(close_distances, 90)),
        95: float(np.percentile(close_distances, 95))
    }


def cluster_embeddings_dbscan(df: pd.DataFrame,
                              eps: float,
                              min_samples: int = 1,
                              metric: str = "cosine") -> pd.DataFrame:
    """
    Perform DBSCAN clustering on all columns except 'RecID'.
    Sort RecIDs within each cluster ascending (numeric).
    Remap cluster IDs so clusters are ordered by the smallest RecID.

    Returns:
        DataFrame with ClusterID and RecIDs columns
    """
    features = df.loc[:, df.columns != "RecID"].values
    rec_ids = df["RecID"].astype(str).reset_index(drop=True)

    db = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric=metric)
    labels = db.fit_predict(features)

    df_lab = pd.DataFrame({"RecID": rec_ids, "cluster_label": labels.astype(int)})

    groups = {}
    for cid, grp in tqdm(df_lab.groupby("cluster_label"), desc="Sorting cluster members", unit="cluster"):
        items = sorted(grp["RecID"].tolist(), key=recid_key)
        groups[int(cid)] = items

    ordered = sorted(groups.items(), key=lambda kv: recid_key(kv[1][0] if kv[1] else "9999999"))
    rows = [{"ClusterID": new_id, "RecIDs": ";".join(items)} for new_id, (_, items) in enumerate(ordered)]
    return pd.DataFrame(rows, columns=["ClusterID", "RecIDs"])


def main():
    """Main execution function for DBSCAN clustering pipeline."""
    start_time = time.time()

    print("Loading parsed names and addresses...")
    names_df = pd.read_csv("c_parsed_names.csv", dtype=str).fillna("")
    names_df["Name"] = names_df[["first_name", "middle_name", "last_name"]].apply(
        lambda parts: " ".join(p for p in parts if p), axis=1
    )

    addr_df = pd.read_csv("c_parsed_addresses.csv", dtype=str).fillna("")
    addr_df["Address"] = addr_df[["house_number", "street_name", "city", "state", "zip"]].apply(
        lambda parts: " ".join(p for p in parts if p), axis=1
    )

    df = pd.merge(
        names_df[["RecID", "Name"]],
        addr_df[["RecID", "Address"]],
        on="RecID",
        how="inner"
    )

    print("Normalizing Name and Address...")
    df["Name_clean"] = df["Name"].apply(simple_normalize)
    df["Address_clean"] = df["Address"].apply(simple_normalize)

    print("Loading embedding model (BAAI/bge-M3)...")
    model = SentenceTransformer("BAAI/bge-M3")

    print("Computing name embeddings...")
    name_embs = model.encode(df["Name_clean"].tolist(), show_progress_bar=True)
    name_embs = l2_normalize_matrix(np.array(name_embs))
    name_emb_df = pd.DataFrame(name_embs, columns=[f"dim_{i}" for i in range(name_embs.shape[1])])
    name_emb_df.insert(0, "RecID", df["RecID"])
    name_emb_df.to_csv("d_bge_M3_name_embeddings.csv", index=False)
    print("→ Saved name embeddings to 'd_bge_M3_name_embeddings.csv'")

    print("Computing address embeddings...")
    addr_embs = model.encode(df["Address_clean"].tolist(), show_progress_bar=True)
    addr_embs = l2_normalize_matrix(np.array(addr_embs))
    addr_emb_df = pd.DataFrame(addr_embs, columns=[f"dim_{i}" for i in range(addr_embs.shape[1])])
    addr_emb_df.insert(0, "RecID", df["RecID"])
    addr_emb_df.to_csv("d_bge_M3_address_embeddings.csv", index=False)
    print("→ Saved address embeddings to 'd_bge_M3_address_embeddings.csv'")

    print("\nCombining embeddings for joint clustering...")
    combined = pd.merge(name_emb_df, addr_emb_df, on="RecID", suffixes=("_name", "_addr"))
    combined_feats = combined.loc[:, combined.columns != "RecID"].values
    combined_feats = l2_normalize_matrix(combined_feats)
    combined.to_csv("d_combined_embeddings.csv", index=False)
    print("→ Combined embeddings saved to 'd_combined_embeddings.csv'")

    standard_eps = {"name": 0.15, "address": 0.15, "combined": 0.08}

    print("\nDetecting epsilon values from duplicate boundaries...")
    suggestions = {}

    for label, features, standard in [
        ("name", name_embs, standard_eps["name"]),
        ("address", addr_embs, standard_eps["address"]),
        ("combined", combined_feats, standard_eps["combined"])
    ]:
        detected = detect_duplicate_boundary(features)
        if detected:
            suggestions[label] = detected
            print(f"  {label:8s}: p85={detected[85]:.4f}, p90={detected[90]:.4f}, p95={detected[95]:.4f}")
        else:
            suggestions[label] = {85: standard, 90: standard, 95: standard}
            print(f"  {label:8s}: Could not detect boundary. Using standard value: {standard:.4f}")

    print("\nChoose epsilon values:")
    print("  [1] p85 - Conservative (fewer false positives)")
    print("  [2] p90 - Balanced (recommended)")
    print("  [3] p95 - Aggressive (more matches)")
    print("  [4] Use standard values (0.15, 0.15, 0.08)")

    choice = input("Selection (1-4) [2]: ").strip() or "2"

    if choice == "1":
        percentile = 85
    elif choice == "2":
        percentile = 90
    elif choice == "3":
        percentile = 95
    else:
        percentile = None

    if percentile:
        chosen_eps = {
            'name': suggestions['name'][percentile],
            'address': suggestions['address'][percentile],
            'combined': suggestions['combined'][percentile]
        }
    else:
        chosen_eps = standard_eps.copy()

    print("\nDBSCAN parameters summary:")
    print(f"  min_samples = 1 (singleton-friendly)")
    print(f"  metric      = 'cosine'")
    print(f"  eps (name)     = {chosen_eps['name']:.4f}")
    print(f"  eps (address)  = {chosen_eps['address']:.4f}")
    print(f"  eps (combined) = {chosen_eps['combined']:.4f}")

    print("\nClustering name embeddings with DBSCAN...")
    name_clusters = cluster_embeddings_dbscan(
        name_emb_df.copy(),
        eps=chosen_eps["name"],
        min_samples=1,
        metric="cosine"
    )
    name_clusters.to_csv("d1_name_clusters.csv", index=False)
    print("→ Name clusters saved to 'd1_name_clusters.csv'")

    print("Clustering address embeddings with DBSCAN...")
    address_clusters = cluster_embeddings_dbscan(
        addr_emb_df.copy(),
        eps=chosen_eps["address"],
        min_samples=1,
        metric="cosine"
    )
    address_clusters.to_csv("d1_address_clusters.csv", index=False)
    print("→ Address clusters saved to 'd1_address_clusters.csv'")

    print("Clustering combined embeddings with DBSCAN...")
    combined_clusters = cluster_embeddings_dbscan(
        combined.copy(),
        eps=chosen_eps["combined"],
        min_samples=1,
        metric="cosine"
    )
    combined_clusters.to_csv("d1_combined_clusters.csv", index=False)
    print("→ Combined clusters saved to 'd1_combined_clusters.csv'")

    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
