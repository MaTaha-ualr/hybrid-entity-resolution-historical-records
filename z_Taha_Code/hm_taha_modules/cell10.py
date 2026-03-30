"""Auto-generated from HM_Taha.ipynb cell 10."""

#Step 7
import pandas as pd

INPUT_CSV = "f_combined_clusters_with_best.csv"
OUT_BY_NAME = "g_grouped_by_name.csv"
OUT_BY_ADDRESS = "g_grouped_by_address.csv"

# ---------- helpers ----------
def recid_key(s: str):
    """
    Turn a RecID like '88.10' or '003.2' into a sortable (major, minor) tuple of ints.
    Non-digits are ignored; missing minor → 0.
    """
    s = str(s).strip()
    if "." in s:
        a, b = s.split(".", 1)
    else:
        a, b = s, "0"
    def to_int(x: str) -> int:
        digits = "".join(ch for ch in str(x) if ch.isdigit())
        return int(digits) if digits else 0
    return (to_int(a), to_int(b))

def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def parse_recids_cell(cell: str):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [x.strip() for x in str(cell).split(";") if x.strip()]

def flatten_links(vals):
    """
    vals: series of "Direct links" cells (strings, possibly ';'-separated).
    Returns a list of unique links (order preserved).
    """
    acc = []
    for v in vals:
        if pd.isna(v) or str(v).strip() == "":
            continue
        parts = [p.strip() for p in str(v).split(";") if p.strip()]
        acc.extend(parts)
    return dedupe_preserve_order(acc)

# ---------- core grouping ----------
def group_by_name(df: pd.DataFrame) -> pd.DataFrame:
    # split and aggregate
    df["RecIDs_list"] = df["RecIDs"].apply(parse_recids_cell)
    grouped = (
        df.groupby("BestName", sort=False)
          .agg({
              "RecIDs_list": "sum",
              "BestAddress": lambda addrs: dedupe_preserve_order(
                  [str(a).strip() for a in addrs if pd.notna(a) and str(a).strip() != ""]
              ),
              "Direct links": flatten_links
          })
          .reset_index()
    )

    # sort RecIDs within groups
    grouped["RecIDs_list"] = grouped["RecIDs_list"].apply(
        lambda lst: sorted([r for r in lst if r], key=recid_key)
    )

    # cluster order by smallest RecID
    grouped["__first_recid_key__"] = grouped["RecIDs_list"].apply(
        lambda lst: recid_key(lst[0]) if lst else (float("inf"), float("inf"))
    )
    grouped = grouped.sort_values("__first_recid_key__", kind="mergesort").reset_index(drop=True)
    grouped.insert(0, "ClusterID", range(len(grouped)))

    # finalize columns
    grouped["RecIDs"] = grouped["RecIDs_list"].apply(lambda lst: ";".join(lst))
    grouped["Addresses"] = grouped["BestAddress"].apply(lambda addrs: "; ".join(addrs))
    grouped["Direct links"] = grouped["Direct links"].apply(lambda links: ";".join(links))

    out = grouped[["ClusterID", "BestName", "RecIDs", "Addresses", "Direct links"]].copy()
    out = out.rename(columns={"BestName": "Name"})
    return out

def group_by_address(df: pd.DataFrame) -> pd.DataFrame:
    # split and aggregate
    df["RecIDs_list"] = df["RecIDs"].apply(parse_recids_cell)
    grouped = (
        df.groupby("BestAddress", sort=False)
          .agg({
              "RecIDs_list": "sum",
              "BestName": lambda names: dedupe_preserve_order(
                  [str(n).strip() for n in names if pd.notna(n) and str(n).strip() != ""]
              ),
              "Direct links": flatten_links
          })
          .reset_index()
    )

    # sort RecIDs within groups
    grouped["RecIDs_list"] = grouped["RecIDs_list"].apply(
        lambda lst: sorted([r for r in lst if r], key=recid_key)
    )

    # cluster order by smallest RecID
    grouped["__first_recid_key__"] = grouped["RecIDs_list"].apply(
        lambda lst: recid_key(lst[0]) if lst else (float("inf"), float("inf"))
    )
    grouped = grouped.sort_values("__first_recid_key__", kind="mergesort").reset_index(drop=True)
    grouped.insert(0, "ClusterID", range(len(grouped)))

    # finalize columns
    grouped["RecIDs"] = grouped["RecIDs_list"].apply(lambda lst: ";".join(lst))
    grouped["Name"] = grouped["BestName"].apply(lambda names: "; ".join(names))
    grouped["Direct links"] = grouped["Direct links"].apply(lambda links: ";".join(links))

    out = grouped[["ClusterID", "BestAddress", "RecIDs", "Name", "Direct links"]].copy()
    out = out.rename(columns={"BestAddress": "Addresses"})
    return out

# ---------- run both in one program ----------
def main():
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    if "Direct links" not in df.columns:
        df["Direct links"] = ""  # ensure column exists

    out_name = group_by_name(df.copy())
    out_name.to_csv(OUT_BY_NAME, index=False)
    print(f"Saved (grouped by BestName, ordered by RecID): {OUT_BY_NAME}")

    out_addr = group_by_address(df.copy())
    out_addr.to_csv(OUT_BY_ADDRESS, index=False)
    print(f"Saved (grouped by BestAddress, ordered by RecID): {OUT_BY_ADDRESS}")

if __name__ == "__main__":
    main()
