"""Auto-generated from HM_Taha.ipynb cell 3."""

from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _normalize_recid(x: object) -> str:
    """Normalize record ID to uppercase and stripped string."""
    return str(x).strip().upper()


def _normalize_value(val: object) -> str:
    """
    Normalize field values by uppercasing, removing punctuation,
    and collapsing whitespace. Keeps only A-Z, 0-9 and spaces.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_index_case_insensitive(names: List[str], target: str) -> Optional[int]:
    """Find column index by case-insensitive name match."""
    tgt = target.lower()
    for i, n in enumerate(names):
        if str(n).strip().lower() == tgt:
            return i
    return None


def _truth_id_col(fieldnames: List[str]) -> Optional[str]:
    """Identify the truth/group ID column from common naming patterns."""
    candidates = [
        "idtruth", "truthid", "truth_id",
        "clusterid", "cluster_id", "cluster",
        "groupid", "group_id", "group"
    ]
    lower_map = {fn.lower(): fn for fn in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def read_truth_map(truth_path: Path) -> Dict[str, int]:
    """
    Read truth file and build mapping from RecID to truth group ID.
    Returns dictionary mapping normalized RecIDs to integer group IDs.
    """
    truth_map: Dict[str, int] = {}
    with truth_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        tf_fields = reader.fieldnames or []

        recid_col = None
        for cand in ["recid", "RecID"]:
            recid_col = recid_col or (cand if cand in tf_fields else None)
        if not recid_col:
            for fn in tf_fields:
                if fn.lower() == "recid":
                    recid_col = fn
                    break

        id_col = _truth_id_col(tf_fields)
        if not recid_col or not id_col:
            return truth_map

        for row in reader:
            recid_raw = row.get(recid_col, "")
            idtruth_raw = row.get(id_col, "")
            recid = _normalize_recid(recid_raw)
            if not recid:
                continue
            if idtruth_raw is None or str(idtruth_raw).strip() == "":
                continue
            try:
                idtruth = int(str(idtruth_raw).strip())
            except ValueError:
                continue
            truth_map[recid] = idtruth
    return truth_map


def process_files(
    source_path: Path,
    truth_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Process source file using truth mapping to group and relabel records.

    Args:
        source_path: Path to input CSV file
        truth_path: Path to truth mapping file
        output_path: Optional output path (defaults to 'a_concatenated.csv')

    Returns:
        Path to the generated output file
    """
    truth_map = read_truth_map(truth_path)

    with source_path.open("r", newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        rows_raw = list(rdr)

    if not rows_raw:
        raise ValueError("Input file is empty.")

    header = rows_raw[0]
    data_rows = rows_raw[1:]

    recid_idx = _find_index_case_insensitive(header, "recid")
    if recid_idx is None:
        recid_idx = 0

    items = []
    for idx, cells in enumerate(data_rows):
        if len(cells) <= recid_idx:
            recid_val = ""
        else:
            recid_val = cells[recid_idx]
        items.append({
            "__idx": idx,
            "cells": cells,
            "recid": recid_val,
            "__recid_norm__": _normalize_recid(recid_val),
        })

    matched = []
    unmatched = []
    for it in items:
        if it["__recid_norm__"] in truth_map:
            matched.append(it)
        else:
            unmatched.append(it)

    present_idtruths = sorted({truth_map[it["__recid_norm__"]] for it in matched}) if matched else []
    idtruth_remap: Dict[int, int] = {orig: i + 1 for i, orig in enumerate(present_idtruths)}

    per_truth_counters: Dict[int, int] = {}
    relabeled = []
    for it in matched:
        orig_truth = truth_map[it["__recid_norm__"]]
        grp = idtruth_remap[orig_truth]
        per_truth_counters[orig_truth] = per_truth_counters.get(orig_truth, 0) + 1
        k = per_truth_counters[orig_truth]
        it2 = dict(it)
        it2["recid"] = f"{grp}.{k}"
        relabeled.append(it2)

    def parse_new_recid(val: str) -> Tuple[int, int]:
        try:
            g_str, k_str = val.split(".", 1)
            return (int(g_str), int(k_str))
        except Exception:
            return (10**12, 10**12)

    relabeled_sorted = sorted(relabeled, key=lambda r: parse_new_recid(str(r["recid"])))
    final_items = relabeled_sorted + unmatched

    out_rows = []
    for it in final_items:
        cells = it["cells"]
        parts: List[str] = []
        for j, cell in enumerate(cells):
            if j == recid_idx:
                continue
            norm = _normalize_value(cell)
            if norm:
                parts.append(norm)
        concatenated = " ".join(parts)
        out_rows.append({
            "RecID": it["recid"],
            "concatenated": concatenated,
        })

    out_path = output_path if output_path else Path("a_concatenated.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["RecID", "concatenated"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Matched rows: {len(matched)}  |  Unmatched rows: {len(unmatched)}")
    if len(matched) == 0 and truth_map:
        print("Note: Truth map loaded, but no RecIDs matched after normalization.")
    if not truth_map:
        print("Warning: No truth IDs were loaded (missing/unknown truth column names in truth file?).")

    return out_path


def main() -> None:
    """Main entry point for interactive command-line usage."""
    import sys
    try:
        src = input("Enter the input filename (e.g., S1G.txt): ").strip()
        if not src:
            print("No input filename provided.")
            sys.exit(1)
        src_path = Path(src)

        default_truth = "truthABCgoodDQ.txt"
        truth_in = input(f"Enter the truth filename [default: {default_truth}]: ").strip() or default_truth
        truth_path = Path(truth_in)

        if not src_path.exists():
            print(f"Input file not found: {src_path}")
            sys.exit(1)
        if not truth_path.exists():
            print(f"Truth file not found: {truth_path}")
            sys.exit(1)

        out = process_files(src_path, truth_path)
        print(f"Output written to: {out}")

    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
