"""Auto-generated from HM_Taha.ipynb cell 13."""

#Code needs to be streamlined
#Step 9 - Complete Async Entity Resolution with Enhanced Reasoning and Unified Numbering
import os
import re
import json
import difflib
import itertools
import pandas as pd
import asyncio
import aiohttp
from typing import List, Dict, Set, Tuple
import time

################################################################################
# LLM ADAPTER (OpenAI ONLY — async version)                                    #
################################################################################

class LLMDecision:
    def __init__(self, merge: bool, confidence: float, reasons: List[str], detailed_analysis: str = ""):
        self.merge = bool(merge)
        self.confidence = float(confidence)
        self.reasons = list(reasons)
        self.detailed_analysis = detailed_analysis

    def __repr__(self):
        return f"LLMDecision(merge={self.merge}, confidence={self.confidence:.2f})"


class AsyncOpenAIAdapter:
    """Async version using aiohttp for parallel API calls."""
    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY", max_concurrent: int = 5):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is not set. Please export it and re-run.")

        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def decide_batch(self, system: str, prompts: List[Tuple[str, str]]) -> List[Tuple[str, LLMDecision]]:
        """Process multiple prompts concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_single(prompt_data):
            pair_key, user_prompt = prompt_data
            async with semaphore:
                try:
                    decision = await self._single_request(system, user_prompt)
                    return (pair_key, decision)
                except Exception as e:
                    print(f"Error processing {pair_key}: {e}")
                    return (pair_key, LLMDecision(False, 0.0, [f"API Error: {e}"]))

        tasks = [process_single(prompt_data) for prompt_data in prompts]
        results = await asyncio.gather(*tasks)
        return results

    async def _single_request(self, system: str, user_prompt: str) -> LLMDecision:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")

                result = await response.json()
                content = result["choices"][0]["message"]["content"] or "{}"
                data = json.loads(content)

                reasons = data.get("reasons", [])
                if isinstance(reasons, str):
                    reasons = [reasons]

                return LLMDecision(
                    merge=bool(data.get("merge", False)),
                    confidence=float(data.get("confidence", 0.5)),
                    reasons=list(reasons),
                    detailed_analysis=data.get("detailed_analysis", "")
                )

##############################
# Utilities for names/addresses/features
##############################

def first_middle_variants(name_str: str) -> Tuple[List[str], List[str]]:
    variants = [v.strip() for v in name_str.split(';') if v.strip()]
    first_names, keys = [], []
    for v in variants:
        parts = v.split()
        first = parts[0].upper() if parts else ""
        middle = parts[1].upper() if len(parts) > 1 else ""
        first_names.append(first)
        keys.append(f"{first} {middle}".strip())
    return first_names, keys

def last_names(name_str: str) -> Set[str]:
    out = set()
    for variant in name_str.split(";"):
        variant = variant.strip()
        parts = variant.split()
        if len(parts) >= 2:
            out.add(parts[-1].upper())
    return out

def make_addr_set(address_str: str) -> Set[str]:
    return {addr.strip().upper() for addr in address_str.split(";") if addr.strip()}

def make_name_tokens(name_str: str) -> Set[str]:
    tokens = set()
    for variant in name_str.split(";"):
        for word in re.findall(r"\b\w+\b", variant):
            w = word.upper()
            if len(w) > 1:
                tokens.add(w)
    return tokens

def looks_like_po_box(addr: str) -> bool:
    addr_u = addr.upper()
    return any(k in addr_u for k in [" PO BOX", " P.O.", "POST OFFICE", "BOX "])

def address_similarity_score(addr1: str, addr2: str) -> float:
    """Calculate similarity between two addresses considering various factors"""
    addr1_clean = re.sub(r'[^\w\s]', '', addr1.upper())
    addr2_clean = re.sub(r'[^\w\s]', '', addr2.upper())

    base_sim = difflib.SequenceMatcher(None, addr1_clean, addr2_clean).ratio()

    nums1 = set(re.findall(r'\b\d+\b', addr1_clean))
    nums2 = set(re.findall(r'\b\d+\b', addr2_clean))

    common_words = {'ST', 'STREET', 'AVE', 'AVENUE', 'RD', 'ROAD', 'DR', 'DRIVE',
                   'LN', 'LANE', 'CT', 'COURT', 'PL', 'PLACE', 'BLVD', 'BOULEVARD',
                   'WAY', 'APT', 'APARTMENT', 'UNIT', 'STE', 'SUITE'}
    words1 = set(addr1_clean.split()) - common_words
    words2 = set(addr2_clean.split()) - common_words

    if nums1 and nums2:
        if nums1 & nums2:
            base_sim += 0.2

    if words1 and words2:
        word_overlap = len(words1 & words2) / max(len(words1), len(words2))
        base_sim += word_overlap * 0.3

    return min(base_sim, 1.0)

def find_similar_addresses(addrs_a: Set[str], addrs_b: Set[str]) -> List[Tuple[str, str, float]]:
    """Find similar addresses between two sets with similarity scores"""
    similar = []
    for a1 in addrs_a:
        for a2 in addrs_b:
            if not (looks_like_po_box(a1) and looks_like_po_box(a2)):
                sim = address_similarity_score(a1, a2)
                if sim > 0.7:
                    similar.append((a1, a2, sim))
    return sorted(similar, key=lambda x: x[2], reverse=True)

#####################################
# Data enrichment & candidate pairs
#####################################

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df[["First", "Middle"]] = df["Name"].apply(
        lambda x: pd.Series(
            first_middle_variants(x)[1][0].split()[:2]
            if first_middle_variants(x)[1]
            else ["", ""]
        )
    )
    df["NameKey"] = (df["First"] + " " + df["Middle"]).str.strip()
    df["AddrSet"] = df["Addresses"].apply(make_addr_set)
    df["NameTokens"] = df["Name"].apply(make_name_tokens)
    df["LastNames"] = df["Name"].apply(last_names)
    return df

def build_address_index(df: pd.DataFrame) -> Dict[str, Set[int]]:
    idx: Dict[str, Set[int]] = {}
    for cid, adds in df[["ClusterID", "AddrSet"]].values:
        for a in adds:
            idx.setdefault(a, set()).add(cid)
    return idx

def find_candidate_pairs(df: pd.DataFrame, skip: Set[int]) -> List[Tuple[str, int, int]]:
    key_map: Dict[str, List[int]] = {}
    for _, row in df.iterrows():
        cid = row["ClusterID"]
        _, keys = first_middle_variants(row["Name"])
        for k in keys:
            if k:
                key_map.setdefault(k, []).append(cid)

    pairs = []
    for key, clist in key_map.items():
        uniq = sorted({c for c in clist if c not in skip})
        for a, b in itertools.combinations(uniq, 2):
            pairs.append((key, a, b))
    return pairs

################################
# Merge operation
################################

def merge_clusters(df: pd.DataFrame, merge_ids: List[int],
                  new_cluster_id: int, name_override: str = None) -> pd.DataFrame:
    to_merge = df[df["ClusterID"].isin(merge_ids)].copy()

    addr_lists: Dict[int, List[str]] = {}
    recid_lists: Dict[int, List[List[str]]] = {}

    for _, row in to_merge.iterrows():
        cid = int(row["ClusterID"])
        adds = [a.strip() for a in row["Addresses"].split(";")]
        addr_lists[cid] = adds

        groups: List[List[str]] = []
        for grp in str(row["RecIDs"]).split(";"):
            core = grp.strip().strip("[]")
            ids = [x.strip() for x in core.split(",") if x.strip()]
            groups.append(ids)
        recid_lists[cid] = groups

        if len(adds) != len(groups):
            raise ValueError(f"Cluster {cid} has {len(adds)} addresses but {len(groups)} recid-groups")

    combined_addrs: List[str] = []
    seen = set()
    for cid in merge_ids:
        for a in addr_lists[cid]:
            if a not in seen:
                combined_addrs.append(a)
                seen.add(a)

    combined_groups: List[List[str]] = []
    for addr in combined_addrs:
        grp: List[str] = []
        for cid in merge_ids:
            if addr in addr_lists[cid]:
                idx = addr_lists[cid].index(addr)
                grp.extend(recid_lists[cid][idx])
        combined_groups.append(grp)

    recid_str = "; ".join(f"[{', '.join(g)}]" for g in combined_groups)
    link_str = "; ".join(str(len(g) * (len(g) - 1) // 2) for g in combined_groups)

    if name_override:
        merged_name = name_override
    else:
        all_names = []
        for _, row in to_merge.iterrows():
            for v in str(row["Name"]).split(";"):
                v = v.strip()
                if v and v not in all_names:
                    all_names.append(v)
        merged_name = "; ".join(all_names)

    rest = df[~df["ClusterID"].isin(merge_ids)].copy()
    new_row = {
        "ClusterID": int(new_cluster_id),
        "Name": merged_name,
        "RecIDs": recid_str,
        "Addresses": "; ".join(combined_addrs),
        "DirectLink": link_str
    }
    out = pd.concat([rest, pd.DataFrame([new_row])], ignore_index=True)
    out["ClusterID"] = out["ClusterID"].astype(int)
    out = out.sort_values("ClusterID").reset_index(drop=True)
    return out[["ClusterID", "Name", "RecIDs", "Addresses", "DirectLink"]]

#############################
# Pair features & associates
#############################

def jaro_like(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance."""
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        ca = a[i-1]
        for j in range(1, n+1):
            cb = b[j-1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # substitute
            )
    return dp[m][n]

def is_adjacent_transposition(a: str, b: str) -> bool:
    """True if b == a with exactly one adjacent character swap."""
    if len(a) != len(b):
        return False
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diffs) != 2:
        return False
    (i, ax, ay), (j, bx, by) = diffs
    return j == i+1 and ax == by and ay == bx

def last_name_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    best = 0.0
    for la in set_a:
        for lb in set_b:
            best = max(best, jaro_like(la, lb))
    return best

def min_edit_info(set_a: Set[str], set_b: Set[str]) -> dict:
    """Compute best (min) edit distance and whether it's an adjacent transposition."""
    best = {
        "min_distance": None,
        "pair": ["", ""],
        "adjacent_transposition": False
    }
    if not set_a or not set_b:
        return best
    for la in set_a:
        for lb in set_b:
            d = levenshtein(la, lb)
            adj = is_adjacent_transposition(la, lb)
            if (best["min_distance"] is None) or (d < best["min_distance"]) or (d == best["min_distance"] and adj):
                best["min_distance"] = d
                best["pair"] = [la, lb]
                best["adjacent_transposition"] = adj
    return best

def summarize_associates(a_id: int, b_id: int, df: pd.DataFrame, addr_index: Dict[str, Set[int]]) -> dict:
    a_row = df[df["ClusterID"] == a_id].iloc[0]
    b_row = df[df["ClusterID"] == b_id].iloc[0]
    a_addrs = a_row["AddrSet"]
    b_addrs = b_row["AddrSet"]

    associates_a = set()
    associates_b = set()
    for ad in a_addrs:
        associates_a |= (addr_index.get(ad, set()) - {a_id, b_id})
    for ad in b_addrs:
        associates_b |= (addr_index.get(ad, set()) - {a_id, b_id})

    bridge_by_address = len(associates_a & associates_b) > 0

    ln_a = a_row["LastNames"]
    ln_b = b_row["LastNames"]

    assoc_rows: Dict[int, pd.Series] = {}
    for cid in associates_a | associates_b:
        assoc_rows[cid] = df[df["ClusterID"] == cid].iloc[0]

    def addr_overlap_count(s: Set[str], t: Set[str]) -> Tuple[int, List[str]]:
        inter = sorted(list(s & t))
        return (len(inter), inter[:3])

    # surname-bridge from A->B and B->A
    bridge_from_a = []
    for cid in associates_a:
        r = assoc_rows[cid]
        ln_info = min_edit_info(r["LastNames"], ln_b)
        if ln_info["min_distance"] is None:
            continue
        very_close = (ln_info["min_distance"] is not None and ln_info["min_distance"] <= 1) or ln_info["adjacent_transposition"]
        if very_close:
            cnt, shared = addr_overlap_count(r["AddrSet"], a_addrs)
            bridge_from_a.append({
                "cid": int(cid),
                "name": r["Name"],
                "associate_last_names": sorted(list(r["LastNames"])),
                "best_pair": ln_info["pair"],
                "min_distance": ln_info["min_distance"],
                "adjacent_transposition": ln_info["adjacent_transposition"],
                "shared_address_count_with_A": cnt,
                "example_shared_addresses_with_A": shared
            })

    bridge_from_b = []
    for cid in associates_b:
        r = assoc_rows[cid]
        ln_info = min_edit_info(r["LastNames"], ln_a)
        if ln_info["min_distance"] is None:
            continue
        very_close = (ln_info["min_distance"] is not None and ln_info["min_distance"] <= 1) or ln_info["adjacent_transposition"]
        if very_close:
            cnt, shared = addr_overlap_count(r["AddrSet"], b_addrs)
            bridge_from_b.append({
                "cid": int(cid),
                "name": r["Name"],
                "associate_last_names": sorted(list(r["LastNames"])),
                "best_pair": ln_info["pair"],
                "min_distance": ln_info["min_distance"],
                "adjacent_transposition": ln_info["adjacent_transposition"],
                "shared_address_count_with_B": cnt,
                "example_shared_addresses_with_B": shared
            })

    # intra-side same-surname support
    same_surname_support_a = []
    for cid in associates_a:
        r = assoc_rows[cid]
        if r["LastNames"] & ln_a:
            cnt, shared = addr_overlap_count(r["AddrSet"], a_addrs)
            same_surname_support_a.append({
                "cid": int(cid),
                "name": r["Name"],
                "shared_address_count_with_A": cnt,
                "example_shared_addresses_with_A": shared
            })

    same_surname_support_b = []
    for cid in associates_b:
        r = assoc_rows[cid]
        if r["LastNames"] & ln_b:
            cnt, shared = addr_overlap_count(r["AddrSet"], b_addrs)
            same_surname_support_b.append({
                "cid": int(cid),
                "name": r["Name"],
                "shared_address_count_with_B": cnt,
                "example_shared_addresses_with_B": shared
            })

    def describe(cid: int) -> str:
        r = assoc_rows[cid]
        return f"{cid}: {r['Name']}"

    examples_a = ", ".join(describe(cid) for cid in sorted(list(associates_a))[:3])
    examples_b = ", ".join(describe(cid) for cid in sorted(list(associates_b))[:3])

    return {
        "associates_a": sorted(list(associates_a)),
        "associates_b": sorted(list(associates_b)),
        "associates_examples_a": examples_a,
        "associates_examples_b": examples_b,
        "bridge_by_address": bridge_by_address,
        "surname_bridge_from_a": bridge_from_a[:4],
        "surname_bridge_from_b": bridge_from_b[:4],
        "same_surname_support_a": same_surname_support_a[:4],
        "same_surname_support_b": same_surname_support_b[:4],
    }

def build_pair_features(a_id: int, b_id: int, df: pd.DataFrame, addr_index: Dict[str, Set[int]]) -> dict:
    row_a = df[df["ClusterID"] == a_id].iloc[0]
    row_b = df[df["ClusterID"] == b_id].iloc[0]

    a_addrs = row_a["AddrSet"]
    b_addrs = row_b["AddrSet"]
    shared_addrs = sorted(list(a_addrs & b_addrs))
    similar_addrs = find_similar_addresses(a_addrs, b_addrs)

    po_flags = {
        "any_po_a": any(looks_like_po_box(x) for x in a_addrs),
        "any_po_b": any(looks_like_po_box(x) for x in b_addrs),
    }

    ln_sim_ratio = last_name_similarity(row_a["LastNames"], row_b["LastNames"])
    ln_edit = min_edit_info(row_a["LastNames"], row_b["LastNames"])

    assoc = summarize_associates(a_id, b_id, df, addr_index)

    return {
        "a": {
            "id": int(a_id),
            "name": row_a["Name"],
            "first_name": row_a["First"],
            "addresses": list(a_addrs),
            "last_names": sorted(list(row_a["LastNames"])),
        },
        "b": {
            "id": int(b_id),
            "name": row_b["Name"],
            "first_name": row_b["First"],
            "addresses": list(b_addrs),
            "last_names": sorted(list(row_b["LastNames"])),
        },
        "shared_addresses": shared_addrs,
        "shared_address_count": len(shared_addrs),
        "similar_addresses": similar_addrs,
        "last_name_similarity_ratio": ln_sim_ratio,
        "last_name_edit": ln_edit,
        "po_flags": po_flags,
        "associates": assoc,
    }

#############################################
# Enhanced Prompt with detailed reasoning
#############################################

SYSTEM_MSG = """You are an entity-resolution expert deciding if two clusters represent the SAME PERSON.
Make a firm decision each time. Be precise and pragmatic: merge when there is compelling evidence
OR multiple weaker signals that converge; skip otherwise.

REASONING PROCESS:
1. Analyze the names - are they consistent with the same person?
2. Look for name change patterns (e.g., marriage - names like Jennifer, Nancy, Brenda with surname changes)
3. Examine address connections - exact matches, high similarity scores
4. Consider supporting evidence from associates and surname bridges
5. Combine all signals to reach a decision

WEIGHTING SYSTEM:
- VERY HIGH: Exact shared residential addresses (not PO boxes)
- HIGH:
  * Name pattern suggesting marriage (same first/middle, different surname, consider if name sounds feminine)
  * Similar addresses (similarity > 0.85) especially residential
  * Surname-bridge via associate with multiple shared residential addresses
- MEDIUM:
  * Same FIRST+MIDDLE + high surname similarity (>0.90) + compatible geography
  * Adjacent transposition in surnames (likely typos)
  * Edit distance ≤1 in surnames with other supporting evidence
- LOW (supporting only):
  * Shared PO boxes only
  * City/ZIP co-location without address similarity

MARRIAGE NAME PATTERNS: When analyzing names, consider whether the first name suggests a person
who might change their surname through marriage. Same first/middle with different surnames can be
a STRONG positive signal if it fits this pattern.

ADDRESS SIMILARITY: Similar addresses carry significant weight. Consider apartment number changes,
formatting differences, etc.

DETAILED REASONING: Provide a "detailed_analysis" field that walks through your reasoning process
step-by-step in 3-5 sentences explaining the logical flow from evidence to conclusion.

Return STRICT JSON only:
{
  "merge": true|false,
  "confidence": 0..1,
  "reasons": ["4-10 specific bullets about THIS case"],
  "detailed_analysis": "Step-by-step reasoning explanation (3-5 sentences)"
}"""

FEWSHOT = """
Examples:

MERGE (Name change pattern):
{
  "merge": true,
  "confidence": 0.92,
  "reasons": [
    "Same first+middle ('Jennifer M') - exact match",
    "First name 'Jennifer' with different surnames ('SMITH' vs 'JOHNSON') suggests marriage name change",
    "High address similarity: '123 Oak St Apt 2' vs '123 Oak Street Unit 2' (0.94 similarity)",
    "Compatible geographic pattern across all addresses"
  ],
  "detailed_analysis": "The exact match on first and middle names establishes the baseline connection. The first name 'Jennifer' combined with different surnames (SMITH/JOHNSON) provides strong evidence for a marriage name change scenario. The very high address similarity (0.94) between primary addresses suggests these are the same physical location with minor formatting differences. Together, these signals strongly indicate a single person whose name changed through marriage."
}

SKIP (Insufficient evidence):
{
  "merge": false,
  "confidence": 0.75,
  "reasons": [
    "Same first+middle but no direct address matches",
    "Address similarity too low (<0.7) for reliable connection",
    "Different surnames with no marriage change indicator",
    "No surname-bridge via associates"
  ],
  "detailed_analysis": "While the first and middle names match, there are no strong connecting signals beyond this. The address similarity scores are all below 0.7, indicating different locations rather than formatting variations. The different surnames lack any marriage change indicators or typo patterns. Without address connections or name change evidence, the matching first/middle alone is insufficient for a merge decision."
}
""".strip()

def make_prompt(features: dict) -> str:
    # Enhanced feature payload
    similar_addrs_summary = []
    for addr1, addr2, sim in features["similar_addresses"][:3]:
        similar_addrs_summary.append({"addr1": addr1, "addr2": addr2, "similarity": sim})

    features_json = json.dumps({
        "a_id": features["a"]["id"],
        "b_id": features["b"]["id"],
        "a_first_name": features["a"]["first_name"],
        "b_first_name": features["b"]["first_name"],
        "first_middle_match": True,
        "shared_address_count": features["shared_address_count"],
        "shared_addresses": features["shared_addresses"],
        "similar_addresses": similar_addrs_summary,
        "last_name_similarity_ratio": features["last_name_similarity_ratio"],
        "last_name_edit": features["last_name_edit"],
        "a_last_names": features["a"]["last_names"],
        "b_last_names": features["b"]["last_names"],
        "po_flags": features["po_flags"],
        "associates": features["associates"],
    }, ensure_ascii=False)

    a_block = f"- {features['a']['id']}: Name = {features['a']['name']}\n        Addresses = " + "; ".join(features['a']['addresses'])
    b_block = f"- {features['b']['id']}: Name = {features['b']['name']}\n        Addresses = " + "; ".join(features['b']['addresses'])

    similar_block = ""
    if features["similar_addresses"]:
        similar_block = "Similar addresses found:\n"
        for addr1, addr2, sim in features["similar_addresses"][:3]:
            similar_block += f"   • '{addr1}' ↔ '{addr2}' (similarity: {sim:.2f})\n"

    assoc = features["associates"]
    assoc_text = []
    if assoc.get("associates_a"):
        assoc_text.append(f"   • A associates: {assoc['associates_examples_a']}")
    if assoc.get("associates_b"):
        assoc_text.append(f"   • B associates: {assoc['associates_examples_b']}")

    def summarize_bridges(kind: str):
        items = assoc.get(kind, [])
        lines = []
        for it in items[:3]:
            if "shared_address_count_with_A" in it:
                lines.append(
                    f"     - {it['cid']} ({it['name']}): ln_pair={it['best_pair']}, edit={it['min_distance']}, "
                    f"adj_transp={it['adjacent_transposition']}, shared_with_A={it['shared_address_count_with_A']} "
                    f"ex_addresses={it['example_shared_addresses_with_A']}"
                )
            else:
                lines.append(
                    f"     - {it['cid']} ({it['name']}): ln_pair={it['best_pair']}, edit={it['min_distance']}, "
                    f"adj_transp={it['adjacent_transposition']}, shared_with_B={it['shared_address_count_with_B']} "
                    f"ex_addresses={it['example_shared_addresses_with_B']}"
                )
        return "\n".join(lines) if lines else "     - (none)"

    def summarize_same_surname(kind: str):
        items = assoc.get(kind, [])
        lines = []
        for it in items[:3]:
            if "shared_address_count_with_A" in it:
                lines.append(
                    f"     - {it['cid']} ({it['name']}): same-surname intra-side; shared_with_A={it['shared_address_count_with_A']} "
                    f"ex_addresses={it['example_shared_addresses_with_A']}"
                )
            else:
                lines.append(
                    f"     - {it['cid']} ({it['name']}): same-surname intra-side; shared_with_B={it['shared_address_count_with_B']} "
                    f"ex_addresses={it['example_shared_addresses_with_B']}"
                )
        return "\n".join(lines) if lines else "     - (none)"

    body = f"""
{FEWSHOT}

=== Candidate pair (same FIRST+MIDDLE key) ===
 {a_block}
 {b_block}

{similar_block}

Associates (share an address with A/B):
{chr(10).join(assoc_text) if assoc_text else "   • (none detected)"}

Surname bridge from A's associates to B:
{summarize_bridges('surname_bridge_from_a')}

Surname bridge from B's associates to A:
{summarize_bridges('surname_bridge_from_b')}

Intra-side same-surname support (acknowledge as positive but not a bridge):
  A-side:
{summarize_same_surname('same_surname_support_a')}
  B-side:
{summarize_same_surname('same_surname_support_b')}

__FEATURES_BEGIN__
{features_json}
__FEATURES_END__

REMEMBER: Consider if first names suggest marriage name changes. Address similarity ≥0.85 carries HIGH weight.
Return STRICT JSON only.
""".strip()
    return body

########################
# Merge Candidate Class
########################

class MergeCandidate:
    def __init__(self, key: str, a_id: int, b_id: int, decision: LLMDecision, features: dict):
        self.key = key
        self.a_id = a_id
        self.b_id = b_id
        self.decision = decision
        self.features = features
        self.human_decision = None
        self.new_cluster_id = None
        self.merged_name = None

########################
# Display Functions
########################

def format_candidate_summary(candidate: MergeCandidate, df: pd.DataFrame, number: int) -> str:
    """Format a candidate for display with unified numbering"""
    row_a = df[df["ClusterID"] == candidate.a_id].iloc[0]
    row_b = df[df["ClusterID"] == candidate.b_id].iloc[0]

    decision_str = "MERGE" if candidate.decision.merge else "SKIP"
    conf_str = f"{candidate.decision.confidence:.2f}"

    shared_addrs = candidate.features.get("shared_addresses", [])
    similar_addrs = candidate.features.get("similar_addresses", [])

    addr_info = ""
    if shared_addrs:
        addr_info = f" [SHARED: {len(shared_addrs)} addrs]"
    elif similar_addrs:
        best_sim = max(sim for _, _, sim in similar_addrs) if similar_addrs else 0
        addr_info = f" [SIMILAR: {best_sim:.2f}]" if similar_addrs else ""

    name_change_hint = ""
    reasons_text = ' '.join(candidate.decision.reasons).lower()
    if 'marriage' in reasons_text or 'name change' in reasons_text:
        name_change_hint = " [POSSIBLE_NAME_CHANGE]"

    summary = f"""
[{number}] {candidate.key} - {decision_str} ({conf_str}){addr_info}{name_change_hint}
    A({candidate.a_id}): {row_a['Name']}
    B({candidate.b_id}): {row_b['Name']}
    Reasons: {'; '.join(candidate.decision.reasons[:2])}{'...' if len(candidate.decision.reasons) > 2 else ''}
"""
    return summary.strip()

def display_batch_results(candidates: List[MergeCandidate], df: pd.DataFrame) -> None:
    """Display all candidates with unified numbering for batch review"""
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE: {len(candidates)} total candidates")
    print(f"{'='*60}")

    for i, candidate in enumerate(candidates, 1):
        print(format_candidate_summary(candidate, df, i))
        if candidate.decision.detailed_analysis:
            print(f"    Analysis: {candidate.decision.detailed_analysis[:200]}...")

    merge_count = sum(1 for c in candidates if c.decision.merge)
    skip_count = len(candidates) - merge_count
    print(f"\n{'='*60}")
    print(f"SUMMARY: {merge_count} recommended MERGE, {skip_count} recommended SKIP")
    print(f"Recommended merges: {', '.join(str(i+1) for i, c in enumerate(candidates) if c.decision.merge)}")
    print(f"Recommended skips: {', '.join(str(i+1) for i, c in enumerate(candidates) if not c.decision.merge)}")
    print(f"{'='*60}")

def get_batch_decisions(candidates: List[MergeCandidate], df: pd.DataFrame) -> List[MergeCandidate]:
    """Get human decisions for all candidates with unified numbering"""

    if not candidates:
        print("\nNo candidates to review.")
        return []

    print(f"\n{'='*60}")
    print("BATCH DECISION INTERFACE")
    print(f"{'='*60}")
    print("\nEnter the numbers of candidates you want to MERGE (comma-separated)")
    print("Options:")
    print("  '1,3,5' - merge specific candidates by number")
    print("  'recommended' - accept all LLM-recommended merges")
    print("  'none' - skip all candidates (no merges)")
    print("  'review' - show all candidates again")
    print("  'detail 3' - show detailed analysis for candidate #3")
    print(f"\nTotal candidates: {len(candidates)}")

    recommended_indices = [i+1 for i, c in enumerate(candidates) if c.decision.merge]

    while True:
        response = input(f"\nEnter merge decisions: ").strip().lower()

        if response == 'recommended':
            selected_indices = recommended_indices
            break
        elif response == 'none':
            selected_indices = []
            break
        elif response == 'review':
            display_batch_results(candidates, df)
            continue
        elif response.startswith('detail '):
            try:
                idx = int(response.split()[1])
                if 1 <= idx <= len(candidates):
                    c = candidates[idx-1]
                    print(f"\n{'='*40}")
                    print(format_candidate_summary(c, df, idx))
                    print(f"\nDetailed Analysis:")
                    print(c.decision.detailed_analysis)
                    print(f"\nAll Reasons:")
                    for j, reason in enumerate(c.decision.reasons, 1):
                        print(f"  {j}. {reason}")
                    print(f"{'='*40}")
                else:
                    print(f"Invalid index: {idx}")
            except (ValueError, IndexError):
                print("Invalid command format. Use 'detail 3' to see details for candidate 3")
            continue
        else:
            try:
                if response:
                    selected_indices = [int(x.strip()) for x in response.split(',')]
                    invalid = [idx for idx in selected_indices if idx < 1 or idx > len(candidates)]
                    if invalid:
                        print(f"Invalid indices: {invalid}. Please use numbers 1-{len(candidates)}")
                        continue
                else:
                    selected_indices = []
                break
            except ValueError:
                print(f"Invalid input. Use numbers like '1,3,5' or commands: 'recommended', 'none', 'review', 'detail N'")
                continue

    confirmed = []
    for idx in selected_indices:
        candidate = candidates[idx-1]
        candidate.new_cluster_id = min(candidate.a_id, candidate.b_id)

        row_a = df[df["ClusterID"] == candidate.a_id].iloc[0]
        row_b = df[df["ClusterID"] == candidate.b_id].iloc[0]

        all_names = []
        for nm in (str(row_a["Name"]) + ";" + str(row_b["Name"])).split(";"):
            nm = nm.strip()
            if nm and nm not in all_names:
                all_names.append(nm)
        candidate.merged_name = "; ".join(all_names)

        candidate.human_decision = "merge"
        confirmed.append(candidate)

    not_selected = [i for i in range(1, len(candidates)+1) if i not in selected_indices]

    print(f"\n{'='*60}")
    print(f"DECISION SUMMARY:")
    print(f"  Merging: {len(confirmed)} candidates - {selected_indices if selected_indices else 'none'}")
    print(f"  Skipping: {len(not_selected)} candidates - {not_selected if not_selected else 'none'}")
    print(f"{'='*60}")

    return confirmed

########################
# Async Batch Processing
########################

async def process_batch(df: pd.DataFrame, pairs: List[Tuple[str, int, int]],
                       addr_index: Dict[str, Set[int]], llm: AsyncOpenAIAdapter) -> List[MergeCandidate]:
    """Process a batch of pairs asynchronously"""
    print(f"Processing {len(pairs)} pairs asynchronously...")

    prompts = []
    features_map = {}

    for key, a_id, b_id in pairs:
        features = build_pair_features(a_id, b_id, df, addr_index)
        prompt = make_prompt(features)
        pair_key = f"{key}|{a_id}|{b_id}"
        prompts.append((pair_key, prompt))
        features_map[pair_key] = features

    start_time = time.time()
    results = await llm.decide_batch(SYSTEM_MSG, prompts)
    elapsed = time.time() - start_time

    print(f"Completed {len(results)} LLM decisions in {elapsed:.1f}s")

    candidates = []
    for pair_key, decision in results:
        key, a_id, b_id = pair_key.split("|")
        a_id, b_id = int(a_id), int(b_id)
        features = features_map[pair_key]

        candidate = MergeCandidate(key, a_id, b_id, decision, features)
        candidates.append(candidate)

    return candidates

########################
# Main Function
########################

async def main():
    INPUT = "h_refined_clusters_new.csv"
    OUTPUT = "i_refined_clusters_merged.csv"

    print("Loading data...")
    df = pd.read_csv(INPUT, dtype=str)
    df["ClusterID"] = df["ClusterID"].astype(int)

    llm = AsyncOpenAIAdapter(max_concurrent=10)

    skipped: Set[int] = set()
    pass_num = 1

    while True:
        print(f"\n{'='*20} Pass {pass_num} {'='*20}")
        df = enrich(df)
        addr_index = build_address_index(df)
        pairs = find_candidate_pairs(df, skipped)

        if not pairs:
            print("\nNo more candidate pairs.")
            break

        candidates = await process_batch(df, pairs, addr_index, llm)
        display_batch_results(candidates, df)
        confirmed_merges = get_batch_decisions(candidates, df)

        if not confirmed_merges:
            print("\nNo merges confirmed in this pass.")
            for _, a, b in pairs:
                skipped.update([a, b])
            pass_num += 1
            continue

        merged_this_pass = False
        for candidate in confirmed_merges:
            try:
                df = merge_clusters(df, [candidate.a_id, candidate.b_id],
                                 candidate.new_cluster_id, candidate.merged_name)
                print(f"✓ Merged clusters {candidate.a_id} & {candidate.b_id} into {candidate.new_cluster_id}")
                merged_this_pass = True
            except Exception as e:
                print(f"✗ Error merging {candidate.a_id} & {candidate.b_id}: {e}")

        if merged_this_pass:
            print(f"\nCompleted {len(confirmed_merges)} merges in pass {pass_num}")

        merged_ids = set()
        for candidate in confirmed_merges:
            merged_ids.update([candidate.a_id, candidate.b_id])

        for _, a, b in pairs:
            if a not in merged_ids and b not in merged_ids:
                skipped.update([a, b])

        pass_num += 1

    df.to_csv(OUTPUT, index=False)
    print(f"\n✅ Final results written to {OUTPUT}")
    print(f"Total passes completed: {pass_num - 1}")


if __name__ == "__main__":
    asyncio.run(main())
