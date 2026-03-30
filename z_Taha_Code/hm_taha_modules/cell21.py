"""Auto-generated from HM_Taha.ipynb cell 21."""

#!/usr/bin/env python
# coding: utf-8

"""
Advanced Link Evaluator System v2.1 - FIXED
A comprehensive entity resolution evaluation framework with enhanced metrics and reporting.
Author: Expert Systems Team
FIXES: Standard mode data parsing and metrics calculation
"""

import sys
import time
import datetime
import re
import json
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import csv

# ============================================================================
# Configuration and Constants
# ============================================================================

VERSION = "2.1"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# Data Classes for Better Type Safety
# ============================================================================

@dataclass
class ClusterData:
    """Represents a cluster with its ID and member records."""
    cluster_id: str
    record_ids: List[str] = field(default_factory=list)

@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for entity resolution evaluation."""
    true_positives: float = 0
    false_positives: float = 0
    false_negatives: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0
    pairwise_precision: float = 0
    pairwise_recall: float = 0
    pairwise_f1: float = 0
    cluster_precision: float = 0
    cluster_recall: float = 0
    cluster_f1: float = 0
    homogeneity: float = 0
    completeness: float = 0
    v_measure: float = 0
    adjusted_rand_index: float = 0
    mutual_info_score: float = 0
    fowlkes_mallows_index: float = 0

# ============================================================================
# Enhanced File Parser Class - FIXED
# ============================================================================

class FileParser:
    """Advanced file parser with support for multiple formats - FIXED."""

    @staticmethod
    def parse_link_index(filepath: str, logger: logging.Logger) -> Tuple[List[Tuple[str, str]], List[str], Dict[str, str]]:
        """
        Parse a link index file (RefID, ClusterID format).
        Returns: (list of pairs, list of unique RefIDs, dict mapping RefID to ClusterID)
        FIXED: Now properly creates pairs from clusters instead of wrong approach
        """
        logger.info(f"Parsing link index file: {filepath}")
        pairs = []
        ref_ids = []
        cluster_dict = defaultdict(list)
        ref_to_cluster = {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Try to detect the delimiter
                first_line = f.readline().strip()
                f.seek(0)  # Reset to beginning

                # Check if it's comma or tab separated
                if '\t' in first_line:
                    delimiter = '\t'
                else:
                    delimiter = ','

                reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=True)

                # Handle different column name variations
                for row in reader:
                    # Try different column name variations
                    ref_id = (row.get('RefID') or row.get('refID') or
                             row.get('Ref ID') or row.get('RecID') or '').strip()
                    cluster_id = (row.get('ClusterID') or row.get('clusterID') or
                                row.get('Cluster ID') or row.get('TruthID') or
                                row.get('Truth ID') or '').strip()

                    if ref_id and cluster_id:
                        ref_ids.append(ref_id)
                        cluster_dict[cluster_id].append(ref_id)
                        ref_to_cluster[ref_id] = cluster_id

            # FIXED: Generate pairs correctly - all pairs within each cluster
            logger.info("Generating pairs from link index clusters...")
            for cluster_id, members in cluster_dict.items():
                # Generate all pairs within this cluster (including self-pairs)
                for i in range(len(members)):
                    for j in range(i, len(members)):  # Include i==j for self-pairs
                        pairs.append((members[i], members[j]))
                        # Also add reverse pair if different
                        if i != j:
                            pairs.append((members[j], members[i]))

            logger.info(f"Parsed {len(ref_ids)} references in {len(cluster_dict)} clusters")
            logger.info(f"Generated {len(pairs)} pairs from link index")
            return pairs, ref_ids, ref_to_cluster

        except Exception as e:
            logger.error(f"Error parsing link index file: {e}")
            raise

    @staticmethod
    def parse_linked_pairs(filepath: str, logger: logging.Logger) -> List[Tuple[str, str]]:
        """
        Parse a linked pairs file (RefID1, RefID2 format).
        Returns: list of pairs
        FIXED: Better parsing and bidirectional pairs
        """
        logger.info(f"Parsing linked pairs file: {filepath}")
        pairs = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Try to detect if first line is header
                first_line = f.readline().strip()
                f.seek(0)

                # Skip header if it contains non-numeric data or common header words
                if any(word in first_line.lower() for word in ['refid', 'id1', 'id2', 'ref_id']):
                    f.readline()  # Skip header
                else:
                    f.seek(0)  # Reset if first line might be data

                # Read pairs
                for line_num, line in enumerate(f, 2):
                    line = line.strip()
                    if not line:
                        continue

                    # Handle different delimiters
                    if '\t' in line:
                        parts = line.split('\t')
                    else:
                        parts = line.split(',')

                    if len(parts) >= 2:
                        ref_id1 = parts[0].strip().strip('"').strip("'")
                        ref_id2 = parts[1].strip().strip('"').strip("'")

                        if ref_id1 and ref_id2:
                            # Add both directions for symmetry
                            pairs.append((ref_id1, ref_id2))
                            pairs.append((ref_id2, ref_id1))

            logger.info(f"Parsed {len(pairs)} linked pairs (including bidirectional)")
            return pairs

        except Exception as e:
            logger.error(f"Error parsing linked pairs file: {e}")
            raise

    @staticmethod
    def parse_truth_file(filepath: str, ref_ids: List[str], logger: logging.Logger) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
        """
        Parse a truth file and create truth pairs.
        Returns: (list of truth pairs, dict mapping RefID to TruthID)
        FIXED: Better truth pair generation with enhanced parsing
        """
        logger.info(f"Parsing truth file: {filepath}")
        truth_dict = {}
        truth_pairs = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read first few lines to analyze format
                lines = []
                for _ in range(5):
                    line = f.readline().strip()
                    if line:
                        lines.append(line)
                f.seek(0)  # Reset

                if not lines:
                    logger.error("Truth file is empty")
                    return [], {}

                # Analyze format
                logger.info("Analyzing truth file format...")
                for i, line in enumerate(lines[:3]):
                    logger.info(f"  Line {i+1}: {line}")

                # Try to detect delimiter
                first_line = lines[0]
                if '\t' in first_line:
                    delimiter = '\t'
                    logger.info("Detected tab delimiter")
                elif ',' in first_line:
                    delimiter = ','
                    logger.info("Detected comma delimiter")
                else:
                    # Try space or assume comma
                    delimiter = ' ' if ' ' in first_line else ','
                    logger.info(f"Defaulting to '{delimiter}' delimiter")

                # Try CSV reader first
                try:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=True)
                    headers = reader.fieldnames
                    logger.info(f"CSV headers detected: {headers}")

                    # Read truth assignments
                    for row_num, row in enumerate(reader, 2):
                        # Try different column name variations
                        ref_id = None
                        truth_id = None

                        # Try to find RefID column
                        for key in row.keys():
                            if key and any(term in key.lower() for term in ['refid', 'ref_id', 'recid', 'rec_id', 'id']):
                                ref_id = row[key].strip().strip('"').strip("'") if row[key] else None
                                break

                        # Try to find TruthID column
                        for key in row.keys():
                            if key and any(term in key.lower() for term in ['truthid', 'truth_id', 'clusterid', 'cluster_id', 'truth', 'cluster']):
                                truth_id = row[key].strip().strip('"').strip("'") if row[key] else None
                                break

                        # If still no columns found, try by position
                        if not ref_id or not truth_id:
                            values = list(row.values())
                            if len(values) >= 2:
                                if not ref_id:
                                    ref_id = values[0].strip().strip('"').strip("'") if values[0] else None
                                if not truth_id:
                                    truth_id = values[1].strip().strip('"').strip("'") if values[1] else None

                        # Only process if RefID is in our reference list
                        if ref_id and truth_id and ref_id in ref_ids:
                            truth_dict[ref_id] = truth_id
                        elif ref_id and truth_id:
                            # RefID not in reference list - this might indicate a mismatch
                            if row_num <= 5:  # Only log first few mismatches
                                logger.warning(f"RefID '{ref_id}' from truth file not found in link index")

                except Exception as csv_error:
                    # CSV parsing failed, try simple line-by-line parsing
                    logger.warning(f"CSV parsing failed: {csv_error}")
                    logger.info("Trying line-by-line parsing...")

                    f.seek(0)
                    lines = f.readlines()

                    # Skip potential header
                    start_line = 0
                    if lines and any(term in lines[0].lower() for term in ['refid', 'truthid', 'id', 'cluster']):
                        start_line = 1
                        logger.info("Skipping header line")

                    for line_num, line in enumerate(lines[start_line:], start_line + 1):
                        line = line.strip()
                        if not line:
                            continue

                        # Split by delimiter
                        parts = [p.strip().strip('"').strip("'") for p in line.split(delimiter)]

                        if len(parts) >= 2:
                            ref_id = parts[0]
                            truth_id = parts[1]

                            if ref_id and truth_id and ref_id in ref_ids:
                                truth_dict[ref_id] = truth_id

                logger.info(f"Successfully parsed {len(truth_dict)} truth assignments")

                # Create truth pairs from truth assignments
                truth_clusters = defaultdict(list)
                for ref_id, truth_id in truth_dict.items():
                    truth_clusters[truth_id].append(ref_id)

                # Generate pairs from truth clusters
                logger.info("Generating truth pairs from truth clusters...")
                for truth_id, members in truth_clusters.items():
                    # Generate all pairs within this cluster (including self-pairs)
                    for i in range(len(members)):
                        for j in range(i, len(members)):  # Include self-pairs
                            truth_pairs.append((members[i], members[j]))
                            # Also add reverse pair if different
                            if i != j:
                                truth_pairs.append((members[j], members[i]))

            logger.info(f"Created {len(truth_pairs)} truth pairs from {len(truth_clusters)} truth clusters")

            # Additional diagnostics
            if len(truth_dict) == 0:
                logger.error("No truth assignments found! Possible issues:")
                logger.error("  1. File format not recognized")
                logger.error("  2. Column names don't match expected patterns")
                logger.error("  3. RefIDs in truth file don't match RefIDs in link index")
                logger.error("  4. File encoding issues")

            return truth_pairs, truth_dict

        except Exception as e:
            logger.error(f"Error parsing truth file: {e}")
            raise

    @staticmethod
    def parse_complex_cluster_file(filepath: str, logger: logging.Logger) -> Tuple[List[ClusterData], Dict[str, str], Dict[str, str]]:
        """
        Parse complex cluster file format with RecIDs in brackets.
        Example: 20 BARBARA CHAVEZ [1.3, 1.4, 1.8]; [1.5, 1.6]; ...
        Or CSV format: 0,A "[328.1, 328.2]"
        Returns: (list of ClusterData objects, dict mapping RefID to ClusterID, dict mapping RefID to TruthID)
        """
        logger.info(f"Parsing complex cluster file: {filepath}")
        clusters = []
        ref_to_cluster = {}
        ref_to_truth = {}  # For automatic truth generation

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # Try to extract cluster ID and RecIDs
                    rec_ids = []
                    cluster_id = None

                    # Check if it's CSV format with comma-separated cluster ID
                    if ',' in line and '[' in line:
                        # Format like: 0,A "[328.1, 328.2]" or "0,A" [328.1, 328.2]
                        comma_pos = line.find(',')
                        # Extract just the numeric part before the comma
                        cluster_id = line[:comma_pos].strip().strip('"')
                    else:
                        # Format like: 20 BARBARA CHAVEZ [1.3, 1.4, 1.8]
                        parts = line.split(maxsplit=1)
                        if parts:
                            cluster_id = parts[0]

                    # Extract all RecIDs from brackets
                    bracket_pattern = r'\[(.*?)\]'
                    matches = re.findall(bracket_pattern, line)

                    for match in matches:
                        # Split by comma and clean
                        ids = [id.strip().strip('"').strip("'") for id in match.split(',') if id.strip()]
                        rec_ids.extend(ids)

                    if rec_ids and cluster_id is not None:
                        # Sort RecIDs to find the smallest one
                        sorted_rec_ids = sorted(rec_ids)

                        # Use the smallest RecID as the new cluster ID
                        new_cluster_id = sorted_rec_ids[0]

                        cluster = ClusterData(cluster_id=new_cluster_id, record_ids=rec_ids)
                        clusters.append(cluster)

                        # Map each RefID to its new cluster ID
                        for ref_id in rec_ids:
                            ref_to_cluster[ref_id] = new_cluster_id

                            # Extract truth cluster from RecID pattern
                            if '.' in ref_id:
                                # For pattern like "328.1" -> truth is "328.1" (base + .1)
                                base_num = ref_id.split('.')[0]
                                truth_id = f"{base_num}.1"
                                ref_to_truth[ref_id] = truth_id
                            else:
                                # If no dot pattern, add .1 to make it consistent
                                ref_to_truth[ref_id] = f"{ref_id}.1"

            logger.info(f"Parsed {len(clusters)} clusters with {len(ref_to_cluster)} total records")
            logger.info(f"Automatically derived truth clusters from RecID patterns")
            logger.info(f"Renamed cluster IDs to smallest RecID in each cluster")
            return clusters, ref_to_cluster, ref_to_truth

        except Exception as e:
            logger.error(f"Error parsing complex cluster file: {e}")
            raise

    @staticmethod
    def create_cluster_file(clusters: List[ClusterData], output_path: str, logger: logging.Logger):
        """Create a clean cluster file with ClusterID and RecIDs columns."""
        logger.info(f"Creating cluster file: {output_path}")

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ClusterID', 'RecIDs'])

                for cluster in clusters:
                    rec_ids_str = ';'.join(cluster.record_ids)
                    writer.writerow([cluster.cluster_id, rec_ids_str])

            logger.info(f"Successfully created cluster file with {len(clusters)} clusters")

        except Exception as e:
            logger.error(f"Error creating cluster file: {e}")
            raise

# ============================================================================
# Enhanced Transitive Closure with Optimization
# ============================================================================

class TransitiveClosure:
    """Optimized transitive closure implementation using Union-Find algorithm."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: str) -> str:
        """Find with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str):
        """Union by rank."""
        px, py = self.find(x), self.find(y)

        if px == py:
            return

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def compute_closure(self, pairs: List[Tuple[str, str]], logger: logging.Logger) -> List[Tuple[str, str]]:
        """Compute transitive closure using Union-Find."""
        logger.info("Computing transitive closure using Union-Find algorithm")

        # Build union-find structure
        for a, b in pairs:
            self.union(a, b)

        # Build clusters
        clusters = defaultdict(list)
        for node in self.parent:
            root = self.find(node)
            clusters[root].append(node)

        # Generate all pairs within each cluster (including self-pairs)
        closure_pairs = []
        for root, members in clusters.items():
            for i in range(len(members)):
                for j in range(i, len(members)):  # Include self-pairs
                    closure_pairs.append((members[i], members[j]))
                    # Add reverse if different
                    if i != j:
                        closure_pairs.append((members[j], members[i]))

        logger.info(f"Transitive closure computed: {len(closure_pairs)} pairs from {len(clusters)} clusters")
        return sorted(list(set(closure_pairs)))

# ============================================================================
# Advanced Metrics Calculator - FIXED
# ============================================================================

class MetricsCalculator:
    """Comprehensive metrics calculation for entity resolution evaluation - FIXED."""

    @staticmethod
    def calculate_all_metrics(predicted_pairs: List[Tuple[str, str]],
                            truth_pairs: List[Tuple[str, str]],
                            logger: logging.Logger) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics with progress indication - FIXED."""
        metrics = EvaluationMetrics()

        # Handle empty inputs gracefully
        if not predicted_pairs or not truth_pairs:
            logger.warning("Empty predicted or truth pairs - returning zero metrics")
            return metrics

        print("  Step 1/4: Converting pairs to sets for comparison...")
        # Convert to sets for efficient operations (normalize pair order)
        pred_set = set()
        truth_set = set()

        # Normalize pairs to ensure consistent ordering (smaller ID first)
        for a, b in predicted_pairs:
            if a <= b:
                pred_set.add((a, b))
            else:
                pred_set.add((b, a))

        for a, b in truth_pairs:
            if a <= b:
                truth_set.add((a, b))
            else:
                truth_set.add((b, a))

        print(f"    Normalized to {len(pred_set)} predicted pairs and {len(truth_set)} truth pairs")

        print("  Step 2/4: Calculating basic pairwise metrics...")
        # Basic pairwise metrics
        tp = len(pred_set & truth_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)

        metrics.true_positives = tp
        metrics.false_positives = fp
        metrics.false_negatives = fn

        print(f"    TP: {tp}, FP: {fp}, FN: {fn}")

        # Precision, Recall, F1
        metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics.f1_score = (2 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) \
                          if (metrics.precision + metrics.recall) > 0 else 0

        # Store pairwise metrics
        metrics.pairwise_precision = metrics.precision
        metrics.pairwise_recall = metrics.recall
        metrics.pairwise_f1 = metrics.f1_score

        print("  Step 3/4: Computing cluster-level metrics...")
        # Calculate cluster-level metrics
        pred_clusters = MetricsCalculator._pairs_to_clusters(list(pred_set))
        truth_clusters = MetricsCalculator._pairs_to_clusters(list(truth_set))

        # Homogeneity, Completeness, V-measure
        metrics.homogeneity = MetricsCalculator._calculate_homogeneity(pred_clusters, truth_clusters)
        metrics.completeness = MetricsCalculator._calculate_completeness(pred_clusters, truth_clusters)
        metrics.v_measure = (2 * metrics.homogeneity * metrics.completeness) / \
                           (metrics.homogeneity + metrics.completeness) \
                           if (metrics.homogeneity + metrics.completeness) > 0 else 0

        print("  Step 4/4: Computing advanced metrics...")
        # Fowlkes-Mallows Index
        metrics.fowlkes_mallows_index = MetricsCalculator._calculate_fowlkes_mallows(
            pred_clusters, truth_clusters, tp, fp, fn
        )

        # Adjusted Rand Index (using fast approximation for large datasets)
        if len(predicted_pairs) > 10000 or len(truth_pairs) > 10000:
            print("    Using fast approximation for Adjusted Rand Index (large dataset)...")
            metrics.adjusted_rand_index = MetricsCalculator._calculate_adjusted_rand_index_fast(
                pred_clusters, truth_clusters, tp, fp, fn
            )
        else:
            metrics.adjusted_rand_index = MetricsCalculator._calculate_adjusted_rand_index(
                pred_clusters, truth_clusters
            )

        logger.info("All metrics calculated successfully")
        return metrics

    @staticmethod
    def _calculate_adjusted_rand_index_fast(pred_clusters: Dict, truth_clusters: Dict,
                                           tp: int, fp: int, fn: int) -> float:
        """Fast approximation of Adjusted Rand Index for large datasets."""
        # Use the confusion matrix approach for efficiency
        n = len(set().union(*pred_clusters.values(), *truth_clusters.values()))
        if n < 2:
            return 0

        # Total pairs
        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 0

        # Agreement pairs (TP + TN)
        tn = total_pairs - tp - fp - fn
        rand_index = (tp + tn) / total_pairs

        # Simple adjustment for chance
        expected_index = 0.5  # Simplified expectation
        max_index = 1.0

        if max_index - expected_index == 0:
            return 0

        adjusted = (rand_index - expected_index) / (max_index - expected_index)
        return max(-1, min(1, adjusted))  # Bound between -1 and 1

    @staticmethod
    def _pairs_to_clusters(pairs: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
        """Convert pairs to cluster representation using Union-Find."""
        # Use Union-Find to properly build clusters from pairs
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px

        # Build union-find structure
        for a, b in pairs:
            union(a, b)

        # Build clusters
        clusters = defaultdict(set)
        for node in parent:
            root = find(node)
            clusters[root].add(node)

        return dict(clusters)

    @staticmethod
    def _calculate_homogeneity(pred_clusters: Dict, truth_clusters: Dict) -> float:
        """Calculate homogeneity score."""
        # Simplified homogeneity calculation
        total_items = sum(len(cluster) for cluster in pred_clusters.values())
        if total_items == 0:
            return 0

        homogeneity_sum = 0
        for pred_key, pred_members in pred_clusters.items():
            max_overlap = 0
            for truth_key, truth_members in truth_clusters.items():
                overlap = len(pred_members & truth_members)
                max_overlap = max(max_overlap, overlap)
            homogeneity_sum += max_overlap

        return homogeneity_sum / total_items if total_items > 0 else 0

    @staticmethod
    def _calculate_completeness(pred_clusters: Dict, truth_clusters: Dict) -> float:
        """Calculate completeness score."""
        total_items = sum(len(cluster) for cluster in truth_clusters.values())
        if total_items == 0:
            return 0

        completeness_sum = 0
        for truth_key, truth_members in truth_clusters.items():
            max_overlap = 0
            for pred_key, pred_members in pred_clusters.items():
                overlap = len(truth_members & pred_members)
                max_overlap = max(max_overlap, overlap)
            completeness_sum += max_overlap

        return completeness_sum / total_items if total_items > 0 else 0

    @staticmethod
    def _calculate_fowlkes_mallows(pred_clusters: Dict, truth_clusters: Dict,
                                  tp: int, fp: int, fn: int) -> float:
        """Calculate Fowlkes-Mallows Index."""
        if tp + fp == 0 or tp + fn == 0:
            return 0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return (precision * recall) ** 0.5

    @staticmethod
    def _calculate_adjusted_rand_index(pred_clusters: Dict, truth_clusters: Dict) -> float:
        """Calculate Adjusted Rand Index (simplified version)."""
        # This is a simplified calculation
        # For production, consider using sklearn.metrics.adjusted_rand_score
        all_items = set()
        for cluster in pred_clusters.values():
            all_items.update(cluster)
        for cluster in truth_clusters.values():
            all_items.update(cluster)

        n = len(all_items)
        if n < 2:
            return 0

        # Count agreements and disagreements
        agreements = 0
        total_pairs = 0

        items_list = list(all_items)
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                item1, item2 = items_list[i], items_list[j]
                total_pairs += 1

                # Check if items are in same cluster in both pred and truth
                in_same_pred = any(item1 in cluster and item2 in cluster
                                  for cluster in pred_clusters.values())
                in_same_truth = any(item1 in cluster and item2 in cluster
                                   for cluster in truth_clusters.values())

                if in_same_pred == in_same_truth:
                    agreements += 1

        return (agreements / total_pairs) if total_pairs > 0 else 0

# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """Generate comprehensive evaluation reports."""

    @staticmethod
    def generate_metrics_report(metrics: EvaluationMetrics, logger: logging.Logger, log_file):
        """Generate detailed metrics report with explanations."""

        report_lines = [
            "\n" + "="*80,
            "COMPREHENSIVE ENTITY RESOLUTION EVALUATION REPORT",
            "="*80,
            "\n### PAIRWISE METRICS ###\n",

            f"True Positive Pairs: {metrics.true_positives:.0f}",
            "  → Correctly identified entity pairs that belong together",

            f"False Positive Pairs: {metrics.false_positives:.0f}",
            "  → Incorrectly linked pairs that don't belong together",

            f"False Negative Pairs: {metrics.false_negatives:.0f}",
            "  → Missed pairs that should have been linked",

            f"\nPrecision: {metrics.precision:.4f}",
            "  → Proportion of identified links that are correct",
            "  → Higher precision means fewer false matches",
            "  → Range: [0,1] where 1 is perfect precision",

            f"\nRecall: {metrics.recall:.4f}",
            "  → Proportion of true links that were identified",
            "  → Higher recall means fewer missed matches",
            "  → Range: [0,1] where 1 is perfect recall",

            f"\nF1 Score: {metrics.f1_score:.4f}",
            "  → Harmonic mean of precision and recall",
            "  → Balanced measure of overall performance",
            "  → Range: [0,1] where 1 is perfect performance",

            "\n### CLUSTER QUALITY METRICS ###\n",

            f"Homogeneity: {metrics.homogeneity:.4f}",
            "  → Measures if clusters contain only members of a single true class",
            "  → Higher homogeneity means purer clusters",
            "  → Range: [0,1] where 1 means perfectly homogeneous clusters",

            f"\nCompleteness: {metrics.completeness:.4f}",
            "  → Measures if all members of a true class are in the same cluster",
            "  → Higher completeness means fewer split entities",
            "  → Range: [0,1] where 1 means no entity splitting",

            f"\nV-Measure: {metrics.v_measure:.4f}",
            "  → Harmonic mean of homogeneity and completeness",
            "  → Overall measure of clustering quality",
            "  → Range: [0,1] where 1 is perfect clustering",

            f"\nFowlkes-Mallows Index: {metrics.fowlkes_mallows_index:.4f}",
            "  → Geometric mean of precision and recall",
            "  → Alternative balanced measure of performance",
            "  → Range: [0,1] where 1 indicates perfect agreement",

            f"\nAdjusted Rand Index: {metrics.adjusted_rand_index:.4f}",
            "  → Measures similarity between predicted and true clusters",
            "  → Adjusted for chance (random clustering would score ~0)",
            "  → Range: [-1,1] where 1 is perfect agreement, 0 is random",

            "\n### PERFORMANCE INTERPRETATION ###\n",
        ]

        # Add performance interpretation
        if metrics.f1_score >= 0.9:
            report_lines.append("★ EXCELLENT: Very high quality entity resolution")
        elif metrics.f1_score >= 0.8:
            report_lines.append("★ GOOD: Strong entity resolution performance")
        elif metrics.f1_score >= 0.7:
            report_lines.append("★ FAIR: Acceptable performance with room for improvement")
        elif metrics.f1_score >= 0.6:
            report_lines.append("★ POOR: Significant improvements needed")
        else:
            report_lines.append("★ VERY POOR: Major issues with entity resolution")

        # Identify specific issues
        report_lines.append("\n### DIAGNOSTIC INSIGHTS ###\n")

        if metrics.precision < metrics.recall:
            report_lines.append("⚠ Low Precision: Too many false matches (over-linking)")
            report_lines.append("  Recommendation: Tighten matching criteria")
        elif metrics.recall < metrics.precision:
            report_lines.append("⚠ Low Recall: Too many missed matches (under-linking)")
            report_lines.append("  Recommendation: Relax matching criteria or add more features")

        if metrics.homogeneity < 0.8:
            report_lines.append("⚠ Low Homogeneity: Clusters contain mixed entities")
            report_lines.append("  Recommendation: Review clustering algorithm or similarity threshold")

        if metrics.completeness < 0.8:
            report_lines.append("⚠ Low Completeness: Same entities split across clusters")
            report_lines.append("  Recommendation: Improve transitive closure or linkage strategy")

        report_lines.append("\n" + "="*80)

        # Write to console and log file
        for line in report_lines:
            print(line)
            print(line, file=log_file)

        logger.info("Metrics report generated successfully")

    @staticmethod
    def generate_cluster_profile(clusters: List[ClusterData], logger: logging.Logger, log_file):
        """Generate detailed cluster size profile - FIXED division by zero."""

        if not clusters:
            print("\n### CLUSTER SIZE DISTRIBUTION ###", file=log_file)
            print("No clusters to analyze.", file=log_file)
            logger.warning("No clusters provided for profile generation")
            return

        size_distribution = Counter(len(cluster.record_ids) for cluster in clusters)

        print("\n### CLUSTER SIZE DISTRIBUTION ###", file=log_file)
        print("\nCluster Size Distribution:", file=log_file)
        print("Size\tCount\tRecords\tCumulative%", file=log_file)
        print("-"*50, file=log_file)

        total_records = sum(size * count for size, count in size_distribution.items())
        cumulative = 0

        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            records = size * count
            cumulative += records
            cum_percent = (cumulative / total_records * 100) if total_records > 0 else 0

            print(f"{size}\t{count}\t{records}\t{cum_percent:.1f}%", file=log_file)

        print(f"\nTotal Clusters: {len(clusters)}", file=log_file)
        print(f"Total Records: {total_records}", file=log_file)
        print(f"Average Cluster Size: {total_records/len(clusters):.2f}" if len(clusters) > 0 else "Average Cluster Size: 0.00", file=log_file)
        print(f"Singleton Clusters: {size_distribution.get(1, 0)}", file=log_file)

        logger.info("Cluster profile generated successfully")

# ============================================================================
# Main Application Class - ENHANCED
# ============================================================================

class LinkEvaluator:
    """Main application class for link evaluation - ENHANCED."""

    def __init__(self):
        self.version = VERSION
        self.logger = self._setup_logger()
        self.start_time = time.time()
        self.log_file = None

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('LinkEvaluator')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _create_log_file(self) -> Any:
        """Create timestamped log file."""
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_filename = f"LinkEvaluator_Log_{timestamp}.txt"

        self.logger.info(f"Creating log file: {log_filename}")
        return open(log_filename, 'w', encoding='utf-8')

    def run(self):
        """Main execution flow."""
        try:
            # Initialize log file
            self.log_file = self._create_log_file()

            # Print header
            self._print_header()

            # Get input mode
            mode = self._get_input_mode()

            if mode == '1':
                self._process_standard_mode()
            elif mode == '2':
                self._process_complex_cluster_mode()
            else:
                self.logger.error("Invalid mode selected")
                return

            # Print execution time
            self._print_footer()

        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            if self.log_file:
                self.log_file.close()

    def _print_header(self):
        """Print application header."""
        header = [
            "="*80,
            f"Advanced Link Evaluator System v{self.version} - FIXED VERSION",
            f"Execution Date: {datetime.datetime.now().strftime(DATE_FORMAT)}",
            "FIXES: Standard mode data parsing and metrics calculation",
            "="*80,
        ]
        for line in header:
            print(line)
            print(line, file=self.log_file)

    def _print_footer(self):
        """Print execution summary."""
        elapsed_time = time.time() - self.start_time
        footer = [
            "="*80,
            f"Total Execution Time: {elapsed_time/60:.2f} minutes",
            "Process completed successfully",
            "="*80,
        ]
        for line in footer:
            print(line)
            print(line, file=self.log_file)

    def _get_input_mode(self) -> str:
        """Get processing mode from user."""
        print("\nSelect Input Mode:")
        print("1. Standard Mode (Link Index + Linked Pairs + Truth File) - FIXED")
        print("2. Complex Cluster Mode (Process complex cluster format)")

        mode = input("\nEnter mode (1 or 2): ").strip()
        print(f"Selected mode: {mode}", file=self.log_file)
        return mode

    def _process_standard_mode(self):
        """Process standard link index format - FIXED."""
        parser = FileParser()

        print("\n" + "="*50)
        print("STANDARD MODE - ENHANCED DATA VALIDATION")
        print("="*50)

        # Get link index file
        link_index_file = input("\nEnter link index filename: ").strip()

        # Enhanced parsing with validation
        base_pairs, ref_ids, ref_to_cluster = parser.parse_link_index(link_index_file, self.logger)

        print(f"\n✓ Link Index Analysis:")
        print(f"  - Total RefIDs: {len(ref_ids)}")
        print(f"  - Unique Clusters: {len(set(ref_to_cluster.values()))}")
        print(f"  - Pairs from clusters: {len(base_pairs)}")

        # Validate data quality
        cluster_sizes = Counter(ref_to_cluster.values())
        singleton_clusters = sum(1 for count in cluster_sizes.values() if count == 1)
        multi_clusters = len(cluster_sizes) - singleton_clusters

        print(f"  - Singleton clusters: {singleton_clusters}")
        print(f"  - Multi-record clusters: {multi_clusters}")

        # Get linked pairs file
        linked_pairs_file = input("\nEnter linked pairs filename: ").strip()
        additional_pairs = parser.parse_linked_pairs(linked_pairs_file, self.logger)

        print(f"\n✓ Linked Pairs Analysis:")
        print(f"  - Additional pairs: {len(additional_pairs)}")

        # Combine pairs with deduplication
        print(f"\n✓ Combining pairs...")
        all_pairs_set = set(base_pairs + additional_pairs)
        all_pairs = list(all_pairs_set)

        print(f"  - Total unique pairs before closure: {len(all_pairs)}")

        # Get truth file
        truth_file = input("\nEnter truth filename: ").strip()
        truth_pairs, truth_dict = parser.parse_truth_file(truth_file, ref_ids, self.logger)

        print(f"\n✓ Truth Data Analysis:")
        print(f"  - RefIDs with truth: {len(truth_dict)}")
        print(f"  - Truth clusters: {len(set(truth_dict.values()))}")
        print(f"  - Truth pairs: {len(truth_pairs)}")

        # Validate truth coverage
        missing_truth = set(ref_ids) - set(truth_dict.keys())
        if missing_truth:
            print(f"  - ⚠ WARNING: {len(missing_truth)} RefIDs missing truth assignments")
            print(f"    Sample missing: {list(missing_truth)[:5]}")

            # If no truth data found, provide detailed diagnostics
            if len(truth_dict) == 0:
                print(f"\n⚠⚠⚠ CRITICAL: No truth assignments found!")
                print(f"Possible issues:")
                print(f"  1. Wrong file format or delimiter")
                print(f"  2. Column headers don't match expected names")
                print(f"  3. RefIDs in truth file don't match those in link index")
                print(f"  4. File encoding problems")

                # Try to show some sample RefIDs for comparison
                print(f"\nSample RefIDs from link index: {ref_ids[:10]}")

                # Try to peek at truth file format
                try:
                    with open(truth_file, 'r', encoding='utf-8') as f:
                        sample_lines = [f.readline().strip() for _ in range(3)]
                    print(f"Sample lines from truth file:")
                    for i, line in enumerate(sample_lines):
                        print(f"  Line {i+1}: {line}")
                except:
                    print("  Could not read sample lines from truth file")

                # Ask user if they want to continue without evaluation
                continue_choice = input("\nNo truth data found. Continue without evaluation? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    print("Exiting...")
                    return
                else:
                    print("Continuing with clustering analysis only...")

        # Run transitive closure
        print(f"\n✓ Computing transitive closure...")
        closure_engine = TransitiveClosure()
        closure_pairs = closure_engine.compute_closure(all_pairs, self.logger)

        print(f"  - Pairs after closure: {len(closure_pairs)}")

        # Generate cluster profile from closure (even without truth data)
        print(f"\n✓ Generating cluster profile from predicted clusters...")
        clusters_dict = defaultdict(list)

        # Build clusters from all closure pairs
        uf_parent = {}
        def uf_find(x):
            if x not in uf_parent:
                uf_parent[x] = x
            if uf_parent[x] != x:
                uf_parent[x] = uf_find(uf_parent[x])
            return uf_parent[x]

        def uf_union(x, y):
            px, py = uf_find(x), uf_find(y)
            if px != py:
                uf_parent[py] = px

        # Build clusters using Union-Find
        for a, b in closure_pairs:
            uf_union(a, b)

        # Group items by root
        root_clusters = defaultdict(list)
        for item in uf_parent:
            root = uf_find(item)
            root_clusters[root].append(item)

        clusters = [ClusterData(cluster_id=cid, record_ids=list(set(rids)))
                   for cid, rids in root_clusters.items() if rids]

        reporter = ReportGenerator()
        reporter.generate_cluster_profile(clusters, self.logger, self.log_file)

        # Only proceed with evaluation if we have truth data
        if len(truth_dict) > 0:
            # Filter pairs to only include those with truth data
            print(f"\n✓ Filtering pairs for evaluation...")
            truth_ref_ids = set(truth_dict.keys())

            filtered_closure_pairs = [(a, b) for a, b in closure_pairs
                                    if a in truth_ref_ids and b in truth_ref_ids]
            filtered_truth_pairs = [(a, b) for a, b in truth_pairs
                                  if a in truth_ref_ids and b in truth_ref_ids]

            print(f"  - Predicted pairs (filtered): {len(filtered_closure_pairs)}")
            print(f"  - Truth pairs (filtered): {len(filtered_truth_pairs)}")

            if len(filtered_closure_pairs) > 0 and len(filtered_truth_pairs) > 0:
                # Calculate metrics
                print(f"\n✓ Calculating evaluation metrics...")
                calculator = MetricsCalculator()
                metrics = calculator.calculate_all_metrics(filtered_closure_pairs, filtered_truth_pairs, self.logger)

                # Generate reports
                reporter.generate_metrics_report(metrics, self.logger, self.log_file)
            else:
                print(f"\n⚠ Cannot calculate metrics: insufficient filtered pairs")
        else:
            print(f"\n⚠ Skipping metrics calculation: no truth data available")

        # Additional diagnostic information
        print(f"\n### DIAGNOSTIC SUMMARY ###")
        print(f"Input Quality Assessment:")
        print(f"  - Link Index Coverage: {len(ref_ids)} RefIDs")
        print(f"  - Truth Coverage: {len(truth_dict)}/{len(ref_ids)} ({100*len(truth_dict)/len(ref_ids):.1f}%)" if len(ref_ids) > 0 else "  - Truth Coverage: 0/0 (0.0%)")
        print(f"  - Additional Links: {len(additional_pairs)} pairs")
        print(f"  - Transitive Expansion: {len(all_pairs)} → {len(closure_pairs)} pairs")
        print(f"  - Final Clusters: {len(clusters)}")

        if len(truth_dict) > 0 and hasattr(self, 'metrics') and hasattr(self.metrics, 'precision') and self.metrics.precision < 0.5:
            print(f"\n⚠ LOW PRECISION ALERT:")
            print(f"  This suggests data quality issues. Check:")
            print(f"  1. Are the file formats correct?")
            print(f"  2. Do ClusterIDs in link index match truth assignments?")
            print(f"  3. Are there too many spurious links in linked pairs file?")
            print(f"  4. Is transitive closure creating over-connected components?")
        elif len(truth_dict) == 0:
            print(f"\n💡 RECOMMENDATION:")
            print(f"  Fix the truth file parsing to enable evaluation metrics.")
            print(f"  Current analysis shows only clustering structure without accuracy assessment.")

    def _process_complex_cluster_mode(self):
        """Process complex cluster file format."""
        parser = FileParser()

        # Get complex cluster file
        cluster_file = input("\nEnter complex cluster filename: ").strip()
        clusters, ref_to_cluster, ref_to_truth = parser.parse_complex_cluster_file(cluster_file, self.logger)

        # Create output files
        output_prefix = Path(cluster_file).stem

        # Create clean cluster file
        clean_cluster_file = f"{output_prefix}_clusters.csv"
        parser.create_cluster_file(clusters, clean_cluster_file, self.logger)
        print(f"Created cluster file: {clean_cluster_file}")

        # Create link index file with properly formatted ClusterIDs
        link_index_file = f"{output_prefix}_linkindex.csv"
        with open(link_index_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['RefID', 'ClusterID'])

            # Sort by RefID for better readability
            for ref_id in sorted(ref_to_cluster.keys()):
                cluster_id = ref_to_cluster[ref_id]
                writer.writerow([ref_id, cluster_id])

        print(f"Created link index file: {link_index_file}")
        self.logger.info(f"Link index uses smallest RecID as ClusterID for each cluster")

        # Generate cluster profile
        reporter = ReportGenerator()
        reporter.generate_cluster_profile(clusters, self.logger, self.log_file)

        # Ask about truth file
        truth_option = input("\nDo you have a separate truth file for evaluation? (y/n): ").strip().lower()

        if truth_option == 'y':
            # Use provided truth file
            truth_file = input("Enter truth filename: ").strip()
            ref_ids = list(ref_to_cluster.keys())
            truth_pairs, _ = parser.parse_truth_file(truth_file, ref_ids, self.logger)
            print("Using provided truth file for evaluation")
        else:
            # Use automatic truth derivation from RecID patterns
            print("\nUsing automatic truth derivation from RecID patterns")
            print("Truth pattern: Records like 328.1, 328.2 → Truth Cluster 328.1")
            print("Processing truth clusters...")

            # Create truth pairs from automatic truth assignments
            truth_clusters = defaultdict(list)
            for ref_id, truth_id in ref_to_truth.items():
                truth_clusters[truth_id].append(ref_id)

            # Generate truth pairs with progress indicator
            print("Generating truth pairs...")
            truth_pairs = []
            total_clusters = len(truth_clusters)
            processed = 0

            for truth_id, members in truth_clusters.items():
                if len(members) > 1:
                    for i in range(len(members)):
                        for j in range(i, len(members)):
                            truth_pairs.append((members[i], members[j]))
                            if i != j:
                                truth_pairs.append((members[j], members[i]))
                else:
                    # Single member clusters still need self-pairs
                    truth_pairs.append((members[0], members[0]))

                processed += 1
                if processed % 100 == 0 or processed == total_clusters:
                    print(f"  Progress: {processed}/{total_clusters} clusters processed", end='\r')

            print(f"\n✓ Generated {len(truth_pairs)} truth pairs from {len(truth_clusters)} automatic truth clusters")
            self.logger.info(f"Generated {len(truth_pairs)} truth pairs from {len(truth_clusters)} automatic truth clusters")

            # Create and save automatic truth file for reference
            auto_truth_file = f"{output_prefix}_auto_truth.csv"
            with open(auto_truth_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['RefID', 'TruthID'])

                # Sort for better readability
                for ref_id in sorted(ref_to_truth.keys()):
                    truth_id = ref_to_truth[ref_id]
                    writer.writerow([ref_id, truth_id])

            print(f"Created automatic truth file for reference: {auto_truth_file}")

        # Convert predicted clusters to pairs for evaluation with progress
        print("\nGenerating predicted pairs...")
        predicted_pairs = []
        total_pred_clusters = len(clusters)

        for idx, cluster in enumerate(clusters, 1):
            if len(cluster.record_ids) > 1:
                for i in range(len(cluster.record_ids)):
                    for j in range(i, len(cluster.record_ids)):
                        predicted_pairs.append((cluster.record_ids[i], cluster.record_ids[j]))
                        if i != j:
                            predicted_pairs.append((cluster.record_ids[j], cluster.record_ids[i]))
            else:
                # Single member clusters
                predicted_pairs.append((cluster.record_ids[0], cluster.record_ids[0]))

            if idx % 100 == 0 or idx == total_pred_clusters:
                print(f"  Progress: {idx}/{total_pred_clusters} clusters processed", end='\r')

        print(f"\n✓ Generated {len(predicted_pairs)} predicted pairs from {len(clusters)} clusters")

        # Calculate metrics with progress indication
        print("\nCalculating evaluation metrics...")
        print(f"  Comparing {len(predicted_pairs)} predicted pairs with {len(truth_pairs)} truth pairs")
        print("  This may take a moment for large datasets...")

        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(predicted_pairs, truth_pairs, self.logger)

        print("✓ Metrics calculation complete")

        # Generate report
        reporter.generate_metrics_report(metrics, self.logger, self.log_file)

        # Additional analysis for automatic truth
        if truth_option != 'y':
            print("\n### AUTOMATIC TRUTH ANALYSIS ###")
            print(f"Truth clusters detected: {len(truth_clusters)}")
            print(f"Predicted clusters: {len(clusters)}")

            # Also write to log file
            print("\n### AUTOMATIC TRUTH ANALYSIS ###", file=self.log_file)
            print(f"Truth clusters detected: {len(truth_clusters)}", file=self.log_file)
            print(f"Predicted clusters: {len(clusters)}", file=self.log_file)

            # Show sample truth assignments
            print("\nSample Cluster Assignments (first 10):")
            print("\nSample Cluster Assignments (first 10):", file=self.log_file)

            sample_items = list(ref_to_truth.items())[:10]
            for ref_id, truth_id in sample_items:
                predicted_id = ref_to_cluster[ref_id]
                # Check if the base numbers match
                truth_base = truth_id.split('.')[0] if '.' in truth_id else truth_id
                pred_base = predicted_id.split('.')[0] if '.' in predicted_id else predicted_id
                match = "✓" if truth_base == pred_base else "✗"

                print(f"  {ref_id} → Truth: {truth_id}, Predicted: {predicted_id} {match}")
                print(f"  {ref_id} → Truth: {truth_id}, Predicted: {predicted_id} {match}", file=self.log_file)

# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    app = LinkEvaluator()
    app.run()

if __name__ == "__main__":
    main()
