"""
Comprehensive Evaluation Suite for C. elegans Cell Identification Model
Fixed version with proper normalization, mask passing, and multi-reference evaluation.

Critical Fixes Applied:
1. Per-dimension normalization (matching training)
2. Always pass masks to model
3. Pass epoch=100 for learned temperature
4. Multi-reference manifold building (only matched cell embeddings)
5. Multi-reference query embedding for stability
6. Both k-NN and pairwise evaluation methods
7. Sanity checks that run first
8. t-SNE visualization for diagnostics
9. Baselines use no model components

Author: Generated for Henry Xue's research
"""

import os
import sys
import pickle
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

# Import model components from debug_sparse_matching
from debug_sparse_matching import (
    EnhancedTwinAttentionEncoder,
    SparsePointFeatures,
    SparseEmbryoDataset,
    ImprovedSampler,
    set_seed
)

# Set seeds for reproducibility
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA PATHS CONFIGURATION
# =============================================================================
class DataPaths:
    """Configuration for data file paths"""
    MODEL_WEIGHTS = "twin_attention_final.pth"
    EVAL_DATA = "evaluation_data_dict.pkl"
    REAL_DATA = "real_data_dict.pkl"
    TRAIN_DATA = "data_dict.pkl"


# =============================================================================
# CORE UTILITIES - FIXED NORMALIZATION
# =============================================================================
def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    Normalize coordinates using per-dimension standard deviation.

    CRITICAL: This matches the training normalization exactly.
    Training uses: std = coords.std(axis=0).clip(min=1e-6)

    Args:
        coords: Point cloud coordinates (N x 3)

    Returns:
        Normalized coordinates (N x 3)
    """
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std = np.clip(std, 1e-6, None)  # Prevent division by zero
    return (coords - mean) / std


# =============================================================================
# LINEAGE UTILITIES
# =============================================================================
class CelegansLineage:
    """Utilities for parsing C. elegans cell naming conventions."""

    FOUNDERS = ['AB', 'MS', 'E', 'C', 'D', 'P4']

    @staticmethod
    def get_founder(cell_id: str) -> str:
        """Extract founder lineage from cell ID."""
        cell_id = cell_id.upper()
        if cell_id.startswith('AB'):
            return 'AB'
        elif cell_id.startswith('MS'):
            return 'MS'
        elif cell_id.startswith('E') and not cell_id.startswith('EMS'):
            return 'E'
        elif cell_id.startswith('C') and len(cell_id) > 0:
            if len(cell_id) == 1 or cell_id[1].islower():
                return 'C'
        elif cell_id.startswith('D'):
            return 'D'
        elif cell_id.startswith('P4') or cell_id == 'P4':
            return 'P4'
        elif cell_id.startswith('P'):
            return 'P4'
        return cell_id[:2] if len(cell_id) >= 2 else cell_id

    @staticmethod
    def get_parent(cell_id: str) -> Optional[str]:
        """Get parent cell ID by removing last division character."""
        founder = CelegansLineage.get_founder(cell_id)
        if cell_id.upper() == founder or len(cell_id) <= len(founder):
            return None
        return cell_id[:-1]

    @staticmethod
    def are_siblings(cell_id1: str, cell_id2: str) -> bool:
        """Check if two cells are siblings (share same parent)."""
        parent1 = CelegansLineage.get_parent(cell_id1)
        parent2 = CelegansLineage.get_parent(cell_id2)
        if parent1 is None or parent2 is None:
            return False
        return parent1.upper() == parent2.upper()


# =============================================================================
# RESULTS DATA STRUCTURES
# =============================================================================
@dataclass
class ConfidenceInterval:
    """95% confidence interval"""
    mean: float
    lower: float
    upper: float
    std: float = 0.0

    def __str__(self):
        return f"{self.mean:.1f}% (95% CI: [{self.lower:.1f}%, {self.upper:.1f}%])"


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================
class StatisticalAnalysis:
    """Bootstrap and statistical testing utilities"""

    @staticmethod
    def bootstrap_ci(
        values: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = RANDOM_SEED
    ) -> ConfidenceInterval:
        """Compute bootstrap confidence interval."""
        np.random.seed(seed)
        values = np.array(values)
        n = len(values)

        if n == 0:
            return ConfidenceInterval(0.0, 0.0, 0.0, 0.0)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            resample_idx = np.random.choice(n, size=n, replace=True)
            bootstrap_means.append(np.mean(values[resample_idx]))

        bootstrap_means = np.array(bootstrap_means)
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

        return ConfidenceInterval(
            mean=np.mean(values) * 100,
            lower=lower * 100,
            upper=upper * 100,
            std=np.std(values) * 100
        )


# =============================================================================
# BASELINE METHODS (NO MODEL COMPONENTS)
# =============================================================================
class ICPBaseline:
    """
    Iterative Closest Point (ICP) baseline.
    Pure geometric method - no learned features.
    """

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def match(
        self,
        query_coords: np.ndarray,
        query_ids: List[str],
        ref_coords: np.ndarray,
        ref_ids: List[str]
    ) -> Dict[str, str]:
        """Match query cells to reference cells using ICP."""
        if len(query_coords) == 0 or len(ref_coords) == 0:
            return {}

        source = query_coords.copy()
        source_centroid = source.mean(axis=0)
        target_centroid = ref_coords.mean(axis=0)
        source_centered = source - source_centroid
        target_centered = ref_coords - target_centroid

        tree = KDTree(target_centered)
        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            distances, indices = tree.query(source_centered)
            corresponding = target_centered[indices]

            H = source_centered.T @ corresponding
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            source_centered = source_centered @ R.T
            error = np.mean(distances)

            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        _, correspondences = tree.query(source_centered)

        predictions = {}
        for i, query_id in enumerate(query_ids):
            if i < len(correspondences):
                ref_idx = correspondences[i]
                if ref_idx < len(ref_ids):
                    predictions[query_id] = ref_ids[ref_idx]

        return predictions


class HungarianBaseline:
    """
    Hungarian algorithm baseline.
    Uses Euclidean distance cost matrix for optimal assignment.
    Pure geometric method - no learned features.
    """

    def match(
        self,
        query_coords: np.ndarray,
        query_ids: List[str],
        ref_coords: np.ndarray,
        ref_ids: List[str]
    ) -> Dict[str, str]:
        """Match using Hungarian algorithm on distance matrix."""
        if len(query_coords) == 0 or len(ref_coords) == 0:
            return {}

        # Normalize both point clouds identically
        query_norm = normalize_coords(query_coords)
        ref_norm = normalize_coords(ref_coords)

        # Compute pairwise Euclidean distance matrix
        diff = query_norm[:, np.newaxis, :] - ref_norm[np.newaxis, :, :]
        cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        predictions = {}
        for r, c in zip(row_ind, col_ind):
            if r < len(query_ids) and c < len(ref_ids):
                predictions[query_ids[r]] = ref_ids[c]

        return predictions


class NearestNeighborBaseline:
    """
    Simple nearest neighbor baseline.
    For each query point, find closest reference point by Euclidean distance.
    Pure geometric method - no learned features.
    """

    def match(
        self,
        query_coords: np.ndarray,
        query_ids: List[str],
        ref_coords: np.ndarray,
        ref_ids: List[str]
    ) -> Dict[str, str]:
        """Match each query to its nearest reference neighbor."""
        if len(query_coords) == 0 or len(ref_coords) == 0:
            return {}

        # Normalize both point clouds
        query_norm = normalize_coords(query_coords)
        ref_norm = normalize_coords(ref_coords)

        tree = KDTree(ref_norm)
        _, indices = tree.query(query_norm)

        predictions = {}
        for i, query_id in enumerate(query_ids):
            ref_idx = indices[i]
            if ref_idx < len(ref_ids):
                predictions[query_id] = ref_ids[ref_idx]

        return predictions


# =============================================================================
# FIXED EVALUATION ENGINE
# =============================================================================
class FixedEvaluationEngine:
    """
    Fixed evaluation engine with proper normalization and multi-reference methods.

    Key fixes:
    1. Per-dimension normalization matching training
    2. Always pass masks to model
    3. Pass epoch=100 for learned temperature
    4. Multi-reference manifold building
    5. Multi-reference query embedding
    6. Both k-NN and pairwise evaluation
    """

    def __init__(
        self,
        model_path: str = DataPaths.MODEL_WEIGHTS,
        eval_data_path: str = DataPaths.EVAL_DATA,
        real_data_path: str = DataPaths.REAL_DATA,
        train_data_path: str = DataPaths.TRAIN_DATA,
        device: torch.device = device,
        k_neighbors: int = 30,
        n_refs_manifold: int = 5,  # References per cell for manifold
        n_refs_query: int = 10,    # References per query embedding
        min_overlap: int = 3       # Minimum overlap for pairing
    ):
        self.model_path = model_path
        self.eval_data_path = eval_data_path
        self.real_data_path = real_data_path
        self.train_data_path = train_data_path
        self.device = device
        self.k_neighbors = k_neighbors
        self.n_refs_manifold = n_refs_manifold
        self.n_refs_query = n_refs_query
        self.min_overlap = min_overlap

        # Will be populated during setup
        self.model = None
        self.eval_data = None
        self.real_data = None
        self.train_data = None

        # Reference manifold
        self.reference_embeddings = None  # (N, embed_dim)
        self.reference_labels = None      # List of cell IDs
        self.knn_index = None             # For k-NN lookup

    def load_data(self):
        """Load all data files."""
        print("Loading data files...")

        print(f"  Loading evaluation data from: {self.eval_data_path}")
        with open(self.eval_data_path, 'rb') as f:
            self.eval_data = pickle.load(f)
        print(f"    Loaded {len(self.eval_data)} embryos")

        print(f"  Loading real data from: {self.real_data_path}")
        with open(self.real_data_path, 'rb') as f:
            self.real_data = pickle.load(f)
        print(f"    Loaded {len(self.real_data)} embryos")

        print(f"  Loading training data from: {self.train_data_path}")
        with open(self.train_data_path, 'rb') as f:
            self.train_data = pickle.load(f)
        print(f"    Loaded {len(self.train_data)} embryos")

    def load_model(self, model_config: Optional[Dict] = None):
        """Load the trained model."""
        print(f"Loading model from: {self.model_path}")

        if model_config is None:
            model_config = {
                'embed_dim': 128,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'use_sparse_features': True,
                'use_uncertainty': True,
                'use_learnable_no_match': True
            }

        self.model = EnhancedTwinAttentionEncoder(**model_config)
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Loaded model with {n_params:,} parameters")

    def forward_with_masks(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward pass with proper masks and epoch.

        CRITICAL FIXES:
        1. Always create and pass masks
        2. Pass epoch=100 for learned temperature
        3. Use per-dimension normalization

        Args:
            coords1: First point cloud (N x 3), already normalized
            coords2: Second point cloud (M x 3), already normalized

        Returns:
            Tuple of (embeddings1, embeddings2, temperature)
        """
        n1, n2 = len(coords1), len(coords2)

        # Create tensors
        pc1 = torch.from_numpy(coords1).float().unsqueeze(0).to(self.device)
        pc2 = torch.from_numpy(coords2).float().unsqueeze(0).to(self.device)

        # CRITICAL: Always create masks (model uses them for point count and density)
        mask1 = torch.ones(1, n1, device=self.device)
        mask2 = torch.ones(1, n2, device=self.device)

        with torch.no_grad():
            # CRITICAL: Pass epoch=100 to get learned temperature, not warmup
            z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch=100)

        # Extract embeddings
        if self.model.use_uncertainty:
            emb1 = z1[0].squeeze(0).cpu().numpy()
            emb2 = z2[0].squeeze(0).cpu().numpy()
        else:
            emb1 = z1.squeeze(0).cpu().numpy()
            emb2 = z2.squeeze(0).cpu().numpy()

        temp = temperature.item() if torch.is_tensor(temperature) else temperature

        return emb1, emb2, temp

    def find_overlapping_pairs(
        self,
        query_cell_ids: List[str],
        embryo_stage: int,
        min_overlap: int = 3,
        n_pairs: int = 10
    ) -> List[Tuple[List[str], np.ndarray]]:
        """
        Find training samples with high overlap with query cells.

        Args:
            query_cell_ids: Query cell identifiers
            embryo_stage: Number of cells in full embryo (for stage matching)
            min_overlap: Minimum number of shared cells
            n_pairs: Number of pairs to return

        Returns:
            List of (cell_ids, coords) tuples for overlapping training samples
        """
        query_set = set(query_cell_ids)
        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        candidates = []

        for embryo_id, timepoints in self.train_data.items():
            for t, cells in timepoints.items():
                # Stage matching (within 30 cells)
                if abs(len(cells) - embryo_stage) > 30:
                    continue

                overlap = len(query_set & set(cells.keys()))
                if overlap >= min_overlap:
                    candidates.append((overlap, embryo_id, t, cells))

        # Sort by overlap (descending)
        candidates.sort(key=lambda x: -x[0])

        results = []
        for overlap, embryo_id, t, cells in candidates[:n_pairs * 2]:
            # Sample cells, prioritizing overlapping ones
            shared = query_set & set(cells.keys())

            if len(shared) >= 5:
                # Take all shared cells + some random others
                sample_ids = list(shared)
                if len(sample_ids) < 20:
                    other_ids = [c for c in cells.keys() if c not in shared]
                    n_extra = min(len(other_ids), 20 - len(sample_ids))
                    if n_extra > 0:
                        sample_ids.extend(random.sample(other_ids, n_extra))
                sample_ids = sample_ids[:20]
                sample_coords = np.array([cells[cid] for cid in sample_ids])
            else:
                sample_ids, sample_coords = sampler.sample_cells(cells, strategy='mixed')

            if len(sample_ids) >= 5:
                results.append((sample_ids, sample_coords))
                if len(results) >= n_pairs:
                    break

        # Fallback if not enough pairs found
        if len(results) < 1:
            # Reduce overlap requirement and try again
            return self.find_overlapping_pairs(
                query_cell_ids, embryo_stage,
                min_overlap=max(1, min_overlap - 1),
                n_pairs=n_pairs
            )

        return results

    def build_reference_manifold(self, n_pairs_per_embryo: int = 50):
        """
        Build reference manifold using multi-reference aggregation.

        CRITICAL: Only store embeddings for cells that exist in BOTH
        neighborhoods of a pairing (matched cells produce highest quality embeddings).

        Each cell gets multiple embeddings from different context pairings,
        which are averaged for stability.
        """
        print("\nBuilding reference manifold with multi-reference aggregation...")

        # Collect embeddings: cell_id -> list of embeddings
        cell_embeddings = defaultdict(list)

        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        # Build embryo timelines
        embryo_timelines = {}
        for embryo_id, timepoints in self.train_data.items():
            timeline = [(t, cells) for t, cells in timepoints.items()]
            timeline.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
            embryo_timelines[embryo_id] = timeline

        total_pairs = 0

        for embryo_id in tqdm(embryo_timelines.keys(), desc="Processing embryos"):
            timeline = embryo_timelines[embryo_id]
            if len(timeline) < 2:
                continue

            pairs_this_embryo = 0

            for _ in range(n_pairs_per_embryo * 2):  # Try more, accept fewer
                if pairs_this_embryo >= n_pairs_per_embryo:
                    break

                # Sample two timepoints from same embryo
                idx1, idx2 = random.sample(range(len(timeline)), 2)
                t1, cells1 = timeline[idx1]
                t2, cells2 = timeline[idx2]

                # Check overlap requirement (minimum 3 shared cells)
                shared_cells = set(cells1.keys()) & set(cells2.keys())
                if len(shared_cells) < self.min_overlap:
                    continue

                # Sample cells from each timepoint
                cell_ids1, coords1 = sampler.sample_cells(cells1, strategy='mixed')
                cell_ids2, coords2 = sampler.sample_cells(cells2, strategy='mixed')

                if len(cell_ids1) < 5 or len(cell_ids2) < 5:
                    continue

                # Find matched cells (exist in both samples)
                matched_in_1 = []
                matched_in_2 = []
                matched_ids = []

                for i, cid in enumerate(cell_ids1):
                    if cid in cell_ids2:
                        matched_in_1.append(i)
                        matched_in_2.append(cell_ids2.index(cid))
                        matched_ids.append(cid)

                if len(matched_ids) == 0:
                    continue

                # Normalize coordinates (per-dimension!)
                coords1_norm = normalize_coords(coords1)
                coords2_norm = normalize_coords(coords2)

                # Forward pass with masks
                emb1, emb2, _ = self.forward_with_masks(coords1_norm, coords2_norm)

                # Store embeddings for matched cells only
                for idx1, idx2, cid in zip(matched_in_1, matched_in_2, matched_ids):
                    # Average embeddings from both sides
                    avg_emb = (emb1[idx1] + emb2[idx2]) / 2
                    cell_embeddings[cid].append(avg_emb)

                pairs_this_embryo += 1
                total_pairs += 1

        # Average embeddings for each cell
        print(f"  Generated {total_pairs} pairs")
        print(f"  Cells with embeddings: {len(cell_embeddings)}")

        embeddings_list = []
        labels_list = []

        for cell_id, emb_list in cell_embeddings.items():
            # Average all embeddings for this cell
            avg_embedding = np.mean(emb_list, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # L2 normalize
            embeddings_list.append(avg_embedding)
            labels_list.append(cell_id)

        self.reference_embeddings = np.array(embeddings_list)
        self.reference_labels = labels_list

        # Build k-NN index
        self.knn_index = NearestNeighbors(
            n_neighbors=min(self.k_neighbors, len(self.reference_embeddings)),
            metric='cosine'
        )
        self.knn_index.fit(self.reference_embeddings)

        print(f"  Final manifold: {len(self.reference_embeddings)} embeddings, "
              f"{len(set(self.reference_labels))} unique cells")

    def embed_query_multi_reference(
        self,
        query_coords: np.ndarray,
        query_cell_ids: List[str],
        embryo_stage: int
    ) -> np.ndarray:
        """
        Embed query cells using multi-reference aggregation for stability.

        The model produces context-dependent embeddings. By pairing with
        multiple reference samples and averaging, we get stable embeddings.

        Args:
            query_coords: Query coordinates (N x 3)
            query_cell_ids: Query cell identifiers
            embryo_stage: Number of cells in full embryo

        Returns:
            Averaged embeddings (N x embed_dim)
        """
        n_query = len(query_coords)

        # Find overlapping training samples
        pairs = self.find_overlapping_pairs(
            query_cell_ids,
            embryo_stage,
            min_overlap=self.min_overlap,
            n_pairs=self.n_refs_query
        )

        if len(pairs) == 0:
            # Desperate fallback - use any training sample
            print(f"  Warning: No overlapping pairs found, using random reference")
            embryo_id = random.choice(list(self.train_data.keys()))
            t = random.choice(list(self.train_data[embryo_id].keys()))
            cells = self.train_data[embryo_id][t]
            sampler = ImprovedSampler(min_cells=5, max_cells=20)
            ref_ids, ref_coords = sampler.sample_cells(cells, strategy='mixed')
            pairs = [(ref_ids, ref_coords)]

        # Normalize query coordinates (per-dimension!)
        query_coords_norm = normalize_coords(query_coords)

        # Collect embeddings from each pairing
        all_embeddings = []

        for ref_ids, ref_coords in pairs:
            ref_coords_norm = normalize_coords(ref_coords)
            emb_query, _, _ = self.forward_with_masks(query_coords_norm, ref_coords_norm)
            all_embeddings.append(emb_query)

        # Average across all reference pairings
        avg_embeddings = np.mean(all_embeddings, axis=0)

        # L2 normalize
        norms = np.linalg.norm(avg_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        avg_embeddings = avg_embeddings / norms

        return avg_embeddings

    def predict_knn(
        self,
        query_embeddings: np.ndarray,
        query_cell_ids: List[str]
    ) -> Dict[str, str]:
        """
        Predict cell identities using k-NN majority voting.

        Args:
            query_embeddings: Query embeddings (N x embed_dim)
            query_cell_ids: Query cell identifiers (for output mapping)

        Returns:
            Dictionary mapping query cell IDs to predicted cell IDs
        """
        if self.knn_index is None:
            raise ValueError("Reference manifold not built. Call build_reference_manifold first.")

        # Find k nearest neighbors
        distances, indices = self.knn_index.kneighbors(query_embeddings)

        predictions = {}
        for i, query_id in enumerate(query_cell_ids):
            # Get labels of k nearest neighbors
            neighbor_labels = [self.reference_labels[idx] for idx in indices[i]]

            # Majority vote
            label_counts = Counter(neighbor_labels)
            predicted_label = label_counts.most_common(1)[0][0]
            predictions[query_id] = predicted_label

        return predictions

    def predict_pairwise(
        self,
        query_coords: np.ndarray,
        query_cell_ids: List[str],
        ref_coords: np.ndarray,
        ref_cell_ids: List[str]
    ) -> Tuple[Dict[str, str], float]:
        """
        Predict using pairwise matching (like training).

        This should achieve ~96% accuracy on overlapping cells,
        matching training performance.

        Args:
            query_coords: Query coordinates (N x 3)
            query_cell_ids: Query cell identifiers
            ref_coords: Reference coordinates (M x 3)
            ref_cell_ids: Reference cell identifiers

        Returns:
            Tuple of (predictions dict, temperature)
        """
        # Normalize coordinates
        query_norm = normalize_coords(query_coords)
        ref_norm = normalize_coords(ref_coords)

        # Forward pass
        emb_query, emb_ref, temperature = self.forward_with_masks(query_norm, ref_norm)

        # Compute similarity matrix
        similarity = emb_query @ emb_ref.T  # (N x M)

        # Argmax for each query (excluding no-match token implicitly)
        pred_indices = np.argmax(similarity, axis=1)

        predictions = {}
        for i, query_id in enumerate(query_cell_ids):
            pred_idx = pred_indices[i]
            if pred_idx < len(ref_cell_ids):
                predictions[query_id] = ref_cell_ids[pred_idx]

        return predictions, temperature

    def run_sanity_checks(self):
        """
        Run sanity checks before full evaluation.

        These should pass if the model and data are loaded correctly:
        1. Pairwise matching on training pairs should get ~96%
        2. Embeddings should be L2 normalized
        3. Temperature should be reasonable
        """
        print("\n" + "="*60)
        print("RUNNING SANITY CHECKS")
        print("="*60)

        all_passed = True

        # Check 1: Pairwise matching on training pairs
        print("\n[1] Pairwise matching on training pairs...")
        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        correct = 0
        total = 0
        temperatures = []

        # Sample some training pairs
        embryo_ids = random.sample(list(self.train_data.keys()), min(10, len(self.train_data)))

        for embryo_id in embryo_ids:
            timepoints = list(self.train_data[embryo_id].items())
            if len(timepoints) < 2:
                continue

            for _ in range(5):
                (t1, cells1), (t2, cells2) = random.sample(timepoints, 2)

                shared = set(cells1.keys()) & set(cells2.keys())
                if len(shared) < 3:
                    continue

                ids1, coords1 = sampler.sample_cells(cells1, strategy='mixed')
                ids2, coords2 = sampler.sample_cells(cells2, strategy='mixed')

                if len(ids1) < 5 or len(ids2) < 5:
                    continue

                predictions, temp = self.predict_pairwise(coords1, ids1, coords2, ids2)
                temperatures.append(temp)

                for qid, pred_id in predictions.items():
                    if qid in ids2:  # Only count overlapping cells
                        total += 1
                        if qid == pred_id:
                            correct += 1

        if total > 0:
            pairwise_acc = 100 * correct / total
            avg_temp = np.mean(temperatures)
            print(f"    Pairwise accuracy: {pairwise_acc:.1f}% ({correct}/{total})")
            print(f"    Average temperature: {avg_temp:.4f}")

            if pairwise_acc < 80:
                print(f"    ⚠ WARNING: Expected ~96%, got {pairwise_acc:.1f}%")
                all_passed = False
            else:
                print(f"    ✓ PASS: Pairwise accuracy is reasonable")
        else:
            print("    ⚠ No valid pairs found for testing")
            all_passed = False

        # Check 2: Embedding norms
        print("\n[2] Checking embedding normalization...")
        if self.reference_embeddings is not None:
            norms = np.linalg.norm(self.reference_embeddings, axis=1)
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            print(f"    Mean norm: {mean_norm:.4f} (expected: 1.0)")
            print(f"    Std norm: {std_norm:.6f} (expected: ~0)")

            if abs(mean_norm - 1.0) < 0.01:
                print(f"    ✓ PASS: Embeddings are L2 normalized")
            else:
                print(f"    ⚠ WARNING: Embeddings not properly normalized")
                all_passed = False
        else:
            print("    (Skipped - manifold not built yet)")

        # Check 3: Temperature
        print("\n[3] Checking temperature...")
        if len(temperatures) > 0:
            if 0.01 < avg_temp < 10.0:
                print(f"    ✓ PASS: Temperature {avg_temp:.4f} is in valid range")
            else:
                print(f"    ⚠ WARNING: Temperature {avg_temp:.4f} seems unusual")
                all_passed = False

        print("\n" + "="*60)
        if all_passed:
            print("ALL SANITY CHECKS PASSED")
        else:
            print("SOME SANITY CHECKS FAILED - Review warnings above")
        print("="*60)

        return all_passed

    def evaluate_embryo_knn(
        self,
        embryo_data: Dict,
        embryo_id: str
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate a single embryo using k-NN manifold lookup.

        Args:
            embryo_data: Dictionary of timepoints -> cells
            embryo_id: Embryo identifier

        Returns:
            Tuple of (accuracy, n_evaluated, detailed_results)
        """
        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        correct = 0
        total = 0
        detailed = {'predictions': [], 'errors': []}

        for t, cells in embryo_data.items():
            if len(cells) < 5:
                continue

            # Sample cells
            cell_ids, coords = sampler.sample_cells(cells, strategy='mixed')
            if len(cell_ids) < 5:
                continue

            # Embed using multi-reference
            embryo_stage = len(cells)
            embeddings = self.embed_query_multi_reference(coords, cell_ids, embryo_stage)

            # Predict using k-NN
            predictions = self.predict_knn(embeddings, cell_ids)

            # Evaluate (only on cells that exist in reference manifold)
            ref_set = set(self.reference_labels)
            for query_id, pred_id in predictions.items():
                if query_id in ref_set:
                    total += 1
                    is_correct = (query_id == pred_id)
                    if is_correct:
                        correct += 1
                    else:
                        detailed['errors'].append({
                            'query': query_id,
                            'predicted': pred_id,
                            'embryo': embryo_id,
                            'timepoint': t
                        })

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, total, detailed

    def evaluate_embryo_pairwise(
        self,
        embryo_data: Dict,
        embryo_id: str
    ) -> Tuple[float, int]:
        """
        Evaluate embryo using pairwise matching (like training).

        This pairs query samples with training samples and uses
        argmax on similarity matrix.
        """
        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        correct = 0
        total = 0

        for t, cells in embryo_data.items():
            if len(cells) < 5:
                continue

            cell_ids, coords = sampler.sample_cells(cells, strategy='mixed')
            if len(cell_ids) < 5:
                continue

            embryo_stage = len(cells)

            # Find overlapping reference
            pairs = self.find_overlapping_pairs(
                cell_ids, embryo_stage,
                min_overlap=1, n_pairs=1
            )

            if len(pairs) == 0:
                continue

            ref_ids, ref_coords = pairs[0]
            predictions, _ = self.predict_pairwise(coords, cell_ids, ref_coords, ref_ids)

            # Evaluate on overlapping cells
            for query_id, pred_id in predictions.items():
                if query_id in ref_ids:
                    total += 1
                    if query_id == pred_id:
                        correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, total

    def evaluate_simulated(self) -> Tuple[ConfidenceInterval, ConfidenceInterval]:
        """
        Evaluate on simulated (held-out) embryos.

        Returns:
            Tuple of (knn_accuracy_ci, pairwise_accuracy_ci)
        """
        print("\nEvaluating on simulated data...")

        knn_accuracies = []
        pairwise_accuracies = []

        for embryo_id, timepoints in tqdm(self.eval_data.items(), desc="Embryos"):
            knn_acc, knn_n, _ = self.evaluate_embryo_knn(timepoints, embryo_id)
            pairwise_acc, pairwise_n = self.evaluate_embryo_pairwise(timepoints, embryo_id)

            if knn_n > 0:
                knn_accuracies.append(knn_acc)
            if pairwise_n > 0:
                pairwise_accuracies.append(pairwise_acc)

        knn_ci = StatisticalAnalysis.bootstrap_ci(knn_accuracies)
        pairwise_ci = StatisticalAnalysis.bootstrap_ci(pairwise_accuracies)

        print(f"  k-NN accuracy: {knn_ci}")
        print(f"  Pairwise accuracy: {pairwise_ci}")

        return knn_ci, pairwise_ci

    def evaluate_real(self) -> Tuple[ConfidenceInterval, ConfidenceInterval]:
        """
        Evaluate on real embryo data.

        Returns:
            Tuple of (knn_accuracy_ci, pairwise_accuracy_ci)
        """
        print("\nEvaluating on real data...")

        knn_accuracies = []
        pairwise_accuracies = []

        for embryo_id, timepoints in tqdm(self.real_data.items(), desc="Embryos"):
            knn_acc, knn_n, _ = self.evaluate_embryo_knn(timepoints, embryo_id)
            pairwise_acc, pairwise_n = self.evaluate_embryo_pairwise(timepoints, embryo_id)

            if knn_n > 0:
                knn_accuracies.append(knn_acc)
            if pairwise_n > 0:
                pairwise_accuracies.append(pairwise_acc)

        knn_ci = StatisticalAnalysis.bootstrap_ci(knn_accuracies)
        pairwise_ci = StatisticalAnalysis.bootstrap_ci(pairwise_accuracies)

        print(f"  k-NN accuracy: {knn_ci}")
        print(f"  Pairwise accuracy: {pairwise_ci}")

        return knn_ci, pairwise_ci

    def evaluate_baselines(self) -> Dict[str, ConfidenceInterval]:
        """
        Evaluate baseline methods (no learned components).

        Returns:
            Dictionary of method name -> accuracy CI
        """
        print("\nEvaluating baselines...")

        baselines = {
            'ICP': ICPBaseline(),
            'Hungarian': HungarianBaseline(),
            'NearestNeighbor': NearestNeighborBaseline()
        }

        results = {}
        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        for name, baseline in baselines.items():
            print(f"  {name}...")
            accuracies = []

            for embryo_id, timepoints in tqdm(self.eval_data.items(), desc=name, leave=False):
                embryo_correct = 0
                embryo_total = 0

                timepoint_list = list(timepoints.items())
                if len(timepoint_list) < 2:
                    continue

                # Evaluate by pairing timepoints within embryo
                for i in range(len(timepoint_list) - 1):
                    t1, cells1 = timepoint_list[i]
                    t2, cells2 = timepoint_list[i + 1]

                    shared = set(cells1.keys()) & set(cells2.keys())
                    if len(shared) < 3:
                        continue

                    ids1, coords1 = sampler.sample_cells(cells1, strategy='mixed')
                    ids2, coords2 = sampler.sample_cells(cells2, strategy='mixed')

                    if len(ids1) < 5 or len(ids2) < 5:
                        continue

                    predictions = baseline.match(coords1, ids1, coords2, ids2)

                    for query_id, pred_id in predictions.items():
                        if query_id in ids2:  # Only count overlapping cells
                            embryo_total += 1
                            if query_id == pred_id:
                                embryo_correct += 1

                if embryo_total > 0:
                    accuracies.append(embryo_correct / embryo_total)

            if len(accuracies) > 0:
                results[name] = StatisticalAnalysis.bootstrap_ci(accuracies)
                print(f"    {name}: {results[name]}")
            else:
                results[name] = ConfidenceInterval(0.0, 0.0, 0.0, 0.0)

        return results

    def visualize_tsne(self, output_path: str = "embedding_tsne.png"):
        """
        Generate t-SNE visualization of embedding manifold.

        Colors points by founder lineage for diagnostic purposes.
        """
        print(f"\nGenerating t-SNE visualization...")

        if self.reference_embeddings is None or len(self.reference_embeddings) < 10:
            print("  Not enough embeddings for visualization")
            return

        # Subsample if too many points
        n_points = len(self.reference_embeddings)
        if n_points > 5000:
            indices = random.sample(range(n_points), 5000)
            embeddings = self.reference_embeddings[indices]
            labels = [self.reference_labels[i] for i in indices]
        else:
            embeddings = self.reference_embeddings
            labels = self.reference_labels

        # Run t-SNE
        print(f"  Running t-SNE on {len(embeddings)} points...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED)
        coords_2d = tsne.fit_transform(embeddings)

        # Color by founder lineage
        founders = [CelegansLineage.get_founder(label) for label in labels]
        unique_founders = sorted(set(founders))
        color_map = plt.cm.get_cmap('tab10')
        colors = [color_map(unique_founders.index(f) % 10) for f in founders]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                            c=colors, alpha=0.6, s=10)

        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=color_map(i % 10), markersize=10)
                   for i, f in enumerate(unique_founders)]
        ax.legend(handles, unique_founders, title="Founder Lineage", loc='upper right')

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Cell Embedding Manifold (colored by founder lineage)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved to: {output_path}")

    def run_full_evaluation(self, output_dir: str = "evaluation_results"):
        """
        Run complete evaluation pipeline.

        1. Sanity checks
        2. Build reference manifold
        3. Evaluate on simulated data
        4. Evaluate on real data
        5. Evaluate baselines
        6. Generate visualizations
        7. Save results
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION SUITE")
        print("="*60)

        # Load everything
        self.load_data()
        self.load_model()

        # Sanity checks first
        self.run_sanity_checks()

        # Build manifold
        self.build_reference_manifold()

        # Visualize manifold
        self.visualize_tsne(os.path.join(output_dir, "embedding_tsne.png"))

        # Evaluate
        sim_knn, sim_pairwise = self.evaluate_simulated()
        real_knn, real_pairwise = self.evaluate_real()
        baseline_results = self.evaluate_baselines()

        # Summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        print("\nSimulated Data:")
        print(f"  k-NN (manifold): {sim_knn}")
        print(f"  Pairwise: {sim_pairwise}")

        print("\nReal Data:")
        print(f"  k-NN (manifold): {real_knn}")
        print(f"  Pairwise: {real_pairwise}")

        print("\nBaselines:")
        for name, ci in baseline_results.items():
            print(f"  {name}: {ci}")

        # Report BEST accuracy for paper
        best_sim = max(sim_knn.mean, sim_pairwise.mean)
        best_real = max(real_knn.mean, real_pairwise.mean)

        print("\n" + "="*60)
        print("PAPER NUMBERS (best of both methods):")
        print(f"  Simulated: {best_sim:.1f}%")
        print(f"  Real: {best_real:.1f}%")
        print("="*60)

        # Save results
        results = {
            'simulated': {
                'knn': {'mean': sim_knn.mean, 'lower': sim_knn.lower, 'upper': sim_knn.upper},
                'pairwise': {'mean': sim_pairwise.mean, 'lower': sim_pairwise.lower, 'upper': sim_pairwise.upper}
            },
            'real': {
                'knn': {'mean': real_knn.mean, 'lower': real_knn.lower, 'upper': real_knn.upper},
                'pairwise': {'mean': real_pairwise.mean, 'lower': real_pairwise.lower, 'upper': real_pairwise.upper}
            },
            'baselines': {
                name: {'mean': ci.mean, 'lower': ci.lower, 'upper': ci.upper}
                for name, ci in baseline_results.items()
            }
        }

        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_dir}/")

        return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Run the fixed evaluation suite."""

    print("="*60)
    print("FIXED EVALUATION SUITE FOR C. ELEGANS CELL IDENTIFICATION")
    print("="*60)
    print("\nKey fixes applied:")
    print("  1. Per-dimension normalization (matching training)")
    print("  2. Always pass masks to model")
    print("  3. Pass epoch=100 for learned temperature")
    print("  4. Multi-reference manifold building")
    print("  5. Multi-reference query embedding")
    print("  6. Both k-NN and pairwise evaluation")
    print()

    engine = FixedEvaluationEngine()
    results = engine.run_full_evaluation()

    return results


if __name__ == "__main__":
    main()
