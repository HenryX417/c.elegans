"""
Comprehensive Evaluation Suite for C. elegans Cell Identification Model
Fixed version with proper normalization, mask passing, and TRUE no-cheating evaluation.

=== EVALUATION PHILOSOPHY ===
The key insight: In real deployment, you DON'T know which cells you're looking at.
You only know:
  - The 3D coordinates of 5-20 observed cells
  - The total embryo cell count (from imaging)

The "no-cheating" evaluation tests this TRUE deployment scenario:
  INPUT: coordinates + embryo_stage (cell count only)
  OUTPUT: predicted cell identities
  NO ACCESS TO: Ground truth cell IDs during inference

=== CRITICAL FIXES APPLIED ===
1. Per-dimension normalization (matching training)
2. Always pass masks to model
3. Pass epoch=100 for learned temperature
4. TRUE NO-CHEATING: find_stage_matched_references() uses stage ONLY
5. Stage-matched voting: identify_cells_no_cheating() for deployment scenario
6. Stage-binned manifold for efficient k-NN lookup
7. Hierarchical accuracy (exact, sublineage, founder, binary AB vs non-AB)
8. Accuracy by neighborhood size (sparse 5-10, medium 11-15, dense 16-20)
9. Comprehensive t-SNE visualizations for paper
10. Sanity checks that run first (pairwise with known overlap)
11. Data diagnostics (diagnose_real_data, check_manifold_coverage)
12. Fair baselines using cross-embryo matching (same task as model)

=== EXPECTED RESULTS ===
No-Cheating (stage-matched voting):
  - Simulated: 75-85%
  - Real: 60-75%

Stage-Aware k-NN:
  - Simulated: 55-70%

Baselines (fair comparison):
  - 40-55%

=== KEY METHODS ===
- identify_cells_no_cheating(): TRUE deployment scenario
- find_stage_matched_references(): Stage-only reference finding
- build_stage_binned_manifold(): Stage-aware k-NN
- evaluate_no_cheating(): Full no-cheating evaluation
- visualize_manifold_comprehensive(): Paper-quality figures

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
# SIAMESE TRANSFORMER (for ablation studies)
# =============================================================================
class SiameseTransformer(nn.Module):
    """
    Siamese Transformer baseline - processes each neighborhood INDEPENDENTLY.

    This is the key architectural comparison:
    - Same capacity as Joint Attention model
    - Same feature extraction
    - BUT: No cross-neighborhood attention during encoding
    - Comparison happens AFTER independent encoding via similarity

    This demonstrates the importance of joint attention - without it,
    the model cannot learn correspondence-dependent representations.
    """

    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_sparse_features: bool = True,
        use_uncertainty: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_sparse_features = use_sparse_features
        self.use_uncertainty = use_uncertainty

        # Feature extraction (shared between both branches)
        if use_sparse_features:
            self.sparse_features = SparsePointFeatures(embed_dim)
            self.feature_projection = nn.Linear(embed_dim, embed_dim)
        else:
            self.point_embed = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, embed_dim) * 0.02)

        # SINGLE shared transformer encoder (Siamese = shared weights)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projections
        if use_uncertainty:
            self.output_mean = nn.Linear(embed_dim, embed_dim)
            self.output_logvar = nn.Linear(embed_dim, embed_dim)
        else:
            self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Temperature
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def encode_single(self, pc: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a SINGLE point cloud independently.

        This is the key difference from Joint Attention:
        - Each neighborhood is processed in isolation
        - No information about the comparison neighborhood
        """
        B, N, _ = pc.shape

        # Extract features
        if self.use_sparse_features:
            z = self.sparse_features(pc, mask)
            z = self.feature_projection(z)
        else:
            z = self.point_embed(pc)

        # Add positional encoding
        z = z + self.pos_encoding[:, :N, :]

        # Create attention mask
        if mask is not None:
            attn_mask = ~mask.bool()
        else:
            attn_mask = None

        # Transform INDEPENDENTLY
        if attn_mask is not None:
            z = self.transformer(z, src_key_padding_mask=attn_mask)
        else:
            z = self.transformer(z)

        # Output projection with L2 normalization
        if self.use_uncertainty:
            z_mean = F.normalize(self.output_mean(z), p=2, dim=-1)
            z_logvar = torch.clamp(self.output_logvar(z), -10, 2)
            return (z_mean, z_logvar)
        else:
            z = F.normalize(self.output_proj(z), p=2, dim=-1)
            return z

    def forward(
        self,
        pc1: torch.Tensor,
        pc2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
        epoch: int = 0
    ):
        """
        Forward pass - encode both neighborhoods INDEPENDENTLY.

        Unlike Joint Attention which concatenates and processes together,
        Siamese processes each neighborhood separately then compares.
        """
        # Encode each neighborhood independently (NO joint attention)
        z1 = self.encode_single(pc1, mask1)
        z2 = self.encode_single(pc2, mask2)

        # Get temperature
        temperature = torch.exp(self.log_temperature).clamp(0.01, 10.0)

        return z1, z2, temperature


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

    @staticmethod
    def get_sublineage(cell_id: str, depth: int = 2) -> str:
        """Extract sublineage at specified depth from founder."""
        founder = CelegansLineage.get_founder(cell_id)
        cell_upper = cell_id.upper()
        if cell_upper.startswith(founder):
            suffix = cell_id[len(founder):]
        else:
            suffix = ""
        return founder + suffix[:depth]

    @staticmethod
    def same_sublineage(cell_id1: str, cell_id2: str, depth: int = 2) -> bool:
        """Check if two cells share sublineage at given depth."""
        return CelegansLineage.get_sublineage(cell_id1, depth).upper() == \
               CelegansLineage.get_sublineage(cell_id2, depth).upper()

    @staticmethod
    def same_founder(cell_id1: str, cell_id2: str) -> bool:
        """Check if two cells share founder lineage."""
        return CelegansLineage.get_founder(cell_id1) == CelegansLineage.get_founder(cell_id2)


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

    def diagnose_real_data(self):
        """Check if real embryo cell names match training data"""
        train_cells = set()
        for embryo in self.train_data.values():
            for cells in embryo.values():
                train_cells.update(cells.keys())

        eval_cells = set()
        for embryo in self.eval_data.values():
            for cells in embryo.values():
                eval_cells.update(cells.keys())

        real_cells = set()
        for embryo in self.real_data.values():
            for cells in embryo.values():
                real_cells.update(cells.keys())

        train_eval_overlap = train_cells & eval_cells
        train_real_overlap = train_cells & real_cells

        print(f"\n{'='*60}")
        print("DATA DIAGNOSTIC")
        print(f"{'='*60}")
        print(f"\nCell name coverage:")
        print(f"  Training unique cells: {len(train_cells)}")
        print(f"  Eval unique cells: {len(eval_cells)}")
        print(f"  Real unique cells: {len(real_cells)}")
        print(f"\n  Train-Eval overlap: {len(train_eval_overlap)} ({100*len(train_eval_overlap)/max(1,len(eval_cells)):.1f}% of eval)")
        print(f"  Train-Real overlap: {len(train_real_overlap)} ({100*len(train_real_overlap)/max(1,len(real_cells)):.1f}% of real)")

        print(f"\n  Sample training names: {sorted(list(train_cells))[:10]}")
        print(f"  Sample eval names: {sorted(list(eval_cells))[:10]}")
        print(f"  Sample real names: {sorted(list(real_cells))[:10]}")

        if len(train_real_overlap) < len(real_cells) * 0.5:
            print("\n  ⚠ WARNING: Real embryo cell names don't match training data!")
            print("    Real data accuracy will be low due to naming mismatch.")

    def check_manifold_coverage(self):
        """Check what % of test cells are in manifold"""
        if self.reference_labels is None:
            print("  Manifold not built yet")
            return

        eval_cells = set()
        for embryo in self.eval_data.values():
            for cells in embryo.values():
                eval_cells.update(cells.keys())

        manifold_cells = set(self.reference_labels)
        coverage = len(eval_cells & manifold_cells) / max(1, len(eval_cells))

        print(f"\n{'='*60}")
        print("MANIFOLD COVERAGE")
        print(f"{'='*60}")
        print(f"  Eval unique cells: {len(eval_cells)}")
        print(f"  Manifold unique cells: {len(manifold_cells)}")
        print(f"  Coverage: {coverage:.1%}")

        if coverage < 0.8:
            missing = eval_cells - manifold_cells
            print(f"\n  ⚠ WARNING: Only {coverage:.1%} of eval cells in manifold!")
            print(f"  Missing cells (sample): {sorted(list(missing))[:20]}")

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

        # Try progressively lower overlap thresholds (no recursion!)
        for current_min_overlap in range(min_overlap, 0, -1):
            candidates = []

            for embryo_id, timepoints in self.train_data.items():
                for t, cells in timepoints.items():
                    # Relaxed stage matching (within 50 cells)
                    if abs(len(cells) - embryo_stage) > 50:
                        continue

                    overlap = len(query_set & set(cells.keys()))
                    if overlap >= current_min_overlap:
                        candidates.append((overlap, embryo_id, t, cells))

            if len(candidates) == 0:
                continue

            # Sort by overlap (descending)
            candidates.sort(key=lambda x: -x[0])

            results = []
            for overlap, embryo_id, t, cells in candidates[:n_pairs * 3]:
                # Sample cells, prioritizing overlapping ones
                shared = query_set & set(cells.keys())

                if len(shared) >= 3:
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

            if len(results) >= 1:
                return results

        # Ultimate fallback: random samples (no overlap requirement)
        print(f"  Warning: No overlapping pairs found for {len(query_cell_ids)} query cells, using random samples")
        results = []
        embryo_ids = random.sample(list(self.train_data.keys()), min(n_pairs * 2, len(self.train_data)))
        for embryo_id in embryo_ids:
            timepoints = list(self.train_data[embryo_id].items())
            if len(timepoints) == 0:
                continue
            t, cells = random.choice(timepoints)
            sample_ids, sample_coords = sampler.sample_cells(cells, strategy='mixed')
            if len(sample_ids) >= 5:
                results.append((sample_ids, sample_coords))
                if len(results) >= n_pairs:
                    break

        return results if len(results) > 0 else [(list(query_cell_ids)[:20], np.zeros((min(20, len(query_cell_ids)), 3)))]

    def build_reference_manifold(self, n_pairs_per_embryo: int = 50):
        """
        Build reference manifold - store ALL cell embeddings, not just shared.

        CRITICAL FIX: Previous version only stored embeddings for cells that
        appeared in BOTH samples of a pair. This version stores embeddings
        for ALL cells in both samples, dramatically increasing manifold coverage.
        """
        print("\nBuilding reference manifold (storing ALL cells)...")

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

                # Sample cells from each timepoint (no overlap requirement)
                cell_ids1, coords1 = sampler.sample_cells(cells1, strategy='mixed')
                cell_ids2, coords2 = sampler.sample_cells(cells2, strategy='mixed')

                if len(cell_ids1) < 5 or len(cell_ids2) < 5:
                    continue

                # Normalize coordinates (per-dimension!)
                coords1_norm = normalize_coords(coords1)
                coords2_norm = normalize_coords(coords2)

                # Forward pass with masks
                emb1, emb2, _ = self.forward_with_masks(coords1_norm, coords2_norm)

                # CRITICAL FIX: Store ALL embeddings from BOTH samples
                # This ensures we capture embeddings for all cells, not just shared ones
                for i, cid in enumerate(cell_ids1):
                    cell_embeddings[cid].append(emb1[i])

                # Also store from pc2 (the reference side)
                # Note: emb2 may have one extra token (no-match) at the end if model uses it
                for i, cid in enumerate(cell_ids2):
                    if i < len(emb2):  # Safety check for no-match token
                        cell_embeddings[cid].append(emb2[i])

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
        # CRITICAL: Sample OVERLAPPING cells specifically to match training setup
        print("\n[1] Pairwise matching on training pairs...")

        correct = 0
        total = 0
        temperatures = []

        # Sample some training pairs
        embryo_ids = random.sample(list(self.train_data.keys()), min(20, len(self.train_data)))

        for embryo_id in embryo_ids:
            timepoints = list(self.train_data[embryo_id].items())
            if len(timepoints) < 2:
                continue

            for _ in range(10):  # More attempts per embryo
                (t1, cells1), (t2, cells2) = random.sample(timepoints, 2)

                # Find shared cells between timepoints
                shared = set(cells1.keys()) & set(cells2.keys())
                if len(shared) < 5:
                    continue

                # CRITICAL: Sample cells that INCLUDE the shared ones
                shared_list = list(shared)
                n_shared_to_sample = min(len(shared_list), 15)
                sampled_shared = random.sample(shared_list, n_shared_to_sample)

                # Build sample 1: shared cells + some unique to cells1
                ids1 = list(sampled_shared)
                unique1 = [c for c in cells1.keys() if c not in shared]
                n_extra1 = min(len(unique1), 20 - len(ids1))
                if n_extra1 > 0:
                    ids1.extend(random.sample(unique1, n_extra1))
                coords1 = np.array([cells1[cid] for cid in ids1])

                # Build sample 2: shared cells + some unique to cells2
                ids2 = list(sampled_shared)
                unique2 = [c for c in cells2.keys() if c not in shared]
                n_extra2 = min(len(unique2), 20 - len(ids2))
                if n_extra2 > 0:
                    ids2.extend(random.sample(unique2, n_extra2))
                coords2 = np.array([cells2[cid] for cid in ids2])

                if len(ids1) < 5 or len(ids2) < 5:
                    continue

                predictions, temp = self.predict_pairwise(coords1, ids1, coords2, ids2)
                temperatures.append(temp)

                # Evaluate on the shared cells (which are in both samples)
                for qid in sampled_shared:
                    if qid in predictions:
                        total += 1
                        if predictions[qid] == qid:
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
        embryo_id: str,
        n_refs: int = 5
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate embryo using pairwise matching with multi-reference voting.

        Uses multiple reference pairs and votes on predictions for stability.
        """
        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        correct = 0
        total = 0
        hierarchical = {'exact': 0, 'sublineage': 0, 'founder': 0, 'binary': 0}
        by_size = {'sparse': {'correct': 0, 'total': 0},  # 5-10 cells
                   'medium': {'correct': 0, 'total': 0},  # 11-15 cells
                   'dense': {'correct': 0, 'total': 0}}   # 16-20 cells

        for t, cells in embryo_data.items():
            if len(cells) < 5:
                continue

            cell_ids, coords = sampler.sample_cells(cells, strategy='mixed')
            n_cells = len(cell_ids)
            if n_cells < 5:
                continue

            # Determine size category
            if n_cells <= 10:
                size_cat = 'sparse'
            elif n_cells <= 15:
                size_cat = 'medium'
            else:
                size_cat = 'dense'

            embryo_stage = len(cells)

            # Find multiple overlapping references for voting
            pairs = self.find_overlapping_pairs(
                cell_ids, embryo_stage,
                min_overlap=1, n_pairs=n_refs
            )

            if len(pairs) == 0:
                continue

            # Collect votes from multiple reference pairings
            votes = defaultdict(list)  # query_id -> list of predicted ids

            for ref_ids, ref_coords in pairs:
                predictions, _ = self.predict_pairwise(coords, cell_ids, ref_coords, ref_ids)
                for query_id, pred_id in predictions.items():
                    if query_id in ref_ids:  # Only count when query exists in reference
                        votes[query_id].append(pred_id)

            # Use majority voting for final predictions
            for query_id, pred_list in votes.items():
                if len(pred_list) == 0:
                    continue

                # Majority vote
                vote_counts = Counter(pred_list)
                final_pred = vote_counts.most_common(1)[0][0]

                total += 1
                by_size[size_cat]['total'] += 1

                # Exact match
                is_correct = (query_id == final_pred)
                if is_correct:
                    correct += 1
                    by_size[size_cat]['correct'] += 1
                    hierarchical['exact'] += 1

                # Hierarchical accuracy (even if not exact match)
                if CelegansLineage.same_sublineage(query_id, final_pred):
                    hierarchical['sublineage'] += 1
                if CelegansLineage.same_founder(query_id, final_pred):
                    hierarchical['founder'] += 1
                # Binary: AB vs non-AB
                query_is_ab = CelegansLineage.get_founder(query_id) == 'AB'
                pred_is_ab = CelegansLineage.get_founder(final_pred) == 'AB'
                if query_is_ab == pred_is_ab:
                    hierarchical['binary'] += 1

        accuracy = correct / total if total > 0 else 0.0

        # Convert hierarchical to percentages
        if total > 0:
            for key in hierarchical:
                hierarchical[key] = hierarchical[key] / total

        # Convert by_size to accuracies
        for size_cat in by_size:
            if by_size[size_cat]['total'] > 0:
                by_size[size_cat]['accuracy'] = by_size[size_cat]['correct'] / by_size[size_cat]['total']
            else:
                by_size[size_cat]['accuracy'] = 0.0

        return accuracy, total, {'hierarchical': hierarchical, 'by_size': by_size}

    def evaluate_simulated(self) -> Dict:
        """
        Evaluate on simulated (held-out) embryos.

        Returns:
            Dictionary with knn_ci, pairwise_ci, hierarchical, by_size
        """
        print("\nEvaluating on simulated data...")

        knn_accuracies = []
        pairwise_accuracies = []

        # Aggregate hierarchical and by_size metrics
        hierarchical_totals = {'exact': 0, 'sublineage': 0, 'founder': 0, 'binary': 0}
        by_size_totals = {'sparse': {'correct': 0, 'total': 0},
                          'medium': {'correct': 0, 'total': 0},
                          'dense': {'correct': 0, 'total': 0}}
        total_cells = 0

        for embryo_id, timepoints in tqdm(self.eval_data.items(), desc="Embryos"):
            knn_acc, knn_n, _ = self.evaluate_embryo_knn(timepoints, embryo_id)
            pairwise_acc, pairwise_n, details = self.evaluate_embryo_pairwise(timepoints, embryo_id)

            if knn_n > 0:
                knn_accuracies.append(knn_acc)
            if pairwise_n > 0:
                pairwise_accuracies.append(pairwise_acc)

                # Aggregate hierarchical metrics (weighted by n cells)
                for key in hierarchical_totals:
                    hierarchical_totals[key] += details['hierarchical'][key] * pairwise_n
                total_cells += pairwise_n

                # Aggregate by_size
                for size_cat in by_size_totals:
                    by_size_totals[size_cat]['correct'] += details['by_size'][size_cat]['correct']
                    by_size_totals[size_cat]['total'] += details['by_size'][size_cat]['total']

        knn_ci = StatisticalAnalysis.bootstrap_ci(knn_accuracies)
        pairwise_ci = StatisticalAnalysis.bootstrap_ci(pairwise_accuracies)

        # Compute final hierarchical percentages
        hierarchical_pct = {}
        if total_cells > 0:
            for key in hierarchical_totals:
                hierarchical_pct[key] = 100 * hierarchical_totals[key] / total_cells
        else:
            hierarchical_pct = {k: 0.0 for k in hierarchical_totals}

        # Compute final by_size accuracies
        by_size_pct = {}
        for size_cat in by_size_totals:
            t = by_size_totals[size_cat]['total']
            c = by_size_totals[size_cat]['correct']
            by_size_pct[size_cat] = 100 * c / t if t > 0 else 0.0

        print(f"  k-NN accuracy: {knn_ci}")
        print(f"  Pairwise accuracy: {pairwise_ci}")
        print(f"\n  Hierarchical Accuracy:")
        print(f"    Exact: {hierarchical_pct['exact']:.1f}%")
        print(f"    Sublineage: {hierarchical_pct['sublineage']:.1f}%")
        print(f"    Founder: {hierarchical_pct['founder']:.1f}%")
        print(f"    Binary (AB vs non-AB): {hierarchical_pct['binary']:.1f}%")
        print(f"\n  Accuracy by Neighborhood Size:")
        print(f"    Sparse (5-10 cells): {by_size_pct['sparse']:.1f}%")
        print(f"    Medium (11-15 cells): {by_size_pct['medium']:.1f}%")
        print(f"    Dense (16-20 cells): {by_size_pct['dense']:.1f}%")

        return {
            'knn_ci': knn_ci,
            'pairwise_ci': pairwise_ci,
            'hierarchical': hierarchical_pct,
            'by_size': by_size_pct
        }

    def evaluate_real(self) -> Dict:
        """
        Evaluate on real embryo data.

        Returns:
            Dictionary with knn_ci, pairwise_ci, hierarchical, by_size
        """
        print("\nEvaluating on real data...")

        knn_accuracies = []
        pairwise_accuracies = []

        # Aggregate hierarchical and by_size metrics
        hierarchical_totals = {'exact': 0, 'sublineage': 0, 'founder': 0, 'binary': 0}
        by_size_totals = {'sparse': {'correct': 0, 'total': 0},
                          'medium': {'correct': 0, 'total': 0},
                          'dense': {'correct': 0, 'total': 0}}
        total_cells = 0

        for embryo_id, timepoints in tqdm(self.real_data.items(), desc="Embryos"):
            knn_acc, knn_n, _ = self.evaluate_embryo_knn(timepoints, embryo_id)
            pairwise_acc, pairwise_n, details = self.evaluate_embryo_pairwise(timepoints, embryo_id)

            if knn_n > 0:
                knn_accuracies.append(knn_acc)
            if pairwise_n > 0:
                pairwise_accuracies.append(pairwise_acc)

                # Aggregate hierarchical metrics (weighted by n cells)
                for key in hierarchical_totals:
                    hierarchical_totals[key] += details['hierarchical'][key] * pairwise_n
                total_cells += pairwise_n

                # Aggregate by_size
                for size_cat in by_size_totals:
                    by_size_totals[size_cat]['correct'] += details['by_size'][size_cat]['correct']
                    by_size_totals[size_cat]['total'] += details['by_size'][size_cat]['total']

        knn_ci = StatisticalAnalysis.bootstrap_ci(knn_accuracies)
        pairwise_ci = StatisticalAnalysis.bootstrap_ci(pairwise_accuracies)

        # Compute final hierarchical percentages
        hierarchical_pct = {}
        if total_cells > 0:
            for key in hierarchical_totals:
                hierarchical_pct[key] = 100 * hierarchical_totals[key] / total_cells
        else:
            hierarchical_pct = {k: 0.0 for k in hierarchical_totals}

        # Compute final by_size accuracies
        by_size_pct = {}
        for size_cat in by_size_totals:
            t = by_size_totals[size_cat]['total']
            c = by_size_totals[size_cat]['correct']
            by_size_pct[size_cat] = 100 * c / t if t > 0 else 0.0

        print(f"  k-NN accuracy: {knn_ci}")
        print(f"  Pairwise accuracy: {pairwise_ci}")
        print(f"\n  Hierarchical Accuracy:")
        print(f"    Exact: {hierarchical_pct['exact']:.1f}%")
        print(f"    Sublineage: {hierarchical_pct['sublineage']:.1f}%")
        print(f"    Founder: {hierarchical_pct['founder']:.1f}%")
        print(f"    Binary (AB vs non-AB): {hierarchical_pct['binary']:.1f}%")
        print(f"\n  Accuracy by Neighborhood Size:")
        print(f"    Sparse (5-10 cells): {by_size_pct['sparse']:.1f}%")
        print(f"    Medium (11-15 cells): {by_size_pct['medium']:.1f}%")
        print(f"    Dense (16-20 cells): {by_size_pct['dense']:.1f}%")

        return {
            'knn_ci': knn_ci,
            'pairwise_ci': pairwise_ci,
            'hierarchical': hierarchical_pct,
            'by_size': by_size_pct
        }

    def evaluate_baselines(self, cross_embryo: bool = True) -> Dict[str, ConfidenceInterval]:
        """
        Evaluate baseline methods (no learned components).

        CRITICAL FIX: Use the SAME task as the model for fair comparison.
        - cross_embryo=True: Match cells across different embryos (like model)
        - cross_embryo=False: Match distant timepoints (10+ apart) in same embryo

        Old approach matched consecutive timepoints (trivially easy, ~85%).
        Fixed approach should give baselines ~40-60%.

        Returns:
            Dictionary of method name -> accuracy CI
        """
        print("\nEvaluating baselines (FAIR comparison - same task as model)...")

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

            if cross_embryo:
                # FAIR EVALUATION: Cross-embryo matching (same as model)
                # Match eval embryo timepoints to TRAINING embryo timepoints
                for embryo_id, timepoints in tqdm(self.eval_data.items(), desc=name, leave=False):
                    embryo_correct = 0
                    embryo_total = 0

                    for t, cells in timepoints.items():
                        if len(cells) < 5:
                            continue

                        # Sample from eval embryo (query)
                        query_ids, query_coords = sampler.sample_cells(cells, strategy='mixed')
                        if len(query_ids) < 5:
                            continue

                        embryo_stage = len(cells)

                        # Find a DIFFERENT embryo from training data with overlapping cells
                        # This matches what the model is tested on
                        pairs = self.find_overlapping_pairs(
                            query_ids, embryo_stage,
                            min_overlap=1, n_pairs=1
                        )

                        if len(pairs) == 0:
                            continue

                        ref_ids, ref_coords = pairs[0]

                        # Run baseline
                        predictions = baseline.match(query_coords, query_ids, ref_coords, ref_ids)

                        # Evaluate only on overlapping cells (fair comparison)
                        for query_id, pred_id in predictions.items():
                            if query_id in ref_ids:  # Only count cells that exist in reference
                                embryo_total += 1
                                if query_id == pred_id:
                                    embryo_correct += 1

                    if embryo_total > 0:
                        accuracies.append(embryo_correct / embryo_total)
            else:
                # ALTERNATIVE: Distant timepoints in same embryo (10+ apart)
                for embryo_id, timepoints in tqdm(self.eval_data.items(), desc=name, leave=False):
                    embryo_correct = 0
                    embryo_total = 0

                    timepoint_list = list(timepoints.items())
                    timepoint_list.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)

                    if len(timepoint_list) < 15:  # Need enough timepoints for 10+ gap
                        continue

                    # Match distant timepoints (gap >= 10)
                    for i in range(len(timepoint_list)):
                        for j in range(i + 10, len(timepoint_list)):  # Gap of at least 10
                            t1, cells1 = timepoint_list[i]
                            t2, cells2 = timepoint_list[j]

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

    # =========================================================================
    # NO-CHEATING EVALUATION METHODS
    # These methods use ONLY stage information, not ground truth cell IDs
    # =========================================================================

    def find_stage_matched_references(
        self,
        embryo_stage: int,
        n_refs: int = 10,
        stage_tolerance: int = 20
    ) -> List[Tuple[List[str], np.ndarray]]:
        """
        Find training references by stage ONLY - no cell ID information used.

        This is the TRUE deployment scenario:
        - You image an embryo and count ~50 cells total
        - You observe 5-20 of them
        - You want to identify them without knowing any IDs

        Args:
            embryo_stage: Total cells in embryo (known from imaging)
            n_refs: Number of references to return
            stage_tolerance: How close stage must match (default ±20 cells)

        Returns:
            List of (cell_ids, coords) tuples from training data
        """
        sampler = ImprovedSampler(min_cells=5, max_cells=20)
        candidates = []

        for embryo_id, timepoints in self.train_data.items():
            for t, cells in timepoints.items():
                stage_diff = abs(len(cells) - embryo_stage)
                if stage_diff <= stage_tolerance:
                    candidates.append((stage_diff, embryo_id, t, cells))

        # Sort by closest stage match
        candidates.sort(key=lambda x: x[0])

        results = []
        used_embryos = set()  # Ensure diversity - don't use same embryo twice

        for stage_diff, embryo_id, t, cells in candidates:
            if embryo_id in used_embryos and len(results) < n_refs // 2:
                continue  # Skip if we have enough diversity

            sample_ids, sample_coords = sampler.sample_cells(cells, strategy='mixed')
            if len(sample_ids) >= 5:
                results.append((sample_ids, sample_coords))
                used_embryos.add(embryo_id)
                if len(results) >= n_refs:
                    break

        return results

    def identify_cells_no_cheating(
        self,
        query_coords: np.ndarray,
        embryo_stage: int,
        n_refs: int = 10
    ) -> Tuple[List[str], Dict]:
        """
        TRUE single-input cell identification - NO cheating.

        This is what we'd use in real deployment:
        1. Get stage-matched references (by cell count only)
        2. Pair query with each reference
        3. For each pairing, model outputs similarity matrix
        4. Argmax gives predicted ID from that reference
        5. Majority vote across all references

        Args:
            query_coords: (N, 3) coordinates of observed cells
            embryo_stage: Total cells in embryo (known from imaging)
            n_refs: Number of references to vote over

        Returns:
            Tuple of (predicted_ids list, debug_info dict)
        """
        n_query = len(query_coords)
        query_norm = normalize_coords(query_coords)

        # Step 1: Find references by STAGE ONLY (no ID information!)
        references = self.find_stage_matched_references(embryo_stage, n_refs=n_refs)

        if len(references) == 0:
            return ["UNKNOWN"] * n_query, {'error': 'no references found'}

        # Step 2: Collect votes from each reference pairing
        votes = defaultdict(list)  # query_idx -> list of (predicted_id, confidence)

        for ref_ids, ref_coords in references:
            ref_norm = normalize_coords(ref_coords)

            # Forward pass - joint encoding
            emb_query, emb_ref, temperature = self.forward_with_masks(query_norm, ref_norm)

            # Compute similarity (exclude no-match token if present)
            n_ref_cells = len(ref_ids)
            emb_ref_cells = emb_ref[:n_ref_cells]  # Exclude no-match token

            similarity = emb_query @ emb_ref_cells.T  # (N_query x N_ref)

            # For each query cell, find best match in this reference
            for q_idx in range(n_query):
                sim_row = similarity[q_idx]
                best_ref_idx = np.argmax(sim_row)
                confidence = sim_row[best_ref_idx]

                predicted_id = ref_ids[best_ref_idx]
                votes[q_idx].append((predicted_id, confidence))

        # Step 3: Majority vote (optionally weighted by confidence)
        predictions = []
        confidences = []

        for q_idx in range(n_query):
            if len(votes[q_idx]) == 0:
                predictions.append("UNKNOWN")
                confidences.append(0.0)
                continue

            # Simple majority vote
            vote_ids = [v[0] for v in votes[q_idx]]
            vote_counts = Counter(vote_ids)
            best_id, best_count = vote_counts.most_common(1)[0]

            # Confidence = fraction of votes for winner
            conf = best_count / len(vote_ids)

            predictions.append(best_id)
            confidences.append(conf)

        debug_info = {
            'n_references_used': len(references),
            'avg_confidence': np.mean(confidences),
            'votes_per_cell': {i: votes[i] for i in range(min(3, n_query))}  # Sample
        }

        return predictions, debug_info

    def evaluate_no_cheating(
        self,
        data: Dict,
        data_name: str = "Test"
    ) -> Dict:
        """
        Evaluate using TRUE no-cheating method.

        Returns detailed results including hierarchical accuracy and by-size breakdown.
        """
        print(f"\nEvaluating {data_name} data (NO CHEATING - stage-matched only)...")

        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        all_correct = []
        hierarchical_totals = {'exact': 0, 'sublineage': 0, 'founder': 0, 'binary': 0}
        by_size_totals = {
            'sparse': {'correct': 0, 'total': 0},
            'medium': {'correct': 0, 'total': 0},
            'dense': {'correct': 0, 'total': 0}
        }
        total_cells = 0

        for embryo_id, timepoints in tqdm(data.items(), desc=f"{data_name} embryos"):
            for t, cells in timepoints.items():
                if len(cells) < 5:
                    continue

                # Sample query cells (we know coords, we know ground truth IDs for eval only)
                query_ids_gt, query_coords = sampler.sample_cells(cells, strategy='mixed')
                n_cells = len(query_ids_gt)
                if n_cells < 5:
                    continue

                # Determine size category
                if n_cells <= 10:
                    size_cat = 'sparse'
                elif n_cells <= 15:
                    size_cat = 'medium'
                else:
                    size_cat = 'dense'

                embryo_stage = len(cells)

                # TRUE NO-CHEATING: Only pass coords and stage
                predictions, _ = self.identify_cells_no_cheating(
                    query_coords,
                    embryo_stage,
                    n_refs=10
                )

                # Evaluate predictions against ground truth
                for i, (gt_id, pred_id) in enumerate(zip(query_ids_gt, predictions)):
                    if pred_id == "UNKNOWN":
                        continue

                    total_cells += 1
                    by_size_totals[size_cat]['total'] += 1

                    # Exact match
                    is_correct = (gt_id == pred_id)
                    all_correct.append(1 if is_correct else 0)

                    if is_correct:
                        hierarchical_totals['exact'] += 1
                        by_size_totals[size_cat]['correct'] += 1

                    # Hierarchical (even if not exact)
                    if CelegansLineage.same_sublineage(gt_id, pred_id):
                        hierarchical_totals['sublineage'] += 1
                    if CelegansLineage.same_founder(gt_id, pred_id):
                        hierarchical_totals['founder'] += 1

                    gt_is_ab = CelegansLineage.get_founder(gt_id) == 'AB'
                    pred_is_ab = CelegansLineage.get_founder(pred_id) == 'AB'
                    if gt_is_ab == pred_is_ab:
                        hierarchical_totals['binary'] += 1

        # Compute final metrics
        accuracy_ci = StatisticalAnalysis.bootstrap_ci(all_correct)

        hierarchical_pct = {}
        for key in hierarchical_totals:
            hierarchical_pct[key] = 100 * hierarchical_totals[key] / max(1, total_cells)

        by_size_pct = {}
        for size_cat in by_size_totals:
            t = by_size_totals[size_cat]['total']
            c = by_size_totals[size_cat]['correct']
            by_size_pct[size_cat] = 100 * c / t if t > 0 else 0.0

        # Print results
        print(f"\n  {data_name} Results (NO CHEATING):")
        print(f"    Overall: {accuracy_ci}")
        print(f"\n    Hierarchical Accuracy:")
        print(f"      Exact: {hierarchical_pct['exact']:.1f}%")
        print(f"      Sublineage: {hierarchical_pct['sublineage']:.1f}%")
        print(f"      Founder: {hierarchical_pct['founder']:.1f}%")
        print(f"      Binary (AB vs non-AB): {hierarchical_pct['binary']:.1f}%")
        print(f"\n    By Neighborhood Size:")
        print(f"      Sparse (5-10): {by_size_pct['sparse']:.1f}%")
        print(f"      Medium (11-15): {by_size_pct['medium']:.1f}%")
        print(f"      Dense (16-20): {by_size_pct['dense']:.1f}%")

        return {
            'accuracy_ci': accuracy_ci,
            'hierarchical': hierarchical_pct,
            'by_size': by_size_pct,
            'total_cells': total_cells
        }

    # =========================================================================
    # STAGE-BINNED MANIFOLD METHODS
    # =========================================================================

    def build_stage_binned_manifold(self, n_pairs_per_embryo: int = 50, bin_size: int = 30):
        """
        Build manifold with stage information for stage-aware k-NN lookup.

        Instead of one big manifold, create bins by developmental stage.
        At inference, only search relevant stage bins.
        """
        print("\nBuilding stage-binned reference manifold...")

        # cell_id -> list of (embedding, stage)
        cell_stage_embeddings = defaultdict(list)
        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        # Build embryo timelines
        embryo_timelines = {}
        for embryo_id, timepoints in self.train_data.items():
            timeline = [(t, cells) for t, cells in timepoints.items()]
            timeline.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
            embryo_timelines[embryo_id] = timeline

        total_embeddings = 0

        for embryo_id in tqdm(embryo_timelines.keys(), desc="Building manifold"):
            timeline = embryo_timelines[embryo_id]
            if len(timeline) < 2:
                continue

            for _ in range(n_pairs_per_embryo):
                idx1, idx2 = random.sample(range(len(timeline)), 2)
                t1, cells1 = timeline[idx1]
                t2, cells2 = timeline[idx2]

                stage1 = len(cells1)
                stage2 = len(cells2)

                cell_ids1, coords1 = sampler.sample_cells(cells1, strategy='mixed')
                cell_ids2, coords2 = sampler.sample_cells(cells2, strategy='mixed')

                if len(cell_ids1) < 5 or len(cell_ids2) < 5:
                    continue

                coords1_norm = normalize_coords(coords1)
                coords2_norm = normalize_coords(coords2)

                emb1, emb2, _ = self.forward_with_masks(coords1_norm, coords2_norm)

                # Store with stage info
                for i, cid in enumerate(cell_ids1):
                    cell_stage_embeddings[cid].append((emb1[i], stage1))
                    total_embeddings += 1

                for i, cid in enumerate(cell_ids2):
                    if i < len(emb2) - 1:  # Exclude no-match token
                        cell_stage_embeddings[cid].append((emb2[i], stage2))
                        total_embeddings += 1

        print(f"  Total embeddings: {total_embeddings}")
        print(f"  Unique cells: {len(cell_stage_embeddings)}")

        # Build stage-binned indices
        self.stage_bins = defaultdict(lambda: {'embeddings': [], 'labels': [], 'stages': []})

        for cell_id, emb_stage_list in cell_stage_embeddings.items():
            for emb, stage in emb_stage_list:
                bin_idx = stage // bin_size
                self.stage_bins[bin_idx]['embeddings'].append(emb)
                self.stage_bins[bin_idx]['labels'].append(cell_id)
                self.stage_bins[bin_idx]['stages'].append(stage)

        # Build k-NN index per bin AND aggregate embeddings per cell within bin
        self.stage_knn = {}

        for bin_idx, data in self.stage_bins.items():
            if len(data['embeddings']) == 0:
                continue

            # Aggregate embeddings per cell within this bin
            cell_embs = defaultdict(list)
            for emb, label in zip(data['embeddings'], data['labels']):
                cell_embs[label].append(emb)

            # Average embeddings per cell
            agg_embeddings = []
            agg_labels = []
            for cell_id, embs in cell_embs.items():
                avg_emb = np.mean(embs, axis=0)
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
                agg_embeddings.append(avg_emb)
                agg_labels.append(cell_id)

            embeddings_array = np.array(agg_embeddings)

            knn = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, len(embeddings_array)),
                metric='cosine'
            )
            knn.fit(embeddings_array)

            self.stage_knn[bin_idx] = {
                'index': knn,
                'labels': agg_labels,
                'embeddings': embeddings_array,
                'n_cells': len(agg_labels)
            }

            print(f"    Bin {bin_idx} (stage {bin_idx*bin_size}-{(bin_idx+1)*bin_size}): {len(agg_labels)} cells")

        # Also keep flat manifold for comparison
        all_embeddings = []
        all_labels = []
        for cell_id, emb_stage_list in cell_stage_embeddings.items():
            avg_emb = np.mean([e[0] for e in emb_stage_list], axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            all_embeddings.append(avg_emb)
            all_labels.append(cell_id)

        self.reference_embeddings = np.array(all_embeddings)
        self.reference_labels = all_labels

        self.knn_index = NearestNeighbors(
            n_neighbors=min(self.k_neighbors, len(self.reference_embeddings)),
            metric='cosine'
        )
        self.knn_index.fit(self.reference_embeddings)

        print(f"\n  Final flat manifold: {len(self.reference_embeddings)} unique cells")

    def predict_knn_stage_aware(
        self,
        query_embeddings: np.ndarray,
        query_cell_ids: List[str],
        embryo_stage: int,
        bin_size: int = 30
    ) -> Dict[str, str]:
        """
        k-NN with stage-aware search - only look in relevant stage bins.

        This is more realistic and should improve accuracy because we're not
        searching cells from wildly different developmental stages.
        """
        bin_idx = embryo_stage // bin_size

        # Search in same bin and ±1 adjacent bins
        search_bins = [bin_idx]
        if bin_idx > 0:
            search_bins.append(bin_idx - 1)
        search_bins.append(bin_idx + 1)

        # Collect from relevant bins
        combined_embeddings = []
        combined_labels = []

        for b in search_bins:
            if b in self.stage_knn:
                combined_embeddings.append(self.stage_knn[b]['embeddings'])
                combined_labels.extend(self.stage_knn[b]['labels'])

        if len(combined_embeddings) == 0:
            # Fallback to full manifold
            return self.predict_knn(query_embeddings, query_cell_ids)

        combined_embeddings = np.vstack(combined_embeddings)

        # Build temp k-NN
        k = min(self.k_neighbors, len(combined_embeddings))
        temp_knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        temp_knn.fit(combined_embeddings)

        distances, indices = temp_knn.kneighbors(query_embeddings)

        predictions = {}
        for i, query_id in enumerate(query_cell_ids):
            neighbor_labels = [combined_labels[idx] for idx in indices[i]]
            label_counts = Counter(neighbor_labels)
            predictions[query_id] = label_counts.most_common(1)[0][0]

        return predictions

    def evaluate_knn_stage_aware(
        self,
        data: Dict,
        data_name: str = "Test"
    ) -> Dict:
        """
        Evaluate using stage-aware k-NN.
        """
        print(f"\nEvaluating {data_name} data (Stage-Aware k-NN)...")

        sampler = ImprovedSampler(min_cells=5, max_cells=20)

        all_correct = []
        total_cells = 0

        for embryo_id, timepoints in tqdm(data.items(), desc=f"{data_name} k-NN"):
            for t, cells in timepoints.items():
                if len(cells) < 5:
                    continue

                query_ids_gt, query_coords = sampler.sample_cells(cells, strategy='mixed')
                n_cells = len(query_ids_gt)
                if n_cells < 5:
                    continue

                embryo_stage = len(cells)

                # Get embeddings via multi-reference
                embeddings = self.embed_query_multi_reference(query_coords, query_ids_gt, embryo_stage)

                # Predict using stage-aware k-NN
                predictions = self.predict_knn_stage_aware(embeddings, query_ids_gt, embryo_stage)

                # Evaluate
                ref_set = set(self.reference_labels)
                for query_id, pred_id in predictions.items():
                    if query_id in ref_set:
                        total_cells += 1
                        is_correct = (query_id == pred_id)
                        all_correct.append(1 if is_correct else 0)

        accuracy_ci = StatisticalAnalysis.bootstrap_ci(all_correct)
        print(f"  {data_name} Stage-Aware k-NN: {accuracy_ci}")

        return {
            'accuracy_ci': accuracy_ci,
            'total_cells': total_cells
        }

    # =========================================================================
    # COMPREHENSIVE VISUALIZATION
    # =========================================================================

    def visualize_manifold_comprehensive(self, output_dir: str = "evaluation_results"):
        """Generate comprehensive t-SNE visualizations for paper."""

        if self.reference_embeddings is None or len(self.reference_embeddings) < 10:
            print("  Not enough embeddings for visualization")
            return {}

        print("\nGenerating comprehensive visualizations...")

        embeddings = self.reference_embeddings
        labels = self.reference_labels

        # Subsample if needed
        if len(embeddings) > 3000:
            indices = random.sample(range(len(embeddings)), 3000)
            embeddings = embeddings[indices]
            labels = [labels[i] for i in indices]

        # Run t-SNE
        print(f"  Running t-SNE on {len(embeddings)} points...")
        tsne = TSNE(n_components=2, perplexity=50, n_iter=2000, random_state=RANDOM_SEED)
        coords_2d = tsne.fit_transform(embeddings)

        # === PLOT 1: Color by Founder Lineage ===
        fig, ax = plt.subplots(figsize=(12, 10))

        founders = [CelegansLineage.get_founder(label) for label in labels]
        unique_founders = sorted(set(founders))

        # Use distinct colors
        colors_map = {
            'AB': '#e41a1c',   # Red
            'MS': '#377eb8',   # Blue
            'E': '#4daf4a',    # Green
            'C': '#984ea3',    # Purple
            'D': '#ff7f00',    # Orange
            'P4': '#a65628',   # Brown
        }

        for founder in unique_founders:
            mask = [f == founder for f in founders]
            mask_coords = coords_2d[np.array(mask)]
            color = colors_map.get(founder, '#999999')
            ax.scatter(mask_coords[:, 0], mask_coords[:, 1],
                      c=color, label=founder, alpha=0.6, s=15)

        ax.legend(title="Founder Lineage", loc='upper right', fontsize=10)
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("Cell Embedding Manifold by Founder Lineage", fontsize=14)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tsne_by_founder.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: tsne_by_founder.png")

        # === PLOT 2: Color by Sublineage (depth 2) ===
        fig, ax = plt.subplots(figsize=(14, 10))

        sublineages = [CelegansLineage.get_sublineage(label, depth=2) for label in labels]
        unique_sublineages = sorted(set(sublineages))

        # Use colormap for many categories
        cmap = plt.cm.get_cmap('tab20')

        for i, sublin in enumerate(unique_sublineages[:20]):  # Top 20 most common
            mask = [s == sublin for s in sublineages]
            if sum(mask) < 5:
                continue
            mask_coords = coords_2d[np.array(mask)]
            ax.scatter(mask_coords[:, 0], mask_coords[:, 1],
                      c=[cmap(i % 20)], label=sublin, alpha=0.6, s=15)

        ax.legend(title="Sublineage", loc='upper right', fontsize=8, ncol=2)
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("Cell Embedding Manifold by Sublineage", fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tsne_by_sublineage.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: tsne_by_sublineage.png")

        # === PLOT 3: Highlight specific well-known cells ===
        fig, ax = plt.subplots(figsize=(12, 10))

        # Background: all points in gray
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c='lightgray', alpha=0.3, s=10)

        # Highlight specific cells
        highlight_cells = ['ABa', 'ABp', 'ABal', 'ABar', 'ABpl', 'ABpr',
                           'MS', 'E', 'C', 'D', 'P4', 'EMS', 'P2', 'P3']

        highlight_colors = plt.cm.get_cmap('tab10')

        for i, cell in enumerate(highlight_cells):
            mask = [label == cell for label in labels]
            if sum(mask) == 0:
                continue
            mask_coords = coords_2d[np.array(mask)]
            ax.scatter(mask_coords[:, 0], mask_coords[:, 1],
                      c=[highlight_colors(i % 10)], label=cell, alpha=0.9, s=50, edgecolors='black')

        ax.legend(title="Cell Identity", loc='upper right', fontsize=9)
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("Cell Embedding Manifold - Key Cells Highlighted", fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tsne_key_cells.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: tsne_key_cells.png")

        # === Cluster quality metrics ===
        print("\n  Computing cluster quality metrics...")

        cell_counts = Counter(labels)
        common_cells = [c for c, count in cell_counts.most_common(50) if count >= 3]

        intra_distances = []
        inter_distances = []

        from scipy.spatial.distance import pdist, cdist

        for cell in common_cells:
            cell_mask = np.array([label == cell for label in labels])
            other_mask = ~cell_mask

            if cell_mask.sum() < 2 or other_mask.sum() < 2:
                continue

            cell_coords = coords_2d[cell_mask]
            other_coords = coords_2d[other_mask]

            # Intra-class: avg distance within cluster
            if len(cell_coords) > 1:
                intra = np.mean(pdist(cell_coords))
                intra_distances.append(intra)

            # Inter-class: avg distance to nearest other cluster
            inter = np.min(cdist(cell_coords, other_coords).mean(axis=1))
            inter_distances.append(inter)

        avg_intra = np.mean(intra_distances) if intra_distances else 0
        avg_inter = np.mean(inter_distances) if inter_distances else 0
        silhouette_approx = (avg_inter - avg_intra) / max(avg_inter, avg_intra) if max(avg_inter, avg_intra) > 0 else 0

        print(f"    Avg intra-cluster distance: {avg_intra:.2f}")
        print(f"    Avg inter-cluster distance: {avg_inter:.2f}")
        print(f"    Silhouette-like score: {silhouette_approx:.3f}")

        return {
            'intra_distance': avg_intra,
            'inter_distance': avg_inter,
            'silhouette': silhouette_approx
        }

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
        Run complete evaluation pipeline with no-cheating methods.

        Pipeline:
        1. Load data and model
        2. Run diagnostics (data coverage, naming)
        3. Sanity checks (pairwise with known overlap - verify model works)
        4. Build stage-binned manifold
        5. Check manifold coverage
        6. Generate comprehensive visualizations
        7. MAIN RESULTS: No-cheating evaluation (stage-matched voting)
        8. Comparison: Stage-aware k-NN
        9. Fair baseline comparison
        10. Save results
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION SUITE")
        print("="*60)

        # Load everything
        self.load_data()
        self.load_model()

        # Diagnostics
        self.diagnose_real_data()

        # Sanity checks (pairwise with known overlap - just to verify model works)
        self.run_sanity_checks()

        # Build stage-binned manifold
        self.build_stage_binned_manifold()

        # Check coverage
        self.check_manifold_coverage()

        # Visualizations (BEFORE evaluation for debugging)
        cluster_metrics = self.visualize_manifold_comprehensive(output_dir)

        # ===== MAIN RESULTS: NO-CHEATING EVALUATION =====
        print("\n" + "="*60)
        print("MAIN RESULTS: TRUE SINGLE-INPUT IDENTIFICATION (NO CHEATING)")
        print("="*60)

        sim_no_cheat = self.evaluate_no_cheating(self.eval_data, "Simulated")
        real_no_cheat = self.evaluate_no_cheating(self.real_data, "Real")

        # Also run k-NN (stage-aware) for comparison
        print("\n" + "="*60)
        print("COMPARISON: Stage-Aware k-NN")
        print("="*60)

        sim_knn = self.evaluate_knn_stage_aware(self.eval_data, "Simulated")

        # Fair baseline comparison
        baseline_results = self.evaluate_baselines(cross_embryo=True)

        # ===== SUMMARY =====
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)

        print("\n  MAIN RESULT (No Cheating - Stage-Matched Voting):")
        print(f"   Simulated: {sim_no_cheat['accuracy_ci']}")
        print(f"   Real: {real_no_cheat['accuracy_ci']}")

        print("\n  k-NN (Stage-Aware Manifold Lookup):")
        print(f"   Simulated: {sim_knn['accuracy_ci']}")

        print("\n  Baselines (Same Task - Fair Comparison):")
        for name, ci in baseline_results.items():
            print(f"   {name}: {ci}")

        # Compute improvement over best baseline
        best_baseline = max(ci.mean for ci in baseline_results.values())
        improvement = sim_no_cheat['accuracy_ci'].mean - best_baseline

        print(f"\n  Improvement over best baseline: +{improvement:.1f}%")

        # Save results
        results = {
            'main_no_cheating': {
                'simulated': {
                    'accuracy': sim_no_cheat['accuracy_ci'].mean,
                    'ci_lower': sim_no_cheat['accuracy_ci'].lower,
                    'ci_upper': sim_no_cheat['accuracy_ci'].upper,
                    'hierarchical': sim_no_cheat['hierarchical'],
                    'by_size': sim_no_cheat['by_size']
                },
                'real': {
                    'accuracy': real_no_cheat['accuracy_ci'].mean,
                    'ci_lower': real_no_cheat['accuracy_ci'].lower,
                    'ci_upper': real_no_cheat['accuracy_ci'].upper,
                    'hierarchical': real_no_cheat['hierarchical'],
                    'by_size': real_no_cheat['by_size']
                }
            },
            'knn_stage_aware': {
                'simulated': {
                    'accuracy': sim_knn['accuracy_ci'].mean,
                    'ci_lower': sim_knn['accuracy_ci'].lower,
                    'ci_upper': sim_knn['accuracy_ci'].upper
                }
            },
            'baselines': {
                name: {'mean': ci.mean, 'lower': ci.lower, 'upper': ci.upper}
                for name, ci in baseline_results.items()
            },
            'cluster_quality': cluster_metrics if cluster_metrics else {}
        }

        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n  Results saved to: {output_dir}/")

        return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Run the complete evaluation suite with no-cheating methods."""

    print("="*60)
    print("C. ELEGANS CELL IDENTIFICATION - EVALUATION SUITE")
    print("="*60)
    print("\n  TRUE NO-CHEATING EVALUATION")
    print("  ----------------------------")
    print("  Input: 3D coordinates + embryo stage (cell count)")
    print("  Output: Predicted cell identities")
    print("  NO ACCESS TO ground truth cell IDs during inference")
    print()
    print("  Key Methods:")
    print("  1. Stage-matched voting (no cell ID information)")
    print("  2. Stage-binned k-NN (efficient manifold lookup)")
    print("  3. Fair baselines (same cross-embryo task)")
    print()
    print("  Outputs:")
    print("  - tsne_by_founder.png (paper figure)")
    print("  - tsne_by_sublineage.png (paper figure)")
    print("  - tsne_key_cells.png (paper figure)")
    print("  - results.json (all metrics)")
    print()

    engine = FixedEvaluationEngine()
    results = engine.run_full_evaluation()

    return results


if __name__ == "__main__":
    main()
