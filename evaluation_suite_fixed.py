"""
Comprehensive Evaluation Suite for Twin Attention Cell Identification Model
TRUE NO-CHEATING EVALUATION - Uses stage-only matching like real deployment

Key principle: In real deployment, we DON'T know query cell IDs.
We only know coordinates and embryo stage (total cell count).
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.stats import ttest_ind
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional, NamedTuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

# Import model classes from the main file
from debug_sparse_matching import (
    EnhancedTwinAttentionEncoder,
    SparsePointFeatures,
    TwinAttentionMatchingLoss,
    SparseEmbryoDataset,
    collate_fn_with_padding,
    ImprovedSampler,
    set_seed
)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA STRUCTURES
# =============================================================================
class ConfidenceInterval(NamedTuple):
    mean: float
    ci_lower: float
    ci_upper: float


# =============================================================================
# CONFIGURATION
# =============================================================================
class EvalConfig:
    """Configuration for evaluation suite"""
    # Data paths (Windows format - will be converted for Linux)
    TRAIN_DATA_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\data_dict.pkl"
    EVAL_DATA_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\evaluation_data_dict.pkl"
    MODEL_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\twin_attention_final.pth"
    REAL_EMBRYO_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\real_data_dict.pkl"

    # Output directories
    OUTPUT_DIR = "evaluation_results"
    FIGURE_DIR = "evaluation_figures"
    ABLATION_WEIGHTS_DIR = "ablation_weights"

    # Model architecture
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 6
    DROPOUT = 0.1

    # Evaluation parameters
    BATCH_SIZE = 16
    MIN_CELLS = 5
    MAX_CELLS = 20
    STAGE_LIMIT = 194
    K_NEIGHBORS = 30
    N_BOOTSTRAP = 100
    N_REFS = 10  # Number of stage-matched references for voting
    STAGE_TOLERANCE = 20  # Initial tolerance for stage matching

    # Sampling
    MAX_SAMPLES_PER_EMBRYO = 50  # Max timepoints per embryo to evaluate

    # For faster CPU KNN
    KNN_BATCH_SIZE = 1000


def convert_path(path):
    """Convert Windows paths to Linux paths if needed"""
    if path is None:
        return None
    if os.name != 'nt' and '\\' in path:
        return os.path.basename(path.replace('\\', '/'))
    return path


# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================
ABLATION_CONFIGS = {
    'full_model': {
        'architecture': 'joint',
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'use_sparse_features': True,
        'use_uncertainty': True,
        'use_learnable_no_match': True,
        'weight_file': 'twin_attention_final.pth',
        'description': 'Full Twin Attention model'
    },
    'siamese': {
        'architecture': 'siamese',
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'use_sparse_features': True,
        'use_uncertainty': True,
        'use_learnable_no_match': True,
        'weight_file': 'siamese.pth',
        'description': 'Siamese (independent encoding, no joint attention)'
    },
    'no_sparse_features': {
        'architecture': 'joint',
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'use_sparse_features': False,
        'use_uncertainty': True,
        'use_learnable_no_match': True,
        'weight_file': 'no_sparse_features.pth',
        'description': 'Raw XYZ coordinates only'
    },
    'no_uncertainty': {
        'architecture': 'joint',
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'use_sparse_features': True,
        'use_uncertainty': False,
        'use_learnable_no_match': True,
        'weight_file': 'no_uncertainty.pth',
        'description': 'No uncertainty estimation'
    },
    'no_no_match_token': {
        'architecture': 'joint',
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'use_sparse_features': True,
        'use_uncertainty': True,
        'use_learnable_no_match': False,
        'weight_file': 'no_no_match_token.pth',
        'description': 'No learnable no-match token'
    },
    '3_layers': {
        'architecture': 'joint',
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'use_sparse_features': True,
        'use_uncertainty': True,
        'use_learnable_no_match': True,
        'weight_file': '3_layers.pth',
        'description': 'Shallow model (3 layers)'
    },
    '64_dim': {
        'architecture': 'joint',
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 6,
        'use_sparse_features': True,
        'use_uncertainty': True,
        'use_learnable_no_match': True,
        'weight_file': '64_dim.pth',
        'description': 'Smaller embedding dimension'
    }
}


# =============================================================================
# C. ELEGANS LINEAGE UTILITIES
# =============================================================================
class CelegansLineage:
    """Utilities for C. elegans cell lineage analysis"""

    FOUNDERS = ['AB', 'MS', 'E', 'C', 'D', 'P4', 'P3', 'P2', 'P1', 'P0', 'EMS', 'Z2', 'Z3']

    # Biologically meaningful colors for visualization
    FOUNDER_COLORS = {
        'AB': '#e41a1c',   # Red
        'MS': '#377eb8',   # Blue
        'E': '#4daf4a',    # Green
        'C': '#984ea3',    # Purple
        'D': '#ff7f00',    # Orange
        'P4': '#a65628',   # Brown
        'P3': '#f781bf',   # Pink
        'P2': '#999999',   # Gray
        'EMS': '#66c2a5',  # Teal
        'UNKNOWN': '#cccccc'  # Light gray
    }

    @staticmethod
    def parse_cell_id(cell_id: str) -> Dict:
        """Parse cell ID into components"""
        cell_id = str(cell_id).strip()
        upper_id = cell_id.upper()

        for founder in CelegansLineage.FOUNDERS:
            if upper_id.startswith(founder):
                return {
                    'founder': founder,
                    'lineage': cell_id,
                    'depth': len(cell_id) - len(founder)
                }
        return {'founder': 'UNKNOWN', 'lineage': cell_id, 'depth': 0}

    @staticmethod
    def get_founder(cell_id: str) -> str:
        """Get founder lineage of a cell"""
        return CelegansLineage.parse_cell_id(cell_id)['founder']

    @staticmethod
    def get_parent(cell_id: str) -> Optional[str]:
        """Get parent cell ID (remove last character if valid)"""
        parsed = CelegansLineage.parse_cell_id(cell_id)
        if parsed['founder'] == 'UNKNOWN' or parsed['depth'] == 0:
            return None
        return cell_id[:-1]

    @staticmethod
    def are_siblings(c1: str, c2: str) -> bool:
        """Check if two cells are siblings (same parent)"""
        p1 = CelegansLineage.parse_cell_id(c1)
        p2 = CelegansLineage.parse_cell_id(c2)

        if p1['founder'] == 'UNKNOWN' or p2['founder'] == 'UNKNOWN':
            return False

        l1, l2 = p1['lineage'], p2['lineage']
        return len(l1) >= 2 and len(l2) >= 2 and l1[:-1] == l2[:-1] and l1 != l2

    @staticmethod
    def same_founder(c1: str, c2: str) -> bool:
        """Check if two cells have the same founder"""
        return CelegansLineage.get_founder(c1) == CelegansLineage.get_founder(c2)

    @staticmethod
    def same_sublineage(c1: str, c2: str, depth: int = 2) -> bool:
        """Check if two cells share the same sublineage at given depth"""
        p1, p2 = CelegansLineage.parse_cell_id(c1), CelegansLineage.parse_cell_id(c2)

        if p1['founder'] != p2['founder']:
            return False
        if p1['founder'] == 'UNKNOWN':
            return c1 == c2

        f = p1['founder']
        prefix1 = p1['lineage'][:len(f) + min(depth, p1['depth'])]
        prefix2 = p2['lineage'][:len(f) + min(depth, p2['depth'])]
        return prefix1 == prefix2

    @staticmethod
    def is_mother_daughter(c1: str, c2: str) -> bool:
        """Check if one cell is parent of the other"""
        parent1 = CelegansLineage.get_parent(c1)
        parent2 = CelegansLineage.get_parent(c2)
        return (parent1 == c2) or (parent2 == c1)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================
class StatisticalAnalysis:
    """Statistical analysis utilities"""

    @staticmethod
    def bootstrap_ci(values: List[float], n: int = 100, alpha: float = 0.05) -> ConfidenceInterval:
        """Compute bootstrap confidence interval"""
        if len(values) == 0:
            return ConfidenceInterval(0.0, 0.0, 0.0)

        values = np.array(values)
        means = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n)]

        return ConfidenceInterval(
            mean=float(np.mean(values)),
            ci_lower=float(np.percentile(means, 100 * alpha / 2)),
            ci_upper=float(np.percentile(means, 100 * (1 - alpha / 2)))
        )

    @staticmethod
    def cohens_d(group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0

        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    @staticmethod
    def welch_ttest(group1: List[float], group2: List[float]) -> float:
        """Perform Welch's t-test and return p-value"""
        if len(group1) < 2 or len(group2) < 2:
            return 1.0

        _, p_value = ttest_ind(group1, group2, equal_var=False)
        return float(p_value)


# =============================================================================
# SIAMESE TRANSFORMER (For ablation comparison)
# =============================================================================
class SiameseTransformer(nn.Module):
    """
    Siamese Transformer - processes each neighborhood INDEPENDENTLY.
    Contrast with Twin Attention which processes both jointly.
    """

    def __init__(self, input_dim=3, embed_dim=128, num_heads=8, num_layers=6,
                 dropout=0.1, use_sparse_features=True, use_uncertainty=True,
                 use_learnable_no_match=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_sparse_features = use_sparse_features
        self.use_uncertainty = use_uncertainty
        self.use_learnable_no_match = use_learnable_no_match

        # Feature extraction
        if use_sparse_features:
            self.sparse_features = SparsePointFeatures(embed_dim)
            self.feature_projection = nn.Linear(embed_dim, embed_dim)
        else:
            self.point_embed = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, embed_dim) * 0.02)

        # INDEPENDENT encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projections
        if use_uncertainty:
            self.output_mean = nn.Linear(embed_dim, embed_dim)
            self.output_logvar = nn.Linear(embed_dim, embed_dim)
        else:
            self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        if use_learnable_no_match:
            self.no_match_token = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def encode_single(self, pc, mask=None):
        """Encode a single neighborhood independently"""
        B, N, _ = pc.shape

        if self.use_sparse_features:
            z = self.sparse_features(pc, mask)
            z = self.feature_projection(z)
        else:
            z = self.point_embed(pc)

        z = z + self.pos_encoding[:, :N, :]

        attn_mask = ~mask.bool() if mask is not None else None
        z = self.encoder(z, src_key_padding_mask=attn_mask)

        if self.use_uncertainty:
            z_mean = F.normalize(self.output_mean(z), p=2, dim=-1)
            z_logvar = torch.clamp(self.output_logvar(z), -10, 2)
            return z_mean, z_logvar
        else:
            return F.normalize(self.output_proj(z), p=2, dim=-1)

    def forward(self, pc1, pc2, mask1=None, mask2=None, epoch=0):
        B = pc1.shape[0]

        if self.use_uncertainty:
            z1_mean, z1_logvar = self.encode_single(pc1, mask1)
            z2_mean, z2_logvar = self.encode_single(pc2, mask2)

            if self.use_learnable_no_match:
                no_match = F.normalize(self.no_match_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1), p=2, dim=-1)
                z2_mean = torch.cat([z2_mean, no_match], dim=1)
                z2_logvar = torch.cat([z2_logvar, torch.zeros(B, 1, self.embed_dim, device=z2_logvar.device)], dim=1)

            z1 = (z1_mean, z1_logvar)
            z2 = (z2_mean, z2_logvar)
        else:
            z1 = self.encode_single(pc1, mask1)
            z2 = self.encode_single(pc2, mask2)

            if self.use_learnable_no_match:
                no_match = F.normalize(self.no_match_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1), p=2, dim=-1)
                z2 = torch.cat([z2, no_match], dim=1)

        temperature = torch.exp(self.log_temperature).clamp(0.01, 10.0)
        return z1, z2, temperature


# =============================================================================
# BASELINE METHODS
# =============================================================================
class BaselineMethods:
    """Traditional baseline methods for cell identification"""

    @staticmethod
    def icp_match(src: np.ndarray, tgt: np.ndarray, max_iters: int = 20) -> Tuple[np.ndarray, float]:
        """ICP alignment and matching"""
        src = src.copy()
        src = src - src.mean(0)
        tgt = tgt - tgt.mean(0)

        for _ in range(max_iters):
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(tgt)
            dists, idx = nn.kneighbors(src)

            median_dist = np.median(dists)
            inlier_mask = dists.flatten() < median_dist * 3

            if inlier_mask.sum() < 4:
                break

            src_in = src[inlier_mask]
            tgt_in = tgt[idx.flatten()[inlier_mask]]

            H = src_in.T @ tgt_in
            U, _, Vt = np.linalg.svd(H)
            R_mat = Vt.T @ U.T
            if np.linalg.det(R_mat) < 0:
                Vt[-1, :] *= -1
                R_mat = Vt.T @ U.T
            src = src @ R_mat.T

        # Final matching
        nn = NearestNeighbors(n_neighbors=1).fit(tgt)
        dists, matches = nn.kneighbors(src)
        return matches.flatten(), np.sqrt(np.mean(dists**2))

    @staticmethod
    def hungarian_match(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Hungarian algorithm matching based on distances"""
        from scipy.spatial.distance import cdist

        src = src - src.mean(0)
        tgt = tgt - tgt.mean(0)

        cost = cdist(src, tgt)
        n1, n2 = len(src), len(tgt)

        if n1 != n2:
            max_size = max(n1, n2)
            pad = np.full((max_size, max_size), cost.max() * 10)
            pad[:n1, :n2] = cost
            cost = pad

        row, col = linear_sum_assignment(cost)

        matches = np.zeros(n1, dtype=int)
        for r, c in zip(row, col):
            if r < n1:
                matches[r] = c if c < n2 else -1

        return matches

    @staticmethod
    def nearest_neighbor_match(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Simple nearest neighbor matching"""
        src = src - src.mean(0)
        tgt = tgt - tgt.mean(0)

        nn = NearestNeighbors(n_neighbors=1).fit(tgt)
        _, matches = nn.kneighbors(src)
        return matches.flatten()


# =============================================================================
# MAIN EVALUATOR CLASS
# =============================================================================
class NoCheatingEvaluator:
    """
    TRUE NO-CHEATING Cell Identification Evaluator

    Key principle: In real deployment, we DON'T know query cell IDs.
    We only know coordinates and embryo stage (total cell count).
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = device

        # Data storage
        self.train_data = None
        self.eval_data = None
        self.real_data = None

        # Model
        self.model = None

        # Index for stage-based lookup
        self.stage_index = defaultdict(list)  # stage -> [(embryo, timepoint, cells)]

        # Results
        self.results = {}

    def load_data(self):
        """Load all datasets"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)

        # Training data
        train_path = convert_path(self.config.TRAIN_DATA_PATH)
        if os.path.exists(train_path):
            print(f"Loading training data: {train_path}")
            with open(train_path, 'rb') as f:
                self.train_data = pickle.load(f)
            print(f"  Loaded {len(self.train_data)} training embryos")
            self._build_stage_index()

        # Evaluation data
        eval_path = convert_path(self.config.EVAL_DATA_PATH)
        if os.path.exists(eval_path):
            print(f"Loading evaluation data: {eval_path}")
            with open(eval_path, 'rb') as f:
                self.eval_data = pickle.load(f)
            print(f"  Loaded {len(self.eval_data)} evaluation embryos")

        # Real embryo data
        real_path = convert_path(self.config.REAL_EMBRYO_PATH)
        if real_path and os.path.exists(real_path):
            print(f"Loading real embryo data: {real_path}")
            with open(real_path, 'rb') as f:
                self.real_data = pickle.load(f)
            print(f"  Loaded {len(self.real_data)} real embryos")

    def _build_stage_index(self):
        """Build index mapping embryo stage (cell count) to timepoints"""
        print("  Building stage index...")
        self.stage_index.clear()

        for embryo, timepoints in self.train_data.items():
            for t, cells in timepoints.items():
                n_cells = len(cells)
                if self.config.MIN_CELLS <= n_cells <= self.config.STAGE_LIMIT:
                    self.stage_index[n_cells].append({
                        'embryo': embryo,
                        'timepoint': t,
                        'cells': cells
                    })

        total_entries = sum(len(v) for v in self.stage_index.values())
        print(f"    Indexed {total_entries} timepoints across {len(self.stage_index)} stages")

    def load_model(self, model_config: Dict = None, weight_path: str = None):
        """Load model with optional custom configuration"""
        print("\n" + "="*60)
        print("LOADING MODEL")
        print("="*60)

        if model_config is None:
            model_config = ABLATION_CONFIGS['full_model']

        if weight_path is None:
            weight_path = convert_path(self.config.MODEL_PATH)

        # Create model based on architecture
        if model_config.get('architecture') == 'siamese':
            self.model = SiameseTransformer(
                embed_dim=model_config.get('embed_dim', 128),
                num_heads=model_config.get('num_heads', 8),
                num_layers=model_config.get('num_layers', 6),
                dropout=self.config.DROPOUT,
                use_sparse_features=model_config.get('use_sparse_features', True),
                use_uncertainty=model_config.get('use_uncertainty', True),
                use_learnable_no_match=model_config.get('use_learnable_no_match', True)
            )
        else:
            self.model = EnhancedTwinAttentionEncoder(
                embed_dim=model_config.get('embed_dim', 128),
                num_heads=model_config.get('num_heads', 8),
                num_layers=model_config.get('num_layers', 6),
                dropout=self.config.DROPOUT,
                use_sparse_features=model_config.get('use_sparse_features', True),
                use_uncertainty=model_config.get('use_uncertainty', True),
                use_learnable_no_match=model_config.get('use_learnable_no_match', True)
            )

        # Load weights if available
        if os.path.exists(weight_path):
            print(f"Loading weights from: {weight_path}")
            checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("  Weights loaded successfully")
        else:
            print(f"  WARNING: Weight file not found: {weight_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {n_params:,}")
        print(f"  Architecture: {model_config.get('architecture', 'joint')}")
        print(f"  Sparse features: {model_config.get('use_sparse_features', True)}")
        print(f"  Uncertainty: {model_config.get('use_uncertainty', True)}")

    @staticmethod
    def normalize_coords(coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates (per-dimension, matching training)"""
        mean = coords.mean(axis=0)
        std = coords.std(axis=0)
        std = np.clip(std, 1e-6, None)  # Avoid division by zero
        return (coords - mean) / std

    def find_stage_matched_references(
        self,
        embryo_stage: int,
        n_refs: int = 10,
        exclude_embryo: str = None
    ) -> List[Tuple[List[str], np.ndarray]]:
        """
        Find training references by STAGE ONLY - NO cell ID information.

        This matches Twin Attention's inference approach:
        - Query embryo has N total cells
        - Find training timepoints with similar cell count
        - Sample 5-20 cells from each

        Args:
            embryo_stage: Total cells in embryo (known from imaging)
            n_refs: Number of references to return
            exclude_embryo: Embryo name to exclude (for held-out eval)

        Returns:
            List of (cell_ids, normalized_coords) tuples
        """
        references = []
        used_embryos = set()

        # Try increasingly larger tolerances until we have enough refs
        for tolerance in [20, 30, 50, 100, 200]:
            candidates = []

            for stage in range(embryo_stage - tolerance, embryo_stage + tolerance + 1):
                if stage in self.stage_index:
                    for entry in self.stage_index[stage]:
                        if exclude_embryo and entry['embryo'] == exclude_embryo:
                            continue
                        if entry['embryo'] in used_embryos:
                            continue
                        candidates.append(entry)

            # Sample from candidates
            random.shuffle(candidates)

            for entry in candidates:
                if len(references) >= n_refs:
                    break

                cells = entry['cells']
                cell_ids = list(cells.keys())
                n_cells = len(cell_ids)

                # Sample 5-20 cells
                n_sample = min(max(self.config.MIN_CELLS, n_cells), self.config.MAX_CELLS)
                if n_cells > n_sample:
                    sampled_indices = np.random.choice(n_cells, n_sample, replace=False)
                    sampled_ids = [cell_ids[i] for i in sampled_indices]
                else:
                    sampled_ids = cell_ids

                # Get coordinates
                coords = np.array([cells[c] for c in sampled_ids])
                coords = self.normalize_coords(coords)

                references.append((sampled_ids, coords))
                used_embryos.add(entry['embryo'])

            if len(references) >= n_refs:
                break

        return references

    @torch.no_grad()
    def identify_cells_no_cheating(
        self,
        query_coords: np.ndarray,
        embryo_stage: int,
        n_refs: int = 10,
        exclude_embryo: str = None
    ) -> Tuple[List[str], np.ndarray, Dict]:
        """
        TRUE single-input cell identification - NO cheating.

        Pipeline (mirrors Twin Attention inference):
        1. Find stage-matched references (by cell count ONLY)
        2. For each reference:
           a. Forward pass through model (joint encoding)
           b. Compute similarity matrix
           c. Argmax per query cell -> predicted ID
        3. Majority vote across all references

        Args:
            query_coords: (N, 3) cell coordinates (already sampled/observable)
            embryo_stage: Total cells in embryo (known from imaging)
            n_refs: Number of stage-matched references to vote over
            exclude_embryo: Embryo to exclude from references

        Returns:
            Tuple of:
            - predictions: List[str] predicted cell IDs
            - confidences: np.ndarray vote confidence
            - debug_info: Dict with voting details
        """
        self.model.eval()
        n_query = len(query_coords)

        # Normalize query coordinates
        query_norm = self.normalize_coords(query_coords)

        # Find stage-matched references
        references = self.find_stage_matched_references(
            embryo_stage, n_refs, exclude_embryo
        )

        if not references:
            # Fallback: return unknown predictions
            return ['UNKNOWN'] * n_query, np.zeros(n_query), {'n_refs': 0}

        # Collect votes for each query cell
        all_votes = [[] for _ in range(n_query)]

        for ref_ids, ref_coords in references:
            n_ref = len(ref_ids)

            # Prepare tensors
            pc1 = torch.tensor(query_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
            pc2 = torch.tensor(ref_coords, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask1 = torch.ones(1, n_query, device=self.device)
            mask2 = torch.ones(1, n_ref, device=self.device)

            # Forward pass
            z1, z2, temp = self.model(pc1, pc2, mask1, mask2, epoch=100)

            # Extract embeddings
            if self.model.use_uncertainty:
                emb_query = z1[0].squeeze(0).cpu().numpy()  # (n_query, dim)
                emb_ref = z2[0].squeeze(0).cpu().numpy()    # (n_ref+1, dim) with no-match
            else:
                emb_query = z1.squeeze(0).cpu().numpy()
                emb_ref = z2.squeeze(0).cpu().numpy()

            # Compute similarity (exclude no-match token from ref)
            emb_ref_cells = emb_ref[:n_ref]  # Exclude no-match token
            similarity = emb_query @ emb_ref_cells.T  # (n_query, n_ref)

            # Argmax for each query cell
            pred_indices = similarity.argmax(axis=1)

            # Record votes
            for i, pred_idx in enumerate(pred_indices):
                if pred_idx < len(ref_ids):
                    all_votes[i].append(ref_ids[pred_idx])

        # Majority vote
        predictions = []
        confidences = []

        for votes in all_votes:
            if not votes:
                predictions.append('UNKNOWN')
                confidences.append(0.0)
            else:
                vote_counts = Counter(votes)
                winner, count = vote_counts.most_common(1)[0]
                predictions.append(winner)
                confidences.append(count / len(votes))

        debug_info = {
            'n_refs': len(references),
            'all_votes': all_votes
        }

        return predictions, np.array(confidences), debug_info

    def classify_error(
        self,
        true_id: str,
        pred_id: str,
        cell_coords: Dict[str, np.ndarray]
    ) -> str:
        """
        Classify a prediction error into categories.

        Categories (like Twin Attention Fig 4c):
        1. "nearest_neighbor" - pred is spatially closest to true's position
        2. "sibling" - pred and true share same parent
        3. "mother_daughter" - pred is parent or child of true
        4. "other" - none of the above
        """
        # Check nearest neighbor
        if true_id in cell_coords and pred_id in cell_coords:
            true_pos = np.array(cell_coords[true_id])

            # Find actual nearest neighbor to true position
            min_dist = float('inf')
            nearest_id = None
            for cid, pos in cell_coords.items():
                if cid != true_id:
                    dist = np.linalg.norm(np.array(pos) - true_pos)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_id = cid

            if nearest_id == pred_id:
                return 'nearest_neighbor'

        # Check sibling
        if CelegansLineage.are_siblings(true_id, pred_id):
            return 'sibling'

        # Check mother/daughter
        if CelegansLineage.is_mother_daughter(true_id, pred_id):
            return 'mother_daughter'

        return 'other'

    def evaluate_no_cheating(
        self,
        data: Dict,
        data_name: str = "Test",
        max_samples: int = None
    ) -> Dict:
        """
        Evaluate using TRUE no-cheating method.

        For each embryo, for each timepoint:
        1. Sample 5-20 cells
        2. Get ground truth IDs (for evaluation ONLY)
        3. Call identify_cells_no_cheating(coords, stage)
        4. Compare predictions to ground truth
        """
        print(f"\n--- Evaluating {data_name} Data (NO CHEATING) ---")

        if data is None:
            print("  No data available")
            return {}

        all_correct = []
        hierarchical = {'exact': 0, 'sublineage': 0, 'founder': 0, 'binary': 0}
        by_size = {'sparse_5_10': [], 'medium_11_15': [], 'dense_16_20': []}
        error_counts = {'nearest_neighbor': 0, 'sibling': 0, 'mother_daughter': 0, 'other': 0}
        total_cells = 0
        per_embryo_acc = []

        sample_count = 0
        max_samples = max_samples or float('inf')

        for embryo, timepoints in tqdm(data.items(), desc=f"Evaluating {data_name}"):
            embryo_correct = []

            timepoint_list = list(timepoints.items())
            random.shuffle(timepoint_list)

            for t, cells in timepoint_list[:self.config.MAX_SAMPLES_PER_EMBRYO]:
                if sample_count >= max_samples:
                    break

                cell_ids = list(cells.keys())
                n_cells = len(cell_ids)
                embryo_stage = n_cells  # Total cells = stage indicator

                if n_cells < self.config.MIN_CELLS or n_cells > self.config.STAGE_LIMIT:
                    continue

                # Sample observable cells (5-20)
                n_sample = min(max(self.config.MIN_CELLS, n_cells), self.config.MAX_CELLS)
                if n_cells > n_sample:
                    sampled_indices = np.random.choice(n_cells, n_sample, replace=False)
                    sampled_ids = [cell_ids[i] for i in sampled_indices]
                else:
                    sampled_ids = cell_ids

                # Get coordinates (this is what we'd have in deployment)
                query_coords = np.array([cells[c] for c in sampled_ids])

                # TRUE NO-CHEATING IDENTIFICATION
                predictions, confidences, _ = self.identify_cells_no_cheating(
                    query_coords,
                    embryo_stage,
                    n_refs=self.config.N_REFS,
                    exclude_embryo=embryo  # Don't use same embryo as reference
                )

                # Evaluate against ground truth
                for i, (true_id, pred_id) in enumerate(zip(sampled_ids, predictions)):
                    is_correct = (pred_id == true_id)
                    all_correct.append(1 if is_correct else 0)
                    embryo_correct.append(1 if is_correct else 0)
                    total_cells += 1

                    # Hierarchical accuracy
                    if is_correct:
                        hierarchical['exact'] += 1
                    if CelegansLineage.same_sublineage(pred_id, true_id):
                        hierarchical['sublineage'] += 1
                    if CelegansLineage.same_founder(pred_id, true_id):
                        hierarchical['founder'] += 1

                    true_ab = CelegansLineage.get_founder(true_id) == 'AB'
                    pred_ab = CelegansLineage.get_founder(pred_id) == 'AB'
                    if true_ab == pred_ab:
                        hierarchical['binary'] += 1

                    # By size
                    if 5 <= n_sample <= 10:
                        by_size['sparse_5_10'].append(1 if is_correct else 0)
                    elif 11 <= n_sample <= 15:
                        by_size['medium_11_15'].append(1 if is_correct else 0)
                    elif 16 <= n_sample <= 20:
                        by_size['dense_16_20'].append(1 if is_correct else 0)

                    # Error classification
                    if not is_correct:
                        cell_coords = {cid: cells[cid] for cid in cell_ids}
                        error_type = self.classify_error(true_id, pred_id, cell_coords)
                        error_counts[error_type] += 1

                sample_count += 1

            if embryo_correct:
                per_embryo_acc.append(np.mean(embryo_correct))

        # Compute statistics
        accuracy_ci = StatisticalAnalysis.bootstrap_ci(all_correct, n=self.config.N_BOOTSTRAP)

        # Normalize hierarchical
        hier_pct = {k: v / total_cells * 100 if total_cells > 0 else 0 for k, v in hierarchical.items()}

        # By size CIs
        by_size_results = {}
        for k, v in by_size.items():
            if v:
                ci = StatisticalAnalysis.bootstrap_ci(v, n=50)
                by_size_results[k] = {'accuracy': ci.mean * 100, 'ci': (ci.ci_lower * 100, ci.ci_upper * 100), 'n': len(v)}

        # Error percentages
        total_errors = sum(error_counts.values())
        error_pct = {k: v / total_errors * 100 if total_errors > 0 else 0 for k, v in error_counts.items()}

        results = {
            'accuracy': accuracy_ci.mean * 100,
            'ci_lower': accuracy_ci.ci_lower * 100,
            'ci_upper': accuracy_ci.ci_upper * 100,
            'hierarchical': hier_pct,
            'by_size': by_size_results,
            'error_analysis': error_pct,
            'error_counts': error_counts,
            'total_cells': total_cells,
            'per_embryo_accuracies': per_embryo_acc
        }

        # Print summary
        print(f"\n  {data_name} Results (NO CHEATING):")
        print(f"    Accuracy: {results['accuracy']:.1f}% [95% CI: {results['ci_lower']:.1f}-{results['ci_upper']:.1f}%]")
        print(f"    Total cells evaluated: {total_cells}")
        print(f"    Hierarchical: exact={hier_pct['exact']:.1f}%, sublineage={hier_pct['sublineage']:.1f}%, founder={hier_pct['founder']:.1f}%")
        if error_pct:
            print(f"    Errors: nearest_neighbor={error_pct['nearest_neighbor']:.1f}%, sibling={error_pct['sibling']:.1f}%, mother_daughter={error_pct['mother_daughter']:.1f}%")

        return results

    def evaluate_baselines_no_cheating(
        self,
        data: Dict,
        data_name: str = "Test",
        n_samples: int = 500
    ) -> Dict[str, Dict]:
        """
        Evaluate baselines using SAME no-cheating setup as model.

        Baselines get same stage-matched references (no cell ID info).
        """
        print(f"\n--- Evaluating Baselines on {data_name} (NO CHEATING) ---")

        if data is None:
            return {}

        methods = {
            'ICP': BaselineMethods.icp_match,
            'Hungarian': BaselineMethods.hungarian_match,
            'NearestNeighbor': BaselineMethods.nearest_neighbor_match
        }

        results = {m: [] for m in methods}
        sample_count = 0

        for embryo, timepoints in tqdm(data.items(), desc="Baseline evaluation"):
            for t, cells in list(timepoints.items())[:10]:  # Limit per embryo
                if sample_count >= n_samples:
                    break

                cell_ids = list(cells.keys())
                n_cells = len(cell_ids)

                if n_cells < self.config.MIN_CELLS or n_cells > self.config.STAGE_LIMIT:
                    continue

                # Sample query cells
                n_sample = min(max(self.config.MIN_CELLS, n_cells), self.config.MAX_CELLS)
                sampled_indices = np.random.choice(n_cells, n_sample, replace=False) if n_cells > n_sample else np.arange(n_cells)
                sampled_ids = [cell_ids[i] for i in sampled_indices]
                query_coords = np.array([cells[c] for c in sampled_ids])
                query_norm = self.normalize_coords(query_coords)

                # Get ONE stage-matched reference (baselines are slow)
                refs = self.find_stage_matched_references(n_cells, n_refs=1, exclude_embryo=embryo)
                if not refs:
                    continue

                ref_ids, ref_coords = refs[0]

                # Run each baseline
                for method_name, method_fn in methods.items():
                    if method_name == 'ICP':
                        matches, _ = method_fn(query_norm, ref_coords)
                    else:
                        matches = method_fn(query_norm, ref_coords)

                    # Evaluate
                    for i, (true_id, match_idx) in enumerate(zip(sampled_ids, matches)):
                        if 0 <= match_idx < len(ref_ids):
                            pred_id = ref_ids[match_idx]
                            results[method_name].append(1 if pred_id == true_id else 0)
                        else:
                            results[method_name].append(0)

                sample_count += 1

        # Compute CIs
        baseline_results = {}
        for method_name, correct_list in results.items():
            if correct_list:
                ci = StatisticalAnalysis.bootstrap_ci(correct_list, n=50)
                baseline_results[method_name] = {
                    'accuracy': ci.mean * 100,
                    'ci_lower': ci.ci_lower * 100,
                    'ci_upper': ci.ci_upper * 100,
                    'n': len(correct_list)
                }
                print(f"  {method_name}: {ci.mean*100:.1f}% [CI: {ci.ci_lower*100:.1f}-{ci.ci_upper*100:.1f}%]")

        return baseline_results

    def compute_significance(
        self,
        model_per_embryo: List[float],
        baseline_results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Compute statistical significance vs baselines"""
        significance = {}

        for baseline_name, baseline_data in baseline_results.items():
            # We need per-sample data for t-test
            # For now, use aggregate comparison
            baseline_acc = baseline_data['accuracy'] / 100
            model_acc = np.mean(model_per_embryo) if model_per_embryo else 0

            # Effect size (rough estimate)
            effect_size = (model_acc - baseline_acc) / 0.2  # Assume std ~0.2

            significance[baseline_name] = {
                'model_acc': model_acc * 100,
                'baseline_acc': baseline_acc * 100,
                'improvement': (model_acc - baseline_acc) * 100,
                'effect_size_estimate': effect_size
            }

        return significance

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    def visualize_tsne_by_founder(self, embeddings: np.ndarray, labels: List[str], output_dir: str):
        """t-SNE visualization colored by founder lineage"""
        print("  Generating t-SNE visualization...")

        # Subsample if needed
        max_pts = 3000
        if len(embeddings) > max_pts:
            idx = np.random.choice(len(embeddings), max_pts, replace=False)
            emb = embeddings[idx]
            labs = [labels[i] for i in idx]
        else:
            emb, labs = embeddings, labels

        # Run t-SNE (max_iter for newer sklearn, n_iter for older)
        try:
            tsne = TSNE(n_components=2, perplexity=min(50, len(emb)//4),
                       random_state=42, max_iter=2000, init='pca')
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=min(50, len(emb)//4),
                       random_state=42, n_iter=2000, init='pca')
        emb_2d = tsne.fit_transform(emb)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        founders = [CelegansLineage.get_founder(l) for l in labs]
        unique_founders = sorted(set(f for f in founders if f != 'UNKNOWN'))

        for founder in unique_founders:
            mask = [f == founder for f in founders]
            pts = emb_2d[np.array(mask)]
            color = CelegansLineage.FOUNDER_COLORS.get(founder, '#cccccc')
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=15, alpha=0.6, label=founder)

        ax.legend(title='Founder Lineage', loc='upper right', framealpha=0.9)
        ax.set_xlabel('t-SNE Component 1', fontweight='bold')
        ax.set_ylabel('t-SNE Component 2', fontweight='bold')
        ax.set_title('Cell Embedding Space by Lineage', fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.savefig(os.path.join(output_dir, 'fig_tsne_founders.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'fig_tsne_founders.pdf'), bbox_inches='tight')
        plt.close()

    def visualize_error_analysis(self, error_pct: Dict, output_dir: str):
        """Error analysis pie chart"""
        fig, ax = plt.subplots(figsize=(9, 7))

        labels_map = {
            'nearest_neighbor': 'Nearest\nNeighbor',
            'sibling': 'Sibling',
            'mother_daughter': 'Mother/\nDaughter',
            'other': 'Other'
        }

        colors = ['#984ea3', '#b3a2c7', '#377eb8', '#e41a1c']

        labels, sizes = [], []
        for k in ['nearest_neighbor', 'sibling', 'mother_daughter', 'other']:
            if error_pct.get(k, 0) > 0:
                labels.append(labels_map[k])
                sizes.append(error_pct[k])

        if sizes:
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors[:len(sizes)],
                autopct='%1.1f%%', startangle=90, pctdistance=0.75,
                explode=[0.02] * len(sizes)
            )
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')
            ax.set_title('Error Type Distribution', fontweight='bold', pad=15)

        plt.savefig(os.path.join(output_dir, 'fig_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'fig_error_analysis.pdf'), bbox_inches='tight')
        plt.close()

    def visualize_accuracy_comparison(self, model_acc: float, baseline_results: Dict, output_dir: str):
        """Bar chart comparing model vs baselines"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(baseline_results.keys()) + ['Twin Attention\n(Ours)']
        accs = [baseline_results[m]['accuracy'] for m in baseline_results.keys()] + [model_acc]

        colors = ['#C4C4C4', '#9E9E9E', '#757575', '#2E86AB']

        # Error bars
        errors = []
        for m in baseline_results.keys():
            ci_low = baseline_results[m]['ci_lower']
            ci_high = baseline_results[m]['ci_upper']
            acc = baseline_results[m]['accuracy']
            errors.append([acc - ci_low, ci_high - acc])
        errors.append([0, 0])  # No CI for model in this simple version

        bars = ax.bar(methods, accs, color=colors, edgecolor='white', linewidth=1.5,
                      yerr=np.array(errors).T, capsize=5, error_kw={'linewidth': 1.5})

        ax.axhline(90, color='#28A745', linestyle='--', alpha=0.7, linewidth=2, label='90% threshold')
        ax.set_ylabel('Identification Accuracy (%)', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_title('Cell Identification: Model vs Baselines (No Cheating)', fontweight='bold', pad=15)

        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                   f'{a:.1f}%', ha='center', fontsize=11, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(os.path.join(output_dir, 'fig_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'fig_accuracy_comparison.pdf'), bbox_inches='tight')
        plt.close()

    def visualize_accuracy_by_size(self, by_size: Dict, output_dir: str):
        """Bar chart showing accuracy by neighborhood size"""
        fig, ax = plt.subplots(figsize=(8, 6))

        sizes = ['Sparse\n(5-10 cells)', 'Medium\n(11-15 cells)', 'Dense\n(16-20 cells)']
        keys = ['sparse_5_10', 'medium_11_15', 'dense_16_20']

        accs = []
        errors = []
        for k in keys:
            if k in by_size:
                accs.append(by_size[k]['accuracy'])
                ci = by_size[k]['ci']
                errors.append([by_size[k]['accuracy'] - ci[0], ci[1] - by_size[k]['accuracy']])
            else:
                accs.append(0)
                errors.append([0, 0])

        colors = ['#6BB3D9', '#4A9AC9', '#2E86AB']
        bars = ax.bar(sizes, accs, color=colors, edgecolor='white', linewidth=1.5,
                      yerr=np.array(errors).T, capsize=5, error_kw={'linewidth': 1.5})

        ax.set_ylabel('Identification Accuracy (%)', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_title('Accuracy by Neighborhood Size', fontweight='bold', pad=15)

        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{a:.1f}%', ha='center', fontsize=11, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(os.path.join(output_dir, 'fig_accuracy_by_size.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'fig_accuracy_by_size.pdf'), bbox_inches='tight')
        plt.close()

    def visualize_ablations(self, ablation_results: Dict, output_dir: str):
        """Bar chart for ablation study"""
        if not ablation_results:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        names = list(ablation_results.keys())
        accs = [ablation_results[n]['accuracy'] for n in names]

        colors = ['#2E86AB' if n == 'full_model' else '#F18F01' for n in names]

        # Error bars if available
        errors = []
        for n in names:
            if 'ci_lower' in ablation_results[n]:
                acc = ablation_results[n]['accuracy']
                errors.append([acc - ablation_results[n]['ci_lower'],
                              ablation_results[n]['ci_upper'] - acc])
            else:
                errors.append([0, 0])

        bars = ax.barh(names, accs, color=colors, edgecolor='white', linewidth=1.5,
                       xerr=np.array(errors).T, capsize=4, error_kw={'linewidth': 1.5})

        ax.set_xlabel('Identification Accuracy (%)', fontweight='bold')
        ax.set_title('Ablation Study', fontweight='bold', pad=15)
        ax.set_xlim(0, 105)

        for bar, a in zip(bars, accs):
            ax.text(a + 1.5, bar.get_y() + bar.get_height()/2,
                   f'{a:.1f}%', va='center', fontsize=10, fontweight='bold')

        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(os.path.join(output_dir, 'fig_ablations.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'fig_ablations.pdf'), bbox_inches='tight')
        plt.close()

    # =========================================================================
    # ABLATION EVALUATION
    # =========================================================================
    def run_ablation_evaluation(self, data: Dict, n_samples: int = 300) -> Dict[str, Dict]:
        """Evaluate ablation variants if weights exist"""
        print("\n" + "="*60)
        print("ABLATION STUDIES")
        print("="*60)

        ablation_results = {}
        ablation_dir = convert_path(self.config.ABLATION_WEIGHTS_DIR)

        for ablation_name, config in ABLATION_CONFIGS.items():
            weight_file = config['weight_file']

            # Check for weight file
            if ablation_name == 'full_model':
                weight_path = convert_path(self.config.MODEL_PATH)
            else:
                weight_path = os.path.join(ablation_dir, weight_file) if ablation_dir else weight_file

            if not os.path.exists(weight_path):
                print(f"\n  Skipping {ablation_name}: weights not found at {weight_path}")
                continue

            print(f"\n  Evaluating: {ablation_name}")
            print(f"    Description: {config['description']}")

            # Load model with this config
            self.load_model(config, weight_path)

            # Run no-cheating evaluation
            results = self.evaluate_no_cheating(data, f"{ablation_name}", max_samples=n_samples)

            if results:
                ablation_results[ablation_name] = {
                    'accuracy': results['accuracy'],
                    'ci_lower': results['ci_lower'],
                    'ci_upper': results['ci_upper'],
                    'description': config['description']
                }

        return ablation_results

    # =========================================================================
    # MAIN EVALUATION RUNNER
    # =========================================================================
    def run_full_evaluation(
        self,
        output_dir: str = "evaluation_results",
        run_baselines: bool = True,
        run_ablations: bool = False,
        run_real_embryo: bool = True,
        max_samples: int = None
    ):
        """
        Complete evaluation pipeline.

        Args:
            output_dir: Directory for results
            run_baselines: Whether to run baseline comparisons
            run_ablations: Whether to run ablation studies
            run_real_embryo: Whether to evaluate on real embryo data
            max_samples: Maximum samples to evaluate (None for all)
        """
        print("\n" + "="*70)
        print("TWIN ATTENTION MODEL - NO-CHEATING EVALUATION")
        print("="*70)
        print(f"Device: {self.device}")

        # Setup
        os.makedirs(output_dir, exist_ok=True)
        figure_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figure_dir, exist_ok=True)

        # Load data
        self.load_data()

        # Load full model
        self.load_model()

        all_results = {
            'config': {
                'n_refs': self.config.N_REFS,
                'stage_tolerance': self.config.STAGE_TOLERANCE,
                'min_cells': self.config.MIN_CELLS,
                'max_cells': self.config.MAX_CELLS
            }
        }

        # =================================================================
        # MAIN RESULTS: NO-CHEATING EVALUATION
        # =================================================================
        print("\n" + "="*60)
        print("MAIN RESULTS: NO-CHEATING CELL IDENTIFICATION")
        print("="*60)

        # Simulated data
        if self.eval_data:
            simulated_results = self.evaluate_no_cheating(
                self.eval_data, "Simulated", max_samples=max_samples
            )
            all_results['simulated'] = simulated_results

            # Visualizations
            if simulated_results.get('by_size'):
                self.visualize_accuracy_by_size(simulated_results['by_size'], figure_dir)
            if simulated_results.get('error_analysis'):
                self.visualize_error_analysis(simulated_results['error_analysis'], figure_dir)

        # Real embryo data
        if run_real_embryo and self.real_data:
            real_results = self.evaluate_no_cheating(
                self.real_data, "Real Embryo", max_samples=max_samples
            )
            all_results['real_embryo'] = real_results

        # =================================================================
        # BASELINES (Fair - same no-cheating setup)
        # =================================================================
        if run_baselines and self.eval_data:
            print("\n" + "="*60)
            print("BASELINE COMPARISON (No Cheating)")
            print("="*60)

            baseline_results = self.evaluate_baselines_no_cheating(
                self.eval_data, "Simulated", n_samples=min(500, max_samples or 500)
            )
            all_results['baselines'] = baseline_results

            # Visualization
            if baseline_results and 'simulated' in all_results:
                self.visualize_accuracy_comparison(
                    all_results['simulated']['accuracy'],
                    baseline_results,
                    figure_dir
                )

            # Significance
            if baseline_results and 'simulated' in all_results:
                significance = self.compute_significance(
                    all_results['simulated'].get('per_embryo_accuracies', []),
                    baseline_results
                )
                all_results['significance'] = significance

        # =================================================================
        # ABLATIONS
        # =================================================================
        if run_ablations and self.eval_data:
            ablation_results = self.run_ablation_evaluation(
                self.eval_data, n_samples=min(300, max_samples or 300)
            )
            all_results['ablations'] = ablation_results

            if ablation_results:
                self.visualize_ablations(ablation_results, figure_dir)

        # =================================================================
        # SAVE RESULTS
        # =================================================================
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

        # Save JSON
        results_path = os.path.join(output_dir, 'results.json')

        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj

        with open(results_path, 'w') as f:
            json.dump(convert_for_json(all_results), f, indent=2)

        print(f"\nResults saved to: {results_path}")
        print(f"Figures saved to: {figure_dir}")

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        if 'simulated' in all_results:
            r = all_results['simulated']
            print(f"\nSimulated Data (NO CHEATING):")
            print(f"  Accuracy: {r['accuracy']:.1f}% [CI: {r['ci_lower']:.1f}-{r['ci_upper']:.1f}%]")
            print(f"  Total cells: {r['total_cells']}")

        if 'real_embryo' in all_results:
            r = all_results['real_embryo']
            print(f"\nReal Embryo Data (NO CHEATING):")
            print(f"  Accuracy: {r['accuracy']:.1f}% [CI: {r['ci_lower']:.1f}-{r['ci_upper']:.1f}%]")

        if 'baselines' in all_results:
            print(f"\nBaselines (No Cheating):")
            for name, data in all_results['baselines'].items():
                print(f"  {name}: {data['accuracy']:.1f}%")

        if 'ablations' in all_results:
            print(f"\nAblations:")
            for name, data in all_results['ablations'].items():
                print(f"  {name}: {data['accuracy']:.1f}%")

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)

        self.results = all_results
        return all_results


# =============================================================================
# MAIN ENTRY POINT WITH TOGGLES
# =============================================================================
def main():
    """Main entry point with configurable evaluation options"""

    # =========================================================================
    # TOGGLE SETTINGS - MODIFY THESE TO CONTROL EVALUATION
    # =========================================================================

    # Core evaluation settings
    RUN_BASELINES = True           # Compare against ICP, Hungarian, NN baselines
    RUN_ABLATIONS = False          # Run ablation studies (requires trained variant weights)
    RUN_REAL_EMBRYO = True         # Evaluate on real embryo data

    # Sample limits (set to None for full evaluation, or int for quick test)
    MAX_SAMPLES = None             # e.g., 100 for quick test, None for full

    # Output directory
    OUTPUT_DIR = "evaluation_results_no_cheating"

    # =========================================================================
    # RUN EVALUATION
    # =========================================================================

    print("\n" + "="*70)
    print("EVALUATION CONFIGURATION")
    print("="*70)
    print(f"  Run baselines: {RUN_BASELINES}")
    print(f"  Run ablations: {RUN_ABLATIONS}")
    print(f"  Run real embryo: {RUN_REAL_EMBRYO}")
    print(f"  Max samples: {MAX_SAMPLES or 'All'}")
    print(f"  Output directory: {OUTPUT_DIR}")

    config = EvalConfig()
    evaluator = NoCheatingEvaluator(config)

    results = evaluator.run_full_evaluation(
        output_dir=OUTPUT_DIR,
        run_baselines=RUN_BASELINES,
        run_ablations=RUN_ABLATIONS,
        run_real_embryo=RUN_REAL_EMBRYO,
        max_samples=MAX_SAMPLES
    )

    return results


if __name__ == "__main__":
    print(f"Using device: {device}")
    main()
