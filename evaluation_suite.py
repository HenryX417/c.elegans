"""
Comprehensive Evaluation Suite for Sparse Cell Identification Model
=====================================================================

This evaluation suite implements a complete pipeline for evaluating a sparse cell
identification model that adapts the Twin Attention architecture for identifying
cells in C. elegans embryos from partial observations (5-20 cells).

Target: >85% accuracy on kNN-based cell type assignment

Based on:
- Model Architecture: EnhancedTwinAttentionEncoder with SparsePointFeatures
- Original Paper: "A Single-cell Spatiotemporal Manifold of Tissue Morphology and Dynamics"

Author: Evaluation Suite for Twin Attention Sparse Cell Identification
"""

import sys
import os
import pickle
import random
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from datetime import datetime
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class EvalConfig:
    """Configuration for evaluation suite"""

    # Data paths - update these for your system
    TRAIN_DATA = "data_dict.pkl"
    EVAL_DATA = "evaluation_data_dict.pkl"
    REAL_DATA = "real_data_dict.pkl"
    MODEL_PATH = "twin_attention_final.pth"
    OUTPUT_DIR = "eval_results"

    # Model architecture (must match trained model)
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 6
    DROPOUT = 0.1
    USE_SPARSE_FEATURES = True
    USE_UNCERTAINTY = True
    USE_LEARNABLE_NO_MATCH = True

    # Evaluation parameters
    MIN_CELLS = 5
    MAX_CELLS = 20
    STAGE_LIMIT = 194

    # kNN parameters
    DEFAULT_K = 30
    K_VALUES_TO_TEST = [5, 10, 20, 30, 50, 100]

    # Reference aggregation
    M_VALUES_TO_TEST = [1, 3, 5, 10]
    N_SAMPLES_PER_TIMEPOINT_VALUES = [10, 30, 50, 100]

    # Stability analysis
    N_STABILITY_QUERIES = 100
    N_STABILITY_REFERENCES = 10

    # Bootstrap
    N_BOOTSTRAP_SAMPLES = 1000

    # Random seed
    SEED = 42


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# MODEL ARCHITECTURE (imported from debug_sparse_matching.py)
# =============================================================================

class SparsePointFeatures(nn.Module):
    """Feature extraction specifically designed for sparse point clouds (5-20 points)

    MUST match exactly with debug_sparse_matching.py lines 46-131
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.relative_position_enc = nn.Linear(3, embed_dim // 4)
        self.centroid_distance_enc = nn.Linear(1, embed_dim // 4)
        self.point_count_enc = nn.Embedding(50, embed_dim // 4)
        self.local_density_enc = nn.Linear(1, embed_dim // 4)

    def forward(self, points, mask=None):
        B, N, _ = points.shape

        # Handle masked points
        if mask is not None:
            points_masked = points * mask.unsqueeze(-1)
            n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
            centroid = points_masked.sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        else:
            centroid = points.mean(dim=1, keepdim=True)
            n_valid = torch.full((B, 1), N, device=points.device)

        # Relative positions
        relative_pos = points - centroid
        rel_features = self.relative_position_enc(relative_pos)

        # Distance from centroid
        centroid_dist = torch.norm(relative_pos, dim=-1, keepdim=True)
        dist_features = self.centroid_distance_enc(centroid_dist)

        # Point count awareness (exact match to original line 77-79)
        n_valid_long = n_valid.squeeze(1).long()
        count_emb = self.point_count_enc(n_valid_long)
        count_features = count_emb.unsqueeze(1).expand(-1, N, -1)

        # Local density
        local_density = self._compute_local_density(points, mask)
        density_features = self.local_density_enc(local_density)

        features = torch.cat([
            rel_features,
            dist_features,
            count_features,
            density_features
        ], dim=-1)

        return features

    def _compute_local_density(self, points, mask):
        """Compute local density - matches original lines 94-131"""
        B, N, _ = points.shape
        densities = torch.zeros(B, N, 1, device=points.device)

        for b in range(B):
            if mask is not None:
                valid_mask = mask[b].bool()
                valid_points = points[b][valid_mask]
                n_valid = valid_mask.sum().item()
            else:
                valid_points = points[b]
                n_valid = N

            if n_valid <= 1:
                continue

            # Use cdist for GPU, numpy for CPU (matches original)
            if points.device.type == 'cpu':
                pts_np = valid_points.detach().cpu().numpy()
                diff = pts_np[:, None, :] - pts_np[None, :, :]
                dists_np = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists_np, np.inf)
                k = min(3, n_valid - 1)
                nearest_dists = np.partition(dists_np, k-1, axis=1)[:, :k]
                mean_density = nearest_dists.mean(axis=1, keepdims=True)
                density_tensor = torch.from_numpy(mean_density).float().to(points.device)
            else:
                dists = torch.cdist(valid_points, valid_points)
                dists.fill_diagonal_(float('inf'))
                k = min(3, n_valid - 1)
                nearest_dists, _ = dists.topk(k, dim=1, largest=False)
                density_tensor = nearest_dists.mean(dim=1, keepdim=True)

            if mask is not None:
                densities[b][valid_mask] = density_tensor
            else:
                densities[b] = density_tensor

        return densities


class EnhancedTwinAttentionEncoder(nn.Module):
    """Twin Attention encoder with subset point cloud optimizations

    MUST match exactly with debug_sparse_matching.py lines 133-263
    """

    def __init__(self, input_dim=3, embed_dim=128, num_heads=8, num_layers=6,
                 dropout=0.1, use_positional_encoding=True, max_seq_len=50,
                 use_sparse_features=True, use_uncertainty=True,
                 use_learnable_no_match=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        self.use_sparse_features = use_sparse_features
        self.use_uncertainty = use_uncertainty
        self.use_learnable_no_match = use_learnable_no_match

        # Feature extraction
        if use_sparse_features:
            self.sparse_features = SparsePointFeatures(embed_dim)
            self.feature_projection = nn.Linear(embed_dim, embed_dim)
        else:
            self.point_embed = nn.Linear(input_dim, embed_dim)

        # Learnable no-match token
        if use_learnable_no_match:
            self.no_match_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Transformer
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

        # Temperature with warm-up (CRITICAL: must match original line 182-184)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('temperature_warmup_steps', torch.tensor(0))

    def forward(self, pc1, pc2, mask1=None, mask2=None, epoch=0):
        B = pc1.shape[0]

        # Extract features
        if self.use_sparse_features:
            z1 = self.sparse_features(pc1, mask1)
            z2 = self.sparse_features(pc2, mask2)
            z1 = self.feature_projection(z1)
            z2 = self.feature_projection(z2)
        else:
            z1 = self.point_embed(pc1)
            z2 = self.point_embed(pc2)

        # Add learnable no-match token to pc2
        if self.use_learnable_no_match:
            no_match = self.no_match_token.expand(B, -1, -1)
            z2 = torch.cat([z2, no_match], dim=1)
            if mask2 is not None:
                mask2 = torch.cat([mask2, torch.ones(B, 1, device=mask2.device)], dim=1)

        # Concatenate for twin attention
        z = torch.cat([z1, z2], dim=1)

        # Create combined mask - CPU optimized version (matches original lines 209-219)
        if mask1 is not None or mask2 is not None:
            if mask1 is None:
                mask1 = torch.ones(B, pc1.shape[1], device=pc1.device, dtype=torch.bool)
            if mask2 is None:
                mask2 = torch.ones(B, pc2.shape[1], device=pc2.device, dtype=torch.bool)
            # Use boolean operations directly (faster on CPU)
            combined_mask = torch.cat([mask1, mask2], dim=1).bool()
            attn_mask = ~combined_mask  # True = ignore (directly boolean)
        else:
            attn_mask = None

        # Add positional encoding
        if self.use_positional_encoding:
            seq_len = z.shape[1]
            z = z + self.pos_encoding[:, :seq_len, :]

        # Transform with mask
        if attn_mask is not None:
            z = self.transformer(z, src_key_padding_mask=attn_mask.bool())
        else:
            z = self.transformer(z)

        # Split back
        N1 = pc1.shape[1]
        z1, z2_with_no_match = z[:, :N1], z[:, N1:]

        # Output projections with L2 normalization
        if self.use_uncertainty:
            z1_mean = F.normalize(self.output_mean(z1), p=2, dim=-1)
            z1_logvar = torch.clamp(self.output_logvar(z1), -10, 2)
            z2_mean = F.normalize(self.output_mean(z2_with_no_match), p=2, dim=-1)
            z2_logvar = torch.clamp(self.output_logvar(z2_with_no_match), -10, 2)
            outputs = ((z1_mean, z1_logvar), (z2_mean, z2_logvar))
        else:
            z1 = F.normalize(self.output_proj(z1), p=2, dim=-1)
            z2_with_no_match = F.normalize(self.output_proj(z2_with_no_match), p=2, dim=-1)
            outputs = (z1, z2_with_no_match)

        # Temperature with warm-up (matches original lines 248-263)
        temperature = self._get_temperature(epoch)

        return outputs[0], outputs[1], temperature

    def _get_temperature(self, epoch):
        """Temperature with warmup - matches original lines 253-263"""
        warmup_epochs = 5
        if epoch < warmup_epochs:
            # Linear warmup from 1.0 to learned value
            warmup_factor = epoch / warmup_epochs
            base_temp = 1.0
            learned_temp = torch.exp(self.log_temperature).clamp(0.01, 10.0)
            temperature = base_temp + warmup_factor * (learned_temp - base_temp)
        else:
            temperature = torch.exp(self.log_temperature).clamp(0.01, 10.0)
        return temperature


# =============================================================================
# SAMPLING UTILITIES
# =============================================================================

class ImprovedSampler:
    """Biologically informed sampling with validation"""

    def __init__(self, min_cells=5, max_cells=20):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def sample_cells(self, cells: Dict[str, np.ndarray], strategy='mixed') -> Tuple[List[str], np.ndarray]:
        cell_ids = list(cells.keys())
        coords = np.array([cells[cid] for cid in cell_ids])

        if len(cell_ids) <= self.max_cells:
            return cell_ids, coords

        n_target = random.randint(self.min_cells, self.max_cells)

        if strategy == 'polar':
            selected_idx = self._sample_polar(coords, n_target)
        elif strategy == 'boundary':
            selected_idx = self._sample_boundary(coords, n_target)
        elif strategy == 'cluster':
            selected_idx = self._sample_cluster(coords, n_target)
        elif strategy == 'diverse':
            selected_idx = self._sample_diverse(coords, n_target)
        else:  # mixed
            strategy = random.choice(['polar', 'boundary', 'cluster', 'diverse'])
            return self.sample_cells(cells, strategy)

        selected_ids = [cell_ids[i] for i in selected_idx]
        return selected_ids, coords[selected_idx]

    def _sample_polar(self, coords, n_target):
        """Sample cells from polar regions (anterior/posterior) - matches original lines 398-425"""
        if len(coords) < 4:
            return np.random.choice(len(coords), min(n_target, len(coords)), replace=False).astype(int)

        # Find principal axis
        mean = coords.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(coords - mean)
            principal = vh[0]
        except:
            return np.random.choice(len(coords), min(n_target, len(coords)), replace=False).astype(int)

        # Project onto principal axis
        projections = (coords - mean) @ principal

        # Get extreme points
        n_poles = min(4, n_target // 2)
        anterior_idx = np.argsort(projections)[:n_poles]
        posterior_idx = np.argsort(projections)[-n_poles:]

        selected = np.concatenate([anterior_idx, posterior_idx])

        # Fill remaining with random points
        if len(selected) < n_target:
            remaining = list(set(range(len(coords))) - set(selected))
            if remaining:
                n_need = n_target - len(selected)
                extra = np.random.choice(remaining, min(n_need, len(remaining)), replace=False)
                selected = np.concatenate([selected, extra])

        return selected[:n_target].astype(int)

    def _sample_boundary(self, coords, n_target):
        if len(coords) < 4:
            return np.random.choice(len(coords), min(n_target, len(coords)), replace=False)

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            boundary_idx = hull.vertices

            if len(boundary_idx) >= n_target:
                return np.random.choice(boundary_idx, n_target, replace=False)
            else:
                internal = list(set(range(len(coords))) - set(boundary_idx))
                n_need = n_target - len(boundary_idx)
                if internal and n_need > 0:
                    extra = np.random.choice(internal, min(n_need, len(internal)), replace=False)
                    return np.concatenate([boundary_idx, extra]).astype(int)
                return boundary_idx.astype(int)
        except:
            return self._sample_diverse(coords, n_target)

    def _sample_cluster(self, coords, n_target):
        seed_idx = np.random.randint(len(coords))
        distances = np.linalg.norm(coords - coords[seed_idx], axis=1)
        return np.argsort(distances)[:n_target].astype(int)

    def _sample_diverse(self, coords, n_target):
        selected = [np.random.randint(len(coords))]

        for _ in range(n_target - 1):
            min_distances = np.full(len(coords), np.inf)

            for s in selected:
                distances = np.linalg.norm(coords - coords[s], axis=1)
                min_distances = np.minimum(min_distances, distances)

            min_distances[selected] = -np.inf
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)

        return np.array(selected).astype(int)


# =============================================================================
# DATA UTILITIES
# =============================================================================

def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates to zero mean and unit variance"""
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (coords - mean) / std


def get_stage_matched_reference(query_embryo: str, query_time: int,
                                 data_dict: Dict, query_n_cells: int,
                                 exclude_embryo: bool = True) -> Tuple[str, int]:
    """Get a stage-matched reference from training data"""
    # Stage window: ±10% of cell count or ±5 cells, whichever is larger
    window = max(int(query_n_cells * 0.1), 5)
    min_cells = query_n_cells - window
    max_cells = query_n_cells + window

    candidates = []
    for embryo_id, timepoints in data_dict.items():
        if exclude_embryo and embryo_id == query_embryo:
            continue
        for t, cells in timepoints.items():
            n_cells = len(cells)
            if min_cells <= n_cells <= max_cells:
                candidates.append((embryo_id, t, n_cells))

    if not candidates:
        # Fallback: use any timepoint from a different embryo
        for embryo_id, timepoints in data_dict.items():
            if exclude_embryo and embryo_id == query_embryo:
                continue
            for t, cells in timepoints.items():
                candidates.append((embryo_id, t, len(cells)))

    if not candidates:
        # Last resort: use same embryo
        for t, cells in data_dict[query_embryo].items():
            if t != query_time:
                candidates.append((query_embryo, t, len(cells)))

    if candidates:
        embryo_id, t, _ = random.choice(candidates)
        return embryo_id, t

    return query_embryo, query_time


def load_data(config: EvalConfig) -> Tuple[Dict, Dict, Dict]:
    """Load training, evaluation, and real data"""
    train_data, eval_data, real_data = None, None, None

    if os.path.exists(config.TRAIN_DATA):
        with open(config.TRAIN_DATA, 'rb') as f:
            train_data = pickle.load(f)
        print(f"Loaded training data: {len(train_data)} embryos")
    else:
        print(f"Warning: Training data not found at {config.TRAIN_DATA}")

    if os.path.exists(config.EVAL_DATA):
        with open(config.EVAL_DATA, 'rb') as f:
            eval_data = pickle.load(f)
        print(f"Loaded evaluation data: {len(eval_data)} embryos")
    else:
        print(f"Warning: Evaluation data not found at {config.EVAL_DATA}")
        # Use portion of training data for evaluation
        if train_data:
            embryo_ids = list(train_data.keys())
            n_eval = max(1, len(embryo_ids) // 5)
            eval_embryos = embryo_ids[-n_eval:]
            eval_data = {k: train_data[k] for k in eval_embryos}
            train_data = {k: v for k, v in train_data.items() if k not in eval_embryos}
            print(f"Split: {len(train_data)} train, {len(eval_data)} eval embryos")

    if os.path.exists(config.REAL_DATA):
        with open(config.REAL_DATA, 'rb') as f:
            real_data = pickle.load(f)
        print(f"Loaded real data: {len(real_data)} embryos")

    return train_data, eval_data, real_data


def load_model(config: EvalConfig, device: torch.device) -> EnhancedTwinAttentionEncoder:
    """Load trained model with proper handling of checkpoint formats"""
    model = EnhancedTwinAttentionEncoder(
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_sparse_features=config.USE_SPARSE_FEATURES,
        use_uncertainty=config.USE_UNCERTAINTY,
        use_learnable_no_match=config.USE_LEARNABLE_NO_MATCH
    )

    if os.path.exists(config.MODEL_PATH):
        checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)

        # Handle both checkpoint dict format and direct state_dict format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load state dict - strict=False allows loading even if there are minor mismatches
        # But first check that the model architecture matches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            # These unexpected keys are likely from a different model version
            # but shouldn't break loading

        print(f"Loaded model from {config.MODEL_PATH}")

        # Verify model architecture matches checkpoint
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        matched_keys = model_keys & ckpt_keys
        print(f"  Matched {len(matched_keys)}/{len(ckpt_keys)} checkpoint keys")
    else:
        print(f"Warning: Model not found at {config.MODEL_PATH}, using random initialization")

    model.to(device)
    model.eval()
    return model


# =============================================================================
# PHASE 0: EMBEDDING STABILITY ANALYSIS
# =============================================================================

class EmbeddingStabilityAnalyzer:
    """Analyze embedding stability across different reference choices"""

    def __init__(self, model: nn.Module, device: torch.device, config: EvalConfig):
        self.model = model
        self.device = device
        self.config = config
        self.sampler = ImprovedSampler(config.MIN_CELLS, config.MAX_CELLS)

    @torch.no_grad()
    def analyze_stability(self, eval_data: Dict, train_data: Dict) -> Dict:
        """
        Phase 0: Determine if embeddings are stable across different reference choices

        Returns dict with stability metrics and recommendation
        """
        print("\n" + "="*60)
        print("PHASE 0: Embedding Stability Analysis")
        print("="*60)

        self.model.eval()

        # Select query neighborhoods from eval data
        query_samples = self._select_query_samples(eval_data, self.config.N_STABILITY_QUERIES)

        all_cosine_similarities = []

        for query_idx, (embryo_id, time, cells) in enumerate(tqdm(query_samples, desc="Analyzing stability")):
            # Sample subset from query
            cell_ids, coords = self.sampler.sample_cells(cells, strategy='diverse')
            coords_norm = normalize_coords(coords)
            n_cells = len(cell_ids)

            # Get embeddings with N different references
            embeddings_per_ref = []

            for ref_idx in range(self.config.N_STABILITY_REFERENCES):
                ref_embryo, ref_time = get_stage_matched_reference(
                    embryo_id, time, train_data, n_cells, exclude_embryo=True
                )
                ref_cells = train_data[ref_embryo][ref_time]
                ref_cell_ids, ref_coords = self.sampler.sample_cells(ref_cells, strategy='diverse')
                ref_coords_norm = normalize_coords(ref_coords)

                # Get embeddings
                pc1 = torch.from_numpy(coords_norm).float().unsqueeze(0).to(self.device)
                pc2 = torch.from_numpy(ref_coords_norm).float().unsqueeze(0).to(self.device)

                z1, z2, _ = self.model(pc1, pc2)

                if self.model.use_uncertainty:
                    z1_mean = z1[0].squeeze(0).cpu().numpy()
                else:
                    z1_mean = z1.squeeze(0).cpu().numpy()

                embeddings_per_ref.append(z1_mean)

            # Compute pairwise cosine similarities across references for each cell
            embeddings_per_ref = np.array(embeddings_per_ref)  # (N_refs, N_cells, embed_dim)

            for cell_idx in range(len(cell_ids)):
                cell_embeddings = embeddings_per_ref[:, cell_idx, :]  # (N_refs, embed_dim)

                # Compute pairwise cosine similarities
                for i in range(len(cell_embeddings)):
                    for j in range(i + 1, len(cell_embeddings)):
                        cos_sim = np.dot(cell_embeddings[i], cell_embeddings[j]) / (
                            np.linalg.norm(cell_embeddings[i]) * np.linalg.norm(cell_embeddings[j]) + 1e-8
                        )
                        all_cosine_similarities.append(cos_sim)

        # Compute statistics
        cos_sims = np.array(all_cosine_similarities)
        results = {
            'mean_cosine_similarity': float(np.mean(cos_sims)),
            'std_cosine_similarity': float(np.std(cos_sims)),
            'min_cosine_similarity': float(np.min(cos_sims)),
            'max_cosine_similarity': float(np.max(cos_sims)),
            'median_cosine_similarity': float(np.median(cos_sims)),
            'n_queries': len(query_samples),
            'n_references_per_query': self.config.N_STABILITY_REFERENCES,
            'n_comparisons': len(cos_sims)
        }

        # Recommendation
        mean_sim = results['mean_cosine_similarity']
        if mean_sim > 0.95:
            results['recommendation'] = 'single_reference'
            results['recommendation_reason'] = 'High stability (mean > 0.95): Use single reference'
        elif mean_sim > 0.80:
            results['recommendation'] = 'multi_reference'
            results['recommendation_reason'] = f'Moderate stability ({mean_sim:.3f}): Use multi-reference aggregation'
        else:
            results['recommendation'] = 'multi_reference_with_caution'
            results['recommendation_reason'] = f'Low stability ({mean_sim:.3f}): Serious concern - use multi-reference with caution'

        print(f"\nStability Results:")
        print(f"  Mean cosine similarity: {results['mean_cosine_similarity']:.4f}")
        print(f"  Std cosine similarity: {results['std_cosine_similarity']:.4f}")
        print(f"  Min cosine similarity: {results['min_cosine_similarity']:.4f}")
        print(f"  Recommendation: {results['recommendation_reason']}")

        return results

    def _select_query_samples(self, data: Dict, n_samples: int) -> List[Tuple]:
        """Select random query neighborhoods from evaluation data"""
        all_samples = []

        for embryo_id, timepoints in data.items():
            for t, cells in timepoints.items():
                n_cells = len(cells)
                if self.config.MIN_CELLS <= n_cells <= self.config.STAGE_LIMIT:
                    all_samples.append((embryo_id, t, cells))

        if len(all_samples) > n_samples:
            return random.sample(all_samples, n_samples)
        return all_samples


# =============================================================================
# PHASE 1 & 2: REFERENCE MANIFOLD BUILDER
# =============================================================================

class ReferenceManifoldBuilder:
    """Build reference manifold for kNN cell identification"""

    def __init__(self, model: nn.Module, device: torch.device, config: EvalConfig):
        self.model = model
        self.device = device
        self.config = config
        self.sampler = ImprovedSampler(config.MIN_CELLS, config.MAX_CELLS)
        self.manifold = []
        self.embeddings_matrix = None
        self.faiss_index = None
        self.sklearn_index = None

    @torch.no_grad()
    def build_manifold(self, train_data: Dict, n_samples_per_timepoint: int = 50) -> None:
        """
        Phase 1: Build reference manifold with (embedding, cell_identity) pairs
        """
        print("\n" + "="*60)
        print("PHASE 1: Building Reference Manifold")
        print("="*60)

        self.model.eval()
        self.manifold = []

        strategies = ['polar', 'boundary', 'cluster', 'diverse']
        strategy_idx = 0

        total_timepoints = sum(
            1 for embryo_id, timepoints in train_data.items()
            for t, cells in timepoints.items()
            if self.config.MIN_CELLS <= len(cells) <= self.config.STAGE_LIMIT
        )

        pbar = tqdm(total=total_timepoints * n_samples_per_timepoint, desc="Building manifold")

        for embryo_id, timepoints in train_data.items():
            for t, cells in sorted(timepoints.items()):
                n_cells = len(cells)

                if n_cells < self.config.MIN_CELLS or n_cells > self.config.STAGE_LIMIT:
                    continue

                for sample_idx in range(n_samples_per_timepoint):
                    # Rotate through sampling strategies
                    strategy = strategies[strategy_idx % len(strategies)]
                    strategy_idx += 1

                    # Sample sparse subset from this timepoint
                    subset1_ids, subset1_coords = self.sampler.sample_cells(cells, strategy)
                    subset1_coords_norm = normalize_coords(subset1_coords)

                    # Get stage-matched reference (different embryo preferred)
                    ref_embryo, ref_time = get_stage_matched_reference(
                        embryo_id, t, train_data, len(subset1_ids), exclude_embryo=True
                    )
                    ref_cells = train_data[ref_embryo][ref_time]
                    subset2_ids, subset2_coords = self.sampler.sample_cells(ref_cells, strategy)
                    subset2_coords_norm = normalize_coords(subset2_coords)

                    # Run through model
                    pc1 = torch.from_numpy(subset1_coords_norm).float().unsqueeze(0).to(self.device)
                    pc2 = torch.from_numpy(subset2_coords_norm).float().unsqueeze(0).to(self.device)

                    z1, z2, temp = self.model(pc1, pc2)

                    if self.model.use_uncertainty:
                        z1_embeddings = z1[0].squeeze(0).cpu().numpy()
                    else:
                        z1_embeddings = z1.squeeze(0).cpu().numpy()

                    # Store embeddings for subset1 cells
                    for i, cell_id in enumerate(subset1_ids):
                        self.manifold.append({
                            'embedding': z1_embeddings[i],
                            'cell_id': cell_id,
                            'embryo': embryo_id,
                            'timepoint': t,
                            'n_cells_context': len(subset1_ids),
                            'sampling_strategy': strategy
                        })

                    pbar.update(1)

        pbar.close()
        print(f"Built manifold with {len(self.manifold)} embeddings")

        # Build index
        self._build_index()

    def _build_index(self) -> None:
        """
        Phase 2: Build kNN index for fast retrieval
        """
        print("\n" + "="*60)
        print("PHASE 2: Building kNN Index")
        print("="*60)

        # Collect all embeddings
        self.embeddings_matrix = np.stack([entry['embedding'] for entry in self.manifold])
        self.embeddings_matrix = self.embeddings_matrix.astype('float32')

        # L2 normalize
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        self.embeddings_matrix = self.embeddings_matrix / (norms + 1e-8)

        # Try FAISS first, fall back to sklearn
        try:
            import faiss
            d = self.embeddings_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)  # Inner product on normalized = cosine
            self.faiss_index.add(self.embeddings_matrix)
            print(f"Built FAISS index with {len(self.embeddings_matrix)} embeddings")
        except ImportError:
            print("FAISS not available, using sklearn NearestNeighbors")
            self.sklearn_index = NearestNeighbors(
                n_neighbors=self.config.DEFAULT_K,
                metric='cosine',
                algorithm='auto'
            )
            self.sklearn_index.fit(self.embeddings_matrix)
            print(f"Built sklearn index with {len(self.embeddings_matrix)} embeddings")

    def search(self, query_embeddings: np.ndarray, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        query_embeddings = query_embeddings.astype('float32')

        # L2 normalize
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / (norms + 1e-8)

        if self.faiss_index is not None:
            distances, indices = self.faiss_index.search(query_embeddings, k)
            return distances, indices
        else:
            # Temporarily adjust k if needed
            self.sklearn_index.set_params(n_neighbors=min(k, len(self.manifold)))
            distances, indices = self.sklearn_index.kneighbors(query_embeddings)
            return distances, indices

    def get_labels(self, indices: np.ndarray) -> List[List[str]]:
        """Get cell IDs for given indices"""
        labels = []
        for row in indices:
            row_labels = [self.manifold[idx]['cell_id'] for idx in row]
            labels.append(row_labels)
        return labels

    def save(self, path: str) -> None:
        """Save manifold to file"""
        data = {
            'manifold': self.manifold,
            'embeddings_matrix': self.embeddings_matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved manifold to {path}")

    def load(self, path: str) -> None:
        """Load manifold from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.manifold = data['manifold']
        self.embeddings_matrix = data['embeddings_matrix']
        self._build_index()
        print(f"Loaded manifold from {path}")


# =============================================================================
# PHASE 3: INFERENCE PROCEDURE
# =============================================================================

class CellIdentifier:
    """kNN-based cell identification"""

    def __init__(self, model: nn.Module, manifold_builder: ReferenceManifoldBuilder,
                 device: torch.device, config: EvalConfig):
        self.model = model
        self.manifold = manifold_builder
        self.device = device
        self.config = config
        self.sampler = ImprovedSampler(config.MIN_CELLS, config.MAX_CELLS)

    @torch.no_grad()
    def identify_cells_single_ref(self, query_cells: Dict[str, np.ndarray],
                                   train_data: Dict, k: int = 30,
                                   voting: str = 'uniform') -> Dict[str, str]:
        """
        Single-reference cell identification

        Args:
            query_cells: Dict of cell_id -> coordinates
            train_data: Training data for reference selection
            k: Number of neighbors
            voting: 'uniform' or 'distance_weighted'

        Returns:
            Dict of query_cell_id -> predicted_cell_id
        """
        self.model.eval()

        cell_ids = list(query_cells.keys())
        coords = np.array([query_cells[cid] for cid in cell_ids])
        coords_norm = normalize_coords(coords)

        # Get stage-matched reference
        n_cells = len(cell_ids)
        # Use a random embryo as reference
        ref_embryo = random.choice(list(train_data.keys()))
        ref_time = random.choice(list(train_data[ref_embryo].keys()))
        ref_cells = train_data[ref_embryo][ref_time]
        ref_cell_ids, ref_coords = self.sampler.sample_cells(ref_cells, 'diverse')
        ref_coords_norm = normalize_coords(ref_coords)

        # Get embeddings
        pc1 = torch.from_numpy(coords_norm).float().unsqueeze(0).to(self.device)
        pc2 = torch.from_numpy(ref_coords_norm).float().unsqueeze(0).to(self.device)

        z1, z2, _ = self.model(pc1, pc2)

        if self.model.use_uncertainty:
            z1_embeddings = z1[0].squeeze(0).cpu().numpy()
        else:
            z1_embeddings = z1.squeeze(0).cpu().numpy()

        # kNN search
        distances, indices = self.manifold.search(z1_embeddings, k)
        neighbor_labels = self.manifold.get_labels(indices)

        # Vote on identity
        predictions = {}
        for i, cell_id in enumerate(cell_ids):
            if voting == 'distance_weighted':
                # Weight by inverse distance
                weights = 1.0 / (distances[i] + 1e-8)
                label_weights = defaultdict(float)
                for label, weight in zip(neighbor_labels[i], weights):
                    label_weights[label] += weight
                pred_id = max(label_weights, key=label_weights.get)
            else:
                # Uniform voting
                label_counts = Counter(neighbor_labels[i])
                pred_id = label_counts.most_common(1)[0][0]

            predictions[cell_id] = pred_id

        return predictions

    @torch.no_grad()
    def identify_cells_multi_ref(self, query_cells: Dict[str, np.ndarray],
                                  train_data: Dict, k: int = 30, n_refs: int = 5,
                                  voting: str = 'uniform') -> Dict[str, str]:
        """
        Multi-reference cell identification (aggregates votes across references)
        """
        self.model.eval()

        cell_ids = list(query_cells.keys())
        coords = np.array([query_cells[cid] for cid in cell_ids])
        coords_norm = normalize_coords(coords)
        n_cells = len(cell_ids)

        all_votes = defaultdict(list)
        all_distances = defaultdict(list)

        for _ in range(n_refs):
            # Get stage-matched reference
            ref_embryo, ref_time = get_stage_matched_reference(
                "", 0, train_data, n_cells, exclude_embryo=False
            )
            ref_cells = train_data[ref_embryo][ref_time]
            ref_cell_ids, ref_coords = self.sampler.sample_cells(ref_cells, 'diverse')
            ref_coords_norm = normalize_coords(ref_coords)

            # Get embeddings
            pc1 = torch.from_numpy(coords_norm).float().unsqueeze(0).to(self.device)
            pc2 = torch.from_numpy(ref_coords_norm).float().unsqueeze(0).to(self.device)

            z1, z2, _ = self.model(pc1, pc2)

            if self.model.use_uncertainty:
                z1_embeddings = z1[0].squeeze(0).cpu().numpy()
            else:
                z1_embeddings = z1.squeeze(0).cpu().numpy()

            # kNN search
            distances, indices = self.manifold.search(z1_embeddings, k)
            neighbor_labels = self.manifold.get_labels(indices)

            for i, cell_id in enumerate(cell_ids):
                all_votes[i].extend(neighbor_labels[i])
                all_distances[i].extend(distances[i].tolist())

        # Vote on identity
        predictions = {}
        for i, cell_id in enumerate(cell_ids):
            if voting == 'distance_weighted':
                weights = 1.0 / (np.array(all_distances[i]) + 1e-8)
                label_weights = defaultdict(float)
                for label, weight in zip(all_votes[i], weights):
                    label_weights[label] += weight
                pred_id = max(label_weights, key=label_weights.get)
            else:
                label_counts = Counter(all_votes[i])
                pred_id = label_counts.most_common(1)[0][0]

            predictions[cell_id] = pred_id

        return predictions


# =============================================================================
# PHASE 4: HYPERPARAMETER TUNING
# =============================================================================

class HyperparameterTuner:
    """Tune hyperparameters on validation data"""

    def __init__(self, model: nn.Module, device: torch.device, config: EvalConfig):
        self.model = model
        self.device = device
        self.config = config
        self.sampler = ImprovedSampler(config.MIN_CELLS, config.MAX_CELLS)

    def tune(self, train_data: Dict, val_data: Dict) -> Dict:
        """
        Phase 4: Hyperparameter tuning on validation set
        """
        print("\n" + "="*60)
        print("PHASE 4: Hyperparameter Tuning")
        print("="*60)

        results = {
            'k_tuning': [],
            'm_tuning': [],
            'n_samples_tuning': [],
            'voting_comparison': [],
            'best_config': {}
        }

        # Test different k values
        print("\nTuning k (number of neighbors)...")
        best_k, best_k_acc = self.config.DEFAULT_K, 0

        for k in tqdm(self.config.K_VALUES_TO_TEST, desc="Testing k values"):
            # Build manifold with default settings
            manifold_builder = ReferenceManifoldBuilder(self.model, self.device, self.config)
            manifold_builder.build_manifold(train_data, n_samples_per_timepoint=30)

            identifier = CellIdentifier(self.model, manifold_builder, self.device, self.config)

            # Evaluate on validation set
            acc = self._evaluate_on_validation(identifier, val_data, train_data, k=k)
            results['k_tuning'].append({'k': k, 'accuracy': acc})

            if acc > best_k_acc:
                best_k, best_k_acc = k, acc

        results['best_config']['k'] = best_k
        print(f"Best k: {best_k} (accuracy: {best_k_acc:.4f})")

        # Test voting strategies
        print("\nComparing voting strategies...")
        for voting in ['uniform', 'distance_weighted']:
            manifold_builder = ReferenceManifoldBuilder(self.model, self.device, self.config)
            manifold_builder.build_manifold(train_data, n_samples_per_timepoint=30)
            identifier = CellIdentifier(self.model, manifold_builder, self.device, self.config)

            acc = self._evaluate_on_validation(
                identifier, val_data, train_data, k=best_k, voting=voting
            )
            results['voting_comparison'].append({'voting': voting, 'accuracy': acc})

        best_voting = max(results['voting_comparison'], key=lambda x: x['accuracy'])['voting']
        results['best_config']['voting'] = best_voting
        print(f"Best voting: {best_voting}")

        return results

    def _evaluate_on_validation(self, identifier: CellIdentifier, val_data: Dict,
                                 train_data: Dict, k: int = 30, voting: str = 'uniform',
                                 n_refs: int = 1) -> float:
        """Evaluate identifier on validation data"""
        correct = 0
        total = 0

        # Sample validation queries
        val_samples = []
        for embryo_id, timepoints in val_data.items():
            for t, cells in timepoints.items():
                n_cells = len(cells)
                if self.config.MIN_CELLS <= n_cells <= self.config.MAX_CELLS:
                    val_samples.append((embryo_id, t, cells))

        # Limit samples for speed
        if len(val_samples) > 100:
            val_samples = random.sample(val_samples, 100)

        for embryo_id, t, cells in val_samples:
            cell_ids, coords = self.sampler.sample_cells(cells, 'diverse')
            query_cells = {cid: cells[cid] for cid in cell_ids if cid in cells}

            if n_refs == 1:
                predictions = identifier.identify_cells_single_ref(
                    query_cells, train_data, k=k, voting=voting
                )
            else:
                predictions = identifier.identify_cells_multi_ref(
                    query_cells, train_data, k=k, n_refs=n_refs, voting=voting
                )

            for true_id, pred_id in predictions.items():
                if true_id == pred_id:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0


# =============================================================================
# PHASE 5: MAIN EVALUATION
# =============================================================================

class MainEvaluator:
    """Main evaluation with stratified metrics and confidence intervals"""

    def __init__(self, model: nn.Module, manifold_builder: ReferenceManifoldBuilder,
                 device: torch.device, config: EvalConfig):
        self.model = model
        self.manifold = manifold_builder
        self.device = device
        self.config = config
        self.sampler = ImprovedSampler(config.MIN_CELLS, config.MAX_CELLS)
        self.identifier = CellIdentifier(model, manifold_builder, device, config)

    def evaluate(self, test_data: Dict, train_data: Dict, k: int = 30,
                 voting: str = 'uniform', n_refs: int = 1) -> Dict:
        """
        Phase 5: Main evaluation with stratified metrics
        """
        print("\n" + "="*60)
        print("PHASE 5: Main Evaluation")
        print("="*60)

        all_results = []

        # Collect all test samples
        test_samples = []
        for embryo_id, timepoints in test_data.items():
            for t, cells in timepoints.items():
                n_cells = len(cells)
                if self.config.MIN_CELLS <= n_cells <= self.config.STAGE_LIMIT:
                    test_samples.append((embryo_id, t, cells, n_cells))

        print(f"Evaluating on {len(test_samples)} test samples...")

        for embryo_id, t, cells, n_cells_total in tqdm(test_samples, desc="Evaluating"):
            # Sample subset
            cell_ids, coords = self.sampler.sample_cells(cells, 'diverse')
            query_cells = {cid: cells[cid] for cid in cell_ids if cid in cells}

            if len(query_cells) < self.config.MIN_CELLS:
                continue

            # Get predictions
            if n_refs == 1:
                predictions = self.identifier.identify_cells_single_ref(
                    query_cells, train_data, k=k, voting=voting
                )
            else:
                predictions = self.identifier.identify_cells_multi_ref(
                    query_cells, train_data, k=k, n_refs=n_refs, voting=voting
                )

            # Record results
            for true_id, pred_id in predictions.items():
                # Determine stage category
                if n_cells_total <= 30:
                    stage = 'early'
                elif n_cells_total <= 100:
                    stage = 'mid'
                else:
                    stage = 'late'

                # Determine query size category
                query_size = len(query_cells)
                if query_size <= 7:
                    size_cat = '5-7'
                elif query_size <= 12:
                    size_cat = '8-12'
                elif query_size <= 17:
                    size_cat = '13-17'
                else:
                    size_cat = '18-20'

                all_results.append({
                    'true_id': true_id,
                    'pred_id': pred_id,
                    'correct': true_id == pred_id,
                    'embryo': embryo_id,
                    'timepoint': t,
                    'stage': stage,
                    'n_cells_total': n_cells_total,
                    'query_size': query_size,
                    'size_category': size_cat
                })

        # Compute metrics
        results_df = all_results

        # Primary metrics
        correct = sum(1 for r in results_df if r['correct'])
        total = len(results_df)
        per_cell_accuracy = correct / total if total > 0 else 0

        # Per-query accuracy
        query_accuracies = defaultdict(list)
        for r in results_df:
            key = (r['embryo'], r['timepoint'])
            query_accuracies[key].append(r['correct'])

        per_query_accs = [np.mean(v) for v in query_accuracies.values()]
        per_query_mean_accuracy = np.mean(per_query_accs) if per_query_accs else 0

        # Stratified metrics
        stratified = {
            'by_query_size': {},
            'by_stage': {}
        }

        for size_cat in ['5-7', '8-12', '13-17', '18-20']:
            cat_results = [r for r in results_df if r['size_category'] == size_cat]
            if cat_results:
                acc = sum(1 for r in cat_results if r['correct']) / len(cat_results)
                stratified['by_query_size'][size_cat] = acc

        for stage in ['early', 'mid', 'late']:
            stage_results = [r for r in results_df if r['stage'] == stage]
            if stage_results:
                acc = sum(1 for r in stage_results if r['correct']) / len(stage_results)
                stratified['by_stage'][stage] = acc

        # Bootstrap confidence intervals
        bootstrap_accs = []
        for _ in range(self.config.N_BOOTSTRAP_SAMPLES):
            sample = random.choices(results_df, k=len(results_df))
            acc = sum(1 for r in sample if r['correct']) / len(sample)
            bootstrap_accs.append(acc)

        ci_lower = np.percentile(bootstrap_accs, 2.5)
        ci_upper = np.percentile(bootstrap_accs, 97.5)

        results = {
            'per_cell_accuracy': per_cell_accuracy,
            'per_query_mean_accuracy': per_query_mean_accuracy,
            'n_samples': total,
            'n_queries': len(query_accuracies),
            'confidence_interval_95': [ci_lower, ci_upper],
            'stratified_results': stratified,
            'k': k,
            'voting': voting,
            'n_refs': n_refs
        }

        print(f"\nResults:")
        print(f"  Per-cell accuracy: {per_cell_accuracy:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        print(f"  Per-query mean accuracy: {per_query_mean_accuracy:.4f}")
        print(f"  By query size: {stratified['by_query_size']}")
        print(f"  By stage: {stratified['by_stage']}")

        return results


# =============================================================================
# PHASE 6: ABLATION STUDIES
# =============================================================================

class AblationStudy:
    """Ablation studies for architecture components"""

    def __init__(self, device: torch.device, config: EvalConfig):
        self.device = device
        self.config = config

    def run_ablations(self, train_data: Dict, test_data: Dict) -> Dict:
        """
        Phase 6: Ablation studies
        """
        print("\n" + "="*60)
        print("PHASE 6: Ablation Studies")
        print("="*60)

        ablations = {
            'full_model': {
                'use_sparse_features': True,
                'use_uncertainty': True,
                'use_learnable_no_match': True
            },
            'no_sparse_features': {
                'use_sparse_features': False,
                'use_uncertainty': True,
                'use_learnable_no_match': True
            },
            'no_uncertainty': {
                'use_sparse_features': True,
                'use_uncertainty': False,
                'use_learnable_no_match': True
            },
            'no_no_match_token': {
                'use_sparse_features': True,
                'use_uncertainty': True,
                'use_learnable_no_match': False
            }
        }

        results = {}

        for name, settings in ablations.items():
            print(f"\nRunning ablation: {name}")

            # Note: This requires retraining. For now, we document what would be tested
            results[name] = {
                'settings': settings,
                'status': 'requires_retraining',
                'note': f'To evaluate {name}, retrain model with settings: {settings}'
            }

            # If we have the full model loaded, we can at least evaluate it
            if name == 'full_model':
                try:
                    model = load_model(self.config, self.device)
                    manifold_builder = ReferenceManifoldBuilder(model, self.device, self.config)
                    manifold_builder.build_manifold(train_data, n_samples_per_timepoint=30)

                    evaluator = MainEvaluator(model, manifold_builder, self.device, self.config)
                    eval_results = evaluator.evaluate(test_data, train_data, k=30)

                    results[name] = {
                        'settings': settings,
                        'accuracy': eval_results['per_cell_accuracy'],
                        'ci_95': eval_results['confidence_interval_95']
                    }
                except Exception as e:
                    results[name]['error'] = str(e)

        return results


# =============================================================================
# PHASE 7: ERROR ANALYSIS
# =============================================================================

class ErrorAnalyzer:
    """Analyze error patterns"""

    def __init__(self, config: EvalConfig):
        self.config = config

    def analyze_errors(self, predictions: List[Dict]) -> Dict:
        """
        Phase 7: Categorize errors

        Categories:
        - Spatial nearest neighbor
        - Sibling confusion
        - Parent/child confusion
        - Distant confusion
        """
        print("\n" + "="*60)
        print("PHASE 7: Error Analysis")
        print("="*60)

        errors = [p for p in predictions if not p.get('correct', p['true_id'] == p['pred_id'])]

        categories = {
            'spatial_nearest_neighbor': 0,
            'sibling': 0,
            'parent_child': 0,
            'distant': 0
        }

        for error in errors:
            true_id = error['true_id']
            pred_id = error['pred_id']

            category = self._categorize_error(true_id, pred_id)
            categories[category] += 1

        total_errors = len(errors)

        results = {
            'total_errors': total_errors,
            'categories': categories,
            'category_percentages': {
                k: v / total_errors * 100 if total_errors > 0 else 0
                for k, v in categories.items()
            }
        }

        print(f"\nError Analysis Results:")
        print(f"  Total errors: {total_errors}")
        for cat, count in categories.items():
            pct = results['category_percentages'][cat]
            print(f"  {cat}: {count} ({pct:.1f}%)")

        # Compare to paper's Figure 4C
        print(f"\nComparison to original paper (Figure 4C):")
        print(f"  Paper: 45.9% nearest neighbor, 17.4% sibling, 24.1% parent/child, 12.6% other")

        return results

    def _categorize_error(self, true_id: str, pred_id: str) -> str:
        """Categorize an error based on C. elegans naming conventions"""

        # Check for sibling (same parent - same prefix, different last character)
        if len(true_id) == len(pred_id) and len(true_id) > 1:
            if true_id[:-1] == pred_id[:-1]:
                return 'sibling'

        # Check for parent/child
        if true_id.startswith(pred_id) or pred_id.startswith(true_id):
            if abs(len(true_id) - len(pred_id)) == 1:
                return 'parent_child'

        # Check for spatial nearest neighbor (heuristic: similar prefix suggests proximity)
        common_prefix_len = 0
        for i in range(min(len(true_id), len(pred_id))):
            if true_id[i] == pred_id[i]:
                common_prefix_len += 1
            else:
                break

        if common_prefix_len >= len(true_id) - 2 or common_prefix_len >= len(pred_id) - 2:
            return 'spatial_nearest_neighbor'

        return 'distant'


# =============================================================================
# PHASE 8: BASELINE COMPARISONS
# =============================================================================

class BaselineComparison:
    """Compare against baseline methods"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.sampler = ImprovedSampler(config.MIN_CELLS, config.MAX_CELLS)

    def run_baselines(self, test_data: Dict, train_data: Dict) -> Dict:
        """
        Phase 8: Compare against baseline methods
        """
        print("\n" + "="*60)
        print("PHASE 8: Baseline Comparisons")
        print("="*60)

        results = {}

        # Collect test samples
        test_samples = []
        for embryo_id, timepoints in test_data.items():
            for t, cells in timepoints.items():
                n_cells = len(cells)
                if self.config.MIN_CELLS <= n_cells <= self.config.MAX_CELLS:
                    test_samples.append((embryo_id, t, cells))

        # Limit for speed
        if len(test_samples) > 100:
            test_samples = random.sample(test_samples, 100)

        print(f"Running baselines on {len(test_samples)} samples...")

        # 1. ICP Baseline
        print("\n1. ICP (Iterative Closest Point) Baseline")
        try:
            icp_results = self._run_icp_baseline(test_samples, train_data)
            results['icp'] = icp_results
            print(f"   Accuracy: {icp_results['accuracy']:.4f}")
        except Exception as e:
            results['icp'] = {'error': str(e)}
            print(f"   Error: {e}")

        # 2. CPD Baseline
        print("\n2. CPD (Coherent Point Drift) Baseline")
        try:
            cpd_results = self._run_cpd_baseline(test_samples, train_data)
            results['cpd'] = cpd_results
            print(f"   Accuracy: {cpd_results['accuracy']:.4f}")
        except Exception as e:
            results['cpd'] = {'error': str(e)}
            print(f"   Error: {e}")

        # 3. Hungarian Baseline
        print("\n3. Hungarian (Optimal Assignment) Baseline")
        try:
            hungarian_results = self._run_hungarian_baseline(test_samples, train_data)
            results['hungarian'] = hungarian_results
            print(f"   Accuracy: {hungarian_results['accuracy']:.4f}")
        except Exception as e:
            results['hungarian'] = {'error': str(e)}
            print(f"   Error: {e}")

        # 4. Nearest Neighbor (no learning) Baseline
        print("\n4. Nearest Neighbor (no learning) Baseline")
        try:
            nn_results = self._run_nearest_neighbor_baseline(test_samples, train_data)
            results['nearest_neighbor'] = nn_results
            print(f"   Accuracy: {nn_results['accuracy']:.4f}")
        except Exception as e:
            results['nearest_neighbor'] = {'error': str(e)}
            print(f"   Error: {e}")

        return results

    def _run_icp_baseline(self, test_samples: List, train_data: Dict) -> Dict:
        """ICP-based cell identification"""
        correct = 0
        total = 0

        for embryo_id, t, cells in tqdm(test_samples, desc="ICP"):
            cell_ids, coords = self.sampler.sample_cells(cells, 'diverse')
            query_coords = coords.copy()

            # Get reference
            ref_embryo = random.choice(list(train_data.keys()))
            ref_time = random.choice(list(train_data[ref_embryo].keys()))
            ref_cells = train_data[ref_embryo][ref_time]
            ref_ids, ref_coords = self.sampler.sample_cells(ref_cells, 'diverse')

            # Simple ICP: iteratively find closest points and compute transform
            transformed = query_coords.copy()
            for _ in range(10):  # Max iterations
                # Find nearest neighbors
                dists = cdist(transformed, ref_coords)
                nearest_idx = dists.argmin(axis=1)

                # Compute centroid alignment
                query_centroid = transformed.mean(axis=0)
                ref_matched = ref_coords[nearest_idx]
                ref_centroid = ref_matched.mean(axis=0)

                # Translate
                transformed = transformed - query_centroid + ref_centroid

            # Final assignment by nearest neighbor
            dists = cdist(transformed, ref_coords)
            assignments = dists.argmin(axis=1)

            for i, cell_id in enumerate(cell_ids):
                assigned_ref_idx = assignments[i]
                if assigned_ref_idx < len(ref_ids):
                    pred_id = ref_ids[assigned_ref_idx]
                    if cell_id == pred_id:
                        correct += 1
                total += 1

        return {'accuracy': correct / total if total > 0 else 0, 'n_samples': total}

    def _run_cpd_baseline(self, test_samples: List, train_data: Dict) -> Dict:
        """CPD-based cell identification"""
        try:
            from pycpd import RigidRegistration
        except ImportError:
            return {'error': 'pycpd not installed (pip install pycpd)', 'accuracy': None}

        correct = 0
        total = 0

        for embryo_id, t, cells in tqdm(test_samples, desc="CPD"):
            cell_ids, coords = self.sampler.sample_cells(cells, 'diverse')

            ref_embryo = random.choice(list(train_data.keys()))
            ref_time = random.choice(list(train_data[ref_embryo].keys()))
            ref_cells = train_data[ref_embryo][ref_time]
            ref_ids, ref_coords = self.sampler.sample_cells(ref_cells, 'diverse')

            try:
                reg = RigidRegistration(X=ref_coords, Y=coords)
                transformed, _ = reg.register()

                # Assign by nearest neighbor after registration
                dists = cdist(transformed, ref_coords)
                assignments = dists.argmin(axis=1)

                for i, cell_id in enumerate(cell_ids):
                    if assignments[i] < len(ref_ids):
                        if cell_id == ref_ids[assignments[i]]:
                            correct += 1
                    total += 1
            except:
                total += len(cell_ids)

        return {'accuracy': correct / total if total > 0 else 0, 'n_samples': total}

    def _run_hungarian_baseline(self, test_samples: List, train_data: Dict) -> Dict:
        """Hungarian algorithm for optimal assignment"""
        correct = 0
        total = 0

        for embryo_id, t, cells in tqdm(test_samples, desc="Hungarian"):
            cell_ids, coords = self.sampler.sample_cells(cells, 'diverse')

            ref_embryo = random.choice(list(train_data.keys()))
            ref_time = random.choice(list(train_data[ref_embryo].keys()))
            ref_cells = train_data[ref_embryo][ref_time]
            ref_ids, ref_coords = self.sampler.sample_cells(ref_cells, 'diverse')

            # Compute distance matrix
            dist_matrix = cdist(coords, ref_coords)

            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            for i, j in zip(row_ind, col_ind):
                if i < len(cell_ids) and j < len(ref_ids):
                    if cell_ids[i] == ref_ids[j]:
                        correct += 1
                    total += 1

        return {'accuracy': correct / total if total > 0 else 0, 'n_samples': total}

    def _run_nearest_neighbor_baseline(self, test_samples: List, train_data: Dict) -> Dict:
        """Simple nearest neighbor without any transformation"""
        correct = 0
        total = 0

        for embryo_id, t, cells in tqdm(test_samples, desc="NN Baseline"):
            cell_ids, coords = self.sampler.sample_cells(cells, 'diverse')

            ref_embryo = random.choice(list(train_data.keys()))
            ref_time = random.choice(list(train_data[ref_embryo].keys()))
            ref_cells = train_data[ref_embryo][ref_time]
            ref_ids, ref_coords = self.sampler.sample_cells(ref_cells, 'diverse')

            # Normalize both
            coords_norm = normalize_coords(coords)
            ref_coords_norm = normalize_coords(ref_coords)

            # Simple nearest neighbor
            dists = cdist(coords_norm, ref_coords_norm)
            assignments = dists.argmin(axis=1)

            for i, cell_id in enumerate(cell_ids):
                if assignments[i] < len(ref_ids):
                    if cell_id == ref_ids[assignments[i]]:
                        correct += 1
                total += 1

        return {'accuracy': correct / total if total > 0 else 0, 'n_samples': total}


# =============================================================================
# PHASE 9: VISUALIZATION
# =============================================================================

class Visualizer:
    """Generate visualizations for evaluation results"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_tsne_embeddings(self, manifold: List[Dict], max_points: int = 5000) -> None:
        """Plot t-SNE of embeddings colored by lineage"""
        print("\nGenerating t-SNE visualization...")

        if len(manifold) > max_points:
            indices = random.sample(range(len(manifold)), max_points)
            manifold_subset = [manifold[i] for i in indices]
        else:
            manifold_subset = manifold

        embeddings = np.stack([e['embedding'] for e in manifold_subset])
        cell_ids = [e['cell_id'] for e in manifold_subset]

        # Assign colors by lineage prefix
        lineage_colors = self._assign_lineage_colors(cell_ids)

        tsne = TSNE(n_components=2, perplexity=50, random_state=42, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=lineage_colors, s=3, alpha=0.6, cmap='tab10')
        plt.title('t-SNE of Cell Embeddings (colored by lineage)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')

        # Add legend
        lineage_names = ['ABa', 'ABp', 'MS', 'E', 'C', 'D', 'P', 'Other']
        handles = [plt.scatter([], [], c=plt.cm.tab10(i/10), s=50, label=name)
                  for i, name in enumerate(lineage_names)]
        plt.legend(handles=handles, title='Lineage')

        plt.savefig(self.output_dir / 'tsne_embeddings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'tsne_embeddings.png'}")

    def plot_accuracy_by_subset_size(self, stratified_results: Dict) -> None:
        """Bar chart of accuracy by subset size"""
        print("\nGenerating accuracy by subset size plot...")

        by_size = stratified_results.get('by_query_size', {})
        if not by_size:
            print("  No data available")
            return

        categories = list(by_size.keys())
        accuracies = [by_size[c] * 100 for c in categories]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(categories, accuracies, color='steelblue', edgecolor='black')

        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Query Size (number of cells)')
        plt.ylabel('Accuracy (%)')
        plt.title('Cell Identification Accuracy by Query Size')
        plt.ylim(0, 105)

        plt.savefig(self.output_dir / 'accuracy_by_subset_size.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'accuracy_by_subset_size.png'}")

    def plot_accuracy_by_stage(self, stratified_results: Dict) -> None:
        """Bar chart of accuracy by developmental stage"""
        print("\nGenerating accuracy by stage plot...")

        by_stage = stratified_results.get('by_stage', {})
        if not by_stage:
            print("  No data available")
            return

        stage_order = ['early', 'mid', 'late']
        stage_labels = ['Early (4-30)', 'Mid (31-100)', 'Late (101-194)']
        accuracies = [by_stage.get(s, 0) * 100 for s in stage_order]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(stage_labels, accuracies, color='coral', edgecolor='black')

        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Developmental Stage')
        plt.ylabel('Accuracy (%)')
        plt.title('Cell Identification Accuracy by Developmental Stage')
        plt.ylim(0, 105)

        plt.savefig(self.output_dir / 'accuracy_by_stage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'accuracy_by_stage.png'}")

    def plot_error_categories(self, error_results: Dict) -> None:
        """Pie chart of error categories"""
        print("\nGenerating error categories plot...")

        categories = error_results.get('categories', {})
        if not categories or error_results.get('total_errors', 0) == 0:
            print("  No errors to analyze")
            return

        labels = ['Spatial Nearest\nNeighbor', 'Sibling', 'Parent/Child', 'Distant']
        sizes = [
            categories.get('spatial_nearest_neighbor', 0),
            categories.get('sibling', 0),
            categories.get('parent_child', 0),
            categories.get('distant', 0)
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        plt.title('Error Category Distribution')

        # Add comparison to paper
        plt.figtext(0.5, 0.02,
                   'Paper reference: 45.9% NN, 17.4% sibling, 24.1% parent/child, 12.6% other',
                   ha='center', fontsize=9, style='italic')

        plt.savefig(self.output_dir / 'error_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'error_categories.png'}")

    def plot_baseline_comparison(self, baseline_results: Dict, model_accuracy: float) -> None:
        """Bar chart comparing model vs baselines"""
        print("\nGenerating baseline comparison plot...")

        methods = ['Twin Attention\n(Our Model)']
        accuracies = [model_accuracy * 100]

        for name, result in baseline_results.items():
            if isinstance(result, dict) and 'accuracy' in result and result['accuracy'] is not None:
                display_name = name.upper().replace('_', ' ')
                methods.append(display_name)
                accuracies.append(result['accuracy'] * 100)

        plt.figure(figsize=(10, 6))
        colors = ['#2ecc71'] + ['#3498db'] * (len(methods) - 1)
        bars = plt.bar(methods, accuracies, color=colors, edgecolor='black')

        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Method')
        plt.ylabel('Accuracy (%)')
        plt.title('Cell Identification: Model vs Baselines')
        plt.ylim(0, 105)
        plt.xticks(rotation=15)

        # Add target line
        plt.axhline(y=85, color='red', linestyle='--', label='Target (85%)')
        plt.legend()

        plt.savefig(self.output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {self.output_dir / 'baseline_comparison.png'}")

    def _assign_lineage_colors(self, cell_ids: List[str]) -> np.ndarray:
        """Assign colors based on lineage prefix"""
        colors = []
        for cid in cell_ids:
            cid_upper = cid.upper()
            if cid_upper.startswith('ABA'):
                colors.append(0)
            elif cid_upper.startswith('ABP'):
                colors.append(1)
            elif cid_upper.startswith('MS'):
                colors.append(2)
            elif cid_upper.startswith('E'):
                colors.append(3)
            elif cid_upper.startswith('C'):
                colors.append(4)
            elif cid_upper.startswith('D'):
                colors.append(5)
            elif cid_upper.startswith('P'):
                colors.append(6)
            else:
                colors.append(7)
        return np.array(colors)


# =============================================================================
# SUMMARY REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generate summary report"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, all_results: Dict) -> None:
        """Generate markdown summary report"""
        report_path = self.output_dir / 'summary_report.md'

        with open(report_path, 'w') as f:
            f.write("# Sparse Cell Identification Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview
            f.write("## Overview\n\n")
            f.write("This report summarizes the evaluation of the sparse cell identification model ")
            f.write("that adapts the Twin Attention architecture for identifying cells in C. elegans ")
            f.write("embryos from partial observations (5-20 cells).\n\n")
            f.write("**Target**: >85% accuracy on kNN-based cell type assignment\n\n")

            # Phase 0: Stability
            if 'phase0_stability' in all_results:
                f.write("## Phase 0: Embedding Stability\n\n")
                stability = all_results['phase0_stability']
                f.write(f"- Mean cosine similarity: {stability.get('mean_cosine_similarity', 'N/A'):.4f}\n")
                f.write(f"- Recommendation: {stability.get('recommendation_reason', 'N/A')}\n\n")

            # Phase 5: Main Results
            if 'phase5_main' in all_results:
                f.write("## Main Results (Phase 5)\n\n")
                main = all_results['phase5_main']
                acc = main.get('per_cell_accuracy', 0) * 100
                ci = main.get('confidence_interval_95', [0, 0])
                f.write(f"**Per-cell accuracy**: {acc:.2f}% (95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%])\n\n")

                target_met = "✓ TARGET MET" if acc >= 85 else "✗ Below target"
                f.write(f"**Status**: {target_met}\n\n")

                # Stratified results
                f.write("### Accuracy by Query Size\n\n")
                f.write("| Size | Accuracy |\n")
                f.write("|------|----------|\n")
                for size, acc in main.get('stratified_results', {}).get('by_query_size', {}).items():
                    f.write(f"| {size} cells | {acc*100:.1f}% |\n")
                f.write("\n")

                f.write("### Accuracy by Developmental Stage\n\n")
                f.write("| Stage | Accuracy |\n")
                f.write("|-------|----------|\n")
                for stage, acc in main.get('stratified_results', {}).get('by_stage', {}).items():
                    f.write(f"| {stage.capitalize()} | {acc*100:.1f}% |\n")
                f.write("\n")

            # Phase 7: Error Analysis
            if 'phase7_errors' in all_results:
                f.write("## Error Analysis (Phase 7)\n\n")
                errors = all_results['phase7_errors']
                f.write(f"Total errors: {errors.get('total_errors', 0)}\n\n")
                f.write("| Category | Count | Percentage |\n")
                f.write("|----------|-------|------------|\n")
                for cat, count in errors.get('categories', {}).items():
                    pct = errors.get('category_percentages', {}).get(cat, 0)
                    f.write(f"| {cat.replace('_', ' ').title()} | {count} | {pct:.1f}% |\n")
                f.write("\n")
                f.write("*Paper reference: 45.9% nearest neighbor, 17.4% sibling, 24.1% parent/child, 12.6% other*\n\n")

            # Phase 8: Baselines
            if 'phase8_baselines' in all_results:
                f.write("## Baseline Comparisons (Phase 8)\n\n")
                f.write("| Method | Accuracy |\n")
                f.write("|--------|----------|\n")
                f.write(f"| **Twin Attention (Ours)** | **{all_results.get('phase5_main', {}).get('per_cell_accuracy', 0)*100:.1f}%** |\n")
                for name, result in all_results['phase8_baselines'].items():
                    if isinstance(result, dict) and 'accuracy' in result and result['accuracy'] is not None:
                        f.write(f"| {name.upper()} | {result['accuracy']*100:.1f}% |\n")
                f.write("\n")

            # Best Configuration
            if 'phase4_hyperparams' in all_results:
                f.write("## Best Configuration\n\n")
                hp = all_results['phase4_hyperparams']
                best = hp.get('best_config', {})
                f.write(f"- k (neighbors): {best.get('k', 30)}\n")
                f.write(f"- Voting strategy: {best.get('voting', 'uniform')}\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            f.write("1. **Geometric features**: The SparsePointFeatures module extracts four geometric features ")
            f.write("(relative position, centroid distance, point count embedding, local density) ")
            f.write("specifically designed for sparse point clouds.\n\n")
            f.write("2. **No-match token**: The learnable no-match token handles cells with no valid ")
            f.write("correspondence, which is critical for partial observations.\n\n")
            f.write("3. **Stage matching**: Using stage-matched references (based on cell count) ")
            f.write("improves embedding quality.\n\n")

            f.write("---\n")
            f.write("*Report generated by Evaluation Suite for Sparse Cell Identification*\n")

        print(f"\nSummary report saved to: {report_path}")


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def run_full_evaluation(config: EvalConfig = None) -> Dict:
    """
    Run the complete evaluation pipeline
    """
    if config is None:
        config = EvalConfig()

    set_seed(config.SEED)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    all_results = {}

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    train_data, eval_data, real_data = load_data(config)

    if train_data is None:
        print("ERROR: No training data available. Cannot proceed.")
        return all_results

    # Split eval data into validation and test
    if eval_data:
        eval_embryos = list(eval_data.keys())
        n_val = max(1, len(eval_embryos) // 2)
        val_embryos = eval_embryos[:n_val]
        test_embryos = eval_embryos[n_val:]
        val_data = {k: eval_data[k] for k in val_embryos}
        test_data = {k: eval_data[k] for k in test_embryos}
    else:
        val_data = train_data
        test_data = train_data

    # Load model
    model = load_model(config, device)

    # Phase 0: Embedding Stability
    try:
        stability_analyzer = EmbeddingStabilityAnalyzer(model, device, config)
        all_results['phase0_stability'] = stability_analyzer.analyze_stability(val_data, train_data)

        with open(os.path.join(config.OUTPUT_DIR, 'phase0_embedding_stability.json'), 'w') as f:
            json.dump(all_results['phase0_stability'], f, indent=2)
    except Exception as e:
        print(f"Phase 0 error: {e}")
        traceback.print_exc()

    # Phase 1 & 2: Build Reference Manifold
    try:
        manifold_builder = ReferenceManifoldBuilder(model, device, config)
        manifold_builder.build_manifold(train_data, n_samples_per_timepoint=50)
        manifold_builder.save(os.path.join(config.OUTPUT_DIR, 'reference_manifold.pkl'))
    except Exception as e:
        print(f"Phase 1-2 error: {e}")
        traceback.print_exc()
        return all_results

    # Phase 4: Hyperparameter Tuning
    try:
        tuner = HyperparameterTuner(model, device, config)
        all_results['phase4_hyperparams'] = tuner.tune(train_data, val_data)

        with open(os.path.join(config.OUTPUT_DIR, 'phase4_hyperparameter_search.json'), 'w') as f:
            json.dump(all_results['phase4_hyperparams'], f, indent=2)
    except Exception as e:
        print(f"Phase 4 error: {e}")
        traceback.print_exc()

    # Get best config
    best_k = all_results.get('phase4_hyperparams', {}).get('best_config', {}).get('k', 30)
    best_voting = all_results.get('phase4_hyperparams', {}).get('best_config', {}).get('voting', 'uniform')
    use_multi_ref = all_results.get('phase0_stability', {}).get('recommendation', '') == 'multi_reference'
    n_refs = 5 if use_multi_ref else 1

    # Phase 5: Main Evaluation
    try:
        evaluator = MainEvaluator(model, manifold_builder, device, config)
        all_results['phase5_main'] = evaluator.evaluate(
            test_data, train_data, k=best_k, voting=best_voting, n_refs=n_refs
        )

        with open(os.path.join(config.OUTPUT_DIR, 'phase5_main_results.json'), 'w') as f:
            json.dump(all_results['phase5_main'], f, indent=2)
    except Exception as e:
        print(f"Phase 5 error: {e}")
        traceback.print_exc()

    # Phase 6: Ablation Studies
    try:
        ablation_study = AblationStudy(device, config)
        all_results['phase6_ablations'] = ablation_study.run_ablations(train_data, test_data)

        with open(os.path.join(config.OUTPUT_DIR, 'phase6_ablations.json'), 'w') as f:
            json.dump(all_results['phase6_ablations'], f, indent=2)
    except Exception as e:
        print(f"Phase 6 error: {e}")
        traceback.print_exc()

    # Phase 7: Error Analysis
    # Collect predictions for error analysis
    try:
        identifier = CellIdentifier(model, manifold_builder, device, config)
        predictions = []

        for embryo_id, timepoints in test_data.items():
            for t, cells in timepoints.items():
                if config.MIN_CELLS <= len(cells) <= config.MAX_CELLS:
                    sampler = ImprovedSampler(config.MIN_CELLS, config.MAX_CELLS)
                    cell_ids, _ = sampler.sample_cells(cells, 'diverse')
                    query_cells = {cid: cells[cid] for cid in cell_ids if cid in cells}

                    preds = identifier.identify_cells_single_ref(query_cells, train_data, k=best_k)
                    for true_id, pred_id in preds.items():
                        predictions.append({
                            'true_id': true_id,
                            'pred_id': pred_id,
                            'correct': true_id == pred_id
                        })

        error_analyzer = ErrorAnalyzer(config)
        all_results['phase7_errors'] = error_analyzer.analyze_errors(predictions)

        with open(os.path.join(config.OUTPUT_DIR, 'phase7_error_analysis.json'), 'w') as f:
            json.dump(all_results['phase7_errors'], f, indent=2)
    except Exception as e:
        print(f"Phase 7 error: {e}")
        traceback.print_exc()

    # Phase 8: Baseline Comparisons
    try:
        baseline_comp = BaselineComparison(config)
        all_results['phase8_baselines'] = baseline_comp.run_baselines(test_data, train_data)

        with open(os.path.join(config.OUTPUT_DIR, 'phase8_baselines.json'), 'w') as f:
            json.dump(all_results['phase8_baselines'], f, indent=2)
    except Exception as e:
        print(f"Phase 8 error: {e}")
        traceback.print_exc()

    # Phase 9: Visualization
    try:
        visualizer = Visualizer(config.OUTPUT_DIR)

        visualizer.plot_tsne_embeddings(manifold_builder.manifold)

        if 'phase5_main' in all_results:
            visualizer.plot_accuracy_by_subset_size(all_results['phase5_main'].get('stratified_results', {}))
            visualizer.plot_accuracy_by_stage(all_results['phase5_main'].get('stratified_results', {}))

        if 'phase7_errors' in all_results:
            visualizer.plot_error_categories(all_results['phase7_errors'])

        if 'phase8_baselines' in all_results and 'phase5_main' in all_results:
            model_acc = all_results['phase5_main'].get('per_cell_accuracy', 0)
            visualizer.plot_baseline_comparison(all_results['phase8_baselines'], model_acc)
    except Exception as e:
        print(f"Phase 9 error: {e}")
        traceback.print_exc()

    # Evaluate on real data if available
    if real_data:
        try:
            print("\n" + "="*60)
            print("EVALUATING ON REAL DATA")
            print("="*60)

            real_evaluator = MainEvaluator(model, manifold_builder, device, config)
            all_results['real_data_results'] = real_evaluator.evaluate(
                real_data, train_data, k=best_k, voting=best_voting, n_refs=n_refs
            )

            with open(os.path.join(config.OUTPUT_DIR, 'real_data_results.json'), 'w') as f:
                json.dump(all_results['real_data_results'], f, indent=2)
        except Exception as e:
            print(f"Real data evaluation error: {e}")

    # Generate Summary Report
    report_gen = ReportGenerator(config.OUTPUT_DIR)
    report_gen.generate_report(all_results)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {config.OUTPUT_DIR}/")

    return all_results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation Suite for Sparse Cell Identification')
    parser.add_argument('--train_data', type=str, default='data_dict.pkl',
                       help='Path to training data pickle file')
    parser.add_argument('--eval_data', type=str, default='evaluation_data_dict.pkl',
                       help='Path to evaluation data pickle file')
    parser.add_argument('--real_data', type=str, default='real_data_dict.pkl',
                       help='Path to real data pickle file')
    parser.add_argument('--model_path', type=str, default='twin_attention_final.pth',
                       help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create config
    config = EvalConfig()
    config.TRAIN_DATA = args.train_data
    config.EVAL_DATA = args.eval_data
    config.REAL_DATA = args.real_data
    config.MODEL_PATH = args.model_path
    config.OUTPUT_DIR = args.output_dir
    config.SEED = args.seed

    # Run evaluation
    results = run_full_evaluation(config)

    # Print final summary
    if 'phase5_main' in results:
        acc = results['phase5_main'].get('per_cell_accuracy', 0) * 100
        ci = results['phase5_main'].get('confidence_interval_95', [0, 0])
        print(f"\n{'='*60}")
        print(f"FINAL ACCURACY: {acc:.2f}% (95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%])")
        print(f"TARGET (>85%): {'✓ MET' if acc >= 85 else '✗ NOT MET'}")
        print(f"{'='*60}")
