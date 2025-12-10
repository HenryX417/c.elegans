"""
Comprehensive Evaluation Suite for Twin Attention Cell Identification Model
Calculates all metrics from the paper's Results section (3.1-3.6)
"""

import os
import sys
import pickle
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
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
print(f"Using device: {device}")


# =============================================================================
# CONFIGURATION
# =============================================================================
class EvalConfig:
    """Configuration for evaluation paths and parameters"""
    # Default paths - Windows paths converted for cross-platform
    TRAIN_DATA_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\data_dict.pkl"
    EVAL_DATA_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\evaluation_data_dict.pkl"
    MODEL_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\twin_attention_final.pth"
    REAL_EMBRYO_PATH = None  # Set when available

    # Output directory
    OUTPUT_DIR = "evaluation_results"
    FIGURE_DIR = "evaluation_figures"

    # Model config
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 6
    DROPOUT = 0.1

    # Evaluation params
    BATCH_SIZE = 16
    MIN_CELLS = 5
    MAX_CELLS = 20
    STAGE_LIMIT = 194
    K_NEIGHBORS = 30

    # Bootstrap params
    N_BOOTSTRAP = 1000
    CONFIDENCE_LEVEL = 0.95


def convert_path(path):
    """Convert Windows path to current OS path"""
    if path is None:
        return None
    # Handle Windows paths on Linux
    if os.name != 'nt' and '\\' in path:
        # Extract just the filename for now
        return os.path.basename(path.replace('\\', '/'))
    return path


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(config: EvalConfig):
    """Load training and evaluation data"""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    data = {}

    # Try to load training data
    train_path = convert_path(config.TRAIN_DATA_PATH)
    if os.path.exists(train_path):
        print(f"Loading training data from: {train_path}")
        with open(train_path, 'rb') as f:
            data['train'] = pickle.load(f)
        print(f"  Loaded {len(data['train'])} embryos")
    else:
        print(f"  Training data not found at: {train_path}")
        data['train'] = None

    # Try to load evaluation data
    eval_path = convert_path(config.EVAL_DATA_PATH)
    if os.path.exists(eval_path):
        print(f"Loading evaluation data from: {eval_path}")
        with open(eval_path, 'rb') as f:
            data['eval'] = pickle.load(f)
        print(f"  Loaded {len(data['eval'])} embryos")
    else:
        print(f"  Evaluation data not found at: {eval_path}")
        data['eval'] = None

    # Try to load real embryo data
    if config.REAL_EMBRYO_PATH:
        real_path = convert_path(config.REAL_EMBRYO_PATH)
        if os.path.exists(real_path):
            print(f"Loading real embryo data from: {real_path}")
            with open(real_path, 'rb') as f:
                data['real'] = pickle.load(f)
            print(f"  Loaded {len(data['real'])} real embryos")
        else:
            print(f"  Real embryo data not found at: {real_path}")
            data['real'] = None
    else:
        data['real'] = None

    return data


def load_model(config: EvalConfig):
    """Load trained model"""
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)

    model = EnhancedTwinAttentionEncoder(
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_sparse_features=True,
        use_uncertainty=True,
        use_learnable_no_match=True
    )

    model_path = convert_path(config.MODEL_PATH)
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("  Model loaded successfully")
    else:
        print(f"  Model not found at: {model_path}")
        print("  Using randomly initialized model (results will not be meaningful)")

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    return model


# =============================================================================
# LINEAGE UTILITIES
# =============================================================================
C_ELEGANS_FOUNDERS = ['AB', 'MS', 'E', 'C', 'D', 'P4', 'P1', 'P2', 'P3', 'EMS', 'P0']

def get_founder_lineage(cell_id: str) -> str:
    """Extract founder lineage from cell ID"""
    cell_id = str(cell_id).upper()
    for founder in ['AB', 'MS', 'E', 'C', 'D', 'P4', 'P3', 'P2', 'P1', 'P0', 'EMS']:
        if cell_id.startswith(founder):
            return founder
    return 'UNKNOWN'

def get_sub_lineage(cell_id: str, depth: int = 2) -> str:
    """Get sub-lineage at specified depth (e.g., ABal, ABar, ABpl, ABpr)"""
    cell_id = str(cell_id).upper()
    founder = get_founder_lineage(cell_id)
    if founder == 'AB' and len(cell_id) > 2:
        return cell_id[:min(len(cell_id), 2 + depth)]
    return founder

def get_parent(cell_id: str) -> str:
    """Get parent cell ID"""
    if len(cell_id) <= 1:
        return None
    return cell_id[:-1]

def are_siblings(cell1: str, cell2: str) -> bool:
    """Check if two cells are siblings"""
    p1, p2 = get_parent(cell1), get_parent(cell2)
    return p1 is not None and p1 == p2

def is_parent_child(cell1: str, cell2: str) -> bool:
    """Check if cells have parent-child relationship"""
    return get_parent(cell1) == cell2 or get_parent(cell2) == cell1

def same_lineage_branch(cell1: str, cell2: str) -> bool:
    """Check if cells are in same lineage branch"""
    return get_founder_lineage(cell1) == get_founder_lineage(cell2)


# =============================================================================
# BOOTSTRAP UTILITIES
# =============================================================================
def bootstrap_ci(values, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval"""
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    values = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    mean = np.mean(values)

    return mean, lower, upper


# =============================================================================
# SECTION 3.1: CORE IDENTIFICATION PERFORMANCE
# =============================================================================
class CorePerformanceEvaluator:
    """Evaluate core identification performance metrics"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.loss_fn = TwinAttentionMatchingLoss(use_uncertainty=True)

    @torch.no_grad()
    def evaluate_dataset(self, dataloader, desc="Evaluating"):
        """Run evaluation on a dataset"""
        self.model.eval()

        all_results = []

        for batch in tqdm(dataloader, desc=desc):
            pc1, pc2, mask1, mask2, match_indices, info_list = batch
            pc1 = pc1.to(device)
            pc2 = pc2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            match_indices = match_indices.to(device)

            # Forward pass
            z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch=100)

            if self.model.use_uncertainty:
                z1_mean, z1_logvar = z1
                z2_mean, z2_logvar = z2
                sims = torch.bmm(z1_mean, z2_mean.transpose(1, 2)) / temperature
            else:
                sims = torch.bmm(z1, z2.transpose(1, 2)) / temperature

            pred_indices = sims.argmax(dim=-1)

            # Process each sample in batch
            B = pc1.shape[0]
            for b in range(B):
                n_valid = int(mask1[b].sum().item())
                info = info_list[b]

                for i in range(n_valid):
                    pred = pred_indices[b, i].item()
                    target = match_indices[b, i].item()
                    n2 = int(mask2[b].sum().item()) + 1  # +1 for no-match token

                    is_match = target < n2 - 1
                    is_correct = pred == target

                    # Get cell IDs
                    cell1_id = info['cells1_ids'][i] if i < len(info['cells1_ids']) else f"cell_{i}"
                    cell2_id = None
                    if is_match and target < len(info['cells2_ids']):
                        cell2_id = info['cells2_ids'][target]
                    pred_cell_id = None
                    if pred < len(info['cells2_ids']):
                        pred_cell_id = info['cells2_ids'][pred]

                    result = {
                        'embryo1': info['run1'],
                        'embryo2': info['run2'],
                        'time1': info['time1'],
                        'time2': info['time2'],
                        'cell1_id': cell1_id,
                        'cell2_id': cell2_id,
                        'pred_cell_id': pred_cell_id,
                        'pred': pred,
                        'target': target,
                        'is_match': is_match,
                        'is_correct': is_correct,
                        'n_cells1': n_valid,
                        'n_cells2': n2 - 1,
                        'confidence': torch.softmax(sims[b, i], dim=0).max().item()
                    }
                    all_results.append(result)

        return all_results

    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute all metrics from results"""
        if not results:
            return {}

        # Overall metrics
        match_results = [r for r in results if r['is_match']]
        nomatch_results = [r for r in results if not r['is_match']]

        match_correct = sum(1 for r in match_results if r['is_correct'])
        nomatch_correct = sum(1 for r in nomatch_results if r['is_correct'])
        total_correct = match_correct + nomatch_correct

        metrics = {
            'total_predictions': len(results),
            'match_total': len(match_results),
            'match_correct': match_correct,
            'match_accuracy': match_correct / len(match_results) if match_results else 0,
            'nomatch_total': len(nomatch_results),
            'nomatch_correct': nomatch_correct,
            'nomatch_accuracy': nomatch_correct / len(nomatch_results) if nomatch_results else 0,
            'overall_correct': total_correct,
            'overall_accuracy': total_correct / len(results) if results else 0,
        }

        # Bootstrap CIs
        correct_flags = [1 if r['is_correct'] else 0 for r in results]
        mean, lower, upper = bootstrap_ci(correct_flags, self.config.N_BOOTSTRAP)
        metrics['overall_accuracy_ci'] = (lower, upper)

        # By neighborhood size
        size_bins = {'sparse_5_10': [], 'medium_11_15': [], 'dense_16_20': []}
        for r in results:
            n = r['n_cells1']
            if 5 <= n <= 10:
                size_bins['sparse_5_10'].append(1 if r['is_correct'] else 0)
            elif 11 <= n <= 15:
                size_bins['medium_11_15'].append(1 if r['is_correct'] else 0)
            elif 16 <= n <= 20:
                size_bins['dense_16_20'].append(1 if r['is_correct'] else 0)

        for key, vals in size_bins.items():
            if vals:
                mean, lower, upper = bootstrap_ci(vals, self.config.N_BOOTSTRAP)
                metrics[f'{key}_accuracy'] = mean
                metrics[f'{key}_ci'] = (lower, upper)
                metrics[f'{key}_count'] = len(vals)

        # By developmental stage (cell count bins)
        stage_bins = {
            '4_20': (4, 20),
            '21_50': (21, 50),
            '51_100': (51, 100),
            '101_150': (101, 150),
            '151_194': (151, 194)
        }

        for stage_name, (low, high) in stage_bins.items():
            stage_results = [r for r in results if low <= r['n_cells1'] <= high]
            if stage_results:
                correct = [1 if r['is_correct'] else 0 for r in stage_results]
                mean, lower, upper = bootstrap_ci(correct, self.config.N_BOOTSTRAP)
                metrics[f'stage_{stage_name}_accuracy'] = mean
                metrics[f'stage_{stage_name}_ci'] = (lower, upper)
                metrics[f'stage_{stage_name}_count'] = len(stage_results)

        # Hierarchical accuracy
        hierarchical = self._compute_hierarchical_accuracy(results)
        metrics.update(hierarchical)

        return metrics

    def _compute_hierarchical_accuracy(self, results: List[dict]) -> dict:
        """Compute hierarchical identification accuracy"""
        exact_correct = 0
        sublineage_correct = 0
        founder_correct = 0
        binary_correct = 0  # AB vs non-AB
        total = 0

        for r in results:
            if not r['is_match']:
                continue

            total += 1
            cell1 = str(r['cell1_id'])
            pred_cell = str(r['pred_cell_id']) if r['pred_cell_id'] else ""
            target_cell = str(r['cell2_id']) if r['cell2_id'] else ""

            # Exact match
            if r['is_correct']:
                exact_correct += 1

            # Sub-lineage match
            if get_sub_lineage(cell1) == get_sub_lineage(pred_cell):
                sublineage_correct += 1

            # Founder lineage match
            if get_founder_lineage(cell1) == get_founder_lineage(pred_cell):
                founder_correct += 1

            # Binary (AB vs non-AB)
            cell1_is_ab = get_founder_lineage(cell1) == 'AB'
            pred_is_ab = get_founder_lineage(pred_cell) == 'AB'
            if cell1_is_ab == pred_is_ab:
                binary_correct += 1

        return {
            'hierarchical_exact': exact_correct / total if total > 0 else 0,
            'hierarchical_sublineage': sublineage_correct / total if total > 0 else 0,
            'hierarchical_founder': founder_correct / total if total > 0 else 0,
            'hierarchical_binary': binary_correct / total if total > 0 else 0,
            'hierarchical_total': total
        }


# =============================================================================
# SECTION 3.2: BASELINE METHODS
# =============================================================================
class BaselineEvaluator:
    """Evaluate baseline methods for comparison"""

    def __init__(self, config):
        self.config = config

    def evaluate_icp(self, pc1: np.ndarray, pc2: np.ndarray,
                     cells1_ids: List[str], cells2_ids: List[str],
                     max_iterations: int = 50) -> List[int]:
        """Iterative Closest Point matching"""
        # Simple ICP implementation
        source = pc1.copy()
        target = pc2.copy()

        # Center both point clouds
        source_centered = source - source.mean(axis=0)
        target_centered = target - target.mean(axis=0)

        # Iterative alignment
        for _ in range(max_iterations):
            # Find closest points
            nbrs = NearestNeighbors(n_neighbors=1).fit(target_centered)
            distances, indices = nbrs.kneighbors(source_centered)

            # Compute optimal rotation using SVD
            matched_target = target_centered[indices.flatten()]
            H = source_centered.T @ matched_target
            U, _, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T

            # Apply rotation
            source_centered = source_centered @ R_opt.T

        # Final matching
        nbrs = NearestNeighbors(n_neighbors=1).fit(target_centered)
        _, indices = nbrs.kneighbors(source_centered)

        return indices.flatten().tolist()

    def evaluate_cpd(self, pc1: np.ndarray, pc2: np.ndarray,
                     cells1_ids: List[str], cells2_ids: List[str]) -> List[int]:
        """Coherent Point Drift matching (simplified)"""
        try:
            from pycpd import RigidRegistration
            reg = RigidRegistration(X=pc2, Y=pc1)
            transformed, _ = reg.register()

            nbrs = NearestNeighbors(n_neighbors=1).fit(pc2)
            _, indices = nbrs.kneighbors(transformed)
            return indices.flatten().tolist()
        except ImportError:
            # Fallback to simple nearest neighbor after centering
            pc1_c = pc1 - pc1.mean(axis=0)
            pc2_c = pc2 - pc2.mean(axis=0)
            nbrs = NearestNeighbors(n_neighbors=1).fit(pc2_c)
            _, indices = nbrs.kneighbors(pc1_c)
            return indices.flatten().tolist()

    def evaluate_hungarian(self, pc1: np.ndarray, pc2: np.ndarray,
                          cells1_ids: List[str], cells2_ids: List[str]) -> List[int]:
        """Hungarian algorithm on distance matrix"""
        # Center point clouds
        pc1_c = pc1 - pc1.mean(axis=0)
        pc2_c = pc2 - pc2.mean(axis=0)

        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        cost_matrix = cdist(pc1_c, pc2_c, metric='euclidean')

        # Handle size mismatch by padding
        n1, n2 = len(pc1), len(pc2)
        if n1 != n2:
            max_n = max(n1, n2)
            padded = np.full((max_n, max_n), cost_matrix.max() * 10)
            padded[:n1, :n2] = cost_matrix
            cost_matrix = padded

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Map back to original indices
        matches = []
        for i in range(len(pc1)):
            if i < len(row_ind):
                match_idx = col_ind[np.where(row_ind == i)[0][0]]
                if match_idx < len(pc2):
                    matches.append(match_idx)
                else:
                    matches.append(len(pc2))  # no-match
            else:
                matches.append(len(pc2))

        return matches

    def run_baseline_evaluation(self, dataloader, method: str) -> List[dict]:
        """Run a baseline method on the dataset"""
        results = []

        for batch in tqdm(dataloader, desc=f"Evaluating {method}"):
            pc1, pc2, mask1, mask2, match_indices, info_list = batch

            B = pc1.shape[0]
            for b in range(B):
                n1 = int(mask1[b].sum().item())
                n2 = int(mask2[b].sum().item())

                pc1_np = pc1[b, :n1].numpy()
                pc2_np = pc2[b, :n2].numpy()

                info = info_list[b]
                cells1_ids = info['cells1_ids'][:n1]
                cells2_ids = info['cells2_ids'][:n2]

                # Run baseline method
                if method == 'icp':
                    pred_matches = self.evaluate_icp(pc1_np, pc2_np, cells1_ids, cells2_ids)
                elif method == 'cpd':
                    pred_matches = self.evaluate_cpd(pc1_np, pc2_np, cells1_ids, cells2_ids)
                elif method == 'hungarian':
                    pred_matches = self.evaluate_hungarian(pc1_np, pc2_np, cells1_ids, cells2_ids)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Compare with ground truth
                for i in range(n1):
                    target = match_indices[b, i].item()
                    pred = pred_matches[i] if i < len(pred_matches) else n2

                    is_match = target < n2
                    is_correct = pred == target

                    results.append({
                        'pred': pred,
                        'target': target,
                        'is_match': is_match,
                        'is_correct': is_correct,
                        'n_cells1': n1,
                        'method': method
                    })

        return results


# =============================================================================
# SECTION 3.3: ROBUSTNESS EVALUATION
# =============================================================================
class RobustnessEvaluator:
    """Evaluate model robustness to perturbations"""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    @torch.no_grad()
    def evaluate_with_perturbation(self, dataloader, perturbation_fn, desc=""):
        """Evaluate with a perturbation applied to input"""
        self.model.eval()
        results = []

        for batch in tqdm(dataloader, desc=desc):
            pc1, pc2, mask1, mask2, match_indices, info_list = batch

            # Apply perturbation
            pc1, pc2 = perturbation_fn(pc1, pc2, mask1, mask2)

            pc1 = pc1.to(device)
            pc2 = pc2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            match_indices = match_indices.to(device)

            z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch=100)

            if self.model.use_uncertainty:
                z1_mean, _ = z1
                z2_mean, _ = z2
                sims = torch.bmm(z1_mean, z2_mean.transpose(1, 2)) / temperature
            else:
                sims = torch.bmm(z1, z2.transpose(1, 2)) / temperature

            pred_indices = sims.argmax(dim=-1)

            B = pc1.shape[0]
            for b in range(B):
                n_valid = int(mask1[b].sum().item())
                for i in range(n_valid):
                    pred = pred_indices[b, i].item()
                    target = match_indices[b, i].item()
                    n2 = int(mask2[b].sum().item()) + 1

                    results.append({
                        'pred': pred,
                        'target': target,
                        'is_match': target < n2 - 1,
                        'is_correct': pred == target
                    })

        accuracy = sum(1 for r in results if r['is_correct']) / len(results) if results else 0
        return accuracy, results

    def missing_cells_perturbation(self, fraction: float):
        """Create perturbation that removes cells"""
        def perturb(pc1, pc2, mask1, mask2):
            pc1 = pc1.clone()
            mask1 = mask1.clone()

            B = pc1.shape[0]
            for b in range(B):
                n_valid = int(mask1[b].sum().item())
                n_remove = int(n_valid * fraction)

                if n_remove > 0 and n_valid - n_remove >= 4:
                    valid_indices = torch.where(mask1[b] > 0)[0]
                    remove_indices = valid_indices[torch.randperm(n_valid)[:n_remove]]
                    mask1[b, remove_indices] = 0
                    pc1[b, remove_indices] = 0

            return pc1, pc2
        return perturb

    def coordinate_noise_perturbation(self, scale: float):
        """Create perturbation that adds coordinate noise"""
        def perturb(pc1, pc2, mask1, mask2):
            pc1 = pc1.clone()
            pc2 = pc2.clone()

            B = pc1.shape[0]
            for b in range(B):
                # Compute mean nearest neighbor distance for scaling
                n1 = int(mask1[b].sum().item())
                if n1 > 1:
                    pts = pc1[b, :n1].numpy()
                    nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
                    dists, _ = nbrs.kneighbors(pts)
                    mean_nn_dist = dists[:, 1].mean()

                    noise_std = scale * mean_nn_dist
                    noise1 = torch.randn_like(pc1[b, :n1]) * noise_std
                    pc1[b, :n1] += noise1

                n2 = int(mask2[b].sum().item())
                if n2 > 1:
                    pts = pc2[b, :n2].numpy()
                    nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
                    dists, _ = nbrs.kneighbors(pts)
                    mean_nn_dist = dists[:, 1].mean()

                    noise_std = scale * mean_nn_dist
                    noise2 = torch.randn_like(pc2[b, :n2]) * noise_std
                    pc2[b, :n2] += noise2

            return pc1, pc2
        return perturb

    def no_perturbation(self):
        """Identity perturbation"""
        def perturb(pc1, pc2, mask1, mask2):
            return pc1, pc2
        return perturb

    def run_missing_cells_sweep(self, dataloader):
        """Sweep over different missing cell fractions"""
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4]
        results = {}

        for frac in fractions:
            print(f"\nEvaluating with {int(frac*100)}% missing cells...")
            if frac == 0:
                acc, _ = self.evaluate_with_perturbation(
                    dataloader, self.no_perturbation(), f"Missing 0%")
            else:
                acc, _ = self.evaluate_with_perturbation(
                    dataloader, self.missing_cells_perturbation(frac),
                    f"Missing {int(frac*100)}%")
            results[frac] = acc
            print(f"  Accuracy: {acc*100:.1f}%")

        return results

    def run_noise_sweep(self, dataloader):
        """Sweep over different noise scales"""
        scales = [0.0, 0.1, 0.2, 0.3, 0.5]
        results = {}

        for scale in scales:
            print(f"\nEvaluating with {scale}x noise...")
            if scale == 0:
                acc, _ = self.evaluate_with_perturbation(
                    dataloader, self.no_perturbation(), f"Noise 0.0x")
            else:
                acc, _ = self.evaluate_with_perturbation(
                    dataloader, self.coordinate_noise_perturbation(scale),
                    f"Noise {scale}x")
            results[scale] = acc
            print(f"  Accuracy: {acc*100:.1f}%")

        return results


# =============================================================================
# SECTION 3.4 & 3.5: ABLATION STUDIES
# =============================================================================
class AblationEvaluator:
    """Evaluate ablated model variants"""

    def __init__(self, config):
        self.config = config

    def create_model_variant(self,
                            use_sparse_features=True,
                            use_uncertainty=True,
                            use_learnable_no_match=True,
                            use_relative_pos=True,
                            use_density=True,
                            use_count=True,
                            use_centroid=True):
        """Create a model variant with specified ablations"""
        model = EnhancedTwinAttentionEncoder(
            embed_dim=self.config.EMBED_DIM,
            num_heads=self.config.NUM_HEADS,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT,
            use_sparse_features=use_sparse_features,
            use_uncertainty=use_uncertainty,
            use_learnable_no_match=use_learnable_no_match
        )
        return model.to(device)

    @torch.no_grad()
    def evaluate_model(self, model, dataloader, desc="Evaluating"):
        """Evaluate a model variant"""
        model.eval()
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc=desc):
            pc1, pc2, mask1, mask2, match_indices, info_list = batch
            pc1 = pc1.to(device)
            pc2 = pc2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            match_indices = match_indices.to(device)

            z1, z2, temperature = model(pc1, pc2, mask1, mask2, epoch=100)

            if model.use_uncertainty:
                z1_mean, _ = z1
                z2_mean, _ = z2
                sims = torch.bmm(z1_mean, z2_mean.transpose(1, 2)) / temperature
            else:
                sims = torch.bmm(z1, z2.transpose(1, 2)) / temperature

            pred_indices = sims.argmax(dim=-1)

            B = pc1.shape[0]
            for b in range(B):
                n_valid = int(mask1[b].sum().item())
                for i in range(n_valid):
                    pred = pred_indices[b, i].item()
                    target = match_indices[b, i].item()
                    if pred == target:
                        correct += 1
                    total += 1

        return correct / total if total > 0 else 0


# =============================================================================
# SECTION 3.6: EMBEDDING ANALYSIS
# =============================================================================
class EmbeddingAnalyzer:
    """Analyze learned embedding structure"""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    @torch.no_grad()
    def extract_embeddings(self, dataloader) -> Tuple[np.ndarray, List[str], List[int]]:
        """Extract embeddings for all cells"""
        self.model.eval()

        all_embeddings = []
        all_cell_ids = []
        all_times = []

        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            pc1, pc2, mask1, mask2, match_indices, info_list = batch
            pc1 = pc1.to(device)
            pc2 = pc2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)

            z1, z2, _ = self.model(pc1, pc2, mask1, mask2, epoch=100)

            if self.model.use_uncertainty:
                z1_mean, _ = z1
            else:
                z1_mean = z1

            B = pc1.shape[0]
            for b in range(B):
                n_valid = int(mask1[b].sum().item())
                info = info_list[b]

                for i in range(n_valid):
                    emb = z1_mean[b, i].cpu().numpy()
                    cell_id = info['cells1_ids'][i] if i < len(info['cells1_ids']) else f"cell_{i}"
                    time = info['time1']

                    all_embeddings.append(emb)
                    all_cell_ids.append(cell_id)
                    all_times.append(time)

        return np.array(all_embeddings), all_cell_ids, all_times

    def categorize_errors(self, results: List[dict]) -> dict:
        """Categorize errors by type"""
        error_types = {
            'sibling': 0,
            'parent_child': 0,
            'same_lineage_nonadjacent': 0,
            'distant_lineage': 0,
            'random': 0
        }

        errors = [r for r in results if not r['is_correct'] and r['is_match']]

        for r in errors:
            cell1 = str(r['cell1_id'])
            pred_cell = str(r['pred_cell_id']) if r['pred_cell_id'] else ""

            if not pred_cell:
                error_types['random'] += 1
            elif are_siblings(cell1, pred_cell):
                error_types['sibling'] += 1
            elif is_parent_child(cell1, pred_cell):
                error_types['parent_child'] += 1
            elif same_lineage_branch(cell1, pred_cell):
                error_types['same_lineage_nonadjacent'] += 1
            elif get_founder_lineage(cell1) != get_founder_lineage(pred_cell):
                error_types['distant_lineage'] += 1
            else:
                error_types['random'] += 1

        total_errors = sum(error_types.values())
        error_percentages = {k: v / total_errors * 100 if total_errors > 0 else 0
                           for k, v in error_types.items()}

        return {
            'counts': error_types,
            'percentages': error_percentages,
            'total_errors': total_errors
        }


# =============================================================================
# FIGURE GENERATION
# =============================================================================
class FigureGenerator:
    """Generate all figures from the paper"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12

    def figure2_core_performance(self, metrics: dict, save_name="figure2_core_performance.png"):
        """Generate Figure 2: Core identification performance"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel A: Overall accuracy (simulated vs real)
        ax = axes[0, 0]
        categories = ['Simulated']
        accuracies = [metrics.get('overall_accuracy', 0) * 100]
        cis = [metrics.get('overall_accuracy_ci', (0, 0))]

        if 'real_overall_accuracy' in metrics:
            categories.append('Real')
            accuracies.append(metrics['real_overall_accuracy'] * 100)
            cis.append(metrics.get('real_overall_accuracy_ci', (0, 0)))

        bars = ax.bar(categories, accuracies, color=['#2196F3', '#4CAF50'][:len(categories)])

        # Add error bars
        for i, (acc, ci) in enumerate(zip(accuracies, cis)):
            yerr = [[acc - ci[0]*100], [ci[1]*100 - acc]]
            ax.errorbar(i, acc, yerr=yerr, fmt='none', color='black', capsize=5)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('A  Overall accuracy (simulated vs. real)')
        ax.set_ylim(80, 100)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        # Panel B: Accuracy by developmental stage
        ax = axes[0, 1]
        stages = ['4-20', '21-50', '51-100', '101-150', '151-194']
        stage_accs = []
        stage_cis = []

        for stage in ['4_20', '21_50', '51_100', '101_150', '151_194']:
            acc = metrics.get(f'stage_{stage}_accuracy', 0) * 100
            ci = metrics.get(f'stage_{stage}_ci', (0, 0))
            stage_accs.append(acc)
            stage_cis.append(ci)

        x = range(len(stages))
        ax.plot(x, stage_accs, 'o-', color='#2196F3', linewidth=2, markersize=8)

        # Add error bands
        lowers = [ci[0]*100 for ci in stage_cis]
        uppers = [ci[1]*100 for ci in stage_cis]
        ax.fill_between(x, lowers, uppers, alpha=0.2, color='#2196F3')

        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_xlabel('Cell-count stage bin')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('B  Accuracy by developmental stage')
        ax.set_ylim(85, 100)

        # Panel C: Hierarchical identification
        ax = axes[1, 0]
        hier_names = ['Exact ID', 'Sub-lineage', 'Founder', 'Binary']
        hier_accs = [
            metrics.get('hierarchical_exact', 0) * 100,
            metrics.get('hierarchical_sublineage', 0) * 100,
            metrics.get('hierarchical_founder', 0) * 100,
            metrics.get('hierarchical_binary', 0) * 100
        ]

        bars = ax.bar(hier_names, hier_accs, color='#2196F3')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('C  Hierarchical identification')
        ax.set_ylim(90, 100)

        for bar, acc in zip(bars, hier_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

        # Panel D: Performance by correspondence type
        ax = axes[1, 1]
        corr_types = ['Match', 'No-match']
        corr_accs = [
            metrics.get('match_accuracy', 0) * 100,
            metrics.get('nomatch_accuracy', 0) * 100
        ]

        bars = ax.bar(corr_types, corr_accs, color='#2196F3')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('D  Performance by correspondence type')
        ax.set_ylim(80, 100)

        for bar, acc in zip(bars, corr_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def figure3_baseline_comparison(self, baseline_results: dict,
                                    model_accuracy: float,
                                    save_name="figure3_baseline_comparison.png"):
        """Generate Figure 3: Baseline method comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['ICP', 'CPD', 'Hungarian', 'Siamese', 'Joint Attention']
        accuracies = [
            baseline_results.get('icp', 0) * 100,
            baseline_results.get('cpd', 0) * 100,
            baseline_results.get('hungarian', 0) * 100,
            baseline_results.get('siamese', 0) * 100,
            model_accuracy * 100
        ]

        colors = ['#FFA726', '#FFA726', '#FFA726', '#42A5F5', '#2196F3']
        bars = ax.bar(methods, accuracies, color=colors)

        ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Baseline method comparison')
        ax.set_ylim(40, 100)
        ax.legend()

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def figure4_robustness(self, missing_results: dict, noise_results: dict,
                          save_name="figure4_robustness.png"):
        """Generate Figure 4: Robustness to perturbations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel A: Missing cells
        ax = axes[0, 0]
        fracs = sorted(missing_results.keys())
        accs = [missing_results[f] * 100 for f in fracs]
        ax.plot([f*100 for f in fracs], accs, 'o-', color='#2196F3', linewidth=2, markersize=8)
        ax.axhline(y=85, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Cells removed (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('A  Missing cells')
        ax.set_ylim(65, 100)

        # Panel B: Coordinate noise
        ax = axes[0, 1]
        scales = sorted(noise_results.keys())
        accs = [noise_results[s] * 100 for s in scales]
        ax.plot(scales, accs, 'o-', color='#2196F3', linewidth=2, markersize=8)
        ax.set_xlabel('Noise scale (× mean NN distance)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('B  Coordinate noise')
        ax.set_ylim(75, 100)

        # Panel C: Summary bar chart
        ax = axes[1, 0]
        conditions = ['Baseline', '10% missing', '20% missing', '0.1× noise', '0.2× noise']
        cond_accs = [
            missing_results.get(0.0, 0) * 100,
            missing_results.get(0.1, 0) * 100,
            missing_results.get(0.2, 0) * 100,
            noise_results.get(0.1, 0) * 100,
            noise_results.get(0.2, 0) * 100
        ]
        bars = ax.bar(conditions, cond_accs, color='#2196F3')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('C  Perturbation summary')
        ax.set_ylim(75, 100)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Panel D: Combined perturbations (placeholder)
        ax = axes[1, 1]
        ax.text(0.5, 0.5, 'Combined perturbations\n(requires temporal data)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('D  Combined perturbations')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def figure5_feature_contributions(self, feature_results: dict,
                                      save_name="figure5_feature_contributions.png"):
        """Generate Figure 5: Feature contributions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Progressive addition
        ax = axes[0]
        if 'progressive' in feature_results:
            stages = list(feature_results['progressive'].keys())
            accs = [feature_results['progressive'][s] * 100 for s in stages]
            bars = ax.bar(stages, accs, color='#2196F3')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('A  Progressive addition of geometric features')
            ax.set_ylim(65, 100)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

        # Panel B: Ablation
        ax = axes[1]
        if 'ablation' in feature_results:
            features = list(feature_results['ablation'].keys())
            accs = [feature_results['ablation'][f] * 100 for f in features]
            colors = ['#4CAF50' if f == 'Full' else '#F44336' for f in features]
            bars = ax.bar(features, accs, color=colors)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('B  Individual feature ablation')
            ax.set_ylim(70, 100)

            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def figure6_architecture_ablation(self, arch_results: dict,
                                      save_name="figure6_architecture_ablation.png"):
        """Generate Figure 6: Architectural component ablation"""
        fig, ax = plt.subplots(figsize=(10, 6))

        components = list(arch_results.keys())
        accs = [arch_results[c] * 100 for c in components]

        colors = ['#4CAF50' if c == 'Full model' else '#F44336' for c in components]
        bars = ax.bar(components, accs, color=colors)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Architectural component ablation')
        ax.set_ylim(60, 100)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")

    def figure7_embedding_analysis(self, embeddings: np.ndarray, cell_ids: List[str],
                                   error_analysis: dict,
                                   save_name="figure7_embedding_analysis.png"):
        """Generate Figure 7: Embedding structure and error analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: t-SNE visualization
        ax = axes[0]
        if len(embeddings) > 0:
            # Subsample if too many points
            max_points = 3000
            if len(embeddings) > max_points:
                idx = np.random.choice(len(embeddings), max_points, replace=False)
                emb_subset = embeddings[idx]
                ids_subset = [cell_ids[i] for i in idx]
            else:
                emb_subset = embeddings
                ids_subset = cell_ids

            # Run t-SNE
            print("Running t-SNE...")
            tsne = TSNE(n_components=2, perplexity=min(30, len(emb_subset)//4),
                       random_state=42, max_iter=1000)
            emb_2d = tsne.fit_transform(emb_subset)

            # Color by founder lineage
            founders = [get_founder_lineage(cid) for cid in ids_subset]
            unique_founders = list(set(founders))
            founder_colors = {f: plt.cm.tab10(i) for i, f in enumerate(unique_founders)}
            colors = [founder_colors[f] for f in founders]

            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=5, alpha=0.6)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('A  t-SNE by founder lineage')

            # Add legend
            for f in unique_founders:
                ax.scatter([], [], c=[founder_colors[f]], label=f, s=30)
            ax.legend(loc='upper right', fontsize=8)

        # Panel B: Error type distribution
        ax = axes[1]
        if error_analysis and 'percentages' in error_analysis:
            labels = ['Sibling', 'Parent-child', 'Same-lineage\nnon-adjacent',
                     'Distant lineage', 'Random']
            sizes = [
                error_analysis['percentages'].get('sibling', 0),
                error_analysis['percentages'].get('parent_child', 0),
                error_analysis['percentages'].get('same_lineage_nonadjacent', 0),
                error_analysis['percentages'].get('distant_lineage', 0),
                error_analysis['percentages'].get('random', 0)
            ]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

            # Filter out zero values
            nonzero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
            if nonzero:
                labels, sizes, colors = zip(*nonzero)
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                      startangle=90, pctdistance=0.75)

            ax.set_title(f'B  Error type distribution (n={error_analysis.get("total_errors", 0)})')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_name}")


# =============================================================================
# MAIN EVALUATION RUNNER
# =============================================================================
class EvaluationRunner:
    """Main class to run all evaluations"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.results = {}

        # Create output directories
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.FIGURE_DIR, exist_ok=True)

        self.fig_gen = FigureGenerator(config.FIGURE_DIR)

    def run_full_evaluation(self):
        """Run all evaluations from the paper"""
        print("\n" + "="*70)
        print("TWIN ATTENTION MODEL EVALUATION SUITE")
        print("="*70)

        # Load data and model
        data = load_data(self.config)
        model = load_model(self.config)

        # Check what data is available
        eval_data = data.get('eval') or data.get('train')
        if eval_data is None:
            print("\nERROR: No evaluation data available!")
            print("Please ensure data files exist at the specified paths.")
            return self.results

        # Create evaluation dataset
        print("\nCreating evaluation dataset...")
        eval_dataset = SparseEmbryoDataset(
            eval_data,
            stage_limit=self.config.STAGE_LIMIT,
            min_cells=self.config.MIN_CELLS,
            max_cells=self.config.MAX_CELLS,
            augment=False,
            num_rotations=1
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_with_padding,
            num_workers=0
        )

        # =================================================================
        # SECTION 3.1: Core Performance
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.1: CORE IDENTIFICATION PERFORMANCE")
        print("="*60)

        core_eval = CorePerformanceEvaluator(model, self.config)
        results = core_eval.evaluate_dataset(eval_loader, "Core evaluation")
        metrics = core_eval.compute_metrics(results)

        self.results['core_performance'] = metrics

        print(f"\nCore Performance Results:")
        print(f"  Total predictions: {metrics.get('total_predictions', 0)}")
        print(f"  Overall accuracy: {metrics.get('overall_accuracy', 0)*100:.1f}%")
        if 'overall_accuracy_ci' in metrics:
            ci = metrics['overall_accuracy_ci']
            print(f"    95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
        print(f"  Match accuracy: {metrics.get('match_accuracy', 0)*100:.1f}%")
        print(f"  No-match accuracy: {metrics.get('nomatch_accuracy', 0)*100:.1f}%")

        print(f"\n  By neighborhood size:")
        for key in ['sparse_5_10', 'medium_11_15', 'dense_16_20']:
            if f'{key}_accuracy' in metrics:
                print(f"    {key}: {metrics[f'{key}_accuracy']*100:.1f}% (n={metrics.get(f'{key}_count', 0)})")

        print(f"\n  By developmental stage:")
        for stage in ['4_20', '21_50', '51_100', '101_150', '151_194']:
            if f'stage_{stage}_accuracy' in metrics:
                print(f"    {stage} cells: {metrics[f'stage_{stage}_accuracy']*100:.1f}%")

        print(f"\n  Hierarchical accuracy:")
        print(f"    Exact: {metrics.get('hierarchical_exact', 0)*100:.1f}%")
        print(f"    Sub-lineage: {metrics.get('hierarchical_sublineage', 0)*100:.1f}%")
        print(f"    Founder: {metrics.get('hierarchical_founder', 0)*100:.1f}%")
        print(f"    Binary: {metrics.get('hierarchical_binary', 0)*100:.1f}%")

        # Generate Figure 2
        self.fig_gen.figure2_core_performance(metrics)

        # =================================================================
        # SECTION 3.2: Baseline Comparisons
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.2: BASELINE METHOD COMPARISON")
        print("="*60)

        baseline_eval = BaselineEvaluator(self.config)
        baseline_results = {}

        for method in ['icp', 'cpd', 'hungarian']:
            print(f"\nEvaluating {method.upper()}...")
            method_results = baseline_eval.run_baseline_evaluation(eval_loader, method)
            accuracy = sum(1 for r in method_results if r['is_correct']) / len(method_results)
            baseline_results[method] = accuracy
            print(f"  {method.upper()} accuracy: {accuracy*100:.1f}%")

        # Note: Siamese baseline would require training a separate model
        baseline_results['siamese'] = 0.0  # Placeholder

        self.results['baseline_comparison'] = baseline_results

        # Generate Figure 3
        self.fig_gen.figure3_baseline_comparison(
            baseline_results,
            metrics.get('overall_accuracy', 0)
        )

        # =================================================================
        # SECTION 3.3: Robustness Evaluation
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.3: ROBUSTNESS TO PERTURBATIONS")
        print("="*60)

        robust_eval = RobustnessEvaluator(model, self.config)

        print("\n--- Missing Cells Sweep ---")
        missing_results = robust_eval.run_missing_cells_sweep(eval_loader)

        print("\n--- Coordinate Noise Sweep ---")
        noise_results = robust_eval.run_noise_sweep(eval_loader)

        self.results['robustness'] = {
            'missing_cells': missing_results,
            'coordinate_noise': noise_results
        }

        # Generate Figure 4
        self.fig_gen.figure4_robustness(missing_results, noise_results)

        # =================================================================
        # SECTION 3.6: Embedding Analysis
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.6: EMBEDDING ANALYSIS")
        print("="*60)

        emb_analyzer = EmbeddingAnalyzer(model, self.config)

        print("\nExtracting embeddings...")
        embeddings, cell_ids, times = emb_analyzer.extract_embeddings(eval_loader)
        print(f"  Extracted {len(embeddings)} embeddings")

        print("\nAnalyzing errors...")
        error_analysis = emb_analyzer.categorize_errors(results)
        print(f"  Total errors: {error_analysis['total_errors']}")
        print(f"  Error breakdown:")
        for k, v in error_analysis['percentages'].items():
            print(f"    {k}: {v:.1f}%")

        self.results['embedding_analysis'] = {
            'n_embeddings': len(embeddings),
            'error_analysis': error_analysis
        }

        # Generate Figure 7
        self.fig_gen.figure7_embedding_analysis(embeddings, cell_ids, error_analysis)

        # =================================================================
        # SAVE RESULTS
        # =================================================================
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

        results_path = os.path.join(self.config.OUTPUT_DIR, 'evaluation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to: {results_path}")

        # Save text summary
        summary_path = os.path.join(self.config.OUTPUT_DIR, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("TWIN ATTENTION MODEL EVALUATION SUMMARY\n")
            f.write("="*50 + "\n\n")

            f.write("CORE PERFORMANCE\n")
            f.write("-"*30 + "\n")
            f.write(f"Overall accuracy: {metrics.get('overall_accuracy', 0)*100:.1f}%\n")
            f.write(f"Match accuracy: {metrics.get('match_accuracy', 0)*100:.1f}%\n")
            f.write(f"No-match accuracy: {metrics.get('nomatch_accuracy', 0)*100:.1f}%\n\n")

            f.write("BASELINE COMPARISON\n")
            f.write("-"*30 + "\n")
            for method, acc in baseline_results.items():
                f.write(f"{method.upper()}: {acc*100:.1f}%\n")
            f.write(f"Joint Attention (ours): {metrics.get('overall_accuracy', 0)*100:.1f}%\n\n")

            f.write("ROBUSTNESS\n")
            f.write("-"*30 + "\n")
            f.write("Missing cells:\n")
            for frac, acc in missing_results.items():
                f.write(f"  {int(frac*100)}%: {acc*100:.1f}%\n")
            f.write("Coordinate noise:\n")
            for scale, acc in noise_results.items():
                f.write(f"  {scale}x: {acc*100:.1f}%\n")

        print(f"Summary saved to: {summary_path}")

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"\nResults directory: {self.config.OUTPUT_DIR}")
        print(f"Figures directory: {self.config.FIGURE_DIR}")

        return self.results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point for evaluation"""
    # Create config with paths
    config = EvalConfig()

    # Allow command-line overrides
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Twin Attention Model')
    parser.add_argument('--train-data', type=str, help='Path to training data')
    parser.add_argument('--eval-data', type=str, help='Path to evaluation data')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--real-data', type=str, help='Path to real embryo data')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    args = parser.parse_args()

    if args.train_data:
        config.TRAIN_DATA_PATH = args.train_data
    if args.eval_data:
        config.EVAL_DATA_PATH = args.eval_data
    if args.model:
        config.MODEL_PATH = args.model
    if args.real_data:
        config.REAL_EMBRYO_PATH = args.real_data
    if args.output:
        config.OUTPUT_DIR = args.output
        config.FIGURE_DIR = os.path.join(args.output, 'figures')

    # Run evaluation
    runner = EvaluationRunner(config)
    results = runner.run_full_evaluation()

    return results


if __name__ == "__main__":
    main()
