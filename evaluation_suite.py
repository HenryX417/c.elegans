"""
Comprehensive Evaluation Suite for Twin Attention Cell Identification Model
Generates all results for paper Sections 3.1-3.6
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
print(f"Using device: {device}")


# =============================================================================
# CONFIGURATION
# =============================================================================
class EvalConfig:
    TRAIN_DATA_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\data_dict.pkl"
    EVAL_DATA_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\evaluation_data_dict.pkl"
    MODEL_PATH = r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\twin_attention_final.pth"
    REAL_EMBRYO_PATH = None

    OUTPUT_DIR = "evaluation_results"
    FIGURE_DIR = "evaluation_figures"

    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 6
    DROPOUT = 0.1

    BATCH_SIZE = 16
    MIN_CELLS = 5
    MAX_CELLS = 20
    STAGE_LIMIT = 194
    K_NEIGHBORS = 30

    N_BOOTSTRAP = 100

    # For faster CPU KNN
    MAX_TRAIN_EMBEDDINGS = 50000
    KNN_BATCH_SIZE = 1000


def convert_path(path):
    if path is None:
        return None
    if os.name != 'nt' and '\\' in path:
        return os.path.basename(path.replace('\\', '/'))
    return path


# =============================================================================
# DATA & MODEL LOADING
# =============================================================================
def load_data(config):
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    data = {}

    train_path = convert_path(config.TRAIN_DATA_PATH)
    if os.path.exists(train_path):
        print(f"Loading training data from: {train_path}")
        with open(train_path, 'rb') as f:
            data['train'] = pickle.load(f)
        print(f"  Loaded {len(data['train'])} embryos")
    else:
        data['train'] = None

    eval_path = convert_path(config.EVAL_DATA_PATH)
    if os.path.exists(eval_path):
        print(f"Loading evaluation data from: {eval_path}")
        with open(eval_path, 'rb') as f:
            data['eval'] = pickle.load(f)
        print(f"  Loaded {len(data['eval'])} embryos")
    else:
        data['eval'] = None

    # Show sample cell IDs
    sample_data = data.get('train') or data.get('eval')
    if sample_data:
        print("\n  Sample cell IDs from data:")
        for embryo, timepoints in list(sample_data.items())[:1]:
            for t, cells in list(timepoints.items())[:1]:
                print(f"    {list(cells.keys())[:8]}")

    return data


def load_model(config, custom_config=None):
    """Load model with optional custom configuration for ablations"""
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)

    # Use custom config for ablations, otherwise default
    embed_dim = custom_config.get('embed_dim', config.EMBED_DIM) if custom_config else config.EMBED_DIM
    num_heads = custom_config.get('num_heads', config.NUM_HEADS) if custom_config else config.NUM_HEADS
    num_layers = custom_config.get('num_layers', config.NUM_LAYERS) if custom_config else config.NUM_LAYERS
    use_sparse = custom_config.get('use_sparse_features', True) if custom_config else True
    use_uncertainty = custom_config.get('use_uncertainty', True) if custom_config else True

    model = EnhancedTwinAttentionEncoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=config.DROPOUT,
        use_sparse_features=use_sparse,
        use_uncertainty=use_uncertainty,
        use_learnable_no_match=True
    )

    model_path = convert_path(config.MODEL_PATH)
    if os.path.exists(model_path) and custom_config is None:
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("  Model loaded successfully")
    elif custom_config:
        print(f"  Using untrained ablation model: {custom_config}")
    else:
        print(f"  WARNING: Model not found at {model_path}")

    model = model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


# =============================================================================
# LINEAGE UTILITIES
# =============================================================================
def parse_cell_id(cell_id):
    """Parse cell ID - handles C. elegans nomenclature"""
    cell_id = str(cell_id).strip()
    founders = ['AB', 'MS', 'E', 'C', 'D', 'P4', 'P3', 'P2', 'P1', 'P0', 'EMS', 'Z2', 'Z3']
    upper_id = cell_id.upper()

    for founder in founders:
        if upper_id.startswith(founder):
            return {'founder': founder, 'lineage': cell_id, 'depth': len(cell_id) - len(founder)}

    return {'founder': 'UNKNOWN', 'lineage': cell_id, 'depth': 0}


def get_founder(cell_id):
    return parse_cell_id(cell_id)['founder']


def same_founder(c1, c2):
    return get_founder(c1) == get_founder(c2)


def are_siblings(c1, c2):
    p1, p2 = parse_cell_id(c1), parse_cell_id(c2)
    if p1['founder'] == 'UNKNOWN' or p2['founder'] == 'UNKNOWN':
        return False
    l1, l2 = p1['lineage'], p2['lineage']
    return len(l1) >= 2 and len(l2) >= 2 and l1[:-1] == l2[:-1] and l1 != l2


def same_sublineage(c1, c2, depth=2):
    p1, p2 = parse_cell_id(c1), parse_cell_id(c2)
    if p1['founder'] != p2['founder']:
        return False
    if p1['founder'] == 'UNKNOWN':
        return c1 == c2
    f = p1['founder']
    prefix1 = p1['lineage'][:len(f) + min(depth, p1['depth'])]
    prefix2 = p2['lineage'][:len(f) + min(depth, p2['depth'])]
    return prefix1 == prefix2


# =============================================================================
# BOOTSTRAP CI
# =============================================================================
def bootstrap_ci(values, n=100):
    if len(values) == 0:
        return 0, 0, 0
    values = np.array(values)
    means = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)


# =============================================================================
# CORE EVALUATION - MATCHES TRAINING CODE EXACTLY
# =============================================================================
class MatchingEvaluator:
    """
    Evaluates using EXACT same logic as TwinAttentionMatchingLoss in training.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.loss_fn = TwinAttentionMatchingLoss(use_uncertainty=True)

    @torch.no_grad()
    def evaluate(self, dataloader, desc="Evaluating"):
        """Evaluate using same logic as training validation"""
        self.model.eval()

        total_loss = 0
        match_correct = 0
        match_total = 0
        outlier_correct = 0
        outlier_total = 0
        batch_count = 0

        all_results = []

        for batch in tqdm(dataloader, desc=desc):
            pc1, pc2, mask1, mask2, match_indices, info_list = batch
            pc1 = pc1.to(device)
            pc2 = pc2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            match_indices = match_indices.to(device)

            z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch=100)
            loss, metrics = self.loss_fn(z1, z2, match_indices, temperature, mask1, mask2)

            total_loss += loss.item()
            match_correct += metrics['match_correct']
            match_total += metrics['match_total']
            outlier_correct += metrics['outlier_correct']
            outlier_total += metrics['outlier_total']
            batch_count += 1

            # Store detailed results
            if self.model.use_uncertainty:
                z1_mean, _ = z1
                z2_mean, _ = z2
                sims = torch.bmm(z1_mean, z2_mean.transpose(1, 2)) / temperature
            else:
                sims = torch.bmm(z1, z2.transpose(1, 2)) / temperature

            pred_indices = sims.argmax(dim=-1)
            B, N1, N2 = sims.shape

            for b in range(B):
                info = info_list[b]
                n_valid = int(mask1[b].sum().item())

                for i in range(n_valid):
                    target = match_indices[b, i].item()
                    pred = pred_indices[b, i].item()
                    is_match = target < N2 - 1

                    all_results.append({
                        'is_match': is_match,
                        'target': target,
                        'pred': pred,
                        'correct': pred == target,
                        'n_cells': n_valid,
                        'cell_id': info['cells1_ids'][i] if i < len(info['cells1_ids']) else None,
                        'confidence': torch.softmax(sims[b, i], dim=0).max().item()
                    })

        # Compute accuracy by neighborhood size
        size_results = {}
        for name, (lo, hi) in [('sparse_5_10', (5, 10)), ('medium_11_15', (11, 15)), ('dense_16_20', (16, 20))]:
            bin_results = [r for r in all_results if r['is_match'] and lo <= r['n_cells'] <= hi]
            if bin_results:
                bin_acc = sum(1 for r in bin_results if r['correct']) / len(bin_results)
                size_results[name] = {'accuracy': bin_acc, 'count': len(bin_results)}

        results = {
            'loss': total_loss / batch_count if batch_count > 0 else 0,
            'match_accuracy': match_correct / match_total if match_total > 0 else 0,
            'match_correct': match_correct,
            'match_total': match_total,
            'outlier_accuracy': outlier_correct / outlier_total if outlier_total > 0 else 0,
            'outlier_correct': outlier_correct,
            'outlier_total': outlier_total,
            'overall_accuracy': (match_correct + outlier_correct) / (match_total + outlier_total) if (match_total + outlier_total) > 0 else 0,
            'by_size': size_results,
            'detailed_results': all_results
        }

        return results


# =============================================================================
# KNN-BASED CELL IDENTIFICATION (Section 2.8) - IMPROVED
# =============================================================================
class KNNIdentifier:
    """KNN-based cell identification with averaged embeddings per unique cell"""

    def __init__(self, k=30):
        self.k = k
        self.embeddings = None
        self.labels = None

    def fit(self, embeddings, labels):
        """Build index from averaged embeddings per unique cell"""
        # Average embeddings for each unique cell label
        print("  Averaging embeddings per unique cell...")
        label_to_embeddings = defaultdict(list)
        for emb, label in zip(embeddings, labels):
            label_to_embeddings[label].append(emb)

        avg_embeddings = []
        avg_labels = []
        for label, embs in label_to_embeddings.items():
            avg_emb = np.mean(embs, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
            avg_embeddings.append(avg_emb)
            avg_labels.append(label)

        self.embeddings = np.array(avg_embeddings)
        self.labels = avg_labels

        self.nn = NearestNeighbors(
            n_neighbors=min(self.k, len(self.embeddings)),
            metric='cosine', algorithm='brute', n_jobs=-1
        )
        self.nn.fit(self.embeddings)
        print(f"  KNN index: {len(self.embeddings)} unique cells, k={self.k}")

    def predict(self, queries, query_labels, batch_size=1000):
        """Predict cell IDs via majority vote (using averaged query embeddings)"""
        # Average query embeddings per unique label
        label_to_embeddings = defaultdict(list)
        label_to_indices = defaultdict(list)
        for idx, (emb, label) in enumerate(zip(queries, query_labels)):
            label_to_embeddings[label].append(emb)
            label_to_indices[label].append(idx)

        # Compute averaged predictions
        unique_labels = list(label_to_embeddings.keys())
        avg_queries = []
        for label in unique_labels:
            avg_emb = np.mean(label_to_embeddings[label], axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
            avg_queries.append(avg_emb)
        avg_queries = np.array(avg_queries)

        # KNN prediction on averaged queries
        predictions_unique = []
        confidences_unique = []

        n_batches = (len(avg_queries) + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc="KNN predictions"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(avg_queries))
            batch = avg_queries[start:end]

            _, indices = self.nn.kneighbors(batch)

            for idx_list in indices:
                neighbor_labels = [self.labels[j] for j in idx_list]
                counts = Counter(neighbor_labels)
                top, count = counts.most_common(1)[0]
                predictions_unique.append(top)
                confidences_unique.append(count / len(idx_list))

        # Map back to original indices
        predictions = [''] * len(queries)
        confidences = [0.0] * len(queries)
        for i, label in enumerate(unique_labels):
            for idx in label_to_indices[label]:
                predictions[idx] = predictions_unique[i]
                confidences[idx] = confidences_unique[i]

        return predictions, confidences, unique_labels, predictions_unique, confidences_unique


class KNNEvaluator:
    """Evaluate KNN-based cell identification with proper averaging"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.knn = KNNIdentifier(k=config.K_NEIGHBORS)

    @torch.no_grad()
    def extract_embeddings(self, dataloader, desc="Extracting"):
        """Extract cell embeddings"""
        self.model.eval()
        embeddings, labels, metadata = [], [], []

        for batch in tqdm(dataloader, desc=desc):
            pc1, pc2, mask1, mask2, _, info_list = batch
            pc1 = pc1.to(device)
            pc2 = pc2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)

            z1, _, _ = self.model(pc1, pc2, mask1, mask2, epoch=100)
            z1_mean = z1[0] if self.model.use_uncertainty else z1

            for b in range(pc1.shape[0]):
                n = int(mask1[b].sum().item())
                info = info_list[b]
                for i in range(n):
                    embeddings.append(z1_mean[b, i].cpu().numpy())
                    labels.append(info['cells1_ids'][i] if i < len(info['cells1_ids']) else f"cell_{i}")
                    metadata.append({'n_cells': n, 'embryo': info['run1']})

        return np.array(embeddings), labels, metadata

    def evaluate(self, train_loader, test_loader):
        """Full KNN evaluation with averaged embeddings per unique cell"""
        print("\nBuilding KNN index from training data...")
        train_emb, train_labels, _ = self.extract_embeddings(train_loader, "Train embeddings")
        self.knn.fit(train_emb, train_labels)

        print("\nEvaluating on test data...")
        test_emb, test_labels, test_meta = self.extract_embeddings(test_loader, "Test embeddings")
        print(f"  Running KNN on {len(test_emb)} test embeddings ({len(set(test_labels))} unique cells)...")

        predictions, confidences, unique_labels, unique_preds, unique_confs = self.knn.predict(
            test_emb, test_labels, batch_size=self.config.KNN_BATCH_SIZE
        )

        # Compute metrics on UNIQUE cells (one vote per cell type)
        unique_correct = [1 if p == t else 0 for p, t in zip(unique_preds, unique_labels)]
        mean_acc, ci_low, ci_high = bootstrap_ci(unique_correct, self.config.N_BOOTSTRAP)

        # Hierarchical accuracy (on unique cells)
        hier = {'exact': 0, 'sublineage': 0, 'founder': 0, 'binary': 0}
        for pred, true in zip(unique_preds, unique_labels):
            if pred == true:
                hier['exact'] += 1
            if same_sublineage(pred, true):
                hier['sublineage'] += 1
            if same_founder(pred, true):
                hier['founder'] += 1
            true_ab = get_founder(true) == 'AB'
            pred_ab = get_founder(pred) == 'AB'
            if true_ab == pred_ab:
                hier['binary'] += 1

        n = len(unique_labels)
        hier = {k: v / n for k, v in hier.items()}

        return {
            'overall_accuracy': mean_acc,
            'ci': (ci_low, ci_high),
            'total': len(test_labels),
            'unique_cells': n,
            'hierarchical': hier,
            'test_embeddings': test_emb,
            'test_labels': test_labels,
            'predictions': predictions,
            'confidences': confidences,
            'unique_labels': unique_labels,
            'unique_predictions': unique_preds
        }


# =============================================================================
# BASELINE METHODS
# =============================================================================
class BaselineEvaluator:
    def __init__(self, config):
        self.config = config

    def icp(self, pc1, pc2, iters=50):
        src = pc1 - pc1.mean(0)
        tgt = pc2 - pc2.mean(0)
        for _ in range(iters):
            nn = NearestNeighbors(n_neighbors=1).fit(tgt)
            _, idx = nn.kneighbors(src)
            matched = tgt[idx.flatten()]
            H = src.T @ matched
            U, _, Vt = np.linalg.svd(H)
            R_mat = Vt.T @ U.T
            if np.linalg.det(R_mat) < 0:
                Vt[-1, :] *= -1
                R_mat = Vt.T @ U.T
            src = src @ R_mat.T
        nn = NearestNeighbors(n_neighbors=1).fit(tgt)
        _, idx = nn.kneighbors(src)
        return idx.flatten()

    def cpd(self, pc1, pc2):
        src = (pc1 - pc1.mean(0)) / (pc1.std() + 1e-8)
        tgt = (pc2 - pc2.mean(0)) / (pc2.std() + 1e-8)
        nn = NearestNeighbors(n_neighbors=1).fit(tgt)
        _, idx = nn.kneighbors(src)
        return idx.flatten()

    def hungarian(self, pc1, pc2):
        from scipy.spatial.distance import cdist
        src = pc1 - pc1.mean(0)
        tgt = pc2 - pc2.mean(0)
        cost = cdist(src, tgt)
        n1, n2 = len(pc1), len(pc2)
        if n1 != n2:
            pad = np.full((max(n1, n2), max(n1, n2)), cost.max() * 10)
            pad[:n1, :n2] = cost
            cost = pad
        row, col = linear_sum_assignment(cost)
        return np.array([col[np.where(row == i)[0][0]] if i in row else n2 for i in range(n1)])

    def evaluate(self, dataloader, method):
        """Evaluate baseline on match cases only"""
        correct, total = 0, 0

        for batch in tqdm(dataloader, desc=f"Baseline: {method}"):
            pc1, pc2, mask1, mask2, match_indices, _ = batch
            B = pc1.shape[0]

            for b in range(B):
                n1 = int(mask1[b].sum().item())
                n2 = int(mask2[b].sum().item())
                p1 = pc1[b, :n1].numpy()
                p2 = pc2[b, :n2].numpy()

                if method == 'icp':
                    preds = self.icp(p1, p2)
                elif method == 'cpd':
                    preds = self.cpd(p1, p2)
                elif method == 'hungarian':
                    preds = self.hungarian(p1, p2)

                for i in range(n1):
                    tgt = match_indices[b, i].item()
                    if tgt < n2:  # Only count match cases
                        total += 1
                        if i < len(preds) and preds[i] == tgt:
                            correct += 1

        return correct / total if total > 0 else 0


# =============================================================================
# ROBUSTNESS EVALUATION
# =============================================================================
class RobustnessEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.loss_fn = TwinAttentionMatchingLoss(use_uncertainty=True)

    @torch.no_grad()
    def evaluate_perturbed(self, dataloader, perturb_fn, desc=""):
        self.model.eval()
        match_correct, match_total = 0, 0

        for batch in tqdm(dataloader, desc=desc):
            pc1, pc2, mask1, mask2, match_indices, _ = batch
            pc1, pc2, mask1, mask2 = perturb_fn(pc1, pc2, mask1, mask2)

            pc1 = pc1.to(device)
            pc2 = pc2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            match_indices = match_indices.to(device)

            z1, z2, temp = self.model(pc1, pc2, mask1, mask2, epoch=100)
            _, metrics = self.loss_fn(z1, z2, match_indices, temp, mask1, mask2)

            match_correct += metrics['match_correct']
            match_total += metrics['match_total']

        return match_correct / match_total if match_total > 0 else 0

    def missing_cells(self, frac):
        def fn(pc1, pc2, m1, m2):
            pc1, m1 = pc1.clone(), m1.clone()
            for b in range(pc1.shape[0]):
                n = int(m1[b].sum())
                n_rm = int(n * frac)
                if n_rm > 0 and n - n_rm >= 4:
                    valid = torch.where(m1[b] > 0)[0]
                    rm = valid[torch.randperm(n)[:n_rm]]
                    m1[b, rm] = 0
                    pc1[b, rm] = 0
            return pc1, pc2, m1, m2
        return fn

    def coord_noise(self, scale):
        def fn(pc1, pc2, m1, m2):
            pc1, pc2 = pc1.clone(), pc2.clone()
            for b in range(pc1.shape[0]):
                for pc, m in [(pc1, m1), (pc2, m2)]:
                    n = int(m[b].sum())
                    if n > 1:
                        pts = pc[b, :n].numpy()
                        nn = NearestNeighbors(n_neighbors=2).fit(pts)
                        d, _ = nn.kneighbors(pts)
                        nn_dist = d[:, 1].mean()
                        pc[b, :n] += torch.randn(n, 3) * (scale * nn_dist)
            return pc1, pc2, m1, m2
        return fn

    def no_perturb(self):
        return lambda p1, p2, m1, m2: (p1, p2, m1, m2)

    def run_sweeps(self, dataloader):
        """Run missing cells and noise sweeps"""
        missing = {}
        for frac in [0.0, 0.1, 0.2, 0.3, 0.4]:
            fn = self.no_perturb() if frac == 0 else self.missing_cells(frac)
            acc = self.evaluate_perturbed(dataloader, fn, f"Missing {int(frac*100)}%")
            missing[frac] = acc
            print(f"  {int(frac*100)}% missing: {acc*100:.1f}%")

        noise = {}
        for scale in [0.0, 0.1, 0.2, 0.3, 0.5]:
            fn = self.no_perturb() if scale == 0 else self.coord_noise(scale)
            acc = self.evaluate_perturbed(dataloader, fn, f"Noise {scale}x")
            noise[scale] = acc
            print(f"  {scale}x noise: {acc*100:.1f}%")

        return {'missing': missing, 'noise': noise}


# =============================================================================
# ABLATION STUDIES - WITH ACTUAL TRAINING
# =============================================================================
class AblationEvaluator:
    """Run ablation studies by training variant models"""

    def __init__(self, config, train_data, eval_data):
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data
        self.ablation_epochs = 15  # Quick training for ablations
        self.ablation_samples = 2000  # Subset for faster training

    def create_loaders(self, train_data, eval_data, n_train=2000, n_eval=500):
        """Create train and eval dataloaders for ablation"""
        train_ds = SparseEmbryoDataset(
            train_data, stage_limit=self.config.STAGE_LIMIT,
            min_cells=self.config.MIN_CELLS, max_cells=self.config.MAX_CELLS,
            augment=True, num_rotations=3
        )
        if len(train_ds) > n_train:
            indices = np.random.choice(len(train_ds), n_train, replace=False)
            train_ds.pairs = [train_ds.pairs[i] for i in indices]

        eval_ds = SparseEmbryoDataset(
            eval_data, stage_limit=self.config.STAGE_LIMIT,
            min_cells=self.config.MIN_CELLS, max_cells=self.config.MAX_CELLS,
            augment=False, num_rotations=1
        )
        if len(eval_ds) > n_eval:
            indices = np.random.choice(len(eval_ds), n_eval, replace=False)
            eval_ds.pairs = [eval_ds.pairs[i] for i in indices]

        train_loader = DataLoader(train_ds, batch_size=self.config.BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_fn_with_padding, num_workers=0)
        eval_loader = DataLoader(eval_ds, batch_size=self.config.BATCH_SIZE,
                                 shuffle=False, collate_fn=collate_fn_with_padding, num_workers=0)
        return train_loader, eval_loader

    def train_and_evaluate(self, model_config, train_loader, eval_loader, name):
        """Train a model variant and evaluate it"""
        print(f"\n  Training: {name}")

        # Create model
        use_uncertainty = model_config.get('use_uncertainty', True)
        model = EnhancedTwinAttentionEncoder(
            embed_dim=model_config.get('embed_dim', 128),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 6),
            dropout=0.1,
            use_sparse_features=model_config.get('use_sparse_features', True),
            use_uncertainty=use_uncertainty,
            use_learnable_no_match=model_config.get('use_learnable_no_match', True)
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        loss_fn = TwinAttentionMatchingLoss(use_uncertainty=use_uncertainty)

        # Training loop
        best_acc = 0
        for epoch in range(self.ablation_epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                pc1, pc2, mask1, mask2, match_indices, _ = batch
                pc1, pc2 = pc1.to(device), pc2.to(device)
                mask1, mask2 = mask1.to(device), mask2.to(device)
                match_indices = match_indices.to(device)

                optimizer.zero_grad()
                z1, z2, temp = model(pc1, pc2, mask1, mask2, epoch=epoch)
                loss, _ = loss_fn(z1, z2, match_indices, temp, mask1, mask2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Quick validation every few epochs
            if (epoch + 1) % 5 == 0 or epoch == self.ablation_epochs - 1:
                model.eval()
                match_correct, match_total = 0, 0
                with torch.no_grad():
                    for batch in eval_loader:
                        pc1, pc2, mask1, mask2, match_indices, _ = batch
                        pc1, pc2 = pc1.to(device), pc2.to(device)
                        mask1, mask2 = mask1.to(device), mask2.to(device)
                        match_indices = match_indices.to(device)

                        z1, z2, temp = model(pc1, pc2, mask1, mask2, epoch=100)
                        _, metrics = loss_fn(z1, z2, match_indices, temp, mask1, mask2)
                        match_correct += metrics['match_correct']
                        match_total += metrics['match_total']

                acc = match_correct / match_total if match_total > 0 else 0
                best_acc = max(best_acc, acc)
                print(f"    Epoch {epoch+1}/{self.ablation_epochs}: loss={epoch_loss/len(train_loader):.3f}, acc={acc*100:.1f}%")

        print(f"    Final accuracy: {best_acc*100:.1f}%")
        return best_acc

    def run_architecture_ablations(self):
        """Test different architecture configurations"""
        print("\n--- Architecture Ablations (Training each variant) ---")
        train_loader, eval_loader = self.create_loaders(
            self.train_data, self.eval_data,
            n_train=self.ablation_samples, n_eval=500
        )

        results = {}

        # Architecture variants - vary one thing at a time
        configs = [
            ('Full (6L, 8H, 128D)', {'num_layers': 6, 'num_heads': 8, 'embed_dim': 128}),
            ('Shallow (2L)', {'num_layers': 2, 'num_heads': 8, 'embed_dim': 128}),
            ('Medium (4L)', {'num_layers': 4, 'num_heads': 8, 'embed_dim': 128}),
            ('Few heads (2H)', {'num_layers': 6, 'num_heads': 2, 'embed_dim': 128}),
            ('Small embed (64D)', {'num_layers': 6, 'num_heads': 8, 'embed_dim': 64}),
        ]

        for name, cfg in configs:
            results[name] = self.train_and_evaluate(cfg, train_loader, eval_loader, name)

        return results

    def run_feature_ablations(self):
        """Test contribution of different features"""
        print("\n--- Feature Ablations (Training each variant) ---")
        train_loader, eval_loader = self.create_loaders(
            self.train_data, self.eval_data,
            n_train=self.ablation_samples, n_eval=500
        )

        results = {}

        # Feature variants
        configs = [
            ('All features', {'use_sparse_features': True, 'use_uncertainty': True, 'use_learnable_no_match': True}),
            ('No sparse features', {'use_sparse_features': False, 'use_uncertainty': True, 'use_learnable_no_match': True}),
            ('No uncertainty', {'use_sparse_features': True, 'use_uncertainty': False, 'use_learnable_no_match': True}),
            ('No no-match token', {'use_sparse_features': True, 'use_uncertainty': True, 'use_learnable_no_match': False}),
        ]

        for name, cfg in configs:
            results[name] = self.train_and_evaluate(cfg, train_loader, eval_loader, name)

        return results


# =============================================================================
# ERROR ANALYSIS - FIXED
# =============================================================================
class ErrorAnalyzer:
    def analyze(self, results):
        """Categorize errors from KNN results"""
        errors = []
        # Use unique predictions for cleaner analysis
        if 'unique_predictions' in results and 'unique_labels' in results:
            for pred, true in zip(results['unique_predictions'], results['unique_labels']):
                if pred != true:
                    errors.append((pred, true))
        else:
            for i in range(len(results['test_labels'])):
                if results['predictions'][i] != results['test_labels'][i]:
                    errors.append((results['predictions'][i], results['test_labels'][i]))

        cats = {'sibling': 0, 'same_sublineage': 0, 'same_founder': 0, 'diff_founder': 0, 'unknown': 0}

        for pred, true in errors:
            p_founder = get_founder(pred)
            t_founder = get_founder(true)

            if p_founder == 'UNKNOWN' or t_founder == 'UNKNOWN':
                cats['unknown'] += 1
            elif are_siblings(pred, true):
                cats['sibling'] += 1
            elif same_sublineage(pred, true):
                cats['same_sublineage'] += 1
            elif p_founder == t_founder:
                cats['same_founder'] += 1
            else:
                cats['diff_founder'] += 1

        total = len(errors)
        pcts = {k: (v / total * 100 if total > 0 else 0) for k, v in cats.items()}

        return {'counts': cats, 'percentages': pcts, 'total': total}


# =============================================================================
# FIGURE GENERATION - IMPROVED AESTHETICS
# =============================================================================
class FigureGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style for publication-quality figures
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
        })

    def save(self, fig, name):
        fig.savefig(os.path.join(self.output_dir, name), dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  Saved: {name}")

    def fig_matching_accuracy(self, metrics, name="fig_matching_accuracy.png"):
        """Main matching performance - bar chart"""
        fig, ax = plt.subplots(figsize=(8, 6))

        cats = ['Match\nAccuracy', 'Outlier\nAccuracy', 'Overall\nAccuracy']
        vals = [metrics['match_accuracy'] * 100,
                metrics['outlier_accuracy'] * 100,
                metrics['overall_accuracy'] * 100]
        counts = [metrics['match_total'], metrics['outlier_total'],
                  metrics['match_total'] + metrics['outlier_total']]

        colors = ['#2ecc71', '#f39c12', '#3498db']
        bars = ax.bar(cats, vals, color=colors, edgecolor='white', linewidth=2)

        ax.axhline(90, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2, label='90% threshold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('Twin Attention Matching Performance', fontweight='bold', pad=15)
        ax.legend(loc='lower right')

        for bar, v, n in zip(bars, vals, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 8,
                   f'n={n:,}', ha='center', fontsize=9, color='white')

        ax.set_ylim(0, 105)
        self.save(fig, name)

    def fig_knn_accuracy(self, results, name="fig_knn_accuracy.png"):
        """KNN identification accuracy with CI"""
        fig, ax = plt.subplots(figsize=(6, 6))

        acc = results['overall_accuracy'] * 100
        ci = results['ci']

        bar = ax.bar(['KNN Cell\nIdentification'], [acc], color='#3498db',
                    edgecolor='white', linewidth=2, width=0.5)
        ax.errorbar(0, acc, yerr=[[acc - ci[0]*100], [ci[1]*100 - acc]],
                   fmt='none', color='#2c3e50', capsize=10, linewidth=3, capthick=2)

        ax.axhline(90, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title(f'KNN Cell Identification (k={30})\n{results["unique_cells"]} unique cells', fontweight='bold', pad=15)

        ax.text(0, acc + 5, f'{acc:.1f}%', ha='center', fontsize=16, fontweight='bold', color='#2c3e50')

        self.save(fig, name)

    def fig_accuracy_by_size(self, results, name="fig_accuracy_by_size.png"):
        """Accuracy by neighborhood size"""
        fig, ax = plt.subplots(figsize=(8, 6))

        sizes = ['Sparse\n(5-10 cells)', 'Medium\n(11-15 cells)', 'Dense\n(16-20 cells)']
        keys = ['sparse_5_10', 'medium_11_15', 'dense_16_20']

        accs = [results['by_size'].get(k, {}).get('accuracy', 0) * 100 for k in keys]
        counts = [results['by_size'].get(k, {}).get('count', 0) for k in keys]

        colors = ['#85c1e9', '#5dade2', '#2980b9']
        bars = ax.bar(sizes, accs, color=colors, edgecolor='white', linewidth=2)

        ax.axhline(90, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2, label='90% threshold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_ylim(50, 100)
        ax.set_title('Identification Accuracy by Neighborhood Density', fontweight='bold', pad=15)
        ax.legend(loc='lower right')

        for bar, a, n in zip(bars, accs, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                   f'{a:.1f}%', ha='center', fontsize=11, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                   f'n={n:,}', ha='center', fontsize=9, color='white')

        self.save(fig, name)

    def fig_hierarchical(self, results, name="fig_hierarchical.png"):
        """Hierarchical accuracy - shows error granularity"""
        fig, ax = plt.subplots(figsize=(9, 6))

        levels = ['Exact\nCell ID', 'Same\nSub-lineage', 'Same\nFounder', 'Binary\n(AB vs other)']
        hier = results['hierarchical']
        accs = [hier['exact'] * 100, hier['sublineage'] * 100,
                hier['founder'] * 100, hier['binary'] * 100]

        colors = ['#1a5276', '#2471a3', '#5499c7', '#85c1e9']
        bars = ax.bar(levels, accs, color=colors, edgecolor='white', linewidth=2)

        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('Hierarchical Identification Accuracy\n(Even errors are biologically close)', fontweight='bold', pad=15)

        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f'{a:.1f}%', ha='center', fontsize=12, fontweight='bold')

        # Add annotation
        ax.annotate('', xy=(3, accs[3]), xytext=(0, accs[0]),
                   arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

        self.save(fig, name)

    def fig_baseline_comparison(self, baselines, our_acc, name="fig_baseline_comparison.png"):
        """Baseline method comparison - shows our method is better"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['ICP', 'CPD', 'Hungarian', 'Twin Attention\n(Ours)']
        accs = [baselines['icp'] * 100, baselines['cpd'] * 100,
                baselines['hungarian'] * 100, our_acc * 100]

        colors = ['#bdc3c7', '#95a5a6', '#7f8c8d', '#3498db']
        bars = ax.bar(methods, accs, color=colors, edgecolor='white', linewidth=2)

        ax.axhline(50, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2, label='Random baseline')
        ax.set_ylabel('Match Accuracy (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('Comparison with Baseline Methods', fontweight='bold', pad=15)
        ax.legend(loc='upper left')

        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f'{a:.1f}%', ha='center', fontsize=12, fontweight='bold')

        # Highlight our method
        bars[-1].set_edgecolor('#2980b9')
        bars[-1].set_linewidth(3)

        self.save(fig, name)

    def fig_robustness_missing(self, results, name="fig_robustness_missing.png"):
        """Missing cells robustness curve"""
        fig, ax = plt.subplots(figsize=(8, 6))

        fracs = sorted(results['missing'].keys())
        accs = [results['missing'][f] * 100 for f in fracs]

        ax.plot([f*100 for f in fracs], accs, 'o-', color='#3498db',
               linewidth=3, markersize=12, markerfacecolor='white', markeredgewidth=3)

        ax.fill_between([f*100 for f in fracs], accs, alpha=0.2, color='#3498db')
        ax.axhline(80, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2, label='80% threshold')

        ax.set_xlabel('Cells Removed (%)', fontweight='bold')
        ax.set_ylabel('Match Accuracy (%)', fontweight='bold')
        ax.set_title('Robustness to Missing Cells', fontweight='bold', pad=15)
        ax.set_ylim(50, 100)
        ax.set_xlim(-2, 42)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3, linestyle='-')

        for f, a in zip(fracs, accs):
            ax.annotate(f'{a:.1f}%', (f*100, a), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

        self.save(fig, name)

    def fig_robustness_noise(self, results, name="fig_robustness_noise.png"):
        """Coordinate noise robustness curve"""
        fig, ax = plt.subplots(figsize=(8, 6))

        scales = sorted(results['noise'].keys())
        accs = [results['noise'][s] * 100 for s in scales]

        ax.plot(scales, accs, 's-', color='#27ae60',
               linewidth=3, markersize=12, markerfacecolor='white', markeredgewidth=3)

        ax.fill_between(scales, accs, alpha=0.2, color='#27ae60')
        ax.axhline(80, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2, label='80% threshold')

        ax.set_xlabel('Noise Scale (× mean NN distance)', fontweight='bold')
        ax.set_ylabel('Match Accuracy (%)', fontweight='bold')
        ax.set_title('Robustness to Coordinate Noise', fontweight='bold', pad=15)
        ax.set_ylim(50, 100)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3, linestyle='-')

        for s, a in zip(scales, accs):
            ax.annotate(f'{a:.1f}%', (s, a), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

        self.save(fig, name)

    def fig_ablation_architecture(self, results, name="fig_ablation_architecture.png"):
        """Architecture ablation bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(results.keys())
        accs = [results[n] * 100 for n in names]

        # Color full model differently
        colors = ['#3498db' if i == 0 else '#95a5a6' for i in range(len(names))]

        bars = ax.barh(names, accs, color=colors, edgecolor='white', linewidth=2)

        ax.set_xlabel('Match Accuracy (%)', fontweight='bold')
        ax.set_title('Architecture Ablation Study', fontweight='bold', pad=15)
        ax.set_xlim(0, 100)

        for bar, a in zip(bars, accs):
            ax.text(a + 1, bar.get_y() + bar.get_height()/2,
                   f'{a:.1f}%', va='center', fontsize=10, fontweight='bold')

        ax.invert_yaxis()
        self.save(fig, name)

    def fig_ablation_features(self, results, name="fig_ablation_features.png"):
        """Feature ablation bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(results.keys())
        accs = [results[n] * 100 for n in names]

        colors = ['#3498db' if i == 0 else '#e67e22' for i in range(len(names))]
        bars = ax.barh(names, accs, color=colors, edgecolor='white', linewidth=2)

        ax.set_xlabel('Match Accuracy (%)', fontweight='bold')
        ax.set_title('Feature Ablation Study', fontweight='bold', pad=15)
        ax.set_xlim(0, 100)

        for bar, a in zip(bars, accs):
            ax.text(a + 1, bar.get_y() + bar.get_height()/2,
                   f'{a:.1f}%', va='center', fontsize=10, fontweight='bold')

        ax.invert_yaxis()
        self.save(fig, name)

    def fig_embedding_tsne(self, embeddings, labels, name="fig_embedding_tsne.png"):
        """t-SNE visualization of embeddings by lineage"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Subsample for speed
        max_pts = 2000
        if len(embeddings) > max_pts:
            idx = np.random.choice(len(embeddings), max_pts, replace=False)
            emb = embeddings[idx]
            labs = [labels[i] for i in idx]
        else:
            emb, labs = embeddings, labels

        print("  Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(emb)//4),
                   random_state=42, max_iter=1000, init='pca')
        emb_2d = tsne.fit_transform(emb)

        founders = [get_founder(l) for l in labs]
        unique = sorted(set(f for f in founders if f != 'UNKNOWN'))

        cmap = plt.cm.get_cmap('tab10', len(unique))
        colors = {f: cmap(i) for i, f in enumerate(unique)}

        for f in unique:
            mask = [fo == f for fo in founders]
            pts = emb_2d[np.array(mask)]
            ax.scatter(pts[:, 0], pts[:, 1], c=[colors[f]], s=15, alpha=0.6, label=f)

        ax.legend(title='Founder Lineage', loc='upper right', framealpha=0.9)
        ax.set_xlabel('t-SNE Component 1', fontweight='bold')
        ax.set_ylabel('t-SNE Component 2', fontweight='bold')
        ax.set_title('Cell Embedding Space by Lineage', fontweight='bold', pad=15)

        self.save(fig, name)

    def fig_error_distribution(self, errors, name="fig_error_distribution.png"):
        """Error distribution pie chart"""
        fig, ax = plt.subplots(figsize=(9, 7))

        labels_map = {
            'sibling': 'Sibling\nConfusion',
            'same_sublineage': 'Same\nSub-lineage',
            'same_founder': 'Same\nFounder',
            'diff_founder': 'Different\nFounder'
        }

        colors = ['#e74c3c', '#f39c12', '#3498db', '#1abc9c']
        labels, sizes = [], []

        for i, (k, lbl) in enumerate(labels_map.items()):
            count = errors['counts'].get(k, 0)
            if count > 0:
                labels.append(f'{lbl}\n(n={count})')
                sizes.append(count)

        if sizes:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(sizes)],
                                              autopct='%1.1f%%', startangle=90, pctdistance=0.75,
                                              explode=[0.02]*len(sizes))
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')
            ax.set_title(f'Error Type Distribution\n(Total errors: {errors["total"]})', fontweight='bold', pad=15)
        else:
            ax.text(0.5, 0.5, 'No errors to categorize', ha='center', va='center', fontsize=14)

        self.save(fig, name)

    def fig_confidence_histogram(self, results, name="fig_confidence_histogram.png"):
        """Confidence distribution for correct vs incorrect predictions"""
        fig, ax = plt.subplots(figsize=(9, 6))

        correct_conf = [results['confidences'][i] for i in range(len(results['test_labels']))
                       if results['predictions'][i] == results['test_labels'][i]]
        wrong_conf = [results['confidences'][i] for i in range(len(results['test_labels']))
                     if results['predictions'][i] != results['test_labels'][i]]

        bins = np.linspace(0, 1, 21)
        ax.hist(correct_conf, bins, alpha=0.7, label=f'Correct (n={len(correct_conf):,})',
               color='#27ae60', edgecolor='white', linewidth=1)
        ax.hist(wrong_conf, bins, alpha=0.7, label=f'Incorrect (n={len(wrong_conf):,})',
               color='#e74c3c', edgecolor='white', linewidth=1)

        ax.set_xlabel('KNN Confidence (fraction of agreeing neighbors)', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Prediction Confidence Distribution', fontweight='bold', pad=15)
        ax.legend(framealpha=0.9)

        self.save(fig, name)


# =============================================================================
# MAIN EVALUATION RUNNER
# =============================================================================
class EvaluationRunner:
    def __init__(self, config):
        self.config = config
        self.results = {}
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.FIGURE_DIR, exist_ok=True)
        self.fig_gen = FigureGenerator(config.FIGURE_DIR)

    def run(self):
        print("\n" + "="*70)
        print("TWIN ATTENTION MODEL - COMPREHENSIVE EVALUATION")
        print("="*70)

        data = load_data(self.config)
        model = load_model(self.config)

        train_data = data.get('train')
        eval_data = data.get('eval') or train_data

        if not eval_data:
            print("ERROR: No data!")
            return

        # Create datasets
        print("\nCreating datasets...")
        if train_data:
            train_ds = SparseEmbryoDataset(train_data, stage_limit=self.config.STAGE_LIMIT,
                                           min_cells=self.config.MIN_CELLS, max_cells=self.config.MAX_CELLS,
                                           augment=False, num_rotations=1)
            train_loader = DataLoader(train_ds, batch_size=self.config.BATCH_SIZE,
                                      shuffle=False, collate_fn=collate_fn_with_padding, num_workers=0)

        eval_ds = SparseEmbryoDataset(eval_data, stage_limit=self.config.STAGE_LIMIT,
                                      min_cells=self.config.MIN_CELLS, max_cells=self.config.MAX_CELLS,
                                      augment=False, num_rotations=1)
        eval_loader = DataLoader(eval_ds, batch_size=self.config.BATCH_SIZE,
                                 shuffle=False, collate_fn=collate_fn_with_padding, num_workers=0)

        # =================================================================
        # SECTION 3.1: Core Performance
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.1: CORE PERFORMANCE")
        print("="*60)

        print("\n--- Direct Matching (Primary Result) ---")
        match_eval = MatchingEvaluator(model, self.config)
        match_results = match_eval.evaluate(eval_loader, "Direct Matching")

        print(f"\n  Match Accuracy: {match_results['match_accuracy']*100:.1f}% ({match_results['match_correct']}/{match_results['match_total']})")
        print(f"  Outlier Accuracy: {match_results['outlier_accuracy']*100:.1f}% ({match_results['outlier_correct']}/{match_results['outlier_total']})")
        print(f"  Overall Accuracy: {match_results['overall_accuracy']*100:.1f}%")
        print(f"  Loss: {match_results['loss']:.4f}")

        self.results['matching'] = match_results

        # Print by_size from matching results
        print(f"\n  By neighborhood size (direct matching):")
        for k, v in match_results['by_size'].items():
            print(f"    {k}: {v['accuracy']*100:.1f}% (n={v['count']})")

        # KNN Identification
        print("\n--- KNN Cell Identification ---")
        knn_eval = KNNEvaluator(model, self.config)
        knn_results = knn_eval.evaluate(train_loader, eval_loader)

        print(f"\n  KNN Accuracy: {knn_results['overall_accuracy']*100:.1f}% (on {knn_results['unique_cells']} unique cells)")
        print(f"    95% CI: [{knn_results['ci'][0]*100:.1f}%, {knn_results['ci'][1]*100:.1f}%]")
        print(f"\n  Hierarchical accuracy (KNN):")
        for k, v in knn_results['hierarchical'].items():
            print(f"    {k}: {v*100:.1f}%")

        self.results['knn'] = knn_results

        # Generate Section 3.1 figures
        print("\nGenerating Section 3.1 figures...")
        self.fig_gen.fig_matching_accuracy(match_results)
        self.fig_gen.fig_knn_accuracy(knn_results)
        self.fig_gen.fig_accuracy_by_size(match_results)  # Use matching results
        self.fig_gen.fig_hierarchical(knn_results)

        # =================================================================
        # SECTION 3.2: Baselines
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.2: BASELINE COMPARISON")
        print("="*60)

        baseline_eval = BaselineEvaluator(self.config)
        baselines = {}
        for method in ['icp', 'cpd', 'hungarian']:
            acc = baseline_eval.evaluate(eval_loader, method)
            baselines[method] = acc
            print(f"  {method.upper()}: {acc*100:.1f}%")

        self.results['baselines'] = baselines

        print("\nGenerating Section 3.2 figure...")
        self.fig_gen.fig_baseline_comparison(baselines, match_results['match_accuracy'])

        # =================================================================
        # SECTION 3.3: Robustness
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.3: ROBUSTNESS")
        print("="*60)

        robust_eval = RobustnessEvaluator(model, self.config)
        robust_results = robust_eval.run_sweeps(eval_loader)
        self.results['robustness'] = robust_results

        print("\nGenerating Section 3.3 figures...")
        self.fig_gen.fig_robustness_missing(robust_results)
        self.fig_gen.fig_robustness_noise(robust_results)

        # =================================================================
        # SECTION 3.4 & 3.5: Ablations
        # =================================================================
        print("\n" + "="*60)
        print("SECTIONS 3.4-3.5: ABLATION STUDIES")
        print("="*60)

        ablation_eval = AblationEvaluator(self.config, train_data, eval_data)

        print("\nRunning architecture ablations...")
        arch_ablations = ablation_eval.run_architecture_ablations()
        self.results['arch_ablations'] = arch_ablations

        print("\nRunning feature ablations...")
        feat_ablations = ablation_eval.run_feature_ablations()
        self.results['feat_ablations'] = feat_ablations

        print("\nGenerating ablation figures...")
        self.fig_gen.fig_ablation_architecture(arch_ablations)
        self.fig_gen.fig_ablation_features(feat_ablations)

        # =================================================================
        # SECTION 3.6: Embedding & Error Analysis
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.6: EMBEDDING & ERROR ANALYSIS")
        print("="*60)

        error_analyzer = ErrorAnalyzer()
        errors = error_analyzer.analyze(knn_results)
        print(f"\n  Total errors: {errors['total']}")
        for k, v in errors['percentages'].items():
            if v > 0:
                print(f"    {k}: {v:.1f}%")

        self.results['errors'] = errors

        print("\nGenerating Section 3.6 figures...")
        self.fig_gen.fig_embedding_tsne(knn_results['test_embeddings'], knn_results['test_labels'])
        self.fig_gen.fig_error_distribution(errors)
        self.fig_gen.fig_confidence_histogram(knn_results)

        # =================================================================
        # SAVE RESULTS
        # =================================================================
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

        # Save pickle
        save_results = {k: v for k, v in self.results.items()}
        if 'knn' in save_results:
            save_results['knn'] = {k: v for k, v in save_results['knn'].items()
                                   if k not in ['test_embeddings', 'predictions', 'confidences', 'test_labels']}
        if 'matching' in save_results:
            save_results['matching'] = {k: v for k, v in save_results['matching'].items()
                                        if k != 'detailed_results'}

        with open(os.path.join(self.config.OUTPUT_DIR, 'evaluation_results.pkl'), 'wb') as f:
            pickle.dump(save_results, f)

        # Save summary text
        with open(os.path.join(self.config.OUTPUT_DIR, 'evaluation_summary.txt'), 'w') as f:
            f.write("TWIN ATTENTION MODEL - EVALUATION SUMMARY\n")
            f.write("="*50 + "\n\n")

            f.write("SECTION 3.1: CORE PERFORMANCE\n")
            f.write("-"*30 + "\n")
            f.write(f"Direct Matching Accuracy: {match_results['match_accuracy']*100:.1f}%\n")
            f.write(f"Outlier Detection Accuracy: {match_results['outlier_accuracy']*100:.1f}%\n")
            f.write(f"Overall Accuracy: {match_results['overall_accuracy']*100:.1f}%\n")
            f.write(f"KNN Identification Accuracy: {knn_results['overall_accuracy']*100:.1f}%\n")
            f.write(f"  95% CI: [{knn_results['ci'][0]*100:.1f}%, {knn_results['ci'][1]*100:.1f}%]\n\n")

            f.write("SECTION 3.2: BASELINE COMPARISON\n")
            f.write("-"*30 + "\n")
            for m, a in baselines.items():
                f.write(f"{m.upper()}: {a*100:.1f}%\n")
            f.write(f"TWIN ATTENTION (Ours): {match_results['match_accuracy']*100:.1f}%\n\n")

            f.write("SECTION 3.3: ROBUSTNESS\n")
            f.write("-"*30 + "\n")
            f.write("Missing cells:\n")
            for frac, acc in robust_results['missing'].items():
                f.write(f"  {int(frac*100)}%: {acc*100:.1f}%\n")
            f.write("Coordinate noise:\n")
            for scale, acc in robust_results['noise'].items():
                f.write(f"  {scale}x: {acc*100:.1f}%\n")

        print(f"\nResults saved to: {self.config.OUTPUT_DIR}")
        print(f"Figures saved to: {self.config.FIGURE_DIR}")

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)

        return self.results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--eval-data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str, default='evaluation_results')
    args = parser.parse_args()

    config = EvalConfig()
    if args.train_data:
        config.TRAIN_DATA_PATH = args.train_data
    if args.eval_data:
        config.EVAL_DATA_PATH = args.eval_data
    if args.model:
        config.MODEL_PATH = args.model
    config.OUTPUT_DIR = args.output
    config.FIGURE_DIR = os.path.join(args.output, 'figures')

    runner = EvaluationRunner(config)
    return runner.run()


if __name__ == "__main__":
    main()
