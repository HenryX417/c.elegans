"""
Comprehensive Evaluation Suite for Twin Attention Cell Identification Model
Fixed version - uses exact same metric computation as training code
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

    N_BOOTSTRAP = 1000


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


def load_model(config):
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
def bootstrap_ci(values, n=1000):
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
    This ensures metrics match what was reported during training.
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

        # For detailed analysis
        all_results = []

        for batch in tqdm(dataloader, desc=desc):
            pc1, pc2, mask1, mask2, match_indices, info_list = batch
            pc1 = pc1.to(device)
            pc2 = pc2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            match_indices = match_indices.to(device)

            # Forward pass - exactly as in training
            z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch=100)

            # Compute loss and metrics using the SAME loss function as training
            loss, metrics = self.loss_fn(z1, z2, match_indices, temperature, mask1, mask2)

            total_loss += loss.item()
            match_correct += metrics['match_correct']
            match_total += metrics['match_total']
            outlier_correct += metrics['outlier_correct']
            outlier_total += metrics['outlier_total']
            batch_count += 1

            # Store detailed results for later analysis
            if self.model.use_uncertainty:
                z1_mean, _ = z1
                z2_mean, _ = z2
                sims = torch.bmm(z1_mean, z2_mean.transpose(1, 2)) / temperature
            else:
                sims = torch.bmm(z1, z2.transpose(1, 2)) / temperature

            pred_indices = sims.argmax(dim=-1)
            B, N1, N2 = sims.shape  # N2 includes no-match token

            for b in range(B):
                info = info_list[b]
                n_valid = int(mask1[b].sum().item())

                for i in range(n_valid):
                    target = match_indices[b, i].item()
                    pred = pred_indices[b, i].item()
                    is_match = target < N2 - 1  # Same check as loss function

                    all_results.append({
                        'is_match': is_match,
                        'target': target,
                        'pred': pred,
                        'correct': pred == target,
                        'n_cells': n_valid,
                        'cell_id': info['cells1_ids'][i] if i < len(info['cells1_ids']) else None,
                        'confidence': torch.softmax(sims[b, i], dim=0).max().item()
                    })

        # Compute final metrics
        results = {
            'loss': total_loss / batch_count if batch_count > 0 else 0,
            'match_accuracy': match_correct / match_total if match_total > 0 else 0,
            'match_correct': match_correct,
            'match_total': match_total,
            'outlier_accuracy': outlier_correct / outlier_total if outlier_total > 0 else 0,
            'outlier_correct': outlier_correct,
            'outlier_total': outlier_total,
            'overall_accuracy': (match_correct + outlier_correct) / (match_total + outlier_total) if (match_total + outlier_total) > 0 else 0,
            'detailed_results': all_results
        }

        return results


# =============================================================================
# KNN-BASED CELL IDENTIFICATION (Section 2.8)
# =============================================================================
class KNNIdentifier:
    """KNN-based cell identification as described in paper Section 2.8"""

    def __init__(self, k=30):
        self.k = k
        self.embeddings = None
        self.labels = None

    def fit(self, embeddings, labels):
        """Build index from training embeddings"""
        self.embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        self.labels = labels
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(embeddings)), metric='cosine')
        self.nn.fit(self.embeddings)
        print(f"  KNN index: {len(embeddings)} embeddings, k={self.k}")

    def predict(self, queries):
        """Predict cell IDs via majority vote"""
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        _, indices = self.nn.kneighbors(queries_norm)

        predictions = []
        confidences = []
        for idx_list in indices:
            neighbor_labels = [self.labels[i] for i in idx_list]
            counts = Counter(neighbor_labels)
            top, count = counts.most_common(1)[0]
            predictions.append(top)
            confidences.append(count / len(idx_list))

        return predictions, confidences


class KNNEvaluator:
    """Evaluate KNN-based cell identification"""

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
        """Full KNN evaluation"""
        print("\nBuilding KNN index from training data...")
        train_emb, train_labels, _ = self.extract_embeddings(train_loader, "Train embeddings")
        self.knn.fit(train_emb, train_labels)

        print("Evaluating on test data...")
        test_emb, test_labels, test_meta = self.extract_embeddings(test_loader, "Test embeddings")
        predictions, confidences = self.knn.predict(test_emb)

        # Compute metrics
        correct = [1 if p == t else 0 for p, t in zip(predictions, test_labels)]
        mean_acc, ci_low, ci_high = bootstrap_ci(correct, self.config.N_BOOTSTRAP)

        # By neighborhood size
        size_results = {}
        for name, (lo, hi) in [('sparse_5_10', (5, 10)), ('medium_11_15', (11, 15)), ('dense_16_20', (16, 20))]:
            mask = [lo <= m['n_cells'] <= hi for m in test_meta]
            if sum(mask) > 0:
                bin_correct = [c for c, m in zip(correct, mask) if m]
                acc, _, _ = bootstrap_ci(bin_correct, self.config.N_BOOTSTRAP)
                size_results[name] = {'accuracy': acc, 'count': sum(mask)}

        # Hierarchical accuracy
        hier = {'exact': 0, 'sublineage': 0, 'founder': 0, 'binary': 0}
        for pred, true in zip(predictions, test_labels):
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

        n = len(test_labels)
        hier = {k: v / n for k, v in hier.items()}

        return {
            'overall_accuracy': mean_acc,
            'ci': (ci_low, ci_high),
            'total': n,
            'by_size': size_results,
            'hierarchical': hier,
            'test_embeddings': test_emb,
            'test_labels': test_labels,
            'predictions': predictions,
            'confidences': confidences
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
            src = src @ (Vt.T @ U.T).T
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
        """Evaluate with perturbation - reports match accuracy"""
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
# ERROR ANALYSIS
# =============================================================================
class ErrorAnalyzer:
    def analyze(self, results):
        """Categorize errors from KNN results"""
        errors = [(r['predictions'][i], r['test_labels'][i])
                  for i in range(len(r['test_labels']))
                  if r['predictions'][i] != r['test_labels'][i]]
        # Fix: use proper variable
        errors = []
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
# FIGURE GENERATION - Individual figures
# =============================================================================
class FigureGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13})

    def save(self, fig, name):
        fig.savefig(os.path.join(self.output_dir, name), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {name}")

    def fig_matching_accuracy(self, metrics, name="fig_matching_accuracy.png"):
        """Match and outlier accuracy - the main training metrics"""
        fig, ax = plt.subplots(figsize=(7, 5))

        cats = ['Match\nAccuracy', 'Outlier\nAccuracy', 'Overall\nAccuracy']
        vals = [metrics['match_accuracy'] * 100,
                metrics['outlier_accuracy'] * 100,
                metrics['overall_accuracy'] * 100]
        counts = [metrics['match_total'], metrics['outlier_total'],
                  metrics['match_total'] + metrics['outlier_total']]

        colors = ['#4CAF50', '#FF9800', '#2196F3']
        bars = ax.bar(cats, vals, color=colors)

        ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% target')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Direct Matching Performance\n(Same as Training Metrics)')
        ax.legend()

        for bar, v, n in zip(bars, vals, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{v:.1f}%\n(n={n})', ha='center', fontsize=10)

        self.save(fig, name)

    def fig_knn_accuracy(self, results, name="fig_knn_accuracy.png"):
        """KNN identification accuracy"""
        fig, ax = plt.subplots(figsize=(5, 5))

        acc = results['overall_accuracy'] * 100
        ci = results['ci']

        ax.bar(['KNN\nIdentification'], [acc], color='#2196F3', width=0.5)
        ax.errorbar(0, acc, yerr=[[acc - ci[0]*100], [ci[1]*100 - acc]],
                   fmt='none', color='black', capsize=8, linewidth=2)

        ax.axhline(90, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.set_title(f'KNN Cell Identification (k={30})\nn={results["total"]}')
        ax.text(0, acc + 3, f'{acc:.1f}%', ha='center', fontsize=14, fontweight='bold')

        self.save(fig, name)

    def fig_accuracy_by_size(self, results, name="fig_accuracy_by_size.png"):
        """Accuracy by neighborhood size"""
        fig, ax = plt.subplots(figsize=(7, 5))

        sizes = ['Sparse\n(5-10)', 'Medium\n(11-15)', 'Dense\n(16-20)']
        keys = ['sparse_5_10', 'medium_11_15', 'dense_16_20']

        accs = [results['by_size'].get(k, {}).get('accuracy', 0) * 100 for k in keys]
        counts = [results['by_size'].get(k, {}).get('count', 0) for k in keys]

        bars = ax.bar(sizes, accs, color=['#81D4FA', '#29B6F6', '#0288D1'])
        ax.axhline(90, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(80, 100)
        ax.set_title('KNN Accuracy by Neighborhood Size')

        for bar, a, n in zip(bars, accs, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{a:.1f}%\n(n={n})', ha='center', fontsize=10)

        self.save(fig, name)

    def fig_hierarchical(self, results, name="fig_hierarchical.png"):
        """Hierarchical accuracy"""
        fig, ax = plt.subplots(figsize=(8, 5))

        levels = ['Exact\nCell ID', 'Sub-\nlineage', 'Founder\nLineage', 'Binary\n(AB vs other)']
        hier = results['hierarchical']
        accs = [hier['exact'] * 100, hier['sublineage'] * 100,
                hier['founder'] * 100, hier['binary'] * 100]

        colors = ['#1565C0', '#1976D2', '#2196F3', '#64B5F6']
        bars = ax.bar(levels, accs, color=colors)

        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Hierarchical Identification Accuracy')

        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{a:.1f}%', ha='center', fontsize=11, fontweight='bold')

        self.save(fig, name)

    def fig_baseline_comparison(self, baselines, our_acc, name="fig_baseline_comparison.png"):
        """Baseline method comparison"""
        fig, ax = plt.subplots(figsize=(9, 5))

        methods = ['ICP', 'CPD', 'Hungarian', 'Twin Attention\n(Ours)']
        accs = [baselines['icp'] * 100, baselines['cpd'] * 100,
                baselines['hungarian'] * 100, our_acc * 100]

        colors = ['#BDBDBD', '#9E9E9E', '#757575', '#2196F3']
        bars = ax.bar(methods, accs, color=colors)

        ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylabel('Match Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Baseline Method Comparison')
        ax.legend()

        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{a:.1f}%', ha='center', fontsize=11, fontweight='bold')

        self.save(fig, name)

    def fig_robustness_missing(self, results, name="fig_robustness_missing.png"):
        """Missing cells robustness"""
        fig, ax = plt.subplots(figsize=(7, 5))

        fracs = sorted(results['missing'].keys())
        accs = [results['missing'][f] * 100 for f in fracs]

        ax.plot([f*100 for f in fracs], accs, 'o-', color='#2196F3',
               linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% threshold')

        ax.set_xlabel('Cells Removed (%)')
        ax.set_ylabel('Match Accuracy (%)')
        ax.set_title('Robustness to Missing Cells')
        ax.set_ylim(50, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

        for f, a in zip(fracs, accs):
            ax.annotate(f'{a:.1f}%', (f*100, a), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

        self.save(fig, name)

    def fig_robustness_noise(self, results, name="fig_robustness_noise.png"):
        """Coordinate noise robustness"""
        fig, ax = plt.subplots(figsize=(7, 5))

        scales = sorted(results['noise'].keys())
        accs = [results['noise'][s] * 100 for s in scales]

        ax.plot(scales, accs, 's-', color='#4CAF50',
               linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% threshold')

        ax.set_xlabel('Noise Scale (× mean NN distance)')
        ax.set_ylabel('Match Accuracy (%)')
        ax.set_title('Robustness to Coordinate Noise')
        ax.set_ylim(50, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

        for s, a in zip(scales, accs):
            ax.annotate(f'{a:.1f}%', (s, a), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

        self.save(fig, name)

    def fig_embedding_tsne(self, embeddings, labels, name="fig_embedding_tsne.png"):
        """t-SNE of embeddings"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Subsample
        max_pts = 3000
        if len(embeddings) > max_pts:
            idx = np.random.choice(len(embeddings), max_pts, replace=False)
            emb = embeddings[idx]
            labs = [labels[i] for i in idx]
        else:
            emb, labs = embeddings, labels

        print("  Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(emb)//4), random_state=42, max_iter=1000)
        emb_2d = tsne.fit_transform(emb)

        founders = [get_founder(l) for l in labs]
        unique = sorted(set(founders))
        cmap = plt.cm.get_cmap('tab10', len(unique))
        colors = {f: cmap(i) for i, f in enumerate(unique)}

        for f in unique:
            mask = [fo == f for fo in founders]
            pts = emb_2d[mask]
            if f != 'UNKNOWN':
                ax.scatter(pts[:, 0], pts[:, 1], c=[colors[f]], s=8, alpha=0.6, label=f)

        ax.legend(title='Founder', loc='upper right')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Embedding Space by Lineage')

        self.save(fig, name)

    def fig_error_distribution(self, errors, name="fig_error_distribution.png"):
        """Error distribution pie"""
        fig, ax = plt.subplots(figsize=(8, 6))

        labels_map = {
            'sibling': 'Sibling',
            'same_sublineage': 'Same Sub-lineage',
            'same_founder': 'Same Founder',
            'diff_founder': 'Different Founder'
        }

        labels, sizes, colors = [], [], ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (k, lbl) in enumerate(labels_map.items()):
            pct = errors['percentages'].get(k, 0)
            if pct > 0:
                labels.append(f'{lbl}\n(n={errors["counts"].get(k, 0)})')
                sizes.append(pct)

        if sizes:
            ax.pie(sizes, labels=labels, colors=colors[:len(sizes)],
                  autopct='%1.1f%%', startangle=90, pctdistance=0.75)
            ax.set_title(f'Error Distribution (n={errors["total"]})')
        else:
            ax.text(0.5, 0.5, 'No categorizable errors', ha='center', va='center')

        self.save(fig, name)

    def fig_confidence_histogram(self, results, name="fig_confidence_histogram.png"):
        """Confidence distribution"""
        fig, ax = plt.subplots(figsize=(8, 5))

        correct_conf = [results['confidences'][i] for i in range(len(results['test_labels']))
                       if results['predictions'][i] == results['test_labels'][i]]
        wrong_conf = [results['confidences'][i] for i in range(len(results['test_labels']))
                     if results['predictions'][i] != results['test_labels'][i]]

        bins = np.linspace(0, 1, 21)
        ax.hist(correct_conf, bins, alpha=0.7, label=f'Correct (n={len(correct_conf)})', color='#4CAF50')
        ax.hist(wrong_conf, bins, alpha=0.7, label=f'Wrong (n={len(wrong_conf)})', color='#F44336')

        ax.set_xlabel('KNN Confidence (% neighbors agreeing)')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Confidence Distribution')
        ax.legend()

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
        # SECTION 3.1: Core Performance - Direct Matching (same as training)
        # =================================================================
        print("\n" + "="*60)
        print("SECTION 3.1: CORE PERFORMANCE")
        print("="*60)

        print("\n--- Direct Matching (Training Metrics) ---")
        match_eval = MatchingEvaluator(model, self.config)
        match_results = match_eval.evaluate(eval_loader, "Direct Matching")

        print(f"\n  Match Accuracy: {match_results['match_accuracy']*100:.1f}% ({match_results['match_correct']}/{match_results['match_total']})")
        print(f"  Outlier Accuracy: {match_results['outlier_accuracy']*100:.1f}% ({match_results['outlier_correct']}/{match_results['outlier_total']})")
        print(f"  Overall Accuracy: {match_results['overall_accuracy']*100:.1f}%")
        print(f"  Loss: {match_results['loss']:.4f}")

        self.results['matching'] = match_results

        # KNN Identification
        print("\n--- KNN Cell Identification ---")
        knn_eval = KNNEvaluator(model, self.config)
        knn_results = knn_eval.evaluate(train_loader, eval_loader)

        print(f"\n  KNN Accuracy: {knn_results['overall_accuracy']*100:.1f}%")
        print(f"    95% CI: [{knn_results['ci'][0]*100:.1f}%, {knn_results['ci'][1]*100:.1f}%]")
        print(f"\n  By size:")
        for k, v in knn_results['by_size'].items():
            print(f"    {k}: {v['accuracy']*100:.1f}% (n={v['count']})")
        print(f"\n  Hierarchical:")
        for k, v in knn_results['hierarchical'].items():
            print(f"    {k}: {v*100:.1f}%")

        self.results['knn'] = knn_results

        # Generate Section 3.1 figures
        print("\nGenerating Section 3.1 figures...")
        self.fig_gen.fig_matching_accuracy(match_results)
        self.fig_gen.fig_knn_accuracy(knn_results)
        self.fig_gen.fig_accuracy_by_size(knn_results)
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
        # SAVE
        # =================================================================
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

        # Save pickle (without large arrays for smaller file)
        save_results = {k: v for k, v in self.results.items()}
        if 'knn' in save_results:
            save_results['knn'] = {k: v for k, v in save_results['knn'].items()
                                   if k not in ['test_embeddings', 'predictions', 'confidences', 'test_labels']}
        if 'matching' in save_results:
            save_results['matching'] = {k: v for k, v in save_results['matching'].items()
                                        if k != 'detailed_results'}

        with open(os.path.join(self.config.OUTPUT_DIR, 'evaluation_results.pkl'), 'wb') as f:
            pickle.dump(save_results, f)

        # Save summary
        with open(os.path.join(self.config.OUTPUT_DIR, 'evaluation_summary.txt'), 'w') as f:
            f.write("TWIN ATTENTION MODEL - EVALUATION SUMMARY\n")
            f.write("="*50 + "\n\n")

            f.write("DIRECT MATCHING (Training Metrics):\n")
            f.write(f"  Match Accuracy: {match_results['match_accuracy']*100:.1f}%\n")
            f.write(f"  Outlier Accuracy: {match_results['outlier_accuracy']*100:.1f}%\n")
            f.write(f"  Overall Accuracy: {match_results['overall_accuracy']*100:.1f}%\n\n")

            f.write("KNN IDENTIFICATION:\n")
            f.write(f"  Accuracy: {knn_results['overall_accuracy']*100:.1f}%\n")
            f.write(f"  95% CI: [{knn_results['ci'][0]*100:.1f}%, {knn_results['ci'][1]*100:.1f}%]\n\n")

            f.write("BASELINES:\n")
            for m, a in baselines.items():
                f.write(f"  {m.upper()}: {a*100:.1f}%\n")
            f.write(f"  OURS: {match_results['match_accuracy']*100:.1f}%\n\n")

            f.write("ROBUSTNESS (Match Accuracy):\n")
            f.write("  Missing cells:\n")
            for frac, acc in robust_results['missing'].items():
                f.write(f"    {int(frac*100)}%: {acc*100:.1f}%\n")
            f.write("  Noise:\n")
            for scale, acc in robust_results['noise'].items():
                f.write(f"    {scale}x: {acc*100:.1f}%\n")

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
