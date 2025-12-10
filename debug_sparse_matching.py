import sys
import os
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
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import traceback
from scipy.spatial import KDTree
from scipy.stats import zscore
warnings.filterwarnings('ignore')

if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == 'cpu':
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(1)
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(os.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("WARNING: Running on CPU. This will be very slow!")

class SparsePointFeatures(nn.Module):
    """Feature extraction specifically designed for sparse point clouds (5-20 points)"""
    
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
        
        # Point count awareness 
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
        
            if device.type == 'cpu':
                pts_np = valid_points.detach().cpu().numpy()
                diff = pts_np[:, None, :] - pts_np[None, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                k = min(3, n_valid - 1)
                nearest_dists = np.partition(dists, k-1, axis=1)[:, :k]
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
    """Twin Attention encoder with subset point cloud optimizations"""
    
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
        
        # Temperature with warm-up
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
        
        # Create combined mask - CPU optimized version
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
        
        # Temperature with warm-up
        temperature = self._get_temperature(epoch)
        
        return outputs[0], outputs[1], temperature
    
    def _get_temperature(self, epoch):
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

class TwinAttentionMatchingLoss(nn.Module):
    """Matching Loss"""
    
    def __init__(self, use_uncertainty=True, margin=0.2):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.margin = margin
        
    def forward(self, z1, z2, match_indices, temperature, mask1=None, mask2=None):
        if self.use_uncertainty:
            z1_mean, z1_logvar = z1
            z2_mean, z2_logvar = z2
            sim = torch.matmul(z1_mean, z2_mean.transpose(-1, -2))
        else:
            sim = torch.matmul(z1, z2.transpose(-1, -2))
        
        sim = sim / temperature
        
        B, N1, N2 = sim.shape
        
        losses = []
        n_matches = 0
        n_outliers = 0
        
        for b in range(B):
            for i in range(N1):
                if mask1 is not None and not mask1[b, i]:
                    continue
                    
                match_idx = match_indices[b, i]
                
                if match_idx < N2 - 1:  # Valid match (not no-match token)
                    # Negative log probability of correct match
                    log_probs = F.log_softmax(sim[b, i], dim=0)
                    loss = -log_probs[match_idx]
                    losses.append(loss)
                    n_matches += 1
                else:  # Outlier - should match to no-match token
                    log_probs = F.log_softmax(sim[b, i], dim=0)
                    loss = -log_probs[-1]  # Last position is no-match token
                    losses.append(loss * 0.5)  # Lower weight for outliers
                    n_outliers += 1
        
        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=sim.device)
        
        # Compute accuracies
        with torch.no_grad():
            pred_indices = sim.argmax(dim=2)
            
            match_correct = 0
            outlier_correct = 0
            
            for b in range(B):
                for i in range(N1):
                    if mask1 is not None and not mask1[b, i]:
                        continue
                        
                    pred = pred_indices[b, i]
                    target = match_indices[b, i]
                    
                    if target < N2 - 1:  # Valid match
                        if pred == target:
                            match_correct += 1
                    else:  # Outlier
                        if pred == N2 - 1:  # Correctly predicted as no-match
                            outlier_correct += 1
        
        metrics = {
            'loss': total_loss.item(),
            'match_correct': match_correct,
            'match_total': n_matches,
            'match_acc': match_correct / n_matches if n_matches > 0 else 0,
            'outlier_correct': outlier_correct,
            'outlier_total': n_outliers,
            'outlier_acc': outlier_correct / n_outliers if n_outliers > 0 else 0,
            'temperature': temperature.item()
        }
        
        return total_loss, metrics

class TemporalConsistencyLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, z1_t, z1_t1, parent_child_pairs):
        """Enforce parent embeddings are similar to children"""
        if not parent_child_pairs:
            return torch.tensor(0.0, device=z1_t.device)
        
        losses = []
        for parent_idx, child_idx in parent_child_pairs:
            parent_emb = z1_t[parent_idx]
            child_emb = z1_t1[child_idx]
            similarity = F.cosine_similarity(parent_emb, child_emb, dim=0)
            losses.append(1 - similarity)
        
        return self.weight * torch.stack(losses).mean()

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
        if len(coords) < 4:
            return np.random.choice(len(coords), min(n_target, len(coords)), replace=False)
        
        # Find principal axis
        mean = coords.mean(axis=0)
        _, _, vh = np.linalg.svd(coords - mean)
        principal = vh[0]
        
        # Project onto principal axis
        projections = (coords - mean) @ principal
        
        # Get extreme points
        n_poles = min(4, n_target // 2)
        anterior_idx = np.argsort(projections)[:n_poles]
        posterior_idx = np.argsort(projections)[-n_poles:]
        
        selected = np.concatenate([anterior_idx, posterior_idx])
        
        # Fill remaining with nearest to poles
        if len(selected) < n_target:
            remaining = list(set(range(len(coords))) - set(selected))
            if remaining:
                n_need = n_target - len(selected)
                extra = np.random.choice(remaining, min(n_need, len(remaining)), replace=False)
                selected = np.concatenate([selected, extra])
        
        return selected[:n_target]
    
    def _sample_boundary(self, coords, n_target):
        if len(coords) < 4:
            return np.random.choice(len(coords), min(n_target, len(coords)), replace=False)
        
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(coords)
            boundary_idx = hull.vertices
            
            if len(boundary_idx) >= n_target:
                return np.random.choice(boundary_idx, n_target, replace=False)
            else:
                # Add random internal points
                internal = list(set(range(len(coords))) - set(boundary_idx))
                n_need = n_target - len(boundary_idx)
                if internal and n_need > 0:
                    extra = np.random.choice(internal, min(n_need, len(internal)), replace=False)
                    return np.concatenate([boundary_idx, extra])
                return boundary_idx
        except:
            return self._sample_diverse(coords, n_target)
    
    def _sample_cluster(self, coords, n_target):
        # Pick random seed and get nearest neighbors
        seed_idx = np.random.randint(len(coords))
        distances = np.linalg.norm(coords - coords[seed_idx], axis=1)
        return np.argsort(distances)[:n_target]
    
    def _sample_diverse(self, coords, n_target):
        # Farthest point sampling
        selected = [np.random.randint(len(coords))]
        
        for _ in range(n_target - 1):
            # Compute minimum distance to any selected point
            min_distances = np.full(len(coords), np.inf)
            
            for s in selected:
                distances = np.linalg.norm(coords - coords[s], axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # Set already selected points to -inf so they won't be selected again
            min_distances[selected] = -np.inf
            
            # Select the farthest point
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)
        
        return np.array(selected)

def validate_point_cloud(coords, min_points=4):
    """Validate point cloud for processing"""
    if len(coords) < min_points:
        return False, f"Too few points: {len(coords)}"
    
    # Check for degeneracy
    std = np.std(coords, axis=0)
    if std.min() < 1e-6:
        return False, "Points are nearly colinear"
    
    # Check for duplicates
    unique_coords = np.unique(coords, axis=0)
    if len(unique_coords) < len(coords):
        return False, f"Contains {len(coords) - len(unique_coords)} duplicate points"
    
    return True, "Valid"

class SparseEmbryoDataset(Dataset):
    def __init__(self, data_dict, stage_limit=194, min_cells=5, max_cells=20,
                 augment=True, num_rotations=10, curriculum_stage=0):
        
        self.data_dict = data_dict
        self.stage_limit = stage_limit
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.augment = augment
        self.num_rotations = num_rotations if augment else 1
        self.curriculum_stage = curriculum_stage
        
        self.sampler = ImprovedSampler(min_cells, max_cells)
        
        # Build dataset
        self.data_items = self._build_dataset()
        self.pairs = self._generate_pairs()
        
        print(f"Dataset created: {len(self.data_items)} items, {len(self.pairs)} pairs")
    
    def _build_dataset(self):
        items = []
        
        for run, timepoints in self.data_dict.items():
            for t, cells in sorted(timepoints.items()):
                n_cells = len(cells)
                
                if n_cells > self.stage_limit or n_cells < self.min_cells:
                    continue
                
                # Normalize coordinates
                coords = np.array([cells[cid] for cid in cells.keys()])
                coords_normalized = self._normalize_coords(coords)
                cells_normalized = {cid: coords_normalized[i] for i, cid in enumerate(cells.keys())}
                
                # Add original
                items.append({
                    'run': run, 'time': t, 'cells': cells_normalized,
                    'n_cells': n_cells, 'rotation_idx': 0
                })
                
                # Add rotations
                if self.augment:
                    for rot_idx in range(1, self.num_rotations):
                        items.append({
                            'run': run, 'time': t, 'cells': cells_normalized,
                            'n_cells': n_cells, 'rotation_idx': rot_idx
                        })
        
        return items
    
    def _normalize_coords(self, coords):
        """Normalize coordinates to zero mean and unit variance"""
        mean = coords.mean(axis=0)
        std = coords.std(axis=0).clip(min=1e-6)
        return (coords - mean) / std
    
    def _generate_pairs(self):
        """Generate pairs based on curriculum stage"""
        pairs = []
        
        # Group by embryo and cell count
        embryo_timeline = defaultdict(list)
        stage_groups = defaultdict(list)
        
        for item in self.data_items:
            embryo_timeline[item['run']].append(item)
            stage_groups[item['n_cells']].append(item)
        
        # Sort timelines
        for run in embryo_timeline:
            embryo_timeline[run].sort(key=lambda x: x['time'])
        
        # Curriculum-based overlap reqs
        overlap_requirements = {
            0: {'min_shared': 4, 'min_overlap_ratio': 0.5},
            1: {'min_shared': 3, 'min_overlap_ratio': 0.3},
            2: {'min_shared': 2, 'min_overlap_ratio': 0.2},
            3: {'min_shared': 1, 'min_overlap_ratio': 0.1}
        }
        
        req = overlap_requirements.get(self.curriculum_stage, overlap_requirements[3])
        
        # Generate pairs with curriculum-appropriate difficulty
        for _ in range(len(self.data_items) * 3):
            if random.random() < 0.7:  # Intra-embryo pairs
                embryo = random.choice(list(embryo_timeline.keys()))
                timeline = embryo_timeline[embryo]
                
                if len(timeline) < 2:
                    continue
                
                idx1, idx2 = random.sample(range(len(timeline)), 2)
                anchor, comparison = timeline[idx1], timeline[idx2]
            else:  # Inter-embryo pairs
                anchor, comparison = random.sample(self.data_items, 2)
                
                if anchor['run'] == comparison['run']:
                    continue
            
            # Check overlap
            cells1, cells2 = anchor['cells'], comparison['cells']
            shared = list(set(cells1.keys()) & set(cells2.keys()))
            
            if len(shared) < req['min_shared']:
                continue
            
            overlap_ratio = len(shared) / len(set(cells1.keys()) | set(cells2.keys()))
            
            if overlap_ratio < req['min_overlap_ratio'] and random.random() > 0.2:
                continue
            
            pairs.append({
                'anchor': anchor,
                'comparison': comparison,
                'shared_cells': shared,
                'overlap_ratio': overlap_ratio
            })
        
        return pairs
    
    def update_curriculum(self, stage):
        """Update curriculum stage and regenerate pairs"""
        self.curriculum_stage = stage
        self.pairs = self._generate_pairs()
        print(f"Updated to curriculum stage {stage}, regenerated {len(self.pairs)} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx % len(self.pairs)]
        anchor, comparison = pair['anchor'], pair['comparison']
        
        # Sample cells with validation
        max_attempts = 5
        for attempt in range(max_attempts):
            cells1_ids, coords1 = self.sampler.sample_cells(anchor['cells'])
            cells2_ids, coords2 = self.sampler.sample_cells(comparison['cells'])
            
            if len(cells1_ids) >= self.min_cells and len(cells2_ids) >= self.min_cells:
                break
        
        # Apply rotation if needed
        if anchor['rotation_idx'] > 0:
            rotation = R.random(random_state=anchor['rotation_idx']).as_matrix()
            coords1 = (coords1 - coords1.mean(axis=0)) @ rotation.T + coords1.mean(axis=0)
        
        if comparison['rotation_idx'] > 0:
            rotation = R.random(random_state=comparison['rotation_idx']).as_matrix()
            coords2 = (coords2 - coords2.mean(axis=0)) @ rotation.T + coords2.mean(axis=0)
        
        # Apply augmentation
        if self.augment and random.random() < 0.5:
            coords1, coords2 = self._apply_augmentation(coords1, coords2)
        
        # Create match indices (with no-match token support)
        match_indices = []
        for cell_id in cells1_ids:
            if cell_id in cells2_ids:
                match_indices.append(cells2_ids.index(cell_id))
            else:
                match_indices.append(len(cells2_ids))  # Points to no-match token
        
        # Convert to tensors
        pc1 = torch.from_numpy(coords1).float()
        pc2 = torch.from_numpy(coords2).float()
        match_indices = torch.tensor(match_indices, dtype=torch.long)
        
        info = {
            'run1': anchor['run'],
            'run2': comparison['run'],
            'time1': anchor['time'],
            'time2': comparison['time'],
            'cells1_ids': cells1_ids,
            'cells2_ids': cells2_ids
        }
        
        return pc1, pc2, match_indices, info
    
    def _apply_augmentation(self, coords1, coords2):
        # Progressive augmentation based on curriculum
        noise_scale = [0.0, 0.01, 0.02, 0.03][min(self.curriculum_stage, 3)]
        rotation_angle = [np.pi/36, np.pi/24, np.pi/18, np.pi/12][min(self.curriculum_stage, 3)]
        
        # Small rotation
        if random.random() < 0.5:
            angle = random.uniform(-rotation_angle, rotation_angle)
            axis = np.random.randn(3)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            rot = R.from_rotvec(angle * axis).as_matrix()
            coords1 = (coords1 - coords1.mean(axis=0)) @ rot.T + coords1.mean(axis=0)
            coords2 = (coords2 - coords2.mean(axis=0)) @ rot.T + coords2.mean(axis=0)
        
        # Noise
        if noise_scale > 0:
            coords1 += np.random.normal(0, noise_scale, coords1.shape)
            coords2 += np.random.normal(0, noise_scale, coords2.shape)
        
        return coords1, coords2

def collate_fn_with_padding(batch):
    """Collate function that pads point clouds and creates masks"""
    pc1_list, pc2_list, match_indices_list, info_list = zip(*batch)
    
    # Find max sizes
    max_n1 = max(pc.shape[0] for pc in pc1_list)
    max_n2 = max(pc.shape[0] for pc in pc2_list)
    
    # Pad point clouds and create masks
    pc1_padded = []
    pc2_padded = []
    mask1_list = []
    mask2_list = []
    match_indices_padded = []
    
    for pc1, pc2, matches in zip(pc1_list, pc2_list, match_indices_list):
        n1, n2 = pc1.shape[0], pc2.shape[0]
        
        # Pad pc1
        if n1 < max_n1:
            pad = torch.zeros(max_n1 - n1, 3)
            pc1 = torch.cat([pc1, pad])
            mask1 = torch.cat([torch.ones(n1), torch.zeros(max_n1 - n1)])
        else:
            mask1 = torch.ones(n1)
        
        # Pad pc2
        if n2 < max_n2:
            pad = torch.zeros(max_n2 - n2, 3)
            pc2 = torch.cat([pc2, pad])
            mask2 = torch.cat([torch.ones(n2), torch.zeros(max_n2 - n2)])
        else:
            mask2 = torch.ones(n2)
        
        # Pad match indices
        if len(matches) < max_n1:
            pad_matches = torch.full((max_n1 - len(matches),), max_n2, dtype=torch.long)
            matches = torch.cat([matches, pad_matches])
        
        pc1_padded.append(pc1)
        pc2_padded.append(pc2)
        mask1_list.append(mask1)
        mask2_list.append(mask2)
        match_indices_padded.append(matches)
    
    # Stack into batches
    pc1_batch = torch.stack(pc1_padded)
    pc2_batch = torch.stack(pc2_padded)
    mask1_batch = torch.stack(mask1_list)
    mask2_batch = torch.stack(mask2_list)
    match_indices_batch = torch.stack(match_indices_padded)
    
    return pc1_batch, pc2_batch, mask1_batch, mask2_batch, match_indices_batch, info_list

class SparseCloudTrainer:
    """Trainer with curriculum learning and proper batching"""
    
    def __init__(self, model, train_dataset, val_dataset, test_dataset=None,
                 batch_size=16, learning_rate=2e-4, num_epochs=100,
                 device='cuda', patience=15, use_wandb=False,
                 checkpoint_dir='checkpoints'):
        
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Dataloaders with proper collation
        num_workers = 0 if os.name == 'nt' else 4
        persistent = False if os.name == 'nt' else True
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn_with_padding, num_workers=num_workers,
            pin_memory=(device.type == 'cuda'), persistent_workers=persistent
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn_with_padding, num_workers=0,
            pin_memory=(device.type == 'cuda'), persistent_workers=False
        )
        
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn_with_padding, num_workers=0
            )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate/25, weight_decay=1e-4
        )
        
        # OneCycleLR scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * num_epochs
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25,
            final_div_factor=1000
        )
        
        # Loss functions
        self.matching_loss = TwinAttentionMatchingLoss(use_uncertainty=model.use_uncertainty)
        self.temporal_loss = TemporalConsistencyLoss(weight=0.1)
        
        # Mixed precision (only on CUDA)
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Tracking
        self.history = defaultdict(list)
        self.best_val_metric = 0
        self.curriculum_schedule = {
            0: 0,
            20: 1,
            40: 2,
            60: 3
        }
        
        # Initialize wandb ehh maybe
        if use_wandb:
            import wandb
            wandb.init(project='twin-attention-sparse', config={
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'model_config': {
                    'embed_dim': model.embed_dim,
                    'use_sparse_features': model.use_sparse_features,
                    'use_uncertainty': model.use_uncertainty
                }
            })
    
    def train(self, resume_from_epoch=None):
        """Train with optional resume capability"""
        print(f"Starting training on {self.device}")
        
        patience_counter = 0
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from_epoch is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{resume_from_epoch}.pth')
            if os.path.exists(checkpoint_path):
                print(f"Resuming from epoch {resume_from_epoch}...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Recreate scheduler with remaining steps
                remaining_epochs = self.num_epochs - resume_from_epoch - 1
                remaining_steps = len(self.train_loader) * remaining_epochs
                if remaining_steps > 0:
                    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        self.optimizer,
                        max_lr=self.optimizer.param_groups[0]['lr'] * 25,
                        total_steps=remaining_steps,
                        epochs=remaining_epochs,
                        steps_per_epoch=len(self.train_loader),
                        pct_start=0.1,
                        anneal_strategy='cos',
                        cycle_momentum=True,
                        base_momentum=0.85,
                        max_momentum=0.95,
                        div_factor=25,
                        final_div_factor=1000
                    )
                
                # Restore history and metrics
                if 'history' in checkpoint:
                    self.history = defaultdict(list, checkpoint['history'])
                if 'metrics' in checkpoint:
                    if 'overall_acc' in checkpoint['metrics']:
                        self.best_val_metric = checkpoint['metrics']['overall_acc']
                
                start_epoch = resume_from_epoch + 1
                print(f"Resumed from epoch {resume_from_epoch}, continuing from epoch {start_epoch}")
            else:
                print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")
        
        for epoch in range(start_epoch, self.num_epochs):
            # Update curriculum
            for epoch_threshold, stage in self.curriculum_schedule.items():
                if epoch >= epoch_threshold:
                    self.train_dataset.update_curriculum(stage)
            
            # Train epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validation
            val_metrics = self._validate(epoch)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save best model
            if val_metrics['overall_acc'] > self.best_val_metric:
                self.best_val_metric = val_metrics['overall_acc']
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Final evaluation
        if self.test_dataset:
            test_metrics = self._test()
            print(f"\nTest Results: {test_metrics}")
        
        return self.history
    
    def _train_epoch(self, epoch):
        self.model.train()
        
        epoch_loss = 0
        match_correct = 0
        match_total = 0
        outlier_correct = 0
        outlier_total = 0
        batch_count = 0
        
        progress = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        
        try:
            for batch_idx, (pc1, pc2, mask1, mask2, match_indices, info_list) in enumerate(progress):
                pc1 = pc1.to(self.device)
                pc2 = pc2.to(self.device)
                mask1 = mask1.to(self.device)
                mask2 = mask2.to(self.device)
                match_indices = match_indices.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch)
                        loss, metrics = self.matching_loss(z1, z2, match_indices, temperature, mask1, mask2)
                else:
                    z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch)
                    loss, metrics = self.matching_loss(z1, z2, match_indices, temperature, mask1, mask2)
                
                # Temperature regularization
                temp_reg = 1e-3 * (self.model.log_temperature ** 2)
                loss = loss + temp_reg
                
                # Backward pass
                if self.device.type == 'cuda' and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                if self.scheduler.last_epoch < self.scheduler.total_steps - 1:
                    self.scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                match_correct += metrics['match_correct']
                match_total += metrics['match_total']
                outlier_correct += metrics['outlier_correct']
                outlier_total += metrics['outlier_total']
                batch_count += 1
                
                try:
                    current_lr = self.scheduler.get_last_lr()[0]
                except (IndexError, AttributeError):
                    current_lr = self.optimizer.param_groups[0]['lr']
                
                # Track LR for vis
                self.history['learning_rate'].append(current_lr)
                    
                progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'match_acc': f'{metrics["match_acc"]:.3f}' if metrics["match_acc"] > 0 else '0.000',
                    'lr': f'{current_lr:.2e}'
                })
                
        except Exception as e:
            print(f"Error in training: {e}")
            traceback.print_exc()
            raise
        
        return {
            'loss': epoch_loss / batch_count if batch_count > 0 else 0,
            'match_acc': match_correct / match_total if match_total > 0 else 0,
            'outlier_acc': outlier_correct / outlier_total if outlier_total > 0 else 0,
            'overall_acc': (match_correct + outlier_correct) / (match_total + outlier_total) if (match_total + outlier_total) > 0 else 0
        }
    
    def _validate(self, epoch):
        self.model.eval()
        
        val_loss = 0
        match_correct = 0
        match_total = 0
        outlier_correct = 0
        outlier_total = 0
        
        strategy_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for pc1, pc2, mask1, mask2, match_indices, info_list in tqdm(self.val_loader, desc='Validation'):
                pc1 = pc1.to(self.device)
                pc2 = pc2.to(self.device)
                mask1 = mask1.to(self.device)
                mask2 = mask2.to(self.device)
                match_indices = match_indices.to(self.device)
                
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch)
                        loss, metrics = self.matching_loss(z1, z2, match_indices, temperature, mask1, mask2)
                else:   
                    z1, z2, temperature = self.model(pc1, pc2, mask1, mask2, epoch)
                    loss, metrics = self.matching_loss(z1, z2, match_indices, temperature, mask1, mask2)

                val_loss += loss.item()
                match_correct += metrics['match_correct']
                match_total += metrics['match_total']
                outlier_correct += metrics['outlier_correct']
                outlier_total += metrics['outlier_total']
                
                # Track by difficulty
                for info in info_list:
                    n_cells = len(info['cells1_ids'])
                    if n_cells <= 10:
                        strategy = 'sparse'
                    elif n_cells <= 15:
                        strategy = 'medium'
                    else:
                        strategy = 'dense'
                    
                    # This is approximate - would need per-sample metrics
                    strategy_metrics[strategy]['total'] += 1
                    if metrics['match_acc'] > 0.5:  # Rough approximation
                        strategy_metrics[strategy]['correct'] += 1
        
        results = {
            'loss': val_loss / len(self.val_loader),
            'match_acc': match_correct / match_total if match_total > 0 else 0,
            'outlier_acc': outlier_correct / outlier_total if outlier_total > 0 else 0,
            'overall_acc': (match_correct + outlier_correct) / (match_total + outlier_total) if (match_total + outlier_total) > 0 else 0
        }
        
        # Add strategy breakdowns
        for strategy, metrics in strategy_metrics.items():
            if metrics['total'] > 0:
                results[f'{strategy}_acc'] = metrics['correct'] / metrics['total']
        
        return results
    
    def _test(self):
        """Test on held-out set"""
        return self._validate(self.num_epochs)
    
    def _find_temporal_pairs(self, info_list):
        """Find parent-child relationships in batch"""
        # Simplified - would need actual lineage information
        pairs = []
        for i, info in enumerate(info_list):
            if info['time2'] == info['time1'] + 1:
                # Would need to check actual parent-child relationships
                pairs.append((i, i))
        return pairs
    
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        # Store history
        for k, v in train_metrics.items():
            self.history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            self.history[f'val_{k}'].append(v)
        
        # Print summary
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Match Acc: {train_metrics['match_acc']:.3f}, Outlier Acc: {train_metrics['outlier_acc']:.3f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Match Acc: {val_metrics['match_acc']:.3f}, Outlier Acc: {val_metrics['outlier_acc']:.3f}")
        
        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_match_acc': train_metrics['match_acc'],
                'val_loss': val_metrics['loss'],
                'val_match_acc': val_metrics['match_acc'],
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
    
    def _save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': dict(self.history)
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

class DevelopmentalManifoldBuilder:
    """Build manifold with batched inference"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.embeddings = {}
        
    def build_manifold(self, dataloader, desc="Building manifold"):
        self.model.eval()
        
        with torch.no_grad():
            for pc1, pc2, mask1, mask2, match_indices, info_list in tqdm(dataloader, desc=desc):
                pc1 = pc1.to(self.device)
                pc2 = pc2.to(self.device)
                mask1 = mask1.to(self.device)
                mask2 = mask2.to(self.device)
                
                # Get embeddings
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        z1, z2, _ = self.model(pc1, pc2, mask1, mask2)
                else:
                    z1, z2, _ = self.model(pc1, pc2, mask1, mask2)
                
                if self.model.use_uncertainty:
                    z1_mean, _ = z1
                    embeddings = z1_mean
                else:
                    embeddings = z1
                
                # Store embeddings for each cell
                for b, info in enumerate(info_list):
                    key = (info['run1'], info['time1'])
                    if key not in self.embeddings:
                        self.embeddings[key] = {}
                    
                    # Only store valid (non-padded) embeddings
                    n_valid = int(mask1[b].sum().item())
                    for i in range(n_valid):
                        cell_id = info['cells1_ids'][i]
                        self.embeddings[key][cell_id] = embeddings[b, i].cpu()

class CellIdentificationKNN:
    """KNN-based cell identification"""
    
    def __init__(self, k=30):
        self.k = k
        self.index = None
        self.labels = None
        
    def fit(self, manifold_builder):
        # Collect embeddings
        embeddings_list = []
        labels_list = []
        
        for (embryo, time), cell_embeddings in manifold_builder.embeddings.items():
            for cell_id, embedding in cell_embeddings.items():
                embeddings_list.append(embedding.numpy())
                labels_list.append(cell_id)
        
        if not embeddings_list:
            raise ValueError("No embeddings found")
        
        embeddings_array = np.array(embeddings_list)
        self.labels = labels_list
        
        # Build FAISS index for fast retrieval
        try:
            import faiss
            d = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
            self.index.add(embeddings_array)
            print(f"Built FAISS index with {len(embeddings_array)} embeddings")
        except ImportError:
            print("FAISS not available, using sklearn")
            from sklearn.neighbors import NearestNeighbors
            self.index = NearestNeighbors(n_neighbors=self.k, metric='cosine')
            self.index.fit(embeddings_array)
    
    def predict(self, query_embeddings):
        if isinstance(self.index, type(None)):
            raise ValueError("Must fit first")
        
        if hasattr(self.index, 'search'):  # FAISS
            distances, indices = self.index.search(query_embeddings, self.k)
            # Vote based on k nearest neighbors
            predictions = []
            for neighbor_indices in indices:
                neighbor_labels = [self.labels[idx] for idx in neighbor_indices]
                # Majority vote
                label_counts = Counter(neighbor_labels)
                predictions.append(label_counts.most_common(1)[0][0])
            return predictions
        else:  # sklearn
            indices = self.index.kneighbors(query_embeddings, return_distance=False)
            predictions = []
            for neighbor_indices in indices:
                neighbor_labels = [self.labels[idx] for idx in neighbor_indices]
                label_counts = Counter(neighbor_labels)
                predictions.append(label_counts.most_common(1)[0][0])
            return predictions

def run_learning_rate_finder(model, train_loader, device='cuda', num_iter=100):
    """Leslie Smith's learning rate finder - find steepest descent, not explosion point"""
    model_state = model.state_dict()  # Save initial state
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    
    lrs = []
    losses = []
    
    # Test range from 1e-7 to 1e-1
    lr_schedule = np.logspace(-7, -1, num_iter)
    
    min_loss = float('inf')
    explosion_threshold = 4.0
    
    for i, (pc1, pc2, mask1, mask2, match_indices, _) in enumerate(train_loader):
        if i >= num_iter:
            break
        
        lr = lr_schedule[i]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        pc1 = pc1.to(device)
        pc2 = pc2.to(device)
        mask1 = mask1.to(device)
        mask2 = mask2.to(device)
        match_indices = match_indices.to(device)
        
        optimizer.zero_grad()
        
        z1, z2, temperature = model(pc1, pc2, mask1, mask2)
        loss_fn = TwinAttentionMatchingLoss(use_uncertainty=model.use_uncertainty)
        loss, _ = loss_fn(z1, z2, match_indices, temperature, mask1, mask2)
        
        # Check for NaN or explosion
        if not torch.isfinite(loss) or (len(losses) > 0 and loss.item() > min_loss * explosion_threshold):
            print(f"Loss exploded at LR={lr:.2e}")
            break
            
        loss.backward()
        
        # Gradient clipping for stability during search
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        current_loss = loss.item()
        lrs.append(lr)
        losses.append(current_loss)
        
        if current_loss < min_loss:
            min_loss = current_loss
    
    # Restore model state
    model.load_state_dict(model_state)
    
    # Find steepest descent (most negative gradient)
    if len(losses) > 10:
        # Smooth losses to reduce noise
        window_size = 5
        smoothed_losses = []
        for i in range(len(losses)):
            start = max(0, i - window_size // 2)
            end = min(len(losses), i + window_size // 2 + 1)
            smoothed_losses.append(np.mean(losses[start:end]))
        
        # Compute gradients
        gradients = np.gradient(smoothed_losses)
        
        # Find steepest descent in first 80% to avoid explosion region
        search_end = int(len(gradients) * 0.8)
        steepest_idx = np.argmin(gradients[:search_end])
        
        suggested_lr = lrs[steepest_idx]
        
        # Apply safety factor - use 0.1x to 0.5x the steepest point
        safety_factor = 0.3
        suggested_lr = suggested_lr * safety_factor
        
        print(f"LR Finder: Steepest descent at {lrs[steepest_idx]:.2e}, suggesting {suggested_lr:.2e} with safety factor")
    else:
        # Fallback if not enough data
        suggested_lr = 3e-4
        print(f"LR Finder: Using default {suggested_lr:.2e}")
    
    # Sanity bounds
    suggested_lr = np.clip(suggested_lr, 1e-5, 1e-2)
    
    return lrs, losses, suggested_lr

def visualize_training_results(history, save_path='training_results.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Match accuracy
    ax = axes[0, 1]
    ax.plot(history['train_match_acc'], label='Train')
    ax.plot(history['val_match_acc'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Match Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Outlier accuracy
    ax = axes[0, 2]
    if 'train_outlier_acc' in history:
        ax.plot(history['train_outlier_acc'], label='Train')
        ax.plot(history['val_outlier_acc'], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Outlier Detection Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
        # Difficulty breakdown
        ax = axes[1, 0]
        if 'val_sparse_acc' in history:
            ax.plot(history['val_sparse_acc'], label='Sparse (5-10)')
            if 'val_medium_acc' in history:
                ax.plot(history['val_medium_acc'], label='Medium (11-15)')
            if 'val_dense_acc' in history:
                ax.plot(history['val_dense_acc'], label='Dense (16-20)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by Density')
            ax.legend()
            ax.grid(True, alpha=0.3)    # Overall accuracy
    ax = axes[1, 1]
    if 'train_overall_acc' in history:
        ax.plot(history['train_overall_acc'], label='Train')
        ax.plot(history['val_overall_acc'], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Overall Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 2]
    if 'learning_rate' in history:
        ax.plot(history['learning_rate'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def _get_one_val_batch(trainer):
    trainer.model.eval()
    for pc1, pc2, mask1, mask2, match_indices, info_list in trainer.val_loader:
        return (pc1.to(trainer.device), pc2.to(trainer.device),
                mask1.to(trainer.device), mask2.to(trainer.device),
                match_indices.to(trainer.device), info_list)

@torch.no_grad()
def plot_match_matrix_heatmap(trainer, outdir="poster_figures"):
    os.makedirs(outdir, exist_ok=True)
    pc1, pc2, m1, m2, match_idx, info_list = _get_one_val_batch(trainer)

    # Forward once; encoder returns ((mean, logvar), (mean, logvar), T)
    (z1, _), (z2, _), temperature = trainer.model(pc1, pc2, m1, m2, epoch=0)
    sims = torch.bmm(z1, z2.transpose(1, 2)) / temperature  # [B,N1,N2(+no-match)]
    P = torch.softmax(sims[0], dim=-1).cpu().numpy()         # take first item in batch

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(P, aspect='auto', origin='upper', cmap='viridis')
    ax.set_xlabel("Comparison index (incl. no-match)")
    ax.set_ylabel("Anchor cell index")
    # Tick labels: last column is no-match
    ax.set_xticks([0, min(5, P.shape[1]-2), min(10, P.shape[1]-2), P.shape[1]-2, P.shape[1]-1])
    ax.set_xticklabels([0, 5, 10, P.shape[1]-2, "no-match"])
    ax.set_yticks([0, min(5, P.shape[0]-1), min(10, P.shape[0]-1), P.shape[0]-1])
    ax.set_yticklabels([0, 5, 10, P.shape[0]-1])
    cbar = plt.colorbar(im); cbar.set_label("Match probability")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "match_matrix_heatmap.png"), dpi=300)
    fig.savefig(os.path.join(outdir, "match_matrix_heatmap.pdf"))
    plt.close(fig)

from matplotlib.patches import FancyArrowPatch
from sklearn.manifold import TSNE

def plot_embedding_manifold(manifold_builder, outdir="poster_figures",
                            reducer="tsne", max_points=4000):
    os.makedirs(outdir, exist_ok=True)

    # Collect embeddings + times
    pts, times = [], []
    for (run, t), cellmap in manifold_builder.embeddings.items():
        for emb in cellmap.values():
            pts.append(emb.numpy())
            times.append(t)
    if len(pts) == 0:
        print("No embeddings in manifold_builder; nothing to plot.")
        return

    pts = np.array(pts); times = np.array(times)
    # Downsample for speed/clarity if very large
    if len(pts) > max_points:
        sel = np.random.choice(len(pts), max_points, replace=False)
        pts, times = pts[sel], times[sel]

    # Dimensionality reduction
    if reducer.lower() == "umap":
        try:
            import umap
            Z = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.15).fit_transform(pts)
        except Exception as e:
            print("UMAP unavailable, falling back to t-SNE:", e)
            Z = TSNE(n_components=2, init="random", learning_rate="auto",
                     perplexity=min(30, max(5, len(pts)//50)), max_iter=1000).fit_transform(pts)
    else:
        Z = TSNE(n_components=2, init="random", learning_rate="auto",
                 perplexity=min(30, max(5, len(pts)//50)), max_iter=1000).fit_transform(pts)

    # Color by coarse time bins (quartiles)
    bins = np.quantile(times, [0, 0.25, 0.5, 0.75, 1.0])
    labels = np.digitize(times, bins[1:-1], right=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sc = ax.scatter(Z[:,0], Z[:,1], s=8, alpha=0.7, c=labels, cmap="tab10")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")

    # Early→Late arrow (global means)
    early = Z[times <= np.percentile(times, 10)].mean(0)
    late  = Z[times >= np.percentile(times, 90)].mean(0)
    ax.add_patch(FancyArrowPatch(early, late, arrowstyle='->', lw=1.5, mutation_scale=12))

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "embedding_manifold.png"), dpi=300)
    fig.savefig(os.path.join(outdir, "embedding_manifold.pdf"))
    plt.close(fig)

def plot_accuracy_bars_by_density(trainer, outdir="poster_figures"):
    os.makedirs(outdir, exist_ok=True)
    model, device = trainer.model, trainer.device
    model.eval()

    bins = {"Sparse (5–10)": [], "Medium (11–15)": [], "Dense (16–20)": []}

    with torch.no_grad():
        for pc1, pc2, m1, m2, match_idx, info_list in trainer.val_loader:
            pc1, pc2, m1, m2, match_idx = pc1.to(device), pc2.to(device), m1.to(device), m2.to(device), match_idx.to(device)
            (z1, _), (z2, _), temperature = model(pc1, pc2, m1, m2, epoch=0)
            sims = torch.bmm(z1, z2.transpose(1, 2)) / temperature
            pred = sims.argmax(dim=-1)

            for b, info in enumerate(info_list):
                n_valid = int(m1[b].sum().item())
                correct = 0
                for i in range(n_valid):
                    if pred[b, i].item() == match_idx[b, i].item():
                        correct += 1
                acc = correct / max(1, n_valid)
                if n_valid <= 10:
                    bins["Sparse (5–10)"].append(acc)
                elif n_valid <= 15:
                    bins["Medium (11–15)"].append(acc)
                else:
                    bins["Dense (16–20)"].append(acc)

    cats = list(bins.keys())
    means = [100*np.mean(bins[c]) if len(bins[c])>0 else np.nan for c in cats]
    stds  = [100*np.std(bins[c])  if len(bins[c])>0 else np.nan for c in cats]
    overall = np.nanmean(means)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.bar(cats, means, yerr=stds, capsize=6)
    ax.set_ylim(80, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Held-out Accuracy by Subset Size (Overall ≈ {overall:.0f}%)")
    for i, v in enumerate(means):
        if not np.isnan(v):
            ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_bars.png"), dpi=300)
    fig.savefig(os.path.join(outdir, "accuracy_bars.pdf"))
    plt.close(fig)

def main(config=None):
    """Main training pipeline"""
    
    print(f" Subset Twin Attention Pipeline")
    print(f"   Device: {device}")
    
    # Default config
    if config is None:
        config = {
            'data_path': 'data_dict.pkl',
            'embed_dim': 128,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'use_sparse_features': True,
            'use_uncertainty': True,
            'use_learnable_no_match': True,
            'stage_limit': 194,
            'min_cells': 5,
            'max_cells': 20,
            'num_rotations': 10,
            'batch_size': 16,
            'num_epochs': 100,
            'patience': 15,
            'use_wandb': False
        }
    
    # Load data
    print("\nLoading data...")
    with open(config['data_path'], 'rb') as f:
        data_dict = pickle.load(f)
    
    # Split data
    embryo_ids = list(data_dict.keys())
    random.shuffle(embryo_ids)
    
    n_train = int(0.7 * len(embryo_ids))
    n_val = int(0.15 * len(embryo_ids))
    
    train_embryos = embryo_ids[:n_train]
    val_embryos = embryo_ids[n_train:n_train+n_val]
    test_embryos = embryo_ids[n_train+n_val:]
    
    print(f"  Train: {len(train_embryos)}, Val: {len(val_embryos)}, Test: {len(test_embryos)}")
    
    # Create datasets
    train_data = {k: v for k, v in data_dict.items() if k in train_embryos}
    val_data = {k: v for k, v in data_dict.items() if k in val_embryos}
    test_data = {k: v for k, v in data_dict.items() if k in test_embryos}
    
    train_dataset = SparseEmbryoDataset(
        train_data,
        stage_limit=config['stage_limit'],
        min_cells=config['min_cells'],
        max_cells=config['max_cells'],
        augment=True,
        num_rotations=config['num_rotations']
    )
    
    val_dataset = SparseEmbryoDataset(
        val_data,
        stage_limit=config['stage_limit'],
        min_cells=config['min_cells'],
        max_cells=config['max_cells'],
        augment=False,
        num_rotations=1
    )
    
    test_dataset = SparseEmbryoDataset(
        test_data,
        stage_limit=config['stage_limit'],
        min_cells=config['min_cells'],
        max_cells=config['max_cells'],
        augment=False,
        num_rotations=1
    )
    
    # Create model
    print("\nBuilding model...")
    model = EnhancedTwinAttentionEncoder(
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_sparse_features=config['use_sparse_features'],
        use_uncertainty=config['use_uncertainty'],
        use_learnable_no_match=config['use_learnable_no_match']
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    # Learning rate finder
    if config.get('use_lr_finder', True):
        print("\nRunning learning rate finder...")
        temp_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'],
            shuffle=True, collate_fn=collate_fn_with_padding
        )
        lrs, losses, suggested_lr = run_learning_rate_finder(model, temp_loader, device)
        config['learning_rate'] = suggested_lr
    else:
        config['learning_rate'] = config.get('learning_rate', 2e-4)
        print(f"\nUsing fixed learning rate: {config['learning_rate']:.2e}")
    
    # Reinitialize model
    model = EnhancedTwinAttentionEncoder(
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_sparse_features=config['use_sparse_features'],
        use_uncertainty=config['use_uncertainty'],
        use_learnable_no_match=config['use_learnable_no_match']
    )
    
    # Create trainer
    print("\nStarting training...")
    trainer = SparseCloudTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        patience=config['patience'],
        device=device,
        use_wandb=config['use_wandb']
    )
    
    # Train
    resume_epoch = None  # Change this to the epoch you want to resume from, or None to start fresh
    history = trainer.train(resume_from_epoch=resume_epoch)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_training_results(history)
    
    # Build manifold
    print("\nBuilding developmental manifold...")
    manifold_builder = DevelopmentalManifoldBuilder(model, device)
    manifold_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_fn_with_padding
    )
    manifold_builder.build_manifold(manifold_loader)
    
    # Test identification
    print("\nTesting cell identification...")
    identifier = CellIdentificationKNN(k=30)
    identifier.fit(manifold_builder)

    outdir = "poster_figures_cibb"
    plot_match_matrix_heatmap(trainer, outdir=outdir)
    plot_embedding_manifold(manifold_builder, outdir=outdir, reducer="tsne")  # or "umap"
    plot_accuracy_bars_by_density(trainer, outdir=outdir)
    print(f"\nPoster figures saved in: {outdir}")

    # Save results
    print("\nSaving results...")
    torch.save(model.state_dict(), 'twin_attention_final.pth')
    
    with open('training_results.pkl', 'wb') as f:
        pickle.dump({
            'config': config,
            'history': dict(history),
            'manifold': manifold_builder.embeddings
        }, f)
    
    print("\nTraining complete!")
    return history

if __name__ == "__main__":
    config = {
        'data_path': r"C:\Users\henry\OneDrive\Documents\Research Folder\Data\data_dict.pkl",
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'use_sparse_features': True,
        'use_uncertainty': True,
        'use_learnable_no_match': True,
        'stage_limit': 194,
        'min_cells': 5,
        'max_cells': 20,
        'num_rotations': 10,
        'batch_size': 16,
        'num_epochs': 100,
        'patience': 15,
        'use_wandb': False,
        'use_lr_finder': True,   
        'learning_rate': 3e-4    # Fallback
    }
    results = main(config)