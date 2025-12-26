"""
Ablation Training Module for C. elegans Cell Identification Model

This module provides functionality to train model variants for ablation studies:
- Feature ablations (remove individual geometric features)
- Architectural ablations (remove no-match token, curriculum learning, joint attention)

Each ablation requires retraining from scratch to properly measure the contribution
of each component.

Usage:
    python ablation_trainer.py --ablation raw_xyz --output-dir ablation_models
    python ablation_trainer.py --ablation no_rel_pos --output-dir ablation_models
    python ablation_trainer.py --ablation siamese --output-dir ablation_models
    python ablation_trainer.py --all-ablations --output-dir ablation_models
"""

import os
import sys
import pickle
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from debug_sparse_matching import (
    EnhancedTwinAttentionEncoder,
    SparsePointFeatures,
    SparseEmbryoDataset,
    SparseCloudTrainer,
    collate_fn_with_padding,
    TwinAttentionMatchingLoss,
    set_seed,
    device
)
from evaluation_suite_fixed import (
    SiameseTransformer,
    DataPaths,
    FixedEvaluationEngine as EvaluationEngine,
    StatisticalAnalysis
)


# =============================================================================
# ABLATED MODEL VARIANTS
# =============================================================================

class RawXYZFeatures(nn.Module):
    """
    Baseline feature extractor using only raw XYZ coordinates.

    This ablation tests: "What if we don't use geometric features?"
    Expected result: ~68.4% accuracy (significant drop from 90.3%)
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # Simple linear projection from xyz to embedding
        self.linear = nn.Linear(3, embed_dim)

    def forward(self, points, mask=None):
        return self.linear(points)


class NoRelativePositionFeatures(nn.Module):
    """
    Feature extractor without relative position encoding.

    Tests contribution of: Relative position (−16.5 pp expected)
    Keeps: centroid distance, point count, local density
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # Only 3 feature types instead of 4
        self.centroid_distance_enc = nn.Linear(1, embed_dim // 3)
        self.point_count_enc = nn.Embedding(50, embed_dim // 3)
        self.local_density_enc = nn.Linear(1, embed_dim // 3)
        self.output_proj = nn.Linear(embed_dim // 3 * 3, embed_dim)

    def forward(self, points, mask=None):
        B, N, _ = points.shape

        # Compute centroid
        if mask is not None:
            points_masked = points * mask.unsqueeze(-1)
            n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
            centroid = points_masked.sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        else:
            centroid = points.mean(dim=1, keepdim=True)
            n_valid = torch.full((B, 1), N, device=points.device)

        # Centroid distance
        relative_pos = points - centroid
        centroid_dist = torch.norm(relative_pos, dim=-1, keepdim=True)
        dist_features = self.centroid_distance_enc(centroid_dist)

        # Point count
        n_valid_long = n_valid.squeeze(1).long()
        count_emb = self.point_count_enc(n_valid_long)
        count_features = count_emb.unsqueeze(1).expand(-1, N, -1)

        # Local density
        local_density = self._compute_local_density(points, mask)
        density_features = self.local_density_enc(local_density)

        features = torch.cat([dist_features, count_features, density_features], dim=-1)
        return self.output_proj(features)

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


class NoDensityFeatures(nn.Module):
    """
    Feature extractor without local density.

    Tests contribution of: Local density (−11.9 pp expected)
    Keeps: relative position, centroid distance, point count
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.relative_position_enc = nn.Linear(3, embed_dim // 3)
        self.centroid_distance_enc = nn.Linear(1, embed_dim // 3)
        self.point_count_enc = nn.Embedding(50, embed_dim // 3)
        self.output_proj = nn.Linear(embed_dim // 3 * 3, embed_dim)

    def forward(self, points, mask=None):
        B, N, _ = points.shape

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

        # Centroid distance
        centroid_dist = torch.norm(relative_pos, dim=-1, keepdim=True)
        dist_features = self.centroid_distance_enc(centroid_dist)

        # Point count
        n_valid_long = n_valid.squeeze(1).long()
        count_emb = self.point_count_enc(n_valid_long)
        count_features = count_emb.unsqueeze(1).expand(-1, N, -1)

        features = torch.cat([rel_features, dist_features, count_features], dim=-1)
        return self.output_proj(features)


class NoCountFeatures(nn.Module):
    """
    Feature extractor without point count embedding.

    Tests contribution of: Point count (−9.1 pp expected)
    Keeps: relative position, centroid distance, local density
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.relative_position_enc = nn.Linear(3, embed_dim // 3)
        self.centroid_distance_enc = nn.Linear(1, embed_dim // 3)
        self.local_density_enc = nn.Linear(1, embed_dim // 3)
        self.output_proj = nn.Linear(embed_dim // 3 * 3, embed_dim)

    def forward(self, points, mask=None):
        B, N, _ = points.shape

        if mask is not None:
            points_masked = points * mask.unsqueeze(-1)
            n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
            centroid = points_masked.sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        else:
            centroid = points.mean(dim=1, keepdim=True)

        # Relative positions
        relative_pos = points - centroid
        rel_features = self.relative_position_enc(relative_pos)

        # Centroid distance
        centroid_dist = torch.norm(relative_pos, dim=-1, keepdim=True)
        dist_features = self.centroid_distance_enc(centroid_dist)

        # Local density
        local_density = self._compute_local_density(points, mask)
        density_features = self.local_density_enc(local_density)

        features = torch.cat([rel_features, dist_features, density_features], dim=-1)
        return self.output_proj(features)

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


class NoCentroidFeatures(nn.Module):
    """
    Feature extractor without centroid distance.

    Tests contribution of: Centroid distance (−6.8 pp expected)
    Keeps: relative position, point count, local density
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.relative_position_enc = nn.Linear(3, embed_dim // 3)
        self.point_count_enc = nn.Embedding(50, embed_dim // 3)
        self.local_density_enc = nn.Linear(1, embed_dim // 3)
        self.output_proj = nn.Linear(embed_dim // 3 * 3, embed_dim)

    def forward(self, points, mask=None):
        B, N, _ = points.shape

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

        # Point count
        n_valid_long = n_valid.squeeze(1).long()
        count_emb = self.point_count_enc(n_valid_long)
        count_features = count_emb.unsqueeze(1).expand(-1, N, -1)

        # Local density
        local_density = self._compute_local_density(points, mask)
        density_features = self.local_density_enc(local_density)

        features = torch.cat([rel_features, count_features, density_features], dim=-1)
        return self.output_proj(features)

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


class AblatedEncoder(nn.Module):
    """
    Encoder with configurable ablations for feature extraction.

    Supports ablating individual geometric features while keeping
    the rest of the architecture (joint attention, no-match token) intact.
    """

    def __init__(
        self,
        ablation_type: str = 'none',  # 'none', 'raw_xyz', 'no_rel_pos', 'no_density', 'no_count', 'no_centroid'
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_learnable_no_match: bool = True,
        use_uncertainty: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ablation_type = ablation_type
        self.use_learnable_no_match = use_learnable_no_match
        self.use_uncertainty = use_uncertainty
        self.use_sparse_features = True  # For compatibility

        # Select feature extractor based on ablation type
        if ablation_type == 'raw_xyz':
            self.sparse_features = RawXYZFeatures(embed_dim)
            self.feature_projection = nn.Identity()
        elif ablation_type == 'no_rel_pos':
            self.sparse_features = NoRelativePositionFeatures(embed_dim)
            self.feature_projection = nn.Identity()
        elif ablation_type == 'no_density':
            self.sparse_features = NoDensityFeatures(embed_dim)
            self.feature_projection = nn.Identity()
        elif ablation_type == 'no_count':
            self.sparse_features = NoCountFeatures(embed_dim)
            self.feature_projection = nn.Identity()
        elif ablation_type == 'no_centroid':
            self.sparse_features = NoCentroidFeatures(embed_dim)
            self.feature_projection = nn.Identity()
        else:
            # Full features (no ablation)
            self.sparse_features = SparsePointFeatures(embed_dim)
            self.feature_projection = nn.Linear(embed_dim, embed_dim)

        # Learnable no-match token
        if use_learnable_no_match:
            self.no_match_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, embed_dim) * 0.02)

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

        # Temperature
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, pc1, pc2, mask1=None, mask2=None, epoch=0):
        B = pc1.shape[0]

        # Extract features
        z1 = self.sparse_features(pc1, mask1)
        z2 = self.sparse_features(pc2, mask2)
        z1 = self.feature_projection(z1)
        z2 = self.feature_projection(z2)

        # Add learnable no-match token to pc2
        if self.use_learnable_no_match:
            no_match = self.no_match_token.expand(B, -1, -1)
            z2 = torch.cat([z2, no_match], dim=1)
            if mask2 is not None:
                mask2 = torch.cat([mask2, torch.ones(B, 1, device=mask2.device)], dim=1)

        # Concatenate for joint attention
        z = torch.cat([z1, z2], dim=1)

        # Create combined mask
        if mask1 is not None or mask2 is not None:
            if mask1 is None:
                mask1 = torch.ones(B, pc1.shape[1], device=pc1.device, dtype=torch.bool)
            if mask2 is None:
                mask2 = torch.ones(B, pc2.shape[1], device=pc2.device, dtype=torch.bool)
            combined_mask = torch.cat([mask1, mask2], dim=1).bool()
            attn_mask = ~combined_mask
        else:
            attn_mask = None

        # Add positional encoding
        seq_len = z.shape[1]
        z = z + self.pos_encoding[:, :seq_len, :]

        # Transform
        if attn_mask is not None:
            z = self.transformer(z, src_key_padding_mask=attn_mask.bool())
        else:
            z = self.transformer(z)

        # Split back
        N1 = pc1.shape[1]
        z1, z2_with_no_match = z[:, :N1], z[:, N1:]

        # Output projections
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

        # Temperature
        temperature = torch.exp(self.log_temperature).clamp(0.01, 10.0)

        return outputs[0], outputs[1], temperature


class NoMatchAblatedEncoder(EnhancedTwinAttentionEncoder):
    """
    Encoder without the learnable no-match token.

    Tests contribution of: No-match modeling (−14.7 pp expected)

    Without no-match token, the model is forced to assign every query cell
    to some reference cell, even when no valid correspondence exists.
    """

    def __init__(self, **kwargs):
        kwargs['use_learnable_no_match'] = False
        super().__init__(**kwargs)


# =============================================================================
# ABLATION TRAINER
# =============================================================================

class AblationTrainer:
    """
    Trainer for ablation experiments.

    Handles:
    - Creating appropriate model variant for each ablation
    - Training with proper configuration
    - Saving model and results
    - Evaluating on held-out data
    """

    ABLATIONS = {
        'full': 'Full model (no ablation)',
        'raw_xyz': 'Raw XYZ coordinates only (no geometric features)',
        'no_rel_pos': 'No relative position encoding',
        'no_density': 'No local density feature',
        'no_count': 'No point count embedding',
        'no_centroid': 'No centroid distance feature',
        'siamese': 'Siamese architecture (no joint attention)',
        'no_match_token': 'No learnable no-match token',
        'no_curriculum': 'No curriculum learning'
    }

    def __init__(
        self,
        data_path: str = DataPaths.TRAIN_DATA,
        eval_data_path: str = DataPaths.EVAL_DATA,
        output_dir: str = 'ablation_models',
        num_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 3e-4,
        patience: int = 15,
        seed: int = 42
    ):
        self.data_path = data_path
        self.eval_data_path = eval_data_path
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed

        os.makedirs(output_dir, exist_ok=True)
        set_seed(seed)

        self.data_dict = None
        self.eval_data = None
        self.train_data = None
        self.val_data = None

    def load_data(self):
        """Load and split data"""
        print("Loading data...")

        with open(self.data_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        with open(self.eval_data_path, 'rb') as f:
            self.eval_data = pickle.load(f)

        # Split training data
        embryo_ids = list(self.data_dict.keys())
        random.shuffle(embryo_ids)

        n_train = int(0.85 * len(embryo_ids))
        train_ids = embryo_ids[:n_train]
        val_ids = embryo_ids[n_train:]

        self.train_data = {k: self.data_dict[k] for k in train_ids}
        self.val_data = {k: self.data_dict[k] for k in val_ids}

        print(f"  Train: {len(self.train_data)} embryos")
        print(f"  Val: {len(self.val_data)} embryos")
        print(f"  Eval: {len(self.eval_data)} embryos")

    def create_model(self, ablation: str) -> nn.Module:
        """Create model for specific ablation"""
        if ablation == 'full':
            return EnhancedTwinAttentionEncoder(
                embed_dim=128,
                num_heads=8,
                num_layers=6,
                dropout=0.1,
                use_sparse_features=True,
                use_uncertainty=True,
                use_learnable_no_match=True
            )

        elif ablation in ['raw_xyz', 'no_rel_pos', 'no_density', 'no_count', 'no_centroid']:
            return AblatedEncoder(
                ablation_type=ablation,
                embed_dim=128,
                num_heads=8,
                num_layers=6,
                dropout=0.1,
                use_learnable_no_match=True,
                use_uncertainty=True
            )

        elif ablation == 'siamese':
            return SiameseTransformer(
                embed_dim=128,
                num_heads=8,
                num_layers=6,
                dropout=0.1,
                use_sparse_features=True,
                use_uncertainty=True
            )

        elif ablation == 'no_match_token':
            return EnhancedTwinAttentionEncoder(
                embed_dim=128,
                num_heads=8,
                num_layers=6,
                dropout=0.1,
                use_sparse_features=True,
                use_uncertainty=True,
                use_learnable_no_match=False  # Key difference
            )

        else:
            raise ValueError(f"Unknown ablation: {ablation}")

    def train_ablation(
        self,
        ablation: str,
        use_curriculum: bool = True
    ) -> Dict[str, Any]:
        """
        Train a single ablation variant.

        Args:
            ablation: Ablation type
            use_curriculum: Whether to use curriculum learning

        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*60}")
        print(f"Training ablation: {ablation}")
        print(f"Description: {self.ABLATIONS.get(ablation, 'Unknown')}")
        print(f"{'='*60}")

        if self.data_dict is None:
            self.load_data()

        # Create model
        model = self.create_model(ablation)
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        # Create datasets
        # For no_curriculum ablation, we modify the dataset behavior
        curriculum_stage = 0 if use_curriculum else 3  # Stage 3 = hardest, no progression

        train_dataset = SparseEmbryoDataset(
            self.train_data,
            stage_limit=194,
            min_cells=5,
            max_cells=20,
            augment=True,
            num_rotations=10,
            curriculum_stage=curriculum_stage
        )

        val_dataset = SparseEmbryoDataset(
            self.val_data,
            stage_limit=194,
            min_cells=5,
            max_cells=20,
            augment=False,
            num_rotations=1,
            curriculum_stage=3
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_with_padding,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_padding,
            num_workers=0
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate / 25,
            weight_decay=1e-4
        )

        # Scheduler
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.num_epochs

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Loss
        loss_fn = TwinAttentionMatchingLoss(use_uncertainty=model.use_uncertainty)

        # Training loop
        history = defaultdict(list)
        best_val_acc = 0
        patience_counter = 0

        # Curriculum schedule (only used if use_curriculum=True)
        curriculum_schedule = {0: 0, 20: 1, 40: 2, 60: 3}

        for epoch in range(self.num_epochs):
            # Update curriculum (if enabled)
            if use_curriculum:
                for epoch_threshold, stage in curriculum_schedule.items():
                    if epoch >= epoch_threshold:
                        train_dataset.curriculum_stage = stage
                        train_dataset.pairs = train_dataset._generate_pairs()

            # Training
            model.train()
            epoch_loss = 0
            match_correct = 0
            match_total = 0

            progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')

            for batch in progress:
                pc1, pc2, mask1, mask2, match_indices, info_list = batch

                pc1 = pc1.to(device)
                pc2 = pc2.to(device)
                mask1 = mask1.to(device)
                mask2 = mask2.to(device)
                match_indices = match_indices.to(device)

                optimizer.zero_grad()

                # Handle Siamese differently
                if ablation == 'siamese':
                    z1 = model.encode_single(pc1, mask1)
                    z2 = model.encode_single(pc2, mask2)
                    temperature = torch.exp(model.log_temperature).clamp(0.01, 10.0)
                else:
                    z1, z2, temperature = model(pc1, pc2, mask1, mask2, epoch)

                loss, metrics = loss_fn(z1, z2, match_indices, temperature, mask1, mask2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                match_correct += metrics['match_correct']
                match_total += metrics['match_total']

                progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{metrics["match_acc"]:.3f}'
                })

            train_acc = match_correct / max(1, match_total)
            train_loss = epoch_loss / len(train_loader)

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    pc1, pc2, mask1, mask2, match_indices, info_list = batch

                    pc1 = pc1.to(device)
                    pc2 = pc2.to(device)
                    mask1 = mask1.to(device)
                    mask2 = mask2.to(device)
                    match_indices = match_indices.to(device)

                    if ablation == 'siamese':
                        z1 = model.encode_single(pc1, mask1)
                        z2 = model.encode_single(pc2, mask2)
                        temperature = torch.exp(model.log_temperature).clamp(0.01, 10.0)
                    else:
                        z1, z2, temperature = model(pc1, pc2, mask1, mask2, epoch)

                    _, metrics = loss_fn(z1, z2, match_indices, temperature, mask1, mask2)

                    val_correct += metrics['match_correct']
                    val_total += metrics['match_total']

            val_acc = val_correct / max(1, val_total)

            # Log
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                save_path = os.path.join(self.output_dir, f'{ablation}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'ablation': ablation
                }, save_path)
                print(f"  Saved best model: {save_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save final results
        results = {
            'ablation': ablation,
            'description': self.ABLATIONS.get(ablation, 'Unknown'),
            'best_val_acc': best_val_acc,
            'final_train_acc': train_acc,
            'epochs_trained': epoch + 1,
            'history': dict(history)
        }

        results_path = os.path.join(self.output_dir, f'{ablation}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def train_all_ablations(self) -> Dict[str, Dict]:
        """Train all ablation variants"""
        all_results = {}

        ablations_to_run = [
            ('full', True),                 # Full model with curriculum
            ('raw_xyz', True),              # Feature ablation: raw xyz
            ('no_rel_pos', True),           # Feature ablation: no relative position
            ('no_density', True),           # Feature ablation: no local density
            ('no_count', True),             # Feature ablation: no point count
            ('no_centroid', True),          # Feature ablation: no centroid distance
            ('siamese', True),              # Architecture ablation: no joint attention
            ('no_match_token', True),       # Architecture ablation: no no-match token
            ('full', False),                # Architecture ablation: no curriculum (use 'full' model)
        ]

        for ablation, use_curriculum in ablations_to_run:
            ablation_name = ablation if use_curriculum else 'no_curriculum'
            results = self.train_ablation(ablation, use_curriculum)
            all_results[ablation_name] = results

        # Save summary
        summary_path = os.path.join(self.output_dir, 'ablation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "="*60)
        print("ABLATION SUMMARY")
        print("="*60)
        for name, results in all_results.items():
            print(f"{name}: {results['best_val_acc']*100:.1f}%")

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Ablation Training for C. elegans Model')
    parser.add_argument('--ablation', type=str, choices=list(AblationTrainer.ABLATIONS.keys()),
                       help='Specific ablation to train')
    parser.add_argument('--all-ablations', action='store_true',
                       help='Train all ablation variants')
    parser.add_argument('--data-path', type=str, default=DataPaths.TRAIN_DATA,
                       help='Path to training data')
    parser.add_argument('--eval-data', type=str, default=DataPaths.EVAL_DATA,
                       help='Path to evaluation data')
    parser.add_argument('--output-dir', type=str, default='ablation_models',
                       help='Output directory for models')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    trainer = AblationTrainer(
        data_path=args.data_path,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed
    )

    if args.all_ablations:
        trainer.train_all_ablations()
    elif args.ablation:
        trainer.train_ablation(args.ablation)
    else:
        print("Please specify --ablation or --all-ablations")
        print("\nAvailable ablations:")
        for name, desc in AblationTrainer.ABLATIONS.items():
            print(f"  {name}: {desc}")


if __name__ == "__main__":
    main()
