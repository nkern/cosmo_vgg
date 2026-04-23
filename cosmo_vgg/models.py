"""
Cosmological Feature Encoder
=============================
VGG-style CNN encoder (2D or 3D) with mid-network multi-head self-attention,
trained with VICReg contrastive loss + optional parameter regression.

Designed for CAMELS CMD 3D grids (25 Mpc/h boxes, fixed resolution).

2D mode (--twodim):
    The 3D input (B, D, H, W) is thinned along the D axis by --thin_factor
    (default 8) via average pooling, then reshaped to (B*D', 1, H, W) so
    each depth slice becomes an independent 2D sample. A 2D VGG encoder +
    2D attention is used. Embeddings are averaged back over slices per volume
    before computing VICReg and FID.

Usage:
    # 3D training
    python cosmo_encoder.py --mode train --data_dir /path/to/camels --epochs 100

    # 2D training
    python cosmo_encoder.py --mode train --twodim --thin_factor 8 \
        --data_dir /path/to/camels --epochs 100

    # Embedding extraction
    python cosmo_encoder.py --mode embed --checkpoint encoder.pt --data_dir /path/to/camels

    # FID computation
    python cosmo_encoder.py --mode fid --checkpoint encoder.pt \
        --train_dir /path/to/train --gen_dir /path/to/generated
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CAMELSDataset(Dataset):
    """
    Loads CAMELS CMD 3D grids from .npy files.

    Expected directory structure:
        data_dir/
            cosmo_{Om}_{s8}_seed{i}_field{j}.npy   # (N, N, N) float32

    Each sample is identified by (Om, s8) — simulations sharing these
    parameters but differing in seed form positive pairs for VICReg.

    Args:
        data_dir:    Path to directory containing .npy grid files.
        fields:      List of field names to load as channels.
        resolution:  Spatial resolution to resize to. None = use as-is.
        transform:   Optional callable applied to the tensor after loading.
        twodim:      If True, thin the D axis and return per-slice tensors.
        thin_factor: Factor by which to reduce the D axis in 2D mode.
    """

    def __init__(
        self,
        data_dir,
        fields=None,
        resolution=None,
        transform=None,
        twodim=False,
        thin_factor=8,
    ):
        self.data_dir = Path(data_dir)
        self.fields = fields or ['Mgas']
        self.resolution = resolution
        self.transform = transform
        self.twodim = twodim
        self.thin_factor = thin_factor

        # Group files by cosmology label (Om, s8) for positive pair sampling
        self.files = sorted(self.data_dir.glob('*.npy'))
        if not self.files:
            raise FileNotFoundError(f'No .npy files found in {data_dir}')

        # Build index: cosmo_key -> list of file paths
        # Assumes filename contains Om and s8 as underscore-separated floats
        # e.g. cosmo_0.30_0.80_seed1_Mgas.npy
        self.cosmo_groups = {}
        for f in self.files:
            key = self._parse_cosmo_key(f)
            self.cosmo_groups.setdefault(key, []).append(f)

        self.cosmo_keys = list(self.cosmo_groups.keys())

    def _parse_cosmo_key(self, fpath):
        """Extract (Om, s8) tuple from filename. Adjust for your naming convention."""
        parts = fpath.stem.split('_')
        try:
            # Expected: cosmo_{Om}_{s8}_...
            om = float(parts[1])
            s8 = float(parts[2])
            return (round(om, 4), round(s8, 4))
        except (IndexError, ValueError):
            # Fallback: treat each file as its own cosmology
            return (fpath.stem,)

    def _load_grid(self, fpath):
        """
        Load a 3D grid, apply log1p, normalize, return tensor.

        3D mode: returns (1, D, H, W)
        2D mode: thins the D axis by strided slicing (step=thin_factor),
                 returns (D', H, W) — each selected slice is a separate image.
                 The training loop reshapes to (D', 1, H, W) before the encoder.
        """
        grid = np.load(fpath).astype(np.float32)

        # Log-compress density-like fields (handles dynamic range)
        grid = np.log1p(np.clip(grid, 0, None))

        # Standardize to zero mean, unit variance per sample
        grid = (grid - grid.mean()) / (grid.std() + 1e-8)

        tensor = torch.from_numpy(grid).unsqueeze(0)  # (1, D, H, W)

        if self.resolution is not None and tensor.shape[-1] != self.resolution:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.resolution,) * 3,
                mode='trilinear',
                align_corners=False,
            ).squeeze(0)

        if self.twodim:
            # Thin D axis by strided slicing: (1, D, H, W) -> (D', H, W)
            # Preserves individual slice structure (no blurring from pooling).
            tensor = tensor[0, ::self.thin_factor, :, :]  # (D', H, W)

        return tensor

    def __len__(self):
        return len(self.cosmo_keys)

    def __getitem__(self, idx):
        """
        Returns a positive pair (two different seeds, same cosmology)
        along with the cosmological parameters as regression targets.

        3D mode: x_a, x_b are (1, D, H, W)
        2D mode: x_a, x_b are (D', H, W) — the training loop handles reshaping
        """
        key = self.cosmo_keys[idx]
        group = self.cosmo_groups[key]

        # Sample two distinct files from the same cosmology group
        if len(group) >= 2:
            i, j = np.random.choice(len(group), size=2, replace=False)
            path_a, path_b = group[i], group[j]
        else:
            path_a = path_b = group[0]

        x_a = self._load_grid(path_a)
        x_b = self._load_grid(path_b)

        if self.transform:
            x_a = self.transform(x_a)
            x_b = self.transform(x_b)

        # Regression targets: Om and s8 (or zeros if not parseable)
        if isinstance(key[0], float):
            params = torch.tensor([key[0], key[1]], dtype=torch.float32)
        else:
            params = torch.zeros(2, dtype=torch.float32)

        return x_a, x_b, params


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

class CosmologicalAugment(nn.Module):
    """
    Physics-motivated augmentations for cosmological fields.

    Works on both 3D (C, D, H, W) and 2D (C, H, W) tensors.
    All augmentations respect statistical isotropy / homogeneity:
        - Random axis-aligned 90-degree rotations (exact symmetry)
        - Random reflections (exact symmetry)
        - Gaussian noise injection
        - Random k-space high-frequency masking (simulates lower resolution)

    Args:
        noise_std:        Std of additive Gaussian noise.
        kspace_mask_prob: Probability of applying k-space masking per sample.
        kmax_frac:        Maximum fraction of Nyquist to retain (randomized).
        twodim:           If True, apply 2D augmentations (C, H, W input).
    """

    def __init__(
        self,
        noise_std=0.05,
        kspace_mask_prob=0.3,
        kmax_frac=0.8,
        twodim=False,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.kspace_mask_prob = kspace_mask_prob
        self.kmax_frac = kmax_frac
        self.twodim = twodim

    def forward(self, x):
        """x: (C, H, W) in 2D mode, (C, D, H, W) in 3D mode."""
        x = self._random_rotation(x)
        x = self._random_flip(x)
        x = self._add_noise(x)
        if torch.rand(1).item() < self.kspace_mask_prob:
            x = self._kspace_mask(x)
        return x

    def _random_rotation(self, x):
        """Random 90-degree rotation in the H-W plane (valid for both 2D/3D)."""
        k = torch.randint(0, 4, (1,)).item()
        # Always rotate in the last two spatial dims (H, W)
        spatial_dims = (-2, -1)
        return torch.rot90(x, k=k, dims=spatial_dims)

    def _random_flip(self, x):
        # Flip along all spatial dims
        for dim in range(1, x.ndim):
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[dim])
        return x

    def _add_noise(self, x):
        return x + self.noise_std * torch.randn_like(x)

    def _kspace_mask(self, x):
        """
        Mask high-k modes in Fourier space to simulate lower resolution.
        Handles both 2D (C, H, W) and 3D (C, D, H, W) inputs.
        """
        spatial_dims = tuple(range(1, x.ndim))
        xf = torch.fft.fftn(x, dim=spatial_dims)

        # Build k-space grid over spatial dims
        shape = x.shape[1:]
        grids = [torch.fft.fftfreq(s) for s in shape]
        mesh = torch.meshgrid(*grids, indexing='ij')
        kgrid = torch.sqrt(sum(g ** 2 for g in mesh)).to(x.device)

        # Random cutoff fraction
        kmax = 0.5 * self.kmax_frac * (0.5 + 0.5 * torch.rand(1).item())
        mask = (kgrid < kmax).float()

        xf = xf * mask.unsqueeze(0)
        return torch.fft.ifftn(xf, dim=spatial_dims).real


# ---------------------------------------------------------------------------
# Model components — 3D
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """
    VGG-style 3D conv block: (Conv3d -> BN -> GELU) x num_convs -> MaxPool3d.
    """

    def __init__(self, in_channels, out_channels, num_convs=2, pool=True):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv3d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
                nn.GELU(),
            ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpatialSelfAttention3D(nn.Module):
    """
    Multi-head self-attention over spatial tokens of a 3D feature map.

    The feature map (B, C, D, H, W) is flattened to D*H*W tokens of
    dimension C. Sinusoidal 3D positional encodings are added before
    attention, giving a global receptive field.

    Args:
        channels:  Number of feature channels (token dimension).
        num_heads: Number of attention heads.
        dropout:   Attention dropout probability.
    """

    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels ({channels}) must be divisible by num_heads ({num_heads})'

        self.channels = channels
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def _positional_encoding_3d(self, D, H, W, device):
        """Sinusoidal 3D positional encoding. Returns (D*H*W, channels)."""
        d_model = self.channels
        d_each = d_model // 3
        remainder = d_model - 3 * d_each

        def sin_enc(positions, d):
            pe = torch.zeros(len(positions), d, device=device)
            div = torch.exp(
                torch.arange(0, d, 2, device=device).float()
                * -(math.log(10000.0) / d)
            )
            pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div)
            pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div[: d // 2])
            return pe

        d_pos = torch.arange(D, device=device).float()
        h_pos = torch.arange(H, device=device).float()
        w_pos = torch.arange(W, device=device).float()

        d_grid, h_grid, w_grid = torch.meshgrid(d_pos, h_pos, w_pos, indexing='ij')
        pe_d = sin_enc(d_grid.reshape(-1), d_each)
        pe_h = sin_enc(h_grid.reshape(-1), d_each)
        pe_w = sin_enc(w_grid.reshape(-1), d_each + remainder)

        return torch.cat([pe_d, pe_h, pe_w], dim=-1)  # (D*H*W, channels)

    def forward(self, x):
        """x: (B, C, D, H, W)"""
        B, C, D, H, W = x.shape
        N = D * H * W

        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)
        pe = self._positional_encoding_3d(D, H, W, x.device)
        tokens = tokens + pe.unsqueeze(0)

        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.norm2(tokens))

        return tokens.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)


# ---------------------------------------------------------------------------
# Model components — 2D
# ---------------------------------------------------------------------------

class ConvBlock2D(nn.Module):
    """
    VGG-style 2D conv block: (Conv2d -> BN -> GELU) x num_convs -> MaxPool2d.
    """

    def __init__(self, in_channels, out_channels, num_convs=2, pool=True):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpatialSelfAttention2D(nn.Module):
    """
    Multi-head self-attention over spatial tokens of a 2D feature map.

    The feature map (B, C, H, W) is flattened to H*W tokens of dimension C.
    Sinusoidal 2D positional encodings are added before attention.

    Args:
        channels:  Number of feature channels (token dimension).
        num_heads: Number of attention heads.
        dropout:   Attention dropout probability.
    """

    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels ({channels}) must be divisible by num_heads ({num_heads})'

        self.channels = channels
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def _positional_encoding_2d(self, H, W, device):
        """Sinusoidal 2D positional encoding. Returns (H*W, channels)."""
        d_model = self.channels
        d_half = d_model // 2
        remainder = d_model - 2 * d_half

        def sin_enc(positions, d):
            pe = torch.zeros(len(positions), d, device=device)
            div = torch.exp(
                torch.arange(0, d, 2, device=device).float()
                * -(math.log(10000.0) / d)
            )
            pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div)
            pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div[: d // 2])
            return pe

        h_pos = torch.arange(H, device=device).float()
        w_pos = torch.arange(W, device=device).float()

        h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing='ij')
        pe_h = sin_enc(h_grid.reshape(-1), d_half)
        pe_w = sin_enc(w_grid.reshape(-1), d_half + remainder)

        return torch.cat([pe_h, pe_w], dim=-1)  # (H*W, channels)

    def forward(self, x):
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        N = H * W

        tokens = x.permute(0, 2, 3, 1).reshape(B, N, C)
        pe = self._positional_encoding_2d(H, W, x.device)
        tokens = tokens + pe.unsqueeze(0)

        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.norm2(tokens))

        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class VGGEncoder3D(nn.Module):
    """
    VGG-style 3D CNN encoder with mid-network self-attention.

    Architecture:
        Blocks 1-4: progressive downsampling (each halves spatial dims)
        Attention:  global receptive field at coarsest spatial resolution
        Block 5:    further refinement post-attention (no pool)
        GAP + MLP:  collapse to fixed-size embedding

    Args:
        in_channels:   Number of input field channels.
        base_channels: Filters in block 1 (doubles each block).
        embed_dim:     Output embedding dimension.
        attn_heads:    Number of self-attention heads.
        attn_dropout:  Dropout in attention layers.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        embed_dim=256,
        attn_heads=8,
        attn_dropout=0.1,
    ):
        super().__init__()
        c = base_channels

        self.block1 = ConvBlock3D(in_channels, c, num_convs=2, pool=True)
        self.block2 = ConvBlock3D(c, c * 2, num_convs=2, pool=True)
        self.block3 = ConvBlock3D(c * 2, c * 4, num_convs=3, pool=True)
        self.block4 = ConvBlock3D(c * 4, c * 8, num_convs=3, pool=True)

        self.attention = SpatialSelfAttention3D(
            channels=c * 8,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )

        self.block5 = ConvBlock3D(c * 8, c * 8, num_convs=2, pool=False)
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.projector = nn.Sequential(
            nn.Linear(c * 8, c * 8),
            nn.BatchNorm1d(c * 8),
            nn.GELU(),
            nn.Linear(c * 8, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, C, D, H, W) -> z: (B, embed_dim)"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.attention(x)
        x = self.block5(x)
        x = self.pool(x).flatten(1)
        return self.projector(x)


class VGGEncoder2D(nn.Module):
    """
    VGG-style 2D CNN encoder with mid-network self-attention.

    Accepts (B, 1, H, W) inputs — individual depth slices from a thinned
    3D volume. After encoding, per-slice embeddings are averaged over
    the D' slices to produce one volume-level embedding.

    Architecture mirrors VGGEncoder3D but uses 2D Conv/Pool/Attention.

    Args:
        in_channels:   Number of input channels (1 per depth slice).
        base_channels: Filters in block 1 (doubles each block).
        embed_dim:     Output embedding dimension.
        attn_heads:    Number of self-attention heads.
        attn_dropout:  Dropout in attention layers.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        embed_dim=256,
        attn_heads=8,
        attn_dropout=0.1,
    ):
        super().__init__()
        c = base_channels

        self.block1 = ConvBlock2D(in_channels, c, num_convs=2, pool=True)
        self.block2 = ConvBlock2D(c, c * 2, num_convs=2, pool=True)
        self.block3 = ConvBlock2D(c * 2, c * 4, num_convs=3, pool=True)
        self.block4 = ConvBlock2D(c * 4, c * 8, num_convs=3, pool=True)

        self.attention = SpatialSelfAttention2D(
            channels=c * 8,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )

        self.block5 = ConvBlock2D(c * 8, c * 8, num_convs=2, pool=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projector = nn.Sequential(
            nn.Linear(c * 8, c * 8),
            nn.BatchNorm1d(c * 8),
            nn.GELU(),
            nn.Linear(c * 8, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_slices(self, x):
        """
        Encode a batch of 2D slices.

        Args:
            x: (B*D', 1, H, W) — all slices across the batch flattened together

        Returns:
            z: (B*D', embed_dim)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.attention(x)
        x = self.block5(x)
        x = self.pool(x).flatten(1)
        return self.projector(x)

    def forward(self, x, n_slices):
        """
        Encode a volume represented as D' depth slices, averaging over slices.

        Args:
            x:        (B*D', 1, H, W) — flattened slices
            n_slices: D' (number of slices per volume)

        Returns:
            z: (B, embed_dim) — mean-pooled over depth slices
        """
        z_slices = self.forward_slices(x)       # (B*D', embed_dim)
        B = z_slices.shape[0] // n_slices
        z = z_slices.reshape(B, n_slices, -1).mean(dim=1)  # (B, embed_dim)
        return z


def build_encoder(args):
    """Construct the appropriate encoder based on the --twodim flag."""
    if args.twodim:
        return VGGEncoder2D(
            in_channels=1,  # one channel per 2D slice
            base_channels=args.base_channels,
            embed_dim=args.embed_dim,
            attn_heads=args.attn_heads,
        )
    else:
        return VGGEncoder3D(
            in_channels=args.in_channels,
            base_channels=args.base_channels,
            embed_dim=args.embed_dim,
            attn_heads=args.attn_heads,
        )


def encode_batch(encoder, x_batch, twodim, thin_factor):
    """
    Run the encoder on a batch, handling 2D vs 3D mode.

    3D mode:
        x_batch: (B, 1, D, H, W)  ->  z: (B, embed_dim)

    2D mode:
        x_batch: (B, D', H, W)  — D' slices per volume
        Reshape to (B*D', 1, H, W), encode, average over slices -> (B, embed_dim)

    Args:
        encoder:     VGGEncoder2D or VGGEncoder3D instance.
        x_batch:     Input tensor from the dataloader.
        twodim:      Whether we are in 2D mode.
        thin_factor: Used to determine D' (only needed for documentation; D' is
                     read from the tensor shape directly).

    Returns:
        z: (B, embed_dim)
    """
    if twodim:
        # x_batch: (B, D', H, W)
        B, D_prime, H, W = x_batch.shape
        # Reshape to (B*D', 1, H, W)
        x_2d = x_batch.reshape(B * D_prime, 1, H, W)
        return encoder(x_2d, n_slices=D_prime)
    else:
        # x_batch: (B, C, D, H, W)
        return encoder(x_batch)


# ---------------------------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------------------------

class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.

    Bardes et al., 2022 (https://arxiv.org/abs/2105.04906)

    Args:
        lambda_inv: Weight for invariance term.
        mu_var:     Weight for variance term (prevents collapse).
        nu_cov:     Weight for covariance term (decorrelates dimensions).
        gamma:      Target standard deviation for variance term.
        eps:        Numerical stability constant.
    """

    def __init__(
        self,
        lambda_inv=25.0,
        mu_var=25.0,
        nu_cov=1.0,
        gamma=1.0,
        eps=1e-4,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.mu_var = mu_var
        self.nu_cov = nu_cov
        self.gamma = gamma
        self.eps = eps

    def forward(self, z_a, z_b):
        """
        Args:
            z_a: (B, D) embeddings from view a
            z_b: (B, D) embeddings from view b

        Returns:
            loss:       Scalar total loss
            components: Dict of individual terms for logging
        """
        B, D = z_a.shape

        # Invariance: MSE between positive pairs
        inv_loss = F.mse_loss(z_a, z_b)

        # Variance: hinge on per-dimension std
        def variance_loss(z):
            std = torch.sqrt(z.var(dim=0) + self.eps)
            return F.relu(self.gamma - std).mean()

        var_loss = 0.5 * (variance_loss(z_a) + variance_loss(z_b))

        # Covariance: penalize off-diagonal elements
        def covariance_loss(z):
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (B - 1)
            off_diag = cov ** 2
            off_diag.fill_diagonal_(0.0)
            return off_diag.sum() / D

        cov_loss = 0.5 * (covariance_loss(z_a) + covariance_loss(z_b))

        loss = (
            self.lambda_inv * inv_loss
            + self.mu_var * var_loss
            + self.nu_cov * cov_loss
        )

        return loss, {
            'inv': inv_loss.item(),
            'var': var_loss.item(),
            'cov': cov_loss.item(),
            'total': loss.item(),
        }


# ---------------------------------------------------------------------------
# Full training model (encoder + heads)
# ---------------------------------------------------------------------------

class CosmologicalEncoderModel(nn.Module):
    """
    Full model wrapping the encoder with:
        - VICReg contrastive objective
        - Optional cosmological parameter regression head (Om, s8)

    Args:
        encoder:           VGGEncoder2D or VGGEncoder3D.
        embed_dim:         Encoder output dimension.
        n_params:          Number of cosmological parameters to regress.
        regression_weight: Weight for regression loss relative to VICReg.
        twodim:            Whether encoder operates in 2D slice mode.
        thin_factor:       D-axis thinning factor (2D mode only).
    """

    def __init__(
        self,
        encoder,
        embed_dim=256,
        n_params=2,
        regression_weight=0.1,
        twodim=False,
        thin_factor=8,
    ):
        super().__init__()
        self.encoder = encoder
        self.vicreg = VICRegLoss()
        self.regression_weight = regression_weight
        self.twodim = twodim
        self.thin_factor = thin_factor

        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_params),
        )

    def forward(self, x_a, x_b, params=None):
        """
        Args:
            x_a, x_b: Input tensors from the dataloader.
                       3D mode: (B, C, D, H, W)
                       2D mode: (B, D', H, W)
            params:    (B, n_params) cosmological parameters, or None.

        Returns:
            loss: Total scalar loss.
            logs: Dict of loss components for logging.
        """
        z_a = encode_batch(self.encoder, x_a, self.twodim, self.thin_factor)
        z_b = encode_batch(self.encoder, x_b, self.twodim, self.thin_factor)

        vicreg_loss, vicreg_logs = self.vicreg(z_a, z_b)
        logs = {f'vicreg/{k}': v for k, v in vicreg_logs.items()}
        loss = vicreg_loss

        if params is not None and self.regression_weight > 0:
            z_avg = 0.5 * (z_a + z_b)
            pred_params = self.reg_head(z_avg)
            reg_loss = F.mse_loss(pred_params, params)
            loss = loss + self.regression_weight * reg_loss
            logs['reg/mse'] = reg_loss.item()

        logs['loss/total'] = loss.item()
        return loss, logs

    @torch.no_grad()
    def embed(self, x):
        """Extract embedding for a single batch (no grad)."""
        return encode_batch(self.encoder, x, self.twodim, self.thin_factor)


# ---------------------------------------------------------------------------
# FID / novelty utilities
# ---------------------------------------------------------------------------

def compute_embeddings(model, dataloader, device):
    """Extract embeddings for all samples in a dataloader. Returns (N, D) array."""
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            z = model.embed(x)
            all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Frechet distance between N(mu1, sigma1) and N(mu2, sigma2).

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
    """
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    mean_term = float(diff @ diff)

    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError('Matrix square root has large imaginary component')
        covmean = covmean.real

    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return mean_term + float(trace_term)


def fit_gaussian_ledoitwolf(embeddings):
    """
    Fit a Gaussian with Ledoit-Wolf shrinkage covariance.
    More stable than sample covariance when N is not >> D.

    Returns:
        mu:    (D,) mean vector
        sigma: (D, D) covariance matrix
    """
    from sklearn.covariance import LedoitWolf

    mu = embeddings.mean(axis=0)
    sigma = LedoitWolf().fit(embeddings).covariance_
    return mu, sigma


def novelty_scores(z_new, z_train, k=5):
    """
    Per-sample novelty and memorization scores.

    Args:
        z_new:   (D,) embedding of the new sample
        z_train: (N, D) training set embeddings
        k:       k for kNN distance

    Returns dict with:
        mahalanobis:         Distance from training distribution
        max_cosine_similarity: Risk of memorization (high = bad)
        knn_dist_k{k}:       Mean distance to k nearest neighbours
    """
    from sklearn.metrics.pairwise import cosine_similarity

    mu, sigma = fit_gaussian_ledoitwolf(z_train)
    sigma_inv = np.linalg.pinv(sigma)

    diff = z_new - mu
    mahal = float(diff @ sigma_inv @ diff)

    cos_sims = cosine_similarity(z_new.reshape(1, -1), z_train)[0]
    max_cos = float(cos_sims.max())

    dists = np.linalg.norm(z_train - z_new, axis=1)
    knn_dist = float(np.sort(dists)[:k].mean())

    return {
        'mahalanobis': mahal,
        'max_cosine_similarity': max_cos,
        f'knn_dist_k{k}': knn_dist,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Mode: {"2D" if args.twodim else "3D"}')

    augment = CosmologicalAugment(
        noise_std=args.noise_std,
        kspace_mask_prob=args.kspace_mask_prob,
        twodim=args.twodim,
    )

    dataset = CAMELSDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        transform=augment,
        twodim=args.twodim,
        thin_factor=args.thin_factor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    encoder = build_encoder(args)
    model = CosmologicalEncoderModel(
        encoder=encoder,
        embed_dim=args.embed_dim,
        regression_weight=args.regression_weight,
        twodim=args.twodim,
        thin_factor=args.thin_factor,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step):
        warmup = args.warmup_steps
        total = args.epochs * len(dataloader)
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_logs = {}

        for batch in dataloader:
            x_a, x_b, params = batch
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            params = params.to(device) if args.regression_weight > 0 else None

            optimizer.zero_grad()
            loss, logs = model(x_a, x_b, params)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            for k, v in logs.items():
                epoch_logs[k] = epoch_logs.get(k, 0) + v

        n_batches = len(dataloader)
        epoch_logs = {k: v / n_batches for k, v in epoch_logs.items()}

        total = epoch_logs.get('loss/total', 0)
        lr = scheduler.get_last_lr()[0]
        print(
            f'Epoch {epoch+1:4d}/{args.epochs} | '
            f'loss={total:.4f} | '
            f'inv={epoch_logs.get("vicreg/inv", 0):.4f} | '
            f'var={epoch_logs.get("vicreg/var", 0):.4f} | '
            f'cov={epoch_logs.get("vicreg/cov", 0):.4f} | '
            f'lr={lr:.2e}'
        )

        if total < best_loss:
            best_loss = total
            torch.save(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'encoder_state': encoder.state_dict(),
                    'args': vars(args),
                    'loss': best_loss,
                },
                args.checkpoint,
            )
            print(f'  -> Saved checkpoint (loss={best_loss:.4f})')


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def embed(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt['args'])

    encoder = build_encoder(ckpt_args)
    model = CosmologicalEncoderModel(
        encoder=encoder,
        embed_dim=ckpt_args.embed_dim,
        twodim=ckpt_args.twodim,
        thin_factor=ckpt_args.thin_factor,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    dataset = CAMELSDataset(
        data_dir=args.data_dir,
        resolution=ckpt_args.resolution,
        twodim=ckpt_args.twodim,
        thin_factor=ckpt_args.thin_factor,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    embeddings = compute_embeddings(model, dataloader, device)
    out_path = Path(args.data_dir) / 'embeddings.npy'
    np.save(out_path, embeddings)
    print(f'Saved {embeddings.shape} embeddings to {out_path}')


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt['args'])

    encoder = build_encoder(ckpt_args)
    model = CosmologicalEncoderModel(
        encoder=encoder,
        embed_dim=ckpt_args.embed_dim,
        twodim=ckpt_args.twodim,
        thin_factor=ckpt_args.thin_factor,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    res = ckpt_args.resolution
    twodim = ckpt_args.twodim
    thin = ckpt_args.thin_factor

    train_ds = CAMELSDataset(data_dir=args.train_dir, resolution=res,
                              twodim=twodim, thin_factor=thin)
    gen_ds = CAMELSDataset(data_dir=args.gen_dir, resolution=res,
                            twodim=twodim, thin_factor=thin)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, shuffle=False)

    print('Embedding training set...')
    z_train = compute_embeddings(model, train_loader, device)

    print('Embedding generated set...')
    z_gen = compute_embeddings(model, gen_loader, device)

    mu_train, sigma_train = fit_gaussian_ledoitwolf(z_train)
    mu_gen, sigma_gen = fit_gaussian_ledoitwolf(z_gen)

    fid = frechet_distance(mu_train, sigma_train, mu_gen, sigma_gen)
    print(f'\nFID: {fid:.4f}')

    print('\nNovelty scores (first 5 generated samples):')
    for i in range(min(5, len(z_gen))):
        scores = novelty_scores(z_gen[i], z_train)
        print(f'  Sample {i}: {scores}')

    return fid
