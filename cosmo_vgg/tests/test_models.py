"""Unit tests for cosmo_vgg.models."""

from pathlib import Path

import numpy as np
import pytest
import torch

from cosmo_vgg.models import (
    CAMELSDataset,
    CosmologicalAugment,
    CosmologicalEncoderModel,
    ConvBlock2D,
    ConvBlock3D,
    SpatialSelfAttention2D,
    SpatialSelfAttention3D,
    VGGEncoder2D,
    VGGEncoder3D,
    VICRegLoss,
    frechet_distance,
    fit_gaussian_ledoitwolf,
    novelty_scores,
)

# ---------------------------------------------------------------------------
# Shared constants — kept small to run fast on CPU
# ---------------------------------------------------------------------------
B = 2         # batch size
C = 1         # input channels
D = H = W = 16
BASE_CH = 8
EMBED_DIM = 32
ATTN_HEADS = 4  # must divide BASE_CH * 8 = 64

# Real test data bundled with the package
DATA_FILE = Path(__file__).parent.parent / "data" / "Grids_HI_IllustrisTNG_CV_128_z=0_small.npy"

# IllustrisTNG CV uses Planck 2015 cosmology
_CV_OM, _CV_S8 = 0.3089, 0.8159


@pytest.fixture(scope="module")
def real_grids():
    """Load all 6 HI grids from the bundled test data. Shape: (6, 128, 128, 128)."""
    return np.load(DATA_FILE)


@pytest.fixture(scope="module")
def real_data_dir(tmp_path_factory):
    """
    Materialize the 6 real HI grids as CAMELS-format .npy files in a temp dir.
    All grids share the same CV cosmology (Om=0.3089, s8=0.8159), different seeds.
    """
    grids = np.load(DATA_FILE)
    tmp = tmp_path_factory.mktemp("camels_real")
    for i, grid in enumerate(grids):
        np.save(tmp / f"cosmo_{_CV_OM}_{_CV_S8}_seed{i}_HI.npy", grid)
    return tmp


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------

class TestConvBlock3D:
    def test_output_shape_with_pool(self):
        block = ConvBlock3D(C, BASE_CH, num_convs=2, pool=True)
        x = torch.randn(B, C, D, H, W)
        out = block(x)
        assert out.shape == (B, BASE_CH, D // 2, H // 2, W // 2)

    def test_output_shape_no_pool(self):
        block = ConvBlock3D(C, BASE_CH, num_convs=2, pool=False)
        x = torch.randn(B, C, D, H, W)
        out = block(x)
        assert out.shape == (B, BASE_CH, D, H, W)


class TestConvBlock2D:
    def test_output_shape_with_pool(self):
        block = ConvBlock2D(C, BASE_CH, num_convs=2, pool=True)
        x = torch.randn(B, C, H, W)
        out = block(x)
        assert out.shape == (B, BASE_CH, H // 2, W // 2)

    def test_output_shape_no_pool(self):
        block = ConvBlock2D(C, BASE_CH, num_convs=2, pool=False)
        x = torch.randn(B, C, H, W)
        out = block(x)
        assert out.shape == (B, BASE_CH, H, W)


# ---------------------------------------------------------------------------
# Self-attention
# ---------------------------------------------------------------------------

class TestSpatialSelfAttention3D:
    def test_output_shape(self):
        channels = BASE_CH * 8  # 64
        attn = SpatialSelfAttention3D(channels=channels, num_heads=ATTN_HEADS)
        # Small spatial dims so the test is fast
        x = torch.randn(B, channels, 2, 2, 2)
        out = attn(x)
        assert out.shape == x.shape

    def test_channels_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            SpatialSelfAttention3D(channels=7, num_heads=4)


class TestSpatialSelfAttention2D:
    def test_output_shape(self):
        channels = BASE_CH * 8
        attn = SpatialSelfAttention2D(channels=channels, num_heads=ATTN_HEADS)
        x = torch.randn(B, channels, 4, 4)
        out = attn(x)
        assert out.shape == x.shape

    def test_channels_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            SpatialSelfAttention2D(channels=7, num_heads=4)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class TestVGGEncoder3D:
    @pytest.fixture
    def encoder(self):
        return VGGEncoder3D(
            in_channels=C,
            base_channels=BASE_CH,
            embed_dim=EMBED_DIM,
            attn_heads=ATTN_HEADS,
        )

    def test_output_shape(self, encoder):
        # 64^3 -> 5 blocks each halving once -> 2^3 before GAP
        x = torch.randn(B, C, 64, 64, 64)
        z = encoder(x)
        assert z.shape == (B, EMBED_DIM)

    def test_output_is_finite(self, encoder):
        x = torch.randn(B, C, 64, 64, 64)
        z = encoder(x)
        assert torch.isfinite(z).all()


class TestVGGEncoder2D:
    @pytest.fixture
    def encoder(self):
        return VGGEncoder2D(
            in_channels=C,
            base_channels=BASE_CH,
            embed_dim=EMBED_DIM,
            attn_heads=ATTN_HEADS,
        )

    def test_forward_slices_shape(self, encoder):
        n_slices = 4
        x = torch.randn(B * n_slices, C, 64, 64)
        z = encoder.forward_slices(x)
        assert z.shape == (B * n_slices, EMBED_DIM)

    def test_forward_averages_slices(self, encoder):
        n_slices = 4
        x = torch.randn(B * n_slices, C, 64, 64)
        z = encoder(x, n_slices=n_slices)
        assert z.shape == (B, EMBED_DIM)

    def test_output_is_finite(self, encoder):
        n_slices = 4
        x = torch.randn(B * n_slices, C, 64, 64)
        z = encoder(x, n_slices=n_slices)
        assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# VICRegLoss
# ---------------------------------------------------------------------------

class TestVICRegLoss:
    @pytest.fixture
    def loss_fn(self):
        return VICRegLoss()

    def test_returns_scalar_and_dict(self, loss_fn):
        z_a = torch.randn(8, EMBED_DIM)
        z_b = torch.randn(8, EMBED_DIM)
        loss, logs = loss_fn(z_a, z_b)
        assert loss.shape == ()
        assert set(logs) == {'inv', 'var', 'cov', 'total'}

    def test_identical_inputs_zero_invariance(self, loss_fn):
        z = torch.randn(8, EMBED_DIM)
        loss, logs = loss_fn(z, z)
        assert logs['inv'] == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_non_negative(self, loss_fn):
        z_a = torch.randn(8, EMBED_DIM)
        z_b = torch.randn(8, EMBED_DIM)
        loss, _ = loss_fn(z_a, z_b)
        assert loss.item() >= 0.0

    def test_loss_is_differentiable(self, loss_fn):
        z_a = torch.randn(8, EMBED_DIM, requires_grad=True)
        z_b = torch.randn(8, EMBED_DIM, requires_grad=True)
        loss, _ = loss_fn(z_a, z_b)
        loss.backward()
        assert z_a.grad is not None
        assert z_b.grad is not None


# ---------------------------------------------------------------------------
# CosmologicalAugment
# ---------------------------------------------------------------------------

class TestCosmologicalAugment:
    def test_3d_shape_preserved(self):
        aug = CosmologicalAugment(noise_std=0.05, kspace_mask_prob=1.0, twodim=False)
        x = torch.randn(C, D, H, W)
        out = aug(x)
        assert out.shape == x.shape

    def test_2d_shape_preserved(self):
        aug = CosmologicalAugment(noise_std=0.05, kspace_mask_prob=1.0, twodim=True)
        x = torch.randn(C, H, W)
        out = aug(x)
        assert out.shape == x.shape

    def test_noise_changes_values(self):
        aug = CosmologicalAugment(noise_std=1.0, kspace_mask_prob=0.0)
        x = torch.zeros(C, H, W)
        out = aug(x)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# CosmologicalEncoderModel
# ---------------------------------------------------------------------------

class TestCosmologicalEncoderModel:
    @pytest.fixture
    def model_3d(self):
        encoder = VGGEncoder3D(C, BASE_CH, EMBED_DIM, ATTN_HEADS)
        return CosmologicalEncoderModel(encoder, embed_dim=EMBED_DIM, twodim=False)

    def test_forward_returns_loss_and_logs(self, model_3d):
        x_a = torch.randn(B, C, 64, 64, 64)
        x_b = torch.randn(B, C, 64, 64, 64)
        params = torch.randn(B, 2)
        loss, logs = model_3d(x_a, x_b, params=params)
        assert loss.shape == ()
        assert 'loss/total' in logs

    def test_embed_no_grad(self, model_3d):
        x = torch.randn(B, C, 64, 64, 64)
        z = model_3d.embed(x)
        assert z.shape == (B, EMBED_DIM)

    def test_no_regression_without_params(self, model_3d):
        x_a = torch.randn(B, C, 64, 64, 64)
        x_b = torch.randn(B, C, 64, 64, 64)
        loss, logs = model_3d(x_a, x_b, params=None)
        assert 'reg/mse' not in logs


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

class TestFrechetDistance:
    def test_identical_distributions_is_zero(self):
        D = 16
        mu = np.zeros(D)
        sigma = np.eye(D)
        fid = frechet_distance(mu, sigma, mu, sigma)
        assert fid == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        D = 8
        mu1, mu2 = rng.standard_normal(D), rng.standard_normal(D)
        A = rng.standard_normal((D, D))
        sigma = A @ A.T + np.eye(D)
        fid = frechet_distance(mu1, sigma, mu2, sigma)
        assert fid >= 0.0


class TestFitGaussianLedoitWolf:
    def test_shapes(self):
        rng = np.random.default_rng(1)
        embeddings = rng.standard_normal((50, 16))
        mu, sigma = fit_gaussian_ledoitwolf(embeddings)
        assert mu.shape == (16,)
        assert sigma.shape == (16, 16)

    def test_covariance_is_positive_semidefinite(self):
        rng = np.random.default_rng(2)
        embeddings = rng.standard_normal((50, 16))
        _, sigma = fit_gaussian_ledoitwolf(embeddings)
        eigvals = np.linalg.eigvalsh(sigma)
        assert (eigvals >= -1e-8).all()


class TestNoveltyScores:
    def test_keys_present(self):
        rng = np.random.default_rng(3)
        z_train = rng.standard_normal((30, 16))
        z_new = rng.standard_normal(16)
        scores = novelty_scores(z_new, z_train, k=3)
        assert 'mahalanobis' in scores
        assert 'max_cosine_similarity' in scores
        assert 'knn_dist_k3' in scores

    def test_max_cosine_similarity_in_range(self):
        rng = np.random.default_rng(4)
        z_train = rng.standard_normal((30, 16))
        z_new = rng.standard_normal(16)
        scores = novelty_scores(z_new, z_train)
        assert -1.0 <= scores['max_cosine_similarity'] <= 1.0


# ---------------------------------------------------------------------------
# CAMELSDataset — synthetic edge-case tests
# ---------------------------------------------------------------------------

class TestCAMELSDataset:
    @pytest.fixture
    def synth_data_dir(self, tmp_path):
        """Two cosmologies, two seeds each — minimal synthetic grids."""
        for om, s8 in [(0.30, 0.80), (0.25, 0.70)]:
            for seed in range(2):
                arr = np.random.rand(16, 16, 16).astype(np.float32)
                np.save(tmp_path / f"cosmo_{om}_{s8}_seed{seed}_Mgas.npy", arr)
        return tmp_path

    def test_len_equals_n_cosmologies(self, synth_data_dir):
        ds = CAMELSDataset(synth_data_dir)
        assert len(ds) == 2

    def test_getitem_shapes_3d(self, synth_data_dir):
        ds = CAMELSDataset(synth_data_dir, twodim=False)
        x_a, x_b, params = ds[0]
        assert x_a.ndim == 4        # (1, D, H, W)
        assert x_a.shape[0] == 1    # channel dim
        assert params.shape == (2,)

    def test_getitem_shapes_2d(self, synth_data_dir):
        ds = CAMELSDataset(synth_data_dir, twodim=True, thin_factor=4)
        x_a, x_b, params = ds[0]
        assert x_a.ndim == 3        # (D', H, W)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CAMELSDataset(tmp_path / "nonexistent")

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CAMELSDataset(tmp_path)


# ---------------------------------------------------------------------------
# CAMELSDataset — real data tests
# ---------------------------------------------------------------------------

class TestCAMELSDatasetRealData:
    def test_one_cosmology_six_seeds(self, real_data_dir):
        ds = CAMELSDataset(real_data_dir)
        assert len(ds) == 1
        assert len(list(ds.cosmo_groups.values())[0]) == 6

    def test_cosmology_params_parsed_correctly(self, real_data_dir):
        ds = CAMELSDataset(real_data_dir)
        key = ds.cosmo_keys[0]
        assert key == pytest.approx((_CV_OM, _CV_S8), abs=1e-4)

    def test_3d_tensor_shape(self, real_data_dir):
        ds = CAMELSDataset(real_data_dir, twodim=False)
        x_a, x_b, params = ds[0]
        assert x_a.ndim == 4        # (1, D, H, W)
        assert x_a.shape[0] == 1
        assert params.shape == (2,)

    def test_2d_tensor_shape(self, real_data_dir):
        ds = CAMELSDataset(real_data_dir, twodim=True, thin_factor=8)
        x_a, _, _ = ds[0]
        assert x_a.ndim == 3        # (D', H, W)
        assert x_a.shape[0] == 128 // 8  # thin_factor=8

    def test_resolution_resize(self, real_data_dir):
        ds = CAMELSDataset(real_data_dir, resolution=32)
        x_a, _, _ = ds[0]
        assert x_a.shape == (1, 32, 32, 32)

    def test_preprocessing_finite(self, real_data_dir):
        """log1p + z-score normalization must produce finite tensors on real data."""
        ds = CAMELSDataset(real_data_dir)
        x_a, x_b, _ = ds[0]
        assert torch.isfinite(x_a).all(), "x_a contains non-finite values"
        assert torch.isfinite(x_b).all(), "x_b contains non-finite values"

    def test_preprocessing_normalized(self, real_data_dir):
        """After log1p + z-score, each volume should have mean≈0 and std≈1."""
        ds = CAMELSDataset(real_data_dir)
        x_a, _, _ = ds[0]
        assert x_a.mean().abs() < 0.1
        assert (x_a.std() - 1.0).abs() < 0.1

    def test_preprocessing_compresses_dynamic_range(self, real_grids):
        """
        Raw HI fields span ~12 orders of magnitude. After log1p the range
        should be finite and much more compact.
        """
        raw = real_grids[0]
        assert raw.max() > 1e6, "test assumption: raw data has large dynamic range"
        compressed = np.log1p(np.clip(raw, 0, None))
        assert np.isfinite(compressed).all()
        assert compressed.max() < 30   # log1p(4.6e12) ≈ 29.1

    def test_positive_pair_different_files(self, real_data_dir):
        """With 6 seeds, __getitem__ should always return two distinct grids."""
        ds = CAMELSDataset(real_data_dir)
        x_a, x_b, _ = ds[0]
        # Different seeds → different values (extremely unlikely to be identical)
        assert not torch.allclose(x_a, x_b)


# ---------------------------------------------------------------------------
# Encoder on real data
# ---------------------------------------------------------------------------

class TestEncoderRealData:
    """Smoke tests: encoder produces finite, correctly-shaped output on real grids."""

    @pytest.fixture(scope="class")
    def preprocessed(self, real_grids):
        """Return two preprocessed 3D tensors at resolution=32 for speed."""
        results = []
        for raw in real_grids[:2]:
            grid = np.log1p(np.clip(raw, 0, None)).astype(np.float32)
            grid = (grid - grid.mean()) / (grid.std() + 1e-8)
            t = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128, 128)
            import torch.nn.functional as F
            t = F.interpolate(t, size=(32, 32, 32), mode='trilinear', align_corners=False)
            results.append(t)
        return torch.cat(results, dim=0)  # (2, 1, 32, 32, 32)

    def test_encoder_3d_output_shape(self, preprocessed):
        encoder = VGGEncoder3D(C, BASE_CH, EMBED_DIM, ATTN_HEADS)
        encoder.eval()
        with torch.no_grad():
            z = encoder(preprocessed)
        assert z.shape == (2, EMBED_DIM)

    def test_encoder_3d_output_finite(self, preprocessed):
        encoder = VGGEncoder3D(C, BASE_CH, EMBED_DIM, ATTN_HEADS)
        encoder.eval()
        with torch.no_grad():
            z = encoder(preprocessed)
        assert torch.isfinite(z).all()

    def test_augment_on_real_data(self, preprocessed):
        aug = CosmologicalAugment(noise_std=0.05, kspace_mask_prob=1.0, twodim=False)
        x = preprocessed[0]  # (1, 32, 32, 32)
        out = aug(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
