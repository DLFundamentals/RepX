"""Tests for letorch.rsa

Run with:  pytest tests/
"""

import pytest
import torch

from letorch.rsa import RSA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _orthogonal_pair(n: int = 50, d_total: int = 256, seed: int = 0):
    """Return two matrices from orthogonal column subspaces."""
    g = torch.Generator()
    g.manual_seed(seed)
    M = torch.randn(n, d_total, generator=g)
    Q, _ = torch.linalg.qr(M.T)   # (d_total, n)
    Q = Q.T                         # (n, d_total)
    d = d_total // 2
    return Q[:, :d], Q[:, d:]


# ---------------------------------------------------------------------------
# compute_rdm
# ---------------------------------------------------------------------------

class TestComputeRDM:
    def test_output_shape(self):
        X = torch.randn(20, 64)
        assert RSA().compute_rdm(X).shape == (20, 20)

    @pytest.mark.parametrize("metric", ["correlation", "cosine", "euclidean", "cityblock"])
    def test_diagonal_zero(self, metric):
        X = torch.randn(20, 64)
        rdm = RSA(rdm_metric=metric).compute_rdm(X)
        assert rdm.diagonal().abs().max().item() < 1e-5

    @pytest.mark.parametrize("metric", ["correlation", "cosine", "euclidean", "cityblock"])
    def test_symmetry(self, metric):
        X = torch.randn(20, 64)
        rdm = RSA(rdm_metric=metric).compute_rdm(X)
        assert (rdm - rdm.T).abs().max().item() < 1e-5

    @pytest.mark.parametrize("metric", ["correlation", "cosine", "euclidean", "cityblock"])
    def test_non_negative(self, metric):
        X = torch.randn(20, 64)
        assert RSA(rdm_metric=metric).compute_rdm(X).min().item() >= -1e-6

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            RSA(rdm_metric="minkowski")


# ---------------------------------------------------------------------------
# rdm_upper_tri
# ---------------------------------------------------------------------------

class TestRdmUpperTri:
    def test_length(self):
        n = 20
        vec = RSA().rdm_upper_tri(torch.zeros(n, n))
        assert vec.shape[0] == n * (n - 1) // 2

    def test_values(self):
        rdm = torch.zeros(3, 3)
        rdm[0, 1] = 1.0
        rdm[0, 2] = 2.0
        rdm[1, 2] = 3.0
        rdm = rdm + rdm.T  # make symmetric
        vec = RSA().rdm_upper_tri(rdm)
        assert torch.allclose(vec, torch.tensor([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# rsa — identical matrices (upper bound = 1.0)
# ---------------------------------------------------------------------------

class TestRSAIdentical:
    @pytest.mark.parametrize("metric", ["correlation", "cosine", "euclidean", "cityblock"])
    def test_identical_spearman(self, metric):
        X = torch.randn(30, 64)
        r = RSA(rdm_metric=metric, compare="spearman").rsa(X, X)
        assert abs(r.item() - 1.0) < 1e-4, f"metric={metric}: r={r.item()}"

    @pytest.mark.parametrize("metric", ["correlation", "cosine", "euclidean", "cityblock"])
    def test_identical_pearson(self, metric):
        X = torch.randn(30, 64)
        r = RSA(rdm_metric=metric, compare="pearson").rsa(X, X)
        assert abs(r.item() - 1.0) < 1e-4, f"metric={metric}: r={r.item()}"


# ---------------------------------------------------------------------------
# rsa — invariances (correlation-distance metric)
# ---------------------------------------------------------------------------

class TestRSAInvariances:
    def test_scaling_invariance(self):
        """correlation-distance RSA is invariant to isotropic row scaling."""
        X = torch.randn(50, 64)
        assert abs(RSA().rsa(X, X * 100).item() - 1.0) < 1e-4

    def test_translation_invariance(self):
        """correlation-distance RSA is invariant to adding the same scalar to all elements."""
        X = torch.randn(50, 64)
        assert abs(RSA().rsa(X, X + 999.0).item() - 1.0) < 1e-4

    def test_combined_scaling_and_translation(self):
        X = torch.randn(50, 64)
        assert abs(RSA().rsa(X, X * 7 + 42).item() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# rsa — orthogonal matrices (near zero)
# ---------------------------------------------------------------------------

class TestRSAOrthogonal:
    def test_independent_random_near_zero(self):
        """Two fully independent random matrices should give RSA close to 0."""
        torch.manual_seed(99)
        X_a = torch.randn(50, 128)
        Y_a = torch.randn(50, 128)
        r = RSA().rsa(X_a, Y_a)
        assert abs(r.item()) < 0.15

    def test_random_gaussian_mean_near_zero(self):
        """Mean RSA over many independent pairs should be ≈ 0."""
        torch.manual_seed(0)
        rsa = RSA()
        scores = [
            rsa.rsa(torch.randn(50, 128), torch.randn(50, 128)).item()
            for _ in range(100)
        ]
        assert abs(sum(scores) / len(scores)) < 0.05


# ---------------------------------------------------------------------------
# rsa — different feature dimensionalities
# ---------------------------------------------------------------------------

class TestRSAMixedDims:
    def test_different_feature_dims(self):
        X = torch.randn(30, 64)
        Y = torch.randn(30, 256)
        r = RSA().rsa(X, Y)
        assert r.shape == torch.Size([])   # scalar

    def test_mismatched_stimuli_raises(self):
        with pytest.raises(ValueError, match="same number of stimuli"):
            RSA().rsa(torch.randn(30, 64), torch.randn(40, 64))


# ---------------------------------------------------------------------------
# rsa — error handling
# ---------------------------------------------------------------------------

class TestRSAErrors:
    def test_unknown_compare_raises(self):
        with pytest.raises(ValueError, match="compare method"):
            RSA(compare="kendall")

    def test_unknown_rdm_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            RSA(rdm_metric="hamming")


# ---------------------------------------------------------------------------
# rsa — GPU (skipped if CUDA unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestRSAGPU:
    def test_identical_on_gpu(self):
        X = torch.randn(30, 64, device="cuda")
        r = RSA().rsa(X, X)
        assert r.device.type == "cuda"
        assert abs(r.item() - 1.0) < 1e-4

    def test_random_on_gpu(self):
        X = torch.randn(30, 64, device="cuda")
        Y = torch.randn(30, 64, device="cuda")
        r = RSA().rsa(X, Y)
        assert r.device.type == "cuda"


# ---------------------------------------------------------------------------
# Cross-validation against scipy (optional — skipped if scipy absent)
# ---------------------------------------------------------------------------

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr as scipy_spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False


@pytest.mark.skipif(not _SCIPY, reason="scipy not installed")
class TestRSAVsScipy:
    """Verify that our results match the scipy-based notebook implementation."""

    def _scipy_rsa(self, X_np, Y_np, metric="correlation"):
        import numpy as np
        rdm_x = squareform(pdist(X_np, metric=metric))
        rdm_y = squareform(pdist(Y_np, metric=metric))
        n = rdm_x.shape[0]
        idx = np.triu_indices(n, k=1)
        r, _ = scipy_spearmanr(rdm_x[idx], rdm_y[idx])
        return float(r)

    def test_identical_matches_scipy(self):
        torch.manual_seed(1)
        X = torch.randn(30, 64)
        r_ours = RSA().rsa(X, X).item()
        r_scipy = self._scipy_rsa(X.numpy(), X.numpy())
        assert abs(r_ours - r_scipy) < 1e-4

    def test_random_pair_matches_scipy(self):
        torch.manual_seed(2)
        X = torch.randn(30, 64)
        Y = torch.randn(30, 64)
        r_ours = RSA().rsa(X, Y).item()
        r_scipy = self._scipy_rsa(X.numpy(), Y.numpy())
        assert abs(r_ours - r_scipy) < 1e-3

    def test_euclidean_metric_matches_scipy(self):
        torch.manual_seed(3)
        X = torch.randn(30, 64)
        Y = torch.randn(30, 64)
        r_ours = RSA(rdm_metric="euclidean").rsa(X, Y).item()
        r_scipy = self._scipy_rsa(X.numpy(), Y.numpy(), metric="euclidean")
        assert abs(r_ours - r_scipy) < 1e-3
