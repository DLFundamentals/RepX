"""Tests for letorch.cka

Run with:  pytest tests/
"""

import pytest
import torch
from letorch.alignment.cka import CKA

# ---------------------------------------------------------------------------
# compute_kernel
# ---------------------------------------------------------------------------


class TestComputeKernel:
    def test_output_shape(self):
        X = torch.randn(20, 64)
        assert CKA().compute_kernel(X).shape == (20, 20)

    def test_symmetry(self):
        X = torch.randn(20, 64)
        K = CKA().compute_kernel(X)
        assert (K - K.T).abs().max().item() < 1e-5

    def test_psd(self):
        """Linear kernel X @ Xᵀ is positive semi-definite."""
        X = torch.randn(20, 64)
        K = CKA().compute_kernel(X)
        eigenvalues = torch.linalg.eigvalsh(K)
        assert eigenvalues.min().item() >= -1e-4

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            CKA(kernel="rbf")


# ---------------------------------------------------------------------------
# cka — identical matrices (score = 1.0)
# ---------------------------------------------------------------------------


class TestCKAIdentical:
    def test_identical(self):
        X = torch.randn(30, 64)
        score = CKA().cka(X, X)
        assert abs(score.item() - 1.0) < 1e-4

    def test_identical_returns_scalar(self):
        X = torch.randn(30, 64)
        assert CKA().cka(X, X).shape == torch.Size([])


# ---------------------------------------------------------------------------
# cka — invariances (linear kernel)
# ---------------------------------------------------------------------------


class TestCKAInvariances:
    def test_isotropic_scaling(self):
        """Linear CKA is invariant to isotropic scaling of rows."""
        X = torch.randn(50, 64)
        assert abs(CKA().cka(X, X * 100).item() - 1.0) < 1e-4

    def test_orthogonal_rotation(self):
        """Linear CKA is invariant to orthogonal rotation of features."""
        torch.manual_seed(0)
        X = torch.randn(50, 64)
        M = torch.randn(64, 64)
        Q, _ = torch.linalg.qr(M)  # 64×64 orthogonal matrix
        Y = X @ Q
        assert abs(CKA().cka(X, Y).item() - 1.0) < 1e-4

    def test_orthogonal_rotation_many(self):
        """Invariance holds across 20 independent random rotations."""
        torch.manual_seed(1)
        X = torch.randn(50, 64)
        cka = CKA()
        for _ in range(20):
            Q, _ = torch.linalg.qr(torch.randn(64, 64))
            assert abs(cka.cka(X, X @ Q).item() - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# cka — orthogonal / independent matrices (score ≈ 0)
# ---------------------------------------------------------------------------


class TestCKAOrthogonal:
    def test_independent_random_near_zero(self):
        """Two independent random matrices should give CKA close to 0."""
        torch.manual_seed(99)
        X = torch.randn(50, 128)
        Y = torch.randn(50, 128)
        assert abs(CKA().cka(X, Y).item()) < 0.15

    def test_random_gaussian_mean_near_zero(self):
        """Mean CKA over many independent pairs should be ≈ 0."""
        torch.manual_seed(0)
        cka = CKA()
        scores = [
            cka.cka(torch.randn(50, 128), torch.randn(50, 128)).item()
            for _ in range(100)
        ]
        assert abs(sum(scores) / len(scores)) < 0.05


# ---------------------------------------------------------------------------
# cka — different feature dimensionalities
# ---------------------------------------------------------------------------


class TestCKAMixedDims:
    def test_different_feature_dims(self):
        X = torch.randn(30, 64)
        Y = torch.randn(30, 256)
        score = CKA().cka(X, Y)
        assert score.shape == torch.Size([])  # scalar

    def test_mismatched_stimuli_raises(self):
        with pytest.raises(ValueError, match="same number of stimuli"):
            CKA().cka(torch.randn(30, 64), torch.randn(40, 64))


# ---------------------------------------------------------------------------
# cka — error handling
# ---------------------------------------------------------------------------


class TestCKAErrors:
    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            CKA(kernel="rbf")

    def test_too_few_stimuli_raises(self):
        with pytest.raises(ValueError, match="At least 4 stimuli"):
            CKA().cka(torch.randn(3, 64), torch.randn(3, 64))


# ---------------------------------------------------------------------------
# cka — GPU (skipped if CUDA unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCKAGPU:
    def test_identical_on_gpu(self):
        X = torch.randn(30, 64, device="cuda")
        score = CKA().cka(X, X)
        assert score.device.type == "cuda"
        assert abs(score.item() - 1.0) < 1e-4

    def test_random_on_gpu(self):
        X = torch.randn(30, 64, device="cuda")
        Y = torch.randn(30, 64, device="cuda")
        score = CKA().cka(X, Y)
        assert score.device.type == "cuda"


# ---------------------------------------------------------------------------
# Cross-validation against NumPy reference implementation
# ---------------------------------------------------------------------------


class TestCKAVsNumpy:
    """Verify PyTorch results match a NumPy reference implementation."""

    @staticmethod
    def _numpy_cka(X_np, Y_np):
        import numpy as np

        K = X_np @ X_np.T
        L = Y_np @ Y_np.T
        np.fill_diagonal(K, 0.0)
        np.fill_diagonal(L, 0.0)

        def hsic(A, B):
            n = A.shape[0]
            ones = np.ones(n)
            t1 = np.trace(A @ B)
            t2 = (ones @ A @ ones) * (ones @ B @ ones) / ((n - 1) * (n - 2))
            t3 = 2.0 / (n - 2) * (ones @ A @ B @ ones)
            return (t1 + t2 - t3) / (n * (n - 3))

        kl = hsic(K, L)
        kk = hsic(K, K)
        ll = hsic(L, L)
        return float(kl / (kk * ll) ** 0.5)

    def test_identical_matches_numpy(self):
        torch.manual_seed(1)
        X = torch.randn(30, 64)
        r_ours = CKA().cka(X, X).item()
        r_ref = self._numpy_cka(X.numpy(), X.numpy())
        assert abs(r_ours - r_ref) < 1e-4

    def test_random_pair_matches_numpy(self):
        torch.manual_seed(2)
        X = torch.randn(30, 64)
        Y = torch.randn(30, 64)
        r_ours = CKA().cka(X, Y).item()
        r_ref = self._numpy_cka(X.numpy(), Y.numpy())
        assert abs(r_ours - r_ref) < 1e-4

    def test_different_dims_matches_numpy(self):
        torch.manual_seed(3)
        X = torch.randn(30, 64)
        Y = torch.randn(30, 256)
        r_ours = CKA().cka(X, Y).item()
        r_ref = self._numpy_cka(X.numpy(), Y.numpy())
        assert abs(r_ours - r_ref) < 1e-4
