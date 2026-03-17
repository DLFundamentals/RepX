"""Tests for letorch.cka

Run with:  pytest tests/
"""

import pytest
import torch

from letorch.cka import CKA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orthogonal_pair(n: int = 64, d: int = 32, seed: int = 0):
    """Return (X, X_orth) where X_orth's column space ⊥ X's column space."""
    g = torch.Generator()
    g.manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    M = torch.randn(n, n, generator=g)
    M[:, :d] = X
    Q, _ = torch.linalg.qr(M)
    return X, Q[:, d:d + d].clone()


# ---------------------------------------------------------------------------
# Identical matrices → CKA ≈ 1.0
# ---------------------------------------------------------------------------

class TestCKAIdentical:
    @pytest.mark.parametrize("debiased", [False, True])
    def test_identical_is_one(self, debiased):
        torch.manual_seed(0)
        X = torch.randn(50, 32)
        score = CKA(debiased=debiased).cka(X, X).item()
        assert abs(score - 1.0) < 1e-4, (
            f"debiased={debiased}: expected 1.0, got {score}"
        )


# ---------------------------------------------------------------------------
# Biased CKA on orthogonal subspaces → near zero
# ---------------------------------------------------------------------------

class TestBiasedCKAOrthogonal:
    def test_biased_orthogonal_is_near_zero(self):
        """Biased CKA for representations from orthogonal subspaces ≈ 0."""
        X, X_orth = _make_orthogonal_pair(n=64, d=16, seed=7)
        score = CKA(debiased=False).cka(X, X_orth).item()
        assert abs(score) < 0.05, f"biased CKA: expected ≈ 0.0, got {score}"


# ---------------------------------------------------------------------------
# Debiased CKA for independent representations → mean ≈ 0
# ---------------------------------------------------------------------------

class TestDebiasedCKAUnbiased:
    def test_mean_near_zero_for_independent(self):
        """Debiased CKA is unbiased: its mean over independent pairs ≈ 0."""
        torch.manual_seed(42)
        cka = CKA(debiased=True)
        scores = [
            cka.cka(torch.randn(50, 32), torch.randn(50, 32)).item()
            for _ in range(100)
        ]
        mean = sum(scores) / len(scores)
        assert abs(mean) < 0.05, f"debiased mean={mean:.4f}; expected ≈ 0"

    def test_biased_mean_positive_for_independent(self):
        """Biased CKA is positively biased for independent representations."""
        torch.manual_seed(42)
        cka = CKA(debiased=False)
        scores = [
            cka.cka(torch.randn(50, 32), torch.randn(50, 32)).item()
            for _ in range(100)
        ]
        mean = sum(scores) / len(scores)
        assert mean > 0.05, (
            f"biased mean={mean:.4f}; expected positive bias > 0.05"
        )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestCKAShape:
    @pytest.mark.parametrize("debiased", [False, True])
    def test_output_is_scalar(self, debiased):
        torch.manual_seed(1)
        X = torch.randn(30, 64)
        Y = torch.randn(30, 128)
        score = CKA(debiased=debiased).cka(X, Y)
        assert score.shape == torch.Size([])

    def test_biased_output_non_negative(self):
        """Biased CKA is always ≥ 0."""
        torch.manual_seed(2)
        for _ in range(20):
            X = torch.randn(40, 32)
            Y = torch.randn(40, 32)
            score = CKA(debiased=False).cka(X, Y).item()
            assert score >= -1e-6, f"biased CKA={score} is negative"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestCKAErrors:
    def test_mismatched_rows_raises(self):
        with pytest.raises(ValueError, match="same number of samples"):
            CKA().cka(torch.randn(30, 64), torch.randn(40, 64))

    def test_unbiased_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 4 samples"):
            CKA(debiased=True).cka(torch.randn(3, 4), torch.randn(3, 4))


# ---------------------------------------------------------------------------
# GPU (skipped if CUDA unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCKAGPU:
    def test_identical_on_gpu(self):
        X = torch.randn(30, 64, device="cuda")
        score = CKA().cka(X, X)
        assert score.device.type == "cuda"
        assert abs(score.item() - 1.0) < 1e-4
