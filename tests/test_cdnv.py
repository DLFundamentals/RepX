"""Tests for repx.collapse.cdnv.

Run with: pytest tests/
"""

import pytest
import torch
from repx.collapse import compute_cdnv, compute_directional_cdnv

# ---------------------------------------------------------------------------
# compute_cdnv
# ---------------------------------------------------------------------------


class TestComputeCDNV:
    def test_known_expected_output(self):
        # Two classes in 1D with hand-computable value.
        # class 0: [0, 2] -> mean=1, variance proxy=1
        # class 1: [3, 5] -> mean=4, variance proxy=1
        # dist^2 between means = 9
        # cdnv = ((1+1)/2)/9 = 1/9
        x = torch.tensor([[0.0], [2.0], [3.0], [5.0]], dtype=torch.float64)
        y = torch.tensor([0, 0, 1, 1])

        score = compute_cdnv(x, y, num_classes=2)
        assert score == pytest.approx(1.0 / 9.0, abs=1e-12)

    def test_returns_float(self):
        x = torch.randn(40, 16)
        y = torch.randint(0, 4, (40,))
        score = compute_cdnv(x, y, num_classes=4)
        assert isinstance(score, float)

    def test_infers_num_classes(self):
        x = torch.randn(30, 10)
        y = torch.randint(0, 3, (30,))

        inferred = compute_cdnv(x, y)
        explicit = compute_cdnv(x, y, num_classes=3)

        assert abs(inferred - explicit) < 1e-10

    def test_single_class_returns_zero(self):
        x = torch.randn(20, 8)
        y = torch.zeros(20, dtype=torch.long)

        assert compute_cdnv(x, y, num_classes=1) == 0.0

    def test_bad_shapes_raise(self):
        x = torch.randn(20, 8)
        y = torch.randint(0, 3, (19,))

        with pytest.raises(ValueError, match="same number of samples"):
            compute_cdnv(x, y)

        with pytest.raises(ValueError, match="2D tensor"):
            compute_cdnv(torch.randn(20), torch.randint(0, 3, (20,)))

        with pytest.raises(ValueError, match="1D tensor"):
            compute_cdnv(torch.randn(20, 8), torch.randint(0, 3, (20, 1)))

    def test_non_positive_eps_raises(self):
        x = torch.randn(20, 8)
        y = torch.randint(0, 3, (20,))

        with pytest.raises(ValueError, match="eps must be positive"):
            compute_cdnv(x, y, eps=0.0)


# ---------------------------------------------------------------------------
# compute_directional_cdnv
# ---------------------------------------------------------------------------


class TestComputeDirectionalCDNV:
    def test_known_expected_output(self):
        # Same setup as CDNV known case.
        # Ordered pairs: (0->1), (1->0), each contributes 1/9
        # Average over 2 ordered pairs -> 1/9
        x = torch.tensor([[0.0], [2.0], [3.0], [5.0]], dtype=torch.float64)
        y = torch.tensor([0, 0, 1, 1])

        score = compute_directional_cdnv(x, y, num_classes=2)
        assert score == pytest.approx(1.0 / 9.0, abs=1e-12)

    def test_returns_float(self):
        x = torch.randn(40, 16)
        y = torch.randint(0, 4, (40,))
        score = compute_directional_cdnv(x, y, num_classes=4)
        assert isinstance(score, float)

    def test_with_and_without_means_match(self):
        torch.manual_seed(12)
        x = torch.randn(60, 20)
        y = torch.randint(0, 5, (60,))

        means = torch.stack(
            [
                x[(y == class_idx).nonzero(as_tuple=True)[0]].mean(dim=0)
                for class_idx in range(5)
            ],
            dim=0,
        )
        from_means = compute_directional_cdnv(x, y, num_classes=5, means=means)
        inferred_means = compute_directional_cdnv(x, y, num_classes=5)

        assert abs(from_means - inferred_means) < 1e-8

    def test_single_class_returns_zero(self):
        x = torch.randn(20, 8)
        y = torch.zeros(20, dtype=torch.long)

        assert compute_directional_cdnv(x, y, num_classes=1) == 0.0

    def test_invalid_means_shape_raises(self):
        x = torch.randn(20, 8)
        y = torch.randint(0, 3, (20,))

        with pytest.raises(ValueError, match="2D tensor"):
            compute_directional_cdnv(x, y, num_classes=3, means=torch.randn(3))

        with pytest.raises(ValueError, match="at least num_classes"):
            compute_directional_cdnv(x, y, num_classes=3, means=torch.randn(2, 8))

    def test_non_positive_eps_raises(self):
        x = torch.randn(20, 8)
        y = torch.randint(0, 3, (20,))

        with pytest.raises(ValueError, match="eps must be positive"):
            compute_directional_cdnv(x, y, eps=0.0)
