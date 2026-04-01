"""Tests for repx.collapse.etf_deviation.

Run with: pytest tests/
"""

import math

import pytest
import torch

from repx.collapse import compute_etf_deviation


class TestETFDeviation:
    def test_returns_float(self):
        x = torch.randn(30, 16)
        y = torch.randint(0, 4, (30,))
        score = compute_etf_deviation(x, y, num_classes=4)
        assert isinstance(score, float)

    def test_perfect_simplex_etf_is_zero(self):
        # Three class means arranged as a 2D simplex ETF.
        x = torch.tensor(
            [
                [1.0, 0.0],
                [-0.5, math.sqrt(3.0) / 2.0],
                [-0.5, -math.sqrt(3.0) / 2.0],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor([0, 1, 2], dtype=torch.long)

        score = compute_etf_deviation(x, y, num_classes=3)
        assert score == pytest.approx(0.0, abs=1e-12)

    def test_translation_invariance(self):
        base = torch.tensor(
            [
                [1.0, 0.0],
                [-0.5, math.sqrt(3.0) / 2.0],
                [-0.5, -math.sqrt(3.0) / 2.0],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor([0, 1, 2], dtype=torch.long)
        shift = torch.tensor([10.0, -7.0], dtype=torch.float64)

        s1 = compute_etf_deviation(base, y, num_classes=3)
        s2 = compute_etf_deviation(base + shift, y, num_classes=3)

        assert abs(s1 - s2) < 1e-12

    def test_single_active_class_returns_zero(self):
        x = torch.randn(20, 8)
        y = torch.zeros(20, dtype=torch.long)
        assert compute_etf_deviation(x, y, num_classes=1) == 0.0

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError, match="2D tensor"):
            compute_etf_deviation(torch.randn(20), torch.randint(0, 3, (20,)))

        with pytest.raises(ValueError, match="1D tensor"):
            compute_etf_deviation(torch.randn(20, 8), torch.randint(0, 3, (20, 1)))

        with pytest.raises(ValueError, match="same number of samples"):
            compute_etf_deviation(torch.randn(20, 8), torch.randint(0, 3, (19,)))

    def test_invalid_num_classes_range_raises(self):
        x = torch.randn(10, 4)
        y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 3])

        with pytest.raises(ValueError, match="labels must be in"):
            compute_etf_deviation(x, y, num_classes=3)

    def test_non_positive_eps_raises(self):
        x = torch.randn(20, 8)
        y = torch.randint(0, 3, (20,))

        with pytest.raises(ValueError, match="eps must be positive"):
            compute_etf_deviation(x, y, eps=0.0)
