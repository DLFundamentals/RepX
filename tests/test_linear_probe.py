"""Tests for repx.transfer.linear_probe."""

import pytest
import torch

from repx.transfer import LinearProbeEvaluator


def _make_separable_dataset(n_per_class: int = 32):
    torch.manual_seed(0)
    centers = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float32)

    x0 = centers[0] + 0.25 * torch.randn(n_per_class, 2)
    x1 = centers[1] + 0.25 * torch.randn(n_per_class, 2)

    train_features = torch.cat([x0, x1], dim=0)
    train_labels = torch.tensor([0] * n_per_class + [1] * n_per_class, dtype=torch.long)

    xt0 = centers[0] + 0.25 * torch.randn(n_per_class, 2)
    xt1 = centers[1] + 0.25 * torch.randn(n_per_class, 2)
    test_features = torch.cat([xt0, xt1], dim=0)
    test_labels = torch.tensor([0] * n_per_class + [1] * n_per_class, dtype=torch.long)

    return train_features, train_labels, test_features, test_labels


class TestLinearProbeEvaluator:
    def test_full_shot_high_accuracy(self):
        train_x, train_y, test_x, test_y = _make_separable_dataset()
        evaluator = LinearProbeEvaluator(
            train_features=train_x,
            train_labels=train_y,
            test_features=test_x,
            test_labels=test_y,
            num_output_classes=2,
            device="cpu",
            lr=5e-2,
            epochs=200,
        )

        train_acc, test_acc = evaluator.evaluate(n_samples=None, repeat=3)
        assert 0.95 <= train_acc <= 1.0
        assert 0.95 <= test_acc <= 1.0

    def test_few_shot_runs_and_returns_valid_range(self):
        train_x, train_y, test_x, test_y = _make_separable_dataset()
        evaluator = LinearProbeEvaluator(
            train_features=train_x,
            train_labels=train_y,
            test_features=test_x,
            test_labels=test_y,
            num_output_classes=2,
            device="cpu",
            lr=1e-2,
            epochs=150,
        )

        train_acc, test_acc = evaluator.evaluate(n_samples=4, repeat=5)
        assert 0.0 <= train_acc <= 1.0
        assert 0.0 <= test_acc <= 1.0
        assert test_acc > 0.8

    def test_selected_classes_subset(self):
        torch.manual_seed(3)
        train_x = torch.randn(90, 8)
        train_y = torch.tensor([0] * 30 + [1] * 30 + [2] * 30, dtype=torch.long)
        test_x = torch.randn(60, 8)
        test_y = torch.tensor([0] * 20 + [1] * 20 + [2] * 20, dtype=torch.long)

        evaluator = LinearProbeEvaluator(
            train_features=train_x,
            train_labels=train_y,
            test_features=test_x,
            test_labels=test_y,
            num_output_classes=2,
            selected_classes=[0, 2],
            device="cpu",
            epochs=2,
        )

        train_acc, test_acc = evaluator.evaluate(repeat=2)
        assert 0.0 <= train_acc <= 1.0
        assert 0.0 <= test_acc <= 1.0

    def test_few_shot_raises_when_class_too_small(self):
        train_x = torch.randn(6, 4)
        train_y = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)
        test_x = torch.randn(4, 4)
        test_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        evaluator = LinearProbeEvaluator(
            train_features=train_x,
            train_labels=train_y,
            test_features=test_x,
            test_labels=test_y,
            num_output_classes=2,
            device="cpu",
            epochs=2,
        )

        with pytest.raises(ValueError, match="has only"):
            evaluator.evaluate(n_samples=3, repeat=1)

    def test_invalid_output_classes_raises(self):
        train_x, train_y, test_x, test_y = _make_separable_dataset(n_per_class=8)

        with pytest.raises(ValueError, match="num_output_classes"):
            LinearProbeEvaluator(
                train_features=train_x,
                train_labels=train_y,
                test_features=test_x,
                test_labels=test_y,
                num_output_classes=1,
                selected_classes=[0, 1],
                device="cpu",
            )