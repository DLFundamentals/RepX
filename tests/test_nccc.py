"""Tests for repx.transfer.nccc.

Run with: pytest tests/
"""

import torch
from repx.transfer import NCCCEvaluator


class TestNCCC:
    def test_compute_class_centers_full_shot(self):
        evaluator = NCCCEvaluator(device="cpu")
        features = torch.tensor(
            [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        centers_list, classes = evaluator.compute_class_centers(features, labels)

        assert classes == [0, 1]
        assert len(centers_list) == 1
        expected = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
        assert torch.allclose(centers_list[0], expected)

    def test_evaluate_perfect_accuracy(self):
        evaluator = NCCCEvaluator(device="cpu")
        features = torch.tensor(
            [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        accs = evaluator.evaluate(features, labels)

        assert len(accs) == 1
        assert abs(accs[0] - 1.0) < 1e-6

    def test_selected_classes_subset(self):
        evaluator = NCCCEvaluator(device="cpu")
        features = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [10.0, 10.0],
                [10.0, 12.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

        accs = evaluator.evaluate(features, labels, selected_classes=[0, 2])
        assert abs(accs[0] - 1.0) < 1e-6

    def test_few_shot_multiple_repeats(self):
        evaluator = NCCCEvaluator(device="cpu")
        torch.manual_seed(0)
        features = torch.randn(30, 8)
        labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10, dtype=torch.long)

        accs = evaluator.evaluate(features, labels, n_shot=3, repeat=5)
        assert len(accs) == 5
        for acc in accs:
            assert 0.0 <= acc <= 1.0
