"""Tests for repx.transfer.nccc.

Run with: pytest tests/
"""

import torch

from repx.transfer import compute_nccc_centers, evaluate_nccc


class TestNCCC:
    def test_compute_class_centers_full_shot(self):
        features = torch.tensor(
            [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        centers_list, classes = compute_nccc_centers(features, labels, device="cpu")

        assert classes == [0, 1]
        assert len(centers_list) == 1
        expected = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
        assert torch.allclose(centers_list[0], expected)

    def test_evaluate_perfect_accuracy(self):
        features = torch.tensor(
            [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        centers_list, classes = compute_nccc_centers(features, labels, device="cpu")
        accs = evaluate_nccc(features, labels, centers_list, classes, device="cpu")

        assert len(accs) == 1
        assert abs(accs[0] - 1.0) < 1e-6

    def test_selected_classes_subset(self):
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

        centers_list, classes = compute_nccc_centers(
            features,
            labels,
            selected_classes=[0, 2],
            device="cpu",
        )

        assert classes == [0, 2]
        expected = torch.tensor([[0.0, 1.0], [10.0, 11.0]])
        assert torch.allclose(centers_list[0], expected)

        accs = evaluate_nccc(features, labels, centers_list, classes, device="cpu")
        assert abs(accs[0] - 1.0) < 1e-6

    def test_few_shot_multiple_repeats(self):
        torch.manual_seed(0)
        features = torch.randn(30, 8)
        labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10, dtype=torch.long)

        centers_list, classes = compute_nccc_centers(
            features,
            labels,
            n_shot=3,
            repeat=5,
            device="cpu",
        )

        assert classes == [0, 1, 2]
        assert len(centers_list) == 5
        for centers in centers_list:
            assert centers.shape == (3, 8)

        accs = evaluate_nccc(features, labels, centers_list, classes, device="cpu")
        assert len(accs) == 5
        for acc in accs:
            assert 0.0 <= acc <= 1.0
