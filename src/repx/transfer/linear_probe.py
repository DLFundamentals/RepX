"""Linear probe transfer evaluation on frozen representation features.

This module provides a simple linear classifier evaluator for downstream
classification quality of frozen embeddings.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from repx.utils.cdnv_utils import _validate_features_and_labels

__all__ = ["LinearProbeEvaluator"]


class LinearProbeEvaluator:
    """Train and evaluate a linear probe on frozen features.

    Parameters
    ----------
    train_features : Tensor, shape (num_train, feature_dim)
        Frozen train embeddings.
    train_labels : Tensor, shape (num_train,)
        Train labels.
    test_features : Tensor, shape (num_test, feature_dim)
        Frozen test embeddings.
    test_labels : Tensor, shape (num_test,)
        Test labels.
    num_output_classes : int
        Number of output classes for the linear probe layer.
    device : str or torch.device, default="cpu"
        Device used for training and evaluation.
    lr : float, default=3e-4
        Learning rate for Adam.
    epochs : int, default=100
        Number of optimization epochs.
    selected_classes : sequence[int], optional
        If provided, restrict training/evaluation to this class subset.
    """

    def __init__(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        num_output_classes: int,
        device: Union[str, torch.device] = "cpu",
        lr: float = 3e-4,
        epochs: int = 100,
        selected_classes: Optional[Sequence[int]] = None,
    ) -> None:
        if num_output_classes <= 0:
            raise ValueError(
                f"num_output_classes must be positive. Got {num_output_classes}."
            )
        if epochs <= 0:
            raise ValueError(f"epochs must be positive. Got {epochs}.")
        if lr <= 0:
            raise ValueError(f"lr must be positive. Got {lr}.")

        self.device = torch.device(device)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.num_output_classes = int(num_output_classes)

        self.train_features, self.train_labels = _validate_features_and_labels(
            train_features,
            train_labels,
            device=self.device,
        )
        self.test_features, self.test_labels = _validate_features_and_labels(
            test_features,
            test_labels,
            device=self.device,
        )

        if self.train_features.shape[1] != self.test_features.shape[1]:
            raise ValueError(
                "Feature dimension mismatch between train and test features. "
                f"Got {self.train_features.shape[1]} and {self.test_features.shape[1]}."
            )

        if selected_classes is None:
            resolved_classes = sorted(
                int(c) for c in torch.unique(self.train_labels).tolist()
            )
        else:
            resolved_classes = [int(c) for c in selected_classes]
            if len(resolved_classes) == 0:
                raise ValueError("selected_classes must contain at least one class.")

        if self.num_output_classes < len(resolved_classes):
            raise ValueError(
                "num_output_classes must be >= number of selected classes. "
                f"Got {self.num_output_classes} and {len(resolved_classes)}."
            )

        self.selected_classes = resolved_classes
        self.label_map: Dict[int, int] = {
            label: idx for idx, label in enumerate(self.selected_classes)
        }

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _map_labels_and_filter(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter to selected classes and map labels to 0..k-1."""
        selected = torch.tensor(
            self.selected_classes,
            device=labels.device,
            dtype=labels.dtype,
        )
        mask = torch.isin(labels, selected)
        filtered_features = features[mask]
        filtered_labels = labels[mask]

        if filtered_labels.numel() == 0:
            raise ValueError("No samples match selected_classes.")

        mapped = torch.empty_like(filtered_labels, dtype=torch.long)
        for original, mapped_id in self.label_map.items():
            mapped[filtered_labels == original] = mapped_id

        return filtered_features, mapped

    def _sample_fewshot(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``n_samples`` examples per mapped class from the train set."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive. Got {n_samples}.")

        class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels.tolist()):
            class_to_indices[int(label)].append(idx)

        chosen_indices: List[int] = []
        for class_id in self.label_map.values():
            idxs = class_to_indices.get(class_id, [])
            if len(idxs) < n_samples:
                raise ValueError(
                    f"Class {class_id} has only {len(idxs)} samples, "
                    f"but n_samples={n_samples} was requested."
                )

            perm = torch.randperm(len(idxs), device=features.device)[:n_samples]
            chosen_indices.extend(idxs[i] for i in perm.tolist())

        index_tensor = torch.tensor(
            chosen_indices, device=features.device, dtype=torch.long
        )
        return features[index_tensor], labels[index_tensor]

    def _train_probe(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        input_dim: int,
    ) -> torch.nn.Module:
        """Train a linear layer with cross-entropy loss."""
        probe = torch.nn.Linear(input_dim, self.num_output_classes, bias=False).to(
            self.device
        )
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            probe.train()
            logits = probe(features)
            loss = self.loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return probe

    @torch.no_grad()
    def _evaluate_probe(
        self,
        probe: torch.nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Return classification accuracy for a trained probe."""
        probe.eval()
        pred_labels = torch.argmax(probe(features), dim=1)
        return float((pred_labels == labels).float().mean().item())

    def evaluate(
        self,
        n_samples: Optional[int] = None,
        repeat: int = 5,
    ) -> Tuple[float, float]:
        """Train and evaluate a linear probe.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples per class for few-shot training. If omitted,
            full filtered train data is used.
        repeat : int, default=5
            Number of repeated runs.

        Returns
        -------
        tuple[float, float]
            ``(mean_train_accuracy, mean_test_accuracy)``.
        """
        if repeat <= 0:
            raise ValueError(f"repeat must be positive. Got {repeat}.")

        train_features, train_labels = self._map_labels_and_filter(
            self.train_features,
            self.train_labels,
        )
        test_features, test_labels = self._map_labels_and_filter(
            self.test_features,
            self.test_labels,
        )

        input_dim = int(train_features.shape[1])
        train_accs: List[float] = []
        test_accs: List[float] = []

        for _ in range(repeat):
            if n_samples is None:
                fewshot_features, fewshot_labels = train_features, train_labels
            else:
                fewshot_features, fewshot_labels = self._sample_fewshot(
                    train_features,
                    train_labels,
                    n_samples=n_samples,
                )

            probe = self._train_probe(
                fewshot_features, fewshot_labels, input_dim=input_dim
            )
            train_accs.append(self._evaluate_probe(probe, train_features, train_labels))
            test_accs.append(self._evaluate_probe(probe, test_features, test_labels))

        train_mean = float(torch.tensor(train_accs).mean().item())
        test_mean = float(torch.tensor(test_accs).mean().item())
        return train_mean, test_mean
