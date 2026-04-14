"""Linear probe transfer evaluation on frozen representation features.

This module provides a simple linear classifier evaluator for downstream
classification quality of frozen embeddings.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch

from repx.utils.helpers import (
    _resolve_selected_classes,
    _validate_features_and_labels,
)
from repx.utils.label_utils import _filter_features_and_map_labels, _sample_per_class
from repx.utils.torch_utils import _resolve_loss, _resolve_optimizer

__all__ = ["LinearProbeEvaluator"]


class LinearProbeEvaluator:
    """Train and evaluate a linear probe on frozen features.

    Parameters
    ----------
    device : str or torch.device, default="cpu"
        Device used for training and evaluation.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)

    def _train_probe(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        input_dim: int,
        num_output_classes: int,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: Mapping[str, Any],
        loss_cls: Type[torch.nn.Module],
        loss_kwargs: Mapping[str, Any],
        epochs: int,
    ) -> torch.nn.Module:
        """Train a linear layer with cross-entropy loss."""
        probe = torch.nn.Linear(input_dim, num_output_classes, bias=False).to(
            self.device
        )
        optimizer = optimizer_cls(probe.parameters(), **dict(optimizer_kwargs))
        loss_fn = loss_cls(**dict(loss_kwargs))

        for _ in range(epochs):
            probe.train()
            logits = probe(features)
            loss = loss_fn(logits, labels)

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
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        num_output_classes: int,
        lr: float = 3e-4,
        epochs: int = 100,
        n_shots: Optional[int] = None,
        repeat: int = 5,
        selected_classes: Optional[Sequence[int]] = None,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "adam",
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        loss_fn: Union[str, Type[torch.nn.Module]] = "cross_entropy",
        loss_fn_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Train and evaluate a linear probe.

        Examples
        --------
        ```python
        import torch
        from repx.transfer import LinearProbeEvaluator

        z_train = torch.randn(256, 128)
        y_train = torch.randint(0, 10, (256,))
        z_test = torch.randn(128, 128)
        y_test = torch.randint(0, 10, (128,))

        evaluator = LinearProbeEvaluator(device="cpu")
        train_acc, test_acc = evaluator.evaluate(
            train_features=z_train,
            train_labels=y_train,
            test_features=z_test,
            test_labels=y_test,
            num_output_classes=10,
            n_shots=5,
            repeat=3,
        )
        ```

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
        lr : float, default=3e-4
            Learning rate for Adam.
        epochs : int, default=100
            Number of optimization epochs.
        n_shots : int, optional
            Number of samples per class for few-shot training. If omitted,
            full filtered train data is used.
        repeat : int, default=5
            Number of repeated runs.
        selected_classes : sequence[int], optional
            If provided, restrict training/evaluation to this class subset.
        optimizer : str or Optimizer class, default="adam"
            Optimizer choice used to train the probe.
        optimizer_kwargs : mapping, optional
            Extra keyword arguments passed to optimizer constructor.
            If ``lr`` is not provided, this method's ``lr`` argument is used.
        loss_fn : str or nn.Module class, default="cross_entropy"
            Loss function used for probe training.
        loss_fn_kwargs : mapping, optional
            Extra keyword arguments passed to the loss constructor.

        Returns
        -------
        tuple[float, float]
            ``(mean_train_accuracy, mean_test_accuracy)``.
        """
        if num_output_classes <= 0:
            raise ValueError(
                f"num_output_classes must be positive. Got {num_output_classes}."
            )
        if epochs <= 0:
            raise ValueError(f"epochs must be positive. Got {epochs}.")
        if lr <= 0:
            raise ValueError(f"lr must be positive. Got {lr}.")
        if repeat <= 0:
            raise ValueError(f"repeat must be positive. Got {repeat}.")

        optimizer_cls = _resolve_optimizer(optimizer)
        loss_cls = _resolve_loss(loss_fn)
        optimizer_kwargs_resolved = dict(optimizer_kwargs or {})
        loss_kwargs_resolved = dict(loss_fn_kwargs or {})
        optimizer_kwargs_resolved.setdefault("lr", lr)

        train_features, train_labels = _validate_features_and_labels(
            train_features,
            train_labels,
            device=self.device,
        )
        test_features, test_labels = _validate_features_and_labels(
            test_features,
            test_labels,
            device=self.device,
        )

        if train_features.shape[1] != test_features.shape[1]:
            raise ValueError(
                "Feature dimension mismatch between train and test features. "
                f"Got {train_features.shape[1]} and {test_features.shape[1]}."
            )

        resolved_classes = _resolve_selected_classes(
            train_labels,
            selected_classes,
        )
        if num_output_classes < len(resolved_classes):
            raise ValueError(
                "num_output_classes must be >= number of selected classes. "
                f"Got {num_output_classes} and {len(resolved_classes)}."
            )

        train_features, train_labels = _filter_features_and_map_labels(
            train_features,
            train_labels,
            resolved_classes,
        )
        test_features, test_labels = _filter_features_and_map_labels(
            test_features,
            test_labels,
            resolved_classes,
        )

        input_dim = int(train_features.shape[1])
        num_classes = len(resolved_classes)
        train_accs: List[float] = []
        test_accs: List[float] = []

        for _ in range(repeat):
            if n_shots is None:
                selected_features, selected_labels = train_features, train_labels
            else:
                selected_features, selected_labels = _sample_per_class(
                    train_features,
                    train_labels,
                    n_shots=n_shots,
                    num_classes=num_classes,
                    strict=True,
                )

            probe = self._train_probe(
                selected_features,
                selected_labels,
                input_dim=input_dim,
                num_output_classes=num_output_classes,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs_resolved,
                loss_cls=loss_cls,
                loss_kwargs=loss_kwargs_resolved,
                epochs=epochs,
            )
            train_accs.append(self._evaluate_probe(probe, train_features, train_labels))
            test_accs.append(self._evaluate_probe(probe, test_features, test_labels))

        train_mean = float(torch.tensor(train_accs).mean().item())
        test_mean = float(torch.tensor(test_accs).mean().item())
        return train_mean, test_mean
