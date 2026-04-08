"""Nearest Class Center Classifier (NCCC) transfer evaluation.

This module provides a nearest-class-center evaluator for learned
representations. Class centers are computed on a support set, and query
samples are classified by nearest center using Euclidean distance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

from repx.utils.helpers import (
    _compute_class_means,
    _resolve_selected_classes,
    _validate_features_and_labels,
)
from repx.utils.label_utils import _map_labels_to_indices, _sample_per_class

__all__ = ["NCCCEvaluator"]


class NCCCEvaluator:
    """Nearest Class Center Classifier (NCCC) evaluator."""

    def __init__(self, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)

    def compute_class_centers(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_shots: Optional[int] = None,
        repeat: int = 1,
        selected_classes: Optional[Sequence[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Compute class centers from full- or few-shot support data."""
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1. Got {repeat}.")
        if n_shots is not None and n_shots <= 0:
            raise ValueError(f"n_shots must be positive when provided. Got {n_shots}.")

        features, labels = _validate_features_and_labels(
            features, labels, device=self.device
        )
        selected = _resolve_selected_classes(labels, selected_classes)
        mapped_labels, mask = _map_labels_to_indices(labels, selected)
        features_sub = features[mask]

        if mapped_labels.numel() == 0:
            raise ValueError("No support samples match selected_classes.")

        if n_shots is None:
            means_all = _compute_class_means(features_sub, mapped_labels, len(selected))
            return [means_all], selected

        centers_per_repeat: List[torch.Tensor] = []
        for repeat_idx in range(repeat):
            sampled_features_t, sampled_labels_t = _sample_per_class(
                features_sub,
                mapped_labels,
                n_shots=n_shots,
                num_classes=len(selected),
                strict=False,
                seed=repeat_idx,
            )
            means_all = _compute_class_means(
                sampled_features_t, sampled_labels_t, len(selected)
            )
            centers_per_repeat.append(means_all)

        return centers_per_repeat, selected

    def evaluate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_shots: Optional[int] = None,
        repeat: int = 1,
        selected_classes: Optional[Sequence[int]] = None,
    ) -> List[float]:
        """Evaluate accuracy using nearest class centers.

        Centers are computed internally by :meth:`compute_class_centers` from
        the same input ``features`` and ``labels``.
        """
        features, labels = _validate_features_and_labels(
            features, labels, device=self.device
        )

        centers_list, resolved_classes = self.compute_class_centers(
            features,
            labels,
            n_shots=n_shots,
            repeat=repeat,
            selected_classes=selected_classes,
        )

        mapped_labels, mask = _map_labels_to_indices(labels, resolved_classes)
        features_sub = features[mask]

        n_classes = len(resolved_classes)
        accuracies: List[float] = []
        for centers in centers_list:
            centers = centers.to(features.device)
            if centers.ndim != 2:
                raise ValueError(
                    f"Each centers tensor must be 2D, got shape {tuple(centers.shape)}."
                )
            if centers.shape[0] != n_classes:
                raise ValueError(
                    "Center rows must match selected_classes length. "
                    f"Got {centers.shape[0]} rows for {n_classes} classes."
                )
            if centers.shape[1] != features_sub.shape[1]:
                raise ValueError(
                    "Feature dimension mismatch between features and centers. "
                    f"Got {features_sub.shape[1]} and {centers.shape[1]}."
                )

            dists = torch.cdist(features_sub, centers)
            pred_idx = torch.argmin(dists, dim=1)
            acc = (pred_idx == mapped_labels).float().mean().item()
            accuracies.append(float(acc))

        return accuracies
