"""Nearest Class Center Classifier (NCCC) transfer evaluation.

This module provides a nearest-class-center evaluator for learned
representations. Class centers are computed on a support set, and query
samples are classified by nearest center using Euclidean distance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

from repx.utils.label_utils import _map_labels_to_indices
from repx.utils.mean_utils import (
    _compute_class_means,
    _resolve_selected_classes,
    _validate_features_and_labels,
)

__all__ = ["NCCCEvaluator"]


class NCCCEvaluator:
    """Nearest Class Center Classifier (NCCC) evaluator."""

    def __init__(self, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)

    def compute_class_centers(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_shot: Optional[int] = None,
        repeat: int = 1,
        selected_classes: Optional[Sequence[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Compute class centers from full- or few-shot support data."""
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1. Got {repeat}.")
        if n_shot is not None and n_shot <= 0:
            raise ValueError(f"n_shot must be positive when provided. Got {n_shot}.")

        features, labels = _validate_features_and_labels(
            features, labels, device=self.device
        )
        selected = _resolve_selected_classes(labels, selected_classes)
        mapped_labels, mask = _map_labels_to_indices(labels, selected)
        features_sub = features[mask]

        if mapped_labels.numel() == 0:
            raise ValueError("No support samples match selected_classes.")

        if n_shot is None:
            means_all = _compute_class_means(features_sub, mapped_labels, len(selected))
            return [means_all], selected

        centers_per_repeat: List[torch.Tensor] = []
        for repeat_idx in range(repeat):
            sampled_features: List[torch.Tensor] = []
            sampled_labels: List[torch.Tensor] = []

            for class_idx in range(len(selected)):
                idxs = (mapped_labels == class_idx).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue

                if idxs.numel() > n_shot:
                    torch.manual_seed(repeat_idx)
                    take = torch.randperm(idxs.numel(), device=idxs.device)[:n_shot]
                    idxs = idxs[take]

                sampled_features.append(features_sub[idxs])
                sampled_labels.append(mapped_labels[idxs])

            if len(sampled_features) == 0:
                raise ValueError("No support samples available to compute centers.")

            sampled_features_t = torch.cat(sampled_features, dim=0)
            sampled_labels_t = torch.cat(sampled_labels, dim=0)
            means_all = _compute_class_means(
                sampled_features_t, sampled_labels_t, len(selected)
            )
            centers_per_repeat.append(means_all)

        return centers_per_repeat, selected

    def evaluate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_shot: Optional[int] = None,
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
            n_shot=n_shot,
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
