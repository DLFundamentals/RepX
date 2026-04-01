"""Nearest Class Center Classifier (NCCC) transfer evaluation.

This module provides a nearest-class-center probe for evaluating learned
representations. Class centers are computed on a support set, and query
samples are classified by nearest center using Euclidean distance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

from repx.utils.cdnv_utils import _compute_class_means, _validate_features_and_labels

__all__ = ["compute_nccc_centers", "evaluate_nccc"]


def _resolve_selected_classes(
    labels: torch.Tensor,
    selected_classes: Optional[Sequence[int]],
) -> List[int]:
    if selected_classes is None:
        return sorted(int(c) for c in torch.unique(labels).tolist())

    resolved = [int(c) for c in selected_classes]
    if len(resolved) == 0:
        raise ValueError("selected_classes must contain at least one class.")
    return resolved


def compute_nccc_centers(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_shot: Optional[int] = None,
    repeat: int = 1,
    selected_classes: Optional[Sequence[int]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[List[torch.Tensor], List[int]]:
    """Compute class centers for NCCC from full- or few-shot support data.

    Parameters
    ----------
    features : Tensor, shape (num_samples, feature_dim)
        Support embeddings.
    labels : Tensor, shape (num_samples,)
        Support labels.
    n_shot : int, optional
        Number of support examples per class. If omitted, all available
        examples per class are used.
    repeat : int, default=1
        Number of repeated center estimations. This is useful for few-shot
        evaluation where each repeat re-samples support examples.
    selected_classes : sequence[int], optional
        Restrict center computation to this subset of class ids.
    device : str or torch.device, default="cpu"
        Device used for computation.

    Returns
    -------
    tuple[list[Tensor], list[int]]
        ``(centers_per_repeat, selected_classes)`` where each center tensor has
        shape ``(num_selected_classes, feature_dim)``.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1. Got {repeat}.")
    if n_shot is not None and n_shot <= 0:
        raise ValueError(f"n_shot must be positive when provided. Got {n_shot}.")

    features, labels = _validate_features_and_labels(features, labels, device=device)
    selected = _resolve_selected_classes(labels, selected_classes)

    selected_tensor = torch.tensor(selected, device=labels.device, dtype=labels.dtype)
    in_subset = torch.isin(labels, selected_tensor)
    features_sub = features[in_subset]
    labels_sub = labels[in_subset]

    if labels_sub.numel() == 0:
        raise ValueError("No support samples match selected_classes.")

    # Compute all means once for full-shot setting.
    max_class = int(max(selected))
    if n_shot is None:
        means_all = _compute_class_means(features_sub, labels_sub, max_class + 1)
        centers = means_all[selected]
        return [centers], selected

    centers_per_repeat: List[torch.Tensor] = []

    for repeat_idx in range(repeat):
        sampled_features: List[torch.Tensor] = []
        sampled_labels: List[torch.Tensor] = []

        for class_id in selected:
            idxs = (labels_sub == class_id).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue

            if idxs.numel() > n_shot:
                torch.manual_seed(repeat_idx)
                take = torch.randperm(idxs.numel(), device=idxs.device)[:n_shot]
                idxs = idxs[take]

            sampled_features.append(features_sub[idxs])
            sampled_labels.append(labels_sub[idxs])

        if len(sampled_features) == 0:
            raise ValueError("No support samples available to compute centers.")

        sampled_features_t = torch.cat(sampled_features, dim=0)
        sampled_labels_t = torch.cat(sampled_labels, dim=0)

        means_all = _compute_class_means(
            sampled_features_t, sampled_labels_t, max_class + 1
        )
        centers_per_repeat.append(means_all[selected])

    return centers_per_repeat, selected


def evaluate_nccc(
    features: torch.Tensor,
    labels: torch.Tensor,
    centers_list: Sequence[torch.Tensor],
    selected_classes: Sequence[int],
    device: Union[str, torch.device] = "cpu",
) -> List[float]:
    """Evaluate NCCC accuracy on query data.

    Parameters
    ----------
    features : Tensor, shape (num_samples, feature_dim)
        Query embeddings.
    labels : Tensor, shape (num_samples,)
        Query labels.
    centers_list : sequence[Tensor]
        Center tensors from :func:`compute_nccc_centers`.
    selected_classes : sequence[int]
        Class ids corresponding to rows in each center tensor.
    device : str or torch.device, default="cpu"
        Device used for computation.

    Returns
    -------
    list[float]
        Accuracy values, one per center tensor in ``centers_list``.
    """
    features, labels = _validate_features_and_labels(features, labels, device=device)

    if len(selected_classes) == 0:
        raise ValueError("selected_classes must contain at least one class.")
    if len(centers_list) == 0:
        return []

    selected = [int(c) for c in selected_classes]
    selected_tensor = torch.tensor(selected, device=labels.device, dtype=labels.dtype)

    in_subset = torch.isin(labels, selected_tensor)
    features_sub = features[in_subset]
    labels_sub = labels[in_subset]

    if labels_sub.numel() == 0:
        raise ValueError("No query samples match selected_classes.")

    accuracies: List[float] = []
    for centers in centers_list:
        centers = centers.to(features.device)
        if centers.ndim != 2:
            raise ValueError(
                f"Each centers tensor must be 2D, got shape {tuple(centers.shape)}."
            )
        if centers.shape[0] != len(selected):
            raise ValueError(
                "Center rows must match selected_classes length. "
                f"Got {centers.shape[0]} rows for {len(selected)} classes."
            )
        if centers.shape[1] != features_sub.shape[1]:
            raise ValueError(
                "Feature dimension mismatch between features and centers. "
                f"Got {features_sub.shape[1]} and {centers.shape[1]}."
            )

        dists = torch.cdist(features_sub, centers)
        pred_idx = torch.argmin(dists, dim=1)
        pred_labels = selected_tensor[pred_idx]
        acc = (pred_labels == labels_sub).float().mean().item()
        accuracies.append(float(acc))

    return accuracies
