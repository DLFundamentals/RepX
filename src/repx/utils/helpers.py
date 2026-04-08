"""Utility helpers for CDNV geometry metrics."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch


def _validate_features_and_labels(
    features: torch.Tensor,
    labels: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate CDNV inputs and move tensors to a target device.

    Parameters
    ----------
    features : Tensor, shape (num_samples, feature_dim)
        Embedding matrix.
    labels : Tensor, shape (num_samples,)
        Class labels for each sample.
    device : str or torch.device, default="cpu"
        Target device.

    Returns
    -------
    tuple[Tensor, Tensor]
        `(features, labels)` moved to `device`.
    """
    if features.ndim != 2:
        raise ValueError(
            "features must be a 2D tensor of shape (num_samples, feature_dim)."
        )
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (num_samples,).")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "features and labels must have the same number of samples. "
            f"Got {features.shape[0]} and {labels.shape[0]}."
        )
    if labels.numel() == 0:
        raise ValueError("labels must be non-empty.")

    resolved_device = torch.device(device)
    return features.to(resolved_device), labels.to(resolved_device)


def _resolve_num_classes(labels: torch.Tensor, num_classes: Optional[int]) -> int:
    """Resolve class count from an explicit value or from labels.

    Parameters
    ----------
    labels : Tensor, shape (num_samples,)
        Label vector.
    num_classes : int, optional
        Explicit class count.

    Returns
    -------
    int
        Resolved number of classes.
    """
    if num_classes is not None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive when provided.")
        return int(num_classes)

    inferred = int(labels.max().item()) + 1
    if inferred <= 0:
        raise ValueError("Could not infer a valid number of classes from labels.")
    return inferred


def _resolve_selected_classes(
    labels: torch.Tensor,
    selected_classes: Optional[Sequence[int]],
) -> List[int]:
    """Resolve selected classes from explicit ids or infer from labels."""
    if selected_classes is None:
        return sorted(int(c) for c in torch.unique(labels).tolist())

    resolved = [int(c) for c in selected_classes]
    if len(resolved) == 0:
        raise ValueError("selected_classes must contain at least one class.")
    return resolved


def _compute_class_means(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute per-class mean vectors.

    Missing classes receive all-zero vectors.

    Parameters
    ----------
    features : Tensor, shape (num_samples, feature_dim)
        Embedding matrix.
    labels : Tensor, shape (num_samples,)
        Label vector.
    num_classes : int
        Number of classes.

    Returns
    -------
    Tensor, shape (num_classes, feature_dim)
        Mean feature vector for each class.
    """
    feature_dim = features.shape[1]
    means = torch.zeros(
        num_classes, feature_dim, device=features.device, dtype=features.dtype
    )

    for class_idx in range(num_classes):
        idxs = (labels == class_idx).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        means[class_idx] = features[idxs].mean(dim=0)

    return means
