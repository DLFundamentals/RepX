"""Equiangular Tight Frame (ETF) deviation metric."""

from __future__ import annotations

from typing import Optional, Union

import torch

from repx.utils.cdnv_utils import (
    _compute_class_means,
    _resolve_num_classes,
    _validate_features_and_labels,
)

__all__ = ["compute_etf_deviation"]


def compute_etf_deviation(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    eps: float = 1e-12,
) -> float:
    """Compute deviation of class means from simplex ETF geometry.

    The metric is based on class means. It first centers class means across
    active classes and L2-normalizes each mean vector. It then compares the
    resulting Gram matrix to the ETF target Gram matrix:

    - diagonal entries = 1
    - off-diagonal entries = ``-1 / (C - 1)``

    where ``C`` is the number of active classes (classes that appear in
    ``labels``). Returned value is the relative Frobenius error:

    ``||G - G_etf||_F / ||G_etf||_F``

    Parameters
    ----------
    features : Tensor, shape (num_samples, feature_dim)
        Embedding matrix.
    labels : Tensor, shape (num_samples,)
        Class labels for each sample.
    num_classes : int, optional
        Total number of classes. If omitted, inferred as ``labels.max() + 1``.
    device : str or torch.device, default="cpu"
        Device used for metric computation.
    eps : float, default=1e-12
        Numerical stability constant used in vector normalization.

    Returns
    -------
    float
        Relative ETF deviation. Lower is better; 0 indicates perfect ETF
        geometry for active class means.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    features, labels = _validate_features_and_labels(features, labels, device=device)
    resolved_num_classes = _resolve_num_classes(labels, num_classes)

    if labels.min().item() < 0 or labels.max().item() >= resolved_num_classes:
        raise ValueError(
            "labels must be in [0, num_classes - 1]. "
            f"Got min={int(labels.min().item())}, "
            f"max={int(labels.max().item())}, "
            f"num_classes={resolved_num_classes}."
        )

    means = _compute_class_means(features, labels, resolved_num_classes)

    active_classes = []
    for class_idx in range(resolved_num_classes):
        if (labels == class_idx).any():
            active_classes.append(class_idx)

    num_active = len(active_classes)
    if num_active <= 1:
        return 0.0

    active_means = means[active_classes]
    active_means = active_means - active_means.mean(dim=0, keepdim=True)

    norms = torch.linalg.vector_norm(active_means, dim=1, keepdim=True)
    active_means = active_means / norms.clamp_min(eps)

    gram = active_means @ active_means.T

    target = torch.full_like(gram, -1.0 / (num_active - 1))
    target.fill_diagonal_(1.0)

    numerator = torch.linalg.matrix_norm(gram - target, ord="fro")
    denominator = torch.linalg.matrix_norm(target, ord="fro").clamp_min(eps)

    return float((numerator / denominator).item())
