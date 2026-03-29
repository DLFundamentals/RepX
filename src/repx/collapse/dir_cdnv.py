"""Directional Class-Distance Normalized Variance (directional CDNV) metric."""

from __future__ import annotations

from typing import Optional, Union

import torch

from repx.utils.cdnv_utils import (
    _compute_class_means,
    _resolve_num_classes,
    _validate_features_and_labels,
)

__all__ = ["compute_directional_cdnv"]


def compute_directional_cdnv(
    features: torch.Tensor,
    labels: torch.Tensor,
    means: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    eps: float = 1e-12,
) -> float:
    """Compute directional CDNV.

    Directional CDNV measures, for ordered class pairs `(c1, c2)`, the variance
    of class-`c1` features projected onto the direction from mean(c1) to
    mean(c2), normalized by squared inter-class distance.

    Parameters
    ----------
    features : Tensor, shape (num_samples, feature_dim)
        Embedding matrix.
    labels : Tensor, shape (num_samples,)
        Class labels for each sample.
    means : Tensor, shape (num_classes, feature_dim), optional
        Precomputed class means. If omitted, class means are computed from
        `features` and `labels`.
    num_classes : int, optional
        Number of classes. If omitted, inferred as `labels.max() + 1`.
    device : str or torch.device, default="cpu"
        Device used for metric computation.
    eps : float, default=1e-12
        Small positive value to avoid division by zero for degenerate class
        directions.

    Returns
    -------
    float
        Average directional CDNV value over ordered class pairs.

    Examples
    --------
    ```python
    import torch
    from repx.geometry import compute_directional_cdnv

    x = torch.randn(100, 128)
    y = torch.randint(0, 10, (100,))

    score = compute_directional_cdnv(x, y, num_classes=10)
    print(score)
    ```
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    features, labels = _validate_features_and_labels(features, labels, device=device)
    resolved_num_classes = _resolve_num_classes(labels, num_classes)

    if means is None:
        means = _compute_class_means(features, labels, resolved_num_classes)
    else:
        means = means.to(features.device)

    if means.ndim != 2:
        raise ValueError(
            "means must be a 2D tensor of shape (num_classes, feature_dim)."
        )
    if means.shape[0] < resolved_num_classes:
        raise ValueError(
            "means first dimension must be at least num_classes. "
            f"Got {means.shape[0]} < {resolved_num_classes}."
        )

    total_num_pairs = resolved_num_classes * (resolved_num_classes - 1)
    if total_num_pairs == 0:
        return 0.0

    avg_dir_cdnv = torch.zeros((), device=features.device, dtype=features.dtype)

    for class1 in range(resolved_num_classes):
        idxs = (labels == class1).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue

        features1 = features[idxs]

        for class2 in range(resolved_num_classes):
            if class2 == class1:
                continue

            v = means[class2] - means[class1]
            v_norm = v.norm()
            if v_norm <= eps:
                continue

            v_hat = v / v_norm
            projections = (features1 - means[class1]) @ v_hat
            dir_var = projections.pow(2).mean()
            dir_cdnv = dir_var / v_norm.pow(2)
            avg_dir_cdnv += dir_cdnv / total_num_pairs

    return float(avg_dir_cdnv.item())
