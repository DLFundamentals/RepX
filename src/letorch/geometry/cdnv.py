"""Class-Distance Normalized Variance (CDNV) metric."""

from __future__ import annotations

from typing import Optional, Union

import torch

from letorch.utils.cdnv_utils import (
    _compute_class_means,
    _resolve_num_classes,
    _validate_features_and_labels,
)

__all__ = ["compute_cdnv"]


def compute_cdnv(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    eps: float = 1e-12,
) -> float:
    """Compute class-distance normalized variance (CDNV).

    For each unordered class pair `(c1, c2)`, this computes:

    - class variance proxy: `E[||x||^2] - ||E[x]||^2` for each class,
    - average variance across the two classes,
    - normalized by squared distance between class means.

    The final result is averaged over `num_classes * (num_classes - 1) / 2`
    possible class pairs.

    Parameters
    ----------
    features : Tensor, shape (num_samples, feature_dim)
        Embedding matrix.
    labels : Tensor, shape (num_samples,)
        Class labels for each sample.
    num_classes : int, optional
        Number of classes. If omitted, inferred as `labels.max() + 1`.
    device : str or torch.device, default="cpu"
        Device used for metric computation.
    eps : float, default=1e-12
        Small positive value to avoid division by zero for degenerate class
        pairs.

    Returns
    -------
    float
        Average CDNV value.

    Examples
    --------
    ```python
    import torch
    from letorch.geometry import compute_cdnv

    x = torch.randn(100, 128)
    y = torch.randint(0, 10, (100,))

    score = compute_cdnv(x, y, num_classes=10)
    print(score)
    ```
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    features, labels = _validate_features_and_labels(features, labels, device=device)
    resolved_num_classes = _resolve_num_classes(labels, num_classes)

    means = _compute_class_means(features, labels, resolved_num_classes)
    second_moments = torch.zeros(
        resolved_num_classes, device=features.device, dtype=features.dtype
    )
    class_present = torch.zeros(
        resolved_num_classes, device=features.device, dtype=torch.bool
    )

    for class_idx in range(resolved_num_classes):
        idxs = (labels == class_idx).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        feats_c = features[idxs]
        second_moments[class_idx] = feats_c.pow(2).sum(dim=1).mean()
        class_present[class_idx] = True

    total_num_pairs = resolved_num_classes * (resolved_num_classes - 1) / 2
    if total_num_pairs == 0:
        return 0.0

    avg_cdnv = 0.0

    for class1 in range(resolved_num_classes):
        for class2 in range(class1 + 1, resolved_num_classes):
            if not class_present[class1] or not class_present[class2]:
                continue

            variance1 = torch.abs(second_moments[class1] - means[class1].pow(2).sum())
            variance2 = torch.abs(second_moments[class2] - means[class2].pow(2).sum())
            variance_avg = (variance1 + variance2) / 2.0

            dist_sq = torch.norm(means[class1] - means[class2]).pow(2)
            if dist_sq <= eps:
                continue

            avg_cdnv += (variance_avg / dist_sq / total_num_pairs).item()

    return float(avg_cdnv)
