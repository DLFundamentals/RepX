"""Utility helpers for label subset filtering and remapping."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def _map_labels_to_indices(
    labels: torch.Tensor,
    selected_classes: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map selected class ids to contiguous indices and return in-subset mask.

    Parameters
    ----------
    labels : Tensor, shape (num_samples,)
        Label vector to filter and remap.
    selected_classes : sequence[int]
        Class ids to keep. Returned mapped labels follow this order.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(mapped_labels, mask)`` where ``mask`` indicates selected samples and
        ``mapped_labels`` are in ``0..len(selected_classes)-1``.
    """
    selected: List[int] = [int(c) for c in selected_classes]
    if len(selected) == 0:
        raise ValueError("selected_classes must contain at least one class.")

    selected_tensor = torch.tensor(selected, device=labels.device, dtype=labels.dtype)
    mask = torch.isin(labels, selected_tensor)
    labels_sub = labels[mask]

    mapped_labels = torch.empty(labels_sub.shape[0], device=labels.device, dtype=torch.long)
    for mapped_idx, class_id in enumerate(selected):
        mapped_labels[labels_sub == class_id] = mapped_idx

    return mapped_labels, mask
