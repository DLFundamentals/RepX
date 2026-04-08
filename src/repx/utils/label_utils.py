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

    mapped_labels = torch.empty(
        labels_sub.shape[0], device=labels.device, dtype=torch.long
    )
    for mapped_idx, class_id in enumerate(selected):
        mapped_labels[labels_sub == class_id] = mapped_idx

    return mapped_labels, mask


def _filter_features_and_map_labels(
    features: torch.Tensor,
    labels: torch.Tensor,
    selected_classes: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter features to selected classes and return mapped labels.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(filtered_features, mapped_labels)`` for selected classes only.
    """
    mapped_labels, mask = _map_labels_to_indices(labels, selected_classes)
    filtered_features = features[mask]

    if mapped_labels.numel() == 0:
        raise ValueError("No samples match selected_classes.")

    return filtered_features, mapped_labels


def _sample_per_class(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_shots: int,
    num_classes: int,
    strict: bool = False,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample up to ``n_shots`` examples per mapped class.

    Parameters
    ----------
    features : Tensor, shape (num_samples, feature_dim)
        Feature matrix.
    labels : Tensor, shape (num_samples,)
        Mapped labels in ``0..num_classes-1``.
    n_shots : int
        Requested examples per class.
    num_classes : int
        Number of mapped classes.
    strict : bool, default=False
        If True, each class must provide at least ``n_shots`` examples.
    seed : int, optional
        Optional seed for deterministic downsampling.

    Returns
    -------
    tuple[Tensor, Tensor]
        Sampled ``(features, labels)``.
    """
    if n_shots <= 0:
        raise ValueError(f"n_shots must be positive. Got {n_shots}.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive. Got {num_classes}.")

    chosen_indices: List[int] = []
    for class_id in range(num_classes):
        idxs = (labels == class_id).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            if strict:
                raise ValueError(
                    f"Class {class_id} has only 0 samples, "
                    f"but n_shots={n_shots} was requested."
                )
            continue

        if strict and idxs.numel() < n_shots:
            raise ValueError(
                f"Class {class_id} has only {idxs.numel()} samples, "
                f"but n_shots={n_shots} was requested."
            )

        take_count = min(n_shots, idxs.numel())
        if take_count < idxs.numel():
            if seed is not None:
                torch.manual_seed(seed)
            take = torch.randperm(idxs.numel(), device=idxs.device)[:take_count]
            idxs = idxs[take]

        chosen_indices.extend(idxs.tolist())

    if len(chosen_indices) == 0:
        raise ValueError("No samples available to sample.")

    index_tensor = torch.tensor(
        chosen_indices, device=features.device, dtype=torch.long
    )
    return features[index_tensor], labels[index_tensor]
