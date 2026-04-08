"""Torch-specific utility helpers for configurable training components."""

from __future__ import annotations

from typing import Type, Union

import torch


def _resolve_optimizer(
    optimizer: Union[str, Type[torch.optim.Optimizer]],
) -> Type[torch.optim.Optimizer]:
    """Resolve an optimizer spec to an optimizer class."""
    if isinstance(optimizer, str):
        key = optimizer.lower()
        registry = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if key not in registry:
            raise ValueError(
                "Unsupported optimizer string. Supported values: "
                "adam, adamw, sgd, rmsprop."
            )
        return registry[key]

    if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        return optimizer

    raise TypeError("optimizer must be a string or a torch.optim.Optimizer class.")


def _resolve_loss(
    loss_fn: Union[str, Type[torch.nn.Module]],
) -> Type[torch.nn.Module]:
    """Resolve a loss spec to a loss module class."""
    if isinstance(loss_fn, str):
        key = loss_fn.lower()
        registry = {
            "cross_entropy": torch.nn.CrossEntropyLoss,
            "ce": torch.nn.CrossEntropyLoss,
            "nll": torch.nn.NLLLoss,
        }
        if key not in registry:
            raise ValueError(
                "Unsupported loss_fn string. Supported values: "
                "cross_entropy, ce, nll."
            )
        return registry[key]

    if isinstance(loss_fn, type) and issubclass(loss_fn, torch.nn.Module):
        return loss_fn

    raise TypeError("loss_fn must be a string or a torch.nn.Module class.")
