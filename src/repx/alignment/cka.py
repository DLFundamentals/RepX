"""Centered Kernel Alignment (CKA) in PyTorch.

This module provides kernel-based representational comparison utilities and
the `CKA` class.
"""

from __future__ import annotations

from typing import Literal

import torch

__all__ = ["CKA"]


# ---------------------------------------------------------------------------
# Internal: kernel functions
# ---------------------------------------------------------------------------


def _linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute linear kernel matrix `K = X @ X.T`."""
    return X @ X.T


_KERNELS: dict[str, object] = {
    "linear": _linear_kernel,
}


# ---------------------------------------------------------------------------
# Internal: debiased HSIC estimator
# ---------------------------------------------------------------------------


def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Compute unbiased HSIC estimator.

    K and L must have zeroed diagonals before calling this.

    The estimator is:

        HSIC(K, L) = [ tr(KL) + (1ᵀK1)(1ᵀL1)/((n-1)(n-2))
                       - 2/(n-2) · 1ᵀKL1 ] / (n(n-3))
    """
    n = K.shape[0]
    ones = torch.ones(n, dtype=K.dtype, device=K.device)
    term1 = torch.trace(K @ L)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (ones @ K @ L @ ones)
    return (term1 + term2 - term3) / (n * (n - 3))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CKA:
    """Centered Kernel Alignment in PyTorch.

    Compares two representation spaces with centered kernel alignment using
    the debiased HSIC estimator.

    Parameters
    ----------
    kernel : str
        Kernel used to build the similarity matrix.

        ``"linear"`` (default)
            K = X @ Xᵀ.  Invariant to **orthogonal transformations** and
            **isotropic scaling** of the rows.

    Notes
    -----
    - Invariant to orthogonal transforms and isotropic scaling with linear kernel.
    - Requires at least 4 stimuli (rows).
    """

    def __init__(self, kernel: Literal["linear"] = "linear") -> None:
        if kernel not in _KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Choose from {sorted(_KERNELS)}."
            )
        self.kernel = kernel

    def compute_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the kernel (Gram) matrix for a set of representations.

        Examples
        --------
        ```python
        import torch
        from repx.alignment import CKA

        X = torch.randn(4, 3)
        K = CKA().compute_kernel(X)
        print(K.shape)  # torch.Size([4, 4])
        ```

        Parameters
        ----------
        X : Tensor, shape (n_samples, n_features)
            One row per stimulus.

        Returns
        -------
        K : Tensor, shape (n_samples, n_samples)
            Symmetric positive semi-definite kernel matrix.
        """
        return _KERNELS[self.kernel](X)  # type: ignore[operator]

    def cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute Centered Kernel Alignment (CKA).

        Builds a kernel matrix for each representation, zeros the diagonals,
        and computes the normalised debiased HSIC.

        Requires at least 4 stimuli (rows) due to the ``n*(n-3)``
        denominator in the unbiased HSIC estimator.

        Examples
        --------
        ```python
        import torch
        from repx.alignment import CKA

        cka = CKA(kernel="linear")
        X = torch.randn(20, 64)
        Y = torch.randn(20, 128)

        score = cka.cka(X, Y)
        print(score.shape)  # torch.Size([])
        ```

        Parameters
        ----------
        X, Y : Tensor, shape (n_samples, n_features_*)
            Row-per-sample matrices.  Both must have the same number of
            rows; feature dimensionalities may differ.

        Returns
        -------
        score : scalar Tensor
            CKA score.  Returns 1.0 for identical representations, ≈ 0.0
            for independent ones.  Call ``.item()`` for a Python float.

        Raises
        ------
        ValueError
            If X and Y have different number of samples or n < 4.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples (rows). "
                f"Got X: {X.shape[0]}, Y: {Y.shape[0]}."
            )
        n = X.shape[0]
        if n < 4:
            raise ValueError(
                f"At least 4 samples are required for the debiased HSIC "
                f"estimator (got {n})."
            )

        K = self.compute_kernel(X).clone()
        L = self.compute_kernel(Y).clone()
        K.fill_diagonal_(0.0)
        L.fill_diagonal_(0.0)

        hsic_kl = _hsic_unbiased(K, L)
        hsic_kk = _hsic_unbiased(K, K)
        hsic_ll = _hsic_unbiased(L, L)

        denom = (hsic_kk * hsic_ll).clamp(min=0.0).sqrt()
        return torch.where(
            denom > 0,
            hsic_kl / denom,
            torch.zeros((), dtype=X.dtype, device=X.device),
        )
