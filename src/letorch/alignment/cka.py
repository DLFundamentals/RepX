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
    """Compute linear (dot-product) kernel matrix.

    Computes the Gram matrix K = X @ X.T, representing inner products
    between all pairs of rows.

    Parameters
    ----------
    X : Tensor, shape (n, d)
        Input matrix with one row per stimulus.

    Returns
    -------
    K : Tensor, shape (n, n)
        Symmetric positive semi-definite kernel (Gram) matrix.
    """
    return X @ X.T


_KERNELS: dict[str, object] = {
    "linear": _linear_kernel,
}


# ---------------------------------------------------------------------------
# Internal: debiased HSIC estimator
# ---------------------------------------------------------------------------

def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Compute unbiased Hilbert–Schmidt Independence Criterion (HSIC).

    Implements the unbiased HSIC estimator from Kornblith et al. (2019),
    which correctly estimates HSIC while avoiding bias when n >> d. Both
    kernel matrices must have their diagonals pre-zeroed.

    The estimator formula is:

        HSIC(K, L) = [tr(KL) + (1^T K 1)(1^T L 1) / ((n-1)(n-2))
                      - 2/(n-2) · 1^T KL 1] / (n(n-3))

    where 1 is the all-ones vector.

    Parameters
    ----------
    K, L : Tensor, shape (n, n)
        Symmetric kernel matrices with diagonals already set to 0.
        (Pre-zeroing is required; not done internally.)

    Returns
    -------
    hsic : scalar Tensor
        Unbiased HSIC estimate. Typically in [0, ∞) but can be negative
        for very small samples.

    References
    ----------
    Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
    Similarity of neural network representations revisited.
    International Conference on Machine Learning (ICML).
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
    kernel : {"linear"}
        Kernel function for building similarity matrices. Default: "linear".

        - "linear": K = X @ X^T (dot-product kernel). Invariant to
          orthogonal transformations and isotropic scaling of rows.

    Notes
    -----
    - Invariant to orthogonal transforms and isotropic scaling with linear kernel.
    - Requires at least 4 stimuli (rows).

    References
    ----------
    Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
    Similarity of neural network representations revisited.
    International Conference on Machine Learning (ICML).
    """

    def __init__(self, kernel: Literal["linear"] = "linear") -> None:
        if kernel not in _KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Choose from {sorted(_KERNELS)}."
            )
        self.kernel = kernel

    def compute_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the kernel (Gram) matrix for a set of representations.

        Computes the kernel matrix using the type specified in ``__init__``.
        For the default linear kernel, this is K = X @ X^T, a symmetric
        positive semi-definite matrix.

        Parameters
        ----------
        X : Tensor, shape (n_stimuli, n_features)
            Input matrix with one row per stimulus.

        Returns
        -------
        K : Tensor, shape (n_stimuli, n_stimuli)
            Symmetric positive semi-definite kernel (Gram) matrix.
        """
        return _KERNELS[self.kernel](X)  # type: ignore[operator]

    def cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute Centered Kernel Alignment (CKA).

        Builds kernel matrices for each input, centers them by zeroing diagonals,
        and computes the normalised unbiased HSIC (Hilbert–Schmidt Independence
        Criterion) using the estimator from Kornblith et al. (2019).

        Requires at least 4 stimuli (rows) due to the n*(n-3) denominator in
        the unbiased HSIC estimator.

        Parameters
        ----------
        X, Y : Tensor, shape (n_stimuli, n_features_*)
            Row-per-stimulus matrices. Both must have the same number of rows;
            feature dimensionalities may differ.

        Returns
        -------
        score : scalar Tensor
            CKA score in [0, 1].
            - 1.0 for identical representations
            - ≈0.0 for independent representations
            Call ``.item()`` to get a Python float.

        Raises
        ------
        ValueError
            If X and Y have different number of stimuli, or if either has
            fewer than 4 stimuli.

        Notes
        -----
        Device and dtype are preserved: if inputs are on GPU or use a specific
        dtype, outputs will match.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of stimuli (rows). "
                f"Got X: {X.shape[0]}, Y: {Y.shape[0]}."
            )
        n = X.shape[0]
        if n < 4:
            raise ValueError(
                f"At least 4 stimuli are required for the debiased HSIC "
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
