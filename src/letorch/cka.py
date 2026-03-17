"""letorch.cka — Centered Kernel Alignment (CKA) in PyTorch.

Implements both the **biased** (standard) and **debiased** (unbiased)
estimators for linear-kernel CKA.

References
----------
* Kornblith et al., "Similarity of Neural Network Representations
  Revisited", ICML 2019.  https://arxiv.org/abs/1905.00414
"""

from __future__ import annotations

import torch

__all__ = ["CKA"]


# ---------------------------------------------------------------------------
# Internal: gram-matrix centering
# ---------------------------------------------------------------------------

def _center_gram_biased(K: torch.Tensor) -> torch.Tensor:
    """Doubly-center a gram matrix: K_c = H K H.

    H = I − (1/n) 1 1ᵀ  is the centering matrix.
    """
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
    return H @ K @ H


def _center_gram_unbiased(K: torch.Tensor) -> torch.Tensor:
    """Unbiased centering for debiased CKA (Kornblith et al., 2019).

    Zeros the diagonal, removes per-column means adjusted for the
    missing diagonal, and zeros the diagonal again.  Produces an
    unbiased estimator of the centered gram matrix suitable for use
    in the debiased CKA formula.
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError(
            f"Debiased CKA requires at least 4 samples; got {n}."
        )
    Kc = K.clone()
    Kc.fill_diagonal_(0.0)
    # Column means with diagonal excluded, then globally adjusted
    means = Kc.sum(dim=0) / (n - 2)
    means = means - means.sum() / (2 * (n - 1))
    Kc = Kc - means.unsqueeze(1) - means.unsqueeze(0)
    Kc.fill_diagonal_(0.0)
    return Kc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CKA:
    """Centered Kernel Alignment (linear kernel) in PyTorch.

    Parameters
    ----------
    debiased : bool
        If ``False`` (default), use the **biased** HSIC estimator
        (standard CKA).  If ``True``, use the **unbiased** estimator
        (debiased CKA), whose expected value for independent
        representations is exactly 0, eliminating the positive bias
        that the standard estimator exhibits at finite sample sizes.

    Notes
    -----
    *Biased CKA* has a systematic positive bias for finite, independent
    representations; the bias shrinks as ``n → ∞`` but can be large
    when ``d/n`` is not small.

    *Debiased CKA* is an unbiased estimator, so its expected value for
    independent representations is 0.  It can therefore return negative
    values for a single pair, whereas biased CKA is always non-negative.

    Examples
    --------
    >>> import torch
    >>> from letorch.cka import CKA
    >>> X = torch.randn(50, 128)
    >>> CKA(debiased=False).cka(X, X).item()   # identical → 1.0
    1.0
    >>> CKA(debiased=True).cka(X, X).item()    # identical → 1.0
    1.0
    """

    def __init__(self, debiased: bool = False) -> None:
        self.debiased = debiased

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gram(X: torch.Tensor) -> torch.Tensor:
        """Linear (dot-product) gram matrix: K = X Xᵀ."""
        return X @ X.T

    def _center(self, K: torch.Tensor) -> torch.Tensor:
        if self.debiased:
            return _center_gram_unbiased(K)
        return _center_gram_biased(K)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear-kernel CKA between two representation matrices.

        Parameters
        ----------
        X, Y : Tensor, shape ``(n_samples, n_features_*)``
            Row-per-sample matrices.  Both must have the same number of
            rows; feature dimensionalities may differ.

        Returns
        -------
        score : scalar Tensor
            CKA similarity.  Biased CKA is in ``[0, 1]``; debiased CKA
            can be negative for independent representations.
            Call ``.item()`` to get a Python float.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples (rows). "
                f"Got X: {X.shape[0]}, Y: {Y.shape[0]}."
            )

        Kc = self._center(self._gram(X))
        Lc = self._center(self._gram(Y))

        hsic_xy = (Kc * Lc).sum()
        norm_x = Kc.norm()
        norm_y = Lc.norm()
        denom = norm_x * norm_y
        return torch.where(
            denom > 0,
            hsic_xy / denom,
            torch.zeros((), dtype=X.dtype, device=X.device),
        )
