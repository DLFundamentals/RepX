"""Representation Similarity Analysis (RSA) in PyTorch.

This module provides distance-based representational comparison utilities and
the `RSA` class.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

__all__ = ["RSA"]


# ---------------------------------------------------------------------------
# Internal: pairwise distance kernels
# ---------------------------------------------------------------------------


def _correlation_rdm(X: torch.Tensor) -> torch.Tensor:
    """Compute correlation-distance RDM.

    Args:
        X: Input tensor of shape `(n_samples, n_features)`.

    Returns:
        Tensor of shape `(n_samples, n_samples)` with zero diagonal.
    """
    X_c = X - X.mean(dim=1, keepdim=True)
    norms = X_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X_n = X_c / norms
    dist = 1.0 - (X_n @ X_n.T)
    dist.fill_diagonal_(0.0)
    return dist


def _cosine_rdm(X: torch.Tensor) -> torch.Tensor:
    """Compute cosine-distance RDM.

    Args:
        X: Input tensor of shape `(n_samples, n_features)`.

    Returns:
        Tensor of shape `(n_samples, n_samples)` with zero diagonal.
    """
    X_n = F.normalize(X, p=2, dim=1)
    dist = 1.0 - (X_n @ X_n.T)
    dist.fill_diagonal_(0.0)
    return dist


def _euclidean_rdm(X: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean-distance RDM.

    Args:
        X: Input tensor of shape `(n_samples, n_features)`.

    Returns:
        Tensor of shape `(n_samples, n_samples)` with pairwise L2 distances.
    """
    return torch.cdist(X, X, p=2)


def _cityblock_rdm(X: torch.Tensor) -> torch.Tensor:
    """Compute Manhattan-distance RDM.

    Args:
        X: Input tensor of shape `(n_samples, n_features)`.

    Returns:
        Tensor of shape `(n_samples, n_samples)` with pairwise L1 distances.
    """
    return torch.cdist(X, X, p=1)


_METRICS: dict[str, object] = {
    "correlation": _correlation_rdm,
    "cosine": _cosine_rdm,
    "euclidean": _euclidean_rdm,
    "cityblock": _cityblock_rdm,
}


# ---------------------------------------------------------------------------
# Internal: rank correlation helpers
# ---------------------------------------------------------------------------


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Return 0-based dense ranks for a 1-D tensor.

    Ties are broken by order-of-appearance (consistent with
    scipy's default for continuous data, where ties are vanishingly rare).
    """
    return x.argsort().argsort().to(x.dtype)


def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation for two 1-D tensors."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = xm.norm() * ym.norm()
    return torch.where(
        denom > 0,
        (xm * ym).sum() / denom,
        torch.zeros((), dtype=x.dtype, device=x.device),
    )


def _spearmanr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Spearman correlation for two 1-D tensors."""
    return _pearsonr(_rank(x), _rank(y))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RSA:
    """Representational Similarity Analysis in PyTorch.

    Compares two representation spaces by correlating their
    Representational Dissimilarity Matrices (RDMs).

    Parameters
    ----------
    rdm_metric : {"correlation", "cosine", "euclidean", "cityblock"}
        Pairwise distance metric used when building RDMs.
        Options: ``"correlation"`` (1 − Pearson r), ``"cosine"``
        (1 − cosine similarity), ``"euclidean"`` (L2 distance),
        and ``"cityblock"`` (L1/Manhattan distance).
    compare : {"spearman", "pearson"}
        Correlation method used when comparing RDM vectors.

    Notes
    -----
    - Identical representations produce scores near 1.
    - Independent representations usually produce scores near 0.
    """

    def __init__(
        self,
        rdm_metric: Literal[
            "correlation", "cosine", "euclidean", "cityblock"
        ] = "correlation",
        compare: Literal["spearman", "pearson"] = "spearman",
    ) -> None:
        if rdm_metric not in _METRICS:
            raise ValueError(
                f"Unknown metric '{rdm_metric}'. Choose from {sorted(_METRICS)}."
            )
        if compare not in ("spearman", "pearson"):
            raise ValueError(
                f"Unknown compare method '{compare}'. Use 'spearman' or 'pearson'."
            )
        self.rdm_metric = rdm_metric
        self.compare = compare

    def compute_rdm(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the Representational Dissimilarity Matrix (RDM).

        Parameters
        ----------
        X : Tensor, shape (n_samples, n_features)
            One row per sample.

        Returns
        -------
        rdm : Tensor, shape (n_samples, n_samples)
            Symmetric matrix with zero diagonal.

        Examples
        --------
        ```python
        import torch
        from letorch.alignment import RSA

        X = torch.randn(5, 3)
        rdm = RSA(rdm_metric="correlation").compute_rdm(X)
        print(rdm.shape)  # torch.Size([5, 5])
        ```
        """
        return _METRICS[self.rdm_metric](X)  # type: ignore[operator]

    def rdm_upper_tri(self, rdm: torch.Tensor) -> torch.Tensor:
        """Extract the strict upper-triangle of an RDM as a flat 1-D vector.

        Parameters
        ----------
        rdm : Tensor, shape (n, n)

        Returns
        -------
        vec : Tensor, shape (n*(n-1)//2,)

        Examples
        --------
        ```python
        import torch
        from letorch.alignment import RSA

        rdm = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
        )
        vec = RSA().rdm_upper_tri(rdm)
        print(vec)  # tensor([1., 2., 3.])
        ```
        """
        n = rdm.shape[0]
        rows, cols = torch.triu_indices(n, n, offset=1, device=rdm.device)
        return rdm[rows, cols]

    def rsa(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute Representational Similarity Analysis (RSA).

        Builds an RDM for each representation matrix, vectorises the upper
        triangles, and correlates them using the settings from ``__init__``.

        Parameters
        ----------
        X, Y : Tensor, shape (n_samples, n_features_*)
            Row-per-sample matrices.  Both must have the same number of rows;
            feature dimensionalities may differ.

        Returns
        -------
        r : scalar Tensor
            Correlation coefficient in [−1, +1].
            Call ``.item()`` to get a Python float.

        Raises
        ------
        ValueError
            If X and Y have different number of samples.

        Examples
        --------
        ```python
        import torch
        from letorch.alignment import RSA

        rsa = RSA(rdm_metric="correlation", compare="spearman")
        X = torch.randn(20, 64)
        Y = torch.randn(20, 128)

        score = rsa.rsa(X, Y)
        print(score.shape)  # torch.Size([])
        ```
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples (rows). "
                f"Got X: {X.shape[0]}, Y: {Y.shape[0]}."
            )

        vec_x = self.rdm_upper_tri(self.compute_rdm(X))
        vec_y = self.rdm_upper_tri(self.compute_rdm(Y))

        if self.compare == "spearman":
            return _spearmanr(vec_x, vec_y)
        else:
            return _pearsonr(vec_x, vec_y)
