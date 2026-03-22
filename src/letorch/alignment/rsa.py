"""letorch.alignment.rsa — Representation Similarity Analysis in PyTorch.

Representation Similarity Analysis (RSA) measures the similarity between
two sets of neural representations by comparing their Representational
Dissimilarity Matrices (RDMs). It works by computing pairwise distances
between stimuli in each representation space, then correlating these distance
structures to quantify representational similarity.

Supported Metrics
-----------------
- correlation: 1 − Pearson r (row-wise scaling and translation invariant)
- cosine: 1 − cosine similarity (row-wise scaling invariant)
- euclidean: L2 distance
- cityblock: L1 / Manhattan distance

Device Support
--------------
All operations are device-agnostic: pass GPU tensors and everything runs on
the GPU with no code changes.
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
    """Compute correlation-based representational dissimilarity matrix.

    Computes 1 − Pearson-r (correlation distance) for every pair of rows.
    Equivalent to cosine distance on row-centred data. Invariant to row-wise
    scaling and translation.

    Parameters
    ----------
    X : Tensor, shape (n, d)
        Input matrix with one row per stimulus.

    Returns
    -------
    dist : Tensor, shape (n, n)
        Symmetric matrix with zero diagonal; dist[i, j] = 1 − r(X[i], X[j]).
    """
    X_c = X - X.mean(dim=1, keepdim=True)
    norms = X_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X_n = X_c / norms
    dist = 1.0 - (X_n @ X_n.T)
    dist.fill_diagonal_(0.0)
    return dist


def _cosine_rdm(X: torch.Tensor) -> torch.Tensor:
    """Compute cosine-based representational dissimilarity matrix.

    Computes 1 − cosine similarity for every pair of rows. Invariant to
    row-wise scaling.

    Parameters
    ----------
    X : Tensor, shape (n, d)
        Input matrix with one row per stimulus.

    Returns
    -------
    dist : Tensor, shape (n, n)
        Symmetric matrix with zero diagonal; dist[i, j] = 1 − cos(X[i], X[j]).
    """
    X_n = F.normalize(X, p=2, dim=1)
    dist = 1.0 - (X_n @ X_n.T)
    dist.fill_diagonal_(0.0)
    return dist


def _euclidean_rdm(X: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance representational dissimilarity matrix.

    Computes L2 distance for every pair of rows.

    Parameters
    ----------
    X : Tensor, shape (n, d)
        Input matrix with one row per stimulus.

    Returns
    -------
    dist : Tensor, shape (n, n)
        Symmetric matrix with zero diagonal; dist[i, j] = ||X[i] − X[j]||_2.
    """
    return torch.cdist(X, X, p=2)


def _cityblock_rdm(X: torch.Tensor) -> torch.Tensor:
    """Compute Manhattan distance representational dissimilarity matrix.

    Computes L1 distance (cityblock/Manhattan) for every pair of rows.

    Parameters
    ----------
    X : Tensor, shape (n, d)
        Input matrix with one row per stimulus.

    Returns
    -------
    dist : Tensor, shape (n, n)
        Symmetric matrix with zero diagonal; dist[i, j] = ||X[i] − X[j]||_1.
    """
    return torch.cdist(X, X, p=1)


_METRICS: dict[str, object] = {
    "correlation": _correlation_rdm,
    "cosine":      _cosine_rdm,
    "euclidean":   _euclidean_rdm,
    "cityblock":   _cityblock_rdm,
}


# ---------------------------------------------------------------------------
# Internal: rank correlation helpers
# ---------------------------------------------------------------------------

def _rank(x: torch.Tensor) -> torch.Tensor:
    """Compute 0-based dense ranks of a 1-D tensor.

    Ties are broken by order-of-appearance (consistent with scipy's default
    for continuous data, where ties are vanishingly rare).

    Parameters
    ----------
    x : Tensor, shape (n,)
        Input 1-D tensor.

    Returns
    -------
    ranks : Tensor, shape (n,)
        Dense ranks as floats in the same dtype as input and device as input.
    """
    return x.argsort().argsort().to(x.dtype)


def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation coefficient between two 1-D tensors.

    Parameters
    ----------
    x, y : Tensor, shape (n,)
        Input 1-D tensors.

    Returns
    -------
    r : scalar Tensor
        Pearson correlation coefficient in [−1, +1].
    """
    xm = x - x.mean()
    ym = y - y.mean()
    denom = xm.norm() * ym.norm()
    return torch.where(
        denom > 0,
        (xm * ym).sum() / denom,
        torch.zeros((), dtype=x.dtype, device=x.device),
    )


def _spearmanr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Spearman rank correlation between two 1-D tensors.

    Parameters
    ----------
    x, y : Tensor, shape (n,)
        Input 1-D tensors.

    Returns
    -------
    rho : scalar Tensor
        Spearman rank correlation coefficient in [−1, +1].
    """
    return _pearsonr(_rank(x), _rank(y))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RSA:
    """Representational Similarity Analysis in PyTorch.

    RSA measures the similarity of two representation spaces by comparing their
    Representational Dissimilarity Matrices (RDMs). Two matrices with identical
    structure (relatedness of stimuli) yield high correlation, while independent
    representations yield correlation near 0.

    Parameters
    ----------
    rdm_metric : {"correlation", "cosine", "euclidean", "cityblock"}
        Pairwise distance metric used when building RDMs. Default: "correlation".

        - "correlation": 1 − Pearson r between rows. Invariant to row-wise
          scaling and translation.
        - "cosine": 1 − cosine similarity. Invariant to row-wise scaling.
        - "euclidean": L2 distance.
        - "cityblock": L1 / Manhattan distance.

    compare : {"spearman", "pearson"}
        Correlation method used when comparing RDM vectors. Default: "spearman".

    Attributes
    ----------
    rdm_metric : str
        The distance metric used for RDM computation.
    compare : str
        The correlation method used for comparing RDMs.

    Examples
    --------
    >>> import torch
    >>> from letorch.alignment import RSA
    >>> rsa = RSA(rdm_metric="correlation", compare="spearman")
    >>> X = torch.randn(50, 128)
    >>> Y = torch.randn(50, 256)
    >>> rsa.rsa(X, X).item()   # identical → 1.0
    1.0
    >>> rsa.rsa(X, Y).item()   # independent → ≈ 0.0
    """

    def __init__(
        self,
        rdm_metric: Literal["correlation", "cosine", "euclidean", "cityblock"] = "correlation",
        compare: Literal["spearman", "pearson"] = "spearman",
    ) -> None:
        """Initialize RSA with specified RDM metric and comparison method.

        Parameters
        ----------
        rdm_metric : {"correlation", "cosine", "euclidean", "cityblock"}
            Distance metric for RDM computation. Default: "correlation".
        compare : {"spearman", "pearson"}
            Correlation method for comparing RDM vectors. Default: "spearman".

        Raises
        ------
        ValueError
            If rdm_metric is not a supported metric or compare is not a supported method.
        """
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

        Builds an RDM using the pairwise distance metric specified in ``__init__``.
        The RDM is a symmetric matrix where each element [i, j] represents the
        dissimilarity between stimuli i and j.

        Parameters
        ----------
        X : Tensor, shape (n_stimuli, n_features)
            Input matrix with one row per stimulus.

        Returns
        -------
        rdm : Tensor, shape (n_stimuli, n_stimuli)
            Symmetric matrix with zero diagonal. rdm[i, j] = distance between
            stimuli i and j, computed using the specified metric.
        """
        return _METRICS[self.rdm_metric](X)  # type: ignore[operator]

    def rdm_upper_tri(self, rdm: torch.Tensor) -> torch.Tensor:
        """Extract the strict upper-triangle of an RDM as a flat 1-D vector.

        Extracts all above-diagonal elements from the symmetric RDM matrix
        and returns them as a flattened vector for correlation operations.

        Parameters
        ----------
        rdm : Tensor, shape (n_stimuli, n_stimuli)
            Symmetric representational dissimilarity matrix.

        Returns
        -------
        vec : Tensor, shape (n_stimuli*(n_stimuli-1)//2,)
            Flattened strict upper-triangle elements in row-major order.
        """
        n = rdm.shape[0]
        rows, cols = torch.triu_indices(n, n, offset=1, device=rdm.device)
        return rdm[rows, cols]

    def rsa(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute Representational Similarity Analysis (RSA).

        Builds a Representational Dissimilarity Matrix (RDM) for each input,
        vectorises the upper triangles, and correlates them using the specified
        comparison method (Pearson or Spearman).

        Parameters
        ----------
        X, Y : Tensor, shape (n_stimuli, n_features_*)
            Row-per-stimulus matrices. Both must have the same number of rows;
            feature dimensionalities may differ.

        Returns
        -------
        r : scalar Tensor
            Correlation coefficient between RDM vectors. Range:
            - For Pearson/Spearman: [−1, +1]
            - 1.0 for identical representations
            - ≈0.0 for independent representations
            Call ``.item()`` to get a Python float.

        Raises
        ------
        ValueError
            If X and Y have different number of stimuli (rows).
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of stimuli (rows). "
                f"Got X: {X.shape[0]}, Y: {Y.shape[0]}."
            )

        vec_x = self.rdm_upper_tri(self.compute_rdm(X))
        vec_y = self.rdm_upper_tri(self.compute_rdm(Y))

        if self.compare == "spearman":
            return _spearmanr(vec_x, vec_y)
        else:
            return _pearsonr(vec_x, vec_y)
