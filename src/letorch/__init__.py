"""letorch — PyTorch-native tools for neural representation analysis.

LeTorch provides efficient, GPU-friendly implementations of neural representation
similarity metrics including Representation Similarity Analysis (RSA) and
Centered Kernel Alignment (CKA).

Features
--------
- **GPU Support**: All operations run on GPU tensors with no code changes
- **PyTorch Integration**: Native PyTorch tensors, no scipy dependency at runtime
- **Debiased Estimators**: Uses unbiased HSIC for robust CKA scores
- **Multiple Metrics**: Support for correlation, cosine, Euclidean, Manhattan distances
- **Flexible Comparisons**: Pearson and Spearman rank correlation options

Quick Start
-----------
>>> import torch
>>> from letorch import RSA, CKA
>>>
>>> X = torch.randn(50, 512)   # Representation from model A
>>> Y = torch.randn(50, 768)   # Representation from model B
>>>
>>> # Representation Similarity Analysis
>>> rsa = RSA(rdm_metric="correlation", compare="spearman")
>>> rsa_score = rsa.rsa(X, Y)
>>>
>>> # Centered Kernel Alignment
>>> cka = CKA(kernel="linear")
>>> cka_score = cka.cka(X, Y)
>>>
>>> # GPU support — just use GPU tensors
>>> rsa_gpu = rsa.rsa(X.cuda(), Y.cuda())

Subpackages
-----------
alignment
    Alignment-based representation metrics (RSA, CKA).
accuracy
    Accuracy-based representation metrics.
geometry
    Geometry-based representation metrics.
core
    Shared utilities and base classes.
"""

from .alignment import RSA, CKA

__version__ = "0.1.1"
__all__ = ["RSA", "CKA"]
