"""letorch.alignment — Alignment-based representation metrics.

This subpackage provides methods for measuring the similarity of neural
representations: how similarly two representations organize the same stimuli.

Classes
-------
RSA : Representation Similarity Analysis
    Compares representations using pairwise distance correlation.
CKA : Centered Kernel Alignment
    Compares representations using centered kernel alignment with debiased HSIC.

Examples
--------
>>> import torch
>>> from letorch.alignment import RSA, CKA
>>>
>>> X = torch.randn(50, 128)  # Representation from model A
>>> Y = torch.randn(50, 256)  # Representation from model B (different dim okay)
>>>
>>> rsa = RSA(rdm_metric="correlation", compare="spearman")
>>> rsa_score = rsa.rsa(X, Y)
>>>
>>> cka = CKA(kernel="linear")
>>> cka_score = cka.cka(X, Y)

Device Support
--------------
All operations are device-agnostic. Pass GPU tensors and everything runs on
GPU with no code changes:

>>> X_gpu = X.cuda()
>>> Y_gpu = Y.cuda()
>>> score = cka.cka(X_gpu, Y_gpu)
"""

from .rsa import RSA
from .cka import CKA

__version__ = "0.1.1"
__all__ = ["RSA", "CKA"]
