"""Alignment-based representation metrics.

This subpackage exposes:
- RSA: Representation Similarity Analysis.
- CKA: Centered Kernel Alignment.
"""

from .rsa import RSA
from .cka import CKA

__version__ = "0.1.1"
__all__ = ["RSA", "CKA"]
