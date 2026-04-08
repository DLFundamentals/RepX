"""PyTorch-native tools for neural representation analysis.

LeTorch provides alignment metrics for comparing neural representations,
including RSA and CKA, with native PyTorch tensor support on CPU and GPU.
"""

from .alignment import CKA, RSA
from .transfer import (
    LinearProbeEvaluator,
    NCCCEvaluator,
)

__version__ = "0.1.1"
__all__ = [
    "RSA",
    "CKA",
    "NCCCEvaluator",
    "LinearProbeEvaluator",
]
