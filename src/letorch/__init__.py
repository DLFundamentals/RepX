"""letorch — PyTorch-native tools for neural representation analysis."""

from .rsa import compute_rdm, rdm_upper_tri, rsa

__version__ = "0.1.0"
__all__ = ["compute_rdm", "rdm_upper_tri", "rsa"]
