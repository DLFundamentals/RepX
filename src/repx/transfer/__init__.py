"""Transfer-based representation metrics.

This subpackage exposes:
- compute_nccc_centers: class center estimation for NCCC.
- evaluate_nccc: nearest-center evaluation.
"""

from .nccc import compute_nccc_centers, evaluate_nccc

__all__ = ["compute_nccc_centers", "evaluate_nccc"]
