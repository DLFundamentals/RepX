"""Transfer-based representation metrics.

This subpackage exposes:
- NCCCEvaluator: class-based nearest class center evaluation.
- LinearProbeEvaluator: linear-probe transfer evaluation.
"""

from .linear_probe import LinearProbeEvaluator
from .nccc import NCCCEvaluator

__all__ = [
    "NCCCEvaluator",
    "LinearProbeEvaluator",
]
