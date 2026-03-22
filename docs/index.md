# LeTorch

PyTorch-native tools for neural representation analysis.

LeTorch currently provides:

- **RSA** (Representation Similarity Analysis)
- **CKA** (Centered Kernel Alignment)

## Quick Start

```python
import torch
from letorch import RSA, CKA

X = torch.randn(50, 512)
Y = torch.randn(50, 768)

rsa = RSA(rdm_metric="correlation", compare="spearman")
cka = CKA(kernel="linear")

print(rsa.rsa(X, Y).item())
print(cka.cka(X, Y).item())
```

For full API docs pulled from docstrings, see [Alignment](api/alignment.md), [Accuracy](api/accuracy.md), [Core](api/core.md), and [Geometry](api/geometry.md).
