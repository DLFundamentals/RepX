# RepX ⛵️

PyTorch-native metrics for representation analysis. 

## Installation

```bash
pip install letorch
```

## Quick Start

### RSA

```python
import torch
from letorch import RSA

# Two representation matrices — rows = stimuli, columns = features
X = torch.randn(50, 512)   # e.g. embeddings from model A
Y = torch.randn(50, 768)   # e.g. embeddings from model B (different dim is fine)

rsa = RSA(rdm_metric="correlation", compare="spearman")

# Full RSA pipeline: RDM → upper triangle → Spearman correlation
r = rsa.rsa(X, Y)
print(r.item())   # scalar in [−1, +1]

# GPU — just pass GPU tensors, no other changes needed
r_gpu = rsa.rsa(X.cuda(), Y.cuda())
```

### CKA

```python
import torch
from letorch import CKA

X = torch.randn(50, 512)
Y = torch.randn(50, 768)

cka = CKA(kernel="linear")

score = cka.cka(X, Y)
print(score.item())   # scalar in [0, 1]

# GPU
score_gpu = cka.cka(X.cuda(), Y.cuda())
```

### Geometry (CDNV)

```python
import torch
from letorch.geometry import compute_cdnv, compute_directional_cdnv

X = torch.randn(100, 128)
y = torch.randint(0, 10, (100,))

cdnv_score = compute_cdnv(X, y, num_classes=10)
dir_cdnv_score = compute_directional_cdnv(X, y, num_classes=10)

print(cdnv_score)
print(dir_cdnv_score)
```

## RSA API

### `RSA(rdm_metric="correlation", compare="spearman")`

Instantiate an RSA object with the desired settings.

| Argument | Options | Default |
|---|---|---|
| `rdm_metric` | `"correlation"`, `"cosine"`, `"euclidean"`, `"cityblock"` | `"correlation"` |
| `compare` | `"spearman"`, `"pearson"` | `"spearman"` |

### `RSA.compute_rdm(X) → Tensor`

Compute the Representational Dissimilarity Matrix for a stimulus set using the instance's `rdm_metric`.

| Metric | Description |
|---|---|
| `"correlation"` | 1 − Pearson r (default) — invariant to row-wise scaling & translation |
| `"cosine"` | 1 − cosine similarity — invariant to row-wise scaling |
| `"euclidean"` | L2 distance |
| `"cityblock"` | L1 / Manhattan distance |

Returns a symmetric `(n_stimuli, n_stimuli)` tensor with zero diagonal.

### `RSA.rdm_upper_tri(rdm) → Tensor`

Extract the strict upper triangle of an RDM as a flat 1-D vector of length `n*(n-1)//2`.

### `RSA.rsa(X, Y) → Tensor`

Full RSA pipeline: build RDMs → vectorise upper triangles → correlate, using the instance's `rdm_metric` and `compare` settings.

Returns a scalar tensor in `[−1, +1]`. Call `.item()` for a Python float.

## CKA API

### `CKA(kernel="linear")`

Instantiate a CKA object. Uses the debiased HSIC estimator (Kornblith et al., 2019) to avoid score inflation when the number of features exceeds the number of stimuli.

| Argument | Options | Default |
|---|---|---|
| `kernel` | `"linear"` | `"linear"` |

### `CKA.compute_kernel(X) → Tensor`

Compute the kernel (Gram) matrix K = X @ Xᵀ. Returns a symmetric `(n_stimuli, n_stimuli)` PSD tensor.

### `CKA.cka(X, Y) → Tensor`

Compute CKA: build kernel matrices → zero diagonals → normalised debiased HSIC.

Requires at least 4 stimuli. Returns a scalar tensor. Call `.item()` for a Python float.

## RSA vs CKA

| | RSA | CKA |
|---|---|---|
| **Range** | [−1, +1] | [0, +1] |
| **Invariant to** | scaling, translation (correlation metric) | orthogonal transforms, isotropic scaling |
| **Based on** | pairwise distance vectors + rank correlation | kernel matrices + normalised inner product |
| **Bias correction** | — | debiased HSIC (handles d >> n) |

## Why PyTorch?

- All operations run on **GPU** when tensors are on GPU
- No `scipy` dependency at runtime
- Integrates naturally into PyTorch model evaluation loops and `torch.no_grad()` blocks
