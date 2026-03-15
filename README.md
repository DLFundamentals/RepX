# LeTorch

PyTorch-native **Representation Similarity Analysis (RSA)** with GPU support.

RSA measures how similarly two sets of neural representations organize the same stimuli, by correlating their pairwise dissimilarity structures.

## Installation

```bash
pip install letorch
```

## Quick Start

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

## API

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

Returns a scalar tensor. Call `.item()` for a Python float.

## Why PyTorch?

- All operations — RDM construction, ranking, correlation — run on **GPU** when tensors are on GPU
- No `scipy` dependency at runtime
- Integrates naturally into PyTorch model evaluation loops and `torch.no_grad()` blocks
