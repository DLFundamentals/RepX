# RepX

## 1. Overview

RepX is built to evaluate representation quality directly from embeddings.
If you can produce a representation tensor, you can analyze it with RepX.

This works across learning paradigms:

- **Supervised learning** (classification, regression backbones)
- **Self-supervised learning** (contrastive, masked modeling, distillation)
- **Unsupervised learning** (autoencoders, clustering-based pipelines)

The only requirement is a representation, e.g. `z = model(x)`.

## 2. Installation

```bash
pip install repx
```

## 3. Examples

Start from the same pattern in every setup:

```python
# model can be any feature extractor: CNN, ViT, MLP, SSL encoder, etc.
z = model(x)
```

### Transfer (downstream performance)

```python
import torch
from repx.transfer import LinearProbeEvaluator, NCCCEvaluator

# model can be any model that outputs representations
z_train = model(x_train)
z_test = model(x_test)

# Linear probe: train a linear classifier on frozen features
lp = LinearProbeEvaluator(device="cpu")
train_acc, test_acc = lp.evaluate(
	train_features=z_train,
	train_labels=y_train,
	test_features=z_test,
	test_labels=y_test,
	num_output_classes=num_classes,
	n_shots=None,  # set to int for few-shot per class
)

# NCCC: nearest class center evaluation
nccc = NCCCEvaluator(device="cpu")
accs = nccc.evaluate(
	features=z_test,
	labels=y_test,
	n_shots=5,
	repeat=3,
)
```

### Geometry (collapse analysis)

```python
from repx.collapse import compute_cdnv, compute_directional_cdnv

# z can come from any model
z = model(x)

cdnv = compute_cdnv(z, y, num_classes=num_classes)
dir_cdnv = compute_directional_cdnv(z, y, num_classes=num_classes)
```

### Alignment (representation similarity)

```python
from repx import RSA, CKA

z_a = model_a(x)
z_b = model_b(x)

rsa = RSA(rdm_metric="correlation", compare="spearman")
cka = CKA(kernel="linear")

print(rsa.rsa(z_a, z_b).item())
print(cka.cka(z_a, z_b).item())
```

For full API docs, see [Alignment](api/alignment.md), [Transfer](api/transfer.md), [Core](api/core.md), and [Collapse](api/collapse.md).

## 4. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and development workflow.

## 5. License

RepX is released under the GNU GPLv3. See [LICENSE.md](LICENSE.md) for details.