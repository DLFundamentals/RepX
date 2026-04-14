# Alignment

Alignment-based representation metrics.

Alignment metrics compare representation spaces from one or more models.

- **RSA** measures whether pairwise similarity structure between examples is
  preserved across representations.
- **CKA** measures global alignment between representation spaces in a way that
  is robust to common linear transformations.

Use these metrics when you want to compare learned representations from
different models, layers, checkpoints, or training regimes.

See:

- [RSA](alignment-rsa.md)
- [CKA](alignment-cka.md)
