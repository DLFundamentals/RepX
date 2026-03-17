"""biased_vs_debiased_demo.py

Demonstrates biased vs. debiased linear-kernel CKA on:
  1. X compared with itself            → similarity ≈ 1.0 for both
  2. X compared with X_orth            → similarity differs:
       • biased CKA   ≈ 0  (orthogonal representations)
       • debiased CKA can be negative, but averages to 0 for independent
         representations (it is an unbiased estimator)

Key insight
-----------
*Biased CKA* has a systematic positive bias at finite sample sizes.
For independent representations its expected value is > 0, growing with
the ratio d/n.

*Debiased CKA* removes this bias: its expected value for independent
representations is exactly 0, but a single estimate can be negative.

Run with:
    python biased_vs_debiased_demo.py
"""

import torch
from letorch.cka import CKA


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Build X and an orthogonal counterpart
# ---------------------------------------------------------------------------

n, d = 64, 32   # 64 samples, 32-dimensional representations

X = torch.randn(n, d)

# Construct a basis for R^n via QR decomposition of a random n×n matrix
# whose first d columns are set to X.  After QR:
#   Q[:, :d]  spans the same column space as X
#   Q[:, d:]  is the orthogonal complement of X's column space
M = torch.randn(n, n)
M[:, :d] = X
Q, _ = torch.linalg.qr(M)          # Q is n×n with orthonormal columns

X_orth = Q[:, d:d + d].clone()     # d columns orthogonal to X's column space

# Sanity-check: columns of X and X_orth are orthogonal
overlap = (X.T @ X_orth).norm().item()
print(f"‖Xᵀ X_orth‖_F  (column-space orthogonality check, should be ≈ 0): {overlap:.6f}")
print()


# ---------------------------------------------------------------------------
# Run both CKA types on X vs X  and  X vs X_orth
# ---------------------------------------------------------------------------

biased_cka   = CKA(debiased=False)
debiased_cka = CKA(debiased=True)

results = {
    ("Biased CKA",   "X  vs  X"):      biased_cka.cka(X, X).item(),
    ("Biased CKA",   "X  vs  X_orth"): biased_cka.cka(X, X_orth).item(),
    ("Debiased CKA", "X  vs  X"):      debiased_cka.cka(X, X).item(),
    ("Debiased CKA", "X  vs  X_orth"): debiased_cka.cka(X, X_orth).item(),
}

col_w = max(len(k[0]) for k in results) + 2
header = f"{'CKA type':<{col_w}} {'Comparison':<20} {'Score':>10}"
print(header)
print("-" * len(header))
for (cka_type, comparison), score in results.items():
    print(f"{cka_type:<{col_w}} {comparison:<20} {score:>10.6f}")

print()
print("Notes:")
print("  • X vs X:      both estimators → ≈ 1.0  (identical representations)")
print("  • X vs X_orth: biased ≈ 0      (orthogonal representations, but still")
print("                                   has positive finite-sample bias)")
print("  • X vs X_orth: debiased can be negative (unbiased estimator;")
print("                 expected value = 0 for independent representations)")
