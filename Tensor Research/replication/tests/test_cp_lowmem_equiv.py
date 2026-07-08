"""Equivalence check: LowMemCPRegressor vs stock tensorly CPRegressor.

Same shapes family as the refit (X: samples x firms x feats x L,
y: samples x firms x feats), same seed. Passes iff fitted weight tensors
and test predictions agree to tight float tolerance across block sizes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorly as tl
from tensorly.regression.cp_regression import CPRegressor

from src.tensors.cp_lowmem import LowMemCPRegressor


def main() -> int:
    tl.set_backend("numpy")
    rng = np.random.default_rng(7)
    n, firms, feats, L, r = 23, 12, 9, 3, 5

    X = rng.normal(size=(n, firms, feats, L)).astype(np.float32)
    y = rng.normal(size=(n, firms, feats)).astype(np.float32)
    X_new = rng.normal(size=(6, firms, feats, L)).astype(np.float32)

    kw = dict(weight_rank=r, reg_W=7.3, n_iter_max=40,
              random_state=42, verbose=0)
    ref = CPRegressor(**kw).fit(X, y)

    ok = True
    for bs in (1, 4, 100):
        lm = LowMemCPRegressor(block_size=bs, **kw).fit(X, y)
        dw = float(np.max(np.abs(lm.weight_tensor_ - ref.weight_tensor_))
                   / (np.max(np.abs(ref.weight_tensor_)) + 1e-30))
        dp = float(np.max(np.abs(lm.predict(X_new) - ref.predict(X_new)))
                   / (np.max(np.abs(ref.predict(X_new))) + 1e-30))
        same_iters = lm.n_iterations_ == ref.n_iterations_
        print(f"block_size={bs:>3}: rel dW={dw:.3e} rel dPred={dp:.3e} "
              f"iters {lm.n_iterations_} vs {ref.n_iterations_}")
        ok &= dw < 1e-8 and dp < 1e-8 and same_iters
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
