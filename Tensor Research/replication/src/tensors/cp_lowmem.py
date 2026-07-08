"""Memory-lean, structure-exploiting drop-in for tensorly's CPRegressor.

(2026-07-06, 499-firm scale-up.) Stock CPRegressor.fit materializes, per
factor update, (a) the full ridge design matrix phi (rows = n_samples x
prod(y dims)) — ~65 GB for the firm factor at 498 firms x rank 13 — and
(b) the joint Khatri-Rao product over all other factors — ~41 GB for the
lookback-factor update. Both OOM the 62 GB lab hosts.

Neither object is needed. The design matrix factorizes row-wise as
phi[(s,b),(d,c)] = Z[s,d,c] * K[b,c], where Z = X_unfolded @ KR(X-side
factors) is tiny and K = KR(y-side factors). Therefore the normal equations
collapse exactly:

    phi'phi[(d,c),(d',c')] = (sum_s Z[s,d,c] Z[s,d',c']) * (K'K)[c,c']
    phi'y[(d,c)]           = sum_s Z[s,d,c] * (y_s @ K)[c]

i.e. a Hadamard product of small Grams (the classic CP-ALS identity) — the
same numbers as the stock implementation up to floating-point summation
order, at a tiny fraction of the memory AND flops (the prod(y dims) factor,
~2e4 at 498 firms, drops out of the Gram cost entirely).

Line-for-line parity with tensorly.regression.cp_regression.CPRegressor.fit
otherwise: same random init order, same per-factor ridge solve with
reg_W * I, same convergence rule on ||W||. predict() inherited unchanged.
Equivalence vs stock is checked by test_cp_lowmem_equiv.py.

block_size is accepted for backward compatibility and ignored (no large
per-sample intermediates remain).
"""
from __future__ import annotations

import numpy as np
import tensorly as tl
from tensorly import backend as T
from tensorly.base import partial_tensor_to_vec, partial_unfold
from tensorly.cp_tensor import cp_to_tensor, cp_to_vec
from tensorly.regression.cp_regression import CPRegressor
from tensorly.tenalg import khatri_rao


def _kr(mats: list, r: int) -> np.ndarray:
    """Khatri-Rao over a possibly-short list (row order = list order)."""
    if not mats:
        return np.ones((1, r))
    if len(mats) == 1:
        return np.asarray(mats[0])
    return khatri_rao(mats)


class LowMemCPRegressor(CPRegressor):
    def __init__(self, weight_rank, tol=10e-7, reg_W=1, n_iter_max=100,
                 random_state=None, verbose=1, block_size=4):
        super().__init__(weight_rank=weight_rank, tol=tol, reg_W=reg_W,
                         n_iter_max=n_iter_max, random_state=random_state,
                         verbose=verbose)
        self.block_size = int(block_size)  # kept for API compat; unused

    def fit(self, X, y):
        rng = T.check_random_state(self.random_state)
        r = self.weight_rank
        ndim_x = T.ndim(X)

        # identical init order to the parent implementation
        W = []
        for i in range(1, ndim_x):
            W.append(T.tensor(rng.randn(X.shape[i], r), **T.context(X)))
        for i in range(1, T.ndim(y)):
            W.append(T.tensor(rng.randn(y.shape[i], r), **T.context(X)))

        norm_W = []
        weights = T.ones(r, **T.context(X))
        n = X.shape[0]
        n_x_factors = ndim_x - 1
        y_flat = np.reshape(np.asarray(y, dtype=np.float64), (n, -1))

        for iteration in range(self.n_iter_max):
            for i in range(len(W)):
                if i < n_x_factors:
                    # --- X-side factor update -------------------------------
                    # phi[(s,b),(d,c)] = Z[s,d,c] * K[b,c]
                    D = X.shape[i + 1]
                    KR_x = _kr([W[j] for j in range(n_x_factors)
                                if j != i], r)                    # (rest_X, r)
                    K = _kr(W[n_x_factors:], r)                   # (y_prod, r)
                    Z = np.dot(partial_unfold(X, i, skip_begin=1),
                               KR_x)                              # (n, D, r)
                    Zmat = np.reshape(Z, (n, D * r))
                    A = Zmat.T @ Zmat                             # (D*r, D*r)
                    H = K.T @ K                                   # (r, r)
                    G = np.reshape(
                        np.reshape(A, (D, r, D, r))
                        * H[None, :, None, :],
                        (D * r, D * r))
                    q = y_flat @ K                                # (n, r)
                    c = np.reshape(np.einsum("sdc,sc->dc", Z, q), (D * r,))
                    G[np.diag_indices_from(G)] += self.reg_W
                    W[i] = np.reshape(np.linalg.solve(G, c), (-1, r))
                else:
                    # --- y-side factor update -------------------------------
                    # phi[(s,b),c] = Z[s,c] * K[b,c]; solve r x r for all D
                    # output slices at once (identical to stock).
                    ax = i - ndim_x + 2
                    D = y.shape[ax]
                    KR_x = _kr(W[:n_x_factors], r)                # (vecX, r)
                    K = _kr([W[j] for j in range(n_x_factors, len(W))
                             if j != i], r)                       # (other_y, r)
                    Z = np.dot(partial_tensor_to_vec(X, skip_begin=1),
                               KR_x)                              # (n, r)
                    G = (Z.T @ Z) * (K.T @ K)
                    Ymv = np.moveaxis(np.asarray(y, dtype=np.float64),
                                      ax, -1)
                    Ymv = np.reshape(Ymv, (n, -1, D))             # (n,other,D)
                    Q = np.einsum("bc,sbd->scd", K, Ymv)          # (n, r, D)
                    C = np.einsum("sc,scd->cd", Z, Q)             # (r, D)
                    G[np.diag_indices_from(G)] += self.reg_W
                    W[i] = np.transpose(np.linalg.solve(G, C))

            weight_tensor_ = cp_to_tensor((weights, W))
            norm_W.append(T.norm(weight_tensor_, 2))

            if iteration > 1:
                weight_evolution = tl.abs(norm_W[-1] - norm_W[-2]) / norm_W[-1]
                if weight_evolution <= self.tol:
                    if self.verbose:
                        print(f"\nConverged in {iteration} iterations")
                    break

        self.weight_tensor_ = weight_tensor_
        self.cp_weight_ = (weights, W)
        self.vec_W_ = cp_to_vec((weights, W))
        self.n_iterations_ = iteration + 1
        self.norm_W_ = norm_W
        return self
