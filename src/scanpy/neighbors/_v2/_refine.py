"""Refiners, see :mod:`scanpy.neighbors.refine`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from fast_array_utils.numba import njit
from scipy import sparse

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..._compat import CSRBase

__all__ = ["binary", "gauss", "phenograph", "seurat", "umap"]


def umap(d: CSRBase, /) -> CSRBase:
    """Fuzzy simplicial set :cite:p:`McInnes2018`.

    For an observation with k neighbors, this is `umap.umap_.fuzzy_simplicial_set`
    with `n_neighbors=k + 1` and `local_connectivity=1`.
    """
    dist = d.data.astype(np.float64)
    rho, sigma = _rho_sigma(d.indptr, dist, dist.sum() / (d.nnz + d.shape[0]))

    # weights of each observation’s edges
    row = d.tocoo().row
    w = sparse.csr_array(  # noqa: TID251
        (np.exp(-np.maximum(dist - rho[row], 0) / sigma[row]), d.indices, d.indptr),
        shape=d.shape,
    )

    # fuzzy union of both directions
    return w + w.T - w.multiply(w.T)


@njit
def _rho_sigma(
    indptr: NDArray[np.integer], dist: NDArray[np.float64], mean_dist: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get each row’s distance to its nearest neighbor that isn’t a duplicate, and its width.

    The width sigma solves `sum(exp(-(dist - rho) / sigma)) = log2(k + 1)`
    and is at least `1e-3` times the mean distance, counting the observation itself.
    """
    n = len(indptr) - 1
    rho, sigma = np.zeros(n), np.ones(n)
    for i in numba.prange(n):
        dist_i = dist[indptr[i] : indptr[i + 1]]

        # rho: distance to the nearest neighbor that isn’t a duplicate
        positive = dist_i[dist_i > 0]
        if len(positive):
            rho[i] = positive.min()

        # sigma: bisection
        excess = np.maximum(dist_i - rho[i], 0.0)
        target = np.log2(len(dist_i) + 1)
        lo, hi, mid = 0.0, np.inf, 1.0
        for _ in range(64):
            psum = np.exp(-excess / mid).sum()
            if abs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2
            else:
                lo = mid
                mid = mid * 2 if hi == np.inf else (lo + hi) / 2

        # lower bound for sigma
        mean = dist_i.sum() / (len(dist_i) + 1) if rho[i] > 0 else mean_dist
        sigma[i] = max(mid, 1e-3 * mean)
    return rho, sigma


def gauss(d: CSRBase, /) -> CSRBase:
    """Gaussian kernel with adaptive width :cite:p:`Coifman2005,Haghverdi2016`.

    Its width for observation i is the median distance to i’s neighbors.
    The weight is symmetric in i and j; an edge in either direction is kept.
    """
    row, k, start = d.tocoo().row, np.diff(d.indptr), d.indptr[:-1]
    sq = d.data.astype(np.float64) ** 2

    # width: median distance, in the middle of each sorted row
    has = k > 0
    sigma_sq = np.zeros(d.shape[0])
    sigma_sq[has] = (sq[(start + (k - 1) // 2)[has]] + sq[(start + k // 2)[has]]) / 2
    sigma = np.sqrt(sigma_sq)

    # weights, symmetric in i and j
    den = sigma_sq[row] + sigma_sq[d.indices]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.sqrt(2 * sigma[row] * sigma[d.indices] / den) * np.exp(-sq / den)
    w[den == 0] = sq[den == 0] == 0  # both widths are 0: only duplicates are connected

    # keep an edge in either direction
    w = sparse.csr_array((w, d.indices, d.indptr), shape=d.shape)  # noqa: TID251
    return w.maximum(w.T)


def phenograph(d: CSRBase, /) -> CSRBase:
    """Jaccard index of the neighbor sets along each edge, as in PhenoGraph :cite:p:`Levine2015`.

    The observation itself is not part of its neighbor set.
    Both directions of an edge are averaged.
    """
    a = _pattern(d)
    row, size = a.tocoo().row, np.diff(a.indptr)

    # Jaccard index along each edge
    shared = a[row].multiply(a[a.indices]).sum(axis=1)
    w = a.copy()
    w.data = shared / (size[row] + size[a.indices] - shared)
    w.eliminate_zeros()

    # average both directions
    return (w + w.T) / 2


def seurat(d: CSRBase, /, *, prune: float = 1 / 15) -> CSRBase:
    """Shared nearest neighbor graph as in Seurat’s `FindNeighbors` :cite:p:`Stuart2019`.

    Each observation’s neighbor set consists of its neighbors and itself.
    Every pair of observations sharing a neighbor is weighted by the Jaccard index
    of their neighbor sets.

    Parameters
    ----------
    d
        kNN distances.
    prune
        Remove weights below this value.
    """
    # neighbor sets including the observation itself
    a = _pattern(d) + sparse.eye_array(d.shape[0], dtype=np.float32, format="csr")
    size = np.diff(a.indptr)

    # Jaccard index for every pair sharing a neighbor
    w = sparse.csr_array(a @ a.T)  # noqa: TID251
    row = w.tocoo().row
    w.data /= size[row] + size[w.indices] - w.data

    # prune, and remove self-loops
    w.data[(w.data < prune) | (row == w.indices)] = 0
    w.eliminate_zeros()
    return w


def binary(d: CSRBase, /) -> CSRBase:
    """Unweighted kNN graph.

    Each edge has weight 1, and an edge in either direction is kept.
    """
    a = _pattern(d)
    return a.maximum(a.T)


def _pattern(d: CSRBase, /) -> CSRBase:
    """Get the unweighted kNN graph: 1 for each neighbor."""
    return sparse.csr_array(  # noqa: TID251
        (np.ones(d.nnz, np.float32), d.indices, d.indptr), shape=d.shape
    )
