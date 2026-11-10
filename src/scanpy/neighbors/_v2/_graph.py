"""The kNN distance graph that searches return and refiners take."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from ..._compat import SpBase

if TYPE_CHECKING:
    from ..._compat import CSRBase


def _knn_graph(d: np.ndarray | SpBase, /) -> CSRBase:
    """Normalize a square distance matrix: CSR, no self-loops, each row sorted by distance.

    Every stored entry of a sparse `d` and every off-diagonal one of a dense `d` is an edge,
    including zero distances between duplicate observations.
    Observations may have any number of neighbors, including none.
    """
    # edges without self-loops
    if isinstance(d, SpBase):
        coo = sparse.coo_array(d)
        keep = coo.row != coo.col
        row, col, dist = coo.row[keep], coo.col[keep], coo.data[keep]
    else:
        row, col = np.nonzero(~np.eye(*d.shape, dtype=bool))
        dist = np.asarray(d)[row, col]

    # sort each row by distance, unless already sorted
    step = np.diff(row)
    if not np.all((step > 0) | ((step == 0) & (np.diff(dist) >= 0))):
        order = np.lexsort((dist, row))
        row, col, dist = row[order], col[order], dist[order]

    indptr = np.r_[0, np.cumsum(np.bincount(row, minlength=d.shape[0]))]
    idx_dtype = np.int32 if len(col) <= np.iinfo(np.int32).max else np.int64
    return sparse.csr_array(  # noqa: TID251
        (dist, col.astype(idx_dtype), indptr.astype(idx_dtype)), shape=d.shape
    )


def _nearest(g: CSRBase, /, n_neighbors: int) -> CSRBase:
    """Keep each observation’s `n_neighbors` nearest neighbors in a graph from :func:`_knn_graph`."""
    counts = np.diff(g.indptr)
    rank = np.arange(g.nnz) - np.repeat(g.indptr[:-1], counts)
    keep = rank < n_neighbors
    indptr = np.r_[0, np.cumsum(np.minimum(counts, n_neighbors))]
    return sparse.csr_array(  # noqa: TID251
        (g.data[keep], g.indices[keep], indptr.astype(g.indptr.dtype)), shape=g.shape
    )
