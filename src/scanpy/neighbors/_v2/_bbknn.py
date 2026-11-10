"""The :attr:`~scanpy.Preset.ScanpyV2Preview` version of :func:`~scanpy.pp.bbknn`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from ... import logging as logg
from ..._utils import _doc_params
from ...get._out import _out_dests
from ...get._uns import _ref_to_uns, _use_to_uns
from ...get._use import _get_arr, _resolve_obs
from .._bbknn import _trim
from .._common import NEIGHBORS_KINDS, _neighbors_dests
from .._doc import doc_out
from . import _search as searches
from ._core import _name, _refine_fn, _result, _write
from ._graph import _knn_graph, _nearest

if TYPE_CHECKING:
    from anndata import AnnData
    from anndata.acc import AdRef

    from ...get._types import GraphAcc, RepAcc
    from .._types import (
        BbknnParams,
        KnnTransformerLike,
        NeighborsOut,
        NeighborsResult,
        _Refine,
        _RefineFn,
    )


@_doc_params(out=doc_out)
def bbknn(
    adata: AnnData,
    *,
    use: RepAcc | str,
    batches: AdRef | str,
    neighbors_within_batch: int = 3,
    search: KnnTransformerLike | None = None,
    refine: _Refine | _RefineFn = "umap",
    trim: int | None = None,
    out: GraphAcc | str | NeighborsOut | bool = False,
) -> NeighborsResult | None:
    r"""Compute a batch balanced neighborhood graph of observations :cite:p:`Polanski2019`.

    First, `search` finds each observation’s `neighbors_within_batch` nearest neighbors
    in each batch of `use`.
    Together, their distances form a kNN graph without self-loops.
    Then, `refine` derives connectivities from that graph,
    and `trim` keeps each observation’s strongest connectivities.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use
        Data to search:
        a :class:`~anndata.acc.LayerAcc` (e.g. `A.X`, `A.layers[...]`) or
        :class:`~anndata.acc.MultiAcc` (e.g. `A.obsm[...]`).
        :class:`str`\ s are :meth:`~anndata.acc.AdAcc.resolve`\ d, e.g. `'obsm.pca'`.
    batches
        The batch of each observation, e.g. `A.obs['batch']`.
        :class:`str`\ s are :meth:`~anndata.acc.AdAcc.resolve`\ d, e.g. `'obs.batch'`.
    neighbors_within_batch
        Number of neighbors of each observation in each batch,
        not counting the observation itself.
    search
        kNN search following the API of :class:`~sklearn.neighbors.KNeighborsTransformer`,
        e.g. from :mod:`scanpy.neighbors.search`.
        A copy made by :func:`sklearn.base.clone` is fitted on each batch
        and queried with all observations.
        It needs to find at least `neighbors_within_batch` neighbors.
        If `None`, :func:`scanpy.neighbors.search.sklearn` with `neighbors_within_batch`.
    refine
        Derives connectivities from the kNN distances
        (a :class:`~scipy.sparse.csr_array` without self-loops, each row sorted by distance).
        A callable, or the name of a function in :mod:`scanpy.neighbors.refine`:
        `'umap'`, `'gauss'`, `'phenograph'`, or `'seurat'`.
    trim
        Keep each observation’s this many strongest connectivities:
        an edge is removed if it is weaker than either observation’s `trim`-th strongest one.
        If `None`, 10 times the number of neighbors of each observation.
        If 0, keep all.
    {out}

    Returns
    -------
    Returns the outputs if `out=True`, else `None`.
    By default sets the following fields (see `out` for others):

    `adata.obsp['connectivities']` : :class:`scipy.sparse.csr_array` (dtype `float32`)
        Weighted adjacency matrix of the neighborhood graph.
    `adata.obsp['distances']` : :class:`scipy.sparse.csr_array` (dtype `float32`)
        Distances to each observation’s neighbors, each row sorted by distance.
    `adata.uns['neighbors']` : :class:`dict`
        neighbors parameters, and the `.obsp` keys the matrices were written to.

    """
    from anndata.acc import A, AdRef
    from sklearn.base import clone

    # check arguments
    refine_fn = _refine_fn(refine)
    if neighbors_within_batch < 1:
        msg = "`neighbors_within_batch` needs to be at least 1."
        raise ValueError(msg)
    dests = _out_dests(
        out, None, dests=_neighbors_dests, kinds=NEIGHBORS_KINDS, uns=True
    )
    use = _resolve_obs(use, rep=True)
    if not isinstance(batches, AdRef):
        batches = A.resolve(batches, vec=True)
    labels = np.asarray(adata[batches])
    transformer = searches.sklearn(neighbors_within_batch) if search is None else search

    start = logg.info("computing batch balanced neighbors")

    # search each batch, keeping each observation’s nearest neighbors in it
    x = _get_arr(adata, use)
    n_obs = adata.n_obs
    edges = []
    for batch in np.unique(labels):
        idx = np.flatnonzero(labels == batch)
        found = sparse.coo_array(clone(transformer).fit(x[idx]).transform(x))
        found = sparse.coo_array(
            (found.data, (found.row, idx[found.col])), shape=(n_obs, n_obs)
        )
        edges.append(_nearest(_knn_graph(found), neighbors_within_batch).tocoo())
        logg.debug(f"    computed neighbors within batch {batch!r}")

    # merge the batches into one kNN graph
    d = _knn_graph(
        sparse.coo_array(
            (
                np.concatenate([e.data for e in edges]),
                (
                    np.concatenate([e.row for e in edges]),
                    np.concatenate([e.col for e in edges]),
                ),
            ),
            shape=(n_obs, n_obs),
        )
    )

    # refine, and trim
    connectivities = sparse.csr_array(refine_fn(d))  # noqa: TID251
    n_neighbors = int(np.diff(d.indptr).max(initial=0))
    trim = 10 * n_neighbors if trim is None else trim
    if trim > 0:
        connectivities = _trim(connectivities, trim)
    vals = _result(connectivities, d)

    params: BbknnParams = dict(
        n_neighbors=n_neighbors,
        search=_name("sklearn" if search is None else search),
        refine=_name(refine),
        metric=getattr(transformer, "metric", None),
        **_use_to_uns(use, rep=True),
        batches=_ref_to_uns(batches, labels),
        neighbors_within_batch=neighbors_within_batch,
        trim=trim,
    )
    return _write(adata, vals, dests, start=start, params=params)
