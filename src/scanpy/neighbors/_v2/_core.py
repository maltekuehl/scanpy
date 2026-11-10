"""The :attr:`~scanpy.Preset.ScanpyV2Preview` version of :func:`~scanpy.pp.neighbors`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from ... import logging as logg
from ..._compat import fullname
from ..._utils import _doc_params
from ...get._out import _out_dests, _write_dests
from ...get._uns import _use_to_uns
from ...get._use import _get_arr, _resolve_obs
from .._common import NEIGHBORS_KINDS, _metadata, _neighbors_dests
from .._doc import doc_out
from .._types import NeighborsResult
from . import _refine as refiners
from . import _search as searches
from ._graph import _knn_graph

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Literal

    from anndata import AnnData

    from ..._compat import CSRBase, SpBase
    from ...get._types import GraphAcc, RepAcc, _SetRep
    from .._types import (
        KnnTransformerLike,
        NeighborsOut,
        NeighborsParams,
        _Refine,
        _RefineFn,
        _Search,
    )


@_doc_params(out=doc_out)
def neighbors(
    adata: AnnData,
    *,
    use: RepAcc | str | None = None,
    distances: np.ndarray | SpBase | None = None,
    search: KnnTransformerLike | _Search | None = None,
    refine: _Refine | _RefineFn = "umap",
    out: GraphAcc | str | NeighborsOut | bool = False,
) -> NeighborsResult | None:
    r"""Compute a neighborhood graph of observations.

    First, `search` finds each observation’s nearest neighbors in `use`,
    unless their `distances` are given.
    These distances form a kNN graph without self-loops,
    with any number of neighbors per observation.
    Then, `refine` derives connectivities from that graph.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use
        Data to search:
        a :class:`~anndata.acc.LayerAcc` (e.g. `A.X`, `A.layers[...]`) or
        :class:`~anndata.acc.MultiAcc` (e.g. `A.obsm[...]`).
        :class:`str`\ s are :meth:`~anndata.acc.AdAcc.resolve`\ d, e.g. `'obsm.pca'`.
        Required unless `distances` are given.
    distances
        Precomputed distances between observations, an `n_obs` × `n_obs` matrix
        whose stored (if sparse) or off-diagonal (if dense) entries are the neighbors.
    search
        kNN search following the API of :class:`~sklearn.neighbors.KNeighborsTransformer`,
        e.g. from :mod:`scanpy.neighbors.search`, see :doc:`/how-to/knn-transformers`.
        Each stored entry of its result is a neighbor, including zero distances,
        and observations may have any number of neighbors.
        A name means the function in :mod:`scanpy.neighbors.search` with its defaults:
        `'sklearn'`.
        If `None`, `'sklearn'`.
    refine
        Derives connectivities from the kNN distances
        (a :class:`~scipy.sparse.csr_array` without self-loops, each row sorted by distance).
        A callable, or the name of a function in :mod:`scanpy.neighbors.refine`:
        `'umap'`, `'gauss'`, `'phenograph'`, or `'seurat'`.
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
    # check arguments
    if isinstance(search, str) and search not in searches.__all__:
        msg = f"`search` needs to be one of {searches.__all__} or a transformer, got {search!r}."
        raise ValueError(msg)
    refine_fn = _refine_fn(refine)
    dests = _out_dests(
        out, None, dests=_neighbors_dests, kinds=NEIGHBORS_KINDS, uns=True
    )

    start = logg.info("computing neighbors")

    # search, or take the given distances, and normalize them to a kNN graph
    if distances is None:
        search = "sklearn" if search is None else search
        transformer = getattr(searches, search)() if isinstance(search, str) else search

        if use is None:
            msg = "Pass `use`, e.g. `use=A.obsm['X_pca']`, or `distances`."
            raise TypeError(msg)
        use = _resolve_obs(use, rep=True)

        d = _knn_graph(transformer.fit_transform(_get_arr(adata, use)))
    else:
        if use is not None or search is not None:
            msg = "`use` and `search` can’t be passed if `distances` are given."
            raise TypeError(msg)
        if distances.shape != (adata.n_obs, adata.n_obs):
            msg = f"`distances` need to have shape {(adata.n_obs, adata.n_obs)}, got {distances.shape}."
            raise ValueError(msg)
        transformer, d = None, _knn_graph(distances)

    # refine
    vals = _result(refine_fn(d), d)

    params: NeighborsParams = dict(
        n_neighbors=int(np.diff(d.indptr).max(initial=0)),
        **({} if search is None else dict(search=_name(search))),
        refine=_name(refine),
        metric=getattr(transformer, "metric", None),
        **_use_to_uns(use, rep=True),
    )
    return _write(adata, vals, dests, start=start, params=params)


def _refine_fn(refine: _Refine | _RefineFn, /) -> _RefineFn:
    """Get the refiner named `refine`, or `refine` itself."""
    if not isinstance(refine, str):
        return refine
    if refine not in refiners.__all__:
        msg = f"`refine` needs to be one of {refiners.__all__} or a callable, got {refine!r}."
        raise ValueError(msg)
    return getattr(refiners, refine)


def _result(connectivities: SpBase, distances: CSRBase, /) -> NeighborsResult:
    """Get the outputs as float32 CSR arrays, keeping the order of `distances`’ rows."""
    return NeighborsResult(
        sparse.csr_array(connectivities, dtype=np.float32),  # noqa: TID251
        sparse.csr_array(  # noqa: TID251
            (distances.data.astype(np.float32), distances.indices, distances.indptr),
            shape=distances.shape,
        ),
    )


def _write[P: NeighborsParams](
    adata: AnnData,
    vals: NeighborsResult,
    dests: tuple[str | None, NeighborsResult[_SetRep | None]] | Literal[True],
    *,
    start: datetime,
    params: P,
) -> NeighborsResult | None:
    """Return `vals` if `out=True`, else write them and `params` to `adata`."""
    if dests is True:
        logg.info("    finished", time=start)
        return vals
    key, dests = dests
    key, neighbors_dict = _metadata(key, dests, params)
    adata.uns[key] = neighbors_dict
    _write_dests(adata, vals, dests)

    logg.info(
        "    finished",
        time=start,
        deep=(
            f"added to `.uns[{key!r}]`\n"
            f"    `.obsp[{neighbors_dict.get('distances_key')!r}]`, distances for each pair of neighbors\n"
            f"    `.obsp[{neighbors_dict.get('connectivities_key')!r}]`, weighted adjacency matrix"
        ),
    )
    return None


def _name(obj: object) -> str:
    """Get a string to store for `search` or `refine` in `.uns`."""
    if isinstance(obj, str):
        return obj
    return fullname(obj if hasattr(obj, "__qualname__") else type(obj))
