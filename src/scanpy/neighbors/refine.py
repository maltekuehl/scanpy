"""Refiners for :func:`~scanpy.pp.neighbors`: derive connectivities from kNN distances.

Each takes the kNN distances as a :class:`~scipy.sparse.csr_array` without self-loops,
each row sorted by distance, with any number of neighbors per observation.
"""

from __future__ import annotations

from ._v2._refine import binary, gauss, phenograph, seurat, umap

__all__ = ["binary", "gauss", "phenograph", "seurat", "umap"]
