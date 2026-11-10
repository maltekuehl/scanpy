"""Searches, see :mod:`scanpy.neighbors.search`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.version_info < (3, 15):
    from types import MappingProxyType as frozendict  # noqa: N813

from ..._settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from numpy.typing import NDArray
    from sklearn.neighbors import KNeighborsTransformer

__all__ = ["sklearn"]


def sklearn(
    n_neighbors: int = 15,
    *,
    metric: str | Callable[[NDArray, NDArray], float] = "euclidean",
    metric_kwds: Mapping[str, Any] = frozendict({}),
) -> KNeighborsTransformer:
    """Exact kNN search by brute force with :class:`~sklearn.neighbors.KNeighborsTransformer`.

    It uses :attr:`scanpy.settings.n_jobs` jobs.

    Parameters
    ----------
    n_neighbors
        Number of neighbors of each observation, not counting the observation itself.
    metric
        A metric :class:`~sklearn.neighbors.KNeighborsTransformer` accepts:
        a name from :func:`~sklearn.metrics.pairwise_distances`
        or a callable that returns the distance between two observations.
    metric_kwds
        Options for the metric.
    """
    from sklearn.neighbors import KNeighborsTransformer

    return KNeighborsTransformer(
        n_neighbors=n_neighbors,
        algorithm="brute",
        metric=metric,
        metric_params=dict(metric_kwds),
        n_jobs=settings.n_jobs,
    )
