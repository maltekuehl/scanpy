"""Preprocessing functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..neighbors._bbknn import bbknn
from ..neighbors._types import NeighborsOut, NeighborsResult
from ._combat import combat
from ._deprecated.sampling import subsample
from ._harmony import harmony_integrate
from ._hashsolo import HashsoloOut, HashsoloResult, hashsolo
from ._highly_variable_genes import highly_variable_genes
from ._normalization import (
    NormalizeTotalOut,
    NormalizeTotalResult,
    normalize_shifted_clr,
    normalize_total,
)
from ._pca import PcaOut, PcaResult, pca
from ._qc import QcMetricsOut, QcMetricsResult, calculate_qc_metrics
from ._recipes import recipe_seurat, recipe_weinreb17, recipe_zheng17
from ._scale import ScaleOut, ScaleResult, scale
from ._scrublet import ScrubletOut, ScrubletResult, scrublet, scrublet_simulate_doublets
from ._simple import (
    downsample_counts,
    filter_cells,
    filter_genes,
    log1p,
    regress_out,
    sample,
    sqrt,
)

if TYPE_CHECKING:
    from typing import Any

    from ..neighbors import neighbors


def __getattr__(name: str) -> Any:
    if name == "neighbors":
        from .._settings import Preset, settings

        if settings.preset is Preset.ScanpyV2Preview:
            from ..neighbors._v2 import neighbors
        else:
            from ..neighbors import neighbors
        return neighbors
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "HashsoloOut",
    "HashsoloResult",
    "NeighborsOut",
    "NeighborsResult",
    "NormalizeTotalOut",
    "NormalizeTotalResult",
    "PcaOut",
    "PcaResult",
    "QcMetricsOut",
    "QcMetricsResult",
    "ScaleOut",
    "ScaleResult",
    "ScrubletOut",
    "ScrubletResult",
    "bbknn",
    "calculate_qc_metrics",
    "combat",
    "downsample_counts",
    "filter_cells",
    "filter_genes",
    "harmony_integrate",
    "hashsolo",
    "highly_variable_genes",
    "log1p",
    "neighbors",
    "normalize_shifted_clr",
    "normalize_total",
    "pca",
    "recipe_seurat",
    "recipe_weinreb17",
    "recipe_zheng17",
    "regress_out",
    "sample",
    "scale",
    "scrublet",
    "scrublet_simulate_doublets",
    "sqrt",
    "subsample",
]
