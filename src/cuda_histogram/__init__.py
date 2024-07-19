"""
`cuda-histogram` is a histogram filling, transformation, and plotting package for GPUs.

The package follows `UHI <https://uhi.readthedocs.io>`__ and keeps its API similar to
`boost-histogram <https://github.com/scikit-hep/boost-histogram>`__
`and hist <https://github.com/scikit-hep/hist>`__.
"""

from __future__ import annotations

from cuda_histogram.hist_tools import Bin, Cat, Hist, Interval, StringBin
from cuda_histogram.plot import (
    clopper_pearson_interval,
    normal_interval,
    plot1d,
    plot2d,
    plotgrid,
    plotratio,
    poisson_interval,
)

from ._version import version as __version__

__all__ = [
    "Hist",
    "Bin",
    "Interval",
    "Cat",
    "StringBin",
    "poisson_interval",
    "clopper_pearson_interval",
    "normal_interval",
    "plot1d",
    "plotratio",
    "plot2d",
    "plotgrid",
    "__version__",
]
