"""
`cuda-histogram` is a histogram filling, transformation, and plotting package for GPUs.

The package follows `UHI <https://uhi.readthedocs.io>`__ and keeps its API similar to
`boost-histogram <https://github.com/scikit-hep/boost-histogram>`__
`and hist <https://github.com/scikit-hep/hist>`__.
"""

from __future__ import annotations

from cuda_histogram.hist_tools import Bin, Cat, Hist, Interval, StringBin

from ._version import version as __version__

__all__: list[str] = [
    "Hist",
    "Bin",
    "Interval",
    "Cat",
    "StringBin",
    "__version__",
]
