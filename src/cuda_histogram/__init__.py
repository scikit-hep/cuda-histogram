"""
`cuda-histogram` is a histogram filling package for GPUs.

The package follows `UHI <https://uhi.readthedocs.io>`__ and keeps its API similar to
`boost-histogram <https://github.com/scikit-hep/boost-histogram>`__
`and hist <https://github.com/scikit-hep/hist>`__.
"""

from __future__ import annotations

from cuda_histogram import axis
from cuda_histogram._version import version as __version__
from cuda_histogram.hist import Hist

__all__: list[str] = [
    "Hist",
    "axis",
    "__version__",
]
