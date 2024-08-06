from __future__ import annotations

import numbers
from typing import Any

import awkward
import boost_histogram.axis as bha
import cupy
import hist
import numpy as np

from cuda_histogram.axis import (
    _overflow_behavior,
)

__all__: list[str] = ["Hist"]


class Hist(hist.Hist):
    """
    Construct a new histogram.

    If you pass in a single argument, this will be treated as a
    histogram and this will convert the histogram to this type of
    histogram.

    Parameters
    ----------
    *args : Axis
        Provide 1 or more axis instances.
    storage : Storage = bh.storage.Double()  -- not implemented
        Select a storage to use in the histogram
    metadata : Any = None
        Data that is passed along if a new histogram is created
    data: np.typing.NDArray[Any] = None  -- not implemented
        Data to be filled in the histogram
    label: str = None
        Histogram's label
    name: str = None
        Histogram's name
    """

    def __init__(
        self,
        *axes,
        storage=None,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            *axes, storage=storage, metadata=metadata, data=data, label=label, name=name
        )
        self._dense_shape = tuple(
            [ax.size for ax in self.axes if isinstance(ax, (bha.Regular, bha.Variable))]
        )
        self._sumw = cupy.zeros(shape=self._dense_shape)
        # Storage of sumw2 starts at first use of weight keyword in fill()
        self._sumw2 = None
        self._overflows = cupy.array(0)
        self._underflows = cupy.array(0)

    def dense_dim(self):
        """Dense dimension of this histogram (number of non-sparse axes)"""
        return len(self._dense_shape)

    def _init_sumw2(self):
        self._sumw2 = cupy.zeros(shape=self._dense_shape)

    def __getitem__(self, keys):
        if isinstance(keys, int):
            return self._sumw[keys]
        else:
            raise ValueError("Convert the histogram to CPU to access full UHI")

    def fill(self, *args, weight=None):
        """
        Insert data into the histogram.

        Parameters
        ----------
        *args : Union[Array[float], Array[int], Array[str], float, int, str]  -- only arrays of numbers supported
            Provide one value or array per dimension
        weight : List[Union[Array[float], Array[int], float, int, str]]]  -- not supported
            Provide weights (only if the histogram storage supports it)
        """
        if isinstance(weight, (awkward.Array, cupy.ndarray, np.ndarray)):
            weight = cupy.array(weight)
        if isinstance(weight, numbers.Number):
            weight = cupy.atleast_1d(weight)
        if not isinstance(args, str) and len(args) != len(self.axes):
            raise TypeError(
                f"Cannot fill {len(self.axis)} axes with {len(args)} arrays."
            )

        if weight is not None and self._sumw2 is None:
            self._init_sumw2()

        if self.dense_dim() > 0:
            self._overflows = tuple(
                sum(value > d.edges[-1])
                for d, value in zip(self.axes, args)
                if isinstance(d, (bha.Regular, bha.Variable))
            )
            self._underflows = tuple(
                sum(value < d.edges[0])
                for d, value in zip(self.axes, args)
                if isinstance(d, (bha.Regular, bha.Variable))
            )
            dense_indices = tuple(
                cupy.asarray(d.index(value))[
                    (value > d.edges[0]) & (value < d.edges[-1])
                ]
                for d, value in zip(self.axes, args)
                if isinstance(d, (bha.Regular, bha.Variable))
            )
            xy = cupy.atleast_1d(
                cupy.ravel_multi_index(dense_indices, self._dense_shape)
            )
            if weight is not None:
                self._sumw[:] += cupy.bincount(
                    xy, weights=weight, minlength=np.array(self._dense_shape).prod()
                ).reshape(self._dense_shape)
                self._sumw2[:] += cupy.bincount(
                    xy,
                    weights=weight**2,
                    minlength=np.array(self._dense_shape).prod(),
                ).reshape(self._dense_shape)
            else:
                self._sumw[:] += cupy.bincount(
                    xy, weights=None, minlength=np.array(self._dense_shape).prod()
                ).reshape(self._dense_shape)
                if self._sumw2 is not None:
                    self._sumw2[:] += cupy.bincount(
                        xy,
                        weights=None,
                        minlength=np.array(self._dense_shape).prod(),
                    ).reshape(self._dense_shape)
        elif weight is not None:
            self._sumw += cupy.sum(weight)
            self._sumw2 += cupy.sum(weight**2)
        else:
            self._sumw += 1.0
            if self._sumw2 is not None:
                self._sumw2 += 1.0

    def values(self, flow: bool = False):
        """
        Return the values in histogram.

        Parameters
        ----------
            flow : bool
                Include flow bins.
        """

        def view_dim(arr):
            if self.dense_dim() == 0:
                return arr
            else:
                return arr[
                    tuple(_overflow_behavior(flow) for _ in range(self.dense_dim()))
                ]

        return view_dim(
            cupy.append(cupy.append(self._underflows, self._sumw), self._overflows)
        )

    def to_cpu(self):
        return hist.Hist(self, data=self._sumw.get())
