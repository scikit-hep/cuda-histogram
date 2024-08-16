from __future__ import annotations

from collections import namedtuple

import cupy
import numpy as np

from cuda_histogram.axis import (
    Axis,
    # Cat,
    DenseAxis,
    Regular,
    SparseAxis,
    Variable,
    _overflow_behavior,
)

__all__: list[str] = ["Hist"]

_MaybeSumSlice = namedtuple("_MaybeSumSlice", ["start", "stop", "sum"])


def _assemble_blocks(array, ndslice, depth=0):
    """
    Turns an n-dimensional slice of array (tuple of slices)
    into a nested list of numpy arrays that can be passed to np.block()

    Under the assumption that index 0 of any dimension is underflow, -2 overflow, -1 nanflow,
    this function will add the range not in the slice to the appropriate (over/under)flow bins
    """
    if depth == 0:
        ndslice = [_MaybeSumSlice(s.start, s.stop, False) for s in ndslice]
    if depth == len(ndslice):
        slice_op = tuple(slice(s.start, s.stop) for s in ndslice)
        sum_op = tuple(i for i, s in enumerate(ndslice) if s.sum)
        return array[slice_op].sum(axis=sum_op, keepdims=True)
    slist = []
    newslice = ndslice[:]
    if ndslice[depth].start is not None:
        newslice[depth] = _MaybeSumSlice(None, ndslice[depth].start, True)
        slist.append(_assemble_blocks(array, newslice, depth + 1))
    newslice[depth] = _MaybeSumSlice(ndslice[depth].start, ndslice[depth].stop, False)
    slist.append(_assemble_blocks(array, newslice, depth + 1))
    if ndslice[depth].stop is not None:
        newslice[depth] = _MaybeSumSlice(ndslice[depth].stop, -1, True)
        slist.append(_assemble_blocks(array, newslice, depth + 1))
        newslice[depth] = _MaybeSumSlice(-1, None, False)
        slist.append(_assemble_blocks(array, newslice, depth + 1))
    return slist


class Hist:
    """
    Construct a new histogram.

    Parameters
    ----------
        *args : Axis
            Provide 1 or more axis instances.
        label: str = None
            Histogram's label
        name: str = None
            Histogram's name
    """

    DEFAULT_DTYPE = "d"

    def __init__(
        self,
        *axes,
        label: str | None = None,
        name: str | None = None,
    ) -> None:
        self._label = label
        self._name = name
        self._dtype = Hist.DEFAULT_DTYPE
        self._axes = axes

        if not all(isinstance(ax, Axis) for ax in self._axes):
            raise TypeError("All axes must be derived from Axis class")
        self._dense_shape = tuple(
            [
                ax.size + 3 if isinstance(ax, Regular) else ax.size + 1
                for ax in self._axes
                if isinstance(ax, DenseAxis)
            ]
        )
        self._sumw = {}
        # Storage of sumw2 starts at first use of weight keyword in fill()
        self._sumw2 = None

    def __repr__(self):
        repr_str = "Hist("
        for ax in self._axes:
            repr_str += f"{ax!r}, "
        return f"{repr_str[:-2]})"

    @property
    def label(self):
        """A label describing the meaning of the sum of weights"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def name(self):
        """A label describing the meaning of the sum of weights"""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def axes(self):
        """Get all axes in this histogram"""
        return self._axes

    def dim(self):
        """Dimension of this histogram (number of axes)"""
        return len(self._axes)

    def dense_dim(self):
        """Dense dimension of this histogram (number of non-sparse axes)"""
        return len(self._dense_shape)

    def sparse_axes(self):
        """All sparse axes"""
        return [ax for ax in self._axes if isinstance(ax, SparseAxis)]

    def _isparse(self, axis):
        return self.sparse_axes().index(axis)

    def _init_sumw2(self):
        self._sumw2 = {}
        for key in self._sumw:
            self._sumw2[key] = self._sumw[key].copy()

    # TODO: should allow better indexing (UHI)
    def __getitem__(self, keys):
        if isinstance(keys, slice) and not all(
            isinstance(s, (int, float)) or s is None
            for s in [keys.start, keys.stop, keys.step]
        ):
            raise ValueError("use to_boost/to_hist to access other UHI functionalities")
        if not isinstance(keys, slice) and not isinstance(keys, (int, float, tuple)):
            raise ValueError("use to_boost/to_hist to access other UHI functionalities")
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) != self.dim():
            raise IndexError("Too many or too less indices for this histogram")
        elif len(keys) < self.dim():
            if Ellipsis in keys:
                idx = keys.index(Ellipsis)
                slices = (slice(None),) * (self.dim() - len(keys) + 1)
                keys = keys[:idx] + slices + keys[idx + 1 :]
            else:
                slices = (slice(None),) * (self.dim() - len(keys))
                keys += slices
        sparse_idx = []
        dense_idx = []
        new_dims = []
        for s, ax in zip(keys, self._axes):
            if isinstance(ax, SparseAxis):
                sparse_idx.append(ax._ireduce(s))
                new_dims.append(ax)
            else:
                islice = ax._ireduce(s)
                dense_idx.append(islice)
                new_dims.append(ax.reduced(islice))
        dense_idx = tuple(dense_idx)

        def dense_op(array):
            as_numpy = array.get()
            blocked = np.block(_assemble_blocks(as_numpy, dense_idx))
            return cupy.asarray(blocked)

        out = Hist(*new_dims, label=self._label)
        if self._sumw2 is not None:
            out._init_sumw2()
        for sparse_key in self._sumw:
            if not all(k in idx for k, idx in zip(sparse_key, sparse_idx)):
                continue
            if sparse_key in out._sumw:
                out._sumw[sparse_key] += dense_op(self._sumw[sparse_key])
                if self._sumw2 is not None:
                    out._sumw2[sparse_key] += dense_op(self._sumw2[sparse_key])
            else:
                out._sumw[sparse_key] = dense_op(self._sumw[sparse_key]).copy()
                if self._sumw2 is not None:
                    out._sumw2[sparse_key] = dense_op(self._sumw2[sparse_key]).copy()
        return out

    def fill(self, *args, weight=None):
        """
        Insert data into the histogram.

        Parameters
        ----------
            *args : cupy.ndarray
                Provide one value or array per dimension.
            weight : cupy.ndarray
                Provide weights.
        """
        if not all(isinstance(a, (cupy.ndarray, str)) for a in args):
            raise TypeError("pass CuPy arrays")
        if weight is not None and not isinstance(weight, cupy.ndarray):
            raise TypeError("pass CuPy arrays")

        if len(self._axes) != len(args):
            raise ValueError("mismatching dimensions for provided values and axes")

        if weight is not None and self._sumw2 is None:
            self._init_sumw2()

        sparse_key = tuple(
            d.index(value)
            for d, value in zip(self._axes, args)
            if isinstance(d, SparseAxis)
        )

        if sparse_key not in self._sumw:
            self._sumw[sparse_key] = cupy.zeros(
                shape=self._dense_shape, dtype=self._dtype
            )
            if self._sumw2 is not None:
                self._sumw2[sparse_key] = cupy.zeros(
                    shape=self._dense_shape, dtype=self._dtype
                )

        if self.dense_dim() > 0:
            dense_indices = tuple(
                cupy.asarray(d.index(value))
                for d, value in zip(self._axes, args)
                if isinstance(d, DenseAxis)
            )
            xy = cupy.atleast_1d(
                cupy.ravel_multi_index(dense_indices, self._dense_shape)
            )
            if weight is not None:
                self._sumw[sparse_key][:] += cupy.bincount(
                    xy, weights=weight, minlength=np.array(self._dense_shape).prod()
                ).reshape(self._dense_shape)
                self._sumw2[sparse_key][:] += cupy.bincount(
                    xy,
                    weights=weight**2,
                    minlength=np.array(self._dense_shape).prod(),
                ).reshape(self._dense_shape)
            else:
                self._sumw[sparse_key][:] += cupy.bincount(
                    xy, weights=None, minlength=np.array(self._dense_shape).prod()
                ).reshape(self._dense_shape)
                if self._sumw2 is not None:
                    self._sumw2[sparse_key][:] += cupy.bincount(
                        xy,
                        weights=None,
                        minlength=np.array(self._dense_shape).prod(),
                    ).reshape(self._dense_shape)
        elif weight is not None:
            self._sumw[sparse_key] += cupy.sum(weight)
            self._sumw2[sparse_key] += cupy.sum(weight**2)
        else:
            self._sumw[sparse_key] += 1.0
            if self._sumw2 is not None:
                self._sumw2[sparse_key] += 1.0

    def _view_dim(self, arr, flow):
        if self.dense_dim() == 0:
            return arr
        else:
            return arr[tuple(_overflow_behavior(flow) for _ in range(self.dense_dim()))]

    def values(self, flow=False):
        """Extract the values from this histogram.

        Parameters
        ----------
        flow : bool
        """

        # TODO: cleanup logic for sparse axis
        # out = {}
        # for sparse_key in self._sumw:
        #     id_key = tuple(ax[k] for ax, k in zip(self.sparse_axes(), sparse_key))
        #     if sumw2:
        #         if self._sumw2 is not None:
        #             w2 = self._view_dim(self._sumw2[sparse_key])
        #         else:
        #             w2 = self._view_dim(self._sumw[sparse_key])
        #         out[id_key] = (self._view_dim(self._sumw[sparse_key]), w2)
        #     else:
        #         out[id_key] = self._view_dim(self._sumw[sparse_key])
        # return out

        return (
            self._view_dim(cupy.zeros(shape=self._dense_shape), flow)
            if self._sumw == {}
            else self._view_dim(next(iter(self._sumw.values())), flow)
        )

    def variance(self, flow=False):
        """Extract the variances from this histogram.

        Parameters
        ----------
        flow : bool
        """
        return (
            None
            if self._sumw2 is None
            else self._view_dim(next(iter(self._sumw2.values())), flow)
        )

    # TODO: cleanup logic for sparse axis
    # def identifiers(self, axis, overflow="none"):
    #     """Return a list of identifiers for an axis

    #     Parameters
    #     ----------
    #         axis
    #             Axis object
    #         overflow
    #             See `sum` description for meaning of allowed values
    #     """
    #     if isinstance(axis, SparseAxis):
    #         out = []
    #         isparse = self._isparse(axis)
    #         for identifier in axis.identifiers():
    #             if any(k[isparse] == axis.index(identifier) for k in self._sumw):
    #                 out.append(identifier)
    #         if axis.sorting == "integral":
    #             hproj = {
    #                 key[0]: integral
    #                 for key, integral in self.project(axis).values().items()
    #             }
    #             out.sort(key=lambda k: hproj[k.name])
    #         return out
    #     elif isinstance(axis, DenseAxis):
    #         return axis.identifiers(overflow=overflow)

    def to_boost(self):
        """
        Convert this cuda_histogram object to a boost_histogram object.

        underflow and overflow are set True and nanflow is lost in the conversion.

        Appropriate boost-histogram axis and storage are automatically chosen.
        All the arguments of cuda-histogram's axis and histogram are passed down.
        """
        import boost_histogram

        newaxes = []
        for axis in self.axes():
            if isinstance(axis, Regular):
                newaxis = boost_histogram.axis.Regular(
                    axis._bins,
                    axis._lo,
                    axis._hi,
                    underflow=True,
                    overflow=True,
                )
                newaxis._ax.metadata["name"] = axis.name
                newaxis.label = axis.label
                newaxes.append(newaxis)
            elif isinstance(axis, Variable):
                newaxis = boost_histogram.axis.Variable(
                    axis.edges(),
                    underflow=True,
                    overflow=True,
                )
                newaxis._ax.metadata["name"] = axis.name
                newaxis.label = axis.label
                newaxes.append(newaxis)
            # TODO: cleanup logic for sparse axis
            # elif isinstance(axis, Cat):
            #     identifiers = self.identifiers(axis)
            #     newaxis = boost_histogram.axis.StrCategory(
            #         [x.name for x in identifiers],
            #         growth=True,
            #     )
            #     newaxis.name = axis.name
            #     newaxis.label = axis.label
            #     newaxis.bin_labels = [x.label for x in identifiers]
            #     newaxes.append(newaxis)

        if self._sumw2 is None:
            storage = boost_histogram.storage.Double()
        else:
            storage = boost_histogram.storage.Weight()

        out = boost_histogram.Histogram(*newaxes, storage=storage)
        out.label = self.label

        view = out.view(flow=True)
        nonan = [slice(None, -1, None)] * (len(newaxes))
        if self._sumw2 is None:
            view[:] = self.values(flow=True)[(*nonan,)].get()
        else:
            view[:] = cupy.stack(
                (
                    self.values(flow=True)[(*nonan,)],
                    self.variance(flow=True)[(*nonan,)],
                ),
                axis=len(newaxes),
            ).get()

        # TODO: cleanup logic for sparse axis
        # def expandkey(key):
        #     kiter = iter(key)
        #     for ax in newaxes:
        #         if isinstance(ax, boost_histogram.axis.StrCategory):
        #             yield ax.index(next(kiter))
        #         else:
        #             yield slice(None)

        # if self._sumw2 is None:
        #     values = self.values(overflow="all")
        #     for sparse_key, sumw in values.items():
        #         index = tuple(expandkey(sparse_key))
        #         view = out.view(flow=True)
        #         view[index] = sumw.get()
        # else:
        #     values = self.values(sumw2=True, overflow="all")
        #     for sparse_key, (sumw, sumw2) in values.items():
        #         index = tuple(expandkey(sparse_key))
        #         view = out.view(flow=True)
        #         view[index].value = sumw.get()
        #         view[index].variance = sumw2.get()

        return out

    def to_hist(self):
        """Convert this cuda_histogram object to a hist object"""
        import hist

        return hist.Hist(self.to_boost(), name=self.name)
