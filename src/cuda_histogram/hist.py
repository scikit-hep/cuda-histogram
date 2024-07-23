from __future__ import annotations

import copy
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Sequence

import awkward
import cupy
import numpy as np

from cuda_histogram.axis import (
    Axis,
    Bin,
    Cat,
    DenseAxis,
    Interval,
    SparseAxis,
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


class AccumulatorABC(metaclass=ABCMeta):
    """Abstract base class for an accumulator

    Accumulators are abstract objects that enable the reduce stage of the typical map-reduce
    scaleout that we do in Coffea. One concrete example is a histogram. The idea is that an
    accumulator definition holds enough information to be able to create an empty accumulator
    (the ``identity()`` method) and add two compatible accumulators together (the ``add()`` method).
    The former is not strictly necessary, but helps with book-keeping. Here we show an example usage
    of a few accumulator types. An arbitrary-depth nesting of dictionary accumulators is supported, much
    like the behavior of directories in ROOT hadd.

    After defining an accumulator::

        from coffea.processor import dict_accumulator, column_accumulator, defaultdict_accumulator
        from cuda_histogram import Hist, Bin
        import numpy as np

        adef = dict_accumulator({
            'cutflow': defaultdict_accumulator(int),
            'pt': Hist("counts", Bin("pt", "$p_T$", 100, 0, 100)),
            'final_pt': column_accumulator(np.zeros(shape=(0,))),
        })

    Notice that this function does not mutate ``adef``::

        def fill(n):
            ptvals = np.random.exponential(scale=30, size=n)
            cut = ptvals > 200.
            acc = adef.identity()
            acc['cutflow']['pt>200'] += cut.sum()
            acc['pt'].fill(pt=ptvals)
            acc['final_pt'] += column_accumulator(ptvals[cut])
            return acc

    As such, we can execute it several times in parallel and reduce the result::

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            outputs = executor.map(fill, [2000, 2000])

        combined = sum(outputs, adef.identity())


    Derived classes must implement
        - ``identity()``: returns a new object of same type as self,
          such that ``self + self.identity() == self``
        - ``add(other)``: adds an object of same type as self to self

    Concrete implementations are then provided for ``__add__``, ``__radd__``, and ``__iadd__``.
    """

    @abstractmethod
    def identity(self):
        """Identity of the accumulator

        A value such that any other value added to it will return
        the other value
        """

    @abstractmethod
    def add(self, other):
        """Add another accumulator to this one in-place"""

    def __add__(self, other):
        ret = self.identity()
        ret.add(self)
        ret.add(other)
        return ret

    def __radd__(self, other):
        ret = self.identity()
        ret.add(other)
        ret.add(self)
        return ret

    def __iadd__(self, other):
        self.add(other)
        return self


class Hist(AccumulatorABC):
    """
    Specify a multidimensional histogram.

    Parameters
    ----------
        label : str
            A description of the meaning of the sum of weights
        ``*axes``
            positional list of `Cat` or `Bin` objects, denoting the axes of the histogram
        axes : collections.abc.Sequence
            list of `Cat` or `Bin` objects, denoting the axes of the histogram (overridden by ``*axes``)
        dtype : str
            Underlying numpy dtype to use for storing sum of weights

    Examples
    --------

    Creating a histogram with a sparse axis, and two dense axes::

        import cuda_histogram as chist

        h = chist.Hist("Observed bird count",
                             chist.Cat("species", "Bird species"),
                             chist.Bin("x", "x coordinate [m]", 20, -5, 5),
                             chist.Bin("y", "y coordinate [m]", 20, -5, 5),
                             )

        # or

        h = chist.Hist(label="Observed bird count",
                             axes=(chist.Cat("species", "Bird species"),
                                   chist.Bin("x", "x coordinate [m]", 20, -5, 5),
                                   chist.Bin("y", "y coordinate [m]", 20, -5, 5),
                                  )
                             )

        # or

        h = chist.Hist(axes=[chist.Cat("species", "Bird species"),
                                   chist.Bin("x", "x coordinate [m]", 20, -5, 5),
                                   chist.Bin("y", "y coordinate [m]", 20, -5, 5),
                                  ],
                             label="Observed bird count",
                             )

    which produces:

    >>> h
    <Hist (species,x,y) instance at 0x10d84b550>

    """

    #: Default numpy dtype to store sum of weights
    DEFAULT_DTYPE = "d"

    def __init__(self, label, *axes, **kwargs):
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        self._label = label
        self._dtype = kwargs.get("dtype", Hist.DEFAULT_DTYPE)
        self._axes = axes
        if len(axes) == 0 and "axes" in kwargs:
            if not isinstance(kwargs["axes"], Sequence):
                raise TypeError("axes must be a sequence type! (tuple, list, etc.)")
            self._axes = tuple(kwargs["axes"])
        elif len(axes) != 0 and "axes" in kwargs:
            warnings.warn(
                "axes defined by both positional arguments and keyword argument, using positional arguments"
            )

        if not all(isinstance(ax, Axis) for ax in self._axes):
            del self._axes
            raise TypeError("All axes must be derived from Axis class")
        # if we stably partition axes to sparse, then dense, some things simplify
        # ..but then the user would then see the order change under them
        self._dense_shape = tuple(
            [ax.size for ax in self._axes if isinstance(ax, DenseAxis)]
        )
        if np.prod(self._dense_shape) > 10000000:
            warnings.warn("Allocating a large (>10M bin) histogram!", RuntimeWarning)
        self._sumw = {}
        # Storage of sumw2 starts at first use of weight keyword in fill()
        self._sumw2 = None

    def __repr__(self):
        return "<{} ({}) instance at 0x{:0x}>".format(
            self.__class__.__name__,
            ",".join(d.name for d in self.axes()),
            id(self),
        )

    @property
    def label(self):
        """A label describing the meaning of the sum of weights"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def copy(self, content=True):
        """Create a deep copy

        Parameters
        ----------
            content : bool
                If set false, only the histogram definition is copied, resetting
                the sum of weights to zero
        """
        out = Hist(self._label, *self._axes, dtype=self._dtype)
        if self._sumw2 is not None:
            out._sumw2 = {}
        if content:
            out._sumw = copy.deepcopy(self._sumw)
            out._sumw2 = copy.deepcopy(self._sumw2)
        return out

    def identity(self):
        """The identity (zero value) of this accumulator"""
        return self.copy(content=False)

    def clear(self):
        """Clear all content in this histogram"""
        self._sumw = {}
        self._sumw2 = None

    def axis(self, axis_name):
        """Get an ``Axis`` object"""
        if axis_name in self._axes:
            return self._axes[self._axes.index(axis_name)]
        raise KeyError(f"No axis {axis_name} found in {self!r}")

    def axes(self):
        """Get all axes in this histogram"""
        return self._axes

    @property
    def fields(self):
        """This is a stub for histbook compatibility"""
        return [ax.name for ax in self._axes]

    def dim(self):
        """Dimension of this histogram (number of axes)"""
        return len(self._axes)

    def dense_dim(self):
        """Dense dimension of this histogram (number of non-sparse axes)"""
        return len(self._dense_shape)

    def sparse_dim(self):
        """Sparse dimension of this histogram (number of sparse axes)"""
        return self.dim() - self.dense_dim()

    def dense_axes(self):
        """All dense axes"""
        return [ax for ax in self._axes if isinstance(ax, DenseAxis)]

    def sparse_axes(self):
        """All sparse axes"""
        return [ax for ax in self._axes if isinstance(ax, SparseAxis)]

    def sparse_nbins(self):
        """Total number of sparse bins"""
        return len(self._sumw)

    def _idense(self, axis):
        return self.dense_axes().index(axis)

    def _isparse(self, axis):
        return self.sparse_axes().index(axis)

    def _init_sumw2(self):
        self._sumw2 = {}
        for key in self._sumw:
            self._sumw2[key] = self._sumw[key].copy()

    def compatible(self, other):
        """Checks if this histogram is compatible with another, i.e. they have identical binning"""
        if self.dim() != other.dim():
            return False
        if {d.name for d in self.sparse_axes()} != {
            d.name for d in other.sparse_axes()
        }:
            return False
        if not all(d1 == d2 for d1, d2 in zip(self.dense_axes(), other.dense_axes())):  # noqa: SIM103
            return False
        return True

    def add(self, other):
        """Add another histogram into this one, in-place"""
        if not self.compatible(other):
            raise ValueError(
                f"Cannot add this histogram with histogram {other!r} of dissimilar dimensions"
            )

        raxes = other.sparse_axes()

        def add_dict(left, right):
            for rkey in right:
                lkey = tuple(
                    self.axis(rax).index(rax[ridx]) for rax, ridx in zip(raxes, rkey)
                )
                if lkey in left:
                    left[lkey] += right[rkey]
                else:
                    left[lkey] = copy.deepcopy(right[rkey])

        if self._sumw2 is None and other._sumw2 is None:
            pass
        elif self._sumw2 is None:
            self._init_sumw2()
            add_dict(self._sumw2, other._sumw2)
        elif other._sumw2 is None:
            add_dict(self._sumw2, other._sumw)
        else:
            add_dict(self._sumw2, other._sumw2)
        add_dict(self._sumw, other._sumw)
        return self

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) > self.dim():
            raise IndexError("Too many indices for this histogram")
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

        out = Hist(self._label, *new_dims, dtype=self._dtype)
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

    def fill(self, **values):
        """Fill sum of weights from columns

        Parameters
        ----------
            ``**values``
                Keyword arguments, one for each axis name, of either flat numpy arrays
                (for dense dimensions) or literals (for sparse dimensions) which will
                be used to fill bins at the corresponding indices.

        Note
        ----
            The reserved keyword ``weight``, if specified, will increment sum of weights
            by the given column values, which must be broadcastable to the same dimension as all other
            columns.  Upon first use, this will trigger the storage of the sum of squared weights.


        Examples
        --------

        Filling the histogram from the `Hist` example:

        >>> h.fill(species='ducks', x=np.random.normal(size=10), y=np.random.normal(size=10), weight=np.ones(size=10) * 3)

        """
        weight = values.pop("weight", None)
        if isinstance(weight, (awkward.Array, cupy.ndarray, np.ndarray)):
            weight = cupy.array(weight)
        if isinstance(weight, numbers.Number):
            weight = cupy.atleast_1d(weight)
        if not all(d.name in values for d in self._axes):
            missing = ", ".join(d.name for d in self._axes if d.name not in values)
            raise ValueError(
                f"Not all axes specified for {self!r}.  Missing: {missing}"
            )
        if not all(name in self._axes for name in values):
            extra = ", ".join(name for name in values if name not in self._axes)
            raise ValueError(
                f"Unrecognized axes specified for {self!r}.  Extraneous: {extra}"
            )

        if weight is not None and self._sumw2 is None:
            self._init_sumw2()

        sparse_key = tuple(d.index(values[d.name]) for d in self.sparse_axes())
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
                cupy.asarray(d.index(values[d.name]))
                for d in self._axes
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

    def sum(self, *axes, **kwargs):
        """Integrates out a set of axes, producing a new histogram

        Parameters
        ----------
            ``*axes``
                Positional list of axes to integrate out (either a string or an Axis object)

            overflow : {'none', 'under', 'over', 'all', 'allnan'}, optional
                How to treat the overflow bins in the sum.  Only applies to dense axes.
                'all' includes both under- and over-flow but not nan-flow bins.
                Default is 'none'.
        """
        overflow = kwargs.pop("overflow", "none")
        axes = [self.axis(ax) for ax in axes]
        reduced_dims = [ax for ax in self._axes if ax not in axes]
        out = Hist(self._label, *reduced_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()

        sparse_drop = []
        dense_slice = [slice(None)] * self.dense_dim()
        dense_sum_dim = []
        for axis in axes:
            if isinstance(axis, DenseAxis):
                idense = self._idense(axis)
                dense_sum_dim.append(idense)
                dense_slice[idense] = _overflow_behavior(overflow)
            elif isinstance(axis, SparseAxis):
                isparse = self._isparse(axis)
                sparse_drop.append(isparse)
        dense_slice = tuple(dense_slice)
        dense_sum_dim = tuple(dense_sum_dim)

        def dense_op(array):
            if len(dense_sum_dim) > 0:
                return np.sum(array[dense_slice], axis=dense_sum_dim)
            return array

        for key in self._sumw:
            new_key = tuple(k for i, k in enumerate(key) if i not in sparse_drop)
            if new_key in out._sumw:
                out._sumw[new_key] += dense_op(self._sumw[key])
                if self._sumw2 is not None:
                    out._sumw2[new_key] += dense_op(self._sumw2[key])
            else:
                out._sumw[new_key] = dense_op(self._sumw[key]).copy()
                if self._sumw2 is not None:
                    out._sumw2[new_key] = dense_op(self._sumw2[key]).copy()
        return out

    def project(self, *axes, **kwargs):
        """Project histogram onto a subset of its axes

        Parameters
        ----------
            ``*axes`` : str or Axis
                Positional list of axes to project on to
            overflow : str
                Controls behavior of integration over remaining axes.
                See `sum` description for meaning of allowed values
                Default is to *not include* overflow bins
        """
        overflow = kwargs.pop("overflow", "none")
        axes = [self.axis(ax) for ax in axes]
        toremove = [ax for ax in self.axes() if ax not in axes]
        return self.sum(*toremove, overflow=overflow)

    def integrate(self, axis_name, int_range=None, overflow="none"):
        """Integrates current histogram along one dimension

        Parameters
        ----------
            axis_name : str or Axis
                Which dimension to reduce on
            int_range : slice
                Any slice, list, string, or other object that the axis will understand
                Default is to integrate over the whole range
            overflow : str
                See `sum` description for meaning of allowed values
                Default is to *not include* overflow bins

        """
        if int_range is None:
            int_range = slice(None)
        axis = self.axis(axis_name)
        full_slice = tuple(
            slice(None) if ax != axis else int_range for ax in self._axes
        )
        if isinstance(int_range, Interval):
            # Handle overflow intervals nicely
            if int_range.nan():
                overflow = "justnan"
            elif int_range.lo == -np.inf:
                overflow = "under"
            elif int_range.hi == np.inf:
                overflow = "over"
        return self[full_slice].sum(
            axis.name, overflow=overflow
        )  # slice may make new axis, use name

    def remove(self, bins, axis):
        """Remove bins from a sparse axis

        Parameters
        ----------
            bins : iterable
                A list of bin identifiers to remove
            axis : str or Axis
                Axis name or SparseAxis instance

        Returns a *copy* of the histogram with specified bins removed, not an in-place operation
        """
        axis = self.axis(axis)
        if not isinstance(axis, SparseAxis):
            raise NotImplementedError(
                "Hist.remove() only supports removing items from a sparse axis."
            )
        bins = [axis.index(binid) for binid in bins]
        keep = [binid.name for binid in self.identifiers(axis) if binid not in bins]
        full_slice = tuple(slice(None) if ax != axis else keep for ax in self._axes)
        return self[full_slice]

    def group(self, old_axes, new_axis, mapping, overflow="none"):
        """Group a set of slices on old axes into a single new axis

        Parameters
        ----------
            old_axes
                Axis or tuple of axes which are being grouped
            new_axis
                A new sparse dimension definition, e.g. a `Cat` instance
            mapping : dict
                A mapping ``{'new_bin': (slice, ...), ...}`` where each
                slice is on the axes being re-binned.  In the case of
                a single axis for ``old_axes``, ``{'new_bin': slice, ...}``
                is admissible.
            overflow : str
                See `sum` description for meaning of allowed values
                Default is to *not include* overflow bins

        Returns a new histogram object
        """
        if not isinstance(new_axis, SparseAxis):
            raise TypeError(
                "New axis must be a sparse axis.  Note: Hist.group() signature has changed to group(old_axes, new_axis, ...)!"
            )
        if new_axis in self.axes() and self.axis(new_axis) is new_axis:
            raise RuntimeError(
                "new_axis is already in the list of axes.  Note: Hist.group() signature has changed to group(old_axes, new_axis, ...)!"
            )
        if not isinstance(old_axes, tuple):
            old_axes = (old_axes,)
        old_axes = [self.axis(ax) for ax in old_axes]
        old_indices = [i for i, ax in enumerate(self._axes) if ax in old_axes]
        new_dims = [new_axis] + [ax for ax in self._axes if ax not in old_axes]
        out = Hist(self._label, *new_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()
        for new_cat in mapping:
            the_slice = mapping[new_cat]
            if not isinstance(the_slice, tuple):
                the_slice = (the_slice,)
            if len(the_slice) != len(old_axes):
                raise Exception("Slicing does not match number of axes being rebinned")
            full_slice = [slice(None)] * self.dim()
            for idx, s in zip(old_indices, the_slice):
                full_slice[idx] = s
            full_slice = tuple(full_slice)
            reduced_hist = self[full_slice].sum(
                *tuple(ax.name for ax in old_axes), overflow=overflow
            )  # slice may change old axis binning
            new_idx = new_axis.index(new_cat)
            for key in reduced_hist._sumw:
                new_key = (new_idx, *key)
                out._sumw[new_key] = reduced_hist._sumw[key]
                if self._sumw2 is not None:
                    out._sumw2[new_key] = reduced_hist._sumw2[key]
        return out

    def rebin(self, old_axis, new_axis):
        """Rebin a dense axis

        This function will construct the mapping from old to new axis, and
        constructs a new histogram, rebinning the sum of weights along that dimension.

        Note
        ----
        No interpolation is performed, so the user must be sure the old
        and new axes have compatible bin boundaries, e.g. that they evenly
        divide each other.

        Parameters
        ----------
            old_axis : str or Axis
                Axis to rebin
            new_axis : str or Axis or int
                A DenseAxis object defining the new axis (e.g. a `Bin` instance).
                If a number N is supplied, the old axis edges are downsampled by N,
                resulting in a histogram with ``old_nbins // N`` bins.

        Returns a new `Hist` object.
        """
        old_axis = self.axis(old_axis)
        if isinstance(new_axis, numbers.Integral):
            new_axis = Bin(old_axis.name, old_axis.label, old_axis.edges()[::new_axis])
        new_dims = [ax if ax != old_axis else new_axis for ax in self._axes]
        out = Hist(self._label, *new_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()

        # would have been nice to use ufunc.reduceat, but we should support arbitrary reshuffling
        idense = self._idense(old_axis)

        def view_ax(idx):
            fullindex = [slice(None)] * self.dense_dim()
            fullindex[idense] = idx
            return tuple(fullindex)

        binmap = [new_axis.index(i) for i in old_axis.identifiers(overflow="allnan")]

        def dense_op(array):
            anew = np.zeros(out._dense_shape, dtype=out._dtype)
            for iold, inew in enumerate(binmap):
                anew[view_ax(inew)] += array[view_ax(iold)]
            return anew

        for key in self._sumw:
            out._sumw[key] = dense_op(self._sumw[key])
            if self._sumw2 is not None:
                out._sumw2[key] = dense_op(self._sumw2[key])
        return out

    def values(self, sumw2=False, overflow="none"):
        """Extract the sum of weights arrays from this histogram

        Parameters
        ----------
            sumw2 : bool
                If True, frequencies is a tuple of arrays (sum weights, sum squared weights)
            overflow
                See `sum` description for meaning of allowed values

        Returns a mapping ``{(sparse identifier, ...): np.array(...), ...}``
        where each array has dimension `dense_dim` and shape matching
        the number of bins per axis, plus 0-3 overflow bins depending
        on the ``overflow`` argument.
        """

        def view_dim(arr):
            if self.dense_dim() == 0:
                return arr
            else:
                return arr[
                    tuple(_overflow_behavior(overflow) for _ in range(self.dense_dim()))
                ]

        out = {}
        for sparse_key in self._sumw:
            id_key = tuple(ax[k] for ax, k in zip(self.sparse_axes(), sparse_key))
            if sumw2:
                if self._sumw2 is not None:
                    w2 = view_dim(self._sumw2[sparse_key])
                else:
                    w2 = view_dim(self._sumw[sparse_key])
                out[id_key] = (view_dim(self._sumw[sparse_key]), w2)
            else:
                out[id_key] = view_dim(self._sumw[sparse_key])
        return out

    def scale(self, factor, axis=None):
        """Scale histogram in-place by factor

        Parameters
        ----------
            factor : float or dict
                A number or mapping of identifier to number
            axis : optional
                Which (sparse) axis the dict applies to, may be a tuples of axes.
                The dict keys must follow the same structure.

        Examples
        --------
        This function is useful to quickly reweight according to some
        weight mapping along a sparse axis, such as the ``species`` axis
        in the `Hist` example:

        >>> h.scale({'ducks': 0.3, 'geese': 1.2}, axis='species')
        >>> h.scale({('ducks',): 0.5}, axis=('species',))
        >>> h.scale({('geese', 'honk'): 5.0}, axis=('species', 'vocalization'))
        """
        if self._sumw2 is None:
            self._init_sumw2()
        if isinstance(factor, numbers.Number) and axis is None:
            for key in self._sumw:
                self._sumw[key] *= factor
                self._sumw2[key] *= factor**2
        elif isinstance(factor, dict):
            if not isinstance(axis, tuple):
                axis = (axis,)
                factor = {(k,): v for k, v in factor.items()}
            axis = tuple(map(self.axis, axis))
            isparse = list(map(self._isparse, axis))
            factor = {
                tuple(a.index(e) for a, e in zip(axis, k)): v for k, v in factor.items()
            }
            for key in self._sumw:
                factor_key = tuple(key[i] for i in isparse)
                if factor_key in factor:
                    self._sumw[key] *= factor[factor_key]
                    self._sumw2[key] *= factor[factor_key] ** 2
        elif isinstance(factor, np.ndarray):
            axis = self.axis(axis)
            raise NotImplementedError("Scale dense dimension by a factor")
        else:
            raise TypeError("Could not interpret scale factor")

    def identifiers(self, axis, overflow="none"):
        """Return a list of identifiers for an axis

        Parameters
        ----------
            axis
                Axis name or Axis object
            overflow
                See `sum` description for meaning of allowed values
        """
        axis = self.axis(axis)
        if isinstance(axis, SparseAxis):  # noqa: RET503
            out = []
            isparse = self._isparse(axis)
            for identifier in axis.identifiers():
                if any(k[isparse] == axis.index(identifier) for k in self._sumw):
                    out.append(identifier)
            if axis.sorting == "integral":
                hproj = {
                    key[0]: integral
                    for key, integral in self.project(axis).values().items()
                }
                out.sort(key=lambda k: hproj[k.name])
            return out
        elif isinstance(axis, DenseAxis):
            return axis.identifiers(overflow=overflow)

    def to_boost(self):
        """Convert this cuda_histogram object to a boost_histogram object"""
        import boost_histogram

        newaxes = []
        for axis in self.axes():
            if isinstance(axis, Bin) and axis._uniform:
                newaxis = boost_histogram.axis.Regular(
                    axis._bins,
                    axis._lo,
                    axis._hi,
                    underflow=True,
                    overflow=True,
                )
                newaxis.name = axis.name
                newaxis.label = axis.label
                newaxes.append(newaxis)
            elif isinstance(axis, Bin) and not axis._uniform:
                newaxis = boost_histogram.axis.Variable(
                    axis.edges(),
                    underflow=True,
                    overflow=True,
                )
                newaxis.name = axis.name
                newaxis.label = axis.label
                newaxes.append(newaxis)
            elif isinstance(axis, Cat):
                identifiers = self.identifiers(axis)
                newaxis = boost_histogram.axis.StrCategory(
                    [x.name for x in identifiers],
                    growth=True,
                )
                newaxis.name = axis.name
                newaxis.label = axis.label
                newaxis.bin_labels = [x.label for x in identifiers]
                newaxes.append(newaxis)

        if self._sumw2 is None:
            storage = boost_histogram.storage.Double()
        else:
            storage = boost_histogram.storage.Weight()

        out = boost_histogram.Histogram(*newaxes, storage=storage)
        out.label = self.label

        def expandkey(key):
            kiter = iter(key)
            for ax in newaxes:
                if isinstance(ax, boost_histogram.axis.StrCategory):
                    yield ax.index(next(kiter))
                else:
                    yield slice(None)

        if self._sumw2 is None:
            values = self.values(overflow="all")
            for sparse_key, sumw in values.items():
                index = tuple(expandkey(sparse_key))
                view = out.view(flow=True)
                view[index] = sumw.get()
        else:
            values = self.values(sumw2=True, overflow="all")
            for sparse_key, (sumw, sumw2) in values.items():
                index = tuple(expandkey(sparse_key))
                view = out.view(flow=True)
                view[index].value = sumw.get()
                view[index].variance = sumw2.get()

        return out

    def to_hist(self):
        """Convert this cuda_histogram object to a hist object"""
        import hist

        return hist.Hist(self.to_boost())
