from __future__ import annotations

import functools
import numbers
import warnings
from typing import Iterable

import awkward
import cupy
import numpy as np

__all__: list[str] = [
    "Regular",
    "Variable",
    # "Cat",
    "Interval",
    # "StringBin",
    "Bin",
]

_replace_nans = cupy.ElementwiseKernel("T v", "T x", "x = isnan(x)?v:x", "replace_nans")

_clip_bins = cupy.ElementwiseKernel(
    "T Nbins, T lo, T hi, T id",
    "T idx",
    """
    const T floored = floor((id - lo) * float(Nbins) / (hi - lo)) + 1;
    idx = floored < 0 ? 0 : floored;
    idx = floored > Nbins ? Nbins + 1 : floored;
    """,
    "clip_bins",
)


def _overflow_behavior(overflow: bool):
    if not overflow:
        return slice(1, -2)
    else:
        return slice(None, None)


@functools.total_ordering
class Interval:
    """Real number interval

    Totally ordered, assuming no overlap in intervals.
    A special nan interval can be constructed, which is defined
    as greater than ``[*, inf)``

    Parameters
    ----------
        lo : float
            Bin lower bound, inclusive
        hi : float
            Bin upper bound, exclusive
    """

    def __init__(self, lo, hi, label=None):
        self._lo = float(lo)
        self._hi = float(hi)
        self._label = label

    def __repr__(self):
        return f"<{self.__class__.__name__} ({self!s}) instance at 0x{id(self):0x}>"

    def __str__(self):
        if self._label is not None:
            return self._label
        if self.nan():
            return "(nanflow)"
        return "{}{}, {})".format(
            "(" if self._lo == -np.inf else "[",
            self._lo,
            self._hi,
        )

    def __hash__(self):
        return hash((self._lo, self._hi))

    def __lt__(self, other):
        if other.nan() and not self.nan():
            return True
        elif self.nan():
            return False
        elif self._lo < other._lo:
            if self._hi > other._lo:
                raise ValueError(
                    f"Intervals {self!r} and {other!r} intersect! What are you doing?!"
                )
            return True
        return False

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        if other.nan() and self.nan():
            return True
        if self._lo == other._lo and self._hi == other._hi:  # noqa: SIM103
            return True
        return False

    def nan(self):
        return np.isnan(self._hi)

    @property
    def lo(self):
        """Lower boundary of this bin, inclusive"""
        return self._lo

    @property
    def hi(self):
        """Upper boundary of this bin, exclusive"""
        return self._hi

    @property
    def mid(self):
        """Midpoint of this bin"""
        return (self._hi + self._lo) / 2

    @property
    def label(self):
        """Label of this bin, mutable"""
        return self._label

    @label.setter
    def label(self, lbl):
        self._label = lbl


# TODO: cleanup logic for sparse axis
# @functools.total_ordering
# class StringBin:
#     """A string used to fill a sparse axis

#     Totally ordered, lexicographically by name.

#     Parameters
#     ----------
#         name : str
#             Name of the bin, as used in `Hist.fill` calls
#         label : str
#             The `str` representation of this bin can be overridden by
#             a custom label.
#     """

#     def __init__(self, name, label=None):
#         if not isinstance(name, str):
#             raise TypeError(
#                 f"StringBin only supports string categories, received a {name!r}"
#             )
#         elif "*" in name:
#             raise ValueError(
#                 "StringBin does not support character '*' as it conflicts with wildcard mapping."
#             )
#         self._name = name
#         self._label = label

#     def __repr__(self):
#         return f"<{self.__class__.__name__} ({self.name}) instance at 0x{id(self):0x}>"

#     def __str__(self):
#         if self._label is not None:
#             return self._label
#         return self._name

#     def __hash__(self):
#         return hash(self._name)

#     def __lt__(self, other):
#         return self._name < other._name

#     def __eq__(self, other):
#         if isinstance(other, StringBin):
#             return self._name == other._name
#         return False

#     @property
#     def name(self):
#         """Name of this bin, *Immutable*"""
#         return self._name

#     @property
#     def label(self):
#         """Label of this bin, mutable"""
#         return self._label

#     @label.setter
#     def label(self, lbl):
#         self._label = lbl


class Axis:
    """
    Axis: Base class for any type of axis
    Derived classes should implement, at least, an equality override
    """

    def __init__(self, name, label):
        self._name = name
        self._label = label

    def __repr__(self):
        return f"<{self.__class__.__name__} (name={self._name}) instance at 0x{id(self):0x}>"

    @property
    def name(self):
        return self._name

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def __eq__(self, other):
        if isinstance(other, Axis):
            if self._name != other._name:  # noqa: SIM103
                return False
            # label doesn't matter
            return True
        elif isinstance(other, str):
            # Convenient for testing axis in list by name
            return not self._name != other
        raise TypeError(f"Cannot compare an Axis with a {other!r}")


class SparseAxis(Axis):
    """
    SparseAxis: ABC for a sparse axis

    Derived should implement:
        **index(identifier)** - return a hashable object for indexing

        **__eq__(axis)** - axis has same definition (not necessarily same bins)

        **__getitem__(index)** - return an identifier

        **_ireduce(slice)** - return a list of hashes, slice is arbitrary

    What we really want here is a hashlist with some slice sugar on top
    It is usually the case that the identifier is already hashable,
    in which case index and __getitem__ are trivial, but this mechanism
    may be useful if the size of the tuple of identifiers in a
    sparse-binned histogram becomes too large
    """


# TODO: cleanup logic for sparse axis
# class Cat(SparseAxis):
#     """A category axis with name and label

#     Parameters
#     ----------
#         name : str
#             is used as a keyword in histogram filling, immutable
#         label : str
#             describes the meaning of the axis, can be changed
#         sorting : {'identifier', 'placement', 'integral'}, optional
#             Axis sorting when listing identifiers.

#     The number of categories is arbitrary, and can be filled sparsely
#     Identifiers are strings
#     """

#     def __init__(self, name, label, sorting="identifier"):
#         super().__init__(name, label)
#         # In all cases key == value.name
#         self._bins = {}
#         self._sorting = sorting
#         self._sorted = []

#     def index(self, identifier):
#         """Index of a identifier or label

#         Parameters
#         ----------
#             identifier : str or StringBin
#                 The identifier to lookup

#         Returns a `StringBin` corresponding to the given argument (trivial in the case
#         where a `StringBin` was passed) and saves a reference internally in the case where
#         the identifier was not seen before by this axis.
#         """
#         if isinstance(identifier, StringBin):
#             index = identifier
#         else:
#             index = StringBin(identifier)
#         if index.name not in self._bins:
#             self._bins[index.name] = index
#             self._sorted.append(index.name)
#             if self._sorting == "identifier":
#                 self._sorted.sort()
#         return self._bins[index.name]

#     def __eq__(self, other):
#         # Sparse, so as long as name is the same
#         return super().__eq__(other)

#     def __getitem__(self, index):
#         if not isinstance(index, StringBin):
#             raise TypeError(f"Expected a StringBin object, got: {index!r}")
#         identifier = index.name
#         if identifier not in self._bins:
#             raise KeyError("No identifier %r in this Category axis")
#         return identifier

#     def _ireduce(self, the_slice):
#         out = None
#         if isinstance(the_slice, StringBin):
#             out = [the_slice.name]
#         elif isinstance(the_slice, re.Pattern):
#             out = [k for k in self._sorted if the_slice.match(k)]
#         elif isinstance(the_slice, str):
#             pattern = "^" + re.escape(the_slice).replace(r"\*", ".*") + "$"
#             m = re.compile(pattern)
#             out = [k for k in self._sorted if m.match(k)]
#         elif isinstance(the_slice, list):
#             if not all(k in self._sorted for k in the_slice):
#                 warnings.warn(
#                     f"Not all requested indices present in {self!r}", RuntimeWarning
#                 )
#             out = [k for k in self._sorted if k in the_slice]
#         elif isinstance(the_slice, slice):
#             if the_slice.step is not None:
#                 raise IndexError("Not sure how to use slice step for categories...")
#             start, stop = 0, len(self._sorted)
#             if isinstance(the_slice.start, str):
#                 start = self._sorted.index(the_slice.start)
#             else:
#                 start = the_slice.start
#             if isinstance(the_slice.stop, str):
#                 stop = self._sorted.index(the_slice.stop)
#             else:
#                 stop = the_slice.stop
#             out = self._sorted[start:stop]
#         else:
#             raise IndexError(f"Cannot understand slice {the_slice!r} on axis {self!r}")
#         return [self._bins[k] for k in out]

#     @property
#     def size(self):
#         """Number of bins"""
#         return len(self._bins)

#     @property
#     def sorting(self):
#         """Sorting definition to adhere to

#         See `Cat` constructor for possible values
#         """
#         return self._sorting

#     @sorting.setter
#     def sorting(self, newsorting):
#         if newsorting == "placement":
#             # not much we can do about already inserted values
#             pass
#         elif newsorting == "identifier":
#             self._sorted.sort()
#         elif newsorting == "integral":
#             # this will be checked in any Hist.identifiers() call accessing this axis
#             pass
#         else:
#             raise AttributeError(f"Invalid axis sorting type: {newsorting}")
#         self._sorting = newsorting

#     def identifiers(self):
#         """List of `StringBin` identifiers"""
#         return [self._bins[k] for k in self._sorted]


class DenseAxis(Axis):
    """
    DenseAxis: ABC for a fixed-size densely-indexed axis

    Derived should implement:
        **index(identifier)** - return an index

        **__eq__(axis)** - axis has same definition and binning

        **__getitem__(index)** - return an identifier

        **_ireduce(slice)** - return a slice or list of indices, input slice to be interpred as values

        **reduced(islice)** - return a new axis with binning corresponding to the index slice (from _ireduce)
    """


class Bin(DenseAxis):
    """Super class for dense axes.

    A binned axis with name, label, and binning.

    Parameters
    ----------
        n_or_arr : int or list or np.ndarray
            Integer number of bins, if uniform binning. Otherwise, a list or
            numpy 1D array of bin boundaries.
        lo : float, optional
            lower boundary of bin range, if uniform binning
        hi : float, optional
            upper boundary of bin range, if uniform binning
        name : str
            is used as a keyword in histogram filling, immutable
        label : str
            describes the meaning of the axis, can be changed

    This axis will generate frequencies for n+3 bins, special bin indices:
    ``0 = underflow, n+1 = overflow, n+2 = nanflow``
    Bin boundaries are [lo, hi)
    """

    def __init__(self, n_or_arr, lo=None, hi=None, *, name="", label=""):
        self._lazy_intervals = None
        if isinstance(n_or_arr, (list, np.ndarray, cupy.ndarray)):
            self._uniform = False
            self._bins = cupy.array(n_or_arr, dtype="d")
            if not all(np.sort(self._bins) == self._bins):
                raise ValueError("Binning not sorted!")
            self._lo = self._bins[0]
            self._hi = self._bins[-1]
            # to make searchsorted differentiate inf from nan
            self._bins = cupy.append(self._bins, cupy.inf)
            self._interval_bins = cupy.r_[-cupy.inf, self._bins, cupy.nan]
            self._bin_names = np.full(self._interval_bins[:-1].size, None)
        elif isinstance(n_or_arr, numbers.Integral):
            self._uniform = True
            self._lo = lo
            self._hi = hi
            self._bins = n_or_arr
            self._interval_bins = cupy.r_[
                -cupy.inf,
                cupy.linspace(self._lo, self._hi, self._bins + 1),
                cupy.inf,
                cupy.nan,
            ]
            self._bin_names = np.full(self._interval_bins[:-1].size, None)
        self._label = label
        self._name = name

    def __repr__(self):
        class_name = self.__class__.__name__
        return (
            f"{class_name}({self._bins[:-1]})"
            if not self._uniform
            else f"{class_name}{self._bins, self._lo, self._hi}"
        )

    @property
    def _intervals(self):
        if not hasattr(self, "_lazy_intervals") or self._lazy_intervals is None:
            self._lazy_intervals = [
                Interval(low, high, bin)
                for low, high, bin in zip(
                    self._interval_bins[:-1], self._interval_bins[1:], self._bin_names
                )
            ]
        return self._lazy_intervals

    def __getstate__(self):
        if hasattr(self, "_lazy_intervals") and self._lazy_intervals is not None:
            self._bin_names = np.array(
                [interval.label for interval in self._lazy_intervals]
            )
        self.__dict__.pop("_lazy_intervals", None)
        return self.__dict__

    def __setstate__(self, d):
        if "_interval_bins" in d and "_bin_names" not in d:
            d["_bin_names"] = np.full(d["_interval_bins"][:-1].size, None)
        self.__dict__ = d

    def index(self, identifier):
        """Index of a identifier or label

        Parameters
        ----------
            identifier : float or Interval or np.ndarray
                The identifier(s) to lookup.  Supports vectorized
                calls when a numpy 1D array of numbers is passed.

        Returns an integer corresponding to the index in the axis where the histogram would be filled.
        The integer range includes flow bins: ``0 = underflow, n+1 = overflow, n+2 = nanflow``
        """
        isarray = isinstance(
            identifier, (awkward.Array, cupy.ndarray, np.ndarray, list)
        )
        if isarray or isinstance(identifier, numbers.Number):
            identifier = awkward.to_cupy(identifier)  # cupy.asarray(identifier)
            if self._uniform:
                idx = None
                if isarray:
                    idx = cupy.zeros_like(identifier)
                    _clip_bins(float(self._bins), self._lo, self._hi, identifier, idx)
                else:
                    idx = np.clip(
                        np.floor(
                            (identifier - self._lo)
                            * float(self._bins)
                            / (self._hi - self._lo)
                        )
                        + 1,
                        0,
                        self._bins + 1,
                    )

                if isinstance(idx, (cupy.ndarray, np.ndarray)):
                    _replace_nans(
                        self.size - 1,
                        idx if idx.dtype.kind == "f" else idx.astype(cupy.float32),
                    )
                    idx = idx.astype(int)
                elif np.isnan(idx):
                    idx = self.size - 1
                else:
                    idx = int(idx)
                return idx
            else:
                return cupy.searchsorted(self._bins, identifier, side="right")
        elif isinstance(identifier, Interval):
            if identifier.nan():
                return self.size - 1
            for idx, interval in enumerate(self._intervals):
                if (
                    interval._lo <= identifier._lo
                    or cupy.isclose(interval._lo, identifier._lo)
                ) and (
                    interval._hi >= identifier._hi
                    or cupy.isclose(interval._hi, identifier._hi)
                ):
                    return idx
            raise ValueError(
                f"Axis {self!r} has no interval that fully contains identifier {identifier!r}"
            )
        raise TypeError("Request bin indices with a identifier or 1-D array only")

    def __eq__(self, other):
        if isinstance(other, DenseAxis):
            if not super().__eq__(other):
                return False
            if self._uniform != other._uniform:
                return False
            if self._uniform and self._bins != other._bins:
                return False
            if not self._uniform and not all(self._bins == other._bins):  # noqa: SIM103
                return False
            return True
        return super().__eq__(other)

    def __getitem__(self, index):
        return self._intervals[index]

    def _ireduce(self, the_slice):
        if isinstance(the_slice, numbers.Number):
            the_slice = slice(the_slice, the_slice)
        elif isinstance(the_slice, Interval):
            if the_slice.nan():
                return slice(-1, None)
            lo = the_slice._lo if the_slice._lo > -np.inf else None
            hi = the_slice._hi if the_slice._hi < np.inf else None
            the_slice = slice(lo, hi)
        if isinstance(the_slice, slice):
            blo, bhi = None, None
            if the_slice.start is not None:
                if the_slice.start < self._lo:
                    raise ValueError(
                        f"Reducing along axis {self!r}: requested start {the_slice.start!r} exceeds bin boundaries (use open slicing, e.g. x[:stop])"
                    )
                if self._uniform:
                    blo_real = (the_slice.start - self._lo) * self._bins / (
                        self._hi - self._lo
                    ) + 1
                    blo = np.clip(
                        np.round(blo_real).astype(int), 0, self._bins + 1
                    ).item()
                    if abs(blo - blo_real) > 1.0e-14:
                        warnings.warn(
                            f"Reducing along axis {self!r}: requested start {the_slice.start!r} between bin boundaries, no interpolation is performed",
                            RuntimeWarning,
                        )
                else:
                    if the_slice.start not in self._bins:
                        warnings.warn(
                            f"Reducing along axis {self!r}: requested start {the_slice.start!r} between bin boundaries, no interpolation is performed",
                            RuntimeWarning,
                        )
                    blo = self.index(the_slice.start).item()
            if the_slice.stop is not None:
                if the_slice.stop > self._hi:
                    raise ValueError(
                        f"Reducing along axis {self!r}: requested stop {the_slice.stop!r} exceeds bin boundaries (use open slicing, e.g. x[start:])"
                    )
                if self._uniform:
                    bhi_real = (the_slice.stop - self._lo) * self._bins / (
                        self._hi - self._lo
                    ) + 1
                    bhi = np.clip(
                        np.round(bhi_real).astype(int), 0, self._bins + 1
                    ).item()
                    if abs(bhi - bhi_real) > 1.0e-14:
                        warnings.warn(
                            f"Reducing along axis {self!r}: requested stop {the_slice.stop!r} between bin boundaries, no interpolation is performed",
                            RuntimeWarning,
                        )
                else:
                    if the_slice.stop not in self._bins:
                        warnings.warn(
                            f"Reducing along axis {self!r}: requested stop {the_slice.stop!r} between bin boundaries, no interpolation is performed",
                            RuntimeWarning,
                        )
                    bhi = self.index(the_slice.stop).item()
                # Assume null ranges (start==stop) mean we want the bin containing the value
                if blo is not None and blo == bhi:
                    bhi += 1
            if the_slice.step is not None:
                raise NotImplementedError(
                    "Step slicing can be interpreted as a rebin factor"
                )
            return slice(blo, bhi, the_slice.step)
        elif isinstance(the_slice, list) and all(
            isinstance(v, Interval) for v in the_slice
        ):
            raise NotImplementedError("Slice histogram from list of intervals")
        raise IndexError(f"Cannot understand slice {the_slice!r} on axis {self!r}")

    @property
    def size(self):
        """Number of bins"""
        return (
            self._bins
            if isinstance(self._bins, (int, np.integer, cupy.integer))
            else len(self._bins)
        )

    def edges(self, flow=False):
        """Bin boundaries

        Parameters
        ----------
            flow : bool
        """
        if self._uniform:
            out = cupy.linspace(self._lo, self._hi, self._bins + 1)
        else:
            out = self._bins[:-1].copy()
        out = cupy.r_[
            2 * out[0] - out[1], out, 2 * out[-1] - out[-2], 3 * out[-1] - 2 * out[-2]
        ]
        return out[_overflow_behavior(flow)]

    def centers(self, flow=False):
        """Bin centers

        Parameters
        ----------
            flow : bool
        """
        edges = self.edges(flow)
        return (edges[:-1] + edges[1:]) / 2

    def identifiers(self, flow=False):
        """List of `Interval` identifiers"""
        return self._intervals[_overflow_behavior(flow)]


class Regular(Bin):
    """
    Make a regular axis with nice keyword arguments for underflow,
    overflow, and growth.

    Parameters
    ----------
        bins : int
            The number of bins between start and stop.
        start : float
            The beginning value for the axis.
        stop : float
            The ending value for the axis.
        name : str
            Axis name.
        label : str
            Axis label.
    """

    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
    ) -> None:
        super().__init__(
            bins,
            start,
            stop,
            name=name,
            label=label,
        )

    def reduced(self, islice):
        """
        Return a new axis with reduced binning
        The new binning corresponds to the slice made on this axis.
        Overflow will be taken care of by ``Hist.__getitem__``

        Parameters
        ----------
            islice : slice
                ``islice.start`` and ``islice.stop`` should be None or within ``[1, ax.size() - 1]``
                This slice is usually as returned from ``Bin._ireduce``
        """
        if islice.step is not None:
            raise NotImplementedError(
                "Step slicing can be interpreted as a rebin factor"
            )
        if islice.start is None and islice.stop is None:
            return self
        lo = self._lo
        ilo = 0
        if islice.start is not None:
            lo += (islice.start - 1) * (self._hi - self._lo) / self._bins
            ilo = islice.start - 1
        hi = self._hi
        ihi = self._bins
        if islice.stop is not None:
            hi = self._lo + (islice.stop - 1) * (self._hi - self._lo) / self._bins
            ihi = islice.stop - 1
        bins = ihi - ilo
        # TODO: remove this once satisfied it works
        rbins = (hi - lo) * self._bins / (self._hi - self._lo)
        assert abs(bins - rbins) < 1e-14, "%d %f %r" % (bins, rbins, self)
        return Regular(bins, lo, hi, name=self._name, label=self._label)


class Variable(Bin):
    """
    Make an axis with irregularly spaced bins. Provide a list
    or array of bin edges, and len(edges)-1 bins will be made.

    Parameters
    ----------
        edges : Array[float]
            The edges for the bins. There will be one less bin than edges.
        name : str
            Axis name.
        label : str
            Axis label.
    """

    def __init__(
        self,
        edges: Iterable[float],
        *,
        name: str = "",
        label: str = "",
    ) -> None:
        super().__init__(edges, name=name, label=label)

    def reduced(self, islice):
        """
        Return a new axis with reduced binning
        The new binning corresponds to the slice made on this axis.
        Overflow will be taken care of by ``Hist.__getitem__``.

        Parameters
        ----------
            islice : slice
                ``islice.start`` and ``islice.stop`` should be None or within ``[1, ax.size() - 1]``
                This slice is usually as returned from ``Bin._ireduce``
        """
        if islice.step is not None:
            raise NotImplementedError(
                "Step slicing can be interpreted as a rebin factor"
            )
        if islice.start is None and islice.stop is None:
            return self
        lo = None if islice.start is None else islice.start - 1
        hi = -1 if islice.stop is None else islice.stop
        bins = self._bins[slice(lo, hi)]
        return Variable(bins, name=self._name, label=self._label)
