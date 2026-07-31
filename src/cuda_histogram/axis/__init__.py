from __future__ import annotations

import functools
import numbers
import warnings
from collections.abc import Iterable
from typing import Any

import awkward
import cupy
import numpy as np

__all__: list[str] = [
    "Bin",
    "Integer",
    "Interval",
    "Regular",
    "StrCategory",
    "Variable",
]

# bounds are float64 so integer input does not truncate them; NaN maps to nanflow
_clip_bins = cupy.ElementwiseKernel(
    "float64 nbins, float64 lo, float64 hi, T x",
    "int64 idx",
    """
    const double floored = floor((x - lo) * nbins / (hi - lo)) + 1;
    idx = isnan(floored)    ? (long long)nbins + 2
          : floored < 0     ? 0
          : floored > nbins ? (long long)nbins + 1
                            : (long long)floored;
    """,
    "clip_bins",
)


def _overflow_behavior(overflow: bool) -> slice:
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

    def __init__(self, lo: float, hi: float, label: str | None = None) -> None:
        self._lo = float(lo)
        self._hi = float(hi)
        self._label = label

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} ({self!s}) instance at 0x{id(self):0x}>"

    def __str__(self) -> str:
        if self._label is not None:
            return self._label
        if self.nan():
            return "(nanflow)"
        return f"{'(' if self._lo == -np.inf else '['}{self._lo}, {self._hi})"

    def __hash__(self) -> int:
        return hash((self._lo, self._hi))

    def __lt__(self, other: Interval) -> bool:
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return False
        if other.nan() and self.nan():
            return True
        return self._lo == other._lo and self._hi == other._hi

    def nan(self) -> bool:
        return bool(np.isnan(self._hi))

    @property
    def lo(self) -> float:
        """Lower boundary of this bin, inclusive"""
        return self._lo

    @property
    def hi(self) -> float:
        """Upper boundary of this bin, exclusive"""
        return self._hi

    @property
    def mid(self) -> float:
        """Midpoint of this bin"""
        return (self._hi + self._lo) / 2

    @property
    def label(self) -> str | None:
        """Label of this bin, mutable"""
        return self._label

    @label.setter
    def label(self, lbl: str | None) -> None:
        self._label = lbl


class Axis:
    """
    Axis: Base class for any type of axis
    Derived classes should implement, at least, an equality override
    """

    def __init__(self, name: str, label: str) -> None:
        self._name = name
        self._label = label

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} (name={self._name}) instance at 0x{id(self):0x}>"

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    @property
    def size(self) -> int:
        """Number of bins, not counting any flow bins"""
        raise NotImplementedError

    __hash__ = None  # type: ignore[assignment]  # mutable label, and __eq__ also accepts str

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Axis):
            if self._name != other._name:  # noqa: SIM103
                return False
            # label doesn't matter
            return True
        elif isinstance(other, str):
            # Convenient for testing axis in list by name
            return self._name == other
        raise TypeError(f"Cannot compare an Axis with a {other!r}")


class SparseAxis(Axis):
    """
    SparseAxis: ABC for a sparse axis

    Derived should implement:
        **index(identifier)** - return a bin index, or None to discard the fill

        **__eq__(axis)** - axis has same definition (not necessarily same bins)

        **__getitem__(index)** - return an identifier

        **_ireduce(slice)** - return a list of bin indices, slice is arbitrary

        **reduced(indices)** - return a new axis with only the given bins
    """

    def index(self, identifier: Any) -> int | None:
        """Bin index for an identifier; ``None`` means the fill is discarded"""
        raise NotImplementedError

    def _ireduce(self, the_slice: Any) -> list[int]:
        raise NotImplementedError

    def reduced(self, indices: list[int]) -> SparseAxis:
        raise NotImplementedError

    @property
    def extent(self) -> int:
        """Number of bins, including the overflow bin if present"""
        raise NotImplementedError

    @property
    def overflow(self) -> bool:
        """Whether unmatched fills are counted in a trailing overflow bin"""
        raise NotImplementedError


def _check_str(category: Any) -> None:
    if not isinstance(category, str):
        raise TypeError(
            f"StrCategory only supports string categories, received {category!r}"
        )


class StrCategory(SparseAxis):
    """A categorical axis with string-valued bins.

    Modeled on ``boost_histogram.axis.StrCategory``. Categories are kept in
    insertion order, and histogram storage is sparse: only filled categories
    occupy memory.

    Parameters
    ----------
        categories : Iterable[str]
            Initial categories, in bin order.
        name : str
            is used as a keyword in histogram filling, immutable
        label : str
            describes the meaning of the axis, can be changed
        growth : bool
            If True, filling a category not on the axis appends it; a growing
            axis matches everything, so it has no overflow bin.
        overflow : bool
            If True (and not growing), unmatched fills are counted in a
            trailing overflow bin; if False they are discarded.
    """

    def __init__(
        self,
        categories: Iterable[str] = (),
        *,
        name: str = "",
        label: str = "",
        growth: bool = False,
        overflow: bool = True,
    ) -> None:
        super().__init__(name, label)
        self._indices: dict[str, int] = {}
        self._growth = growth
        self._overflow = overflow and not growth
        for category in categories:
            self._add(category)

    def _add(self, category: str) -> int:
        _check_str(category)
        if category in self._indices:
            raise ValueError(f"Duplicate category {category!r}")
        index = len(self._indices)
        self._indices[category] = index
        return index

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.identifiers()})"

    @property
    def growth(self) -> bool:
        """Whether filling an unknown category appends it to the axis"""
        return self._growth

    @property
    def overflow(self) -> bool:
        """Whether unmatched fills are counted in a trailing overflow bin"""
        return self._overflow

    @property
    def size(self) -> int:
        """Number of categories, not counting the overflow bin"""
        return len(self._indices)

    @property
    def extent(self) -> int:
        """Number of categories, including the overflow bin if present"""
        return self.size + 1 if self._overflow else self.size

    def index(self, identifier: str) -> int | None:
        """Index of a category

        Parameters
        ----------
            identifier : str
                The category to look up.

        Returns the integer bin index of the category. An unknown category
        is appended if the axis was constructed with ``growth=True``;
        otherwise it maps to the overflow bin (index ``size``), or to
        ``None`` (meaning: discard) if ``overflow=False``.
        """
        _check_str(identifier)
        idx = self._indices.get(identifier)
        if idx is not None:
            return idx
        if self._growth:
            return self._add(identifier)
        return self.size if self._overflow else None

    def __getitem__(self, index: int) -> str:
        return self.identifiers()[index]

    def _ireduce(self, the_slice: Any) -> list[int]:
        if isinstance(the_slice, str):
            if the_slice not in self._indices:
                raise KeyError(f"No category {the_slice!r} in axis {self!r}")
            out = [the_slice]
        elif isinstance(the_slice, list):
            if not all(k in self._indices for k in the_slice):
                warnings.warn(
                    f"Not all requested categories present in {self!r}", RuntimeWarning
                )
            out = [k for k in the_slice if k in self._indices]
        elif the_slice == slice(None):
            out = self.identifiers()
        else:
            raise IndexError(f"Cannot understand slice {the_slice!r} on axis {self!r}")
        return [self._indices[k] for k in out]

    def reduced(self, indices: list[int]) -> StrCategory:
        """Return a new axis with only the categories at the given indices

        Parameters
        ----------
            indices : list[int]
                Category indices, usually as returned from ``StrCategory._ireduce``
        """
        categories = self.identifiers()
        return StrCategory(
            [categories[i] for i in indices],
            name=self._name,
            label=self._label,
            growth=self._growth,
            overflow=self._overflow,
        )

    def identifiers(self) -> list[str]:
        """List of string categories"""
        return list(self._indices)


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

    def index(self, identifier: Any) -> Any:
        raise NotImplementedError

    def _ireduce(self, the_slice: Any) -> slice:
        raise NotImplementedError

    def reduced(self, islice: slice) -> DenseAxis:
        raise NotImplementedError

    @property
    def underflow(self) -> bool:
        """Whether fills below the first bin edge are counted"""
        return True

    @property
    def overflow(self) -> bool:
        """Whether fills at or above the last bin edge are counted"""
        return True


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
        underflow : bool
            If False, fills below ``lo`` are discarded instead of counted.
        overflow : bool
            If False, fills at or above ``hi`` are discarded instead of
            counted.  NaN always lands in the nanflow bin, regardless.

    This axis will generate frequencies for n+3 bins, special bin indices:
    ``0 = underflow, n+1 = overflow, n+2 = nanflow``
    Bin boundaries are [lo, hi)

    The flow bins are always present in the dense storage layout (and thus in
    ``values(flow=True)``); a disabled flow bin simply stays empty.
    """

    def __init__(
        self,
        n_or_arr: Any,
        lo: float | None = None,
        hi: float | None = None,
        *,
        name: str = "",
        label: str = "",
        underflow: bool = True,
        overflow: bool = True,
    ) -> None:
        # _bins is the number of bins when uniform, else the array of edges
        self._bins: Any
        self._lo: Any
        self._hi: Any
        self._lazy_intervals: list[Interval] | None = None
        if isinstance(n_or_arr, list | np.ndarray | cupy.ndarray):
            self._uniform = False
            self._bins = cupy.array(n_or_arr, dtype="d")
            if not bool((self._bins[:-1] < self._bins[1:]).all()):
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
        else:
            raise TypeError(
                f"Expected an integer number of bins or an array of bin edges, got {n_or_arr!r}"
            )
        self._label = label
        self._name = name
        self._underflow = underflow
        self._overflow = overflow

    @property
    def underflow(self) -> bool:
        """Whether fills below the first bin edge are counted"""
        return self._underflow

    @property
    def overflow(self) -> bool:
        """Whether fills at or above the last bin edge are counted"""
        return self._overflow

    @property
    def _nanflow_index(self) -> int:
        """Dense index of the nanflow bin (``n+2``)"""
        return int(self._bins) + 2 if self._uniform else len(self._bins)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}({self._bins[:-1]})"
            if not self._uniform
            else f"{class_name}{self._bins, self._lo, self._hi}"
        )

    @property
    def _intervals(self) -> list[Interval]:
        if not hasattr(self, "_lazy_intervals") or self._lazy_intervals is None:
            self._lazy_intervals = [
                Interval(low, high, bin)
                for low, high, bin in zip(
                    self._interval_bins[:-1],
                    self._interval_bins[1:],
                    self._bin_names,
                    strict=False,
                )
            ]
        return self._lazy_intervals

    def __getstate__(self) -> dict[str, Any]:
        if hasattr(self, "_lazy_intervals") and self._lazy_intervals is not None:
            self._bin_names = np.array(
                [interval.label for interval in self._lazy_intervals]
            )
        self.__dict__.pop("_lazy_intervals", None)
        return self.__dict__

    def __setstate__(self, d: dict[str, Any]) -> None:
        if "_interval_bins" in d and "_bin_names" not in d:
            d["_bin_names"] = np.full(d["_interval_bins"][:-1].size, None)
        # pickles predating the flow toggles
        d.setdefault("_underflow", True)
        d.setdefault("_overflow", True)
        self.__dict__ = d

    def index(self, identifier: Any) -> Any:
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
            identifier, awkward.Array | cupy.ndarray | np.ndarray | list
        )
        if isarray or isinstance(identifier, numbers.Number):
            identifier = awkward.to_cupy(identifier)  # cupy.asarray(identifier)
            if self._uniform:
                return _clip_bins(self._bins, self._lo, self._hi, identifier)
            else:
                return cupy.searchsorted(self._bins, identifier, side="right")
        elif isinstance(identifier, Interval):
            if identifier.nan():
                return self._nanflow_index
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

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Bin):
            if not super().__eq__(other):
                return False
            if self._uniform != other._uniform:
                return False
            if (self._underflow, self._overflow) != (
                other._underflow,
                other._overflow,
            ):
                return False
            if self._uniform:
                return bool(self._bins == other._bins)
            return bool(all(self._bins == other._bins))
        return super().__eq__(other)

    def __getitem__(self, index: int) -> Interval:
        return self._intervals[index]

    def _ireduce(self, the_slice: Any) -> slice:
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
    def size(self) -> int:
        """Number of bins"""
        return (
            int(self._bins)
            if isinstance(self._bins, int | np.integer | cupy.integer)
            else len(self._bins)
        )

    def edges(self, flow: bool = False) -> Any:
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

    def centers(self, flow: bool = False) -> Any:
        """Bin centers

        Parameters
        ----------
            flow : bool
        """
        edges = self.edges(flow)
        return (edges[:-1] + edges[1:]) / 2

    def identifiers(self, flow: bool = False) -> list[Interval]:
        """List of `Interval` identifiers"""
        return self._intervals[_overflow_behavior(flow)]


class Regular(Bin):
    """
    Make a regular axis with uniform binning.

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
        underflow : bool
            If False, fills below ``start`` are discarded.
        overflow : bool
            If False, fills at or above ``stop`` are discarded (NaN still
            lands in the nanflow bin).
    """

    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        underflow: bool = True,
        overflow: bool = True,
    ) -> None:
        super().__init__(
            bins,
            start,
            stop,
            name=name,
            label=label,
            underflow=underflow,
            overflow=overflow,
        )

    def reduced(self, islice: slice) -> Regular:
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
        return Regular(
            bins,
            lo,
            hi,
            name=self._name,
            label=self._label,
            underflow=self._underflow,
            overflow=self._overflow,
        )


class Integer(Regular):
    """
    Make an axis with unit-width integer bins from start (inclusive)
    to stop (exclusive), modeled on ``boost_histogram.axis.Integer``.

    Parameters
    ----------
        start : int
            The first bin value, inclusive.
        stop : int
            One past the last bin value (there are ``stop - start`` bins).
        name : str
            Axis name.
        label : str
            Axis label.
        underflow : bool
            If False, fills below ``start`` are discarded.
        overflow : bool
            If False, fills at or above ``stop`` are discarded (NaN still
            lands in the nanflow bin).

    Float fills bin by floor: any value in ``[v, v + 1)`` lands in the
    bin for integer ``v``.
    """

    def __init__(
        self,
        start: int,
        stop: int,
        *,
        name: str = "",
        label: str = "",
        underflow: bool = True,
        overflow: bool = True,
    ) -> None:
        start = int(start)
        stop = int(stop)
        if stop <= start:
            raise ValueError(f"stop ({stop}) must be greater than start ({start})")
        super().__init__(
            stop - start,
            start,
            stop,
            name=name,
            label=label,
            underflow=underflow,
            overflow=overflow,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._lo}, {self._hi})"

    def reduced(self, islice: slice) -> Integer:
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
        reduced = super().reduced(islice)
        if reduced is self:
            return self
        return Integer(
            int(reduced._lo),
            int(reduced._hi),
            name=self._name,
            label=self._label,
            underflow=self._underflow,
            overflow=self._overflow,
        )


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
        underflow : bool
            If False, fills below the first edge are discarded.
        overflow : bool
            If False, fills at or above the last edge are discarded (NaN
            still lands in the nanflow bin).
    """

    def __init__(
        self,
        edges: Iterable[float],
        *,
        name: str = "",
        label: str = "",
        underflow: bool = True,
        overflow: bool = True,
    ) -> None:
        super().__init__(
            edges, name=name, label=label, underflow=underflow, overflow=overflow
        )

    def reduced(self, islice: slice) -> Variable:
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
        return Variable(
            bins,
            name=self._name,
            label=self._label,
            underflow=self._underflow,
            overflow=self._overflow,
        )
