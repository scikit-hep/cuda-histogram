from __future__ import annotations

import functools
import re
import warnings
from typing import Any, Iterable

import boost_histogram.axis as bha
import cupy
import hist
import numpy as np

__all__: list[str] = [
    "Bin",
    "Regular",
    "Variable",
    "Cat",
    "Interval",
    "StringBin",
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


def _overflow_behavior(flow):
    if not flow:
        return slice(1, -1)
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


@functools.total_ordering
class StringBin:
    """A string used to fill a sparse axis

    Totally ordered, lexicographically by name.

    Parameters
    ----------
        name : str
            Name of the bin, as used in `Hist.fill` calls
        label : str
            The `str` representation of this bin can be overridden by
            a custom label.
    """

    def __init__(self, name, label=None):
        if not isinstance(name, str):
            raise TypeError(
                f"StringBin only supports string categories, received a {name!r}"
            )
        elif "*" in name:
            raise ValueError(
                "StringBin does not support character '*' as it conflicts with wildcard mapping."
            )
        self._name = name
        self._label = label

    def __repr__(self):
        return f"<{self.__class__.__name__} ({self.name}) instance at 0x{id(self):0x}>"

    def __str__(self):
        if self._label is not None:
            return self._label
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __lt__(self, other):
        return self._name < other._name

    def __eq__(self, other):
        if isinstance(other, StringBin):
            return self._name == other._name
        return False

    @property
    def name(self):
        """Name of this bin, *Immutable*"""
        return self._name

    @property
    def label(self):
        """Label of this bin, mutable"""
        return self._label

    @label.setter
    def label(self, lbl):
        self._label = lbl


class Axis:
    """
    Axis: Base class for any type of axis
    Derived classes should implement, at least, an equality override
    """

    def __init__(self, name, label):
        if name == "weight":
            raise ValueError(
                "Cannot create axis: 'weight' is a reserved keyword for Hist.fill()"
            )
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


class Cat(SparseAxis):
    """A category axis with name and label

    Parameters
    ----------
        name : str
            is used as a keyword in histogram filling, immutable
        label : str
            describes the meaning of the axis, can be changed
        sorting : {'identifier', 'placement', 'integral'}, optional
            Axis sorting when listing identifiers.

    The number of categories is arbitrary, and can be filled sparsely
    Identifiers are strings
    """

    def __init__(self, name, label, sorting="identifier"):
        super().__init__(name, label)
        # In all cases key == value.name
        self._bins = {}
        self._sorting = sorting
        self._sorted = []

    def index(self, identifier):
        """Index of a identifier or label

        Parameters
        ----------
            identifier : str or StringBin
                The identifier to lookup

        Returns a `StringBin` corresponding to the given argument (trivial in the case
        where a `StringBin` was passed) and saves a reference internally in the case where
        the identifier was not seen before by this axis.
        """
        if isinstance(identifier, StringBin):
            index = identifier
        else:
            index = StringBin(identifier)
        if index.name not in self._bins:
            self._bins[index.name] = index
            self._sorted.append(index.name)
            if self._sorting == "identifier":
                self._sorted.sort()
        return self._bins[index.name]

    def __eq__(self, other):
        # Sparse, so as long as name is the same
        return super().__eq__(other)

    def __getitem__(self, index):
        if not isinstance(index, StringBin):
            raise TypeError(f"Expected a StringBin object, got: {index!r}")
        identifier = index.name
        if identifier not in self._bins:
            raise KeyError("No identifier %r in this Category axis")
        return identifier

    def _ireduce(self, the_slice):
        out = None
        if isinstance(the_slice, StringBin):
            out = [the_slice.name]
        elif isinstance(the_slice, re.Pattern):
            out = [k for k in self._sorted if the_slice.match(k)]
        elif isinstance(the_slice, str):
            pattern = "^" + re.escape(the_slice).replace(r"\*", ".*") + "$"
            m = re.compile(pattern)
            out = [k for k in self._sorted if m.match(k)]
        elif isinstance(the_slice, list):
            if not all(k in self._sorted for k in the_slice):
                warnings.warn(
                    f"Not all requested indices present in {self!r}", RuntimeWarning
                )
            out = [k for k in self._sorted if k in the_slice]
        elif isinstance(the_slice, slice):
            if the_slice.step is not None:
                raise IndexError("Not sure how to use slice step for categories...")
            start, stop = 0, len(self._sorted)
            if isinstance(the_slice.start, str):
                start = self._sorted.index(the_slice.start)
            else:
                start = the_slice.start
            if isinstance(the_slice.stop, str):
                stop = self._sorted.index(the_slice.stop)
            else:
                stop = the_slice.stop
            out = self._sorted[start:stop]
        else:
            raise IndexError(f"Cannot understand slice {the_slice!r} on axis {self!r}")
        return [self._bins[k] for k in out]

    @property
    def size(self):
        """Number of bins"""
        return len(self._bins)

    @property
    def sorting(self):
        """Sorting definition to adhere to

        See `Cat` constructor for possible values
        """
        return self._sorting

    @sorting.setter
    def sorting(self, newsorting):
        if newsorting == "placement":
            # not much we can do about already inserted values
            pass
        elif newsorting == "identifier":
            self._sorted.sort()
        elif newsorting == "integral":
            # this will be checked in any Hist.identifiers() call accessing this axis
            pass
        else:
            raise AttributeError(f"Invalid axis sorting type: {newsorting}")
        self._sorting = newsorting

    def identifiers(self):
        """List of `StringBin` identifiers"""
        return [self._bins[k] for k in self._sorted]


class Regular(hist.axis.Regular, family="cuda_histogram"):
    """A binned axis with name, label, and binning.

    Parameters
    ----------
        name : str
            is used as a keyword in histogram filling, immutable
        label : str
            describes the meaning of the axis, can be changed
        n_or_arr : int or list or np.ndarray
            Integer number of bins, if uniform binning. Otherwise, a list or
            numpy 1D array of bin boundaries.
        lo : float, optional
            lower boundary of bin range, if uniform binning
        hi : float, optional
            upper boundary of bin range, if uniform binning

    This axis will generate frequencies for n+3 bins, special bin indices:
    ``0 = underflow, n+1 = overflow, n+2 = nanflow``
    Bin boundaries are [lo, hi)
    """

    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        flow: bool = True,
        underflow: bool | None = None,
        overflow: bool | None = None,
        growth: bool = False,
        circular: bool = False,
        # pylint: disable-next=redefined-outer-name
        transform: bha.transform.AxisTransform | None = None,
        __dict__: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            bins,
            start,
            stop,
            metadata=metadata,
            underflow=flow if underflow is None else underflow,
            overflow=flow if overflow is None else overflow,
            growth=growth,
            circular=circular,
            transform=transform,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label


class Variable(hist.axis.Variable, family="cuda_histogram"):
    """A binned axis with name, label, and binning.

    Parameters
    ----------
        name : str
            is used as a keyword in histogram filling, immutable
        label : str
            describes the meaning of the axis, can be changed
        n_or_arr : int or list or np.ndarray
            Integer number of bins, if uniform binning. Otherwise, a list or
            numpy 1D array of bin boundaries.
        lo : float, optional
            lower boundary of bin range, if uniform binning
        hi : float, optional
            upper boundary of bin range, if uniform binning

    This axis will generate frequencies for n+3 bins, special bin indices:
    ``0 = underflow, n+1 = overflow, n+2 = nanflow``
    Bin boundaries are [lo, hi)
    """

    def __init__(
        self,
        edges: Iterable[float],
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        flow: bool = True,
        underflow: bool | None = None,
        overflow: bool | None = None,
        growth: bool = False,
        circular: bool = False,
        __dict__: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            edges,
            metadata=metadata,
            underflow=flow if underflow is None else underflow,
            overflow=flow if overflow is None else overflow,
            growth=growth,
            circular=circular,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label
