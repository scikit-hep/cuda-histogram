from __future__ import annotations

from typing import Any

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import cuda_histogram

# cupy might be installable on a device with no GPUs
try:
    cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA not found", allow_module_level=True)


def dummy_jagged_eta_pt() -> tuple[Any, Any, Any]:
    np.random.seed(42)
    counts = np.random.exponential(2, size=50).astype(int)
    entries = cp.sum(counts)
    test_eta = cp.random.uniform(-3.0, 3.0, size=entries)
    test_pt = cp.random.exponential(10.0, size=entries) + cp.random.exponential(
        10, size=entries
    )
    return (counts, test_eta, test_pt)


def test_hist() -> None:
    _counts, test_eta, test_pt = dummy_jagged_eta_pt()

    h = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(20, 0, 2, label="x", name="x"),
        cuda_histogram.axis.Variable([0, 5, 15, 30], label="y", name="why"),
        name="regular joe",
        label="joe",
    )
    assert (h.values() == cp.zeros((20, 3))).all()
    assert h.variance() is None
    assert h.variance(flow=True) is None
    assert h.values().shape == (20, 3)
    assert h.values(flow=True).shape == (23, 6)

    h.fill(test_eta, test_pt)

    assert not (h.values() == cp.zeros((20, 3))).all()
    assert isinstance(h.values(), cp.ndarray)
    assert h.values().shape == (20, 3)
    assert h.values(flow=True).shape == (23, 6)
    assert h.variance() is None
    assert h.variance(flow=True) is None
    assert h.__repr__() == "Hist(Regular(20, 0, 2), Variable([ 0.  5. 15. 30.]))"
    assert list(h._sumw.keys()) == [()]
    assert (next(iter(h._sumw.values())) == h.values(flow=True)).all()
    assert h._sumw2 is None
    assert h.name == "regular joe"
    assert h.label == "joe"
    assert h[:, :].axes() == h.axes()
    assert isinstance(h[0, :], cuda_histogram.Hist)
    assert isinstance(h[0, :].values(), cp.ndarray)
    assert (h[0, :].values() == h.values()[0, :]).all()
    assert isinstance(h[0, :], cuda_histogram.Hist)
    assert isinstance(h[0, :].values(), cp.ndarray)
    assert (h[0, :].values() == h.values()[0]).all()
    assert isinstance(h[:, 0], cuda_histogram.Hist)
    assert isinstance(h[:, 0].values(), cp.ndarray)
    assert (h[:, 0].values() == h.values()[:, 0].reshape((20, 1))).all()

    h = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(20, 0, 2, label="x", name="x"),
        cuda_histogram.axis.Variable([0, 5, 15, 30], label="y", name="why"),
        name="regular joe",
        label="joe",
    )
    assert (h.values() == cp.zeros((20, 3))).all()
    assert h.values().shape == (20, 3)
    assert h.values(flow=True).shape == (23, 6)
    assert h.variance() is None
    assert h.variance(flow=True) is None

    h.fill(test_eta, test_pt, weight=test_eta)

    assert not (h.values() == cp.zeros((20, 3))).all()
    assert isinstance(h.values(), cp.ndarray)
    assert isinstance(h.variance(), cp.ndarray)
    assert h.values().shape == (20, 3)
    assert h.values(flow=True).shape == (23, 6)
    assert h.variance().shape == (20, 3)
    assert h.variance(flow=True).shape == (23, 6)
    assert h.__repr__() == "Hist(Regular(20, 0, 2), Variable([ 0.  5. 15. 30.]))"
    assert list(h._sumw.keys()) == [()]
    assert (next(iter(h._sumw.values())) == h.values(flow=True)).all()
    assert h._sumw2 is not None
    assert list(h._sumw2.keys()) == [()]
    assert (next(iter(h._sumw2.values())) == h.variance(flow=True)).all()
    assert h.name == "regular joe"
    assert h.label == "joe"
    assert h[:, :].axes() == h.axes()
    assert isinstance(h[0, :], cuda_histogram.Hist)
    assert isinstance(h[0, :].values(), cp.ndarray)
    assert isinstance(h[0, :].variance(), cp.ndarray)
    assert (h[0, :].values() == h.values()[0, :]).all()
    assert (h[0, :].variance() == h.variance()[0, :]).all()
    assert isinstance(h[0, :], cuda_histogram.Hist)
    assert isinstance(h[0, :].values(), cp.ndarray)
    assert isinstance(h[0, :].variance(), cp.ndarray)
    assert (h[0, :].values() == h.values()[0]).all()
    assert (h[0, :].variance() == h.variance()[0]).all()
    assert isinstance(h[:, 0], cuda_histogram.Hist)
    assert isinstance(h[:, 0].values(), cp.ndarray)
    assert isinstance(h[:, 0].variance(), cp.ndarray)
    assert (h[:, 0].values() == h.values()[:, 0].reshape((20, 1))).all()
    assert (h[:, 0].variance() == h.variance()[:, 0].reshape((20, 1))).all()


def test_partial_indexing() -> None:
    _counts, test_eta, test_pt = dummy_jagged_eta_pt()

    h = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(20, 0, 2, label="x", name="x"),
        cuda_histogram.axis.Variable([0, 5, 15, 30], label="y", name="why"),
    )
    h.fill(test_eta, test_pt)

    assert (h[0.1].values() == h[0.1, :].values()).all()
    assert (h[...].values() == h.values()).all()
    assert h[...].axes() == h.axes()
    assert (h[0.1, ...].values() == h[0.1, :].values()).all()
    assert (h[..., 3].values() == h[:, 3].values()).all()

    with pytest.raises(IndexError):
        h[0.1, 3, 5]
    with pytest.raises(IndexError):
        h[..., ...]
    # strings are category selections, meaningless on a dense axis
    with pytest.raises(IndexError):
        h["x"]


def test_underflow_index_and_fill() -> None:
    ax = cuda_histogram.axis.Regular(10, 0, 1)
    assert int(ax.index(cp.array([-5.0]))[0]) == 0
    assert int(ax.index(cp.array([5.0]))[0]) == 11

    h = cuda_histogram.Hist(cuda_histogram.axis.Regular(10, 0, 1, name="x"))
    h.fill(cp.array([-5.0]))
    values = h.values(flow=True)
    assert int(values[0]) == 1
    assert int(values.sum()) == 1


def test_nanflow_index_and_fill() -> None:
    ax = cuda_histogram.axis.Regular(10, 0, 1)
    assert int(ax.index(cp.array([np.nan]))[0]) == 12
    assert int(cp.asarray(ax.index(float("nan"))).reshape(-1)[0]) == 12
    assert ax.index(cuda_histogram.axis.Interval(3.0, np.nan)) == 12

    axv = cuda_histogram.axis.Variable([0, 1, 2, 3])
    assert int(axv.index(cp.array([np.nan]))[0]) == 5
    assert axv.index(cuda_histogram.axis.Interval(3.0, np.nan)) == 5

    h = cuda_histogram.Hist(cuda_histogram.axis.Regular(10, 0, 1, name="x"))
    h.fill(cp.array([np.nan]))
    values = h.values(flow=True)
    assert int(values[-1]) == 1
    assert int(values.sum()) == 1


def test_bin_validation() -> None:
    with pytest.raises(TypeError):
        cuda_histogram.axis.Bin(3.5, 0, 1)
    with pytest.raises(TypeError):
        cuda_histogram.axis.Regular(3.5, 0, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not sorted"):
        cuda_histogram.axis.Variable([0, 2, 1])
    with pytest.raises(ValueError, match="not sorted"):
        cuda_histogram.axis.Variable([0, 1, 1, 2])


@pytest.mark.parametrize("weighted_first", [False, True])
def test_mixed_weighted_unweighted_fill(weighted_first: bool) -> None:
    x = cp.array([0.15, 0.25, 0.25, 0.35])
    w = cp.array([2.0, 3.0, 4.0, 5.0])

    h = cuda_histogram.Hist(cuda_histogram.axis.Regular(4, 0, 1, name="x"))
    if weighted_first:
        h.fill(x, weight=w)
        assert (h.variance() == cp.array([4.0, 50.0, 0.0, 0.0])).all()
        h.fill(x)
    else:
        h.fill(x)
        assert h.variance() is None
        h.fill(x, weight=w)

    # an unweighted entry contributes 1 to the variance
    assert (h.values() == cp.array([3.0, 15.0, 0.0, 0.0])).all()
    assert (h.variance() == cp.array([5.0, 53.0, 0.0, 0.0])).all()


def test_fill_integer_array() -> None:
    h = cuda_histogram.Hist(cuda_histogram.axis.Regular(4, 0, 8, name="x"))
    h.fill(cp.array([1, 3, 3, 5]))
    assert (h.values() == cp.array([1.0, 2.0, 1.0, 0.0])).all()


def test_fill_integer_array_fractional_bounds() -> None:
    ax = cuda_histogram.axis.Regular(4, 0.5, 8.5)
    assert (ax.index(cp.array([0, 1, 8, 9])) == cp.array([0, 1, 4, 5])).all()
    assert (
        ax.index(cp.array([0, 1, 8, 9])) == ax.index(cp.array([0.0, 1.0, 8.0, 9.0]))
    ).all()

    h = cuda_histogram.Hist(cuda_histogram.axis.Regular(4, 0.5, 8.5, name="x"))
    h.fill(cp.array([0, 1, 8, 9]))
    assert (h.values(flow=True) == cp.array([1.0, 1, 0, 0, 1, 1, 0])).all()


def test_cpu_conversion() -> None:
    import boost_histogram as bh

    dummy = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(1, 0, 1, name="dummy", label="Number of events"),
        cuda_histogram.axis.Variable([0, 2, 3, 10]),
        name="Dummy",
        label="Just Dummy",
    )
    dummy.fill(cp.array(0.05), cp.array(0.05))
    dummy.fill(cp.array(-0.05), cp.array(-0.05))

    h = dummy.to_boost()
    assert len(h.axes) == 2
    assert isinstance(h.axes[0], bh.axis.Regular)
    assert isinstance(h.axes[1], bh.axis.Variable)
    assert h.axes[0].name == "dummy"
    assert h.axes[0].label == "Number of events"
    assert h.label == "Just Dummy"
    assert h[0, 0] == 1.0
    assert h.values().shape == dummy.values().shape
    assert h[bh.underflow, bh.underflow] == 1.0
    assert h.storage_type == bh.storage.Double

    dummy = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(1, 0, 1, name="dummy", label="Number of events"),
        cuda_histogram.axis.Variable([0, 2, 3, 10]),
        name="Dummy",
        label="Just Dummy",
    )
    dummy.fill(cp.array(0.05), cp.array(0.05), weight=cp.array([2]))
    dummy.fill(cp.array(-0.05), cp.array(-0.05), weight=cp.array([2]))

    h = dummy.to_boost()
    assert len(h.axes) == 2
    assert isinstance(h.axes[0], bh.axis.Regular)
    assert isinstance(h.axes[1], bh.axis.Variable)
    assert h.axes[0].name == "dummy"
    assert h.axes[0].label == "Number of events"
    assert h.label == "Just Dummy"
    assert h[0, 0].value == 2.0  # type: ignore[union-attr]
    assert h[0, 0].variance == 4.0  # type: ignore[union-attr]
    assert h.values().shape == dummy.values().shape
    assert h[bh.underflow, bh.underflow].variance == 4.0  # type: ignore[union-attr]
    assert h.storage_type == bh.storage.Weight


def _cat_axis(h: cuda_histogram.Hist, i: int = 0) -> cuda_histogram.axis.StrCategory:
    ax = h.axes()[i]
    assert isinstance(ax, cuda_histogram.axis.StrCategory)
    return ax


def test_str_category_axis() -> None:
    ax = cuda_histogram.axis.StrCategory(["one", "two"], name="s", label="sample")
    assert ax.size == 2
    assert ax.name == "s"
    assert ax.label == "sample"
    assert not ax.growth
    assert ax.index("one") == 0
    assert ax.index("two") == 1
    assert ax[0] == "one"
    assert ax[-1] == "two"
    assert ax.identifiers() == ["one", "two"]
    assert repr(ax) == "StrCategory(['one', 'two'])"
    assert ax.overflow
    assert ax.index("three") == 2  # overflow bin
    assert ax.size == 2
    with pytest.raises(TypeError):
        ax.index(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Duplicate"):
        cuda_histogram.axis.StrCategory(["a", "a"])

    no_overflow = cuda_histogram.axis.StrCategory(["one", "two"], overflow=False)
    assert not no_overflow.overflow
    assert no_overflow.index("three") is None

    grow = cuda_histogram.axis.StrCategory(growth=True)
    assert not grow.overflow  # a growing axis matches everything
    assert grow.size == 0
    assert grow.index("b") == 0
    assert grow.index("a") == 1
    assert grow.index("b") == 0
    assert grow.identifiers() == ["b", "a"]


def test_str_category_fill() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.StrCategory(growth=True, name="sample"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    assert h.values().shape == (0, 4)
    assert h.dense_dim() == 1
    assert h.sparse_axes() == [h.axes()[0]]

    h.fill("ttbar", cp.array([0.1, 0.3, 0.3]))
    h.fill("qcd", cp.array([0.6]))
    assert h.values().shape == (2, 4)
    assert (h.values() == cp.array([[1, 2, 0, 0], [0, 0, 1, 0]])).all()
    assert h.values(flow=True).shape == (2, 7)
    assert (h.values(flow=True).sum(axis=1) == cp.array([3, 1])).all()
    assert h.variance() is None

    h.fill("qcd", cp.array([0.6]), weight=cp.array([2.0]))
    assert h.variance() is not None
    assert (h.values() == cp.array([[1, 2, 0, 0], [0, 0, 3, 0]])).all()
    assert (h.variance() == cp.array([[1, 2, 0, 0], [0, 0, 5, 0]])).all()

    # unknown category on a non-growing axis lands in the overflow bin
    fixed = cuda_histogram.Hist(
        cuda_histogram.axis.StrCategory(["a", "b"], name="sample"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    assert fixed.values().shape == (2, 4)
    assert fixed.values(flow=True).shape == (3, 7)
    fixed.fill("b", cp.array([0.1]))
    fixed.fill("c", cp.array([0.1]))
    assert (fixed.values() == cp.array([[0, 0, 0, 0], [1, 0, 0, 0]])).all()
    assert (fixed.values(flow=True)[2] == cp.array([0, 1, 0, 0, 0, 0, 0])).all()

    # with overflow off, unknown categories are discarded
    discard = cuda_histogram.Hist(
        cuda_histogram.axis.StrCategory(["a", "b"], overflow=False, name="sample"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    assert discard.values(flow=True).shape == (2, 7)
    discard.fill("c", cp.array([0.1]), weight=cp.array([2.0]))
    assert discard.values(flow=True).sum() == 0
    assert discard.variance() is None  # the discarded fill never seeded sumw2
    discard.fill("a", cp.array([0.1]))
    assert (discard.values() == cp.array([[1, 0, 0, 0], [0, 0, 0, 0]])).all()


def test_str_category_only() -> None:
    h = cuda_histogram.Hist(cuda_histogram.axis.StrCategory(growth=True, name="s"))
    h.fill("a")
    h.fill("a")
    h.fill("b", weight=cp.array([3.0]))
    assert (h.values() == cp.array([2.0, 3.0])).all()
    assert (h.variance() == cp.array([2.0, 9.0])).all()

    sel = h["b"]
    assert _cat_axis(sel).identifiers() == ["b"]
    assert (sel.values() == cp.array([3.0])).all()


def test_str_category_getitem() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.StrCategory(growth=True, name="sample"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    h.fill("ttbar", cp.array([0.1, 0.3, 0.3]))
    h.fill("qcd", cp.array([0.6]), weight=cp.array([2.0]))
    h.fill("zjets", cp.array([0.9, 0.9]))

    sel = h["qcd"]
    assert isinstance(sel, cuda_histogram.Hist)
    assert _cat_axis(sel).identifiers() == ["qcd"]
    assert sel.values().shape == (1, 4)
    assert (sel.values()[0] == h.values()[1]).all()
    assert (sel.variance()[0] == h.variance()[1]).all()

    # list selection keeps the requested order
    sub = h[["zjets", "ttbar"], :]
    assert _cat_axis(sub).identifiers() == ["zjets", "ttbar"]
    assert (sub.values() == cp.stack([h.values()[2], h.values()[0]])).all()

    # a full slice keeps everything; dense slicing works alongside
    assert (h[:, :].values() == h.values()).all()
    assert (h["ttbar", 0.3].values()[0, 0] == h.values()[0, 1]).all()

    with pytest.raises(KeyError):
        h["nope"]
    with pytest.warns(RuntimeWarning):
        empty = h[["nope"], :]
    assert empty.values().shape == (0, 4)
    with pytest.raises(IndexError):
        h[0.5, :]


def test_str_category_overflow_slicing_and_convert() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.StrCategory(["a", "b"], name="s"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    h.fill("a", cp.array([0.1, 0.3]))
    h.fill("b", cp.array([0.5]))
    h.fill("zzz", cp.array([0.7, 0.7, 0.7]))
    assert h.values().sum() == 3
    assert h.values(flow=True).sum() == 6

    # overflow content survives selection, like boost-histogram picks
    pick = h["b"]
    assert pick.values(flow=True).shape == (2, 7)
    assert (pick.values(flow=True)[1] == h.values(flow=True)[2]).all()
    assert pick.values(flow=True)[1].sum() == 3

    out = h.to_boost()
    assert not out.axes[0].traits.growth
    assert out.axes[0].traits.overflow
    assert out.axes[0].extent == 3
    assert (out.values() == h.values().get()).all()
    # the overflow bin round-trips (drop only the dense nanflow on our side)
    assert (out.view(flow=True) == h.values(flow=True)[:, :-1].get()).all()

    no_overflow = cuda_histogram.Hist(
        cuda_histogram.axis.StrCategory(["a", "b"], overflow=False, name="s"),
    )
    no_overflow.fill("a")
    out2 = no_overflow.to_boost()
    assert not out2.axes[0].traits.overflow
    assert out2.axes[0].extent == 2
    assert (out2.values() == no_overflow.values().get()).all()


def test_str_category_axis_order() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(3, 0, 1, name="x"),
        cuda_histogram.axis.StrCategory(growth=True, name="s"),
    )
    h.fill(cp.array([0.1, 0.5]), "a")
    h.fill(cp.array([0.9]), "b")
    assert h.values().shape == (3, 2)
    assert (h.values()[:, 0] == cp.array([1, 1, 0])).all()
    assert (h.values()[:, 1] == cp.array([0, 0, 1])).all()

    out = h.to_boost()
    assert (out.values() == h.values().get()).all()


def test_str_category_to_boost() -> None:
    import boost_histogram as bh

    h = cuda_histogram.Hist(
        cuda_histogram.axis.StrCategory(growth=True, name="dataset", label="Sample"),
        cuda_histogram.axis.Regular(3, 0, 1, name="x"),
        label="events",
    )
    h.fill("ttbar", cp.array([0.1, 0.5, 1.5]))
    h.fill("qcd", cp.array([-0.5, 0.9]))

    out = h.to_boost()
    assert isinstance(out.axes[0], bh.axis.StrCategory)
    assert list(out.axes[0]) == ["ttbar", "qcd"]
    assert out.axes[0].name == "dataset"
    assert out.axes[0].label == "Sample"
    assert out.storage_type == bh.storage.Double
    assert (out.values() == h.values().get()).all()
    assert out[bh.loc("ttbar"), bh.overflow] == 1.0
    assert out[bh.loc("qcd"), bh.underflow] == 1.0

    h.fill("qcd", cp.array([0.9]), weight=cp.array([2.0]))
    out = h.to_boost()
    assert out.storage_type == bh.storage.Weight
    assert (out.values() == h.values().get()).all()
    assert (out.variances() == h.variance().get()).all()

    hh = h.to_hist()
    assert (hh.values() == h.values().get()).all()


def _int_cat_axis(
    h: cuda_histogram.Hist, i: int = 0
) -> cuda_histogram.axis.IntCategory:
    ax = h.axes()[i]
    assert isinstance(ax, cuda_histogram.axis.IntCategory)
    return ax


def test_int_category_axis() -> None:
    ax = cuda_histogram.axis.IntCategory([4, 7], name="i", label="ids")
    assert ax.size == 2
    assert ax.name == "i"
    assert ax.label == "ids"
    assert not ax.growth
    assert ax.index(4) == 0
    assert ax.index(7) == 1
    assert ax[0] == 4
    assert ax[-1] == 7
    assert ax.identifiers() == [4, 7]
    assert repr(ax) == "IntCategory([4, 7])"
    assert ax.overflow
    assert ax.index(3) == 2  # overflow bin
    assert ax.size == 2
    with pytest.raises(TypeError):
        ax.index("4")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cuda_histogram.axis.IntCategory(["a"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Duplicate"):
        cuda_histogram.axis.IntCategory([1, 1])

    no_overflow = cuda_histogram.axis.IntCategory([4, 7], overflow=False)
    assert not no_overflow.overflow
    assert no_overflow.index(3) is None

    grow = cuda_histogram.axis.IntCategory(growth=True)
    assert not grow.overflow  # a growing axis matches everything
    assert grow.size == 0
    assert grow.index(7) == 0
    assert grow.index(-2) == 1
    assert grow.index(7) == 0
    assert grow.identifiers() == [7, -2]


def test_int_category_fill() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.IntCategory(growth=True, name="pdgid"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    assert h.values().shape == (0, 4)
    assert h.dense_dim() == 1
    assert h.sparse_axes() == [h.axes()[0]]

    h.fill(211, cp.array([0.1, 0.3, 0.3]))
    h.fill(13, cp.array([0.6]))
    assert h.values().shape == (2, 4)
    assert (h.values() == cp.array([[1, 2, 0, 0], [0, 0, 1, 0]])).all()
    assert h.values(flow=True).shape == (2, 7)
    assert (h.values(flow=True).sum(axis=1) == cp.array([3, 1])).all()
    assert h.variance() is None

    h.fill(13, cp.array([0.6]), weight=cp.array([2.0]))
    assert h.variance() is not None
    assert (h.values() == cp.array([[1, 2, 0, 0], [0, 0, 3, 0]])).all()
    assert (h.variance() == cp.array([[1, 2, 0, 0], [0, 0, 5, 0]])).all()

    # unknown category on a non-growing axis lands in the overflow bin
    fixed = cuda_histogram.Hist(
        cuda_histogram.axis.IntCategory([1, 2], name="pdgid"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    assert fixed.values().shape == (2, 4)
    assert fixed.values(flow=True).shape == (3, 7)
    fixed.fill(2, cp.array([0.1]))
    fixed.fill(5, cp.array([0.1]))
    assert (fixed.values() == cp.array([[0, 0, 0, 0], [1, 0, 0, 0]])).all()
    assert (fixed.values(flow=True)[2] == cp.array([0, 1, 0, 0, 0, 0, 0])).all()

    # with overflow off, unknown categories are discarded
    discard = cuda_histogram.Hist(
        cuda_histogram.axis.IntCategory([1, 2], overflow=False, name="pdgid"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    assert discard.values(flow=True).shape == (2, 7)
    discard.fill(5, cp.array([0.1]), weight=cp.array([2.0]))
    assert discard.values(flow=True).sum() == 0
    assert discard.variance() is None  # the discarded fill never seeded sumw2
    discard.fill(1, cp.array([0.1]))
    assert (discard.values() == cp.array([[1, 0, 0, 0], [0, 0, 0, 0]])).all()


def test_int_category_getitem() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.IntCategory(growth=True, name="pdgid"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    h.fill(211, cp.array([0.1, 0.3, 0.3]))
    h.fill(13, cp.array([0.6]), weight=cp.array([2.0]))
    h.fill(-11, cp.array([0.9, 0.9]))

    sel = h[13]
    assert isinstance(sel, cuda_histogram.Hist)
    assert _int_cat_axis(sel).identifiers() == [13]
    assert sel.values().shape == (1, 4)
    assert (sel.values()[0] == h.values()[1]).all()
    assert (sel.variance()[0] == h.variance()[1]).all()

    # list selection keeps the requested order
    sub = h[[-11, 211], :]
    assert _int_cat_axis(sub).identifiers() == [-11, 211]
    assert (sub.values() == cp.stack([h.values()[2], h.values()[0]])).all()

    assert (h[:, :].values() == h.values()).all()
    assert (h[211, 0.3].values()[0, 0] == h.values()[0, 1]).all()

    with pytest.raises(KeyError):
        h[999]
    with pytest.warns(RuntimeWarning):
        empty = h[[999], :]
    assert empty.values().shape == (0, 4)


def test_int_category_overflow_slicing_and_convert() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.IntCategory([1, 2], name="i"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    h.fill(1, cp.array([0.1, 0.3]))
    h.fill(2, cp.array([0.5]))
    h.fill(99, cp.array([0.7, 0.7, 0.7]))
    assert h.values().sum() == 3
    assert h.values(flow=True).sum() == 6

    # overflow content survives selection, like boost-histogram picks
    pick = h[2]
    assert pick.values(flow=True).shape == (2, 7)
    assert (pick.values(flow=True)[1] == h.values(flow=True)[2]).all()
    assert pick.values(flow=True)[1].sum() == 3

    out = h.to_boost()
    assert not out.axes[0].traits.growth
    assert out.axes[0].traits.overflow
    assert out.axes[0].extent == 3
    assert (out.values() == h.values().get()).all()
    # the overflow bin round-trips (drop only the dense nanflow on our side)
    assert (out.view(flow=True) == h.values(flow=True)[:, :-1].get()).all()

    no_overflow = cuda_histogram.Hist(
        cuda_histogram.axis.IntCategory([1, 2], overflow=False, name="i"),
    )
    no_overflow.fill(1)
    out2 = no_overflow.to_boost()
    assert not out2.axes[0].traits.overflow
    assert out2.axes[0].extent == 2
    assert (out2.values() == no_overflow.values().get()).all()


def test_int_category_to_boost() -> None:
    import boost_histogram as bh

    h = cuda_histogram.Hist(
        cuda_histogram.axis.IntCategory(growth=True, name="pdgid", label="PDG ID"),
        cuda_histogram.axis.Regular(3, 0, 1, name="x"),
        label="events",
    )
    h.fill(211, cp.array([0.1, 0.5, 1.5]))
    h.fill(-13, cp.array([-0.5, 0.9]))

    out = h.to_boost()
    assert isinstance(out.axes[0], bh.axis.IntCategory)
    assert list(out.axes[0]) == [211, -13]
    assert out.axes[0].name == "pdgid"
    assert out.axes[0].label == "PDG ID"
    assert out.storage_type == bh.storage.Double
    assert (out.values() == h.values().get()).all()
    assert out[bh.loc(211), bh.overflow] == 1.0
    assert out[bh.loc(-13), bh.underflow] == 1.0

    h.fill(-13, cp.array([0.9]), weight=cp.array([2.0]))
    out = h.to_boost()
    assert out.storage_type == bh.storage.Weight
    assert (out.values() == h.values().get()).all()
    assert (out.variances() == h.variance().get()).all()

    hh = h.to_hist()
    assert (hh.values() == h.values().get()).all()


def test_mixed_category_axes() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.StrCategory(growth=True, name="sample"),
        cuda_histogram.axis.Regular(3, 0, 1, name="x"),
        cuda_histogram.axis.IntCategory(growth=True, name="pdgid"),
    )
    h.fill("ttbar", cp.array([0.1, 0.5]), 211)
    h.fill("qcd", cp.array([0.9]), 13)
    assert h.values().shape == (2, 3, 2)
    assert (h.values()[0, :, 0] == cp.array([1, 1, 0])).all()
    assert (h.values()[1, :, 1] == cp.array([0, 0, 1])).all()

    out = h.to_boost()
    assert (out.values() == h.values().get()).all()


def test_boolean_axis() -> None:
    ax = cuda_histogram.axis.Boolean(name="flag", label="Flag")
    assert ax.size == 2
    assert ax.name == "flag"
    assert ax.label == "Flag"
    assert repr(ax) == "Boolean()"
    assert (ax.edges() == cp.array([0.0, 1.0, 2.0])).all()
    assert (ax.centers() == cp.array([0.5, 1.5])).all()
    # dense indices: 1 = False bin, 2 = True bin; flow bins never fill
    assert (ax.index(cp.array([False, True])) == cp.array([1, 2])).all()
    # nonzero coerces to True; NaN is truthy, so it lands in the True bin
    assert (
        ax.index(cp.array([0.0, 0.5, -2.0, np.nan])) == cp.array([1, 2, 2, 2])
    ).all()
    assert ax == cuda_histogram.axis.Boolean(name="flag")
    assert ax == "flag"
    assert ax != cuda_histogram.axis.Regular(2, 0, 2, name="flag")
    assert cuda_histogram.axis.Regular(2, 0, 2, name="flag") != ax


def test_boolean_fill() -> None:
    h = cuda_histogram.Hist(cuda_histogram.axis.Boolean(name="flag"))
    assert h.values().shape == (2,)
    assert h.values(flow=True).shape == (5,)

    h.fill(cp.array([True, False, True, True]))
    assert (h.values() == cp.array([1.0, 3.0])).all()
    assert h.variance() is None
    assert (h.values(flow=True) == cp.array([0, 1, 3, 0, 0])).all()

    # non-bool numerics coerce to truthiness; NaN counts as True, not nanflow
    h.fill(cp.array([0.0, 2.0, -1.0, np.nan]))
    assert (h.values() == cp.array([2.0, 6.0])).all()
    assert h.values(flow=True).sum() == 8

    h.fill(cp.array([0, 7]))  # integer input
    assert (h.values() == cp.array([3.0, 7.0])).all()


def test_boolean_weighted_fill() -> None:
    h = cuda_histogram.Hist(cuda_histogram.axis.Boolean(name="flag"))
    h.fill(cp.array([True, False, True]), weight=cp.array([2.0, 3.0, 4.0]))
    assert (h.values() == cp.array([3.0, 6.0])).all()
    assert (h.variance() == cp.array([9.0, 20.0])).all()


def test_boolean_with_regular() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.Boolean(name="flag"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    h.fill(cp.array([True, False, True]), cp.array([0.1, 0.3, 0.9]))
    assert h.values().shape == (2, 4)
    assert (h.values() == cp.array([[0, 1, 0, 0], [1, 0, 0, 1]])).all()

    # full slices work; the Regular axis can still be sliced by value
    assert (h[...].values() == h.values()).all()
    assert h[...].axes() == h.axes()
    sel = h[:, 0.3]
    assert (sel.values() == h.values()[:, 1].reshape(2, 1)).all()

    # anything but a full slice on the Boolean axis is an error
    with pytest.raises(IndexError, match="Boolean"):
        h[1, :]
    with pytest.raises(IndexError, match="Boolean"):
        h[0.5:, :]


def test_boolean_to_hist() -> None:
    import boost_histogram as bh
    import hist as hist_mod

    h = cuda_histogram.Hist(
        cuda_histogram.axis.Boolean(name="flag", label="Flag"),
        cuda_histogram.axis.Regular(4, 0, 1, name="x"),
    )
    h.fill(cp.array([True, False, True]), cp.array([0.1, 0.3, 1.5]))

    out = h.to_hist()
    assert isinstance(out, hist_mod.Hist)
    assert isinstance(out.axes[0], bh.axis.Boolean)
    assert out.axes[0].name == "flag"
    assert out.axes[0].label == "Flag"
    assert out.axes[0].extent == 2  # no flow bins
    assert out.storage_type == bh.storage.Double
    assert (out.values() == h.values().get()).all()
    # only the Regular axis has flow content to preserve
    assert out.values(flow=True).sum() == h.values(flow=True).sum()
    assert out[bh.loc(True), bh.overflow] == 1.0

    h.fill(cp.array([False]), cp.array([0.5]), weight=cp.array([2.0]))
    out2 = h.to_boost()
    assert out2.storage_type == bh.storage.Weight
    assert (out2.values() == h.values().get()).all()
    assert (out2.variances() == h.variance().get()).all()


def test_hist_conversion() -> None:
    import hist as hist_mod

    dummy = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(1, 0, 1, name="dummy", label="Number of events"),
        cuda_histogram.axis.Variable([0, 2, 3, 10], name="var", label="Variable axis"),
        name="Dummy",
        label="Just Dummy",
    )
    dummy.fill(cp.array(0.05), cp.array(0.05))

    h = dummy.to_hist()
    assert isinstance(h, hist_mod.Hist)
    assert h.axes[0].name == "dummy"
    assert h.axes[0].label == "Number of events"
    assert h.axes[1].name == "var"
    assert h.axes[1].label == "Variable axis"
    assert h[0, 0] == 1.0
    assert h.values().shape == dummy.values().shape
