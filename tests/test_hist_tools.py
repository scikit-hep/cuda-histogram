from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import cuda_histogram  # noqa: E402

# cupy might be installable on a device with no GPUs
try:
    cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA not found", allow_module_level=True)


def dummy_jagged_eta_pt():
    np.random.seed(42)
    counts = np.random.exponential(2, size=50).astype(int)
    entries = cp.sum(counts)
    test_eta = cp.random.uniform(-3.0, 3.0, size=entries)
    test_pt = cp.random.exponential(10.0, size=entries) + cp.random.exponential(
        10, size=entries
    )
    return (counts, test_eta, test_pt)


def test_hist():
    counts, test_eta, test_pt = dummy_jagged_eta_pt()

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


def test_partial_indexing():
    counts, test_eta, test_pt = dummy_jagged_eta_pt()

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
    with pytest.raises(ValueError):
        h["x"]


def test_underflow_index_and_fill():
    ax = cuda_histogram.axis.Regular(10, 0, 1)
    assert int(ax.index(cp.array([-5.0]))[0]) == 0
    assert int(ax.index(cp.array([5.0]))[0]) == 11

    h = cuda_histogram.Hist(cuda_histogram.axis.Regular(10, 0, 1, name="x"))
    h.fill(cp.array([-5.0]))
    values = h.values(flow=True)
    assert int(values[0]) == 1
    assert int(values.sum()) == 1


def test_nanflow_index_and_fill():
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


def test_bin_validation():
    with pytest.raises(TypeError):
        cuda_histogram.axis.Bin(3.5, 0, 1)
    with pytest.raises(TypeError):
        cuda_histogram.axis.Regular(3.5, 0, 1)
    with pytest.raises(ValueError, match="not sorted"):
        cuda_histogram.axis.Variable([0, 2, 1])
    with pytest.raises(ValueError, match="not sorted"):
        cuda_histogram.axis.Variable([0, 1, 1, 2])


@pytest.mark.parametrize("weighted_first", [False, True])
def test_mixed_weighted_unweighted_fill(weighted_first):
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


def test_fill_integer_array():
    h = cuda_histogram.Hist(cuda_histogram.axis.Regular(4, 0, 8, name="x"))
    h.fill(cp.array([1, 3, 3, 5]))
    assert (h.values() == cp.array([1.0, 2.0, 1.0, 0.0])).all()


def test_cpu_conversion():
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
    assert h[0, 0].value == 2.0
    assert h[0, 0].variance == 4.0
    assert h.values().shape == dummy.values().shape
    assert h[bh.underflow, bh.underflow].variance == 4.0
    assert h.storage_type == bh.storage.Weight


def test_hist_conversion():
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
