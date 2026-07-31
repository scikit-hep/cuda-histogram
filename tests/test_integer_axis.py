from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import cuda_histogram

# cupy might be installable on a device with no GPUs
try:
    cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA not found", allow_module_level=True)


def test_integer_axis() -> None:
    ax = cuda_histogram.axis.Integer(-2, 3, name="n", label="count")
    assert ax.size == 5
    assert ax.name == "n"
    assert ax.label == "count"
    assert repr(ax) == "Integer(-2, 3)"
    assert (ax.edges() == cp.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])).all()
    with pytest.raises(ValueError, match="greater than"):
        cuda_histogram.axis.Integer(2, 2)


def test_integer_fill_and_values() -> None:
    h = cuda_histogram.Hist(cuda_histogram.axis.Integer(-2, 3, name="n"))
    assert h.values().shape == (5,)
    assert h.values(flow=True).shape == (8,)

    h.fill(cp.array([-2, 0, 0, 2, 2, 2]))
    assert (h.values() == cp.array([1.0, 0.0, 2.0, 0.0, 3.0])).all()

    # floats bin by floor: value v lands in bin v - start
    h.fill(cp.array([0.5, 2.999, -1.0]))
    assert (h.values() == cp.array([1.0, 1.0, 3.0, 0.0, 4.0])).all()


def test_integer_flow_bins() -> None:
    h = cuda_histogram.Hist(cuda_histogram.axis.Integer(0, 4, name="n"))
    h.fill(cp.array([-1.0, -100.0, 4.0, 17.0, np.nan, np.nan, np.nan, 1.0]))
    flow = h.values(flow=True)
    assert flow[0] == 2.0  # underflow
    assert flow[-2] == 2.0  # overflow
    assert flow[-1] == 3.0  # nanflow
    assert (h.values() == cp.array([0.0, 1.0, 0.0, 0.0])).all()


def test_integer_slicing() -> None:
    h = cuda_histogram.Hist(cuda_histogram.axis.Integer(-2, 3, name="n"))
    h.fill(cp.array([-2, -1, 0, 0, 1, 2, 2, 5, -7]))

    sliced = h[0:2]
    (ax,) = sliced.axes()
    assert isinstance(ax, cuda_histogram.axis.Integer)
    assert ax.size == 2
    assert repr(ax) == "Integer(0, 2)"
    assert (sliced.values() == cp.array([2.0, 1.0])).all()
    # out-of-slice content is summed into the flow bins
    assert sliced.values(flow=True)[0] == 3.0
    assert sliced.values(flow=True)[-2] == 3.0

    single = h[0]
    (ax,) = single.axes()
    assert isinstance(ax, cuda_histogram.axis.Integer)
    assert ax.size == 1
    assert (single.values() == cp.array([2.0])).all()

    assert h[:].axes() == h.axes()
    assert (h[:].values(flow=True) == h.values(flow=True)).all()


def test_integer_to_hist_roundtrip() -> None:
    hist = pytest.importorskip("hist")

    h = cuda_histogram.Hist(cuda_histogram.axis.Integer(-2, 3, name="n", label="count"))
    h.fill(cp.array([-5.0, -2.0, 0.0, 0.0, 2.0, 9.0, np.nan]))

    converted = h.to_hist()
    (ax,) = converted.axes
    assert isinstance(ax, hist.axis.Integer)
    assert ax.name == "n"
    assert ax.label == "count"
    assert ax.size == 5
    assert ax.edges[0] == -2
    assert ax.edges[-1] == 3
    assert (converted.values() == h.values().get()).all()
    # nanflow is dropped; under/overflow carry over
    assert (converted.values(flow=True) == h.values(flow=True)[:-1].get()).all()

    h.fill(cp.array([1.0]), weight=cp.array([2.0]))
    weighted = h.to_hist()
    variance = weighted.variances()
    assert variance is not None
    assert (variance == h.variance().get()).all()
