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


def _flow_hist(*, underflow: bool = True, overflow: bool = True) -> cuda_histogram.Hist:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(
            4, 0.0, 4.0, name="x", underflow=underflow, overflow=overflow
        )
    )
    h.fill(cp.array([-1.0, 0.5, 1.5, 2.5, 3.5, 5.0, np.nan]))
    return h


def _dense_axis(h: cuda_histogram.Hist, i: int = 0) -> cuda_histogram.axis.Bin:
    ax = h.axes()[i]
    assert isinstance(ax, cuda_histogram.axis.Bin)
    return ax


def test_default_flow_unchanged() -> None:
    h = _flow_hist()
    ax = _dense_axis(h)
    assert ax.underflow
    assert ax.overflow
    # [underflow, 4 bins, overflow, nanflow]
    assert h.values(flow=True).get() == pytest.approx([1, 1, 1, 1, 1, 1, 1])


def test_underflow_disabled() -> None:
    h = _flow_hist(underflow=False)
    assert not _dense_axis(h).underflow
    assert h.values(flow=True).get() == pytest.approx([0, 1, 1, 1, 1, 1, 1])
    assert h.values().get() == pytest.approx([1, 1, 1, 1])


def test_overflow_disabled_keeps_nanflow() -> None:
    h = _flow_hist(overflow=False)
    assert not _dense_axis(h).overflow
    # 5.0 is discarded, but NaN still lands in nanflow
    assert h.values(flow=True).get() == pytest.approx([1, 1, 1, 1, 1, 0, 1])


def test_weighted_fill_discards_matching_weights() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(2, 0.0, 2.0, name="x", underflow=False)
    )
    h.fill(cp.array([-1.0, 0.5, 1.5]), weight=cp.array([100.0, 2.0, 3.0]))
    assert h.values(flow=True).get() == pytest.approx([0, 2, 3, 0, 0])
    assert h.variance(flow=True).get() == pytest.approx([0, 4, 9, 0, 0])


def test_variable_flow() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.Variable(
            [0.0, 1.0, 3.0], name="x", underflow=False, overflow=False
        )
    )
    h.fill(cp.array([-1.0, 0.5, 2.0, 4.0, np.nan]))
    assert h.values(flow=True).get() == pytest.approx([0, 1, 1, 0, 1])


def test_integer_flow() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.Integer(0, 3, name="n", underflow=False, overflow=False)
    )
    h.fill(cp.array([-1, 0, 1, 2, 5]))
    assert h.values(flow=True).get() == pytest.approx([0, 1, 1, 1, 0, 0])
    converted = h.to_hist()
    assert converted.axes[0].traits.underflow is False
    assert converted.axes[0].traits.overflow is False
    assert converted.values(flow=True) == pytest.approx([1, 1, 1])


def test_multiaxis_mask_combines() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(2, 0.0, 2.0, name="x", underflow=False),
        cuda_histogram.axis.Regular(2, 0.0, 2.0, name="y", overflow=False),
    )
    # (-1, 0.5) dropped by x underflow; (0.5, 5) dropped by y overflow
    h.fill(cp.array([-1.0, 0.5, 0.5]), cp.array([0.5, 5.0, 1.5]))
    assert h.values().sum() == 1
    assert h.values(flow=True).sum() == 1


def test_slicing_drops_cropped_content() -> None:
    h = cuda_histogram.Hist(
        cuda_histogram.axis.Regular(4, 0.0, 4.0, name="x", overflow=False)
    )
    h.fill(cp.array([0.5, 1.5, 2.5, 3.5]))
    reduced = h[1.0:3.0]
    assert not _dense_axis(reduced).overflow
    # 0.5 folds into underflow (enabled); 3.5 would fold into overflow (disabled)
    assert reduced.values(flow=True).get() == pytest.approx([1, 1, 1, 0, 0])


def test_axis_equality_includes_flow() -> None:
    reg = cuda_histogram.axis.Regular(4, 0.0, 4.0, name="x")
    assert reg != cuda_histogram.axis.Regular(4, 0.0, 4.0, name="x", underflow=False)
    assert reg == cuda_histogram.axis.Regular(4, 0.0, 4.0, name="x")


def test_to_hist_carries_flow_flags() -> None:
    pytest.importorskip("hist")
    h = _flow_hist(underflow=False)
    converted = h.to_hist()
    ax = converted.axes[0]
    assert ax.traits.underflow is False
    assert ax.traits.overflow is True
    assert converted.values(flow=True) == pytest.approx([1, 1, 1, 1, 1])
    assert converted.values() == pytest.approx([1, 1, 1, 1])

    both = _flow_hist(underflow=False, overflow=False).to_hist()
    assert both.axes[0].traits.underflow is False
    assert both.axes[0].traits.overflow is False
    assert both.values(flow=True) == pytest.approx([1, 1, 1, 1])
