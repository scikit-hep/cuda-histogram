# Agent instructions

This is a CUDA version of boost-histogram.

## Commands

```bash
uv run pytest                          # run tests (dev dependency-group is automatic)
uv run pytest tests/test_hist_tools.py::test_hist   # single test
uv run pytest --cov=cuda_histogram
prek -a --quiet                        # lint/format (ruff, prettier, codespell, ...)
```

Nox is the documented entry point for contributors and is what CI runs:
`nox -s tests`, `nox -s coverage-3.12`, `nox -s docs -- --serve`,
`nox -s build`, `nox -s build_api_docs` (regenerates `docs/api/*.rst` with
sphinx-apidoc).

Dev/test/docs dependencies live in PEP 735 `[dependency-groups]` (`test`,
`docs`, `dev` includes `test`); nox resolves them via
`nox.project.dependency_groups`. Extras are reserved for public, user-facing
options: `cuda12x` and `cuda13x` select the matching CuPy wheel, and they
conflict, so select exactly one.

## Testing reality

`.github/workflows/ci.yml` runs on `ubuntu-latest` with **no GPU**, so
`tests/test_hist_tools.py` skips entirely there (`pytest.importorskip("cupy")`
plus a `getDeviceCount` check). `tests/test_dummy.py` exists only so pytest
doesn't fail on an empty collection.

Real coverage comes from `.github/workflows/gpu.yml`, which runs the full suite
on the scikit-hep self-hosted CUDA runners, matrixed over CUDA 12/13 and Python
3.10/3.14 (oldest and newest supported). It triggers nightly, on pushes to
`main`, on demand, and on pull requests — but only same-repo PRs, since
self-hosted runners must not run fork code. It gets CuPy from conda-forge with
micromamba, so it installs the package without a `cuda12x`/`cuda13x` extra.
Still run the suite locally on a CUDA machine before claiming a change works;
PRs from a fork get no GPU coverage at all.

`filterwarnings = ["error", ...]` in `pyproject.toml`, so unexpected warnings
fail tests. mypy runs strict via pre-commit (`uv run mypy` locally); `cupy` and
`awkward` ship no stubs, so anything touching them is `Any`.

## Architecture

Two modules do all the work: `src/cuda_histogram/axis/__init__.py` (axes) and
`src/cuda_histogram/hist.py` (`Hist`). The design descends from coffea's `hist`,
retrofitted onto a boost-histogram-like public API.

**Storage.** `Hist._sumw` is a `dict` mapping a _sparse key_ (tuple of
sparse-axis indices) to a dense CuPy array of shape `Hist._dense_shape`.
`_sumw2` stays `None` until the first weighted `fill()`, at which point
`_init_sumw2()` seeds it from `_sumw`; `variance()` returns `None` while it is
`None`, and `to_boost()` picks `Double` vs `Weight` storage from it. The sparse
key holds one bin index per category axis (empty tuple when there are none);
index `size` is the overflow bin of a non-growing `overflow=True` axis, and
unmatched fills on an `overflow=False` axis are discarded entirely.
`values()`/`variance()` stack the sparse dict into a dense array via
`Hist._stack_sparse`, reordering so category dimensions land in axis order.

**Bin layout.** Every dense axis stores `nbins + 3` bins: index `0` = underflow,
`n+1` = overflow, `n+2` = **nanflow**. The nanflow bin is the main deviation
from boost-histogram. `_dense_shape` computes this as `size + 3` for `Regular`
and `size + 1` for `Variable` — these agree because `Variable.size` counts the
extra `inf` edge appended in `Bin.__init__`. `_overflow_behavior(flow)` is the
single place that translates `flow=False` into `slice(1, -2)`.

**Filling.** `Hist.fill()` requires CuPy arrays (a plain string/int per
`StrCategory`/`IntCategory` axis). It maps values to indices via `axis.index()`,
flattens with `cupy.ravel_multi_index`, and accumulates with `cupy.bincount`.
Uniform (`Regular`) axes compute indices with the `_clip_bins`
`cupy.ElementwiseKernel` at the top of `axis/__init__.py` (clamps to the flow
bins and sends NaN to nanflow); `Variable` axes use `cupy.searchsorted` against
edges with `inf` appended so NaN sorts past it.

**Indexing.** `Hist.__getitem__` is deliberately narrow: indices are _values in
bin-edge coordinates_, not integer bin numbers, no interpolation (a
`RuntimeWarning` fires for a boundary miss); `...` and partial indexing pad with
`slice(None)`, and anything fancier raises "use to_boost/to_hist".
`Axis._ireduce` converts a value slice to an index slice and `Axis.reduced`
builds the new axis. `_assemble_blocks` + `np.block` then re-sum the
out-of-slice content into the flow bins — note this path round-trips GPU→CPU→GPU
(`array.get()` … `cupy.asarray`).

**Escape hatch.** `to_boost()` / `to_hist()` are how users get full UHI. The
nanflow bin is dropped in the conversion (the `nonan` slice); underflow and
overflow are always enabled on the produced axes, which are built as
`hist.axis.*` (subclasses of the boost-histogram axes) so `name`/`label`
survive.

## Conventions

- `from __future__ import annotations` is a required first import (ruff isort).
- Axes are `Regular`, `Variable`, `StrCategory`, and `IntCategory` only; when
  adding features, check whether boost-histogram already names the concept and
  match that name.
