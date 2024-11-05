# cuda-histogram

<!-- SPHINX-START -->

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![pre-commit.ci status][pre-commit-badge]][pre-commit-link]
[![codecov percentage][codecov-badge]][codecov-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]
[![Conda latest release][conda-version]][conda-link]
[![LICENSE][license-badge]][license-link] [![Scikit-HEP][sk-badge]][sk-link]

`cuda-histogram` is a histogram filling package for GPUs. The package tries to
follow [UHI](https://uhi.readthedocs.io) and keeps its API similar to
[boost-histogram](https://github.com/scikit-hep/boost-histogram) and
[hist](https://github.com/scikit-hep/hist).

Main features of cuda-histogram:

- Implements a subset of the features of boost-histogram using CuPy (see API
  documentation for a complete list):
  - Axes
    - `Regular` and `Variable` axes
      - `edges()`
      - `centers()`
      - `index(...)`
      - ...
  - Histogram
    - `fill(..., weight=...)` (including `Nan` flow)
    - simple indexing with slicing (see example below)
    - `values(flow=...)`
    - `variance(flow=...)`
- Allows users to detach the generated GPU histogram to CPU -
  - `to_boost()` - converts to `boost-histogram.Histogram`
  - `to_hist()` - converts to `hist.Hist`

Near future goals for the package -

- Implement support for `Categorical` axes (exists internally but need
  refactoring to match boost-histogram's API)
- Improve indexing (`__getitem__`) to exactly match boost-histogram's API

## Installation

cuda-histogram is available on [PyPI](https://pypi.org/project/cuda-histogram/)
as well as on [conda](https://anaconda.org/conda-forge/cuda_histogram). The
library can be installed using `pip` -

```
pip install cuda-histogram
```

or using `conda` -

```
conda install -c conda-forge cuda_histogram
```

## Usage

Ideally, a user would want to create a cuda-histogram, fill values on GPU, and
convert the filled histogram to boost-histogram/Hist object to access all the
UHI functionalities.

### Creating a histogram

```py
import cuda_histogram; import cupy as cp

ax1 = cuda_histogram.axis.Regular(10, 0, 1)
ax2 = cuda_histogram.axis.Variable([0, 2, 3, 6])

h = cuda_histogram.Hist(ax1, ax2)

>>> ax1, ax2, h
(Regular(10, 0, 1), Variable([0. 2. 3. 6.]), Hist(Regular(10, 0, 1), Variable([0. 2. 3. 6.])))
```

### Filling a histogram

Differences in API (from boost-histogram) -

- Has an additional `NaN` flow
- Accepts only CuPy arrays

```py
h.fill(cp.random.normal(size=1_000_000), cp.random.normal(size=1_000_000))  # set weight=... for weighted fills

>>> h.values(), type(h.values())  # set flow=True for flow bins (underflow, overflow, nanflow)
(array([[28532.,  1238.,    64.],
       [29603.,  1399.,    61.],
       [30543.,  1341.,    78.],
       [31478.,  1420.,    98.],
       [32692.,  1477.,    92.],
       [32874.,  1441.,    96.],
       [33584.,  1515.,    88.],
       [34304.,  1490.,   114.],
       [34887.,  1598.,   116.],
       [35341.,  1472.,   103.]]), <class 'cupy.ndarray'>)
```

### Indexing axes and histograms

Differences in API (from boost-histogram) -

- `underflow` is indexed as `0` and not `-1`
- `ax[...]` will return a `cuda_histogram.Interval` object
- No interpolation is performed
- `Hist` indices should be in the range of bin edges, instead of integers

```py
>>> ax1.index(0.5)
array([6])

>>> ax1.index(-1)
array([0])

>>> ax1[0]
<Interval ((-inf, 0.0)) instance at 0x1c905208790>

>>> h[0, 0], type(h[0, 0])
(Hist(Regular(1, 0.0, 0.1), Variable([0. 2.])), <class 'cuda_histogram.hist.Hist'>)

>>> h[0, 0].values(), type(h[0, 0].values())
(array([[28532.]]), <class 'cupy.ndarray'>)

>>> h[0, :].values(), type(h[0, 0].values())
(array([[28532.,  1238.,    64.]]), <class 'cupy.ndarray'>)

>>> h[0.2, :].values(), type(h[0, 0].values()) # indices in range of bin edges
(array([[30543.,  1341.,    78.]]), <class 'cupy.ndarray'>)

>>> h[:, 1:2].values(), type(h[0, 0].values()) # no interpolation
C:\Users\Saransh\Saransh_softwares\OpenSource\Python\cuda-histogram\src\cuda_histogram\axis\__init__.py:580: RuntimeWarning: Reducing along axis Variable([0. 2. 3. 6.]): requested start 1 between bin boundaries, no interpolation is performed
  warnings.warn(
(array([[28532.],
       [29603.],
       [30543.],
       [31478.],
       [32692.],
       [32874.],
       [33584.],
       [34304.],
       [34887.],
       [35341.]]), <class 'cupy.ndarray'>)
```

### Converting to CPU

All the existing functionalities of boost-histogram and Hist can be used on the
converted histogram.

```py
h.to_boost()

>>> h.to_boost().values(), type(h.to_boost().values())
(array([[28532.,  1238.,    64.],
       [29603.,  1399.,    61.],
       [30543.,  1341.,    78.],
       [31478.,  1420.,    98.],
       [32692.,  1477.,    92.],
       [32874.,  1441.,    96.],
       [33584.,  1515.,    88.],
       [34304.,  1490.,   114.],
       [34887.,  1598.,   116.],
       [35341.,  1472.,   103.]]), <class 'numpy.ndarray'>)

h.to_hist()

>>> h.to_hist().values(), type(h.to_hist().values())
(array([[28532.,  1238.,    64.],
       [29603.,  1399.,    61.],
       [30543.,  1341.,    78.],
       [31478.,  1420.,    98.],
       [32692.,  1477.,    92.],
       [32874.,  1441.,    96.],
       [33584.,  1515.,    88.],
       [34304.,  1490.,   114.],
       [34887.,  1598.,   116.],
       [35341.,  1472.,   103.]]), <class 'numpy.ndarray'>)
```

## Getting help

- `cuda-histogram`'s code is hosted on
  [GitHub](https://github.com/scikit-hep/cuda-histogram).
- If something is not working the way it should, or if you want to request a new
  feature, create a new
  [issue](https://github.com/scikit-hep/cuda-histogram/issues) on GitHub.
- To discuss something related to `cuda-histogram`, use the
  [discussions](https://github.com/scikit-hep/cuda-histogram/discussions/) tab
  on GitHub.

## Contributing

Contributions of any kind welcome! See
[CONTRIBUTING.md](./.github/CONTRIBUTING.md) for information on setting up a
development environment.

## Acknowledgements

This library was primarily developed by Lindsey Gray, Saransh Chopra, and Jim
Pivarski.

Support for this work was provided by the National Science Foundation
cooperative agreement OAC-1836650 and PHY-2323298 (IRIS-HEP). Any opinions,
findings, conclusions or recommendations expressed in this material are those of
the authors and do not necessarily reflect the views of the National Science
Foundation.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/scikit-hep/cuda-histogram/workflows/CI/badge.svg
[actions-link]:             https://github.com/scikit-hep/cuda-histogram/actions
[codecov-badge]:            https://codecov.io/gh/Saransh-cpp/cuda-histogram/branch/main/graph/badge.svg?token=YBv60ueORQ
[codecov-link]:             https://codecov.io/gh/Saransh-cpp/cuda-histogram
[conda-version]:            https://img.shields.io/conda/vn/conda-forge/cuda_histogram.svg
[conda-link]:               https://github.com/conda-forge/cuda_histogram-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/scikit-hep/cuda-histogram/discussions
[license-badge]:            https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
[license-link]:             https://opensource.org/licenses/BSD-3-Clause
[pre-commit-badge]:         https://results.pre-commit.ci/badge/github/Saransh-cpp/cuda-histogram/main.svg
[pre-commit-link]:          https://results.pre-commit.ci/repo/github/Saransh-cpp/cuda-histogram
[pypi-link]:                https://pypi.org/project/cuda-histogram/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/cuda-histogram
[pypi-version]:             https://img.shields.io/pypi/v/cuda-histogram
[rtd-badge]:                https://readthedocs.org/projects/cuda-histogram/badge/?version=latest
[rtd-link]:                 https://cuda-histogram.readthedocs.io/en/latest/?badge=latest
[sk-badge]:                 https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
[sk-link]:                  https://scikit-hep.org/

<!-- prettier-ignore-end -->
