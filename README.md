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

`cuda-histogram` is a histogram filling package for GPUs. The package follows
[UHI](https://uhi.readthedocs.io) and keeps its API similar to
[boost-histogram](https://github.com/scikit-hep/boost-histogram) and
[hist](https://github.com/scikit-hep/hist).

The package is under active development at the moment.

## Example

```py
In [1]: import cuda_histogram; import cupy as cp

In [2]: a = cuda_histogram.axis.Regular(10, 0, 1)

In [3]: b = cuda_histogram.axis.Variable([0, 2, 3, 6])

In [4]: c = cuda_histogram.Hist(a, b)

In [5]: a, b, c
Out[5]:
(Regular(10, 0, 1),
 Variable([0. 2. 3. 6.]),
 Hist(Regular(10, 0, 1), Variable([0. 2. 3. 6.])))

In [6]: c.fill(cp.random.normal(size=1_000_000), cp.random.normal(size=1_000_000))

In [7]: c.values(), type(c.values())
Out[7]:
(array([[28493.,  1282.,    96.],
        [29645.,  1366.,    91.],
        [30465.,  1397.,    80.],
        [31537.,  1473.,    81.],
        [32608.,  1454.,   102.],
        [33015.,  1440.,    83.],
        [33992.,  1482.,    87.],
        [34388.,  1482.,   111.],
        [34551.,  1517.,    90.],
        [35177.,  1515.,    85.]]),
 cupy.ndarray)

In [8]: c[0, 0], type(c[0, 0])
Out[8]: (array(28493.), cupy.ndarray)

In [9]: c[0:2, 0], type(c[0, 0]) # should ideally return a reduced histogram
Out[9]: (array([28493., 29645.]), cupy.ndarray)

In [10]: c.to_boost()
Out[10]:
Histogram(
  Regular(10, 0, 1),
  Variable([0, 2, 3, 6]),
  storage=Double()) # Sum: 339185.0 (945991.0 with flow)

In [11]: c.to_boost().values(), type(c.to_boost().values())
Out[11]:
(array([[28493.,  1282.,    96.],
        [29645.,  1366.,    91.],
        [30465.,  1397.,    80.],
        [31537.,  1473.,    81.],
        [32608.,  1454.,   102.],
        [33015.,  1440.,    83.],
        [33992.,  1482.,    87.],
        [34388.,  1482.,   111.],
        [34551.,  1517.,    90.],
        [35177.,  1515.,    85.]]),
 numpy.ndarray)

In [12]: c.to_hist()
Out[12]:
Hist(
  Regular(10, 0, 1, label='Axis 0'),
  Variable([0, 2, 3, 6], label='Axis 1'),
  storage=Double()) # Sum: 339185.0 (945991.0 with flow)

In [13]: c.to_hist().values(), type(c.to_hist().values())
Out[13]:
(array([[28493.,  1282.,    96.],
        [29645.,  1366.,    91.],
        [30465.,  1397.,    80.],
        [31537.,  1473.,    81.],
        [32608.,  1454.,   102.],
        [33015.,  1440.,    83.],
        [33992.,  1482.,    87.],
        [34388.,  1482.,   111.],
        [34551.,  1517.,    90.],
        [35177.,  1515.,    85.]]),
    numpy.ndarray)
```

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/Saransh-cpp/cuda-histogram/workflows/CI/badge.svg
[actions-link]:             https://github.com/Saransh-cpp/cuda-histogram/actions
[codecov-badge]:            https://codecov.io/gh/Saransh-cpp/cuda-histogram/branch/main/graph/badge.svg?token=YBv60ueORQ
[codecov-link]:             https://codecov.io/gh/Saransh-cpp/cuda-histogram
[conda-version]:            https://img.shields.io/conda/vn/Saransh-cpp/cuda-histogram.svg
[conda-link]:               https://github.com/Saransh-cpp/cuda-histogram
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/Saransh-cpp/cuda-histogram/discussions
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
