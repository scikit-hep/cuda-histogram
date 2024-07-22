from __future__ import annotations


# pytest will fail if no tests are executed,
# that is, either CUDA is not available or cupy
# is not installable.
def test_dummy():
    pass
