import pytest
import numpy as np
from numpy import random

from src.svd import apply_givens_left, apply_givens_right


@pytest.mark.parametrize("n, m", [4, 5, 6])
def test_givens_left(n): ...


@pytest.mark.parametrize("n, m", [4, 5, 6])
def test_givens_right(n): ...


@pytest.mark.parametrize("l1, l2", [(1.0, 2.0), (5.0, -1.0)])
def test_2_2_eigevalues(l1, l2): ...


@pytest.mark.parametrize("n, m", [4, 5, 6])
def test_householder_reflection(n): ...
