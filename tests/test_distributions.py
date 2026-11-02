"""Tests for distributions.py."""

import numpy as np
import pytest

from ecoscape_connectivity import half_cauchy


def test_half_cauchy_range() -> None:
    """Check that samples are integers from 1 to truncation, both included."""
    # half_cauchy samples with np.random.
    np.random.seed(0)
    sample = half_cauchy(median=2, truncation=5)
    samples = [sample() for _ in range(1000)]
    assert all(isinstance(s, int) for s in samples)
    assert set(samples) == {1, 2, 3, 4, 5}


def test_half_cauchy_median() -> None:
    """Check that the median of many samples is close to the given median."""
    np.random.seed(0)
    sample = half_cauchy(median=10, truncation=100)
    samples = [sample() for _ in range(10_000)]
    # Samples are integers, so their median can be off by one.
    assert abs(np.median(samples) - 10) <= 1


def test_half_cauchy_truncation_too_small() -> None:
    """Check that truncation + 0.5 must be more than twice the median."""
    with pytest.raises(AssertionError):
        half_cauchy(median=10, truncation=19)
    # 19.5 is less than 20, but 20.5 is more.
    half_cauchy(median=10, truncation=20)
