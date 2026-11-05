"""Tests for distributions.py."""

import math

import numpy as np
import pytest
import torch

from ecoscape_connectivity import half_cauchy


def test_half_cauchy_range() -> None:
    """Check that samples are integers from 1 to truncation, both included."""
    # half_cauchy samples with torch.
    torch.manual_seed(0)
    sample = half_cauchy(median=2, truncation=5)
    samples = [sample() for _ in range(1000)]
    assert all(isinstance(s, int) for s in samples)
    assert set(samples) == {1, 2, 3, 4, 5}


def test_half_cauchy_median() -> None:
    """Check that the median of many samples is close to the given median."""
    torch.manual_seed(0)
    sample = half_cauchy(median=10, truncation=100)
    samples = [sample() for _ in range(10_000)]
    # Samples are integers, so their median can be off by one.
    assert abs(np.median(samples) - 10) <= 1


def test_half_cauchy_truncation_too_small() -> None:
    """Check that truncation + 0.5 must be more than twice the median."""
    with pytest.raises(AssertionError):
        half_cauchy(median=10, truncation=19)
    # Exactly twice the median also fails: 20.5 = 2 * 10.25.
    with pytest.raises(AssertionError):
        half_cauchy(median=10.25, truncation=20)
    # 19.5 is less than 20, but 20.5 is more.
    half_cauchy(median=10, truncation=20)


def half_cauchy_sigma(median: float, c: float) -> float:
    """Solve atan(c / sigma) = 2 * atan(median / sigma) for sigma by
    bisection, so the test doesn't depend on the closed-form formula in
    half_cauchy.

    This makes the median of the half-Cauchy truncated at c equal to median.
    """
    low, high = 1e-6, 1e6
    for _ in range(200):
        sigma = math.sqrt(low * high)
        if math.atan(c / sigma) < 2 * math.atan(median / sigma):
            low = sigma
        else:
            high = sigma
    return sigma


def half_cauchy_probabilities(median: float, truncation: int) -> np.ndarray:
    """Return the probability of each value from 1 to truncation: the
    half-Cauchy's probability of rounding to it, normalized over the values.
    """
    sigma = half_cauchy_sigma(median, truncation + 0.5)
    values = np.arange(1, truncation + 1)
    weights = np.arctan((values + 0.5) / sigma) - np.arctan((values - 0.5) / sigma)
    return weights / weights.sum()


@pytest.mark.parametrize(
    ("median", "truncation"),
    # A truncation just above twice the median, where sigma changes fastest,
    # and a small median, where the distribution is skewed toward 1.
    [(10, 20), (2, 20)],
)
def test_half_cauchy_probabilities(median: float, truncation: int) -> None:
    """Check that the share of each value in many samples matches its
    probability under the truncated half-Cauchy.

    With 50,000 samples, each share has a standard error of
    sqrt(p * (1 - p) / 50,000); the test allows 4 standard errors. Each
    mutant of half_cauchy's sigma formula and bucket edges differs by at
    least 9 standard errors in one of the two cases.
    """
    torch.manual_seed(0)
    sample = half_cauchy(median=median, truncation=truncation)
    num_samples = 50_000
    samples = np.array([sample() for _ in range(num_samples)])
    shares = np.bincount(samples, minlength=truncation + 1)[1:] / num_samples
    expected = half_cauchy_probabilities(median, truncation)
    standard_errors = np.sqrt(expected * (1 - expected) / num_samples)
    assert (np.abs(shares - expected) < 4 * standard_errors).all()
