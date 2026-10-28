"""Tests for SingleIterator."""

from collections.abc import Callable, Iterable

import pytest

from ecoscape_connectivity import SingleIterator


@pytest.mark.parametrize("make_items", [list, iter], ids=["list", "iterator"])
def test_single_iterator(
    make_items: Callable[[Iterable[int]], Iterable[int]],
) -> None:
    """Check that each item is paired with None."""
    items = make_items([1, 2, 3, 4, 5])
    assert list(SingleIterator(items)) == [
        (None, 1),
        (None, 2),
        (None, 3),
        (None, 4),
        (None, 5),
    ]
