"""Tests for util.py."""

from collections.abc import Callable, Iterable
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ecoscape_connectivity import SingleIterator, draw_layer, read_transmission_csv
from ecoscape_connectivity.util import (
    createdir_for_file,
    dict_translate,
    rescale_resistance,
)
from tests.helpers import write_geotiff


def test_dict_translate() -> None:
    """Check that each code maps to its value, keeping the array's shape."""
    # Shaped like a tile: one band, 2 x 3 pixels.
    terrain = np.array([[[11, 21, 11], [42, 21, 42]]])
    permeability = {11: 0.5, 21: 1.0, 42: 0.25}
    expected = np.array([[[0.5, 1.0, 0.5], [0.25, 1.0, 0.25]]])
    np.testing.assert_array_equal(dict_translate(terrain, permeability), expected)


def test_dict_translate_missing_code() -> None:
    """Check that codes missing from the dictionary map to default_val."""
    terrain = np.array([[11, 99], [7, 11]])
    permeability = {11: 0.5}
    np.testing.assert_array_equal(
        dict_translate(terrain, permeability), [[0.5, 0], [0, 0.5]]
    )
    np.testing.assert_array_equal(
        dict_translate(terrain, permeability, default_val=0.75),
        [[0.5, 0.75], [0.75, 0.5]],
    )


def test_read_transmission_csv_resistance(tmp_path: Path) -> None:
    """Check that a resistance column is converted to 1 - resistance."""
    csv_path = tmp_path / "resistance.csv"
    csv_path.write_text("map_code,resistance\n11,0.25\n21,0\n42,1\n")
    assert read_transmission_csv(str(csv_path)) == {11: 0.75, 21: 1.0, 42: 0.0}


def test_read_transmission_csv_transmission(tmp_path: Path) -> None:
    """Check that a transmission column is read as is."""
    csv_path = tmp_path / "transmission.csv"
    csv_path.write_text("map_code,transmission\n11,0.25\n21,0\n42,1\n")
    assert read_transmission_csv(str(csv_path)) == {11: 0.25, 21: 0.0, 42: 1.0}


def test_rescale_resistance() -> None:
    """Check that values are unchanged for a 100 m hop and squared for 200 m."""
    resistance = {11: 0.5, 21: 1.0, 42: 0.0}
    # resolution_m * hop_length is the hop in meters.
    assert rescale_resistance(resistance, resolution_m=100, hop_length=1) == {
        11: 0.5,
        21: 1.0,
        42: 0.0,
    }
    assert rescale_resistance(resistance, resolution_m=100, hop_length=2) == {
        11: 0.25,
        21: 1.0,
        42: 0.0,
    }


def test_createdir_for_file(tmp_path: Path) -> None:
    """Check that the file's folders are created, and that existing folders
    are kept.
    """
    file_path = tmp_path / "outputs" / "2026" / "connectivity.tif"
    createdir_for_file(str(file_path))
    assert file_path.parent.is_dir()
    # Only the folders are created, not the file.
    assert not file_path.exists()
    # Calling it again, when the folders exist, is not an error.
    createdir_for_file(str(file_path))
    assert file_path.parent.is_dir()


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


def test_draw_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that draw_layer draws the layer's first band, with its title,
    size and a color bar.

    plt.show is replaced by a mock, so no window opens.
    """
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = write_geotiff(tmp_path / "layer.tif", values)
    show = MagicMock()
    monkeypatch.setattr(plt, "show", show)
    draw_layer(path, "Connectivity", size=3)
    show.assert_called_once()
    figure = plt.gcf()
    image_axes, colorbar_axes = figure.axes
    np.testing.assert_array_equal(image_axes.images[0].get_array(), values)
    assert image_axes.get_title() == "Connectivity"
    assert tuple(figure.get_size_inches()) == (3, 3)
    assert colorbar_axes is not None
    plt.close(figure)
