"""Tests for raster.py."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from ecoscape_connectivity.raster import (
    TileWindow,
    create_output,
    read_tile,
    tile_windows,
    write_core,
)
from tests.helpers import PIXEL_SIZE, write_geotiff

# (width, height, tile_size, border, padding): padding 0, equal to the border
# and in between; cores that tile the raster exactly (48 - 2 * 4 = 40 and
# 28 - 2 * 4 = 20, with tiles of 10) and cores with partial last tiles; and
# a raster smaller than one tile.
TILINGS = [
    (37, 29, 10, 4, 0),
    (37, 29, 10, 4, 4),
    (37, 29, 10, 4, 2),
    (48, 28, 10, 4, 0),
    (12, 9, 20, 3, 1),
]


def test_tile_windows() -> None:
    """Check the tiles of a small raster: cores start border - padding in,
    and the last row and column of cores are partial.
    """
    # The margin is 2 - 1 = 1, so cores cover columns 1-8 and rows 1-5.
    assert list(tile_windows(10, 7, 4, 2, 1)) == [
        TileWindow(column=1, row=1, width=4, height=4, border=2),
        TileWindow(column=5, row=1, width=4, height=4, border=2),
        TileWindow(column=1, row=5, width=4, height=1, border=2),
        TileWindow(column=5, row=5, width=4, height=1, border=2),
    ]
    # A raster with no room for a core inside its margin has no tiles.
    assert list(tile_windows(4, 4, 10, 2, 0)) == []


@pytest.mark.parametrize(("width", "height", "tile_size", "border", "padding"), TILINGS)
def test_tile_windows_cover_core_once(
    width: int, height: int, tile_size: int, border: int, padding: int
) -> None:
    """Check that the cores cover every pixel inside the margin exactly once,
    and nothing outside it, with no empty tiles.
    """
    margin = border - padding
    covered = np.zeros((height, width), dtype=int)
    for window in tile_windows(width, height, tile_size, border, padding):
        assert 0 < window.width <= tile_size
        assert 0 < window.height <= tile_size
        assert window.border == border
        covered[
            window.row : window.row + window.height,
            window.column : window.column + window.width,
        ] += 1
    expected = np.zeros((height, width), dtype=int)
    expected[margin : height - margin, margin : width - margin] = 1
    np.testing.assert_array_equal(covered, expected)


def test_read_tile_padding() -> None:
    """Check that a tile whose border reaches past every edge of a two-band
    raster holds the raster in the middle and the padding value around it.
    """
    values = np.arange(2 * 4 * 5, dtype=np.uint8).reshape(2, 4, 5)
    with (
        MemoryFile() as memory_file,
        memory_file.open(
            driver="GTiff", width=5, height=4, count=2, dtype="uint8"
        ) as dataset,
    ):
        dataset.write(values)
        window = TileWindow(column=0, row=0, width=5, height=4, border=2)
        tile = read_tile(dataset, window, 0.5)
    assert tile.shape == (2, 8, 9)
    assert tile.dtype == np.float64
    np.testing.assert_array_equal(tile[:, 2:6, 2:7], values)
    # The padding value survives in a uint8 raster: the tile is float64.
    padding = np.ones(tile.shape, dtype=bool)
    padding[:, 2:6, 2:7] = False
    assert (tile[padding] == 0.5).all()


def test_write_core_clipping() -> None:
    """Check that write_core writes the core without its border, moved up and
    left by offset, clipped at the output's right and bottom edges, and
    converted to the output's data type.
    """
    window = TileWindow(column=3, row=2, width=4, height=4, border=1)
    # A 6 x 6 tile, core and border, with values 0.5 to 35.5.
    tile = (np.arange(36, dtype=np.float64) + 0.5).reshape(1, 6, 6)
    with (
        MemoryFile() as memory_file,
        memory_file.open(
            driver="GTiff", width=5, height=4, count=1, dtype="uint8"
        ) as output,
    ):
        write_core(output, window, tile, offset=1)
        written = output.read()
    # With an offset of 1, the core starts at column 2, row 1, and the
    # 5 x 4 output clips it to 3 columns and 3 rows.
    expected = np.zeros((1, 4, 5), dtype=np.uint8)
    expected[:, 1:4, 2:5] = tile[:, 1:4, 1:4].astype(np.uint8)
    np.testing.assert_array_equal(written, expected)


@pytest.mark.parametrize("in_memory", [False, True], ids=["disk", "memory"])
def test_create_output(tmp_path: Path, in_memory: bool) -> None:
    """Check that the output is the source cropped by border on each side:
    its size, position on the map, projection and data type, with every
    pixel 0, on disk and in memory.
    """
    source_path = write_geotiff(
        tmp_path / "source.tif", np.ones((20, 30), dtype=np.float32)
    )
    with rasterio.open(source_path) as source, MemoryFile() as memory_file:
        destination = memory_file if in_memory else tmp_path / "output.tif"
        with create_output(source, 3, "uint8", destination) as output:
            assert (output.width, output.height) == (24, 14)
            assert output.transform == source.transform @ rasterio.Affine.translation(
                3, 3
            )
            assert output.transform.c == source.transform.c + 3 * PIXEL_SIZE
            assert output.crs == source.crs
            assert output.dtypes == ("uint8",)
            assert output.nodata is None
            assert (output.read() == 0).all()
