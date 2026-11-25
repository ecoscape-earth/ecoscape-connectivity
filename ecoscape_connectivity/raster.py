"""Reading and writing rasters in tiles, with rasterio.

A raster is read in tiles: each tile has a core, which the analysis
produces output for, and a border around it, which gives the analysis
context. Cores tile the raster without overlap, except for a margin of
border - padding pixels on each side, which no core covers. Where a tile's
border reaches past the raster, it is filled with a padding value.

Nothing here knows about the simulation.

Ported from scgt 0.0.32 (https://github.com/ecoscape-earth/scgt): its
tiled Reader, GeoTiff.set_tile and GeoTiff.crop_to_new_file, keeping only
what EcoScape uses.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile
from rasterio.windows import Window


@dataclass(frozen=True)
class TileWindow:
    """Position of a tile in a raster.

    column and row are the raster coordinates of the core's top left
    pixel; width and height are the core's size; border is the width of the
    border on each side.
    """

    column: int
    row: int
    width: int
    height: int
    border: int


def tile_windows(
    width: int, height: int, tile_size: int, border: int, padding: int
) -> Iterator[TileWindow]:
    """Return the tiles of a width x height raster, row by row.

    Cores are tile_size x tile_size, smaller at the right and bottom edges,
    and start border - padding pixels in from the top left. A raster too
    small to leave room for a core has no tiles.
    """
    margin = border - padding
    for row in range(margin, height - margin, tile_size):
        for column in range(margin, width - margin, tile_size):
            yield TileWindow(
                column=column,
                row=row,
                width=min(tile_size, width - margin - column),
                height=min(tile_size, height - margin - row),
                border=border,
            )


def read_tile(
    dataset: DatasetReader | DatasetWriter, window: TileWindow, pad_value: float
) -> np.ndarray:
    """Return a tile's core and border as a float64 array of shape
    (bands, height + 2 * border, width + 2 * border).

    Pixels past the edge of the raster are pad_value.
    """
    border = window.border
    tile = np.full(
        (
            dataset.count,
            window.height + 2 * border,
            window.width + 2 * border,
        ),
        pad_value,
        dtype=np.float64,
    )
    # The part of the tile inside the raster.
    left = max(0, window.column - border)
    top = max(0, window.row - border)
    right = min(dataset.width, window.column + window.width + border)
    bottom = min(dataset.height, window.row + window.height + border)
    tile_left = left - (window.column - border)
    tile_top = top - (window.row - border)
    tile[
        :, tile_top : tile_top + bottom - top, tile_left : tile_left + right - left
    ] = dataset.read(window=Window(left, top, right - left, bottom - top))
    return tile


def write_core(
    dataset: DatasetWriter, window: TileWindow, tile: np.ndarray, offset: int
) -> None:
    """Write the core of a tile, without its border, to an output.

    tile has the shape read_tile returns. The core goes offset pixels up
    and to the left of its position in the input, since the output is
    cropped by offset on each side; it is clipped at the output's right and
    bottom edges, and converted to the output's data type.
    """
    border = window.border
    column = window.column - offset
    row = window.row - offset
    width = min(window.width, dataset.width - column)
    height = min(window.height, dataset.height - row)
    core = tile[:, border : border + height, border : border + width]
    dataset.write(
        core.astype(dataset.dtypes[0]), window=Window(column, row, width, height)
    )


def create_output(
    source: DatasetReader | DatasetWriter,
    border: int,
    dtype: str,
    destination: str | Path | MemoryFile,
) -> DatasetWriter:
    """Create an output for a source raster, cropped by border on each
    side, and return it open for writing and reading.

    destination is a path, or a MemoryFile for an output in memory; the
    caller keeps the MemoryFile, since the output can't outlive it. The
    output has the source's position on the map, starts at 0, and has no
    nodata value.
    """
    cropped = Window(
        border, border, source.width - 2 * border, source.height - 2 * border
    )
    profile = dict(source.profile)
    profile.update(
        width=cropped.width,
        height=cropped.height,
        transform=source.window_transform(cropped),
        dtype=dtype,
        nodata=None,
        compress="LZW",
        bigtiff="YES",
    )
    if isinstance(destination, MemoryFile):
        return destination.open(**profile)
    return rasterio.open(destination, "w+", **profile)
