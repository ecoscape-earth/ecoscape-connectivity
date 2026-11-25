"""Helpers shared by the tests: GeoTIFFs, and small landscapes with a fake
analysis.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32610"
PIXEL_SIZE = 300
# Upper left corner of the input, in meters (UTM zone 10N).
ORIGIN_X, ORIGIN_Y = 500_000, 4_100_000


def write_geotiff(path: Path, array: np.ndarray) -> str:
    """Write a one-band GeoTIFF with a known position on the map."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=CRS,
        transform=from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_SIZE, PIXEL_SIZE),
    ) as dataset:
        dataset.write(array, 1)
    return str(path)


def read_geotiff(path: Path) -> np.ndarray:
    """Return the first band of a GeoTIFF."""
    with rasterio.open(path) as dataset:
        return dataset.read(1)


# Size, border and tile size of the small landscapes in the tests with a fake
# analysis: 2 rows by 3 columns of tiles, with partial tiles at the edges.
HEIGHT, WIDTH = 50, 70
BORDER_SIZE = 10
TILE_SIZE = 20


def pattern(values: list[float], dtype: type = np.float32) -> np.ndarray:
    """Return a HEIGHT x WIDTH array that cycles through values, so that
    neighboring pixels differ and a tile written in the wrong place shows.
    """
    rows, columns = np.mgrid[:HEIGHT, :WIDTH]
    return np.array(values, dtype=dtype)[(3 * rows + 5 * columns) % len(values)]


class Recorder:
    """Fake analysis function that records the tiles it receives.

    It returns the permeability as connectivity and the habitat as flow,
    so the output shows the inputs after clipping and scaling.
    """

    def __init__(self) -> None:
        self.tiles: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(
        self, habitat: np.ndarray, permeability: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.tiles.append((habitat.copy(), permeability.copy()))
        # Tiles arrive with a leading band axis, (1, height, width); the
        # result can leave it out.
        return permeability[0], habitat[0]
