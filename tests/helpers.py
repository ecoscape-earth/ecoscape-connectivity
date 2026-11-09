"""Helpers shared by the tests, for writing and reading GeoTIFFs."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32610"
PIXEL_SIZE = 300
# Upper left corner of the input, in meters (UTM zone 10N).
ORIGIN_X, ORIGIN_Y = 500_000, 4_100_000


def write_geotiff(
    path: Path,
    array: np.ndarray,
    crs: str = CRS,
    origin: tuple[float, float] = (ORIGIN_X, ORIGIN_Y),
    pixel_size: float = PIXEL_SIZE,
) -> str:
    """Write a one-band GeoTIFF with a known position on the map.

    By default it is in CRS, with its upper left corner at (ORIGIN_X,
    ORIGIN_Y) and pixels of PIXEL_SIZE meters; tests of mismatched rasters
    change them.
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=from_origin(*origin, pixel_size, pixel_size),
    ) as dataset:
        dataset.write(array, 1)
    return str(path)


def read_geotiff(path: Path) -> np.ndarray:
    """Return the first band of a GeoTIFF."""
    with rasterio.open(path) as dataset:
        return dataset.read(1)
