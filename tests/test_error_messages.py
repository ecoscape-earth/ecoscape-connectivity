"""Tests for the error messages of common failures (issue #41).

Each test checks that a wrong file path or mismatched layers give a message
that names the cause and the fix, instead of an error from inside a library.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ecoscape_connectivity import (
    analyze_geotiffs,
    compute_connectivity,
    read_transmission_csv,
)
from tests.helpers import ORIGIN_X, ORIGIN_Y, PIXEL_SIZE, write_geotiff

HEIGHT, WIDTH = 20, 30


@pytest.fixture
def habitat(tmp_path: Path) -> str:
    """Return the file name of a habitat GeoTIFF that is habitat everywhere."""
    return write_geotiff(tmp_path / "habitat.tif", np.ones((HEIGHT, WIDTH), np.uint8))


@pytest.fixture
def permeability(tmp_path: Path) -> str:
    """Return the file name of a permeability GeoTIFF on the same grid as habitat."""
    return write_geotiff(
        tmp_path / "permeability.tif", np.ones((HEIGHT, WIDTH), np.float32)
    )


def run(tmp_path: Path, **kwargs: Any) -> None:
    """Run compute_connectivity on small inputs; kwargs override the defaults."""
    compute_connectivity(
        **{
            "connectivity_fn": str(tmp_path / "connectivity.tif"),
            "dispersal": 2,
            "num_simulations": 1,
            "tile_size": 10,
            "border_size": 2,
            "device": "cpu",
            **kwargs,
        }
    )


@pytest.mark.parametrize("name", ["habitat_fn", "permeability_fn", "landcover_fn"])
def test_missing_input_file(tmp_path: Path, permeability: str, name: str) -> None:
    """Check that a wrong input path names the argument and the full path."""
    missing = tmp_path / "typo.tif"
    inputs: dict[str, Any] = {"permeability_fn": permeability, name: str(missing)}
    if name == "landcover_fn":
        inputs = {"landcover_fn": str(missing), "permeability_dict": {1: 1.0}}
    with pytest.raises(FileNotFoundError, match=f"{name} file not found") as error:
        run(tmp_path, **inputs)
    assert str(missing) in str(error.value)
    assert "Check the path for typos" in str(error.value)


def test_input_is_folder(tmp_path: Path, permeability: str) -> None:
    """Check that a folder given as an input file says so."""
    with pytest.raises(IsADirectoryError, match="habitat_fn is a folder"):
        run(tmp_path, habitat_fn=str(tmp_path), permeability_fn=permeability)


def test_input_is_not_geotiff(tmp_path: Path, permeability: str) -> None:
    """Check that a file that isn't a GeoTIFF, such as the permeability CSV, says so."""
    csv_path = tmp_path / "permeability.csv"
    csv_path.write_text("map_code,transmission\n1,1\n")
    with pytest.raises(ValueError, match="habitat_fn can't be opened as a GeoTIFF"):
        run(tmp_path, habitat_fn=str(csv_path), permeability_fn=permeability)


@pytest.mark.parametrize("name", ["habitat_fn", "permeability_fn", "landcover_fn"])
def test_input_of_wrong_type(tmp_path: Path, permeability: str, name: str) -> None:
    """Check that a pathlib.Path input says to convert it with str()."""
    inputs: dict[str, Any] = {
        "permeability_fn": permeability,
        "permeability_dict": {1: 1.0},
        name: Path(permeability),
    }
    with pytest.raises(
        TypeError,
        match=rf"{name} must be a file name \(str\) or a GeoTiff, not \w*Path",
    ):
        run(tmp_path, **inputs)


def test_no_permeability_input(tmp_path: Path, habitat: str) -> None:
    """Check that leaving out both permeability inputs lists them."""
    with pytest.raises(ValueError, match="No permeability input: give permeability_fn"):
        run(tmp_path, habitat_fn=habitat)


def test_landcover_without_dictionary(tmp_path: Path, permeability: str) -> None:
    """Check that landcover without a dictionary says how to read one."""
    with pytest.raises(
        ValueError, match=r"landcover_fn needs permeability_dict.*read_transmission_csv"
    ):
        run(tmp_path, landcover_fn=permeability)


@pytest.mark.parametrize(
    "output, description",
    [("connectivity_fn", "connectivity"), ("flow_fn", "flow")],
)
def test_missing_output_folder(
    tmp_path: Path, permeability: str, output: str, description: str
) -> None:
    """Check that an output in a folder that doesn't exist names the folder."""
    folder = tmp_path / "missing"
    with pytest.raises(
        FileNotFoundError,
        match=f"The folder for the {description} output doesn't exist: "
        + re.escape(str(folder)),
    ):
        run(tmp_path, permeability_fn=permeability, **{output: str(folder / "x.tif")})


def test_missing_output_file_name(tmp_path: Path, permeability: str) -> None:
    """Check that a missing output file name says how to give one."""
    with pytest.raises(ValueError, match="No file name for the connectivity output"):
        run(tmp_path, permeability_fn=permeability, connectivity_fn=None)


def test_missing_output_file_name_analyze_geotiffs(permeability: str) -> None:
    """Check analyze_geotiffs, which names the output output_repop_fn."""
    with pytest.raises(ValueError, match="output_repop_fn in analyze_geotiffs"):
        analyze_geotiffs(permeability_fn=permeability, analysis_fn=lambda h, p: (h, p))


@pytest.mark.parametrize("name", ["permeability_fn", "landcover_fn"])
def test_different_size(tmp_path: Path, habitat: str, name: str) -> None:
    """Check that rasters of different sizes give both sizes."""
    wider = write_geotiff(
        tmp_path / "wider.tif", np.ones((HEIGHT, WIDTH + 1), np.uint8)
    )
    with pytest.raises(
        ValueError,
        match=f"habitat_fn and {name} must have the same size, but habitat_fn is "
        f"{WIDTH} x {HEIGHT} pixels and {name} is {WIDTH + 1} x {HEIGHT} pixels",
    ):
        run(tmp_path, habitat_fn=habitat, **{name: wider, "permeability_dict": {1: 1}})


def test_different_crs(tmp_path: Path, habitat: str) -> None:
    """Check that rasters in different coordinate reference systems give both."""
    # UTM zone 11N, next to the zone 10N of the habitat.
    other_zone = write_geotiff(
        tmp_path / "other_zone.tif",
        np.ones((HEIGHT, WIDTH), np.float32),
        crs="EPSG:32611",
    )
    with pytest.raises(
        ValueError,
        match="must use the same coordinate reference system, "
        "but habitat_fn uses EPSG:32610 and permeability_fn uses EPSG:32611",
    ):
        run(tmp_path, habitat_fn=habitat, permeability_fn=other_zone)


@pytest.mark.parametrize(
    "origin, pixel_size",
    [
        pytest.param(
            (ORIGIN_X + PIXEL_SIZE / 2, ORIGIN_Y), PIXEL_SIZE, id="half-pixel-shift"
        ),
        pytest.param((ORIGIN_X, ORIGIN_Y), PIXEL_SIZE * 2, id="other-pixel-size"),
    ],
)
def test_different_grid(
    tmp_path: Path, habitat: str, origin: tuple[float, float], pixel_size: float
) -> None:
    """Check that rasters of the same size on different pixel grids give both grids."""
    shifted = write_geotiff(
        tmp_path / "shifted.tif",
        np.ones((HEIGHT, WIDTH), np.float32),
        origin=origin,
        pixel_size=pixel_size,
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "habitat_fn and permeability_fn must be on the same pixel grid, but "
            f"habitat_fn has pixels of {float(PIXEL_SIZE)} x {float(PIXEL_SIZE)} "
            f"with the top left corner at ({float(ORIGIN_X)}, {float(ORIGIN_Y)}), "
            f"and permeability_fn has pixels of {float(pixel_size)} x "
            f"{float(pixel_size)} with the top left corner at "
            f"({float(origin[0])}, {float(origin[1])})"
        ),
    ):
        run(tmp_path, habitat_fn=habitat, permeability_fn=shifted)


def test_tiny_grid_difference_is_allowed(tmp_path: Path, habitat: str) -> None:
    """Check that a grid shifted by a ten-thousandth of a pixel, as from
    rounding, is accepted.
    """
    nearly_aligned = write_geotiff(
        tmp_path / "nearly_aligned.tif",
        np.ones((HEIGHT, WIDTH), np.float32),
        origin=(ORIGIN_X + PIXEL_SIZE * 1e-4, ORIGIN_Y),
    )
    run(tmp_path, habitat_fn=habitat, permeability_fn=nearly_aligned)
    assert (tmp_path / "connectivity.tif").exists()


def test_missing_csv(tmp_path: Path) -> None:
    """Check that a wrong permeability CSV path gives the full path."""
    missing = tmp_path / "typo.csv"
    with pytest.raises(
        FileNotFoundError,
        match="Permeability CSV file not found: " + re.escape(str(missing)),
    ):
        read_transmission_csv(str(missing))


@pytest.mark.parametrize(
    "contents, columns",
    [
        pytest.param(
            "code,transmission\n1,1\n", "code, transmission", id="no-map-code"
        ),
        pytest.param(
            "map_code,permeability\n1,1\n", "map_code, permeability", id="no-value"
        ),
        pytest.param("", "none, the file is empty", id="empty"),
    ],
)
def test_csv_without_columns(tmp_path: Path, contents: str, columns: str) -> None:
    """Check that a CSV without the needed columns lists the columns it has."""
    csv_path = tmp_path / "permeability.csv"
    csv_path.write_text(contents)
    with pytest.raises(
        ValueError,
        match="needs a map_code column, and a transmission or resistance column, "
        f"but its columns are: {columns}",
    ):
        read_transmission_csv(str(csv_path))


@pytest.mark.parametrize(
    "row, message",
    [
        pytest.param(
            "forest,0.5", "map_code 'forest' and transmission '0.5'", id="code"
        ),
        pytest.param("21,high", "map_code '21' and transmission 'high'", id="value"),
        pytest.param("21", "map_code '21' and transmission None", id="missing-value"),
    ],
)
def test_csv_with_bad_value(tmp_path: Path, row: str, message: str) -> None:
    """Check that a value that isn't a number gives its line."""
    csv_path = tmp_path / "permeability.csv"
    csv_path.write_text(f"map_code,transmission\n11,0.25\n{row}\n")
    with pytest.raises(
        ValueError, match=f"Line 3 of the permeability CSV .* has {message}"
    ):
        read_transmission_csv(str(csv_path))


def test_csv_with_byte_order_mark(tmp_path: Path) -> None:
    """Check that a CSV saved by Excel, which starts with a byte order mark, is read."""
    csv_path = tmp_path / "permeability.csv"
    csv_path.write_text("map_code,transmission\n11,0.25\n", encoding="utf-8-sig")
    assert read_transmission_csv(str(csv_path)) == {11: 0.25}
