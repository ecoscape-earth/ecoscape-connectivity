"""Tests for compute_connectivity on small synthetic GeoTIFFs.

The landscape is 170 x 150 pixels. With tile_size=40 and border_size=10,
the tile cores start at x = 10, 50, 90, 130 and y = 10, 50, 90, 130, so
there are 4 x 4 tiles, and those on the right and bottom edges are partial.
Output pixel (row, column) is input pixel (row + 10, column + 10).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import ecoscape_connectivity
from ecoscape_connectivity import half_cauchy
from tests.conftest import Dump
from tests.helpers import (
    CRS,
    ORIGIN_X,
    ORIGIN_Y,
    PIXEL_SIZE,
    read_geotiff,
    write_geotiff,
)

HEIGHT, WIDTH = 150, 170
BORDER_SIZE = 10
# Terrain codes: 1 is open land, 2 is under the habitat, 3 is hard to cross,
# and 9 is missing from the dictionary, so its permeability is 0.
PERMEABILITY = {1: 0.95, 2: 1.0, 3: 0.2}
MISSING_CODE_ROWS = slice(70, 80)


def make_habitat() -> np.ndarray:
    """Return round habitat patches every 20 pixels.

    The top left tile, with its border (input rows and columns 0-59), has
    only a 3 x 3 patch: 0.25% habitat, so minimum_habitat can skip it.
    """
    rows, columns = np.mgrid[:HEIGHT, :WIDTH]
    habitat = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for row in range(10, HEIGHT, 20):
        for column in range(10, WIDTH, 20):
            habitat[(rows - row) ** 2 + (columns - column) ** 2 <= 36] = 1
    habitat[:60, :60] = 0
    habitat[29:32, 29:32] = 1
    return habitat


def make_terrain(habitat: np.ndarray) -> np.ndarray:
    """Return terrain codes: 2 under the habitat and 1 elsewhere, with a
    column band of code 3 and a row band of the missing code 9.
    """
    terrain = np.where(habitat == 1, 2, 1).astype(np.uint8)
    terrain[:, 100:110] = 3
    terrain[MISSING_CODE_ROWS, :] = 9
    return terrain


def to_output(array: np.ndarray, padding_size: int = 0) -> np.ndarray:
    """Crop an input-sized array to the output's area."""
    offset = BORDER_SIZE - padding_size
    return array[offset : HEIGHT - offset, offset : WIDTH - offset]


@pytest.fixture
def landscape(tmp_path: Path) -> dict[str, Any]:
    """Write the habitat and terrain GeoTIFFs, and return the arguments
    compute_connectivity needs to read them.
    """
    habitat = make_habitat()
    return {
        "habitat_fn": write_geotiff(tmp_path / "habitat.tif", habitat),
        "landcover_fn": write_geotiff(tmp_path / "terrain.tif", make_terrain(habitat)),
        "permeability_dict": PERMEABILITY,
    }


def run(
    tmp_path: Path, landscape: dict[str, Any], name: str, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Run compute_connectivity on the CPU and return connectivity and flow.

    kwargs override the defaults below. The outputs are written to
    tmp_path, with name to tell runs apart.
    """
    connectivity_path = tmp_path / f"{name}_connectivity.tif"
    flow_path = tmp_path / f"{name}_flow.tif"
    arguments = {
        "tile_size": 40,
        "border_size": BORDER_SIZE,
        "dispersal": 5,
        "num_simulations": 20,
        "random_seed": 1,
        "device": "cpu",
        **landscape,
        **kwargs,
    }
    # On disk, the results are only in the files.
    assert ecoscape_connectivity.compute_connectivity(
        connectivity_fn=str(connectivity_path), flow_fn=str(flow_path), **arguments
    ) == (None, None)
    return read_geotiff(connectivity_path), read_geotiff(flow_path)


def test_reproducible(tmp_path: Path, landscape: dict[str, Any]) -> None:
    """Check that the same random_seed gives identical output, and a
    different one gives different output.
    """
    connectivity_a, flow_a = run(tmp_path, landscape, "a", random_seed=1)
    connectivity_b, flow_b = run(tmp_path, landscape, "b", random_seed=1)
    connectivity_c, _ = run(tmp_path, landscape, "c", random_seed=2)
    np.testing.assert_array_equal(connectivity_a, connectivity_b)
    np.testing.assert_array_equal(flow_a, flow_b)
    assert not np.array_equal(connectivity_a, connectivity_c)


def test_reproducible_seed_zero(tmp_path: Path, landscape: dict[str, Any]) -> None:
    """Check that random_seed=0 gives identical output."""
    connectivity_a, _ = run(tmp_path, landscape, "a", random_seed=0)
    connectivity_b, _ = run(tmp_path, landscape, "b", random_seed=0)
    np.testing.assert_array_equal(connectivity_a, connectivity_b)


def test_reproducible_half_cauchy(tmp_path: Path, landscape: dict[str, Any]) -> None:
    """Check that the same random_seed gives identical output with a
    random dispersal.
    """
    dispersal = half_cauchy(median=3, truncation=9)
    connectivity_a, _ = run(tmp_path, landscape, "a", dispersal=dispersal)
    connectivity_b, _ = run(tmp_path, landscape, "b", dispersal=dispersal)
    np.testing.assert_array_equal(connectivity_a, connectivity_b)


def test_tiling(tmp_path: Path, landscape: dict[str, Any], dump: Dump) -> None:
    """Check that tiling doesn't change the results.

    Runs are random, so tiled and single-tile output can't be compared
    exactly. Instead, they must agree as well as two single-tile runs with
    different seeds do: tiling may add no difference beyond the randomness.
    """
    settings = {"num_simulations": 100}
    tiled, _ = run(tmp_path, landscape, "tiled", **settings)
    # A tile larger than the raster gives a single tile.
    single, _ = run(tmp_path, landscape, "single", tile_size=1000, **settings)
    other_seed, _ = run(
        tmp_path, landscape, "other", tile_size=1000, random_seed=2, **settings
    )
    dump(tiled=tiled, single=single, other_seed=other_seed)

    habitat = to_output(make_habitat()) == 1
    # np.corrcoef is the Pearson correlation coefficient
    # one number from −1 to 1 that measures how closely two sets of values rise and
    # fall together
    tiled_vs_single = np.corrcoef(tiled[habitat], single[habitat])[0, 1]
    seed_vs_seed = np.corrcoef(other_seed[habitat], single[habitat])[0, 1]
    assert tiled_vs_single > seed_vs_seed - 0.05
    assert abs(tiled[habitat].mean() - single[habitat].mean()) < 0.02


def test_integer_matches_float(tmp_path: Path, landscape: dict[str, Any]) -> None:
    """Check that integer output is the float output rescaled to 0-255.

    Float flow is already on the log scale, 20 * log10(1 + flow), so integer
    flow is the float flow clipped to 0-255.
    """
    float_connectivity, float_flow = run(
        tmp_path, landscape, "float", float_output=True
    )
    int_connectivity, int_flow = run(tmp_path, landscape, "int", float_output=False)
    assert int_connectivity.dtype == np.uint8
    assert int_flow.dtype == np.uint8
    np.testing.assert_array_equal(
        int_connectivity, np.clip(float_connectivity * 255, 0, 255).astype(np.uint8)
    )
    # Integer flow is computed in float32 and float flow in float64, so
    # truncating to an integer can differ by 1.
    expected_flow = np.clip(float_flow, 0, 255).astype(np.uint8)
    assert np.abs(int_flow.astype(int) - expected_flow).max() <= 1


def test_zero_outside_habitat(
    tmp_path: Path, landscape: dict[str, Any], dump: Dump
) -> None:
    """Check that connectivity is zero wherever there is no habitat."""
    connectivity, flow = run(tmp_path, landscape, "run")
    habitat = to_output(make_habitat())
    dump(habitat=habitat, connectivity=connectivity, flow=flow)
    assert (connectivity[habitat == 0] == 0).all()
    assert (connectivity[habitat == 1] > 0).any()


def test_missing_terrain_code(
    tmp_path: Path, landscape: dict[str, Any], dump: Dump
) -> None:
    """Check that a terrain code missing from the dictionary has permeability
    0, so there is no connectivity or flow on it.
    """
    connectivity, flow = run(tmp_path, landscape, "run")
    terrain = to_output(make_terrain(make_habitat()))
    missing = 9
    assert missing not in PERMEABILITY
    dump(terrain=terrain / missing, connectivity=connectivity, flow=flow)
    assert (connectivity[terrain == missing] == 0).all()
    assert (flow[terrain == missing] == 0).all()
    assert (flow[terrain != missing] > 0).any()


def test_skipped_tile(tmp_path: Path, landscape: dict[str, Any], dump: Dump) -> None:
    """Check that a tile with less habitat than minimum_habitat is skipped,
    leaving zero output, while the same tile is computed when it isn't skipped.
    """
    # The top left tile core, input rows and columns 10-49.
    tile = (slice(0, 40), slice(0, 40))
    computed, _ = run(tmp_path, landscape, "computed", minimum_habitat=0)
    skipped, skipped_flow = run(tmp_path, landscape, "skipped", minimum_habitat=0.01)
    dump(computed=computed, skipped=skipped, skipped_flow=skipped_flow)
    assert (computed[tile] > 0).any()
    assert (skipped[tile] == 0).all()
    assert (skipped_flow[tile] == 0).all()
    # The other tiles are still computed.
    assert (skipped[40:, 40:] > 0).any()


@pytest.mark.parametrize("padding_size", [0, 5])
def test_map_position(
    tmp_path: Path, landscape: dict[str, Any], padding_size: int
) -> None:
    """Check that the output is the input cropped by border_size - padding_size
    pixels on each side, in the same place on the map.
    """
    run(tmp_path, landscape, "run", padding_size=padding_size)
    offset = BORDER_SIZE - padding_size
    for output in ("connectivity", "flow"):
        with rasterio.open(tmp_path / f"run_{output}.tif") as dataset:
            assert dataset.crs == CRS
            assert (dataset.width, dataset.height) == (
                WIDTH - 2 * offset,
                HEIGHT - 2 * offset,
            )
            assert dataset.transform == from_origin(
                ORIGIN_X + offset * PIXEL_SIZE,
                ORIGIN_Y - offset * PIXEL_SIZE,
                PIXEL_SIZE,
                PIXEL_SIZE,
            )


def test_in_memory(landscape: dict[str, Any]) -> None:
    """Check that in_memory=True returns the layers as arrays, with flow
    when generate_flow_memory is True.
    """
    layers = ecoscape_connectivity.compute_connectivity(
        tile_size=40,
        border_size=BORDER_SIZE,
        dispersal=5,
        num_simulations=20,
        random_seed=1,
        device="cpu",
        in_memory=True,
        generate_flow_memory=True,
        **landscape,
    )
    assert isinstance(layers, ecoscape_connectivity.ConnectivityLayers)
    assert layers.flow is not None
    assert (layers.connectivity > 0).any()
    assert (layers.flow > 0).any()
