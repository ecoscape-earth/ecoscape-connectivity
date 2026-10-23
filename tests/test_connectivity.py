"""Tests for compute_connectivity with habitat and landcover inputs."""

from pathlib import Path

import pytest
from scgt import GeoTiff

import ecoscape_connectivity

ASSETS_PATH = Path(__file__).parent.parent / "assets"
HABITAT_PATH = ASSETS_PATH / "habitat_small.tif"
BORDER_SIZE = 50


@pytest.mark.parametrize(
    "tile_size, padding_size",
    [
        (200, 0),
        # Larger than the 814 x 689 test raster, so it is a single tile.
        (1000, 0),
        # Padding equal to the border keeps the border in the output.
        (1000, BORDER_SIZE),
    ],
    ids=["tiled", "single_tile", "with_border"],
)
def test_connectivity(
    tmp_path: Path, tile_size: int, padding_size: int
) -> None:
    """Check that the outputs are the input size minus the cropped border."""
    connectivity_path = tmp_path / "connectivity.tif"
    flow_path = tmp_path / "flow.tif"
    permeability_dict = ecoscape_connectivity.read_transmission_csv(
        str(ASSETS_PATH / "terrain_permeability.csv")
    )
    ecoscape_connectivity.compute_connectivity(
        habitat_fn=str(HABITAT_PATH),
        landcover_fn=str(ASSETS_PATH / "terrain_small.tif"),
        permeability_dict=permeability_dict,
        connectivity_fn=str(connectivity_path),
        flow_fn=str(flow_path),
        tile_size=tile_size,
        border_size=BORDER_SIZE,
        padding_size=padding_size,
        num_simulations=2,
        dispersal=ecoscape_connectivity.half_cauchy(15, 40),
    )
    cropped = 2 * (BORDER_SIZE - padding_size)
    with GeoTiff.from_file(str(HABITAT_PATH)) as habitat:
        expected_size = (habitat.width - cropped, habitat.height - cropped)
    for output_path in (connectivity_path, flow_path):
        with GeoTiff.from_file(str(output_path)) as output:
            assert output.size == expected_size
