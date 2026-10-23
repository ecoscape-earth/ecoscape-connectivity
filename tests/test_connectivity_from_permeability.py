"""Tests for compute_connectivity with a permeability raster input."""

from pathlib import Path

import numpy as np
import pytest
from scgt import GeoTiff

import ecoscape_connectivity


@pytest.fixture
def scaled_permeability_path(tmp_path: Path) -> Path:
    """Write the test permeability clipped to [0, 1] with a maximum of 1."""
    path = tmp_path / "scaled_permeability.tif"
    permeability_path = (
        Path(__file__).parent.parent / "assets" / "permeability.tif"
    )
    with GeoTiff.from_file(str(permeability_path)) as permeability:
        tile = permeability.get_all_as_tile()
        tile.m = np.clip(tile.m, 0, 1)
        tile.m = tile.m / np.max(tile.m)
        with permeability.clone_shape(str(path)) as scaled:
            scaled.set_tile(tile)
    return path


@pytest.mark.parametrize(
    "float_output", [True, False]
)
def test_connectivity_from_permeability(
    scaled_permeability_path: Path,
    tmp_path: Path,
    float_output: bool,
) -> None:
    """Check output size and data type for float and integer output."""
    connectivity_path = tmp_path / "connectivity.tif"
    flow_path = tmp_path / "flow.tif"
    ecoscape_connectivity.compute_connectivity(
        permeability_fn=str(scaled_permeability_path),
        connectivity_fn=str(connectivity_path),
        flow_fn=str(flow_path),
        tile_size=512,
        # Padding equal to the border keeps the output the input size.
        border_size=10,
        padding_size=10,
        num_simulations=1,
        dispersal=10,
        float_output=float_output,
    )
    with GeoTiff.from_file(str(scaled_permeability_path)) as permeability:
        expected_size = permeability.size
    for output_path in (connectivity_path, flow_path):
        with GeoTiff.from_file(str(output_path)) as output:
            assert output.size == expected_size
            assert output.data_types[0] == ("float32" if float_output else "uint8")

