"""Tests for connectivity computed from a permeability raster input."""

from pathlib import Path

import numpy as np
import pytest
from scgt import GeoTiff

import ecoscape_connectivity


@pytest.fixture
def scaled_permeability_path(tmp_path: Path) -> Path:
    """Write the test permeability clipped to [0, 1] with a maximum of 1."""
    path = tmp_path / "scaled_permeability.tif"
    permeability_path = Path(__file__).parent.parent / "assets" / "permeability.tif"
    with GeoTiff.from_file(str(permeability_path)) as permeability:
        tile = permeability.get_all_as_tile()
        tile.m = np.clip(tile.m, 0, 1)
        tile.m = tile.m / np.max(tile.m)
        with permeability.clone_shape(str(path)) as scaled:
            scaled.set_tile(tile)
    return path


@pytest.mark.parametrize("float_output", [True, False])
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


def numpy_analysis(
    habitat: np.ndarray, permeability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the permeability as connectivity, and a constant flow of 9.

    Tiles arrive with a leading band axis, (1, w, h); analyze_geotiffs expects
    numpy results without it. A flow of 9 is written as log10(1 + 9) * 20 = 20.
    """
    connectivity = np.squeeze(permeability, axis=0)
    return connectivity, np.full_like(connectivity, 9.0)


@pytest.mark.parametrize("float_output", [True, False])
def test_numpy_analysis_function(
    scaled_permeability_path: Path,
    tmp_path: Path,
    float_output: bool,
) -> None:
    """Check analyze_geotiffs writes numpy analysis results to the outputs."""
    connectivity_path = tmp_path / "connectivity.tif"
    flow_path = tmp_path / "flow.tif"
    ecoscape_connectivity.analyze_geotiffs(
        permeability_fn=str(scaled_permeability_path),
        analysis_fn=numpy_analysis,
        tile_size=512,
        # Padding equal to the border keeps the output aligned with the input.
        border_size=10,
        padding_size=10,
        # Analyze every tile; skipped tiles are written as zeros.
        minimum_habitat=0,
        output_repop_fn=str(connectivity_path),
        output_grad_fn=str(flow_path),
        float_output=float_output,
    )
    with GeoTiff.from_file(str(scaled_permeability_path)) as permeability:
        permeability_values = permeability.get_all_as_tile().m
    if float_output:
        expected_connectivity = permeability_values
        expected_flow = np.full_like(permeability_values, 20.0)
    else:
        expected_connectivity = np.clip(permeability_values * 255, 0, 255).astype(
            np.uint8
        )
        expected_flow = np.full_like(permeability_values, 20, dtype=np.uint8)
    with GeoTiff.from_file(str(connectivity_path)) as connectivity:
        np.testing.assert_array_equal(
            connectivity.get_all_as_tile().m, expected_connectivity
        )
    with GeoTiff.from_file(str(flow_path)) as flow:
        np.testing.assert_array_equal(flow.get_all_as_tile().m, expected_flow)
