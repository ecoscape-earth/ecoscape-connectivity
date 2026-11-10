"""Tests for connectivity computed from a permeability raster input."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

import ecoscape_connectivity


@pytest.fixture
def scaled_permeability_path(tmp_path: Path) -> Path:
    """Write the test permeability clipped to [0, 1] with a maximum of 1."""
    path = tmp_path / "scaled_permeability.tif"
    permeability_path = Path(__file__).parent.parent / "assets" / "permeability.tif"
    with rasterio.open(permeability_path) as permeability:
        values = np.clip(permeability.read(), 0, 1)
        profile = permeability.profile
    with rasterio.open(path, "w", **profile) as scaled:
        scaled.write(values / np.max(values))
    return path


def read(path: Path) -> np.ndarray:
    """Return every band of a raster, as (bands, height, width)."""
    with rasterio.open(path) as dataset:
        return dataset.read()


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
        device="cpu",
    )
    with rasterio.open(scaled_permeability_path) as permeability:
        expected_size = (permeability.width, permeability.height)
    for output_path in (connectivity_path, flow_path):
        with rasterio.open(output_path) as output:
            assert (output.width, output.height) == expected_size
            assert output.dtypes[0] == ("float32" if float_output else "uint8")


def numpy_analysis(
    habitat: np.ndarray, permeability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the permeability as connectivity, and a constant flow of 9.

    Tiles arrive with a leading band axis, (1, w, h); the results can leave it
    out. A flow of 9 is written as log10(1 + 9) * 20 = 20.
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
        # Analyze every tile; skipped tiles are left at zero.
        minimum_habitat=0,
        output_repop_fn=str(connectivity_path),
        output_grad_fn=str(flow_path),
        float_output=float_output,
    )
    permeability_values = read(scaled_permeability_path)
    if float_output:
        expected_connectivity = permeability_values
        expected_flow = np.full_like(permeability_values, 20.0)
    else:
        expected_connectivity = np.clip(permeability_values * 255, 0, 255).astype(
            np.uint8
        )
        expected_flow = np.full_like(permeability_values, 20, dtype=np.uint8)
    np.testing.assert_array_equal(read(connectivity_path), expected_connectivity)
    np.testing.assert_array_equal(read(flow_path), expected_flow)
