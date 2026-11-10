"""Tests for connectivity.py.

Most tests use fake analysis functions for predictable output. They cover
the code around the simulation: reading and clipping the inputs, scaling the
permeability, padding, skipping tiles, converting the output and writing each
tile in its place. Input values are multiples of 1/16 or 1/32, which float32
and float64 hold exactly, so converting to uint8 gives the same result in
numpy and torch.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ecoscape_connectivity import (
    analyze_tiles,
    compute_connectivity,
    connectivity,
    connectivity_and_flow,
)
from tests.helpers import (
    BORDER_SIZE,
    HEIGHT,
    TILE_SIZE,
    WIDTH,
    Recorder,
    pattern,
    read_geotiff,
    write_geotiff,
)


def crop(array: np.ndarray, padding_size: int = 0) -> np.ndarray:
    """Crop an input-sized array to the output's area."""
    offset = BORDER_SIZE - padding_size
    return array[offset : HEIGHT - offset, offset : WIDTH - offset]


def run_analyze_tiles(
    tmp_path: Path, analysis: Any, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Run analyze_tiles, writing to files, and return connectivity and flow.

    kwargs override the defaults below; the inputs are among them.
    """
    connectivity_path = tmp_path / "connectivity.tif"
    flow_path = tmp_path / "flow.tif"
    analyze_tiles(
        analysis,
        **{
            "tile_size": TILE_SIZE,
            "border_size": BORDER_SIZE,
            "connectivity_layer": str(connectivity_path),
            "flow_layer": str(flow_path),
            **kwargs,
        },
    )
    return read_geotiff(connectivity_path), read_geotiff(flow_path)


@pytest.mark.parametrize("source", ["table", "layer"])
def test_inputs(tmp_path: Path, source: str) -> None:
    """Check the habitat and permeability the analysis receives: clipped to
    [0, 1], raised to the permeability exponent, padded, and in the right
    place.

    The exponent of 3 tells p ** 3 apart from p * 3, and keeps the sign of
    negative values, so clipping at 0 shows.
    """
    padding_size = 4
    habitat = pattern([-0.5, 0, 0.25, 0.75, 1, 1.5])
    # Habitat values are clipped to [0, 1].
    expected_habitat = pattern([0, 0, 0.25, 0.75, 1, 1])
    if source == "table":
        # Class 5 is missing from the table, so its permeability is 0.
        # Values are clipped to [0, 1] after the exponent: (-0.5) ** 3 to 0,
        # 1.5 ** 3 to 1.
        landcover = pattern([1, 2, 3, 4, 5], dtype=np.uint8)
        permeability_input = {
            "landcover_layer": write_geotiff(tmp_path / "landcover.tif", landcover),
            "permeability_table": {1: -0.5, 2: 0.5, 3: 1.5, 4: 1.0},
            # The padding is a landcover class: 2 has a permeability of 0.5.
            "permeability_padding": 2,
        }
        expected_permeability = pattern([0, 0.125, 1, 1, 0])
    else:
        # Values are clipped to [0, 1] before the exponent.
        permeability = pattern([-0.5, 0.25, 0.5, 1.5])
        permeability_input = {
            "permeability_layer": write_geotiff(
                tmp_path / "permeability.tif", permeability
            ),
            "permeability_padding": 0.5,
        }
        expected_permeability = pattern([0, 1 / 64, 0.125, 1])
    recorder = Recorder()
    connectivity_values, flow = run_analyze_tiles(
        tmp_path,
        recorder,
        habitat_layer=write_geotiff(tmp_path / "habitat.tif", habitat),
        permeability_exponent=3,
        padding_size=padding_size,
        habitat_padding=0.75,
        **permeability_input,
    )

    np.testing.assert_array_equal(
        connectivity_values, crop(expected_permeability, padding_size)
    )
    # The recorder returns the habitat as flow, written as log10(1 + flow) * 20.
    np.testing.assert_allclose(
        flow, crop(np.log10(1 + expected_habitat) * 20, padding_size), rtol=1e-6
    )
    # The first tile's border reaches padding_size pixels past the top of the
    # input, which is filled with the padding values: 0.5 ** 3 for the
    # permeability, and 0.75 for the habitat.
    first_habitat, first_permeability = recorder.tiles[0]
    assert (first_habitat[0, :padding_size] == 0.75).all()
    assert (first_permeability[0, :padding_size] == 0.125).all()
    # The rest of the tile is the input's top left corner: its core starts at
    # BORDER_SIZE - padding_size, and it ends a border past the core.
    end = BORDER_SIZE - padding_size + TILE_SIZE + BORDER_SIZE
    np.testing.assert_array_equal(
        first_habitat[0, padding_size:, padding_size:], expected_habitat[:end, :end]
    )


"""
Permeability values, multiples of 1/16 from 0 to 1, and the fake analysis's
results for them: connectivity from -0.25 to 1.25, and flow written as
400 * p - 19.5, from -19.5 to 380.5. Both go past [0, 255] once converted
to uint8, and none is close to a whole number, so truncation is exact.
"""
PERMEABILITY_GRID = [k / 16 for k in range(17)]


def fake_connectivity[T: (np.ndarray, torch.Tensor)](permeability: T) -> T:
    """Return connectivity from -0.25 to 1.25, for numpy or torch."""
    return 1.5 * permeability - 0.25


def fake_flow[T: (np.ndarray, torch.Tensor)](permeability: T) -> T:
    """Return a flow that is written as 400 * permeability - 19.5."""
    return 10 ** (20 * permeability - 0.975) - 1


def numpy_analysis(
    habitat: np.ndarray, permeability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fake analysis returning 2-D numpy arrays."""
    return fake_connectivity(permeability[0]), fake_flow(permeability[0])


def torch_analysis(
    habitat: np.ndarray, permeability: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake analysis returning (1, height, width) tensors, like
    analyze_tile_torch.
    """
    tensor = torch.from_numpy(permeability)
    return fake_connectivity(tensor), fake_flow(tensor)


@pytest.mark.parametrize("float_output", [True, False], ids=["float", "uint8"])
@pytest.mark.parametrize(
    "analysis", [numpy_analysis, torch_analysis], ids=["numpy", "torch"]
)
def test_output_conversion(tmp_path: Path, analysis: Any, float_output: bool) -> None:
    """Check the conversion of connectivity and flow to the output.

    Float output keeps connectivity as is, and writes flow as
    log10(1 + flow) * 20. uint8 output writes clip(connectivity * 255) and
    clip(log10(1 + flow) * 20), both clipped to [0, 255] and truncated.
    """
    permeability = pattern(PERMEABILITY_GRID)
    connectivity_values, flow = run_analyze_tiles(
        tmp_path,
        analysis,
        permeability_layer=write_geotiff(tmp_path / "permeability.tif", permeability),
        float_output=float_output,
    )
    expected_connectivity = crop(fake_connectivity(permeability.astype(float)))
    expected_flow = crop(400 * permeability.astype(float) - 19.5)
    if float_output:
        np.testing.assert_array_equal(connectivity_values, expected_connectivity)
        np.testing.assert_allclose(flow, expected_flow, atol=1e-3)
    else:
        np.testing.assert_array_equal(
            connectivity_values, np.floor(np.clip(expected_connectivity * 255, 0, 255))
        )
        np.testing.assert_array_equal(flow, np.floor(np.clip(expected_flow, 0, 255)))


def test_tile_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Check which tiles are analyzed, and that the layers' sizes must match.

    - A tile whose mean habitat equals minimum_tile_habitat exactly is
      analyzed: only a mean below it skips the tile.
    - Low permeability skips a tile only when there is no habitat, with or
      without a flow output; low habitat skips it whatever the permeability.
    - report_progress prints the number of each analyzed tile.
    - A habitat and permeability layer of different widths fail.
    """
    ones = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    ones_path = write_geotiff(tmp_path / "ones.tif", ones)

    # With padding equal to the border and padded with the same value, every
    # pixel of every tile is 0.5, so the mean is exactly 0.5.
    half = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)
    at_minimum = Recorder()
    run_analyze_tiles(
        tmp_path,
        at_minimum,
        habitat_layer=write_geotiff(tmp_path / "half.tif", half),
        permeability_layer=ones_path,
        padding_size=BORDER_SIZE,
        habitat_padding=0.5,
        minimum_tile_habitat=0.5,
    )
    # 3 rows by 4 columns of tiles.
    assert len(at_minimum.tiles) == 12

    # The left of the input has permeability 0. With 2 rows by 3 columns of
    # tiles, the first column, with its border (input columns 0-39), has a
    # mean permeability of 8 / 40 = 0.2, below minimum_tile_habitat; its
    # habitat has a mean of 1.
    left_closed = ones.copy()
    left_closed[:, :32] = 0
    left_closed_path = write_geotiff(tmp_path / "left_closed.tif", left_closed)
    without_habitat = Recorder()
    run_analyze_tiles(
        tmp_path,
        without_habitat,
        permeability_layer=left_closed_path,
        minimum_tile_habitat=0.25,
    )
    assert len(without_habitat.tiles) == 4
    # The same without a flow output, reporting progress: tiles are numbered
    # by row, so the skipped first column is tiles 0 and 3.
    without_flow = Recorder()
    analyze_tiles(
        without_flow,
        permeability_layer=left_closed_path,
        tile_size=TILE_SIZE,
        border_size=BORDER_SIZE,
        minimum_tile_habitat=0.25,
        connectivity_layer=str(tmp_path / "without_flow_connectivity.tif"),
        compute_flow=False,
        report_progress=True,
    )
    assert len(without_flow.tiles) == 4
    assert capsys.readouterr().out == "1 2 4 5 \n"
    with_habitat = Recorder()
    run_analyze_tiles(
        tmp_path,
        with_habitat,
        habitat_layer=ones_path,
        permeability_layer=left_closed_path,
        minimum_tile_habitat=0.25,
    )
    assert len(with_habitat.tiles) == 6
    # Low habitat skips the same tiles, whatever the permeability.
    low_habitat = Recorder()
    run_analyze_tiles(
        tmp_path,
        low_habitat,
        habitat_layer=left_closed_path,
        permeability_layer=ones_path,
        minimum_tile_habitat=0.25,
    )
    assert len(low_habitat.tiles) == 4

    wider = np.ones((HEIGHT, WIDTH + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="different sizes"):
        run_analyze_tiles(
            tmp_path,
            Recorder(),
            habitat_layer=ones_path,
            permeability_layer=write_geotiff(tmp_path / "wider.tif", wider),
        )


def test_flow_on_or_off(tmp_path: Path) -> None:
    """Check that analyze_tiles writes flow only when compute_flow is True,
    and that it needs a flow layer on disk.
    """
    arguments: dict[str, Any] = {
        "permeability_layer": write_geotiff(
            tmp_path / "permeability.tif", pattern([0.25, 0.5, 1])
        ),
        "tile_size": TILE_SIZE,
        "border_size": BORDER_SIZE,
    }
    # With compute_flow=False, the flow file isn't created, even when
    # flow_layer names it.
    flow_path = tmp_path / "flow.tif"
    connectivity_path = tmp_path / "connectivity.tif"
    result = analyze_tiles(
        Recorder(),
        **arguments,
        connectivity_layer=str(connectivity_path),
        flow_layer=str(flow_path),
        compute_flow=False,
    )
    assert result is None
    assert not flow_path.exists()
    assert (read_geotiff(connectivity_path) > 0).any()

    with pytest.raises(ValueError, match="flow_layer"):
        analyze_tiles(
            Recorder(),
            **arguments,
            connectivity_layer=str(tmp_path / "other_connectivity.tif"),
        )


def test_in_memory(tmp_path: Path) -> None:
    """Check that without a connectivity layer, analyze_tiles returns the
    layers, the same as written to files, with their profile; and without
    flow when compute_flow is False.
    """
    permeability = pattern(PERMEABILITY_GRID)
    permeability_path = write_geotiff(tmp_path / "permeability.tif", permeability)
    on_disk = run_analyze_tiles(
        tmp_path, numpy_analysis, permeability_layer=permeability_path
    )
    layers = analyze_tiles(
        numpy_analysis,
        permeability_layer=permeability_path,
        tile_size=TILE_SIZE,
        border_size=BORDER_SIZE,
    )
    assert layers is not None
    assert layers.flow is not None
    np.testing.assert_array_equal(layers.connectivity, on_disk[0])
    np.testing.assert_array_equal(layers.flow, on_disk[1])
    assert (layers.profile["width"], layers.profile["height"]) == (
        WIDTH - 2 * BORDER_SIZE,
        HEIGHT - 2 * BORDER_SIZE,
    )
    assert layers.profile["dtype"] == "float32"

    without_flow = analyze_tiles(
        numpy_analysis,
        permeability_layer=permeability_path,
        tile_size=TILE_SIZE,
        border_size=BORDER_SIZE,
        compute_flow=False,
    )
    assert without_flow is not None
    assert without_flow.flow is None


def test_layer_checks(tmp_path: Path) -> None:
    """Check that the permeability comes from exactly one source, and that
    connectivity_and_flow needs a dispersal distance or a number of
    gap-crossings.
    """
    path = write_geotiff(tmp_path / "ones.tif", np.ones((HEIGHT, WIDTH)))
    with pytest.raises(ValueError, match="either"):
        analyze_tiles(Recorder())
    with pytest.raises(ValueError, match="either"):
        analyze_tiles(Recorder(), permeability_layer=path, landcover_layer=path)
    with pytest.raises(ValueError, match="permeability_table"):
        analyze_tiles(Recorder(), landcover_layer=path)
    with pytest.raises(ValueError, match="num_gap_crossings"):
        connectivity_and_flow(permeability_layer=path)


def test_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that connectivity_and_flow passes each setting on unchanged.

    The test replaces analyze_tile_torch and analyze_tiles with mocks, so no
    simulation runs. The simulation counts gap-crossings in pixels crossed,
    g - 1; without a dispersal distance, it is n * g.
    """
    mock_analyze_tile_torch = MagicMock()
    mock_analyze_tiles = MagicMock()
    monkeypatch.setattr(connectivity, "analyze_tile_torch", mock_analyze_tile_torch)
    monkeypatch.setattr(connectivity, "analyze_tiles", mock_analyze_tiles)
    permeability_table = {1: 0.5}
    connectivity_and_flow(
        habitat_layer="habitat.tif",
        permeability_layer="permeability.tif",
        landcover_layer="landcover.tif",
        permeability_table=permeability_table,
        permeability_exponent=0.5,
        connectivity_layer="connectivity.tif",
        flow_layer="flow.tif",
        compute_flow=False,
        dispersal_distance=7,
        gap_crossing_distance=3,
        num_simulations=12,
        seed_density=3,
        batch_size=3,
        connectivity_only_in_habitat=False,
        device="cpu",
        tile_size=50,
        border_size=8,
        padding_size=4,
        permeability_padding=0.25,
        habitat_padding=0.75,
        minimum_tile_habitat=0.01,
        float_output=False,
        report_progress=True,
    )
    mock_analyze_tile_torch.assert_called_once_with(
        device="cpu",
        seed_density=3,
        produce_gradient=False,
        dispersal=7,
        batch_size=3,
        num_simulations=12,
        gap_crossing=2,
        repopulation_only_in_habitat=False,
    )
    mock_analyze_tiles.assert_called_once_with(
        mock_analyze_tile_torch.return_value,
        habitat_layer="habitat.tif",
        permeability_layer="permeability.tif",
        landcover_layer="landcover.tif",
        permeability_table=permeability_table,
        permeability_exponent=0.5,
        connectivity_layer="connectivity.tif",
        flow_layer="flow.tif",
        compute_flow=False,
        tile_size=50,
        border_size=8,
        padding_size=4,
        permeability_padding=0.25,
        habitat_padding=0.75,
        minimum_tile_habitat=0.01,
        float_output=False,
        report_progress=True,
    )

    connectivity_and_flow(
        permeability_layer="permeability.tif",
        num_gap_crossings=4,
        gap_crossing_distance=3,
    )
    assert mock_analyze_tile_torch.call_args.kwargs["dispersal"] == 12


def test_matches_compute_connectivity(tmp_path: Path) -> None:
    """Check that connectivity_and_flow gives exactly the same layers as
    compute_connectivity, which it replaces, for the same inputs and seed.

    compute_connectivity's gap_crossing of 0 is a gap-crossing distance of 1.
    """
    habitat_path = write_geotiff(
        tmp_path / "habitat.tif", pattern([0, 1, 1], dtype=np.uint8)
    )
    permeability_path = write_geotiff(
        tmp_path / "permeability.tif", pattern([0.25, 0.5, 1])
    )
    settings: dict[str, Any] = {
        "tile_size": TILE_SIZE,
        "border_size": BORDER_SIZE,
        "num_simulations": 4,
        "random_seed": 1,
        "device": "cpu",
        "float_output": False,
    }
    compute_connectivity(
        habitat_fn=habitat_path,
        permeability_fn=permeability_path,
        connectivity_fn=str(tmp_path / "old_connectivity.tif"),
        flow_fn=str(tmp_path / "old_flow.tif"),
        dispersal=5,
        gap_crossing=0,
        **settings,
    )
    connectivity_and_flow(
        habitat_layer=habitat_path,
        permeability_layer=permeability_path,
        connectivity_layer=str(tmp_path / "new_connectivity.tif"),
        flow_layer=str(tmp_path / "new_flow.tif"),
        dispersal_distance=5,
        gap_crossing_distance=1,
        **settings,
    )
    for output in ("connectivity", "flow"):
        new = read_geotiff(tmp_path / f"new_{output}.tif")
        np.testing.assert_array_equal(new, read_geotiff(tmp_path / f"old_{output}.tif"))
        assert (new > 0).any()
