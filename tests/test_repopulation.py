"""Tests for repopulation.py."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ecoscape_connectivity import RandomPropagate, analyze_geotiffs, analyze_tile_torch
from tests.conftest import Dump
from tests.helpers import read_geotiff, write_geotiff


def propagate(
    habitat: torch.Tensor,
    permeability: torch.Tensor,
    seed: torch.Tensor,
    num_spreads: int,
    spread_size: int = 1,
    coin_toss_probability: float = 0,
) -> torch.Tensor:
    """Run RandomPropagate once on the CPU with a fixed random seed.

    With the default coin_toss_probability of 0, the spread reaches every
    pixel it can, so the result is the same for any random seed.
    """
    torch.manual_seed(0)
    propagator = RandomPropagate(
        habitat,
        permeability,
        num_spreads=num_spreads,
        spread_size=spread_size,
        coin_toss_probability=coin_toss_probability,
    )
    with torch.no_grad():
        return propagator(seed)


def single_seed(height: int, width: int, row: int, column: int) -> torch.Tensor:
    """Return a seed tensor with one seed at (row, column)."""
    seed = torch.zeros(height, width)
    seed[row, column] = 1
    return seed


@pytest.mark.parametrize(
    ("num_spreads", "spread_size"), [(1, 1), (3, 1), (2, 2), (1, 3)]
)
def test_spreads_into_square(num_spreads: int, spread_size: int, dump: Dump) -> None:
    """Check that one seed spreads num_spreads * spread_size pixels each way."""
    ones = torch.ones(15, 15)
    seed = single_seed(15, 15, 7, 7)
    population = propagate(ones, ones, seed, num_spreads, spread_size)
    distance = num_spreads * spread_size
    expected = torch.zeros(15, 15, dtype=torch.bool)
    expected[7 - distance : 8 + distance, 7 - distance : 8 + distance] = True
    dump(seed=seed, population=population, expected=expected.float())
    assert torch.equal(population > 0, expected)


def test_barrier(dump: Dump) -> None:
    """Check that a column of zero permeability stops the spread without gap
    crossing, and is crossed with a gap crossing of 1.
    """
    habitat = torch.ones(9, 15)
    permeability = torch.ones(9, 15)
    permeability[:, 7] = 0
    seed = single_seed(9, 15, 4, 2)
    no_gap_crossing = propagate(habitat, permeability, seed, 20, spread_size=1)
    gap_crossing = propagate(habitat, permeability, seed, 20, spread_size=2)
    dump(
        permeability=permeability,
        seed=seed,
        no_gap_crossing=no_gap_crossing,
        gap_crossing=gap_crossing,
    )

    assert (no_gap_crossing[:, :7] > 0).all()
    assert (no_gap_crossing[:, 7:] == 0).all()
    assert (gap_crossing[:, 7] == 0).all()
    assert (gap_crossing[:, 8:] > 0).all()


def test_seed_outside_habitat(dump: Dump) -> None:
    """Check that a seed where there is no habitat produces no population."""
    habitat = torch.ones(9, 9)
    habitat[4, 4] = 0
    seed = single_seed(9, 9, 4, 4)
    population = propagate(habitat, torch.ones(9, 9), seed, 5)
    dump(habitat=habitat, seed=seed, population=population)
    assert (population == 0).all()


def test_no_habitat(dump: Dump) -> None:
    """Check that a habitat of all zeros produces no population."""
    population = propagate(torch.zeros(9, 9), torch.ones(9, 9), torch.ones(9, 9), 5)
    dump(population=population)
    assert (population == 0).all()


def test_population_within_permeability(dump: Dump) -> None:
    """Check that the population is between 0 and the permeability everywhere."""
    torch.manual_seed(1)
    habitat = torch.rand(30, 30)
    permeability = torch.rand(30, 30)
    seed = (torch.rand(30, 30) < 0.05).float()
    population = propagate(
        habitat, permeability, seed, 20, spread_size=2, coin_toss_probability=0.5
    )
    dump(habitat=habitat, permeability=permeability, seed=seed, population=population)
    assert (population >= 0).all()
    assert (population <= permeability).all()


def analyze_tile(
    habitat: np.ndarray,
    permeability: np.ndarray,
    produce_gradient: bool = True,
    repopulation_only_in_habitat: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run analyze_tile_torch on 2-D arrays on the CPU with a fixed random seed.

    Returns 2-D connectivity and flow. Flow is only computed when
    produce_gradient is True; otherwise it is all zeros.
    """
    torch.manual_seed(0)
    analyze = analyze_tile_torch(
        device="cpu",
        dispersal=5,
        num_simulations=20,
        produce_gradient=produce_gradient,
        repopulation_only_in_habitat=repopulation_only_in_habitat,
    )
    # analyze_tile_torch expects tiles with one band: (1, height, width).
    connectivity, flow = analyze(habitat[np.newaxis], permeability[np.newaxis])
    return connectivity[0].detach(), flow[0].detach()


def test_tile_bounds(dump: Dump) -> None:
    """Check that connectivity is in [0, 1] and flow is never negative."""
    rng = np.random.default_rng(1)
    habitat = rng.random((30, 30))
    permeability = rng.random((30, 30))
    connectivity, flow = analyze_tile(habitat, permeability)
    dump(
        habitat=torch.from_numpy(habitat),
        permeability=torch.from_numpy(permeability),
        connectivity=connectivity,
        flow=flow,
    )
    assert (connectivity >= 0).all()
    assert (connectivity <= 1).all()
    assert (flow >= 0).all()


def test_tile_connectivity_only_in_habitat(dump: Dump) -> None:
    """Check that repopulation_only_in_habitat confines connectivity to the
    habitat, and that without it connectivity spreads outside the habitat.
    """
    habitat = np.ones((30, 30))
    habitat[:, 10:20] = 0
    permeability = np.ones((30, 30))
    only_in_habitat, _ = analyze_tile(
        habitat, permeability, produce_gradient=False, repopulation_only_in_habitat=True
    )
    everywhere, _ = analyze_tile(
        habitat,
        permeability,
        produce_gradient=False,
        repopulation_only_in_habitat=False,
    )
    dump(
        habitat=torch.from_numpy(habitat),
        only_in_habitat=only_in_habitat,
        everywhere=everywhere,
    )
    assert (only_in_habitat[:, 10:20] == 0).all()
    assert (everywhere[:, 10:20] > 0).any()


def test_tile_no_flow_without_permeability(dump: Dump) -> None:
    """Check that flow is zero where permeability is zero."""
    habitat = np.ones((30, 30))
    permeability = np.ones((30, 30))
    permeability[:, 15] = 0
    connectivity, flow = analyze_tile(habitat, permeability)
    dump(
        permeability=torch.from_numpy(permeability),
        connectivity=connectivity,
        flow=flow,
    )
    assert (flow[:, 15] == 0).all()
    # Flow elsewhere, so the zero column is not just an empty result.
    assert (flow > 0).any()


def test_tile_higher_permeability(dump: Dump) -> None:
    """Check that higher permeability gives higher connectivity.

    Connectivity can never exceed the permeability, so a lower permeability
    trivially gives lower connectivity. Instead, compare connectivity as a
    fraction of the permeability: with 0.5, the population fades within a few
    spreads, so it reaches less of the landscape than with 1.
    """
    habitat = np.ones((30, 30))
    low, _ = analyze_tile(habitat, np.full((30, 30), 0.5), produce_gradient=False)
    high, _ = analyze_tile(habitat, np.ones((30, 30)), produce_gradient=False)
    dump(low_permeability=low, high_permeability=high)
    assert low.mean() / 0.5 < high.mean()


def test_tile_batch_size() -> None:
    """Check that num_simulations must be a multiple of batch_size."""
    analyze = analyze_tile_torch(device="cpu", num_simulations=10, batch_size=3)
    with pytest.raises(AssertionError, match="multiple"):
        analyze(np.ones((1, 10, 10)), np.ones((1, 10, 10)))


"""
These tests for analyze_geotiffs with fake analysis functions for predictable
output. These cover the code around the simulation: reading and clipping the
inputs, scaling the permeability, padding, skipping tiles, converting the output
and writing each tile in its place. Input values are multiples of 1/16 or 1/32,
which float32 and float64 hold exactly, so converting to uint8 gives the same
result in numpy and torch.
"""


HEIGHT, WIDTH = 50, 70
BORDER_SIZE = 10
TILE_SIZE = 20


def pattern(values: list[float], dtype: type = np.float32) -> np.ndarray:
    """Return a HEIGHT x WIDTH array that cycles through values, so that
    neighboring pixels differ and a tile written in the wrong place shows.
    """
    rows, columns = np.mgrid[:HEIGHT, :WIDTH]
    return np.array(values, dtype=dtype)[(3 * rows + 5 * columns) % len(values)]


def crop(array: np.ndarray, padding_size: int = 0) -> np.ndarray:
    """Crop an input-sized array to the output's area."""
    offset = BORDER_SIZE - padding_size
    return array[offset : HEIGHT - offset, offset : WIDTH - offset]


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
        # Tiles arrive with a leading band axis, (1, height, width);
        # analyze_geotiffs expects numpy results without it.
        return permeability[0], habitat[0]


def run_analyze_geotiffs(
    tmp_path: Path, analysis_fn: Any, **kwargs: Any
) -> list[np.ndarray]:
    """Run analyze_geotiffs and return the connectivity and flow outputs.

    kwargs override the defaults below; the inputs are among them.
    """
    connectivity_path = tmp_path / "connectivity.tif"
    flow_path = tmp_path / "flow.tif"
    analyze_geotiffs(
        **{
            "analysis_fn": analysis_fn,
            "tile_size": TILE_SIZE,
            "border_size": BORDER_SIZE,
            "output_repop_fn": str(connectivity_path),
            "output_grad_fn": str(flow_path),
            **kwargs,
        }
    )
    return [read_geotiff(connectivity_path), read_geotiff(flow_path)]


@pytest.mark.parametrize("source", ["dictionary", "file"])
def test_inputs(tmp_path: Path, source: str) -> None:
    """Check the habitat and permeability the analysis receives: clipped to
    [0, 1], scaled, padded, and in the right place.

    The permeability scaling of 3 tells p ** 3 apart from p * 3, and keeps
    the sign of negative values, so clipping at 0 shows.
    """
    padding_size = 4
    habitat = pattern([-0.5, 0, 0.25, 0.75, 1, 1.5])
    # Habitat values are clipped to [0, 1].
    expected_habitat = pattern([0, 0, 0.25, 0.75, 1, 1])
    if source == "dictionary":
        # Code 5 is missing from the dictionary, so its permeability is 0.
        # Scaled values are clipped to [0, 1]: (-0.5) ** 3 to 0, 1.5 ** 3 to 1.
        landcover = pattern([1, 2, 3, 4, 5], dtype=np.uint8)
        permeability_input = {
            "landcover_fn": write_geotiff(tmp_path / "landcover.tif", landcover),
            "permeability_dictionary": {1: -0.5, 2: 0.5, 3: 1.5, 4: 1.0},
            # The padding is a landcover code: 2 has a permeability of 0.5.
            "permeability_padding": 2,
        }
        expected_permeability = pattern([0, 0.125, 1, 1, 0])
    else:
        # Values are clipped to [0, 1] before scaling.
        permeability = pattern([-0.5, 0.25, 0.5, 1.5])
        permeability_input = {
            "permeability_fn": write_geotiff(
                tmp_path / "permeability.tif", permeability
            ),
            "permeability_padding": 0.5,
        }
        expected_permeability = pattern([0, 1 / 64, 0.125, 1])
    recorder = Recorder()
    connectivity, flow = run_analyze_geotiffs(
        tmp_path,
        recorder,
        habitat_fn=write_geotiff(tmp_path / "habitat.tif", habitat),
        permeability_scaling=3,
        padding_size=padding_size,
        habitat_padding=0.75,
        float_output=True,
        **permeability_input,
    )

    np.testing.assert_array_equal(
        connectivity, crop(expected_permeability, padding_size)
    )
    # The recorder returns the habitat as flow, written as log10(1 + flow) * 20.
    np.testing.assert_allclose(
        flow,
        crop(np.log10(1 + expected_habitat) * 20, padding_size),
        rtol=1e-6,
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
    "analysis_fn", [numpy_analysis, torch_analysis], ids=["numpy", "torch"]
)
def test_output_conversion(
    tmp_path: Path, analysis_fn: Any, float_output: bool
) -> None:
    """Check the conversion of connectivity and flow to the output.

    Float output keeps connectivity as is, and writes flow as
    log10(1 + flow) * 20. uint8 output writes clip(connectivity * 255) and
    clip(log10(1 + flow) * 20), both clipped to [0, 255] and truncated.
    """
    permeability = pattern(PERMEABILITY_GRID)
    connectivity, flow = run_analyze_geotiffs(
        tmp_path,
        analysis_fn,
        permeability_fn=write_geotiff(tmp_path / "permeability.tif", permeability),
        float_output=float_output,
    )
    expected_connectivity = crop(fake_connectivity(permeability.astype(float)))
    expected_flow = crop(400 * permeability.astype(float) - 19.5)
    if float_output:
        np.testing.assert_array_equal(connectivity, expected_connectivity)
        np.testing.assert_allclose(flow, expected_flow, atol=1e-3)
    else:
        np.testing.assert_array_equal(
            connectivity, np.floor(np.clip(expected_connectivity * 255, 0, 255))
        )
        np.testing.assert_array_equal(flow, np.floor(np.clip(expected_flow, 0, 255)))


def test_tile_checks(tmp_path: Path) -> None:
    """Check which tiles are analyzed, and that the inputs' sizes must match.

    - A tile whose mean habitat equals minimum_habitat exactly is analyzed:
      only a mean below it skips the tile.
    - Low permeability skips a tile only when there is no habitat.
    - A habitat and permeability of different widths fail.
    """
    ones = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    ones_fn = write_geotiff(tmp_path / "ones.tif", ones)

    # With padding equal to the border and padded with the same value, every
    # pixel of every tile is 0.5, so the mean is exactly 0.5.
    half = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)
    at_minimum = Recorder()
    run_analyze_geotiffs(
        tmp_path,
        at_minimum,
        habitat_fn=write_geotiff(tmp_path / "half.tif", half),
        permeability_fn=ones_fn,
        padding_size=BORDER_SIZE,
        habitat_padding=0.5,
        minimum_habitat=0.5,
    )
    # 3 rows by 4 columns of tiles.
    assert len(at_minimum.tiles) == 12

    # The left of the input has permeability 0. With 2 rows by 3 columns of
    # tiles, the first column, with its border (input columns 0-39), has a
    # mean permeability of 8 / 40 = 0.2, below minimum_habitat; its habitat
    # has a mean of 1.
    left_closed = ones.copy()
    left_closed[:, :32] = 0
    left_closed_fn = write_geotiff(tmp_path / "left_closed.tif", left_closed)
    without_habitat = Recorder()
    run_analyze_geotiffs(
        tmp_path, without_habitat, permeability_fn=left_closed_fn, minimum_habitat=0.25
    )
    assert len(without_habitat.tiles) == 4
    with_habitat = Recorder()
    run_analyze_geotiffs(
        tmp_path,
        with_habitat,
        habitat_fn=ones_fn,
        permeability_fn=left_closed_fn,
        minimum_habitat=0.25,
    )
    assert len(with_habitat.tiles) == 6

    wider = np.ones((HEIGHT, WIDTH + 1), dtype=np.float32)
    with pytest.raises(AssertionError, match="same size"):
        run_analyze_geotiffs(
            tmp_path,
            Recorder(),
            habitat_fn=ones_fn,
            permeability_fn=write_geotiff(tmp_path / "wider.tif", wider),
        )
