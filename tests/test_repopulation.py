"""Tests for repopulation.py."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
import torch

from ecoscape_connectivity import (
    RandomPropagate,
    analyze_geotiffs,
    analyze_tile_torch,
    compute_connectivity,
    constant,
    repopulation,
)
from tests.conftest import Dump
from tests.helpers import (
    BORDER_SIZE,
    TILE_SIZE,
    Recorder,
    pattern,
    read_geotiff,
    write_geotiff,
)


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


def test_update_threshold(dump: Dump) -> None:
    """Check that a pixel takes a later, higher value only if it rises by more
    than update_threshold (0.05).

    One row of 7 pixels, with a seed at each end:

        seed  a  target  1  1  1  seed

    The near seed reaches the target at step 2 with a * target = 0.5; the
    far seed reaches it at step 4 with the target's own permeability. A
    rise to 0.52 is ignored; a rise to 0.6 is kept. The random jitter
    changes values by less than 0.1%.
    """
    habitat = torch.ones(1, 7)
    seed = torch.zeros(1, 7)
    seed[0, 0] = seed[0, 6] = 1
    populations: dict[float, torch.Tensor] = {}
    for target in (0.52, 0.6):
        permeability = torch.ones(1, 7)
        permeability[0, 1] = 0.5 / target
        permeability[0, 2] = target
        populations[target] = propagate(habitat, permeability, seed, 10)
    dump(below_threshold=populations[0.52], above_threshold=populations[0.6])
    assert populations[0.52][0, 2].item() == pytest.approx(0.5, abs=0.005)
    assert populations[0.6][0, 2].item() == pytest.approx(0.6, abs=0.005)


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


class FakePropagate(RandomPropagate):
    """Fake analysis class for analyze_tile_torch.

    It records its arguments and the seeds it receives, and returns
    habitat * terrain as the population for every seed, so the averaging
    in analyze_tile_torch has a known result. instances lists every
    FakePropagate created; tests clear it first.
    """

    instances: ClassVar[list["FakePropagate"]] = []

    def __init__(
        self,
        habitat: torch.Tensor,
        terrain: torch.Tensor,
        num_spreads: int = 100,
        spread_size: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            habitat,
            terrain,
            num_spreads=num_spreads,
            spread_size=spread_size,
            device=device,
        )
        self.arguments = {
            "num_spreads": num_spreads,
            "spread_size": spread_size,
            "device": device,
        }
        self.seeds: list[torch.Tensor] = []
        FakePropagate.instances.append(self)

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        self.seeds.append(seed)
        # One copy per simulation in the batch. goodness is the terrain, so
        # the gradient flows back to it.
        return (self.habitat * self.goodness).repeat(seed.shape[0], 1, 1)


def test_tile_spreads() -> None:
    """Check the number and size of spreads for a dispersal of 7 pixels:
    num_spreads = int(0.5 + dispersal / spread_size), spread_size =
    gap_crossing + 1. A fixed dispersal makes one analysis class for all
    batches; a callable one makes one per batch.
    """
    tile = np.ones((1, 10, 10))
    # gap_crossing: (num_spreads, spread_size)
    expected_spreads = {0: (7, 1), 1: (4, 2)}
    for gap_crossing, (num_spreads, spread_size) in expected_spreads.items():
        for dispersal, num_instances in ((7, 1), (constant(7), 2)):
            FakePropagate.instances.clear()
            analyze = analyze_tile_torch(
                device="cpu",
                analysis_class=FakePropagate,
                dispersal=dispersal,
                gap_crossing=gap_crossing,
                num_simulations=2,
            )
            analyze(tile, tile)
            assert len(FakePropagate.instances) == num_instances
            for instance in FakePropagate.instances:
                assert instance.arguments == {
                    "num_spreads": num_spreads,
                    "spread_size": spread_size,
                    "device": "cpu",
                }


def test_tile_seed_probability() -> None:
    """Check that each pixel is a seed with probability
    seed_density / (1 + 2 * dispersal) ** 2: 4 / 49 here.

    With 4 batches of 100 x 100 pixels, the count of seeds has a standard
    deviation of about 110 around its mean of about 3,270; a 10% tolerance
    is about 3 standard deviations. The formula's mutants change the count
    by 20% or more.
    """
    FakePropagate.instances.clear()
    torch.manual_seed(0)
    tile = np.ones((1, 100, 100))
    analyze = analyze_tile_torch(
        device="cpu",
        analysis_class=FakePropagate,
        seed_density=4,
        dispersal=3,
        num_simulations=4,
    )
    analyze(tile, tile)
    seeds = FakePropagate.instances[0].seeds
    assert len(seeds) == 4
    share = torch.cat(seeds).float().mean().item()
    assert share == pytest.approx(4 / 49, rel=0.1)


@pytest.mark.parametrize(
    "dispersal",
    [
        pytest.param(
            7,
            id="fixed",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="with a fixed dispersal, one repopulator serves all batches, "
                "and its gradient accumulates across backward() calls, so flow "
                "counts earlier batches again",
            ),
        ),
        pytest.param(constant(7), id="callable"),
    ],
)
def test_tile_averaging(dispersal: Any) -> None:
    """Check that connectivity is the mean population over all simulations,
    and flow the gradient of the population divided by num_simulations.

    The fake population is habitat * terrain for every simulation, so
    connectivity is habitat * terrain. Each batch's gradient of
    sum(population) / num_batches with respect to the terrain, times the
    terrain, is batch_size * habitat * terrain / num_batches; summed over
    the batches and divided by num_simulations, flow is
    habitat * terrain / num_batches.
    """
    FakePropagate.instances.clear()
    rng = np.random.default_rng(1)
    habitat = (rng.random((1, 10, 10)) < 0.5).astype(float)
    terrain = rng.random((1, 10, 10))
    analyze = analyze_tile_torch(
        device="cpu",
        analysis_class=FakePropagate,
        dispersal=dispersal,
        produce_gradient=True,
        num_simulations=6,
        batch_size=2,
    )
    connectivity, flow = analyze(habitat, terrain)
    expected = torch.from_numpy(habitat * terrain).float()
    torch.testing.assert_close(connectivity, expected)
    torch.testing.assert_close(flow, expected / 3)


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


"""
compute_connectivity and analyze_geotiffs are deprecated wrappers around
connectivity_and_flow and analyze_tiles. These tests replace the new
functions with mocks, so no simulation runs, and check that each wrapper
passes every setting on, translated to the new names, with its own old
defaults: a dropped argument would otherwise fall back silently to the new
function's default.
"""


def test_compute_connectivity_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that compute_connectivity passes every setting to
    connectivity_and_flow, and its own defaults.
    """
    mock = MagicMock(return_value=None)
    monkeypatch.setattr(repopulation, "connectivity_and_flow", mock)
    permeability_dict = {1: 0.5}
    result = compute_connectivity(
        habitat_fn="habitat.tif",
        permeability_fn="permeability.tif",
        permeability_scaling=0.5,
        landcover_fn="landcover.tif",
        permeability_dict=permeability_dict,
        connectivity_fn="connectivity.tif",
        flow_fn="flow.tif",
        gap_crossing=2,
        dispersal=7,
        num_gaps=5,
        num_simulations=12,
        seed_density=3,
        tile_size=50,
        border_size=8,
        padding_size=4,
        permeability_padding=0.25,
        habitat_padding=0.75,
        minimum_habitat=0.01,
        float_output=False,
        random_seed=9,
        batch_size=3,
        repopulation_only_in_habitat=False,
        device="cpu",
    )
    assert result == (None, None)
    mock.assert_called_once_with(
        habitat_layer="habitat.tif",
        permeability_layer="permeability.tif",
        landcover_layer="landcover.tif",
        permeability_table=permeability_dict,
        permeability_exponent=0.5,
        connectivity_layer="connectivity.tif",
        flow_layer="flow.tif",
        # Flow is computed because flow_fn is given.
        compute_flow=True,
        dispersal_distance=7,
        num_gap_crossings=5,
        # gap_crossing counts the pixels crossed: the distance is one more.
        gap_crossing_distance=3,
        num_simulations=12,
        seed_density=3,
        random_seed=9,
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
    )

    # The old defaults, and in memory, flow only with generate_flow_memory.
    # In memory, the layers come back as they are.
    mock.reset_mock()
    mock.return_value = layers = MagicMock()
    assert (
        compute_connectivity(
            permeability_fn="permeability.tif", num_gaps=3, in_memory=True
        )
        is layers
    )
    mock.assert_called_once_with(
        habitat_layer=None,
        permeability_layer="permeability.tif",
        landcover_layer=None,
        permeability_table=None,
        permeability_exponent=1.0,
        connectivity_layer=None,
        flow_layer=None,
        compute_flow=False,
        dispersal_distance=None,
        num_gap_crossings=3,
        gap_crossing_distance=1,
        num_simulations=400,
        seed_density=4,
        random_seed=None,
        batch_size=1,
        connectivity_only_in_habitat=True,
        device=None,
        tile_size=1000,
        border_size=100,
        padding_size=0,
        permeability_padding=0,
        habitat_padding=0,
        minimum_tile_habitat=1e-4,
        float_output=True,
    )


def test_analyze_geotiffs_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that analyze_geotiffs passes every setting to analyze_tiles, and
    its own defaults, which differ from analyze_tiles': tiles of 1024, a
    border of 64, and integer output.
    """
    mock = MagicMock(return_value=None)
    monkeypatch.setattr(repopulation, "analyze_tiles", mock)
    analysis = Recorder()
    permeability_dictionary = {1: 0.5}
    result = analyze_geotiffs(
        habitat_fn="habitat.tif",
        landcover_fn="landcover.tif",
        permeability_dictionary=permeability_dictionary,
        permeability_fn="permeability.tif",
        permeability_scaling=0.5,
        analysis_fn=analysis,
        tile_size=50,
        border_size=8,
        padding_size=4,
        permeability_padding=0.25,
        habitat_padding=0.75,
        generate_gradient=False,
        minimum_habitat=0.01,
        output_repop_fn="connectivity.tif",
        output_grad_fn="flow.tif",
        report_progress=True,
        float_output=True,
    )
    assert result == (None, None)
    mock.assert_called_once_with(
        analysis,
        habitat_layer="habitat.tif",
        permeability_layer="permeability.tif",
        landcover_layer="landcover.tif",
        permeability_table=permeability_dictionary,
        permeability_exponent=0.5,
        connectivity_layer="connectivity.tif",
        flow_layer="flow.tif",
        # No flow, even with output_grad_fn, without generate_gradient.
        compute_flow=False,
        tile_size=50,
        border_size=8,
        padding_size=4,
        permeability_padding=0.25,
        habitat_padding=0.75,
        minimum_tile_habitat=0.01,
        float_output=True,
        report_progress=True,
    )

    # The old defaults; in memory, flow by default.
    mock.reset_mock()
    mock.return_value = layers = MagicMock()
    assert (
        analyze_geotiffs(
            permeability_fn="permeability.tif", analysis_fn=analysis, in_memory=True
        )
        is layers
    )
    mock.assert_called_once_with(
        analysis,
        habitat_layer=None,
        permeability_layer="permeability.tif",
        landcover_layer=None,
        permeability_table=None,
        permeability_exponent=1.0,
        connectivity_layer=None,
        flow_layer=None,
        compute_flow=True,
        tile_size=1024,
        border_size=64,
        padding_size=0,
        permeability_padding=0,
        habitat_padding=0,
        minimum_tile_habitat=1e-4,
        float_output=False,
        report_progress=False,
    )


def test_wrapper_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the wrappers' warnings and errors:

    - each warns that it is deprecated, naming its replacement;
    - display_tiles warns that it is ignored;
    - on disk, each needs a connectivity output, and analyze_geotiffs an
      analysis function.
    """
    monkeypatch.setattr(repopulation, "connectivity_and_flow", MagicMock())
    monkeypatch.setattr(repopulation, "analyze_tiles", MagicMock())
    with pytest.warns(FutureWarning, match="use connectivity_and_flow"):
        compute_connectivity(permeability_fn="p.tif", connectivity_fn="c.tif")
    with pytest.warns(FutureWarning, match="use analyze_tiles"):
        analyze_geotiffs(
            permeability_fn="p.tif", analysis_fn=Recorder(), output_repop_fn="c.tif"
        )
    with pytest.warns(FutureWarning, match="display_tiles"):
        analyze_geotiffs(
            permeability_fn="p.tif",
            analysis_fn=Recorder(),
            output_repop_fn="c.tif",
            display_tiles=True,
        )
    with (
        pytest.warns(FutureWarning),
        pytest.raises(ValueError, match="connectivity_fn"),
    ):
        compute_connectivity(permeability_fn="p.tif")
    with (
        pytest.warns(FutureWarning),
        pytest.raises(ValueError, match="output_repop_fn"),
    ):
        analyze_geotiffs(permeability_fn="p.tif", analysis_fn=Recorder())
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="analysis_fn"):
        analyze_geotiffs(permeability_fn="p.tif", output_repop_fn="c.tif")


"""
compute_connectivity and analyze_geotiffs accept each input as a file name or
as an open scgt GeoTiff: any object with a rasterio dataset in .dataset, which
is all EcoScape reads. The tests stand in for a caller's GeoTiff with such an
object, since EcoScape no longer depends on scgt. Each checks that:
- compute_connectivity, with connectivity_and_flow replaced by a mock, passes
  the GeoTiff on as its file's path;
- analyze_geotiffs, with a fake analysis, gives the same output for the
  GeoTiff as for its file name.
"""


def check_geotiff_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    layer: str,
    path: str,
    compute_inputs: dict[str, Any],
    analyze_inputs: dict[str, Any],
) -> None:
    """Check that argument can be a GeoTiff opened from path, which reaches
    the new code as layer.

    compute_inputs and analyze_inputs are the other inputs that
    compute_connectivity and analyze_geotiffs need.
    """
    mock_connectivity_and_flow = MagicMock(return_value=None)
    monkeypatch.setattr(
        repopulation, "connectivity_and_flow", mock_connectivity_and_flow
    )
    by_file_name = run_analyze_geotiffs(
        tmp_path, Recorder(), **{argument: path}, **analyze_inputs
    )
    with rasterio.open(path) as dataset:
        geotiff: Any = SimpleNamespace(dataset=dataset)
        compute_connectivity(
            **{argument: geotiff},
            **compute_inputs,
            connectivity_fn="connectivity.tif",
            dispersal=5,
        )
        assert mock_connectivity_and_flow.call_args.kwargs[layer] == path
        by_geotiff = run_analyze_geotiffs(
            tmp_path, Recorder(), **{argument: geotiff}, **analyze_inputs
        )
    for output_by_file_name, output_by_geotiff in zip(
        by_file_name, by_geotiff, strict=True
    ):
        np.testing.assert_array_equal(output_by_geotiff, output_by_file_name)
    # The fake analysis returns the permeability as connectivity.
    assert (by_file_name[0] > 0).any()


def test_habitat_geotiff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that the habitat can be a GeoTiff."""
    permeability_fn = write_geotiff(
        tmp_path / "permeability.tif", pattern([0.25, 0.5, 1])
    )
    check_geotiff_input(
        tmp_path,
        monkeypatch,
        "habitat_fn",
        "habitat_layer",
        write_geotiff(tmp_path / "habitat.tif", pattern([0, 1, 1], dtype=np.uint8)),
        compute_inputs={"permeability_fn": permeability_fn},
        analyze_inputs={"permeability_fn": permeability_fn},
    )


def test_landcover_geotiff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that the landcover can be a GeoTiff."""
    permeability_dict = {1: 0.25, 2: 0.5, 3: 1.0}
    check_geotiff_input(
        tmp_path,
        monkeypatch,
        "landcover_fn",
        "landcover_layer",
        write_geotiff(tmp_path / "landcover.tif", pattern([1, 2, 3], dtype=np.uint8)),
        compute_inputs={"permeability_dict": permeability_dict},
        analyze_inputs={"permeability_dictionary": permeability_dict},
    )


def test_permeability_geotiff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that the permeability can be a GeoTiff."""
    check_geotiff_input(
        tmp_path,
        monkeypatch,
        "permeability_fn",
        "permeability_layer",
        write_geotiff(tmp_path / "permeability.tif", pattern([0.25, 0.5, 1])),
        compute_inputs={},
        analyze_inputs={},
    )
