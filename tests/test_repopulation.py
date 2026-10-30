"""Tests for repopulation.py."""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from ecoscape_connectivity import RandomPropagate, analyze_tile_torch

Dump = Callable[..., None]


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


@pytest.fixture
def dump(request: pytest.FixtureRequest) -> Dump:
    """Return a function that draws tensors side by side as an image, for inspection.

    Each test writes to tests/output/<test file>/<test name>.png, which
    is kept after the run and ignored by git. Parametrized names such as
    test_spreads[1-1] become test_spreads_1-1.
    """
    file_name = re.sub(r"[^\w-]+", "_", request.node.name).strip("_") + ".png"
    path = Path(__file__).parent / "output" / request.path.stem / file_name
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(**tensors: torch.Tensor) -> None:
        figure = Figure(figsize=(3 * len(tensors) + 1, 3.5), layout="constrained")
        figure.suptitle(f"{request.node.name} (gray: exactly 0)")
        # Exact zeros are masked and drawn in gray, so they stand out from
        # small values, which are nearly black.
        colormap = colormaps["inferno"].with_extremes(bad="lightgray")
        axes = figure.subplots(1, len(tensors), squeeze=False)[0]
        # Values in [0, 1] share one colorbar; others, such as flow, get their own.
        shared_image: AxesImage | None = None
        for ax, (name, tensor) in zip(axes, tensors.items(), strict=True):
            values = tensor.detach().numpy()
            vmax = max(1.0, values.max())
            image = ax.imshow(
                np.ma.masked_equal(values, 0), cmap=colormap, vmin=0, vmax=vmax
            )
            if vmax > 1:
                figure.colorbar(image, ax=ax, shrink=0.8)
            else:
                shared_image = image
            ax.set_title(f"{name}\nmin {values.min():.3g}, max {values.max():.3g}")
            ax.set_xticks([])
            ax.set_yticks([])
        if shared_image is not None:
            figure.colorbar(shared_image, ax=axes, shrink=0.8)
        figure.savefig(path)

    return write


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
