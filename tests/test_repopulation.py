"""Tests for repopulation.py."""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib import colormaps
from matplotlib.figure import Figure

from ecoscape_connectivity import RandomPropagate

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
        for ax, (name, tensor) in zip(axes, tensors.items(), strict=True):
            values = tensor.numpy()
            image = ax.imshow(
                np.ma.masked_equal(values, 0), cmap=colormap, vmin=0, vmax=1
            )
            ax.set_title(f"{name}\nmin {values.min():.3g}, max {values.max():.3g}")
            ax.set_xticks([])
            ax.set_yticks([])
        figure.colorbar(image, ax=axes, shrink=0.8)
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
