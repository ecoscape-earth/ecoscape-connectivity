"""Ecological behavior on the paper's artificial landscape (#10).

The landscape follows the figure of habitat patches in a landscape matrix
in the EcoScape paper, measured from the image: 60 pixels tall and 40
wide, with six round habitat patches in three rows. A permeable matrix
connects the top four; the bottom two sit in low-permeability land. The
tests compare mean connectivity and flow over regions of the landscape,
never fixed values, since the simulation is random.

The ranges in the docstrings come from runs with seeds 0 to 9; the tests
use seed 1.
"""

import numpy as np
import pytest
import torch

from ecoscape_connectivity import analyze_tile_torch, constant
from tests.conftest import Dump

HEIGHT, WIDTH = 60, 40
# Habitat patches: (row, column, radius) of each, measured from the figure.
PATCHES = {
    "top_left_small": (8, 7, 3),
    "top_right_small": (8, 31, 3),
    "connected_left_large": (27, 9, 6),
    "connected_right_large": (27, 33, 6),
    "isolated_large": (52, 9, 6),
    "isolated_small": (52, 32, 3),
}

type Run = dict[str, np.ndarray]


def disk(row: int, column: int, radius: float) -> np.ndarray:
    """Return a HEIGHT x WIDTH mask of the pixels within radius of
    (row, column).
    """
    rows, columns = np.mgrid[:HEIGHT, :WIDTH]
    return (rows - row) ** 2 + (columns - column) ** 2 <= radius**2


def landscape(
    matrix_permeability: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return the habitat, the permeability and the masks of the regions
    the tests compare.

    Habitat patches have permeability 1, the matrix matrix_permeability,
    and the land elsewhere 0.05. The masks are each patch, by name, and:
    - matrix: the permeable land around the top four patches;
    - corridor: the matrix between the two connected large patches;
    - diagonals: the matrix between the rows, left of the notch, where
      paths cross from the top patches to the large patches;
    - around_isolated_large: the land within 4 pixels of the isolated
      large patch;
    - land: all the low-permeability land.
    """
    masks = {name: disk(*patch) for name, patch in PATCHES.items()}
    habitat = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for patch in PATCHES:
        habitat |= masks[patch]
    matrix = np.zeros((HEIGHT, WIDTH), dtype=bool)
    matrix[5:34, 3:35] = True
    # The notch on the right, between the top and middle rows.
    matrix[12:22, 21:35] = False
    matrix &= ~habitat
    masks["matrix"] = matrix
    masks["corridor"] = np.zeros((HEIGHT, WIDTH), dtype=bool)
    masks["corridor"][23:32, 16:27] = True
    masks["corridor"] &= matrix
    masks["diagonals"] = np.zeros((HEIGHT, WIDTH), dtype=bool)
    masks["diagonals"][12:21, 10:21] = True
    masks["diagonals"] &= matrix
    land = ~habitat & ~matrix
    masks["land"] = land
    masks["around_isolated_large"] = disk(52, 9, 10) & land
    permeability = np.full((HEIGHT, WIDTH), 0.05)
    permeability[matrix] = matrix_permeability
    permeability[habitat] = 1.0
    return habitat.astype(float), permeability, masks


def run_landscape(matrix_permeability: float) -> Run:
    """Run the landscape on the CPU with seed 1, and return the habitat,
    permeability, connectivity and flow, and the region masks.

    A dispersal of 40 pixels reaches the top-right small patch around the
    notch, as the figure's flow does.
    """
    habitat, permeability, masks = landscape(matrix_permeability)
    torch.manual_seed(1)
    analyze = analyze_tile_torch(
        device="cpu",
        dispersal=constant(40),
        produce_gradient=True,
        num_simulations=1000,
    )
    connectivity, flow = analyze(habitat[np.newaxis], permeability[np.newaxis])
    return {
        "habitat": habitat,
        "permeability": permeability,
        "connectivity": connectivity[0].detach().numpy(),
        "flow": flow[0].detach().numpy(),
        **masks,
    }


@pytest.fixture(scope="module")
def runs() -> dict[str, Run]:
    """Run the landscape once with the permeable matrix (0.95), and once
    with the matrix at the land's low permeability (0.05), as a control.
    """
    return {"permeable": run_landscape(0.95), "control": run_landscape(0.05)}


def mean(run: Run, output: str, region: str) -> float:
    """Return the mean of output (connectivity or flow) over a region."""
    return float(run[output][run[region]].mean())


def relative(values: np.ndarray) -> np.ndarray:
    """Return values divided by their maximum, so the saved images show
    them from low to high, like the figure.
    """
    return values / values.max()


def test_small_isolated_patch(runs: dict[str, Run], dump: Dump) -> None:
    """Small patches, far apart, low-permeability matrix: low connectivity.

    The isolated small patch has the lowest mean connectivity of the six
    patches: 0.008 to 0.025 over 10 seeds, against at least 0.048 for the
    others.
    """
    run = runs["permeable"]
    dump(
        habitat=run["habitat"],
        permeability=run["permeability"],
        connectivity_relative=relative(run["connectivity"]),
        flow_relative=relative(run["flow"]),
    )
    isolated_small = mean(run, "connectivity", "isolated_small")
    for patch in PATCHES:
        if patch != "isolated_small":
            assert isolated_small < mean(run, "connectivity", patch), patch


def test_large_isolated_patch(runs: dict[str, Run]) -> None:
    """Large patches, far apart, low-permeability matrix: more connectivity
    within the patch, but none between patches.

    The isolated large patch has more connectivity than the isolated small
    one (0.054 to 0.082 against at most 0.025 over 10 seeds). There is
    flow inside it, but none in the land around it: land permeability
    (0.05) is at most the update threshold, so animals never settle there.
    """
    run = runs["permeable"]
    assert mean(run, "connectivity", "isolated_large") > mean(
        run, "connectivity", "isolated_small"
    )
    assert mean(run, "flow", "isolated_large") > 0
    assert mean(run, "flow", "around_isolated_large") == 0


def test_permeable_matrix(runs: dict[str, Run], dump: Dump) -> None:
    """Patches connected through a permeable matrix: flow between them, and
    more connectivity.

    With the permeable matrix:
    - there is flow in the corridor between the connected large patches
      (0.00076 to 0.0010 over 10 seeds), and none in the land;
    - the connected large patches have more connectivity than the isolated
      large one: their mean is 0.098 to 0.122, at least 0.025 more than
      the isolated large patch in every seed.
    With the matrix at low permeability, the corridor has no flow, and the
    connected large patches' connectivity drops (0.052 to 0.084).
    """
    permeable, control = runs["permeable"], runs["control"]
    dump(
        permeable_connectivity=relative(permeable["connectivity"]),
        permeable_flow=relative(permeable["flow"]),
        control_connectivity=relative(control["connectivity"]),
        control_flow=relative(control["flow"]),
    )
    assert mean(permeable, "flow", "corridor") > mean(permeable, "flow", "land")
    assert mean(control, "flow", "corridor") == 0
    for patch in ("connected_left_large", "connected_right_large"):
        assert mean(permeable, "connectivity", patch) > mean(
            permeable, "connectivity", "isolated_large"
        ), patch
        assert mean(permeable, "connectivity", patch) > mean(
            control, "connectivity", patch
        ), patch


def test_multiple_paths(runs: dict[str, Run]) -> None:
    """Large-to-large and large-to-small connections: more flow on the
    large-to-large connection, and several paths, not only the strongest.

    Flow in the corridor between the large patches (0.00076 to 0.0010 over
    10 seeds) is at least twice the flow on the diagonal paths to the
    small patches (0.00027 to 0.00035), which is still above the land's.
    """
    run = runs["permeable"]
    corridor = mean(run, "flow", "corridor")
    diagonals = mean(run, "flow", "diagonals")
    assert corridor > diagonals
    assert diagonals > mean(run, "flow", "land")
