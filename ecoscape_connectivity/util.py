import csv
import os
from collections.abc import Callable, Iterable
from typing import Self

import numpy as np
import torch
from torch import nn

# We collect here a few generally useful functions, and the repopulation
# simulation.

# A dispersal distance in pixels, or a function that samples one, such as
# distributions.half_cauchy.
type Dispersal = float | Callable[[], float]
# Takes habitat and permeability tiles, and returns connectivity and flow.
type AnalysisFunction = Callable[
    [np.ndarray, np.ndarray],
    tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray],
]


def dict_translate(
    np_array: np.ndarray, my_dict: dict[int, float], default_val: float = 0
) -> np.ndarray:
    """Translates the terrain type according to a dictionary mapping
    terrain type to values.
    """
    unique_codes, codes = np.unique(np_array, return_inverse=True)
    return np.array([(my_dict.get(code, default_val)) for code in unique_codes])[
        codes
    ].reshape(np_array.shape)


def read_transmission_csv(fn: str) -> dict[int, float]:
    """Reads a dictionary of terrain resistance or transmission in csv, producing a dictionary."""
    d: dict[int, float] = {}
    with open(fn) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "resistance" in row:
                d[int(row["map_code"])] = 1.0 - float(row["resistance"])
            else:
                d[int(row["map_code"])] = float(row["transmission"])
    return d


def rescale_resistance(
    d: dict[int, float], resolution_m: float, hop_length: float
) -> dict[int, float]:
    """Resistance dictionaries are based on decay over 100m.
    This function rescales the resistance to the value of the
    actual hop length."""
    alpha = resolution_m * hop_length / 100.0
    return {k: v**alpha for k, v in d.items()}


def createdir_for_file(fn: str) -> None:
    """Ensures that the path to a file exists."""
    dirs, _ffn = os.path.split(fn)
    # print("Creating", dirs)
    os.makedirs(dirs, exist_ok=True)


class SingleIterator[T]:
    """Given an iterator, this class builds an iterator that returns
    pairs of the form (None, i), where i is given by the iterator."""

    def __init__(self, it: Iterable[T]) -> None:
        self.it = iter(it)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[None, T]:
        x = next(self.it)
        return (None, x)


class RandomPropagate(nn.Module):
    """
    Important: THIS is the function to use in the repopulation experiments.
    This module models the repopulation of the habitat from a chosen percentage
    of the seed places.  The terrain and habitat are parameters, and the input is a
    similarly sized 0-1 (float) tensor of seed points."""

    def __init__(
        self,
        habitat: torch.Tensor,
        terrain: torch.Tensor,
        num_spreads: int = 100,
        spread_size: int = 1,
        device: str | torch.device | None = None,
        update_threshold: float = 0.05,
        coin_toss_probability: float = 0.5,
    ) -> None:
        """
        :param habitat: torch tensor (2-dim) representing the habitat.
        :param terrain: torch tensor (2-dim) representing the terrain.
        :param num_spreads: number of bird spreads to use.  This is
            the number of kernel spreads, in which birds that move by
            spread_size pixels.
        :param spread_size: by how much (in pixels) birds spread in each spread.
            This is 1 + gap_crossing.
        :param device: device to use for computation (cpu, cuda, mps, etc).
        """
        super().__init__()
        # spread_size and num_spreads have to be integers, not callables here.
        self.device = device or torch.device("cpu")
        assert type(spread_size) == int, (
            "spread_size must be an int"  # pragma: no mutate
        )
        assert type(num_spreads) == int, (
            "num_spreads must be an int"  # pragma: no mutate
        )
        self.habitat = habitat
        self.goodness = torch.nn.Parameter(terrain, requires_grad=True)
        self.h, self.w = habitat.shape
        self.num_spreads = num_spreads
        self.spread_size = spread_size
        # Defines spread operator.
        self.min_transmission = 1e-4
        self.update_threshold = update_threshold
        self.coin_toss_probability = coin_toss_probability
        self.kernel_size = 1 + 2 * spread_size
        self.spreader = torch.nn.MaxPool2d(
            self.kernel_size, stride=1, padding=spread_size
        )

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        """
        seed: a 0-1 (float) tensor of seed points.
        """
        # First, we multiply the seed by the habitat, to confine the seeds to
        # where birds can live.
        x = seed * self.habitat * self.goodness
        if x.ndim < 3:
            # We put it into shape (1, w, h) because the pooling operator expects this.
            x = torch.unsqueeze(x, dim=0)
        # Now we must propagate n times.
        empty_spreads = 0
        for i in range(self.num_spreads):
            xx = x
            # Randomizes the source.
            x = x * (
                1 - self.min_transmission + self.min_transmission * torch.rand_like(x)
            )
            # Then, we propagate.
            x = self.spreader(x)
            # Coin-flip at destination.
            x = x * (torch.rand_like(x) > self.coin_toss_probability)
            x = x * self.goodness
            # And finally we combine the results.
            delta = x - xx
            x = x * (delta > self.update_threshold)
            x = torch.maximum(x, xx)
            if torch.sum(delta) == 0:
                empty_spreads += 1
            else:
                empty_spreads = 0
            if empty_spreads > 4:
                break
        if seed.ndim < 3:
            x = torch.squeeze(x, dim=0)
        return x

    def get_grad(self) -> torch.Tensor:
        assert self.goodness.grad is not None, (
            "Call backward() before get_grad()."  # pragma: no mutate
        )
        return self.goodness.grad * self.goodness


def analyze_tile_torch(
    device: str | torch.device | None = None,
    analysis_class: type[RandomPropagate] = RandomPropagate,
    seed_density: float = 4.0,
    produce_gradient: bool = False,
    batch_size: int = 1,
    dispersal: Dispersal = 20,
    num_simulations: int = 100,
    gap_crossing: int = 0,
    repopulation_only_in_habitat: bool = True,
) -> Callable[[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]:
    """This is the function that performs the analysis on a single tile.
    The input and output to this function are in cpu, but the computation occurs in
    the specified device.
    :param device: the device to be used, e.g., cpu, cuda, mps.
    :param analysis_class: class to be used for the analysis.  We recommend RandomPropagate.
        You can change this if you wish to experiment with different classes.
    :param seed_density: Consider a square of edge (1 + 2 * dispersal) pixels.
        In that square, there will be seed_density seeds on average.
    :param produce_gradient: boolean, whether to produce a gradient as result or not.
    :param batch_size: batch size for GPU calculations. For speed, use the largest
        batch size that fits in memory.
    :param dispersal: dispersal distance in pixels.
        As above, if this is an integer, we do this constanst number of spreads
        for all batches. Otherwise, If this is of the form of a function
        (probability distribution), we run the function (and sample the distribution)
        to get the dispersal distance.
    :param num_simulations: number of simulations to run. Must be a multiple of batch_size.
    :param gap_crossing: maximum number of pixels a bird can jump. 0 means only contiguous pixels.
    :param repopulation_only_in_habitat: if True, then the repopulation only occurs in the habitat.
        If False, the repopulation can be non-zero all over the output raster.
    """

    # The device depends on the machine, so tests pass device="cpu" and
    # mutmut leaves this alone. Checking for MPS in a forked process, as
    # mutmut runs tests, crashes on macOS.
    # pragma: no mutate start
    device = device or (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    # pragma: no mutate end

    assert type(gap_crossing) == int, (
        "gap_crossing must be an int"  # pragma: no mutate
    )

    # Computes the seed probability
    # seed_probability =  seed_density / ((2 * hop_length * total_spreads) ** 2)

    def f(
        habitat: np.ndarray, terrain: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, w, h = habitat.shape
        assert habitat.shape == terrain.shape, (
            "Habitat and terrain must have the same shape."  # pragma: no mutate
        )
        # We may need to do multiple spreads if the batch size is smaller than the total spreads.
        assert num_simulations % batch_size == 0, (
            "Simulations not multiples of batch"  # pragma: no mutate
        )
        num_batches = num_simulations // batch_size
        tot_grad = torch.zeros((1, w, h), dtype=torch.float, device=device)
        tot_pop = torch.zeros((1, w, h), dtype=torch.float, device=device)
        hab = torch.tensor(
            habitat.astype(float), requires_grad=False, dtype=torch.float, device=device
        ).view(w, h)
        ter = torch.tensor(
            terrain.astype(float), requires_grad=False, dtype=torch.float, device=device
        ).view(w, h)
        # If the num_spreads and spread_size are constant, then we can use a fixed repopulator, which is more efficient.
        if not callable(dispersal):
            num_spreads = int(0.5 + dispersal / (gap_crossing + 1))
            repopulator = analysis_class(
                hab,
                ter,
                num_spreads=num_spreads,
                spread_size=gap_crossing + 1,
                device=device,
            ).to(device)
        for i in range(num_batches):
            # Decides on the total spread and hop length.
            spread_size = 1 + gap_crossing
            dispersal_tmp = dispersal() if callable(dispersal) else dispersal
            num_spreads = int(0.5 + dispersal_tmp / spread_size)
            if callable(dispersal):
                repopulator = analysis_class(
                    hab,
                    ter,
                    num_spreads=num_spreads,
                    spread_size=spread_size,
                    device=device,
                ).to(device)
            # Creates the seeds.
            seed_probability = seed_density / ((1 + 2 * dispersal_tmp) ** 2)
            seeds = torch.rand((batch_size, w, h), device=device) < seed_probability
            ## Sample the hop and spreads if necessary.
            # And passes them through the repopulation.
            pop = repopulator(seeds)
            if repopulation_only_in_habitat:
                pop *= hab
            # We need to take the mean over each batch.  This will tell us what is the
            # average repopulation.
            tot_pop += torch.mean(pop, 0)
            # This is the sum across all batches.  So, the gradient will be for the total
            # of the batch. This is why the gradient will need to be divided by the number
            # of simulations.
            if produce_gradient:
                q = torch.sum(pop / num_batches)
                q.backward()
                tot_grad += repopulator.get_grad()
        # Normalizes by number of batches.
        avg_pop, avg_grad = tot_pop / num_batches, tot_grad / num_simulations
        return avg_pop.to("cpu"), avg_grad.to("cpu")

    if not produce_gradient:
        # We remove all memory/time requirements due to gradient computation.
        f = torch.no_grad()(f)
    return f
