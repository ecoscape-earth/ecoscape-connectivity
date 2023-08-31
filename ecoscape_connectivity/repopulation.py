# Standard imports
import numpy as np
import torch
from torch import nn
from contextlib import nullcontext

# Our imports
from scgt import GeoTiff, Tile
from .util import dict_translate


class StochasticRepopulateFast(nn.Module):
    """
    Important: THIS is the function to use in the repopulation experiments.
    This module models the repopulation of the habitat from a chosen percentage
    of the seed places.  The terrain and habitat are parameters, and the input is a
    similarly sized 0-1 (float) tensor of seed points."""

    def __init__(self, habitat, terrain, num_spreads=100, spread_size=1, min_transmission=0.9,
                 randomize_source=True, randomize_dest=False):
        """
        :param habitat: torch tensor (2-dim) representing the habitat.
        :param terrain: torch tensor (2-dim) representing the terrain.
        :param num_spreads: number of bird spreads to use
        :param spread_size: by how much (in pixels) birds spread
        :param min_transmission: min value used in randomizations.
        :param randomize_source: whether to randomize the source of the spread.
        :param randomize_dest: whether to randomize the destination of the spread.
        """
        super().__init__()
        self.habitat = habitat
        self.goodness = torch.nn.Parameter(torch.max(habitat, terrain), requires_grad=True)
        self.h, self.w = habitat.shape
        self.num_spreads = num_spreads
        self.spread_size = spread_size
        # Defines spread operator.
        self.min_transmission = min_transmission
        self.randomize_source = randomize_source
        self.randomize_dest = randomize_dest
        self.kernel_size = 1 + 2 * spread_size
        self.spreader = torch.nn.MaxPool2d(self.kernel_size, stride=1, padding=spread_size)


    def forward(self, seed):
        """
        seed: a 0-1 (float) tensor of seed points.
        """
        # First, we multiply the seed by the habitat, to confine the seeds to
        # where birds can live.
        x = seed * self.habitat
        if x.ndim < 3:
            # We put it into shape (1, w, h) because the pooling operator expects this.
            x = torch.unsqueeze(x, dim=0)
        # Now we must propagate n times.
        for _ in range(self.num_spreads):
            # First, we randomly suppress some bird origin locations.
            xx = x
            if self.randomize_source:
                x = x * (self.min_transmission + (1. - self.min_transmission) * torch.rand_like(x))
            # Then, we propagate.
            x = self.spreader(x) * self.goodness
            # We randomize the destinations too.
            if self.randomize_dest:
                x *= (self.min_transmission + (1. - self.min_transmission) * torch.rand_like(x))
            # And finally we combine the results.
            x = torch.max(x, xx)
        x *= self.habitat
        if seed.ndim < 3:
            x = torch.squeeze(x, dim=0)
        return x

    def get_grad(self):
        return self.goodness.grad * self.goodness


###############################################################################
# Analyze geotiff and tile.

def analyze_tile_torch(
        device=None,
        analysis_class=StochasticRepopulateFast,
        seed_density=4.0, produce_gradient=False,
        batch_size=1, total_spreads=100, num_simulations=100,
        hop_length=1):
    """This is the function that performs the analysis on a single tile.
    The input and output to this function are in cpu, but the computation occurs in
    the specified device.
    hop_length: length in pixels of a bird hop.
    total_spreads: total number of spreads used.
    seed_density: Consider a square of edge 2 * hop_length * total_spreads.
        In that square, there will be seed_density seeds on average.
    str device: the device to be used, either cpu or cuda.
    analysis_class: class to be used for the analysis.  The signature is somewhat constrained
        (see code) but keeping it as a parameter enables at least a modicum of flexibility.
    produce_gradient: boolean, whether to produce a gradient as result or not.
    int batch_size: batch size for GPU calculations.
    int num_simulations: how many simulations to run.  Must be multiple of batch_size.

    """

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Computes the seed probability
    seed_probability =  seed_density / ((2 * hop_length * total_spreads) ** 2)

    def f(habitat, terrain):
        _, w, h = habitat.shape
        # We may need to do multiple spreads if the batch size is smaller than the total spreads.
        assert num_simulations % batch_size == 0, "Simulations not multiples of batch"
        num_batches = num_simulations // batch_size
        tot_grad = torch.zeros((1, w, h), dtype=torch.float, device=device)
        tot_pop = torch.zeros((1, w, h), dtype=torch.float, device=device)
        hab = torch.tensor(habitat.astype(np.float), requires_grad=False, dtype=torch.float, device = device).view(w, h)
        ter = torch.tensor(terrain.astype(np.float), requires_grad=False, dtype=torch.float, device = device).view(w, h)
        repopulator = analysis_class(hab, ter, num_spreads=total_spreads, spread_size=hop_length).to(device)
        for i in range(num_batches):
            # Creates the seeds.
            seeds = torch.rand((batch_size, w, h), device=device) < seed_probability
            # And passes them through the repopulation.
            pop = repopulator(seeds)
            # We need to take the mean over each batch.  This will tell us what is the
            # average repopulation.
            tot_pop += torch.mean(pop, 0)
            # This is the sum across all batches.  So, the gradient will be for the total
            # of the batch. This is why the gradient will need to be divided by the number
            # of simulations.
            if produce_gradient:
                q = torch.sum(pop)
                q.backward()
                tot_grad += repopulator.get_grad()
        # Normalizes by number of batches.
        avg_pop, avg_grad = tot_pop / num_batches, tot_grad / num_simulations
        return avg_pop.to("cpu"), avg_grad.to("cpu")
    if not produce_gradient:
        # We remove all memory/time requirements due to gradient computation.
        f = torch.no_grad()(f)
    return f



def analyze_geotiffs(habitat_fn, terrain_fn,
                     terr_to_transmission,
                     analysis_fn=None,
                     single_tile=False,
                     hab_tile=None, ter_tile=None,
                     block_size=128,
                     border_size=64,
                     generate_gradient=True,
                     interesting_tiles=None,
                     display_tiles=False,
                     disp_fn=None,
                     minimum_habitat=1e-4,
                     output_repop_fn=None,
                     output_grad_fn=None,
                     report_progress=False):
    '''
    Reads a geotiff (or better, a pair of habitat and terrain geotiffs),
    iterating over the tiles, analyzing it with a specified analysis function,
    and then writing the results back.

    str habitat_fn: filename of habitat geotiff
    str terrain_fn: filename of terrain geotiff
    dict terr_to_transmission: terrain to transmission mapping dictionary
    analysis_fn: function used for analysis.
    disp_fn: function used to display a tile for debugging purposes.
    Tile hab_tile, Tile ter_tile: If provided, runs on this particular tile, and
        disregards the other parameters.
    bool single_tile: if true, reads the entire geotiff as a single tile (no iteration).
    int block_size: dimensions of tile
    int border_size: pixel border on each side
    list interesting_tiles: for running only a specified tile list
    display_tiles: True, to display tiles, or list of tiles interesting enough to display.
    minimum_habitat: minimum average of habitat to skip the tile.
    string output_grad: file path for outputting the grad tif file.
    string output_repop: file path for outputting the repop tif file.
        For these last two, if None, then no file is generated.
    '''
    if display_tiles is False:
        display_tiles = []

    do_gradient = generate_gradient and output_grad_fn is not None
    habitat_geotiff = GeoTiff.from_file(habitat_fn)
    terrain_geotiff = GeoTiff.from_file(terrain_fn)
    def do_analysis(repop_file, grad_file):
        do_output = (repop_file is not None)
        if hab_tile is not None and ter_tile is not None:
            hab_reader = [hab_tile]
            ter_reader = [ter_tile]
        else:
            # Reads the files.
            # Iterates through the tiles.
            if single_tile:
                # We read the geotiffs as a single tile.
                hab_reader = [habitat_geotiff.get_all_as_tile()]
                ter_reader = [terrain_geotiff.get_all_as_tile()]
            else:
                # We create readers to iterate over the tiles.
                hab_reader = habitat_geotiff.get_reader(b=border_size, w=block_size, h=block_size)
                ter_reader = terrain_geotiff.get_reader(b=border_size, w=block_size, h=block_size)
        # We process each tile.
        for i, (hab_tile_iter, ter_tile_iter) in enumerate(zip(hab_reader, ter_reader)):
            raw_habitat = hab_tile_iter.m # Habitat tile
            raw_terrain = ter_tile_iter.m # Terrain tile
            # Deals with missing values in the habitat.
            habitat = np.maximum(raw_habitat, 0)
            if display_tiles is True or i in display_tiles:
                disp_fn(habitat, title="Raw habitat")
                disp_fn(raw_terrain, title="Raw terrain")
            # We skip the tile if too little habitat, or if we specify only a list of tiles
            # to be analyzed.
            skip_tile = ((interesting_tiles is not None and i not in interesting_tiles)
                         or
                         (np.mean(habitat) < minimum_habitat))
            if skip_tile:
                if do_output:
                    # These lines are here just to fix a bug into the production of the output geotiff,
                    # which does not set all to zero before the output is done.
                    repop_tile = Tile(ter_tile_iter.w, ter_tile_iter.h, ter_tile_iter.b, ter_tile_iter.c,
                                     ter_tile_iter.x, ter_tile_iter.y, np.zeros_like(habitat))
                    repop_file.set_tile(repop_tile)
                    if do_gradient:
                        grad_tile = Tile(ter_tile_iter.w, ter_tile_iter.h, ter_tile_iter.b, ter_tile_iter.c,
                                         ter_tile_iter.x, ter_tile_iter.y, np.zeros_like(habitat))
                        grad_file.set_tile(grad_tile)
                continue
            # We process the tile.
            # First, we translate the terrain to correct resistance
            terrain = dict_translate(raw_terrain, terr_to_transmission, default_val=0)
            if display_tiles is True or i in display_tiles:
                print("Terrain types", np.unique(raw_terrain))
                disp_fn(terrain, title="Terrain transmission")
            # Then, we processes the terrain and habitat tiles.
            pop, grad = analysis_fn(habitat, terrain)
            # Normalizes the tiles, to fit into the geotiff format.
            # The population is simply normalized with a max of 255. After all it is in [0, 1].
            # We need to use type float because clam is not implemented for all types.
            # print(isinstance(pop))
            if isinstance(pop, np.ndarray):
                # print('prev sum:', np.sum(pop), np.sum(pop)*255)
                # np.set_printoptions(threshold=2000)
                # print(pop[400:450,400:450])
                norm_pop = np.expand_dims(np.clip(pop * 255, 0, 255).astype(np.uint8), axis=0)
                norm_grad = np.expand_dims(np.clip(np.log10(1. + grad) * 20., 0, 255).astype(np.uint8), axis=0)
                # print(norm_pop.shape)
                # print('new sum:', np.sum(norm_pop.astype(int)))
                # print(norm_pop.shape, norm_grad.shape)
            else:
                norm_pop = torch.clamp(pop.type(torch.float) * 255, 0, 255).type(torch.uint8)
                norm_grad = torch.clamp(torch.log10(1. + grad.type(torch.float)) * 20., 0, 255).type(torch.uint8)

            # Displays the output if so asked.
            if display_tiles is True or i in display_tiles:
                disp_fn(norm_pop, title="Repopulation")
                disp_fn(norm_grad, title="Gradient")

            # Prepares the tiles for writing.
            if do_output:
                # Writes the tiles.
                repop_tile = Tile(ter_tile_iter.w, ter_tile_iter.h, ter_tile_iter.b, ter_tile_iter.c, ter_tile_iter.x, ter_tile_iter.y, norm_pop)
                repop_file.set_tile(repop_tile)
                if do_gradient:
                    grad_tile = Tile(ter_tile_iter.w, ter_tile_iter.h, ter_tile_iter.b, ter_tile_iter.c, ter_tile_iter.x, ter_tile_iter.y, norm_grad)
                    grad_file.set_tile(grad_tile)
            if report_progress:
                print(i, end=' ', flush=True)
        if report_progress:
           print()

    if output_repop_fn is not None:
        with habitat_geotiff.clone_shape(output_repop_fn) as repop_output:
            with habitat_geotiff.clone_shape(output_grad_fn) if do_gradient else nullcontext() as grad_output:
                do_analysis(repop_output, grad_output)
    else:
        do_analysis(None, None)


def compute_connectivity(
        habitat_fn=None,
        terrain_fn=None,
        connectivity_fn=None,
        flow_fn=None,
        permeability_dict=None,
        gap_crossing=2,
        num_gaps=10,
        num_simulations=400,
        seed_density=4,
        single_tile=False,
        tile_size=1000,
        tile_border=256,
        minimum_habitat=1e-4
    ):
    """
    Function that computes the connectivity. This is the main function in the module.
    The outputs are encoded as follows:
    - For connectivity, the values from [0, 1] are rescaled to the range 0..255 and encoded
      as integers.
    - For flow, the values from [0, infty) are encoded in log-scale via 20 * log_10(1 + f)
      (so that the flow is expressed in dB, like sound intensity), and clipped to the 0..255 range.
    :param habitat_fn: name of habitat geotiff. This file must contain 0 = non habitat,
        and 1 = habitat.
    :param terrain_fn: name of terrain file.  This file contains terrain categories that are
        translated via permeability_dict.
    :param connectivity_fn: output file name for connectivity.
    :param flow_fn: output file name for flow.  If None, the flow is not computed, and the
        computation is faster.
    :param permeability_dict: Permeability dictionary.  Gives the permeability of each
        terrain type, translating from the terrain codes, to the permeability in [0, 1].
        If a terrain type is not found in the dictionary, it is assumed it has permeability 0.
    :param gap_crossing: size of gap crossing in pixels.
    :param num_gaps: number of gaps that can be crossed during dispersal.
    :param num_simulations: Number of simulations that are done.
    :param seed_density: density of seeds.  There are this many seeds for every square with edge of
        dispersal distance.
    :param single_tile: if True, instead of iterating over small tiles, tries to read the input as a
        single large tile.  This is faster, but might not fit into memory.
    :param tile_size: size of (square) tile in pixels.
    :param tile_border: size of tile border in pixels.
    :param minimum_habitat: if a tile has a fraction of habitat smaller than this, it is skipped.
        This saves time in countries where the habitat is only on a small portion.
    """
    assert habitat_fn is not None and terrain_fn is not None
    assert connectivity_fn is not None
    assert permeability_dict is not None
    # Builds the analysis function.
    analysis_fn = analyze_tile_torch(
        seed_density=seed_density,
        produce_gradient=flow_fn is not None,
        total_spreads=num_gaps,
        num_simulations=num_simulations,
        hop_length=gap_crossing)
    # Applies it.
    analyze_geotiffs(
        habitat_fn, terrain_fn,
        permeability_dict,
        analysis_fn=analysis_fn,
        single_tile=single_tile,
        block_size=tile_size,
        border_size=tile_border,
        generate_gradient=flow_fn is not None,
        minimum_habitat=minimum_habitat,
        output_repop_fn=connectivity_fn,
        output_grad_fn=flow_fn,
    )
