# Standard imports
import argparse
import numpy as np
import torch
from torch import nn
from contextlib import nullcontext

# Our imports
from scgt import GeoTiff, Tile
from .util import dict_translate, SingleIterator

from osgeo import gdal
gdal.UseExceptions()

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
        # spread_size and num_spreads have to be integers, not callables here. 
        assert type(spread_size) == int, "spread_size must be an int"
        assert type(num_spreads) == int, "num_spreads must be an int"
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
        for i in range(self.num_spreads):
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
        seed_density=4.0,
        produce_gradient=False,
        batch_size=1,
        dispersal=20,
        num_simulations=100,
        gap_crossing=0):
    """This is the function that performs the analysis on a single tile.
    The input and output to this function are in cpu, but the computation occurs in
    the specified device.
    gap_crossing: maximum number of pixels a bird can jump. 0 means only contiguous pixels.
    dispersal: dispersal distance in pixels.
        As above, if this is an integer, we do this constanst number of spreads
        for all batches. Otherwise, If this is of the form of a function
        (probability distribution), we run the function (and sample the distribution)
        to get the dispersal distance.
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
    assert type(gap_crossing) == int, "gap_crossing must be an int"

    # Computes the seed probability
    # seed_probability =  seed_density / ((2 * hop_length * total_spreads) ** 2)

    def f(habitat, terrain):
        _, w, h = habitat.shape
        # We may need to do multiple spreads if the batch size is smaller than the total spreads.
        assert num_simulations % batch_size == 0, "Simulations not multiples of batch"
        num_batches = num_simulations // batch_size
        tot_grad = torch.zeros((1, w, h), dtype=torch.float, device=device)
        tot_pop = torch.zeros((1, w, h), dtype=torch.float, device=device)
        hab = torch.tensor(habitat.astype(float), requires_grad=False, dtype=torch.float, device = device).view(w, h)
        ter = torch.tensor(terrain.astype(float), requires_grad=False, dtype=torch.float, device = device).view(w, h)
        # If the num_spreads and spread_size are constant, then we can use a fixed repopulator, which is more efficient.
        if not callable(dispersal):
            num_spreads = int(0.5 + dispersal / (gap_crossing + 1))
            repopulator = analysis_class(hab, ter, num_spreads=num_spreads, spread_size=gap_crossing + 1).to(device)
        for i in range(num_batches):
            # Decides on the total spread and hop length.
            spread_size = 1 + gap_crossing
            dispersal_tmp = dispersal() if callable(dispersal) else dispersal
            num_spreads = int(0.5 + dispersal_tmp / spread_size)
            if callable(dispersal):
                repopulator = analysis_class(hab, ter, num_spreads=num_spreads, spread_size=spread_size).to(device)
            # Creates the seeds.
            seed_probability =  seed_density / ((1 + 2 * dispersal_tmp) ** 2)
            seeds = torch.rand((batch_size, w, h), device=device) < seed_probability
            ## Sample the hop and spreads if necessary.
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


def analyze_geotiffs(habitat_fn=None, 
                     terrain_fn=None,
                     permeability_dictionary=None,
                     permeability_fn=None,
                     permeability_scaling=1.0,
                     analysis_fn=None,
                     single_tile=False,
                     tile_size=1024,
                     border_size=64,
                     generate_gradient=True,
                     interesting_tiles=None,
                     display_tiles=False,
                     disp_fn=None,
                     minimum_habitat=1e-4,
                     output_repop_fn=None,
                     output_grad_fn=None,
                     report_progress=False,
                     in_memory=False,
                     float_output=False):
    '''
    Reads a geotiff (or better, a pair of habitat and terrain geotiffs),
    iterating over the tiles, analyzing it with a specified analysis function,
    and then writing the results back.

    str/GeoTiff habitat_fn: filename of habitat geotiff, or GeoTiff object from habitat geotiff
    str/GeoTiff terrain_fn: filename of terrain geotiff, or GeoTiff object from terrain geotiff
    dict permeability_dictionary: terrain to permeability mapping dictionary. 
        Terrains not listed are assigned a permeability of 0. 
    analysis_fn: function used for analysis.
    disp_fn: function used to display a tile for debugging purposes.
    bool single_tile: if true, reads the entire geotiff as a single tile (no iteration).
    int tile_size: dimensions of tile
    int border_size: pixel border on each side
    display_tiles: True, to display tiles, or list of tiles interesting enough to display.
    minimum_habitat: minimum average of habitat to skip the tile.
    string output_grad: file path for outputting the grad tif file.
    string output_repop: file path for outputting the repop tif file.
        For this and output_grad, if None, then no file is generated.
    bool in_memory: whether the connectivity and flow should be saved in memory only. If so, then
        the files are not saved to disk, so the open files for connectivity and flow are returned.
    '''
    if display_tiles is False:
        display_tiles = []

    do_gradient = generate_gradient and (output_grad_fn is not None if not in_memory else True)
    habitat_geotiff = None
    if habitat_fn is not None:
        habitat_geotiff = GeoTiff.from_file(habitat_fn) if type(habitat_fn) == str else habitat_fn
    # We specify permeability either via a terrain and a dictionary, or via a permeability file 
    # with scaling.  In either case, we have a permeability geotiff, which we then have to 
    # either scale or process via a dictionary.
    if permeability_fn is not None:
        # We are directly given a permeability file, which we scale via a constant. 
        scale_via_dictionary = False
        permeability_geotiff = GeoTiff.from_file(permeability_fn) if type(permeability_fn) == str else permeability_fn
    else:
        # We use terrain and dictionary. 
        scale_via_dictionary = True
        permeability_geotiff = GeoTiff.from_file(terrain_fn) if type(terrain_fn) == str else terrain_fn
        
    def do_analysis(repop_file, grad_file):
        do_output = (repop_file is not None)
        # Reads the files.
        # Iterates through the tiles.
        if single_tile:
            # We read the geotiffs as a single tile.
            joint_reader = [
                (habitat_geotiff.get_all_as_tile() if habitat_geotiff is not None else None,
                 permeability_geotiff.get_all_as_tile())
            ]
        else:
            # We create readers to iterate over the tiles.
            per_reader = permeability_geotiff.get_reader(b=border_size, w=tile_size, h=tile_size)
            if habitat_geotiff is None:
                # We create a reader where the habitat is always None.
                joint_reader = SingleIterator(per_reader)
            else:
                hab_reader = habitat_geotiff.get_reader(b=border_size, w=tile_size, h=tile_size)
                joint_reader = zip(hab_reader, per_reader)        
        # We process each tile.
        for i, (hab_tile_iter, per_tile_iter) in enumerate(joint_reader):
            raw_habitat = hab_tile_iter.m if hab_tile_iter is not None else None # Habitat tile
            raw_permeability = per_tile_iter.m # Permeability tile            
            if display_tiles is True or i in display_tiles:
                disp_fn(habitat, title="Raw habitat")
                disp_fn(raw_permeability, title="Raw terrain")
            # We skip a tile if: 
            # - the habitat is not None, and too low, or, 
            # - the habitat is None, and the permeability is too low. 
            skip_tile = False
            if raw_habitat is not None: 
                habitat = np.maximum(raw_habitat, 0)
                skip_tile = np.mean(habitat) < minimum_habitat
            if not skip_tile:            
                # Scales the permeability.
                if scale_via_dictionary:
                    permeability = dict_translate(raw_permeability, permeability_dictionary, default_val=0)
                else:
                    # Some software uses -infty for 0. 
                    raw_permeability = np.maximum(raw_permeability, 0)
                    permeability = raw_permeability * permeability_scaling
                    # Clip it between 0 and 1.
                    permeability = np.clip(permeability, 0, 1)
                # Checks whether we have to skip due to low permeability. 
                if raw_habitat is None:
                    skip_tile = np.mean(permeability) < minimum_habitat         
            if skip_tile:
                if do_output:
                    # These lines are here just to fix a bug into the production of the output geotiff,
                    # which does not set all to zero before the output is done.
                    repop_tile = Tile(per_tile_iter.w, per_tile_iter.h, per_tile_iter.b, per_tile_iter.c,
                                     per_tile_iter.x, per_tile_iter.y, np.zeros_like(raw_permeability))
                    repop_file.set_tile(repop_tile, geotiff_includes_border=False)
                    if do_gradient:
                        grad_tile = Tile(per_tile_iter.w, per_tile_iter.h, per_tile_iter.b, per_tile_iter.c,
                                         per_tile_iter.x, per_tile_iter.y, np.zeros_like(raw_permeability))
                        grad_file.set_tile(grad_tile, geotiff_includes_border=False)
                continue
            # We process the tile.
            pop, grad = analysis_fn(permeability if raw_habitat is None else habitat, permeability)
            # Normalizes the tiles, to fit into the geotiff format.
            # The population is simply normalized with a max of 255. After all it is in [0, 1].
            # We need to use type float because clam is not implemented for all types.
            if float_output:
                if isinstance(pop, np.ndarray):
                    norm_pop = np.expand_dims(pop.numpy().astype(float), axis=0)
                    norm_grad = np.expand_dims(np.log10(1. + grad) * 20., axis=0)
                else:
                    norm_pop = pop.detach().numpy().astype(float)
                    norm_grad = np.log10(1. + grad.detach().numpy().astype(float)) * 20.
            else:                        
                if isinstance(pop, np.ndarray):
                    norm_pop = np.expand_dims(np.clip(pop * 255, 0, 255).astype(np.uint8), axis=0)
                    norm_grad = np.expand_dims(np.clip(np.log10(1. + grad) * 20., 0, 255).astype(np.uint8), axis=0)
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
                repop_tile = Tile(per_tile_iter.w, per_tile_iter.h, per_tile_iter.b, per_tile_iter.c, per_tile_iter.x, per_tile_iter.y, norm_pop)
                repop_file.set_tile(repop_tile, geotiff_includes_border=False)
                if do_gradient:
                    grad_tile = Tile(per_tile_iter.w, per_tile_iter.h, per_tile_iter.b, per_tile_iter.c, per_tile_iter.x, per_tile_iter.y, norm_grad)
                    grad_file.set_tile(grad_tile, geotiff_includes_border=False)
            if report_progress:
                print(i, end=' ', flush=True)
        if report_progress:
           print()

    if in_memory:
        # Produce the outputs in memory.
        repop_output = GeoTiff.create_memory_file(permeability_geotiff.profile)
        grad_output = GeoTiff.create_memory_file(permeability_geotiff.profile) if do_gradient else nullcontext()
        do_analysis(repop_output, grad_output)
        # These are open memory files. The caller should close them with scgt's GeoTiff.close_memory_file()
        # once they are not needed anymore.
        return repop_output, grad_output
    
    elif output_repop_fn is not None:
        # Produce the outputs on disk.
        profile = permeability_geotiff.profile
        if float_output:
            profile['dtype'] = 'float32'
        # Computes the output bounds, keeping into account the border size.
        profile['width'] -= 2 * border_size
        profile['height'] -= 2 * border_size 
        with GeoTiff.copy_to_new_file(output_repop_fn, profile) as repop_output:
            with GeoTiff.copy_to_new_file(output_grad_fn, profile) if do_gradient else nullcontext() as grad_output:
                do_analysis(repop_output, grad_output)
    else:
        # Just run the analysis without outputs.
        do_analysis(None, None)
    
    return None, None


def compute_connectivity(
        habitat_fn=None,
        permeability_fn=None,
        permeability_scaling=1.0,
        terrain_fn=None,
        permeability_dict=None,
        connectivity_fn=None,
        flow_fn=None,
        gap_crossing=0,
        dispersal=None,
        num_gaps=None,
        num_simulations=400,
        seed_density=4,
        single_tile=False,
        tile_size=1000,
        border_size=200,
        minimum_habitat=1e-4,
        random_seed=None,
        in_memory=False,
        generate_flow_memory=False,
        float_output=True,
    ):
    """
    Function that computes the connectivity. This is the main function in the module.
    The outputs are encoded as follows:

    If floating point output is selected: 
    - The connectivity output is in [0, 1].
    - The flow output is in [0, infty), obtained via 20 * log_10(1 + f).

    If integer output is selected: 
    - For connectivity, the values from [0, 1] are rescaled to the range 0..255 and encoded
      as integers.
    - For flow, the values from [0, infty) are encoded in log-scale via 20 * log_10(1 + f)
      (so that the flow is expressed in dB, like sound intensity), and clipped to the 0..255 range.

    Integer output saves space, but floating point output is more accurate and intuitive to use.
    
    :param habitat_fn: name of habitat geotiff, or GeoTiff (from the scgt packaage) object from habitat geotiff. This file must contain
        0 = non habitat, and 1 = habitat.
        If this file is missing, then it is assumed that everywhere is suitable habitat, and that
        only the permeability determines possible movement. This is useful for modeling mammals. 
    :param permeability_fn: File name for permeability, or GeoTiff object (from scgt package) for the permeability. 
        If this is given, the permeability is read from this file, and scaled according to 
        permeability_scaling.  If this is not given, then the permeability
        is derived from the terrain_fn file, and the dictionary. 
    :param permeability_scaling: scaling factor for the permeability.  This is used only if the permeability
        is read from a file.
    :param terrain_fn: name of terrain geotiff, or GeoTiff object from terrain geotiff.  This file contains
        terrain categories that are translated via permeability_dict.
    :param permeability_dict: Permeability dictionary.  Gives the permeability of each
        terrain type, translating from the terrain codes, to the permeability in [0, 1].
        If a terrain type is not found in the dictionary, it is assumed it has permeability 0.
    :param connectivity_fn: output file name for connectivity.
    :param flow_fn: output file name for flow.  If None, the flow is not computed, and the
        computation is faster.
    :param gap_crossing: size of gap crossing in pixels. 0 means animals move via contiguous pixels.
    :param dispersal: dispersal distance in pixels.
    :param num_gaps: number of gaps to cross. Deprecated.  If dispersal is None, then this is used to 
        compute the dispersal distance.  At least one of dispersal, num_gaps must be provided.
    :param num_simulations: Number of simulations that are done.
    :param seed_density: density of seeds.  There are this many seeds for every square with edge of
        dispersal distance.
    :param single_tile: if True, instead of iterating over small tiles, tries to read the input as a
        single large tile.  This is faster, but might not fit into memory.
    :param tile_size: size of (square) tile in pixels.  This is the size that is processsed 
        in one go.  Choose the tile as large as possible, so that it fits into the GPU memory.
    :param border_size: size of analysis border in pixels. This has to be at least 
        equal to the dispersal distance. 
    :param minimum_habitat: if a tile has a fraction of habitat smaller than this, it is skipped.
        This saves time in countries where the habitat is only on a small portion.
    :param random_seed: random seed, if desired. 
    :param in_memory: whether the connectivity and flow should be saved in memory only.
        If so, then the files are not saved to disk. Because such files would be deleted on close,
        the open memory files will be returned as (repop_file, grad_file). Note that the parameters
        connectivity_fn and flow_fn are ignored if this is set to True, and at least connectivity
        will be returned. Flow is also generated only if generate_flow_memory is True.
    :param generate_flow_memory: whether the flow should be generated in memory. Only used if
        in_memory is True.
    :param float_output: use floating point output, generating a floating point tiff. 
    :return: (None, None) if in_memory is False, (repop_file, grad_file) if in_memory is True.
        If in_memory is True, the caller should close the files with scgt's GeoTiff.close_memory_file()
        once they are not needed anymore.
    """
    assert habitat_fn is None or type(habitat_fn) == str or type(habitat_fn) == GeoTiff
    assert terrain_fn is not None or permeability_fn is not None
    assert terrain_fn is None or permeability_dict is not None
    assert terrain_fn is None or type(terrain_fn) == str or type(terrain_fn) == GeoTiff
    assert permeability_fn is None or type(permeability_fn) == str or type(permeability_fn) == GeoTiff
    
    if random_seed:
        torch.manual_seed(random_seed)
    if dispersal is None:
        assert num_gaps is not None, "One of dispersal and gap crossing should be specified."
        dispersal = (1 + gap_crossing) * num_gaps
    # Builds the analysis function.
    analysis_function = analyze_tile_torch(
        seed_density=seed_density,
        produce_gradient=flow_fn is not None,
        dispersal=dispersal,
        num_simulations=num_simulations,
        gap_crossing=gap_crossing)
    
    # Applies it.
    return analyze_geotiffs(
        habitat_fn=habitat_fn, 
        terrain_fn=terrain_fn,
        permeability_dictionary=permeability_dict,
        permeability_fn=permeability_fn, 
        permeability_scaling=permeability_scaling,
        analysis_fn=analysis_function,
        single_tile=single_tile,
        tile_size=tile_size,
        border_size=border_size,
        generate_gradient=(flow_fn is not None if not in_memory else generate_flow_memory),
        minimum_habitat=minimum_habitat,
        output_repop_fn=connectivity_fn,
        output_grad_fn=flow_fn,
        float_output=float_output,
        in_memory=in_memory
    )
