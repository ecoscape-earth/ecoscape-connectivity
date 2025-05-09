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


def compute_connectivity(
        habitat_fn=None,
        permeability_fn=None,
        permeability_scaling=1.0,
        landcover_fn=None,
        permeability_dict=None,
        connectivity_fn=None,
        flow_fn=None,
        gap_crossing=0,
        dispersal=None,
        num_gaps=None,
        num_simulations=400,
        seed_density=4,
        tile_size=1000,
        border_size=100,
        padding_size=0,
        permeability_padding=0,
        habitat_padding=0,
        minimum_habitat=1e-4,
        float_output=True,
        random_seed=None,
        in_memory=False,
        batch_size=1,
        generate_flow_memory=False,
        repopulation_only_in_habitat=True,
        device=None
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
        is derived from the landcover_fn file, and the dictionary. 
    :param permeability_scaling: scaling factor for the permeability.  The permeability values
        are used per-pixel.  If you have them instead of per-pixel, per n-pixels (e.g., each pixel is 100m 
        but you computed the permeability for 1km), then you would specify here a scaling factor of 0.1 = 1 km / 100 m.  The permeability values p are rescaled to p ** permeability_scaling. 
    :param landcover_fn: name of terrain geotiff, or GeoTiff object from terrain geotiff.  This file contains
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
    :param tile_size: size of (square) tile in pixels.  This is the size that is processsed 
        in one go.  Choose the tile as large as possible, so that it fits into the GPU memory.
    :param border_size: size of analysis border used on each tile in pixels. This has to be at least 
        equal to the dispersal distance. 
    :param padding_size: amount of padding around each tile.  If you specify 0 (no padding), then 
        the output raster will have the same size as the input, except for a border of size border_size
        all around it: so if the input is of size w, h and the border is of size b, then the output
        will be of size w - 2 * b, h - 2 * b.  If you specify a padding of p, then the output will
        have a border of size b - p, and the output will be of size w - 2 * (b - p), h - 2 * (b - p).
        The padding cannot be greater than the border size. 
    :param permeability_padding: value to be used to pad the permeability or terrain raster. 
    :param habitat_padding: value to be used to pad the habitat raster.
    :param batch_size: batch size for GPU calculations.  Use the largest batch size that makes the 
        computation fit into the GPU memory.  
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
    :param repopulation_only_in_habitat: if True, then the repopulation only occurs in the habitat. 
        That's the default.  If False, the repopulation can be non-zero all over the output raster.
    :param device: the device to be used for the computation. If None, then the device is chosen
        automatically.  Valid values include: 'cpu', 'cuda', 'mps'.
    :return: (None, None) if in_memory is False, (repop_file, grad_file) if in_memory is True.
        If in_memory is True, the caller should close the files with scgt's GeoTiff.close_memory_file()
        once they are not needed anymore.
    """
    assert habitat_fn is None or type(habitat_fn) == str or type(habitat_fn) == GeoTiff
    assert landcover_fn is not None or permeability_fn is not None
    assert landcover_fn is None or permeability_dict is not None
    assert landcover_fn is None or type(landcover_fn) == str or type(landcover_fn) == GeoTiff
    assert permeability_fn is None or type(permeability_fn) == str or type(permeability_fn) == GeoTiff
    
    if random_seed:
        torch.manual_seed(random_seed)
    if dispersal is None:
        assert num_gaps is not None, "One of dispersal and gap crossing should be specified."
        dispersal = (1 + gap_crossing) * num_gaps
    # Builds the analysis function.
    analysis_function = analyze_tile_torch(
        device=device,
        seed_density=seed_density,
        produce_gradient=flow_fn is not None,
        dispersal=dispersal,
        batch_size=batch_size,
        num_simulations=num_simulations,
        gap_crossing=gap_crossing,
        repopulation_only_in_habitat=repopulation_only_in_habitat)
    
    # Applies it.
    return analyze_geotiffs(
        habitat_fn=habitat_fn, 
        landcover_fn=landcover_fn,
        permeability_dictionary=permeability_dict,
        permeability_fn=permeability_fn, 
        permeability_scaling=permeability_scaling,
        analysis_fn=analysis_function,
        tile_size=tile_size,
        border_size=border_size,
        padding_size=padding_size,
        permeability_padding=permeability_padding,
        habitat_padding=habitat_padding,
        generate_gradient=(flow_fn is not None if not in_memory else generate_flow_memory),
        minimum_habitat=minimum_habitat,
        output_repop_fn=connectivity_fn,
        output_grad_fn=flow_fn,
        float_output=float_output,
        in_memory=in_memory
    )


class RandomPropagate(nn.Module):
    """
    Important: THIS is the function to use in the repopulation experiments.
    This module models the repopulation of the habitat from a chosen percentage
    of the seed places.  The terrain and habitat are parameters, and the input is a
    similarly sized 0-1 (float) tensor of seed points."""

    def __init__(self, habitat, terrain, num_spreads=100, spread_size=1, device=None, 
                 update_threshold=0.05, coin_toss_probability=0.5):
        """
        :param habitat: torch tensor (2-dim) representing the habitat.
        :param terrain: torch tensor (2-dim) representing the terrain.
        :param num_spreads: number of bird spreads to use
        :param spread_size: by how much (in pixels) birds spread in each spread. 
        :param device: device to use for computation (cpu, cuda, mps, etc). 
        """
        super().__init__()
        # spread_size and num_spreads have to be integers, not callables here. 
        self.device = device or torch.device("cpu")
        assert type(spread_size) == int, "spread_size must be an int"
        assert type(num_spreads) == int, "num_spreads must be an int"
        self.habitat = habitat
        self.goodness = torch.nn.Parameter(terrain, requires_grad=True)
        self.h, self.w = habitat.shape
        self.num_spreads = num_spreads
        self.spread_size = spread_size
        # Defines spread operator.
        self.min_transmission =  1e-4
        self.update_threshold = update_threshold
        self.coin_toss_probability = coin_toss_probability
        self.kernel_size = 1 + 2 * spread_size
        self.spreader = torch.nn.MaxPool2d(self.kernel_size, stride=1, padding=spread_size)


    def forward(self, seed):
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
            x = x * (1 - self.min_transmission + self.min_transmission * torch.rand_like(x))
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

    def get_grad(self):
        return self.goodness.grad * self.goodness


###############################################################################
# Analyze geotiff and tile.

def analyze_tile_torch(
        device=None,
        analysis_class=RandomPropagate,
        seed_density=4.0,
        produce_gradient=False,
        batch_size=1,
        dispersal=20,
        num_simulations=100,
        gap_crossing=0,
        repopulation_only_in_habitat=True):
    """This is the function that performs the analysis on a single tile.
    The input and output to this function are in cpu, but the computation occurs in
    the specified device.
    :param device: the device to be used, e.g., cpu, cuda, mps.
    :param analysis_class: class to be used for the analysis.  We recommend RandomPropagate.
        You can change this if you wish to experiment with different classes.
    :param seed_density: Consider a square of edge 2 * hop_length * total_spreads.
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

    device = device or (torch.device('cuda') if torch.cuda.is_available() else 
                        torch.device('mps') if torch.backends.mps.is_available() else
                        torch.device('cpu'))
        
    assert type(gap_crossing) == int, "gap_crossing must be an int"

    # Computes the seed probability
    # seed_probability =  seed_density / ((2 * hop_length * total_spreads) ** 2)

    def f(habitat, terrain):
        _, w, h = habitat.shape
        assert habitat.shape == terrain.shape, "Habitat and terrain must have the same shape."
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
            repopulator = analysis_class(hab, ter, num_spreads=num_spreads, spread_size=gap_crossing + 1, device=device).to(device)
        for i in range(num_batches):
            # Decides on the total spread and hop length.
            spread_size = 1 + gap_crossing
            dispersal_tmp = dispersal() if callable(dispersal) else dispersal
            num_spreads = int(0.5 + dispersal_tmp / spread_size)
            if callable(dispersal):
                repopulator = analysis_class(hab, ter, num_spreads=num_spreads, spread_size=spread_size, device=device).to(device)
            # Creates the seeds.
            seed_probability =  seed_density / ((1 + 2 * dispersal_tmp) ** 2)
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


def analyze_geotiffs(habitat_fn=None, 
                     landcover_fn=None,
                     permeability_dictionary=None,
                     permeability_fn=None,
                     permeability_scaling=1.0,
                     analysis_fn=None,
                     tile_size=1024,
                     border_size=64,
                     padding_size=0,
                     permeability_padding=0,
                     habitat_padding=0,
                     generate_gradient=True,
                     display_tiles=False,
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

    :param habitat_fn: filename of habitat geotiff, or GeoTiff object from habitat geotiff
    :param landcover_fn: filename of terrain geotiff, or GeoTiff object from terrain geotiff
    :param permeability_dictionary: terrain to permeability mapping dictionary. 
        Terrains not listed are assigned a permeability of 0. 
    :param permeability_fn: filename of permeability geotiff, or GeoTiff object from permeability geotiff. 
        This can be given in alternative to the above dictionary. 
    :param permeability_scaling: scaling factor for the permeability.  The permeability values
        are used per-pixel.  If you have them instead of per-pixel, per n-pixels (e.g., each pixel is 100m 
        but you computed the permeability for 1km), then you would specify here a scaling factor of 0.1 = 1 km / 100 m.  The permeability values p are rescaled to p ** permeability_scaling. 
    :param analysis_fn: function used for analysis.
    :param tile_size: dimensions of tile
    :param border_size: pixel border on each side of the tile. 
    :param padding_size: padding for the file.  The final file will have the same size of the input,
        except that a border of size border_size - padding_size is trimmed all around it. 
    :param permeability_padding: value to be used to pad the permeability.
    :param habitat_padding: value to be used to pad the habitat.
    :param include_border: whether to include the border in the output or not.
    :param display_tiles: True, to display tiles, or list of tiles interesting enough to display.
    :param minimum_habitat: minimum average of habitat to skip the tile.
    :param output_grad: file path for outputting the grad tif file.
    :param output_repop: file path for outputting the repop tif file.
        For this and output_grad, if None, then no file is generated.
    :param in_memory: whether the connectivity and flow should be saved in memory only. If so, then
        the files are not saved to disk, so the open files for connectivity and flow are returned.
    :param float_output: whether the output should be in floating point (True) or integer (False).
        If integer, values are rescaled to 0..255.
    '''
    if display_tiles is False:
        display_tiles = [] 

    compute_flow = generate_gradient and (output_grad_fn is not None if not in_memory else True)
    habitat_geotiff = None
    if habitat_fn is not None:
        habitat_geotiff = GeoTiff.from_file(habitat_fn) if type(habitat_fn) == str else habitat_fn
    # We specify permeability either via a terrain and a dictionary, or via a permeability file 
    # with scaling.  In either case, we have a permeability geotiff, which we then have to 
    # either scale or process via a dictionary.
    if permeability_fn is not None:
        # We are directly given a permeability file, which we scale via a constant. 
        permeability_via_dictionary = False
        permeability_geotiff = GeoTiff.from_file(permeability_fn) if type(permeability_fn) == str else permeability_fn
    else:
        # We use terrain and dictionary. 
        permeability_via_dictionary = True
        scaled_dictionary = {k: np.clip(v ** permeability_scaling, 0, 1) 
                             for k, v in permeability_dictionary.items()}
        permeability_geotiff = GeoTiff.from_file(landcover_fn) if type(landcover_fn) == str else landcover_fn
        
    # Checks that the habitat and permeability have the same size.
    if habitat_geotiff is not None:
        assert habitat_geotiff.width == permeability_geotiff.width and habitat_geotiff.height == permeability_geotiff.height, "Habitat and permeability must have the same size."
        
    def do_analysis(conn_file, flow_file):
        do_output = (conn_file is not None)
        # Reads the files.
        # Iterates through the tiles.
        per_reader = permeability_geotiff.get_reader(b=border_size, p=padding_size, w=tile_size, h=tile_size, pad_value=permeability_padding)
        if habitat_geotiff is None:
            # We create a reader where the habitat is always None.
            joint_reader = SingleIterator(per_reader)
        else:
            hab_reader = habitat_geotiff.get_reader(b=border_size, p=padding_size, w=tile_size, h=tile_size, pad_value=habitat_padding)
            joint_reader = zip(hab_reader, per_reader)        
        # We process each tile.
        for i, (hab_tile_iter, per_tile_iter) in enumerate(joint_reader):
            # print("Habitat tile:", "w:", hab_tile_iter.w, "h:", hab_tile_iter.h, "b:", hab_tile_iter.b, "c:", hab_tile_iter.c, "x:", hab_tile_iter.x, "y:", hab_tile_iter.y)
            # print("Habitat tile shape", hab_tile_iter.m.shape)
            # print("Permeability tile:", "w:", per_tile_iter.w, "h:", per_tile_iter.h, "b:", per_tile_iter.b, "c:", per_tile_iter.c, "x:", per_tile_iter.x, "y:", per_tile_iter.y)
            # print("Permeability tile shape", per_tile_iter.m.shape)
            habitat = np.clip(np.nan_to_num(hab_tile_iter.m), 0, 1) if hab_tile_iter is not None else None # Habitat tile
            raw_permeability = np.nan_to_num(per_tile_iter.m)  # Raw Permeability tile  
            if display_tiles is True or i in display_tiles:
                if habitat is not None:
                    hab_tile_iter.draw_tile(title="Habitat tile")
                per_tile_iter.draw_tile(title="Terrain tile")
            # We skip a tile if: 
            # - the habitat is not None, and too low, or, 
            # - the habitat is None, and the permeability is too low. 
            skip_tile = False
            if habitat is not None: 
                skip_tile = np.mean(habitat) < minimum_habitat
            if not skip_tile:            
                # Scales the permeability.
                if permeability_via_dictionary:
                    permeability = dict_translate(raw_permeability, scaled_dictionary, default_val=0)
                else:
                    raw_permeability = np.clip(raw_permeability, 0, 1)
                    # Scales the permeability.
                    permeability = raw_permeability ** permeability_scaling
                # Checks whether we have to skip due to low permeability. 
                if habitat is None:
                    skip_tile = np.mean(permeability) < minimum_habitat         
            if skip_tile:
                if do_output:
                    # These lines are here just to fix a bug into the production of the output geotiff,
                    # which does not set all to zero before the output is done.
                    conn_tile = per_tile_iter.clone_shape()
                    conn_file.set_tile(conn_tile, offset=border_size - padding_size) 
                    if compute_flow:
                        flow = per_tile_iter.clone_shape()
                        flow_file.set_tile(flow, offset=border_size - padding_size)
                continue
            # We process the tile.
            connectivity, flow = analysis_fn(np.ones_like(permeability) if habitat is None else habitat, permeability)
            # Normalizes the tiles, to fit into the geotiff format.
            # The population is simply normalized with a max of 255. After all it is in [0, 1].
            # We need to use type float because clam is not implemented for all types.
            if float_output:
                if isinstance(connectivity, np.ndarray):
                    connectivity_raster = np.expand_dims(connectivity.numpy().astype(float), axis=0)
                    flow_raster = np.expand_dims(np.log10(1. + flow) * 20., axis=0)
                else:
                    connectivity_raster = connectivity.detach().numpy().astype(float)
                    flow_raster = np.log10(1. + flow.detach().numpy().astype(float)) * 20.
            else:                        
                if isinstance(connectivity, np.ndarray):
                    connectivity_raster = np.expand_dims(np.clip(connectivity * 255, 0, 255).astype(np.uint8), axis=0)
                    flow_raster = np.expand_dims(np.clip(np.log10(1. + flow) * 20., 0, 255).astype(np.uint8), axis=0)
                else:
                    connectivity_raster = torch.clamp(connectivity.type(torch.float) * 255, 0, 255).type(torch.uint8)
                    flow_raster = torch.clamp(torch.log10(1. + flow.type(torch.float)) * 20., 0, 255).type(torch.uint8)


            # Prepares the tiles for writing.
            if do_output:
                # Writes the tiles.
                conn_tile = per_tile_iter.clone_shape()
                conn_tile.m = connectivity_raster
                conn_file.set_tile(conn_tile, offset=border_size - padding_size)
                if compute_flow:
                    flow_tile = per_tile_iter.clone_shape()
                    flow_tile.m = flow_raster
                    flow_file.set_tile(flow_tile, offset=border_size - padding_size)
            # Displays the output if so asked.
            if display_tiles is True or i in display_tiles:
                conn_tile.draw_tile(title="Connectivity")
                if compute_flow:
                    flow_tile.draw_tile(title="Flow")
            if report_progress:
                print(i, end=' ', flush=True)
        if report_progress:
           print()

    # Produce the outputs on disk.
    data_type = 'float32' if float_output else None
    # Computes the border size of the output geotiff, according to whether we wish 
    # to include the border in the output or not. 
    output_border_size = border_size - padding_size
    with permeability_geotiff.crop_to_new_file(
        output_border_size, data_type=data_type, filename=output_repop_fn, 
        in_memory=in_memory) as connectivity_output:
        with permeability_geotiff.crop_to_new_file(
            output_border_size, data_type=data_type, filename=output_grad_fn,
            in_memory=in_memory) if compute_flow else nullcontext() as flow_output:
            do_analysis(connectivity_output, flow_output)
    return connectivity_output, flow_output
