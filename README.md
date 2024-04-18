# EcoScape Connectivity Computation

This package implements the computation of connectivity and flow according 
to the EcoScape algorithm. 

## Authors

* Luca de Alfaro (luca@ucsc.edu)
* Natalia Ocampo-Peñuela (nocampop@ucsc.edu)
* Coen Adler (ctadler@ucsc.edu)
* Artie Nazarov (anazarov@ucsc.edu)
* Natalie Valett (nvalett@ucsc.edu)
* Jasmine Tai (cjtai@ucsc.edu)

## Usage

The package can be used both from the command line, and as a python module. 
For command line options, do: 

    ecoscape-connectivity --help

As a Python module, the main function is `compute_connectivity`: 

```python
from ecoscape_connectivity import compute_connectivity

def compute_connectivity(habitat_fn=None,
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
                         minimum_habitat=1e-4,
                         in_memory=False,
                         generate_flow_memory=False)
```

The computation will be much faster if you run it with GPU support. 

The output connectivity and flow are encoded in the output geotiffs as follows: 

- For connectivity, the values from [0, 1] are linearly rescaled to the range 0..255 and encoded as integers, so that 0 corresponds to no connectivity, and 255 to maximum connectivity. 
- For flow, the values of $f \in [0, \infty)$ are encoded in log-scale via 
  $20 \cdot log_{10} (1 + f)$ (so that the flow is expressed in dB, like 
  sound intensity), and clipped to integers in the 0..255 range.

**Arguments**:

- `habitat_fn`: name of habitat geotiff, or GeoTiff object from habitat geotiff. This file
must contain 0 = non habitat, and 1 = habitat.
- `terrain_fn`: name of the landscape matrix geotiff, or GeoTiff object from landscape
matrix geotiff.  This file contains terrain categories that are translated via permeability_dict.
- `connectivity_fn`: output file name for connectivity.
- `flow_fn`: output file name for flow.  If None, the flow is not computed, and the
computation is faster.
- `permeability_dict`: Permeability dictionary.  Gives the permeability of each
terrain type, translating from the terrain codes, to the permeability in [0, 1].
If a terrain type is not found in the dictionary, it is assumed it has permeability 0.
- `gap_crossing`: size of gap crossing in pixels. This can be either a 
  constant, or a function that returns a value each time it is called. 
- `dispersal`: dispersal distance.  This can 
  be either a constant, or a function that, each time called, returns a 
  value; the latter is useful for simulating dispersal distance 
  distributions. The dispersal distance is measured in pixels. The 
  underlying simulation will simulate a number `num_gaps` of gap-crossing 
  events equal to `num_gaps = dispersal / (gap_crossing + 1)`, where of 
  course `gap_crossing` can be 0. 
  So if animals cannot cross any gap (`gap_crossing = 0`) and the dispersal 
  is 40 px, then the number of simulated gap crossings (or better, pixels 
  expansions) will be 40. 
- `num_simulations`: Number of simulations that are done.
- `seed_density`: density of seeds.  There are this many seeds for every square with edge of
dispersal distance.
- `single_tile`: if True, instead of iterating over small tiles, tries to read the input as a
single large tile.  This is faster, but might not fit into memory.
- `tile_size`: size of (square) tile in pixels.
- `tile_border`: size of tile border in pixels.
- `minimum_habitat`: if a tile has a fraction of habitat smaller than this, it is skipped.
This saves time in countries where the habitat is only on a small portion.
- `in_memory`: whether the connectivity and flow should be saved in memory only.
If so, then the files are not saved to disk. Because such files would be deleted on close,
the open memory files will be returned as `(repop_file, grad_file)`. Note that the parameters
`connectivity_fn` and `flow_fn` are ignored if this is set to True, and at least connectivity
will be returned. Flow is also generated only if `generate_flow_memory` is True.
- `generate_flow_memory`: whether the flow should be generated in memory. Only used if
in_memory is True.

## Example Notebooks

Here you can find a [Colab Notebook](https://drive.google.com/file/d/1Pz6lLyIs8Ju2UGkNtZqcNR72cFzn8UYc/view?usp=sharing) that 
demonstrates connectivity computation. 

## Notes on Parameters

We are currently using these algorithms with a pixel size of about 300m x 300m. 
With this pixel size, we use a gap_crossing of 0 (animals move via 
contiguous pixels). 

## Dispersal Distance Distributions

Distributions that can be passed to `dispersal` are defined in `distributions.
py`, and are: 

- `constant`: always returns the same value (not very useful; one might as 
  well pass a constant).
- `half_cauchy`: using `half_cauchy(median, truncation)` returns a function 
  that, each time it is called, returns an integer sampled from a 
  half-Cauchy distribution with given median and truncation.  One can use 
  this function as input for `num_gaps` to simulate dispersal distance 
  distributions via, for instance, `num_gaps=half_cauchy(40, 160)`. Please 
  refer to `distributions.py` for the details. 



