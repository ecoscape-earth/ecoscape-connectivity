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

## Example Notebooks

Here you can find a [Colab Notebook](https://drive.google.com/file/d/1Pz6lLyIs8Ju2UGkNtZqcNR72cFzn8UYc/view?usp=sharing) that 
demonstrates connectivity computation. 

## Usage

The package can be used both from the command line, and as a python module. 
For command line options, do: 

    ecoscape-connectivity --help

As a Python module, the main function is `compute_connectivity`.  It can be imported with: 

```python
from ecoscape_connectivity import compute_connectivity
```

It has the following signature: 

```python
def compute_connectivity()
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
        single_tile=False,
        tile_size=1000,
        border_size=200,
        minimum_habitat=1e-4,
        float_output=True,
        random_seed=None,
        in_memory=False,
        generate_flow_memory=False,
```

### GPU recommended

The computation will be much faster if you run it with GPU support.
We recommend at least a T2-equivalent GPU or better. 

### Output

Given an input raster of size $n \times m$, and a border size for the analysis of $b$, the output will consist in rasters of size $(n - 2b) \times (m - 2b)$, so that the border is _not_ included in the output. 
The output consists in the connectivity raster, and if requested, in the flow raster. 
Such layers are encoded as geotiffs as follows. 

If `float_output` is True (the default), the outputs are floating point rasters, with the following encoding:

- For connectivity, the values are in the range [0, 1], where 0 corresponds to no connectivity, and 1 to maximum connectivity.
- For flow, the values are in the range [0, $\infty$), and are encoded as $20 \log_{10}(1 + f)$, where $f$ is the flow.  Thus, the flow is expressed in a logarithmic scale, which helps in visualizing it, given its wide range. 

If `float_output` is False, the outputs are 8-bit integer rasters, with the following encoding:

- For connectivity, the values from [0, 1] are linearly rescaled to the range 0..255 and encoded as integers, so that 0 corresponds to no connectivity, and 255 to maximum connectivity. 
- For flow, the values of $f \in [0, \infty)$ are encoded in log-scale via 
  $20 \cdot log_{10} (1 + f)$ (so that the flow is expressed in dB, like 
  sound intensity), and clipped to integers in the 0..255 range.

The use of `float_output = False` is not recommended, and it may become deprecated in the future. 

### Arguments: 

- `habitat_fn`: name of habitat geotiff, or GeoTiff object from habitat geotiff. This file
must contain 0 = non habitat, and 1 = habitat.
- `permeability_fn`: name of the permeability geotiff, or GeoTiff object from permeability geotiff.  This is the file that specifies the permeability values for each pixel.  The permeability values are in [0, 1].  Alternatively, if this parameter is missing, one can specify the permeability via `terrain_fn` and `permeability_dict` (see below). 
- `permeability_scaling`: optional scaling factor for the permeability layer.  This is useful if the permeability values are not in [0, 1], but in a different range.  The permeability values are multiplied by this factor.
- `terrain_fn`: name of the landscape matrix geotiff, or GeoTiff object from landscape matrix geotiff.  This file contains terrain categories, encoded as integers.  These terrain categories are then translated via `permeability_dict` to permeability values. 
- `permeability_dict`: Permeability dictionary.  Gives the permeability of each terrain type, translating from the terrain codes, to the permeability in [0, 1]. If a terrain type is not found in the dictionary, it is assumed it has permeability 0.
- `connectivity_fn`: output file name for the computed connectivity raster.
- `flow_fn`: output file name for flow raster.  If None, the flow is not computed, and the computation is faster and  uses less memory.
- `gap_crossing`: size of gap crossing in pixels.  
- `dispersal`: dispersal distance, in pixels, of the species.  This can be either a constant, or a function that, each time called, returns a value; the latter is useful for simulating dispersal distance distributions. EcoScape will simulate bird spread from seed sites `dispersal / (gap_crossing + 1)` times, where each time, birds can move up to `gap_crossing + 1` pixels from their previous location.
- `num_simulations`: Number of simulations. EcoScape performs this number of simulations, and outputs the average; more simulations means more accurate results, but also more time.  A value between 100-1000 is recommended. 
- `seed_density`: density of seeds.  There are this many seeds for every square with edge of 2 * dispersal distance.  The value of 4, the default, thus means on average 1 seed for each dispersal x dispersal square.  The seeds are placed at random within the square. We recommend not changing this value. 
- `single_tile`: if True, instead of iterating over small tiles, tries to read the input as a single large tile.  This is faster, but might not fit into memory.
- `tile_size`: size of (square) tile in pixels. If `single_tile` is False, the computation is done by iterating over tiles of this size.
- `border_size`: size of analysis border in pixels.  This border size must be greater than the maximum dispersal distance for the birds.  The border is used to avoid edge effects in the computation of connectivity and flow, since the flow from a pixel depends on the pixels around it for a radius equal to the dispersal distance. 
- `minimum_habitat`: if a tile has a fraction of habitat smaller than this, it is skipped, since the connectivity and flow would be 0. 
- `float_output`: if True (the default), the output consists of geotiffs with floating point values.  If False, the output is rescaled to 8-bit integers via multiplication by 255 and conversion to integer. 
- `random_seed`: used to initialize the random number generator used in seed selection and bird movement.  If None, the random number generator is initialized with a random seed.
- `in_memory`: whether the connectivity and flow should be saved in memory only. If so, the results of the computation are returned as a pair `(repop_file, grad_file)`. Note that the parameters `connectivity_fn` and `flow_fn` are ignored if this is set to True, and at least connectivity will be returned. Flow is also generated only if `generate_flow_memory` is True.
- `generate_flow_memory`: whether the flow should be generated in memory. Only used if in_memory is True.

## Notes on Parameters

We are currently using these algorithms with a pixel size of about 300m x 300m. 
With this pixel size, we use a gap_crossing of 0 (movement via 
contiguous pixels). 

## Dispersal Distance Distributions

Distributions that can be passed to `dispersal` are defined in `distributions.
py`. 

To use them, you can do:

```python
from ecoscape_connectivity.distributions import constant, half_cauchy
```

The distributions are: 

- `constant`: always returns the same value (not very useful; one might as 
  well pass a constant).
- `half_cauchy`: using `half_cauchy(median, truncation)` returns a function 
  that, each time it is called, returns an integer sampled from a 
  half-Cauchy distribution with given median and truncation.  One can use 
  this function as input for `num_gaps` to simulate dispersal distance 
  distributions via, for instance, `num_gaps=half_cauchy(40, 160)`. Please 
  refer to `distributions.py` for the details. 

Make sure that the border is greater than the maximum dispersal distance 
that can be generated (i.e., the truncation)



