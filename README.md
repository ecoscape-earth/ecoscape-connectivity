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


The computation will be much faster if you run it with GPU support.
We recommend at least a T2-equivalent GPU or better. 

The main method is `compute_connectivity`, which computes the connectivity
and flow for a given raster.





<a id="repopulation"></a>

# repopulation

<a id="repopulation.RandomPropagate"></a>

## RandomPropagate Objects

```python
class RandomPropagate(nn.Module)
```

Important: THIS is the function to use in the repopulation experiments.
This module models the repopulation of the habitat from a chosen percentage
of the seed places.  The terrain and habitat are parameters, and the input is a
similarly sized 0-1 (float) tensor of seed points.

<a id="repopulation.RandomPropagate.__init__"></a>

#### \_\_init\_\_

```python
def __init__(habitat, terrain, num_spreads=100, spread_size=1, device=None)
```

**Arguments**:

- `habitat`: torch tensor (2-dim) representing the habitat.
- `terrain`: torch tensor (2-dim) representing the terrain.
- `num_spreads`: number of bird spreads to use
- `spread_size`: by how much (in pixels) birds spread in each spread.
- `device`: device to use for computation (cpu, cuda, mps, etc).

<a id="repopulation.RandomPropagate.forward"></a>

#### forward

```python
def forward(seed)
```

seed: a 0-1 (float) tensor of seed points.

<a id="repopulation.analyze_tile_torch"></a>

#### analyze\_tile\_torch

```python
def analyze_tile_torch(device=None,
                       analysis_class=RandomPropagate,
                       seed_density=4.0,
                       produce_gradient=False,
                       batch_size=1,
                       dispersal=20,
                       num_simulations=100,
                       gap_crossing=0,
                       repopulation_only_in_habitat=True)
```

This is the function that performs the analysis on a single tile.

The input and output to this function are in cpu, but the computation occurs in
the specified device.

**Arguments**:

- `device`: the device to be used, e.g., cpu, cuda, mps.
- `analysis_class`: class to be used for the analysis.  We recommend RandomPropagate.
You can change this if you wish to experiment with different classes.
- `seed_density`: Consider a square of edge 2 * hop_length * total_spreads.
In that square, there will be seed_density seeds on average.
- `produce_gradient`: boolean, whether to produce a gradient as result or not.
- `batch_size`: batch size for GPU calculations. For speed, use the largest 
batch size that fits in memory.
- `dispersal`: dispersal distance in pixels.
As above, if this is an integer, we do this constanst number of spreads
for all batches. Otherwise, If this is of the form of a function
(probability distribution), we run the function (and sample the distribution)
to get the dispersal distance.
- `num_simulations`: number of simulations to run. Must be a multiple of batch_size.
- `gap_crossing`: maximum number of pixels a bird can jump. 0 means only contiguous pixels.
- `repopulation_only_in_habitat`: if True, then the repopulation only occurs in the habitat.
If False, the repopulation can be non-zero all over the output raster.

<a id="repopulation.analyze_geotiffs"></a>

#### analyze\_geotiffs

```python
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
                     float_output=False)
```

Reads a geotiff (or better, a pair of habitat and terrain geotiffs),

iterating over the tiles, analyzing it with a specified analysis function,
and then writing the results back.

**Arguments**:

- `habitat_fn`: filename of habitat geotiff, or GeoTiff object from habitat geotiff
- `landcover_fn`: filename of terrain geotiff, or GeoTiff object from terrain geotiff
- `permeability_dictionary`: terrain to permeability mapping dictionary. 
Terrains not listed are assigned a permeability of 0.
- `permeability_fn`: filename of permeability geotiff, or GeoTiff object from permeability geotiff. 
This can be given in alternative to the above dictionary.
- `permeability_scaling`: scaling factor for the permeability.  The permeability values
are used per-pixel.  If you have them instead of per-pixel, per n-pixels (e.g., each pixel is 100m 
but you computed the permeability for 1km), then you would specify here a scaling factor of 0.1 = 1 km / 100 m.  The permeability values p are rescaled to p ** permeability_scaling.
- `analysis_fn`: function used for analysis.
- `tile_size`: dimensions of tile
- `border_size`: pixel border on each side of the tile.
- `padding_size`: padding for the file.  The final file will have the same size of the input,
except that a border of size border_size - padding_size is trimmed all around it.
- `permeability_padding`: value to be used to pad the permeability.
- `habitat_padding`: value to be used to pad the habitat.
- `include_border`: whether to include the border in the output or not.
- `display_tiles`: True, to display tiles, or list of tiles interesting enough to display.
- `minimum_habitat`: minimum average of habitat to skip the tile.
- `output_grad`: file path for outputting the grad tif file.
- `output_repop`: file path for outputting the repop tif file.
For this and output_grad, if None, then no file is generated.
- `in_memory`: whether the connectivity and flow should be saved in memory only. If so, then
the files are not saved to disk, so the open files for connectivity and flow are returned.
- `float_output`: whether the output should be in floating point (True) or integer (False).
If integer, values are rescaled to 0..255.

<a id="repopulation.compute_connectivity"></a>

#### compute\_connectivity

```python
def compute_connectivity(habitat_fn=None,
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
                         device=None)
```

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

**Arguments**:

- `habitat_fn`: name of habitat geotiff, or GeoTiff (from the scgt packaage) object from habitat geotiff. This file must contain
0 = non habitat, and 1 = habitat.
If this file is missing, then it is assumed that everywhere is suitable habitat, and that
only the permeability determines possible movement. This is useful for modeling mammals.
- `permeability_fn`: File name for permeability, or GeoTiff object (from scgt package) for the permeability. 
If this is given, the permeability is read from this file, and scaled according to 
permeability_scaling.  If this is not given, then the permeability
is derived from the landcover_fn file, and the dictionary.
- `permeability_scaling`: scaling factor for the permeability.  The permeability values
are used per-pixel.  If you have them instead of per-pixel, per n-pixels (e.g., each pixel is 100m 
but you computed the permeability for 1km), then you would specify here a scaling factor of 0.1 = 1 km / 100 m.  The permeability values p are rescaled to p ** permeability_scaling.
- `landcover_fn`: name of terrain geotiff, or GeoTiff object from terrain geotiff.  This file contains
terrain categories that are translated via permeability_dict.
- `permeability_dict`: Permeability dictionary.  Gives the permeability of each
terrain type, translating from the terrain codes, to the permeability in [0, 1].
If a terrain type is not found in the dictionary, it is assumed it has permeability 0.
- `connectivity_fn`: output file name for connectivity.
- `flow_fn`: output file name for flow.  If None, the flow is not computed, and the
computation is faster.
- `gap_crossing`: size of gap crossing in pixels. 0 means animals move via contiguous pixels.
- `dispersal`: dispersal distance in pixels.
- `num_gaps`: number of gaps to cross. Deprecated.  If dispersal is None, then this is used to 
compute the dispersal distance.  At least one of dispersal, num_gaps must be provided.
- `num_simulations`: Number of simulations that are done.
- `seed_density`: density of seeds.  There are this many seeds for every square with edge of
dispersal distance.
- `tile_size`: size of (square) tile in pixels.  This is the size that is processsed 
in one go.  Choose the tile as large as possible, so that it fits into the GPU memory.
- `border_size`: size of analysis border used on each tile in pixels. This has to be at least 
equal to the dispersal distance.
- `padding_size`: amount of padding around each tile.  If you specify 0 (no padding), then 
the output raster will have the same size as the input, except for a border of size border_size
all around it: so if the input is of size w, h and the border is of size b, then the output
will be of size w - 2 * b, h - 2 * b.  If you specify a padding of p, then the output will
have a border of size b - p, and the output will be of size w - 2 * (b - p), h - 2 * (b - p).
The padding cannot be greater than the border size.
- `permeability_padding`: value to be used to pad the permeability or terrain raster.
- `habitat_padding`: value to be used to pad the habitat raster.
- `batch_size`: batch size for GPU calculations.  Use the largest batch size that makes the 
computation fit into the GPU memory.
- `minimum_habitat`: if a tile has a fraction of habitat smaller than this, it is skipped.
This saves time in countries where the habitat is only on a small portion.
- `random_seed`: random seed, if desired.
- `in_memory`: whether the connectivity and flow should be saved in memory only.
If so, then the files are not saved to disk. Because such files would be deleted on close,
the open memory files will be returned as (repop_file, grad_file). Note that the parameters
connectivity_fn and flow_fn are ignored if this is set to True, and at least connectivity
will be returned. Flow is also generated only if generate_flow_memory is True.
- `generate_flow_memory`: whether the flow should be generated in memory. Only used if
in_memory is True.
- `float_output`: use floating point output, generating a floating point tiff.
- `repopulation_only_in_habitat`: if True, then the repopulation only occurs in the habitat. 
That's the default.  If False, the repopulation can be non-zero all over the output raster.
- `device`: the device to be used for the computation. If None, then the device is chosen
automatically.  Valid values include: 'cpu', 'cuda', 'mps'.

**Returns**:

(None, None) if in_memory is False, (repop_file, grad_file) if in_memory is True.
If in_memory is True, the caller should close the files with scgt's GeoTiff.close_memory_file()
once they are not needed anymore.

<a id="distributions"></a>

# distributions

<a id="distributions.constant"></a>

#### constant

```python
def constant(value)
```

Returns a function that always returns the same value.
This is useful to allow a deterministic number of spreads.

<a id="distributions.half_cauchy"></a>

#### half\_cauchy

```python
def half_cauchy(median, truncation)
```

A distribution that has been found useful in modeling animal dispersal is the
half-Cauchy distribution; see e.g. Paradis, Emmanuel, Stephen R. Baillie, and William J. Sutherland.
“Modeling Large-Scale Dispersal Distances.” Ecological Modelling 151, no. 2–3 (June 2002): 279–92.
https://doi.org/10.1016/S0304-3800(01)00487-2.

This function returns a function that, when called, returns a random sample from
a truncated half-Cauchy probability distribution.
The function takes as input the mean of the desired samples, and the truncation,
corresponding to the largest integer that can be returned.
To generate a sample, the function will sample a half-Cauchy distribution p.
If x ~ p is the sample, the function will return round(x), that is, x rounded to the
nearest integer.  We do not return the value 0, since for dispersal distances,
0 is not a useful value for a simulation.

Given the truncation, we obtain p by considering a half-Cauchy distribution truncated
to the interval [0 + 0.5, truncation + 0.5], where the +0.5 is there to accommodate for the
rounding.  We select the parameter sigma of the half-Cauchy distribution such that the
median, after such truncation, is equal to the input median.

<a id="util"></a>

# util

<a id="util.dict_translate"></a>

#### dict\_translate

```python
def dict_translate(np_arr, my_dict, default_val=0)
```

Translates the terrain type according to a dictionary mapping
terrain type to values.

<a id="util.read_resistance_csv"></a>

#### read\_resistance\_csv

```python
def read_resistance_csv(fn)
```

Reads a dictionary of terrain to resistance in csv, producing a dictionary.

<a id="util.read_transmission_csv"></a>

#### read\_transmission\_csv

```python
def read_transmission_csv(fn)
```

Reads a dictionary of terrain resistance or transmission in csv, producing a dictionary.

<a id="util.rescale_resistance"></a>

#### rescale\_resistance

```python
def rescale_resistance(d, resolution_m, hop_length)
```

Resistance dictionaries are based on decay over 100m.
This function rescales the resistance to the value of the
actual hop length.

<a id="util.createdir_for_file"></a>

#### createdir\_for\_file

```python
def createdir_for_file(fn)
```

Ensures that the path to a file exists.

<a id="util.SingleIterator"></a>

## SingleIterator Objects

```python
class SingleIterator(object)
```

Given an iterator, this class builds an iterator that returns
pairs of the form (None, i), where i is given by the iterator.

