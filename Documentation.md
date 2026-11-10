<a id="connectivity"></a>

# connectivity

Compute connectivity and flow layers, the EcoScape algorithm's outputs.

These are the preferred entry points; they replace `compute_connectivity`
and `analyze_geotiffs`, which take and return scgt `GeoTiff` objects.
Parameter names follow the EcoScape paper.

Every layer is a path: anything `rasterio.open` accepts, so besides local
files also URLs, GDAL virtual paths (`/vsicurl/...`, `/vsis3/...`) and the
`.name` of a rasterio `MemoryFile` for a raster in memory. Paths, rather than
open datasets, keep the interface simple:

- reading is cheap: profiling showed that opening and reading the inputs
  takes under 0.2% of an analysis, and reusing an open dataset across
  several analyses on the same input saves nothing measurable;
- ownership is simple: EcoScape opens every file itself, read-only, and
  closes it when done;
- there are no surprises from how a caller opened a dataset: read/write
  mode, a dataset closed too early, or a MemoryFile collected while in use.

<a id="connectivity.ConnectivityLayers"></a>

## ConnectivityLayers Objects

```python
@dataclass(frozen=True)
class ConnectivityLayers()
```

Connectivity and flow computed in memory.

connectivity and flow are 2-D arrays; flow is None if it wasn't
computed. profile is the rasterio profile of the layers, with their
size, position on the map and data type, for writing them to a file.

<a id="connectivity.connectivity_and_flow"></a>

#### connectivity\_and\_flow

```python
def connectivity_and_flow(
        habitat_layer: str | None = None,
        permeability_layer: str | None = None,
        landcover_layer: str | None = None,
        permeability_table: dict[int, float] | None = None,
        permeability_exponent: float = 1.0,
        connectivity_layer: str | None = None,
        flow_layer: str | None = None,
        compute_flow: bool = True,
        dispersal_distance: Dispersal | None = None,
        num_gap_crossings: int | None = None,
        gap_crossing_distance: int = 1,
        num_simulations: int = 400,
        seed_density: float = 4,
        random_seed: int | None = None,
        batch_size: int = 1,
        connectivity_only_in_habitat: bool = True,
        device: str | torch.device | None = None,
        tile_size: int = 1000,
        border_size: int = 100,
        padding_size: int = 0,
        permeability_padding: float = 0,
        habitat_padding: float = 0,
        minimum_tile_habitat: float = 1e-4,
        float_output: bool = True,
        report_progress: bool = False) -> ConnectivityLayers | None
```

Compute the connectivity and flow layers of a species' habitat.

The landscape matrix permeability comes either from permeability_layer,
or from landcover_layer and permeability_table, which gives the
permeability of each landcover class.

**Arguments**:

- `habitat_layer`: path of the habitat layer: 1 in the habitat, 0
elsewhere. Without it, every pixel counts as habitat.
- `permeability_layer`: path of the landscape matrix permeability
layer, with values in [0, 1].
- `landcover_layer`: path of the landcover layer, with a landcover
class for each pixel; used with permeability_table.
- `permeability_table`: permeability of each landcover class; classes
not in the table have permeability 0.
- `permeability_exponent`: permeability is raised to this power, for
a pixel size other than the one the permeability is for. For
example, with 100 m pixels and permeability measured over 1 km, use
0.1.
- `connectivity_layer`: path to write the connectivity layer to. If
None, the layers are computed in memory and returned.
- `flow_layer`: path to write the flow layer to; required on disk if
compute_flow is True.
- `compute_flow`: whether to compute the flow layer. Connectivity
alone is faster and uses less memory.
- `dispersal_distance`: dispersal distance d, in pixels, or a
function that returns a random one for each batch of simulations,
such as half_cauchy. If None, it is num_gap_crossings times
gap_crossing_distance.
- `num_gap_crossings`: number of gap-crossings n, so that d = n * g;
used if dispersal_distance is None.
- `gap_crossing_distance`: gap-crossing distance g, in pixels: in
each propagation round, the species can cross up to g - 1 pixels of
any landcover.
- `num_simulations`: number of repopulation simulations to average.
- `seed_density`: density of seeds: each habitat pixel is a seed
with probability seed_density / (1 + 2 * d) ** 2.
- `random_seed`: seed for PyTorch's random number generator, which
every random choice in the simulation uses. The same inputs and seed
give identical results on the same device with the same PyTorch
version.
- `batch_size`: number of simulations run together; use the largest
that fits in GPU memory. num_simulations must be a multiple of it.
- `connectivity_only_in_habitat`: whether connectivity is 0 outside
the habitat, as in the paper.
- `device`: device for the computation, such as "cpu", "cuda" or
"mps"; if None, a GPU if there is one.
- `tile_size`: size of the square tiles the layers are processed in,
in pixels.
- `border_size`: border around each tile, in pixels, which gives the
simulation context; it should be larger than the dispersal distance.
- `padding_size`: padding around the layers, at most border_size.
The outputs are the inputs cropped by border_size - padding_size on
each side.
- `permeability_padding`: permeability (or landcover class) of the
padding.
- `habitat_padding`: habitat value of the padding.
- `minimum_tile_habitat`: tiles whose mean habitat, border included,
is below this are skipped and left at 0; without a habitat layer,
the mean permeability is used.
- `float_output`: whether the outputs are floating point; otherwise
they are bytes, with connectivity scaled to 0-255 and flow to
20 * log10(1 + flow), clipped to 0-255.
- `report_progress`: whether to print the number of each tile
analyzed.

**Returns**:

None if the layers are written to files; otherwise the layers.

<a id="connectivity.analyze_tiles"></a>

#### analyze\_tiles

```python
def analyze_tiles(analysis: AnalysisFunction,
                  habitat_layer: str | None = None,
                  permeability_layer: str | None = None,
                  landcover_layer: str | None = None,
                  permeability_table: dict[int, float] | None = None,
                  permeability_exponent: float = 1.0,
                  connectivity_layer: str | None = None,
                  flow_layer: str | None = None,
                  compute_flow: bool = True,
                  tile_size: int = 1000,
                  border_size: int = 100,
                  padding_size: int = 0,
                  permeability_padding: float = 0,
                  habitat_padding: float = 0,
                  minimum_tile_habitat: float = 1e-4,
                  float_output: bool = True,
                  report_progress: bool = False) -> ConnectivityLayers | None
```

Run an analysis over the layers tile by tile, and write or return its

connectivity and flow.

analysis takes a tile's habitat and permeability, each of shape
(1, height, width), border included, and returns connectivity and flow
of the same size, as numpy arrays or tensors; numpy arrays may leave
out the first axis. The other parameters are as for
connectivity_and_flow.

**Returns**:

None if the layers are written to files; otherwise the layers.

<a id="repopulation"></a>

# repopulation

Deprecated entry points, kept so existing code keeps working.

compute_connectivity and analyze_geotiffs translate their arguments and call
connectivity_and_flow and analyze_tiles in connectivity.py, which take paths
and use the paper's parameter names. They keep their own defaults, so their
results don't change.

<a id="repopulation.GeoTiffLike"></a>

## GeoTiffLike Objects

```python
class GeoTiffLike(Protocol)
```

An object holding an open rasterio dataset, such as a scgt GeoTiff.

<a id="repopulation.compute_connectivity"></a>

#### compute\_connectivity

```python
@deprecated(
    "compute_connectivity is deprecated: use connectivity_and_flow, which "
    "takes paths and the paper's parameter names.",
    category=FutureWarning,
)
def compute_connectivity(
    habitat_fn: str | GeoTiffLike | None = None,
    permeability_fn: str | GeoTiffLike | None = None,
    permeability_scaling: float = 1.0,
    landcover_fn: str | GeoTiffLike | None = None,
    permeability_dict: dict[int, float] | None = None,
    connectivity_fn: str | None = None,
    flow_fn: str | None = None,
    gap_crossing: int = 0,
    dispersal: Dispersal | None = None,
    num_gaps: int | None = None,
    num_simulations: int = 400,
    seed_density: float = 4,
    tile_size: int = 1000,
    border_size: int = 100,
    padding_size: int = 0,
    permeability_padding: float = 0,
    habitat_padding: float = 0,
    minimum_habitat: float = 1e-4,
    float_output: bool = True,
    random_seed: int | None = None,
    in_memory: bool = False,
    batch_size: int = 1,
    generate_flow_memory: bool = False,
    repopulation_only_in_habitat: bool = True,
    device: str | torch.device | None = None
) -> tuple[None, None] | ConnectivityLayers
```

Function that computes the connectivity.

.. deprecated::
    Use connectivity_and_flow, which takes paths and the paper's parameter
    names, and gives the same results. compute_connectivity calls it.

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

- `habitat_fn`: name of habitat geotiff, or GeoTiff (from the scgt package) object from habitat geotiff. This file must contain
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
(1 + 2 * dispersal) pixels.
The default of 4 means 1 seed every dispersal^2 pixels, approximately, and works well.
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
- `random_seed`: seed for PyTorch's random number generator, which every random choice
in the simulation uses, including half_cauchy dispersal. The same inputs and seed give
identical results on the same device with the same PyTorch version.
- `in_memory`: whether the connectivity and flow should be computed in memory only.
If so, then the files are not saved to disk, and connectivity_fn and flow_fn are
ignored. Flow is generated only if generate_flow_memory is True.
- `generate_flow_memory`: whether the flow should be generated in memory. Only used if
in_memory is True.
- `float_output`: use floating point output, generating a floating point tiff.
- `repopulation_only_in_habitat`: if True, then the repopulation only occurs in the habitat.
That's the default.  If False, the repopulation can be non-zero all over the output raster.
- `device`: the device to be used for the computation. If None, then the device is chosen
automatically.  Valid values include: 'cpu', 'cuda', 'mps'.

**Returns**:

(None, None) if in_memory is False; if in_memory is True, a ConnectivityLayers
with the connectivity and flow arrays and their rasterio profile; flow is None if
generate_flow_memory is False.

<a id="repopulation.analyze_geotiffs"></a>

#### analyze\_geotiffs

```python
@deprecated(
    "analyze_geotiffs is deprecated: use analyze_tiles, which takes paths "
    "and the paper's parameter names.",
    category=FutureWarning,
)
def analyze_geotiffs(
        habitat_fn: str | GeoTiffLike | None = None,
        landcover_fn: str | GeoTiffLike | None = None,
        permeability_dictionary: dict[int, float] | None = None,
        permeability_fn: str | GeoTiffLike | None = None,
        permeability_scaling: float = 1.0,
        analysis_fn: AnalysisFunction | None = None,
        tile_size: int = 1024,
        border_size: int = 64,
        padding_size: int = 0,
        permeability_padding: float = 0,
        habitat_padding: float = 0,
        generate_gradient: bool = True,
        display_tiles: bool | list[int] = False,
        minimum_habitat: float = 1e-4,
        output_repop_fn: str | None = None,
        output_grad_fn: str | None = None,
        report_progress: bool = False,
        in_memory: bool = False,
        float_output: bool = False) -> tuple[None, None] | ConnectivityLayers
```

Reads a geotiff (or better, a pair of habitat and terrain geotiffs),

iterating over the tiles, analyzing it with a specified analysis function,
and then writing the results back.

.. deprecated::
    Use analyze_tiles, which takes paths and the paper's parameter names,
    and gives the same results. analyze_geotiffs calls it.

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
- `generate_gradient`: whether to compute the flow.
- `display_tiles`: no longer supported, and ignored with a warning.
- `minimum_habitat`: minimum average of habitat to skip the tile.
- `output_repop_fn`: file path for the connectivity output.
- `output_grad_fn`: file path for the flow output; if None, no flow is written.
- `report_progress`: whether to print the number of each tile analyzed.
- `in_memory`: whether the connectivity and flow should be computed in memory only.
If so, then the files are not saved to disk.
- `float_output`: whether the output should be in floating point (True) or integer (False).
If integer, values are rescaled to 0..255.

**Returns**:

(None, None) if in_memory is False; if in_memory is True, a ConnectivityLayers
with the connectivity and flow arrays and their rasterio profile.

<a id="distributions"></a>

# distributions

<a id="distributions.constant"></a>

#### constant

```python
def constant(value: T) -> Callable[[], T]
```

Returns a function that always returns the same value.
This is useful to allow a deterministic number of spreads.

<a id="distributions.half_cauchy"></a>

#### half\_cauchy

```python
def half_cauchy(median: float, truncation: int) -> Callable[[], int]
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
def dict_translate(np_array: np.ndarray,
                   my_dict: dict[int, float],
                   default_val: float = 0) -> np.ndarray
```

Translates the terrain type according to a dictionary mapping
terrain type to values.

<a id="util.read_transmission_csv"></a>

#### read\_transmission\_csv

```python
def read_transmission_csv(fn: str) -> dict[int, float]
```

Reads a dictionary of terrain resistance or transmission in csv, producing a dictionary.

<a id="util.rescale_resistance"></a>

#### rescale\_resistance

```python
def rescale_resistance(d: dict[int, float], resolution_m: float,
                       hop_length: float) -> dict[int, float]
```

Resistance dictionaries are based on decay over 100m.
This function rescales the resistance to the value of the
actual hop length.

<a id="util.createdir_for_file"></a>

#### createdir\_for\_file

```python
def createdir_for_file(fn: str) -> None
```

Ensures that the path to a file exists.

<a id="util.SingleIterator"></a>

## SingleIterator Objects

```python
class SingleIterator()
```

Given an iterator, this class builds an iterator that returns
pairs of the form (None, i), where i is given by the iterator.

<a id="util.RandomPropagate"></a>

## RandomPropagate Objects

```python
class RandomPropagate(nn.Module)
```

Important: THIS is the function to use in the repopulation experiments.
This module models the repopulation of the habitat from a chosen percentage
of the seed places.  The terrain and habitat are parameters, and the input is a
similarly sized 0-1 (float) tensor of seed points.

<a id="util.RandomPropagate.__init__"></a>

#### \_\_init\_\_

```python
def __init__(habitat: torch.Tensor,
             terrain: torch.Tensor,
             num_spreads: int = 100,
             spread_size: int = 1,
             device: str | torch.device | None = None,
             update_threshold: float = 0.05,
             coin_toss_probability: float = 0.5) -> None
```

**Arguments**:

- `habitat`: torch tensor (2-dim) representing the habitat.
- `terrain`: torch tensor (2-dim) representing the terrain.
- `num_spreads`: number of bird spreads to use.  This is
the number of kernel spreads, in which birds that move by
spread_size pixels.
- `spread_size`: by how much (in pixels) birds spread in each spread.
This is 1 + gap_crossing.
- `device`: device to use for computation (cpu, cuda, mps, etc).

<a id="util.RandomPropagate.forward"></a>

#### forward

```python
def forward(seed: torch.Tensor) -> torch.Tensor
```

seed: a 0-1 (float) tensor of seed points.

<a id="util.analyze_tile_torch"></a>

#### analyze\_tile\_torch

```python
def analyze_tile_torch(
    device: str | torch.device | None = None,
    analysis_class: type[RandomPropagate] = RandomPropagate,
    seed_density: float = 4.0,
    produce_gradient: bool = False,
    batch_size: int = 1,
    dispersal: Dispersal = 20,
    num_simulations: int = 100,
    gap_crossing: int = 0,
    repopulation_only_in_habitat: bool = True
) -> Callable[[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]
```

This is the function that performs the analysis on a single tile.

The input and output to this function are in cpu, but the computation occurs in
the specified device.

**Arguments**:

- `device`: the device to be used, e.g., cpu, cuda, mps.
- `analysis_class`: class to be used for the analysis.  We recommend RandomPropagate.
You can change this if you wish to experiment with different classes.
- `seed_density`: Consider a square of edge (1 + 2 * dispersal) pixels.
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

