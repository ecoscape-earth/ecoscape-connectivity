# EcoScape Connectivity Computation

This package implements the computation of connectivity and flow according 
to the EcoScape algorithm. 

## Authors

* Luca de Alfaro (main author, code, luca@ucsc.edu)
* Natalia Ocampo-Peñuela (main contributor, ecology, nocampop@ucsc.edu)
* Coen Adler (ctadler@ucsc.edu)
* Artie Nazarov (anazarov@ucsc.edu)
* Natalie Valett (nvalett@ucsc.edu)
* Jasmine Tai (cjtai@ucsc.edu)

## Usage.

The package can be used both from the command line, and as a python module. 
For command line options, 

As a Python module, the main function is compute_connectivity: 

```python
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
                         minimum_habitat=1e-4)
```

Function that computes the connectivity. This is the main function in the module.

**Arguments**:

- `habitat_fn`: name of habitat geotiff. This file must contain 0 = non habitat,
and 1 = habitat.
- `terrain_fn`: name of terrain file.  This file contains terrain categories that are
translated via permeability_dict.
- `connectivity_fn`: output file name for connectivity.
- `flow_fn`: output file name for flow.  If None, the flow is not computed, and the
computation is faster.
- `permeability_dict`: Permeability dictionary.  Gives the permeability of each
terrain type, translating from the terrain codes, to the permeability in [0, 1].
If a terrain type is not found in the dictionary, it is assumed it has permeability 0.
- `gap_crossing`: size of gap crossing in pixels.
- `num_gaps`: number of gaps that can be crossed during dispersal.
- `num_simulations`: Number of simulations that are done.
- `seed_density`: density of seeds.  There are this many seeds for every square with edge of
dispersal distance.
- `single_tile`: if True, instead of iterating over small tiles, tries to read the input as a
single large tile.  This is faster, but might not fit into memory.
- `tile_size`: size of (square) tile in pixels.
- `tile_border`: size of tile border in pixels.
- `minimum_habitat`: if a tile has a fraction of habitat smaller than this, it is skipped.
This saves time in countries where the habitat is only on a small portion.


