<!-- README.md is generated from README-prototype.md and the docstrings.
Edit README-prototype.md or the docstrings, then run `make docs`. -->
# EcoScape Connectivity Computation

[![Coverage badge](https://raw.githubusercontent.com/ecoscape-earth/ecoscape-connectivity/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/ecoscape-earth/ecoscape-connectivity/blob/python-coverage-comment-action-data/htmlcov/index.html)
![Mutants caught](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ecoscape-earth/ecoscape-connectivity/main/badges/mutation-score.json)

This package implements the computation of connectivity and flow according to the EcoScape algorithm.

## Authors

* Luca de Alfaro (luca@ucsc.edu)
* Natalia Ocampo-Peñuela (nocampop@ucsc.edu)
* Coen Adler (ctadler@ucsc.edu)
* Artie Nazarov (anazarov@ucsc.edu)
* Natalie Valett (nvalett@ucsc.edu)
* Jasmine Tai (cjtai@ucsc.edu)

## Installing

    pip install ecoscape-connectivity

EcoScape reads and writes rasters with rasterio. The rasterio package includes GDAL; you don't need to install it yourself.

The computation will be much faster if you run it with GPU support. We recommend at least a T2-equivalent GPU or better.

## Usage

`connectivity_and_flow` computes the connectivity and flow layers of a species' habitat. Its parameters are named as in the EcoScape paper, except `gap_crossing`, which counts the pixels the species can cross: one less than the paper's gap-crossing distance g. Each layer is a path: a local file, or anything else `rasterio.open` accepts, such as a URL.

```python
import ecoscape_connectivity

permeability_table = ecoscape_connectivity.read_transmission_csv("permeability.csv")
ecoscape_connectivity.connectivity_and_flow(
    habitat_layer="habitat.tif",
    landcover_layer="landcover.tif",
    permeability_table=permeability_table,
    connectivity_layer="connectivity.tif",
    flow_layer="flow.tif",
    dispersal_distance=40,
    # Pixels crossed in each propagation round: the paper's gap-crossing
    # distance g is one more, here g = 2.
    gap_crossing=1,
)
```

The landscape matrix permeability comes either from a landcover layer and a permeability table, as above, or from a permeability layer (`permeability_layer`). Without output paths, `connectivity_and_flow` returns the layers as numpy arrays with their rasterio profile:

```python
layers = ecoscape_connectivity.connectivity_and_flow(
    habitat_layer="habitat.tif",
    landcover_layer="landcover.tif",
    permeability_table=permeability_table,
    dispersal_distance=40,
)
layers.connectivity, layers.flow, layers.profile
```

`analyze_tiles` runs your own analysis function over the layers, tile by tile. The full documentation of every function follows below.

## Command line

    ecoscape-connectivity --habitat habitat.tif --landcover landcover.tif \
        --permeability permeability.csv --connectivity connectivity.tif \
        --flow flow.tif --dispersal 40

Run `ecoscape-connectivity --help` for every option.

## Migrating from `compute_connectivity`

`compute_connectivity` and `analyze_geotiffs` are deprecated, and a future version will remove them. They still work and give the same results, but warn with a `FutureWarning`. Replace them with `connectivity_and_flow` and `analyze_tiles`:

| `compute_connectivity` | `connectivity_and_flow` |
| --- | --- |
| `habitat_fn` | `habitat_layer` |
| `landcover_fn` | `landcover_layer` |
| `permeability_fn` | `permeability_layer` |
| `permeability_dict` | `permeability_table` |
| `permeability_scaling` | `permeability_exponent` |
| `connectivity_fn` | `connectivity_layer` |
| `flow_fn` | `flow_layer`; `connectivity_and_flow` computes flow unless `compute_flow=False` |
| `dispersal` | `dispersal_distance` |
| `num_gaps` | `num_gap_crossings` |
| `minimum_habitat` | `minimum_tile_habitat` |
| `repopulation_only_in_habitat` | `connectivity_only_in_habitat` |
| `in_memory=True`, `generate_flow_memory` | leave out `connectivity_layer`; `compute_flow` |

`analyze_geotiffs` becomes `analyze_tiles`, which takes the analysis function first, with the same new names, and `output_repop_fn`, `output_grad_fn` and `generate_gradient` become `connectivity_layer`, `flow_layer` and `compute_flow`. `analyze_tiles` has the same defaults as `connectivity_and_flow`: tiles of 1000 pixels, a border of 100, and floating point output. 

Pass paths instead of scgt `GeoTiff` objects. In memory, both the old and the new functions now return the layers as numpy arrays.

## Running the tests

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). Install the exact dependency versions from `uv.lock` into `.venv`, then run the tests:

    make install
    make test

