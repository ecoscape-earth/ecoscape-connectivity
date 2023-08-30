Ecoscape Connectivity Version 0.0.1

SYNOPSIS
========

**ecoscape** --H <habitat_path> --T <terrain_path> --R <resistance_path> --r <repop_path> [--g <grad_path>] [--d <hop_distance>] [--s <num_spreads>] [--S <num_simulations>] [--D <seed_density>]  

DESCRIPTION
===========

Ecoscape is a command-line utility for modeling the connectivity and habitat quality by simulating the spread of bird species across a landscape. It computes the repopulation of a bird species based on current habitat and its terrain compatibility. It can be used to inform scientists and conservationists what areas are most for a certian species, and to simulate the effects that changing landscapes will have on bird habitat health.

Options
-------

-H, --habitat=<habitat_path>

:   Filename to a geotiff of the bird\'s habitat.

-T, --terrain=<terrain_path>

:   Filename to a geotiff of the terrain.

-R, --resistance=<resistance_path>

:   Filename to a CSV dictionary of the terrain value resistance.

-r, --repopulation=<repop_path>

:   Filename to output geotiff file for repopulation (connectivity).

-g, --gradient=<grad_path>

:   Filename to output geotiff file for gradient. (optional, if not specified gradient will not be computed)

-d, --hop_distance=<hop_distance>

:   the length of a bird hop, measured in pixels. So if each square has of 300m, so a hop distance of 2 corresponds to 600m. (default 4)

-s, --num_spreads=<num_spreads>

:   the number of hops a bird can do during dispersal (default 400)

-S, --num_simulations=<num_simulations>

:   the number of simulations performed for the spread process; a typical value is several hundreds. (default 200)

FILES
=====

Input Files:
* Habitat: a Geotiff of the habitat raster for the bird
* Terrain: a Geotiff of the terrain description for California
* Resistance: CSV file raw terrain resistance for each bird. This can be obtained from IUCN data

NOTE: Habitat and Terrain must be of type .tif, and have the same CRS, resolution and size

Optput Files:
* Repopulation: The connectivity Geotiff containing the output of the repopulation, in geotiff format. Repopulation is represented as values between 0 and 1, multiplied by 255 and encoded as integers. 
* Gradient: Geotiff containing the gradient computed for the repopulation, in geotiff format. For a repopulation value of $r$, we output $20 log_10(1 + r)$, clipped between 0 and 255. 

ENVIRONMENT
===========

**CONNECTIVITY_ENV**

:   TBD

BUGS
====

See GitHub Issues: <https://github.com/ecoscape-earth/connectivity/issues>

Contributors
======

## Contributors (alphabetically)

* Coen Adler
* Luca de Alfaro
* Artie Nazarov
* Natalia Ocampo-Peñuela
* Tyler Sorensen
* Jasmine Tai
* Natalie Valett