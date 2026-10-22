<!-- README.md is generated from README-prototype.md and the docstrings.
Edit README-prototype.md or the docstrings, then run `make docs`. -->
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

## Running the tests

Install the dependencies and development dependencies, then run the
tests:

    python3 -m pip install -r requirements.txt -r dev-dependencies.txt
    make test

The tests also need GDAL, which is not yet listed in `requirements.txt`
(see issue #17).





