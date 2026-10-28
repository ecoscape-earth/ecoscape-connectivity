<!-- README.md is generated from README-prototype.md and the docstrings.
Edit README-prototype.md or the docstrings, then run `make docs`. -->
# EcoScape Connectivity Computation

This package implements the computation of connectivity and flow according to the EcoScape algorithm.

## Authors

* Luca de Alfaro (luca@ucsc.edu)
* Natalia Ocampo-Peñuela (nocampop@ucsc.edu)
* Coen Adler (ctadler@ucsc.edu)
* Artie Nazarov (anazarov@ucsc.edu)
* Natalie Valett (nvalett@ucsc.edu)
* Jasmine Tai (cjtai@ucsc.edu)


The computation will be much faster if you run it with GPU support. We recommend at least a T2-equivalent GPU or better.

The main method is `compute_connectivity`, which computes the connectivity and flow for a given raster.

## Running the tests

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and the GDAL library (for example, `brew install gdal` on macOS). The GDAL version must match the one set in `pyproject.toml` (currently 3.13.3); check yours with `gdal-config --version`.

Install the exact dependency versions from `uv.lock` into `.venv`, then run the tests:

    make install
    make test





