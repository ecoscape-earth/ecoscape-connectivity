"""Deprecated entry points, kept so existing code keeps working.

compute_connectivity and analyze_geotiffs translate their arguments and call
connectivity_and_flow and analyze_tiles in connectivity.py, which take paths
and use the paper's parameter names. They keep their own defaults, so their
results don't change.
"""

import warnings
from typing import Any, Protocol

import torch
from typing_extensions import deprecated

from .connectivity import ConnectivityLayers, analyze_tiles, connectivity_and_flow
from .util import AnalysisFunction, Dispersal


class GeoTiffLike(Protocol):
    """An object holding an open rasterio dataset, such as a scgt GeoTiff."""

    dataset: Any


def _layer_path(layer: str | GeoTiffLike | None) -> str | None:
    """Return the path of a layer given as a path or a GeoTiff.

    A GeoTiff's dataset knows its path, for a file on disk or in memory.
    """
    if layer is None or isinstance(layer, str):
        return layer
    return layer.dataset.name


# FutureWarning, unlike DeprecationWarning, shows by default in notebooks and
# scripts; the decorator also flags calls in editors and type checkers.
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
    device: str | torch.device | None = None,
) -> tuple[None, None] | ConnectivityLayers:
    """
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

    :param habitat_fn: name of habitat geotiff, or GeoTiff (from the scgt package) object from habitat geotiff. This file must contain
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
        (1 + 2 * dispersal) pixels.
        The default of 4 means 1 seed every dispersal^2 pixels, approximately, and works well.
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
    :param random_seed: seed for PyTorch's random number generator, which every random choice
        in the simulation uses, including half_cauchy dispersal. The same inputs and seed give
        identical results on the same device with the same PyTorch version.
    :param in_memory: whether the connectivity and flow should be computed in memory only.
        If so, then the files are not saved to disk, and connectivity_fn and flow_fn are
        ignored. Flow is generated only if generate_flow_memory is True.
    :param generate_flow_memory: whether the flow should be generated in memory. Only used if
        in_memory is True.
    :param float_output: use floating point output, generating a floating point tiff.
    :param repopulation_only_in_habitat: if True, then the repopulation only occurs in the habitat.
        That's the default.  If False, the repopulation can be non-zero all over the output raster.
    :param device: the device to be used for the computation. If None, then the device is chosen
        automatically.  Valid values include: 'cpu', 'cuda', 'mps'.
    :return: (None, None) if in_memory is False; if in_memory is True, a ConnectivityLayers
        with the connectivity and flow arrays and their rasterio profile; flow is None if
        generate_flow_memory is False.
    """
    if not in_memory and connectivity_fn is None:
        raise ValueError("Give connectivity_fn, or set in_memory to True.")
    layers = connectivity_and_flow(
        habitat_layer=_layer_path(habitat_fn),
        permeability_layer=_layer_path(permeability_fn),
        landcover_layer=_layer_path(landcover_fn),
        permeability_table=permeability_dict,
        permeability_exponent=permeability_scaling,
        connectivity_layer=None if in_memory else connectivity_fn,
        flow_layer=None if in_memory else flow_fn,
        compute_flow=generate_flow_memory if in_memory else flow_fn is not None,
        dispersal_distance=dispersal,
        num_gap_crossings=num_gaps,
        # gap_crossing counts the pixels crossed; the paper's gap-crossing
        # distance is one more.
        gap_crossing_distance=gap_crossing + 1,
        num_simulations=num_simulations,
        seed_density=seed_density,
        random_seed=random_seed,
        batch_size=batch_size,
        connectivity_only_in_habitat=repopulation_only_in_habitat,
        device=device,
        tile_size=tile_size,
        border_size=border_size,
        padding_size=padding_size,
        permeability_padding=permeability_padding,
        habitat_padding=habitat_padding,
        minimum_tile_habitat=minimum_habitat,
        float_output=float_output,
    )
    return layers if layers is not None else (None, None)


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
    float_output: bool = False,
) -> tuple[None, None] | ConnectivityLayers:
    """
    Reads a geotiff (or better, a pair of habitat and terrain geotiffs),
    iterating over the tiles, analyzing it with a specified analysis function,
    and then writing the results back.

    .. deprecated::
        Use analyze_tiles, which takes paths and the paper's parameter names,
        and gives the same results. analyze_geotiffs calls it.

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
    :param generate_gradient: whether to compute the flow.
    :param display_tiles: no longer supported, and ignored with a warning.
    :param minimum_habitat: minimum average of habitat to skip the tile.
    :param output_repop_fn: file path for the connectivity output.
    :param output_grad_fn: file path for the flow output; if None, no flow is written.
    :param report_progress: whether to print the number of each tile analyzed.
    :param in_memory: whether the connectivity and flow should be computed in memory only.
        If so, then the files are not saved to disk.
    :param float_output: whether the output should be in floating point (True) or integer (False).
        If integer, values are rescaled to 0..255.
    :return: (None, None) if in_memory is False; if in_memory is True, a ConnectivityLayers
        with the connectivity and flow arrays and their rasterio profile.
    """
    if analysis_fn is None:
        raise ValueError("analysis_fn is required.")
    if not in_memory and output_repop_fn is None:
        raise ValueError("Give output_repop_fn, or set in_memory to True.")
    if display_tiles:
        warnings.warn(
            "display_tiles is no longer supported, and is ignored.",
            FutureWarning,
            stacklevel=2,
        )
    layers = analyze_tiles(
        analysis_fn,
        habitat_layer=_layer_path(habitat_fn),
        permeability_layer=_layer_path(permeability_fn),
        landcover_layer=_layer_path(landcover_fn),
        permeability_table=permeability_dictionary,
        permeability_exponent=permeability_scaling,
        connectivity_layer=None if in_memory else output_repop_fn,
        flow_layer=None if in_memory else output_grad_fn,
        compute_flow=generate_gradient and (in_memory or output_grad_fn is not None),
        tile_size=tile_size,
        border_size=border_size,
        padding_size=padding_size,
        permeability_padding=permeability_padding,
        habitat_padding=habitat_padding,
        minimum_tile_habitat=minimum_habitat,
        float_output=float_output,
        report_progress=report_progress,
    )
    return layers if layers is not None else (None, None)
