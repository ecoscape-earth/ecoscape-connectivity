"""Compute connectivity and flow layers, the EcoScape algorithm's outputs.

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
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile

from .raster import create_output, read_tile, tile_windows, write_core
from .util import AnalysisFunction, Dispersal, analyze_tile_torch, dict_translate


@dataclass(frozen=True)
class ConnectivityLayers:
    """Connectivity and flow computed in memory.

    connectivity and flow are 2-D arrays; flow is None if it wasn't
    computed. profile is the rasterio profile of the layers, with their
    size, position on the map and data type, for writing them to a file.
    """

    connectivity: np.ndarray
    flow: np.ndarray | None
    profile: dict[str, Any]


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
    report_progress: bool = False,
) -> ConnectivityLayers | None:
    """Compute the connectivity and flow layers of a species' habitat.

    The landscape matrix permeability comes either from permeability_layer,
    or from landcover_layer and permeability_table, which gives the
    permeability of each landcover class.

    :param habitat_layer: path of the habitat layer: 1 in the habitat, 0
        elsewhere. Without it, every pixel counts as habitat.
    :param permeability_layer: path of the landscape matrix permeability
        layer, with values in [0, 1].
    :param landcover_layer: path of the landcover layer, with a landcover
        class for each pixel; used with permeability_table.
    :param permeability_table: permeability of each landcover class; classes
        not in the table have permeability 0.
    :param permeability_exponent: permeability is raised to this power, for
        a pixel size other than the one the permeability is for. For
        example, with 100 m pixels and permeability measured over 1 km, use
        0.1.
    :param connectivity_layer: path to write the connectivity layer to. If
        None, the layers are computed in memory and returned.
    :param flow_layer: path to write the flow layer to; required on disk if
        compute_flow is True.
    :param compute_flow: whether to compute the flow layer. Connectivity
        alone is faster and uses less memory.
    :param dispersal_distance: dispersal distance d, in pixels, or a
        function that returns a random one for each batch of simulations,
        such as half_cauchy. If None, it is num_gap_crossings times
        gap_crossing_distance.
    :param num_gap_crossings: number of gap-crossings n, so that d = n * g;
        used if dispersal_distance is None.
    :param gap_crossing_distance: gap-crossing distance g, in pixels: in
        each propagation round, the species can cross up to g - 1 pixels of
        any landcover.
    :param num_simulations: number of repopulation simulations to average.
    :param seed_density: density of seeds: each habitat pixel is a seed
        with probability seed_density / (1 + 2 * d) ** 2.
    :param random_seed: seed for PyTorch's random number generator, which
        every random choice in the simulation uses. The same inputs and seed
        give identical results on the same device with the same PyTorch
        version.
    :param batch_size: number of simulations run together; use the largest
        that fits in GPU memory. num_simulations must be a multiple of it.
    :param connectivity_only_in_habitat: whether connectivity is 0 outside
        the habitat, as in the paper.
    :param device: device for the computation, such as "cpu", "cuda" or
        "mps"; if None, a GPU if there is one.
    :param tile_size: size of the square tiles the layers are processed in,
        in pixels.
    :param border_size: border around each tile, in pixels, which gives the
        simulation context; it should be larger than the dispersal distance.
    :param padding_size: padding around the layers, at most border_size.
        The outputs are the inputs cropped by border_size - padding_size on
        each side.
    :param permeability_padding: permeability (or landcover class) of the
        padding.
    :param habitat_padding: habitat value of the padding.
    :param minimum_tile_habitat: tiles whose mean habitat, border included,
        is below this are skipped and left at 0; without a habitat layer,
        the mean permeability is used.
    :param float_output: whether the outputs are floating point; otherwise
        they are bytes, with connectivity scaled to 0-255 and flow to
        20 * log10(1 + flow), clipped to 0-255.
    :param report_progress: whether to print the number of each tile
        analyzed.
    :return: None if the layers are written to files; otherwise the layers.
    """
    if dispersal_distance is None:
        if num_gap_crossings is None:
            raise ValueError("Give dispersal_distance or num_gap_crossings.")
        dispersal_distance = num_gap_crossings * gap_crossing_distance
    if random_seed is not None:
        torch.manual_seed(random_seed)
    analysis = analyze_tile_torch(
        device=device,
        seed_density=seed_density,
        produce_gradient=compute_flow,
        dispersal=dispersal_distance,
        batch_size=batch_size,
        num_simulations=num_simulations,
        # The simulation's gap_crossing is the number of pixels crossed,
        # g - 1.
        gap_crossing=gap_crossing_distance - 1,
        repopulation_only_in_habitat=connectivity_only_in_habitat,
    )
    return analyze_tiles(
        analysis,
        habitat_layer=habitat_layer,
        permeability_layer=permeability_layer,
        landcover_layer=landcover_layer,
        permeability_table=permeability_table,
        permeability_exponent=permeability_exponent,
        connectivity_layer=connectivity_layer,
        flow_layer=flow_layer,
        compute_flow=compute_flow,
        tile_size=tile_size,
        border_size=border_size,
        padding_size=padding_size,
        permeability_padding=permeability_padding,
        habitat_padding=habitat_padding,
        minimum_tile_habitat=minimum_tile_habitat,
        float_output=float_output,
        report_progress=report_progress,
    )


def analyze_tiles(
    analysis: AnalysisFunction,
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
    report_progress: bool = False,
) -> ConnectivityLayers | None:
    """Run an analysis over the layers tile by tile, and write or return its
    connectivity and flow.

    analysis takes a tile's habitat and permeability, each of shape
    (1, height, width), border included, and returns connectivity and flow
    of the same size, as numpy arrays or tensors; numpy arrays may leave
    out the first axis. The other parameters are as for
    connectivity_and_flow.

    :return: None if the layers are written to files; otherwise the layers.
    """
    if (permeability_layer is None) == (landcover_layer is None):
        raise ValueError("Give either permeability_layer or landcover_layer.")
    if landcover_layer is not None and permeability_table is None:
        raise ValueError("landcover_layer needs a permeability_table.")
    in_memory = connectivity_layer is None
    if not in_memory and compute_flow and flow_layer is None:
        raise ValueError("Give flow_layer, or set compute_flow to False.")
    source_path = (
        permeability_layer if permeability_layer is not None else landcover_layer
    )
    assert source_path is not None
    with (
        (
            rasterio.open(habitat_layer) if habitat_layer is not None else nullcontext()
        ) as habitat,
        rasterio.open(source_path) as source,
    ):
        if habitat is not None and (habitat.width, habitat.height) != (
            source.width,
            source.height,
        ):
            raise ValueError(
                "The habitat and permeability layers have different sizes."
            )
        offset = border_size - padding_size
        dtype = "float32" if float_output else "uint8"
        if in_memory:
            # The MemoryFiles must outlive the outputs created in them.
            with (
                MemoryFile() as connectivity_file,
                MemoryFile() as flow_file,
                create_output(source, offset, dtype, connectivity_file) as connectivity,
                (
                    create_output(source, offset, dtype, flow_file)
                    if compute_flow
                    else nullcontext()
                ) as flow,
            ):
                _run_tiles(
                    analysis,
                    habitat,
                    source,
                    connectivity,
                    flow,
                    permeability_table,
                    permeability_exponent,
                    tile_size,
                    border_size,
                    padding_size,
                    permeability_padding,
                    habitat_padding,
                    minimum_tile_habitat,
                    float_output,
                    report_progress,
                )
                return ConnectivityLayers(
                    connectivity=connectivity.read(1),
                    flow=flow.read(1) if flow is not None else None,
                    profile=dict(connectivity.profile),
                )
        assert connectivity_layer is not None
        with (
            create_output(source, offset, dtype, connectivity_layer) as connectivity,
            (
                create_output(source, offset, dtype, flow_layer)
                if compute_flow and flow_layer is not None
                else nullcontext()
            ) as flow,
        ):
            _run_tiles(
                analysis,
                habitat,
                source,
                connectivity,
                flow,
                permeability_table,
                permeability_exponent,
                tile_size,
                border_size,
                padding_size,
                permeability_padding,
                habitat_padding,
                minimum_tile_habitat,
                float_output,
                report_progress,
            )
    return None


def _run_tiles(
    analysis: AnalysisFunction,
    habitat_dataset: DatasetReader | None,
    source: DatasetReader,
    connectivity: DatasetWriter,
    flow: DatasetWriter | None,
    permeability_table: dict[int, float] | None,
    permeability_exponent: float,
    tile_size: int,
    border_size: int,
    padding_size: int,
    permeability_padding: float,
    habitat_padding: float,
    minimum_tile_habitat: float,
    float_output: bool,
    report_progress: bool,
) -> None:
    """Analyze each tile and write its outputs; skipped tiles stay at 0."""
    scaled_table = (
        {
            code: np.clip(value**permeability_exponent, 0, 1)
            for code, value in permeability_table.items()
        }
        if permeability_table is not None
        else None
    )
    offset = border_size - padding_size
    windows = tile_windows(
        source.width, source.height, tile_size, border_size, padding_size
    )
    for i, window in enumerate(windows):
        habitat = (
            np.clip(
                np.nan_to_num(read_tile(habitat_dataset, window, habitat_padding)), 0, 1
            )
            if habitat_dataset is not None
            else None
        )
        if habitat is not None and np.mean(habitat) < minimum_tile_habitat:
            continue
        raw = np.nan_to_num(read_tile(source, window, permeability_padding))
        if scaled_table is not None:
            permeability = dict_translate(raw, scaled_table, default_val=0)
        else:
            permeability = np.clip(raw, 0, 1) ** permeability_exponent
        if habitat is None and np.mean(permeability) < minimum_tile_habitat:
            continue
        tile_connectivity, tile_flow = analysis(
            np.ones_like(permeability) if habitat is None else habitat, permeability
        )
        write_core(
            connectivity,
            window,
            _connectivity_output(tile_connectivity, float_output),
            offset,
        )
        if flow is not None:
            write_core(flow, window, _flow_output(tile_flow, float_output), offset)
        if report_progress:  # pragma: no mutate block
            print(i, end=" ", flush=True)
    if report_progress:  # pragma: no mutate block
        print()


def _connectivity_output(
    values: np.ndarray | torch.Tensor, float_output: bool
) -> np.ndarray:
    """Return a tile's connectivity as written: as is, or scaled to 0-255,
    with a band axis.
    """
    if isinstance(values, torch.Tensor):
        if float_output:
            array = values.detach().numpy().astype(float)
        else:
            array = (
                torch.clamp(values.detach().type(torch.float) * 255, 0, 255)
                .type(torch.uint8)
                .numpy()
            )
    elif float_output:
        array = values.astype(float)
    else:
        array = np.clip(values * 255, 0, 255).astype(np.uint8)
    return array if array.ndim == 3 else array[np.newaxis]


def _flow_output(values: np.ndarray | torch.Tensor, float_output: bool) -> np.ndarray:
    """Return a tile's flow as written: 20 * log10(1 + flow), in bytes
    clipped to 0-255, with a band axis.
    """
    if isinstance(values, torch.Tensor):
        if float_output:
            array = np.log10(1.0 + values.detach().numpy().astype(float)) * 20.0
        else:
            array = (
                torch.clamp(
                    torch.log10(1.0 + values.detach().type(torch.float)) * 20.0, 0, 255
                )
                .type(torch.uint8)
                .numpy()
            )
    elif float_output:
        array = np.log10(1.0 + values) * 20.0
    else:
        array = np.clip(np.log10(1.0 + values) * 20.0, 0, 255).astype(np.uint8)
    return array if array.ndim == 3 else array[np.newaxis]
