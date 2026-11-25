import argparse
import os

from .connectivity import connectivity_and_flow
from .util import createdir_for_file, read_transmission_csv


def main(args: argparse.Namespace) -> None:
    """Compute connectivity, and flow if --flow is given, from a habitat and a
    landcover layer, with the permeability of each landcover class from the
    --permeability table.
    """
    # Creates output folders, if missing.
    createdir_for_file(args.connectivity)
    if args.flow is not None:
        createdir_for_file(args.flow)

    connectivity_and_flow(
        habitat_layer=args.habitat,
        landcover_layer=args.landcover,
        permeability_table=read_transmission_csv(args.permeability),
        connectivity_layer=args.connectivity,
        flow_layer=args.flow,
        compute_flow=args.flow is not None,
        num_simulations=args.num_simulations,
        gap_crossing=args.gap_crossing,
        dispersal_distance=args.dispersal,
        num_gap_crossings=args.num_gap_crossings,
        seed_density=args.seed_density,
        tile_size=args.tile_size,
        border_size=args.border_size,
        padding_size=args.padding_size,
        batch_size=args.batch_size,
    )


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat",
        type=os.path.abspath,
        default=None,
        help="Path of input habitat geotiff.",
    )
    parser.add_argument(
        "--landcover",
        type=os.path.abspath,
        default=None,
        help="Path of input landcover geotiff.",
    )
    parser.add_argument(
        "--permeability",
        type=os.path.abspath,
        default=None,
        help="Path of input CSV dictionary of the terrain permeability."
        "This should be a CSV with at least two columns, map_code, and transmission, the latter between 0 and 1.",
    )
    parser.add_argument(
        "--connectivity",
        type=os.path.abspath,
        default=None,
        help="Path to output geotiff file for connectivity.",
    )
    parser.add_argument(
        "--flow",
        type=os.path.abspath,
        default=None,
        help="Path to output geotiff file for flow. If missing, no flow is computed.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=400,
        help="Number of simulations to perform.",
    )
    parser.add_argument(
        "--gap_crossing", type=int, default=0, help="Gap-crossing distance in pixels."
    )
    parser.add_argument(
        "--dispersal", type=float, default=40, help="Dispersal distance in pixels."
    )
    parser.add_argument(
        "--num_gap_crossings",
        type=int,
        default=None,
        help="Number of gap crossings in a dispersal. Deprecated. "
        + "Use the dissipation parameter instead.",
    )
    parser.add_argument(
        "--seed_density",
        type=int,
        default=4,
        help="Density of random seeds in the simulation.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1000,
        help="Edge of (square) tiles for analysis, in pixels.",
    )
    parser.add_argument(
        "--border_size",
        type=int,
        default=256,
        help="Border needed for analysis, in pixels",
    )
    parser.add_argument(
        "--padding_size",
        type=int,
        default=256,
        help="Padding for the analysis, in pixels",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for GPU computation."
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
