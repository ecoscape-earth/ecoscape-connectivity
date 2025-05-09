import argparse
import os
from .repopulation import compute_connectivity
from .util import read_transmission_csv, createdir_for_file

def main(args):
    # Reads and transltes the resistance dictionary.
    transmission_d = read_transmission_csv(args.permeability)
    # Creates output folders, if missing.
    createdir_for_file(args.connectivity)
    if args.flow is not None:
        createdir_for_file(args.flow)

    compute_connectivity(
        habitat_fn=args.habitat,
        landcover_fn=args.landcover,
        permeability_fn=args.permeability,
        permeability_dict=transmission_d,
        connectivity_fn=args.connectivity,
        flow_fn=args.flow,
        num_simulations=args.num_simulations,
        gap_crossing=args.gap_crossing,
        dispersal=args.dispersal,
        num_gaps=args.num_gap_crossings,
        seed_density=args.seed_density,
        tile_size=args.tile_size,
        border_size=args.border_size,
        padding_size=args.padding_size,
        batch_size=args.batch_size,
    )

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--habitat', type=os.path.abspath, default=None,
                        help='Path of input habitat geotiff.')
    parser.add_argument('--landcover', type=os.path.abspath, default=None,
                        help='Path of input landcover geotiff.')
    parser.add_argument('--permeability', type=os.path.abspath, default=None,
                        help='Path of input CSV dictionary of the terrain permeability.'
                        'This should be a CSV with at least two columns, map_code, and transmission, the latter between 0 and 1.')
    parser.add_argument('--connectivity', type=os.path.abspath, default=None,
                        help='Path to output geotiff file for connectivity.')
    parser.add_argument('--flow', type=os.path.abspath, default=None,
                        help='Path to output geotiff file for flow. If missing, no flow is computed.')
    parser.add_argument('--num_simulations', type=int, default=400,
                        help='Number of simulations to perform.')
    parser.add_argument('--gap_crossing', type=int, default=0,
                        help='Gap-crossing distance in pixels.')
    parser.add_argument('--dispersal', type=float, default=40,
                        help='Dispersal distance in pixels.')
    parser.add_argument('--num_gap_crossings', type=int, default=None,
                        help='Number of gap crossings in a dispersal. Deprecated. ' +
                        'Use the dissipation parameter instead.')
    parser.add_argument('--seed_density', type=int, default=4,
                        help='Density of random seeds in the simulation.')
    parser.add_argument('--tile_size', type=int, default=1000,
                        help="Edge of (square) tiles for analysis, in pixels.")
    parser.add_argument('--border_size', type=int, default=256,
                        help="Border needed for analysis, in pixels")
    parser.add_argument('--padding_size', type=int, default=256,
                        help="Padding for the analysis, in pixels")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for GPU computation.")

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()

