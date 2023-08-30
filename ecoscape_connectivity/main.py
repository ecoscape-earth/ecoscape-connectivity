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
        terrain_fn=args.terrain,
        permeability_dict=transmission_d,
        connectivity_fn=args.connectivity,
        flow_fn=args.flow,
        num_simulations=args.num_simulations,
        gap_crossing=args.gap_crossing,
        num_gaps=args.num_gap_crossings,
        single_tile=args.single_tile,
        tile_size=args.tile_size,
        tile_border=args.tile_border,
    )

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--habitat', type=os.path.abspath, default=None,
                        help='Filename to a geotiff of the bird\'s habitat.')
    parser.add_argument('--terrain', type=os.path.abspath, default=None,
                        help='Filename to a geotiff of the terrain.')
    parser.add_argument('--permeability', type=os.path.abspath, default=None,
                        help='Filename to a CSV dictionary of the terrain permeability.'
                        'This should be a CSV with two columns, map_code, and transmission, the latter between 0 and 1.')
    parser.add_argument('--connectivity', type=os.path.abspath, default=None,
                        help='Filename to output geotiff file for connectivity.')
    parser.add_argument('--flow', type=os.path.abspath, default=None,
                        help='Filename to output geotiff file for flow. If missing, no flow is computed.')
    parser.add_argument('--num_simulations', type=int, default=400,
                        help='Number of simulations to perform.')
    parser.add_argument('--gap_crossing', type=int, default=2,
                        help='Gap-crossing distance in pixels.')
    parser.add_argument('--num_gap_crossings', type=int, default=10,
                        help='Number of gap crossings in a dispersal.')
    parser.add_argument('--seed_density', type=int, default=4,
                        help='Density of random seeds in the simulation.')
    parser.add_argument('--single_tile', action="store_true", default=False,
                        help="Processes the geotiffs in a single tile.")
    parser.add_argument('--tile_size', type=int, default=1000,
                        help="Edge of (square) tiles for analysis, in pixels.")
    parser.add_argument('--tile_border', type=int, default=256,
                        help="Border around a tile, in pixels")

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()

