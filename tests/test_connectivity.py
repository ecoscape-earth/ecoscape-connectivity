import os
import csv
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ecoscape_connectivity

DATA_PATH="tests/assets"

HABITAT_PATH = os.path.join(DATA_PATH, "habitat_small.tif")
LANDCOVER_PATH = os.path.join(DATA_PATH, "terrain_small.tif")
PERMEABILITY_PATH = os.path.join(DATA_PATH, "terrain_permeability.csv")

with open(PERMEABILITY_PATH, mode='r') as infile:
    reader = csv.reader(infile)
    permeability_dict = {rows[0]:rows[1] for rows in reader}

CONNECTIVITY_PATH = os.path.join(DATA_PATH, "Outputs/connectivity.tif")
FLOW_PATH = os.path.join(DATA_PATH, "Outputs/flow.tif")

def test_connectivity():
    ecoscape_connectivity.compute_connectivity(
        habitat_fn=HABITAT_PATH,
        landcover_fn=LANDCOVER_PATH,
        permeability_dict=permeability_dict,
        connectivity_fn=CONNECTIVITY_PATH,
        flow_fn=FLOW_PATH,
        single_tile=False,
        border_size=50,
        tile_size=200,
        num_simulations=2,
        dispersal=ecoscape_connectivity.half_cauchy(15, 40),
    )

test_connectivity()

def test_connectivity_single():
    ecoscape_connectivity.compute_connectivity(
        habitat_fn=HABITAT_PATH,
        landcover_fn=LANDCOVER_PATH,
        permeability_dict=permeability_dict,
        connectivity_fn=CONNECTIVITY_PATH,
        flow_fn=FLOW_PATH,
        single_tile=True,
        border_size=50,
        num_simulations=2,
        dispersal=ecoscape_connectivity.half_cauchy(15, 40),
    )

test_connectivity_single()
