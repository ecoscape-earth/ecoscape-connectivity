import os
import ecoscape_connectivity

DATA_PATH="tests/assets"

HABITAT_PATH = os.path.join(DATA_PATH, "habitat_small.tif")
TERRAIN_PATH = os.path.join(DATA_PATH, "terrain_small.tif")
PERMEABILITY_PATH = os.path.join(DATA_PATH, "terrain_permeability.csv")

CONNECTIVITY_PATH = os.path.join(DATA_PATH, "Outputs/connectivity.tif")
FLOW_PATH = os.path.join(DATA_PATH, "Outputs/flow.tif")

def test_connectivity():
    ecoscape_connectivity.compute_connectivity(
        habitat_fn=HABITAT_PATH,
        terrain_fn=TERRAIN_PATH,
        permeability_dict=PERMEABILITY_PATH,
        connectivity_fn=CONNECTIVITY_PATH,
        flow_fn=FLOW_PATH,
        num_simulations=20
    )
