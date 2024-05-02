import os
import csv
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ecoscape_connectivity

DATA_PATH="tests/assets"

PERMEABILITY_PATH = os.path.join(DATA_PATH, "permeability.tif")
PERMEABILITY_SCALING = 1 / 0.5731019
CONNECTIVITY_PATH = os.path.join(DATA_PATH, "Outputs/connectivity.tif")
FLOW_PATH = os.path.join(DATA_PATH, "Outputs/flow.tif")

def test_connectivity(float_output=False):
    ecoscape_connectivity.compute_connectivity(
        permeability_fn=PERMEABILITY_PATH,   
        permeability_scaling=PERMEABILITY_SCALING,     
        connectivity_fn=CONNECTIVITY_PATH,
        flow_fn=FLOW_PATH,
        single_tile=False,
        num_simulations=2,
        dispersal=ecoscape_connectivity.half_cauchy(4, 8),
        float_output=True,
    )

test_connectivity()
test_connectivity(float_output=True)
