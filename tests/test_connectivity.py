import os
import ecoscape_connectivity
import numpy as np

DATA_PATH="tests/assets"

HABITAT_PATH = os.path.join(DATA_PATH, "habitat_small.tif")
TERRAIN_PATH = os.path.join(DATA_PATH, "terrain_small.tif")
PERMEABILITY_PATH = os.path.join(DATA_PATH, "terrain_permeability.csv")

CONNECTIVITY_PATH = os.path.join(DATA_PATH, "Outputs/connectivity.tif")
FLOW_PATH = os.path.join(DATA_PATH, "Outputs/flow.tif")

def gap_calc(r, p):
    # returns a random sample from the negative binomial probability distribution
    def f():
        return np.random.negative_binomial(r, p) + 1
    return f

def num_gaps_calc(n):
    # returns a random sample from the normal probability distribution
    def f():
            return np.random.normal(n)
    return f

def test_connectivity():
    ecoscape_connectivity.compute_connectivity(
        habitat_fn=HABITAT_PATH,
        terrain_fn=TERRAIN_PATH,
        permeability_dict=PERMEABILITY_PATH,
        connectivity_fn=CONNECTIVITY_PATH,
        flow_fn=FLOW_PATH,
        num_simulations=20,
        gap_crossing=2,
        num_gaps=10,
    )

    # ecoscape_connectivity.compute_connectivity(
    #     habitat_fn=HABITAT_PATH,
    #     terrain_fn=TERRAIN_PATH,
    #     permeability_dict=PERMEABILITY_PATH,
    #     connectivity_fn=CONNECTIVITY_PATH,
    #     flow_fn=FLOW_PATH,
    #     num_simulations=20,
    #     gap_crossing=gap_calc(4, 0.8),
    #     num_gaps=num_gaps_calc(10),
    # )

test_connectivity()
