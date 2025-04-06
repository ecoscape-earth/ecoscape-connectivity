import os
import csv
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ecoscape_connectivity

DATA_PATH="tests/assets"

PERMEABILITY_PATH = os.path.join(DATA_PATH, "permeability.tif")
SCALED_PERMEABILITY_PATH = os.path.join(DATA_PATH, "scaled_permeability.tif")

from scgt import GeoTiff
g = GeoTiff.from_file(PERMEABILITY_PATH)
t = g.get_all_as_tile()
t.m = np.clip(t.m, 0, 1)
t.m = t.m * (1 / np.max(t.m))
with g.clone_shape(SCALED_PERMEABILITY_PATH) as gg:
    gg.set_tile(t)

CONNECTIVITY_PATH = os.path.join(DATA_PATH, "Outputs/connectivity.tif")
FLOW_PATH = os.path.join(DATA_PATH, "Outputs/flow.tif")

def test_connectivity(float_output=False):
    ecoscape_connectivity.compute_connectivity(
        permeability_fn=SCALED_PERMEABILITY_PATH,   
        connectivity_fn=CONNECTIVITY_PATH,
        flow_fn=FLOW_PATH,
        tile_size=512,
        border_size=10,
        padding_size=10,
        num_simulations=1,
        dispersal=10,
        float_output=True,
    )

test_connectivity()
test_connectivity(float_output=True)
