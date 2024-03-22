import os
import ecoscape_connectivity
import numpy as np
import csv
import math
import scipy

DATA_PATH="tests/assets"

HABITAT_PATH = os.path.join(DATA_PATH, "habitat_small.tif")
TERRAIN_PATH = os.path.join(DATA_PATH, "terrain_small.tif")
PERMEABILITY_PATH = os.path.join(DATA_PATH, "terrain_permeability.csv")

with open(PERMEABILITY_PATH, mode='r') as infile:
    reader = csv.reader(infile)
    permeability_dict = {rows[0]:rows[1] for rows in reader}

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
            return int(np.random.normal(n))
    return f

def half_cauchy(median, truncation):
    # returns a random sample from a truncated half cauchy probability distribution
    # Requirement: median < truncation/2
    assert median < truncation/2
    # Solving for sigma using cdf formula knowing the median probability
    sigma = (2*truncation-1)**0.5*(2*median-1) / (8*truncation-16*median+4)**0.5
    cdf_dif = []
    prev_cdf = 2/math.pi * math.atan((0.5)/sigma)
    for i in range(1,truncation+1):
        cur_cdf = 2/math.pi * math.atan((i+0.5)/sigma)
        cdf_dif.append(cur_cdf-prev_cdf)
        prev_cdf = cur_cdf
    probs = np.array(cdf_dif) * (1/np.sum(cdf_dif))
    
    def f():
        return int(np.random.choice(range(1,truncation+1), p=probs))
    return f
    

def test_connectivity():
    # ecoscape_connectivity.compute_connectivity(
    #     habitat_fn=HABITAT_PATH,
    #     terrain_fn=TERRAIN_PATH,
    #     permeability_dict=permeability_dict,
    #     connectivity_fn=CONNECTIVITY_PATH,
    #     flow_fn=FLOW_PATH,
    #     num_simulations=20,
    #     gap_crossing=2,
    #     num_gaps=10,
    # )

    ecoscape_connectivity.compute_connectivity(
        habitat_fn=HABITAT_PATH,
        terrain_fn=TERRAIN_PATH,
        permeability_dict=permeability_dict,
        connectivity_fn=CONNECTIVITY_PATH,
        flow_fn=FLOW_PATH,
        num_simulations=20,
        gap_crossing=1,
        num_gaps=half_cauchy(15, 40),
    )

test_connectivity()
