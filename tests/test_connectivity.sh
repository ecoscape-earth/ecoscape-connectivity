# #!/bin/sh

DATA_PATH=$(pwd)"/tests/assets"

HABITAT_PATH=$DATA_PATH"/habitat_small.tif"
TERRAIN_PATH=$DATA_PATH"/terrain_small.tif"
PERMEABILITY_PATH=$DATA_PATH"/terrain_permeability.csv"

CONNECTIVITY_PATH=$DATA_PATH"/Outputs/connectivity.tif"
FLOW_PATH=$DATA_PATH"/Outputs/flow.tif"


echo "Test ecoscape.py"

python3 ecoscape_connectivity/main.py --habitat "$HABITAT_PATH" --terrain "$TERRAIN_PATH" --permeability "$PERMEABILITY_PATH" --connectivity "$CONNECTIVITY_PATH" --flow "$FLOW_PATH" --gap_crossing 2 --num_gap_crossing 10 --seed_density 4 --num_simulations 4
