# Notebooks

These Jupyter notebooks show how EcoScape computes connectivity and flow, and produce some of the figures used in the paper. They are examples and experiments, not tests: `make test` does not run them.

| Notebook | What it shows | Data | Run time |
|---|---|---|---|
| [display_connectivity.ipynb](display_connectivity.ipynb) | Runs `compute_connectivity` on a small habitat and landcover map, as one tile and as 200-pixel tiles, and draws the connectivity and flow maps. | `habitat_small.tif`, `terrain_small.tif`, `terrain_permeability.csv` | Minutes on a CPU |
| [half_cauchy.ipynb](half_cauchy.ipynb) | Draws 4,000 samples from the half-Cauchy dispersal distribution (median 40, truncated at 160 pixels) and plots a histogram. | None | Seconds |
| [propagation.ipynb](propagation.ipynb) | Runs `compute_connectivity` on a California habitat with a fixed dispersal distance and with a half-Cauchy dispersal distance, then spreads animals from a single central point on a uniform 100 x 100 landscape, with and without coin tosses. | `habitat_CA.tif`, `terrain_CA.tif`, `permeability_CA.csv` | Slow without a GPU (1,000 simulations per run) |
| [SimulatedLandscapes.ipynb](SimulatedLandscapes.ipynb) | Builds a small artificial landscape of circular habitat patches joined by corridors (like Figure 1 in the paper), then computes connectivity and flow with the default settings, with an update threshold of 0, and without coin tosses. Saves each figure as a PDF. | Generated in the notebook | Slow without a GPU |

## Running the notebooks

The notebooks need the EcoScape dependencies, Jupyter and matplotlib.

1. Install the dependencies (Jupyter and matplotlib are in the development dependencies):

       python3 -m pip install -r requirements.txt -r dev-dependencies.txt

2. Start Jupyter from this `docs` folder, so the notebooks find the `ecoscape_connectivity` package in the folder above:

       cd docs
       jupyter notebook

3. Open a notebook and choose **Run > Run All Cells**.

To run them in VS Code instead, open the `.ipynb` file, click **Select Kernel** in the top right, and choose the Python environment you installed the dependencies into.

The notebooks pick a GPU automatically (CUDA, or MPS on Apple computers) and fall back to the CPU. The two slow notebooks can also be uploaded to [Google Colab](https://colab.research.google.com) and run on a free GPU; you will need to upload the package and data files too.

## Outputs

- `display_connectivity` and `propagation` read their data from, and write GeoTIFFs to, `assets/` and `assets/Outputs/` at the top of the repository.
- `SimulatedLandscapes` writes PDF figures (`connectivity.pdf`, `flow.pdf` and variants) to this folder.
