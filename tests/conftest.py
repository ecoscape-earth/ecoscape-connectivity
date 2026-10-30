"""Fixtures shared by the tests."""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

Dump = Callable[..., None]


@pytest.fixture
def dump(request: pytest.FixtureRequest) -> Dump:
    """Return a function that draws 2-D arrays side by side as an image, for inspection.

    Each test writes to tests/output/<test file>/<test name>.png, which
    is kept after the run and ignored by git. Parametrized names such as
    test_spreads[1-1] become test_spreads_1-1.
    """
    file_name = re.sub(r"[^\w-]+", "_", request.node.name).strip("_") + ".png"
    path = Path(__file__).parent / "output" / request.path.stem / file_name
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(**arrays: torch.Tensor | np.ndarray) -> None:
        figure = Figure(figsize=(3 * len(arrays) + 1, 3.5), layout="constrained")
        figure.suptitle(f"{request.node.name} (gray: exactly 0)")
        # Exact zeros are masked and drawn in gray, so they stand out from
        # small values, which are nearly black.
        colormap = colormaps["inferno"].with_extremes(bad="lightgray")
        axes = figure.subplots(1, len(arrays), squeeze=False)[0]
        # Values in [0, 1] share one colorbar, placed after the panels that use
        # it; others, such as flow, get their own.
        shared_image: AxesImage | None = None
        shared_axes: list[Axes] = []
        for ax, (name, array) in zip(axes, arrays.items(), strict=True):
            values = (
                array.detach().numpy() if isinstance(array, torch.Tensor) else array
            )
            vmax = max(1.0, values.max())
            image = ax.imshow(
                np.ma.masked_equal(values, 0), cmap=colormap, vmin=0, vmax=vmax
            )
            if vmax > 1:
                figure.colorbar(image, ax=ax, shrink=0.8)
            else:
                shared_image = image
                shared_axes.append(ax)
            ax.set_title(f"{name}\nmin {values.min():.3g}, max {values.max():.3g}")
            ax.set_xticks([])
            ax.set_yticks([])
        if shared_image is not None:
            figure.colorbar(shared_image, ax=shared_axes, shrink=0.8)
        figure.savefig(path)

    return write
