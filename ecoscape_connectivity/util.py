import csv
import os
from collections.abc import Iterable
from typing import Self

import numpy as np

# We collect here a few generally useful functions.


def dict_translate(
    np_array: np.ndarray, my_dict: dict[int, float], default_val: float = 0
) -> np.ndarray:
    """Translates the terrain type according to a dictionary mapping
    terrain type to values.
    """
    unique_codes, codes = np.unique(np_array, return_inverse=True)
    return np.array([(my_dict.get(code, default_val)) for code in unique_codes])[
        codes
    ].reshape(np_array.shape)


def read_transmission_csv(fn: str) -> dict[int, float]:
    """Reads a dictionary of terrain resistance or transmission in csv, producing a dictionary.

    The CSV needs a map_code column with the landcover codes, and either a
    transmission column with the permeability of each code, or a resistance
    column, which is converted to a permeability of 1 - resistance.

    :param fn: file name of the CSV.
    :return: the permeability of each landcover code.
    :raises FileNotFoundError: if there is no file at fn.
    :raises ValueError: if the CSV lacks the columns above, or a value isn't a number.
    """
    if not os.path.isfile(fn):
        raise FileNotFoundError(
            f"Permeability CSV file not found: {os.path.abspath(fn)}. Check the "  # pragma: no mutate
            "path for typos. Relative paths start from the folder Python runs "  # pragma: no mutate
            f"in, {os.getcwd()}."  # pragma: no mutate
        )
    d: dict[int, float] = {}
    # utf-8-sig skips the byte order mark that Excel writes at the start of a
    # CSV; otherwise the first column would be named "\ufeffmap_code".
    with open(fn, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        if "map_code" not in columns or (
            "resistance" not in columns and "transmission" not in columns
        ):
            raise ValueError(
                f"The permeability CSV {fn} needs a map_code column, and a "  # pragma: no mutate
                "transmission or resistance column, but its columns are: "  # pragma: no mutate
                f"{', '.join(columns) or 'none, the file is empty'}. Check the "  # pragma: no mutate
                "column names in the first row."  # pragma: no mutate
            )
        value_column = "resistance" if "resistance" in columns else "transmission"
        for row in reader:
            try:
                code = int(row["map_code"])
                value = float(row[value_column])
            # A missing value is None, for which int() and float() raise TypeError.
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Line {reader.line_num} of the permeability CSV {fn} has "  # pragma: no mutate
                    f"map_code {row['map_code']!r} and {value_column} "  # pragma: no mutate
                    f"{row[value_column]!r}, but map_code must be a whole number "  # pragma: no mutate
                    f"and {value_column} a number."  # pragma: no mutate
                ) from error
            d[code] = 1.0 - value if value_column == "resistance" else value
    return d


def rescale_resistance(
    d: dict[int, float], resolution_m: float, hop_length: float
) -> dict[int, float]:
    """Resistance dictionaries are based on decay over 100m.
    This function rescales the resistance to the value of the
    actual hop length."""
    alpha = resolution_m * hop_length / 100.0
    return {k: v**alpha for k, v in d.items()}


def createdir_for_file(fn: str) -> None:
    """Ensures that the path to a file exists."""
    dirs, _ffn = os.path.split(fn)
    # print("Creating", dirs)
    os.makedirs(dirs, exist_ok=True)


class SingleIterator[T]:
    """Given an iterator, this class builds an iterator that returns
    pairs of the form (None, i), where i is given by the iterator."""

    def __init__(self, it: Iterable[T]) -> None:
        self.it = iter(it)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[None, T]:
        x = next(self.it)
        return (None, x)
