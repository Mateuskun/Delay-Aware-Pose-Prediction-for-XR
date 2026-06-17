import math
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import numpy.typing as npt

NUMBER_OF_NS_IN = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}
TIME_UNITS = NUMBER_OF_NS_IN.keys()
DEFAULT_TIME_UNITS = "ms"
COMPLETION_FULL_SINCE = 0.98  # Completion ratio to consider a complete run
DEFAULT_TIMING_COLS = (4, -1)  # We skip first columns since usually they have frame timestamps in a different clock
DEFAULT_SEGMENT_DRIFT_TOLERANCE_M = 0.01
COLORS = [  # Regular cycle
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FFC107",  # amber
    "#E91E63",  # pink
    "#673AB7",  # deeppurple
    "#00BCD4",  # cyan
    "#CDDC39",  # lime
    "#FF5722",  # deeporange
    "#9C27B0",  # purple
    "#03A9F4",  # lightblue
    "#8BC34A",  # lightgreen
    "#FF9800",  # orange
    "#3F51B5",  # indigo
    "#009688",  # teal
    "#FFEB3B",  # yellow
    "#F44336",  # red
    "#795548",  # brown
    "#607D8B",  # bluegrey
]
DARK_COLORS = [  # Regular cycle
    "#1976D2",  # blue
    "#388E3C",  # green
    "#FFA000",  # amber
    "#C2185B",  # pink
    "#512DA8",  # deeppurple
    "#0097A7",  # cyan
    "#AFB42B",  # lime
    "#E64A19",  # deeporange
    "#7B1FA2",  # purple
    "#0288D1",  # lightblue
    "#689F38",  # lightgreen
    "#F57C00",  # orange
    "#303F9F",  # indigo
    "#00796B",  # teal
    "#FBC02D",  # yellow
    "#D32F2F",  # red
    "#5D4037",  # brown
    "#455A64",  # bluegrey
]

make_color_iterator = lambda: cycle(COLORS)
make_dark_color_iterator = lambda: cycle(DARK_COLORS)

# Types: These are mostly just aliases to generic numpy arrays in the hopes that
# one day numpy has proper type support and we can just use it here, also it
# helps for documentation
Indices = np.ndarray
ArrayOfFloats = npt.ArrayLike
ArrayOfPoints = npt.ArrayLike  # either 2D or 3D points
Matrix4x4 = np.ndarray
Matrix3x3 = np.ndarray
Vector2 = np.ndarray
Vector4 = np.ndarray
SE3 = Matrix4x4
SO3 = Matrix3x3
Quaternion = Vector4


def moving_average_smooth(values: np.ndarray, window_size=100):
    n = window_size
    cs = np.cumsum(values)
    moving_avg = (cs[n:] - cs[:-n]) / n  # type: ignore
    padded_w_zeros = np.zeros(values.shape)
    padded_w_zeros[n:] = moving_avg
    return padded_w_zeros


def isnan(obj):
    if isinstance(obj, Iterable):
        for item in obj:
            if not isnan(item):
                return False
        return True
    return math.isnan(obj)


def is_int(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False


def check_monotonic_rows(rows: np.ndarray) -> None:
    last_ts = 0
    for row in rows:
        ts = row[0]
        assert last_ts <= ts, f"Failed assertion {last_ts=} < {ts=}"
        last_ts = ts


def load_csv_unsafe(csv_fn: Path, dtype=np.int64) -> np.ndarray:
    # print(csv_fn)
    data = np.genfromtxt(csv_fn, delimiter=",", comments="#", dtype=dtype, invalid_raise=False)
    return data


def load_csv(csv_fn: Path, dtype=np.int64) -> np.ndarray:
    # print(csv_fn)
    try:
        data = np.genfromtxt(csv_fn, delimiter=",", comments="#", dtype=dtype, invalid_raise=True)
    except Exception as e:
        print(f"{csv_fn=}")
        print(e)
    check_monotonic_rows(data)
    return data


def load_trajectory(csv_fn: Path, dtype=np.int64) -> np.ndarray:
    # TODO: Add unit quaternion check
    return load_csv(csv_fn, dtype)


def load_csv_safer(csv_fn: Path, dtype=np.int64) -> Tuple[List[str], np.ndarray]:
    timing_data = load_csv(csv_fn, dtype)

    with open(csv_fn, "r", encoding="utf8") as f:
        first_line = next(f)
    assert first_line[0] == "#" and first_line[-1] == "\n", "first csv line should be a comment with column names"

    column_names = first_line[1:-1].split(",")
    assert len(column_names) == timing_data.shape[1], "number of column names differ from data columns"

    return column_names, timing_data


# Logging utils


def color_string(string, fg=None):
    ANSI_COLORS = {  # List some colors that may be needed
        None: "\033[31m",  # Red
        "red": "\033[31m",
        "pink": "\033[38;5;206m",
        "green": "\033[32m",
        "orange": "\033[33m",
        "blue": "\033[34m",
        "cyan": "\033[36m",
        "lightred": "\033[91m",
        "lightgreen": "\033[92m",
        "yellow": "\033[93m",
        "lightblue": "\033[94m",
        "lightcyan": "\033[96m",
        "brightwhite": "\u001b[37;1m",
        "brightmagenta": "\u001b[35;1m",
        "strikethrough": "\033[9m",
        "bold": "\033[1m",
        "faint": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
    }
    endcolor = "\033[0m"
    return f"{ANSI_COLORS.get(fg, '')}{string}{endcolor}"


def info(*lines):
    for line in lines:
        print(f"{color_string('[I] ', fg='cyan')}{line}")


def error(*lines, should_exit=True):
    for line in lines:
        print(f"{color_string('[E] ', fg='lightred')}{line}")
    if should_exit:
        exit(1)


def warn(*lines):
    for line in lines:
        print(f"{color_string('[W] ', fg='yellow')}{line}")
