#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    COLORS,
    DARK_COLORS,
    DEFAULT_TIMING_COLS,
    NUMBER_OF_NS_IN,
    TIME_UNITS,
    DEFAULT_TIME_UNITS,
    load_csv_safer,
    is_int,
)


class TimingStats:
    def __init__(
        self,
        csv_fn: Optional[Path] = None,
        column_names: Optional[List[str]] = None,
        timing_data: Optional[np.ndarray] = None,
        cols: Optional[Tuple[str, str]] = None,
        units: str = DEFAULT_TIME_UNITS,
        skip_rows: int = 0,
        skip_last_rows: int = 0,
    ):
        self.units = units
        self.csv_fn = csv_fn
        if column_names and timing_data:
            self.column_names = column_names
            self.timing_data = timing_data
        elif csv_fn:
            self.column_names, self.timing_data = load_csv_safer(csv_fn)
        else:
            raise Exception(f"Invalid parameters for {TimingStats.__name__}")

        # silence mypy
        assert self.column_names is not None and self.timing_data is not None

        if skip_rows > 0:
            self.timing_data = self.timing_data[skip_rows:]

        if skip_last_rows > 0:
            self.timing_data = self.timing_data[:-skip_last_rows]

        assert len(self.column_names) == self.timing_data.shape[1], "column names differ from data columns"

        if cols:
            self.set_cols(*cols)

    def set_cols(self, first: str, last: str):
        if is_int(first):
            i = int(first)
            first = self.column_names[min(i, len(self.column_names) - 1)]
        if is_int(last):
            i = int(last)
            last = self.column_names[min(i, len(self.column_names) - 1)]

        assert (
            first in self.column_names and last in self.column_names
        ), f"columns '{first}' or '{last}' not in {self.column_names=}"
        self.first_column = first
        self.last_column = last
        self.i = self.column_names.index(first)
        self.j = self.column_names.index(last)

        self.diffs = (self.timing_data[:, self.j] - self.timing_data[:, self.i]) / NUMBER_OF_NS_IN[self.units]

    @property
    def mean(self):
        return np.mean(self.diffs)

    @property
    def std(self):
        return np.std(self.diffs)

    @property
    def min(self):
        return np.min(self.diffs)

    @property
    def q1(self):
        return np.quantile(self.diffs, 0.25)

    @property
    def q2(self):
        return np.median(self.diffs)

    @property
    def q3(self):
        return np.quantile(self.diffs, 0.75)

    @property
    def max(self):
        return np.max(self.diffs)

    def __str__(self) -> str:
        return f"[{self.csv_fn}]\nTimingStats(mean={self.mean}, std={self.std}, min={self.min}, q1={self.q1}, q2={self.q2}, q3={self.q3}, max={self.max}) from '{self.first_column}' to '{self.last_column}'"

    def plot(self, save_path: Optional[Path] = None, ylim: Optional[float] = None, title: Optional[str] = None) -> None:
        td = self.timing_data

        framepose_tss = (td[:, 0] - td[0, 0]) / NUMBER_OF_NS_IN["s"]

        td1 = np.concatenate((td[:, 0].reshape(td.shape[0], 1), td[:, :-1]), axis=1)
        td2 = td - td1
        assert not td2[:, 0].any(), "first column should be zeroed out by this point"
        # assert td2[123, 7] == td[123, 7] - td[123, 6] # random check that should pass
        td3 = td2 / NUMBER_OF_NS_IN[self.units]

        deltas_by_column = [td3[:, i] for i, cn in enumerate(self.column_names)]

        means = [np.mean(deltas_by_column[j]) for j in range(self.i, self.j + 1)]
        cumsum_means = np.cumsum(means)
        labels = [
            f"{self.i + i}. [{(means[i] / cumsum_means[-1]) * 100:.1f}% {means[i]:.2f} {self.units}] {cn}"
            for i, cn in enumerate(self.column_names[self.i : self.j + 1])
        ]

        # Number of legend columns: scale with number of labels
        num_labels = len(labels) + 1  # +1 for mean line
        ncols = 4 if num_labels <= 16 else 5 if num_labels <= 25 else 6

        dpi = 150
        fig_w = max(2560, ncols * 512) / dpi
        fig_h = 1200 / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.tight_layout(pad=3.0)

        # Plot stacked plot
        ax.stackplot(framepose_tss, deltas_by_column[self.i : self.j + 1], labels=labels, colors=COLORS, alpha=1)

        # Plot each diff mean
        for i, cmean in enumerate(cumsum_means):
            ax.axhline(cmean, color=DARK_COLORS[i % len(DARK_COLORS)], linewidth=0.5, alpha=0.25)

        # Plot total mean
        mean = cumsum_means[-1]
        ax.axhline(mean, color="k", linestyle="--", label=f"Mean: {mean:.2f} {self.units}", linewidth=1, alpha=0.7)

        fig.suptitle(title if title else "Stacked Timing")
        ax.set_title(f"{self.csv_fn}", fontsize="8", pad=-1)
        ax.legend(ncols=ncols, loc="upper right", mode="expand", fontsize="6", framealpha=0)
        ax.set_xlabel("Dataset time (s)")
        ax.set_ylabel("Processing duration (ms)")

        # Auto y-limit: use 95th percentile of stacked totals with headroom, or manual override
        if ylim is not None:
            ax.set_ylim(0, ylim)
        else:
            stacked_totals = sum(deltas_by_column[self.i : self.j + 1])
            p95 = np.percentile(stacked_totals, 95)
            auto_ylim = max(30, np.ceil(p95 * 1.2 / 5) * 5)  # Round up to nearest 5, at least 30
            ax.set_ylim(0, auto_ylim)

        ax.set_xlim(0 - 1, framepose_tss[-1] + 1)

        if save_path is not None:
            fig.savefig(save_path)
            if save_path.suffix.lower() == ".png":  # Compress PNG
                im = Image.open(save_path)
                im = im.convert("P", palette=Image.ADAPTIVE, colors=256)
                im.save(save_path, optimize=True)
        else:
            plt.show()


def parse_args():
    parser = ArgumentParser(
        description="Evaluate timing data for Monado visual-inertial tracking",
    )
    parser.add_argument(
        "timing_csvs",
        type=Path,
        nargs="+",
        help="Timing file generated from Monado",
    )
    parser.add_argument(
        "-fc",
        "--first_column",
        type=str,
        default=DEFAULT_TIMING_COLS[0],  # First columns usually have frame timestamps in a different clock
        help="Column name of timing_csvs to use as first timestamp (default: frames_received)",
    )
    parser.add_argument(
        "-lc",
        "--last_column",
        type=str,
        default=DEFAULT_TIMING_COLS[1],
        help="Column name of timing_csvs to use as last timestamp (default: pose_produced)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="Whether to plot a stacked timing graph",
        action="store_true",
    )
    parser.add_argument(
        "--save_plot", type=Path, default=None, help="Do not show plot but save it instead with this filename"
    )
    parser.add_argument(
        "--units",
        type=str,
        help="Time units to show things on",
        default=DEFAULT_TIME_UNITS,
        choices=TIME_UNITS,
    )
    parser.add_argument(
        "--ylim",
        type=float,
        default=None,
        help="Manual y-axis upper limit in time units (default: auto-scale to 95th percentile)",
    )
    parser.add_argument(
        "--skip_rows",
        type=int,
        default=0,
        help="Number of initial data rows to skip (useful for pipeline warmup)",
    )
    parser.add_argument(
        "--skip_last_rows",
        type=int,
        default=0,
        help="Number of trailing data rows to skip (useful for end-of-dataset garbage)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title (default: 'Stacked Timing')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_files = args.timing_csvs
    first_column = args.first_column
    last_column = args.last_column
    plot = args.plot
    save_plot = args.save_plot
    units = args.units

    ylim = args.ylim
    skip_rows = args.skip_rows
    skip_last_rows = args.skip_last_rows
    title = args.title

    for csv_file in csv_files:
        s = TimingStats(csv_fn=csv_file, cols=(first_column, last_column), units=units, skip_rows=skip_rows, skip_last_rows=skip_last_rows)
        print(s)

        if plot:
            s.plot(save_plot, ylim=ylim, title=title)


if __name__ == "__main__":
    main()
