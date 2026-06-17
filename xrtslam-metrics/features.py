#!/usr/bin/env python

from argparse import ArgumentParser
from itertools import cycle
from pathlib import Path
from cycler import cycler

from matplotlib import pyplot as plt
import numpy as np

from utils import (
    COLORS,
    DARK_COLORS,
    NUMBER_OF_NS_IN,
    load_csv_safer,
    moving_average_smooth,
)


class FeaturesStats:
    def __init__(self, csv_fn: Path):
        ds_name = csv_fn.parent.name
        sys_name = csv_fn.parent.parent.name
        self.name = f"{sys_name}/{ds_name}"
        self.column_names, self.features_data = load_csv_safer(csv_fn)
        self.nof_features = self.features_data[:, 1:3]

    @property
    def mean(self):
        return np.mean(self.nof_features, axis=0)

    @property
    def std(self):
        return np.std(self.nof_features, axis=0)

    @property
    def min(self):
        return np.min(self.nof_features, axis=0)

    @property
    def q1(self):
        return np.quantile(self.nof_features, 0.25, axis=0)

    @property
    def q2(self):
        return np.median(self.nof_features, axis=0)

    @property
    def q3(self):
        return np.quantile(self.nof_features, 0.75, axis=0)

    @property
    def max(self):
        return np.max(self.nof_features, axis=0)

    def __str__(self) -> str:
        return f"FeaturesStats(mean={self.mean}, std={self.std}, min={self.min}, q1={self.q1}, q2={self.q2}, q3={self.q3}, max={self.max})"

    def plot(self, ax, i) -> None:
        fd = self.features_data
        framepose_tss = (fd[:, 0] - fd[0, 0]) / NUMBER_OF_NS_IN["s"]

        # Background stackplot of feature counts
        color_cycler = cycler(color=[DARK_COLORS[i], COLORS[i]])
        ax.set_prop_cycle(color_cycler)
        ax.stackplot(
            framepose_tss,
            fd[:, 1:3].T,
            labels=[f"{self.name} cam0", f"{self.name} cam1"],
            alpha=0.4,
        )

        # Moving average lines of those stackplots
        fc_sum = np.zeros_like(fd[:, 1])
        for j, color in zip(range(1, fd.shape[1]), cycle(color_cycler)):
            fc_sum += fd[:, j]
            plt.plot(
                framepose_tss,
                moving_average_smooth(fc_sum),
                label=f"{self.name} cam{j - 1} smoothed",
                color=color["color"],
            )

        ax.legend(loc="upper right")
        ax.set_title("Tracked feature count")
        ax.set_xlabel("Dataset time (s)")
        ax.set_ylabel("Feature count (#)")


def parse_args():
    parser = ArgumentParser(
        description="Measure visual features metrics for Monado visual-inertial tracking",
    )
    parser.add_argument(
        "features_csvs",
        type=Path,
        nargs="+",
        help="Features file generated from Monado",
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="Whether to plot a stacked feature count graph",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features_csvs = args.features_csvs
    plot = args.plot

    if plot:
        _, ax = plt.subplots()

    for i, feature_csv in enumerate(features_csvs):
        s = FeaturesStats(csv_fn=feature_csv)
        print(s)
        if plot:
            s.plot(ax, i)

    if plot:
        plt.show()


if __name__ == "__main__":
    main()
