#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tabulate import tabulate

from utils import TIME_UNITS, NUMBER_OF_NS_IN, load_csv

DEFAULT_TIME_UNITS = "s"  # Do not use utils.DEFAULT_TIME_UNITS


@dataclass
class CompletionStats:
    first_tracked_ts: int
    last_tracked_ts: int
    first_gt_ts: int
    last_gt_ts: int
    nof_tracked_poses: int
    nof_gt_poses: int
    units: str = DEFAULT_TIME_UNITS

    @property
    def tracking_duration(self):
        return self.last_tracked_ts - self.first_tracked_ts

    @property
    def gt_duration(self):
        return self.last_gt_ts - self.first_gt_ts

    @property
    def tracking_completion(self):
        return self.tracking_duration / self.gt_duration

    @property
    def tracking_success(self):
        # NOTE: When measuring success nof_gt_poses should be the number of frames in dataset
        return self.nof_tracked_poses / self.nof_gt_poses

    def __str__(self):
        units = self.units
        table = [
            ["Groundtruth poses", f"{self.nof_gt_poses}"],
            ["Estimated poses", f"{self.nof_tracked_poses}"],
            ["Tracking success", f"{self.tracking_success * 100:.2f}%"],
            ["Tracking duration", f"{self.tracking_duration:.2f}{units}"],
            ["Groundtruth duration", f"{self.gt_duration:.2f}{units}"],
            ["Tracking completion", f"{self.tracking_completion * 100:.2f}%"],
        ]
        return tabulate(table, tablefmt="pipe")


def parse_args():
    parser = ArgumentParser(description="Determine information about tracking completion based on groundtruth duration")
    parser.add_argument(
        "tracking_csv",
        type=Path,
        help="File generated from Monado (either tracking.csv or timing.csv)",
    )
    parser.add_argument(
        "cam_timestamps_csv",
        type=Path,
        help="Dataset cam0.csv (groundtruth file also works but is less precise)",
    )
    parser.add_argument(
        "--units",
        type=str,
        help="Time units to show things on",
        default=DEFAULT_TIME_UNITS,
        choices=TIME_UNITS,
    )
    return parser.parse_args()


def get_completion_stats(tdata: np.ndarray, gdata: np.ndarray, units: str = DEFAULT_TIME_UNITS) -> CompletionStats:
    ttss = tdata[:, 0]
    gtss = gdata[:, 0]

    div = NUMBER_OF_NS_IN[units]
    t0, t1, g0, g1 = np.array([ttss[0], ttss[-1], gtss[0], gtss[-1]]) / div

    stats = CompletionStats(t0, t1, g0, g1, ttss.size, gtss.size, units)
    return stats


def load_completion_stats(tracking_csv: Path, gt_csv: Path, units=DEFAULT_TIME_UNITS) -> CompletionStats:
    tdata = load_csv(tracking_csv)
    gdata = load_csv(gt_csv)
    return get_completion_stats(tdata, gdata, units)


def main():
    args = parse_args()
    tracking_csv = args.tracking_csv
    cam_timestamps_csv = args.cam_timestamps_csv
    units = args.units

    stats = load_completion_stats(tracking_csv, cam_timestamps_csv, units)
    print(str(stats))


if __name__ == "__main__":
    main()
