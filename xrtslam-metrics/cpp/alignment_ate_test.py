#!/usr/bin/env python

import alignment as al
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
import mplcursors

SCALAR = np.float32

previous_lines = []


def enable_mouse_hover_action(ax):
    # Show vertical line over mouse hover
    cursor = mplcursors.cursor(ax, hover=True)

    def draw_vertical_line(sel):
        global previous_lines
        # Remove previous lines
        for line in previous_lines:
            line.remove()
        previous_lines = []

        # Draw new vertical lines
        for a in ax:
            line = a.axvline(x=sel.target[0], color="gray", linestyle="--", alpha=0.5)
            previous_lines.append(line)
        plt.draw()

    cursor.connect("add", draw_vertical_line)


def main():
    t = time.time()

    parser = argparse.ArgumentParser(
        description="ATE evolution", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("ref", type=Path, help="Reference trajectory CSV file")
    parser.add_argument("est", type=Path, help="Estimated trajectory CSV file")
    args = parser.parse_args()
    ref_csv = args.ref
    est_csv = args.est

    with open(est_csv, "r", encoding="utf-8") as f:
        est_lines = f.readlines()
    with open(ref_csv, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()

    est_lines = [line.strip().split(",") for line in est_lines]
    ref_lines = [line.strip().split(",") for line in ref_lines]

    if est_lines[0][0].startswith("#"):
        est_lines = est_lines[1:]
    if ref_lines[0][0].startswith("#"):
        ref_lines = ref_lines[1:]

    est_ts = np.array([int(line[0]) for line in est_lines], dtype=np.int64).reshape((-1, 1))
    ref_ts = np.array([int(line[0]) for line in ref_lines], dtype=np.int64).reshape((-1, 1))
    est_xyz = np.array([(float(line[1]), float(line[2]), float(line[3])) for line in est_lines], dtype=SCALAR).T
    ref_xyz = np.array([(float(line[1]), float(line[2]), float(line[3])) for line in ref_lines], dtype=SCALAR).T

    # Associate, align, and compute, all at once
    # joint_rmse = al.compute_ate_and_align_ref(est_ts, ref_ts, est_xyz, ref_xyz)
    # print(f"{joint_rmse=}")

    # Associate, align, and compute, in separate steps
    pose_count = al.associate(est_ts, ref_ts, est_xyz, ref_xyz)

    ref_ts = ref_ts[:pose_count]
    est_ts = est_ts[:pose_count]
    ref_xyz = ref_xyz[:, :pose_count]
    est_xyz = est_xyz[:, :pose_count]
    T_ref_est = al.align_ref(est_xyz, ref_xyz, 0, pose_count)

    split_rmse = al.compute_ate(est_xyz, ref_xyz, 0, pose_count, T_ref_est)
    print(f"{split_rmse=}")

    divisions = min(pose_count, 10000)
    assert divisions <= pose_count, f"{divisions=} {pose_count=}"
    ts = (est_ts[:, 0] - est_ts[0, 0]) / 1e9  # conver to seconds
    ys = [0]
    for i, t in enumerate(ts[:-1]):
        T_ref_est = al.align_ref(est_xyz, ref_xyz, 0, i + 1)
        err = al.compute_ate(est_xyz, ref_xyz, 0, i + 1, T_ref_est)
        ys += [err]
    ys = np.array(ys)
    dys = np.diff(ys)

    fig, ax = plt.subplots(2, 1)
    # marker = "-" if divisions > 300 else "-." if divisions > 100 else "o-"
    # marker = ".-"
    marker = "-"

    UNIT_ATE = "cm"
    UNIT_D_ATE = "mm"
    UNITS = {"mm": 1000, "cm": 100, "m": 1}
    mult_ate = UNITS[UNIT_ATE]
    mult_d_ate = UNITS[UNIT_D_ATE]

    ax[0].plot(ts, ys * mult_ate, marker, label=f"ATE [{UNIT_ATE}]")
    ax[1].plot(ts[1:], dys * mult_d_ate, marker, label=f"dATE [{UNIT_D_ATE}]", alpha=0.7)
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")
    ax[0].set_xticks([])
    plt.subplots_adjust(hspace=0)

    enable_mouse_hover_action(ax)

    plt.show()


if __name__ == "__main__":
    main()
