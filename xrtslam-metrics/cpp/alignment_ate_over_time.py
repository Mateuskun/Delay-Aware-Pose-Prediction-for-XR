#!/usr/bin/env python

import alignment as al
import numpy as np
import matplotlib.pyplot as plt
import time
import mplcursors
import argparse
from pathlib import Path

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
        description="ATE evolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trim",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        help="Trim 0.0 to 1.0 of the reference",
    )
    parser.add_argument("ref", type=Path, help="Reference trajectory CSV file")
    parser.add_argument("ests", type=Path, nargs="+", help="Estimated trajectory CSV file")
    parser.add_argument("--names", type=str, nargs="*", help="Names of each estimate")
    parser.add_argument("--title", type=str, default="PLOT", help="Title of this plot")
    parser.add_argument("--save", type=Path, default=None, help="Save plot as ")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot")

    args = parser.parse_args()
    trim_start, trim_end = args.trim
    ref_csv = args.ref
    est_csvs = args.ests
    names = args.names
    title = args.title
    save_file = args.save
    plot = not args.no_plot

    assert len(names) == len(est_csvs)

    dpi = 200
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(2048 / dpi, 1024 / dpi), dpi=dpi)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0, bottom=0.05, top=0.99)
    title = f"{title} ATE"
    fig.suptitle(title)
    ax[-1].set_xlabel("Time [s]")
    ax[0].set_ylabel("ATE [cm]")
    ax[1].set_ylabel("dATE [mm]")

    for est_csv, name in zip(est_csvs, names):
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

        # Trim
        start = int(trim_start * len(ref_lines))
        end = int(trim_end * len(ref_lines))
        ref_lines = ref_lines[start:end]

        est_ts = np.array([int(line[0]) for line in est_lines], dtype=np.int64).reshape((-1, 1))
        ref_ts = np.array([int(line[0]) for line in ref_lines], dtype=np.int64).reshape((-1, 1))
        est_xyz = np.array(
            [(float(line[1]), float(line[2]), float(line[3])) for line in est_lines],
            dtype=SCALAR,
        ).T
        ref_xyz = np.array(
            [(float(line[1]), float(line[2]), float(line[3])) for line in ref_lines],
            dtype=SCALAR,
        ).T

        # Associate, align, and compute, all at once
        # joint_rmse = al.compute_ate_and_align_ref(est_ts, ref_ts, est_xyz, ref_xyz)
        # print(f"{joint_rmse=}")

        # Associate, align, and compute, in separate steps
        pose_count = al.associate(est_ts, ref_ts, est_xyz, ref_xyz)
        if pose_count == 0:
            print(f"Skipping {name}")
            continue

        # Trim numpy arrays
        ref_ts = ref_ts[:pose_count]
        est_ts = est_ts[:pose_count]
        ref_xyz = ref_xyz[:, :pose_count]
        est_xyz = est_xyz[:, :pose_count]
        T_ref_est = al.align_ref(est_xyz, ref_xyz, 0, pose_count)
        print(T_ref_est)
        split_rmse = al.compute_ate(est_xyz, ref_xyz, 0, pose_count, T_ref_est)
        print(f"{split_rmse=}")

        divisions = min(pose_count, 10000)
        assert divisions <= pose_count, f"{divisions=} {pose_count=}"
        # xs = np.linspace(start, pose_count, divisions, endpoint=True).astype(int)
        ts = (est_ts[:, 0] - est_ts[0, 0]) / 1e9  # conver to seconds
        ys = [0]
        for i, t in enumerate(ts[:-1]):
            T_ref_est = al.align_ref(est_xyz, ref_xyz, 0, i + 1)
            err = al.compute_ate(est_xyz, ref_xyz, 0, i + 1, T_ref_est)
            ys += [err]
        ys = np.array(ys)
        dys = np.diff(ys)

        # Apply moving average to dys
        # window = 100
        # dys = np.convolve(dys, np.ones(window) / window, mode="same")

        # Filter first points
        # ts = ts[1000:]
        # dys = dys[1000:]

        # Normalize
        # dys /= dys.max()
        # ys /= ys.max()

        # marker = "-" if divisions > 300 else "-." if divisions > 100 else "o-"
        # marker = ".-"
        # marker = "-"
        marker = "o-"

        UNIT_ATE = "cm"
        UNIT_D_ATE = "mm"
        UNITS = {"mm": 1000, "cm": 100, "m": 1}
        mult_ate = UNITS[UNIT_ATE]
        mult_d_ate = UNITS[UNIT_D_ATE]

        # ax.plot(ts, ys, marker, label=f"ATE [m] {name}")
        # prev_color = ax.get_lines()[-1].get_color() # #1f77b4
        # prev_color_darker = "#" + "".join([hex(int(c, 16) // 2)[2:] for c in prev_color[1:]])
        # ax.plot(xs[1:], dys, marker, label=f"dATE [cm] {name}", color=prev_color_darker)

        # ax.plot(ts[1:], dys, marker, label=f"dATE [cm] {name}", alpha=0.7)
        #
        ax[0].plot(ts, ys * mult_ate, "-", label=f"ATE [{UNIT_ATE}] {name}")
        ax[1].plot(
            ts[1:],
            dys * mult_d_ate,
            marker,
            markersize=0.5,
            label=f"dATE [{UNIT_D_ATE}] {name}",
            alpha=0.33,
        )

    for a in ax:
        a.legend()
    if plot:
        enable_mouse_hover_action(ax)
        plt.show()
    if save_file:
        fig.savefig(save_file, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
