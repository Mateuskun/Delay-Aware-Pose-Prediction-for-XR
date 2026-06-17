#!/usr/bin/env python

import alignment as al
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
import mplcursors
from scipy.spatial.transform import Rotation as R
from math import sqrt

SCALAR = np.float32
DELTA = 6


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


def compute_rte(ts, est_xyz, ref_xyz, est_quat, ref_quat, i, j):
    assert j > i

    def se3(xyz, quat):
        mat = np.eye(4, dtype=SCALAR)
        mat[:3, :3] = R.from_quat(quat).as_matrix()
        mat[:3, 3] = xyz
        return mat

    def se3_inv(mat):
        inv_mat = np.eye(4, dtype=SCALAR)
        inv_mat[:3, :3] = mat[:3, :3].T
        inv_mat[:3, 3] = -inv_mat[:3, :3] @ mat[:3, 3]
        return inv_mat

    # number of pose pairs + 1 for the initial pose/timestamp
    rel_count = (j - i) // DELTA
    rel_count += 1 if (j - i) % DELTA != 0 else 0  # Edge case when j-i is mult. of DELTA
    timestamps = np.zeros(rel_count, dtype=np.int64)
    residuals = np.zeros(rel_count, dtype=SCALAR)
    timestamps[0] = ts[i, 0]
    residuals[0] = 0

    for k in range(i + DELTA, j, DELTA):
        k0 = k - DELTA
        est0 = se3(est_xyz[:, k0], est_quat[:, k0])
        ref0 = se3(ref_xyz[:, k0], ref_quat[:, k0])

        k1 = k
        est1 = se3(est_xyz[:, k1], est_quat[:, k1])
        ref1 = se3(ref_xyz[:, k1], ref_quat[:, k1])

        est_delta = se3_inv(est0) @ est1
        ref_delta = se3_inv(ref0) @ ref1
        estref_delta = se3_inv(est_delta) @ ref_delta
        rel_err = norm(estref_delta[:3, 3])

        timestamps[k1 // DELTA] = ts[k1, 0]
        residuals[k1 // DELTA] = rel_err

    return timestamps, residuals


def main():
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
    title = f"{title} RTE"
    fig.suptitle(title)
    ax[-1].set_xlabel("Time [s]")
    ax[0].set_ylabel("RTE [cm]")
    ax[1].set_ylabel("dRTE [mm]")

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
        est_quat = np.array(
            [(float(line[5]), float(line[6]), float(line[7]), float(line[4])) for line in est_lines],
            dtype=SCALAR,
        ).T
        ref_quat = np.array(
            [(float(line[5]), float(line[6]), float(line[7]), float(line[4])) for line in ref_lines],
            dtype=SCALAR,
        ).T

        # Associate, align, and compute, all at once
        # joint_rmse = al.compute_ate_and_align_ref(est_ts, ref_ts, est_xyz, ref_xyz)
        # print(f"{joint_rmse=}")

        # Associate, align, and compute, in separate steps
        pose_count = al.associate_full(est_ts, ref_ts, est_xyz, ref_xyz, est_quat, ref_quat)
        if pose_count == 0:
            print(f"Skipping {name}")
            continue

        # TODO@mateosss: document that after associate it requires pruning on the python side
        ref_ts = ref_ts[:pose_count]
        est_ts = est_ts[:pose_count]
        ref_xyz = ref_xyz[:, :pose_count]
        est_xyz = est_xyz[:, :pose_count]
        est_quat = est_quat[:, :pose_count]
        ref_quat = ref_quat[:, :pose_count]

        timestamps, residuals = compute_rte(est_ts, est_xyz, ref_xyz, est_quat, ref_quat, 0, pose_count)

        # TODO@mateosss: document this behavor:
        # timestamps = [ts0, ts1, ts2, ...]
        # residuals = [0, diff0_1, diff1_2, diff2_3, ...]

        ts_s = (timestamps[:] - est_ts[0]) / 1e9  # convert to seconds
        # ts_s = timestamps[:]
        # ts_s = np.arange(0, len(residuals)) * DELTA
        n = len(ts_s)
        ys = np.zeros(n, dtype=SCALAR)
        res2_sum = 0
        for i, _ in enumerate(ts_s[1:], start=1):
            res2_sum += residuals[i] ** 2
            ys[i] = sqrt(res2_sum / i)
            # print(f"{i=}, {residuals[i]**2=:.10f}, {res2_sum=:.10f}, {ys[i]=:.10f}")
        assert ys[0] == 0
        print(f"total rmse={ys[-1]}")

        dys = np.diff(ys)

        # Apply moving average to dys
        # window = 100 // DELTA
        # dys = np.convolve(dys, np.ones(window) / window, mode="same")

        # Filter first points
        # ts_s = ts_s[1000 // DELTA :]
        # ys = ys[1000 // DELTA :]
        # dys = dys[1000 // DELTA :]

        # Normalize
        # dys /= dys.max()
        # ys /= ys.max()

        # marker = "-" if divisions > 300 else "-." if divisions > 100 else "o-"
        marker = "-o"
        # marker = "-"
        # marker = "."

        UNIT_RTE = "cm"
        UNIT_D_RTE = "mm"
        UNITS = {"mm": 1000, "cm": 100, "m": 1}
        mult_rte = UNITS[UNIT_RTE]
        mult_d_rte = UNITS[UNIT_D_RTE]

        ax[0].plot(ts_s, ys * mult_rte, "-", label=f"RTE [{UNIT_RTE}] {name}")
        ax[1].plot(
            ts_s[1:],
            dys * mult_d_rte,
            marker,
            markersize=1,
            label=f"dRTE [{UNIT_D_RTE}] {name}",
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
