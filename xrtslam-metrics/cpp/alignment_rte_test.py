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

    n = (j - i) // DELTA
    timestamps = np.zeros(n + 1, dtype=np.int64)
    residuals = np.zeros(n + 1, dtype=SCALAR)
    timestamps[0] = ts[i, 0]
    residuals[0] = 0

    for k in range(0, n):
        k_off = i + k
        k0 = k_off * DELTA
        k1 = (k_off + 1) * DELTA
        est0 = se3(est_xyz[:, k0], est_quat[:, k0])
        est1 = se3(est_xyz[:, k1], est_quat[:, k1])

        ref0 = se3(ref_xyz[:, k0], ref_quat[:, k0])
        ref1 = se3(ref_xyz[:, k1], ref_quat[:, k1])

        est_delta = se3_inv(est0) @ est1
        ref_delta = se3_inv(ref0) @ ref1

        estref_delta = se3_inv(est_delta) @ ref_delta

        rel_err = norm(estref_delta[:3, 3])
        timestamps[k + 1] = ts[k1, 0]
        residuals[k + 1] = rel_err

    return timestamps, residuals


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
    print(f"TIME={time.time()-t:.4f}s [START]")
    t = time.time()

    with open(est_csv, "r", encoding="utf-8") as f:
        est_lines = f.readlines()
    with open(ref_csv, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()
    print(f"TIME={time.time()-t:.4f}s [READ]")
    t = time.time()

    est_lines = [line.strip().split(",") for line in est_lines]
    ref_lines = [line.strip().split(",") for line in ref_lines]
    print(f"TIME={time.time()-t:.4f}s [STRIPED]")
    t = time.time()

    if est_lines[0][0].startswith("#"):
        est_lines = est_lines[1:]
    if ref_lines[0][0].startswith("#"):
        ref_lines = ref_lines[1:]
    print(f"TIME={time.time()-t:.4f}s [UNCOMMENTED]")
    t = time.time()

    est_ts = np.array([int(line[0]) for line in est_lines], dtype=np.int64).reshape((-1, 1))
    ref_ts = np.array([int(line[0]) for line in ref_lines], dtype=np.int64).reshape((-1, 1))
    est_xyz = np.array([(float(line[1]), float(line[2]), float(line[3])) for line in est_lines], dtype=SCALAR).T
    ref_xyz = np.array([(float(line[1]), float(line[2]), float(line[3])) for line in ref_lines], dtype=SCALAR).T
    est_quat = np.array(
        [(float(line[5]), float(line[6]), float(line[7]), float(line[4])) for line in est_lines], dtype=SCALAR
    ).T
    ref_quat = np.array(
        [(float(line[5]), float(line[6]), float(line[7]), float(line[4])) for line in ref_lines], dtype=SCALAR
    ).T
    print(f"TIME={time.time()-t:.4f}s [DATACONVERSION]")
    t = time.time()

    # Associate, align, and compute, all at once
    # joint_rmse = al.compute_ate_and_align_ref(est_ts, ref_ts, est_xyz, ref_xyz)
    # print(f"{joint_rmse=}")

    # Associate, align, and compute, in separate steps
    pose_count = al.associate_full(est_ts, ref_ts, est_xyz, ref_xyz, est_quat, ref_quat)
    print(f"TIME={time.time()-t:.4f}s [ASSOCIATION]")
    t = time.time()

    # TODO@mateosss: document that after associate it requires pruning on the python side
    ref_ts = ref_ts[:pose_count]
    est_ts = est_ts[:pose_count]
    ref_xyz = ref_xyz[:, :pose_count]
    est_xyz = est_xyz[:, :pose_count]
    est_quat = est_quat[:, :pose_count]
    ref_quat = ref_quat[:, :pose_count]
    print(f"TIME={time.time()-t:.4f}s [CUT]")
    t = time.time()

    timestamps, residuals = compute_rte(est_ts, est_xyz, ref_xyz, est_quat, ref_quat, 0, pose_count)

    # TODO@mateosss: document this behavor:
    # timestamps = [ts0, ts1, ts2, ...]
    # residuals = [0, diff0_1, diff1_2, diff2_3, ...]

    print(f"TIME={time.time()-t:.4f}s [COMPUTE RESIDUALS]")
    t = time.time()

    # ts_s = (timestamps[:] - est_ts[0]) / 1e9  # convert to seconds
    # ts_s = timestamps[:]
    ts_s = np.arange(0, len(residuals)) * DELTA
    n = len(ts_s)
    ys = np.zeros(n, dtype=SCALAR)
    res2_sum = 0
    for i, _ in enumerate(ts_s):
        res2_sum += residuals[i] ** 2
        ys[i] = sqrt(res2_sum / (i + 1))
        t = time.time()
    assert ys[0] == 0
    print(f"total rmse={ys[-1]}")

    dys = np.diff(ys)
    print(f"TIME={time.time()-t:.4f}s [COMPUTE ARRAYS]")
    t = time.time()

    fig, ax = plt.subplots(2, 1)
    # marker = "-" if divisions > 300 else "-." if divisions > 100 else "o-"
    marker = ".-"
    # marker = "."

    UNIT_RTE = "cm"
    UNIT_D_RTE = "mm"
    UNITS = {"mm": 1000, "cm": 100, "m": 1}
    mult_rte = UNITS[UNIT_RTE]
    mult_d_rte = UNITS[UNIT_D_RTE]

    ax[0].plot(ts_s, ys * mult_rte, marker, label=f"RTE [{UNIT_RTE}]")
    ax[1].plot(ts_s[1:], dys * mult_d_rte, marker, label=f"dRTE [{UNIT_D_RTE}]", alpha=0.7)
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")
    ax[0].set_xticks([])
    plt.subplots_adjust(hspace=0)

    enable_mouse_hover_action(ax)
    print(f"TIME={time.time()-t:.4f}s [PREPARE PLOT]")
    t = time.time()

    plt.show()
    print(f"TIME={time.time()-t:.4f}s [SHOW PLOT]")
    t = time.time()


if __name__ == "__main__":
    main()
