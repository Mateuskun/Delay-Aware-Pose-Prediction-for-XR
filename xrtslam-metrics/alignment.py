#!/usr/bin/env python

from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
from math import sqrt

import numpy as np
from numpy.linalg import norm
import numpy.typing as npt
import pandas as pd
from scipy.spatial.transform import Rotation as R

import cpp.alignment as al

# Global configs
# TODO@mateosss: move these to utils.py
SCALAR = np.float32
RTE_DELTA = 6

# Type aliases
# TODO@mateosss: maybe move these to utils.py
Timestamps = npt.NDArray[np.int64]
Vector3 = npt.NDArray[SCALAR]  # xyz
Quaternion = npt.NDArray[SCALAR]  # xyzw
Positions = npt.NDArray[Vector3]
Quaternions = npt.NDArray[Quaternion]


@dataclass
class Trajectory:
    ts: Timestamps
    xyz: Positions
    quat: Optional[Quaternions]

    def copy(self):
        # We want col-major (fortran-order) for eigen interoperability
        return Trajectory(self.ts.copy(), self.xyz.copy(order="F"), self.quat.copy(order="F"))


# def euroc_csv_to_trajectory(csv_path: Path) -> Trajectory:
#     # TODO@mateosss: add pyarrow to poetry

#     # Check if file has header, every other comment in the csv will be an error
#     has_title = open(csv_path, "r", encoding="utf-8").readline().startswith("#")
#     skipfirst = 1 if has_title else 0

#     csv_cols = ["ts", "x", "y", "z", "qw", "qx", "qy", "qz"]
#     dtypes = {"ts": np.int64} | {
#         f: SCALAR for f in ["x", "y", "z", "qw", "qx", "qy", "qz"]
#     }

#     df = pd.read_csv(
#         csv_path,
#         skiprows=skipfirst,
#         names=csv_cols,
#         dtype=dtypes,
#         index_col=0,
#         engine="pyarrow",
#     )

#     ts = df.index.to_numpy().reshape(-1, 1)
#     xyz = df[["x", "y", "z"]].to_numpy().T
#     quat = df[["qx", "qy", "qz", "qw"]].to_numpy().T


#     return Trajectory(ts, xyz, quat)
def euroc_csv_to_trajectory(csv_path: Path) -> Trajectory:
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [line.strip().split(",") for line in lines]

    if lines[0][0].startswith("#"):
        lines = lines[1:]

    # NOTE: transposing makes them have col-major layout, which is what we want
    ts = np.array([int(l[0]) for l in lines], dtype=np.int64).reshape((-1, 1))
    xyz = np.array([(float(l[1]), float(l[2]), float(l[3])) for l in lines], dtype=SCALAR).T
    quat = np.array([(float(l[5]), float(l[6]), float(l[7]), float(l[4])) for l in lines], dtype=SCALAR).T

    return Trajectory(ts, xyz, quat)


def associate(est: Trajectory, ref: Trajectory, use_quats=False) -> None:
    if use_quats:
        pose_count = al.associate_full(est.ts, ref.ts, est.xyz, ref.xyz, est.quat, ref.quat)
    else:
        pose_count = al.associate(est.ts, ref.ts, est.xyz, ref.xyz)

    # Native associate requires resizeing array last elements afterwards
    # TODO@mateosss: use ndarray.resize() to resize in place
    est.ts = est.ts[:pose_count]
    est.xyz = est.xyz[:, :pose_count]
    if est.quat is not None:
        est.quat = est.quat[:, :pose_count]

    ref.ts = ref.ts[:pose_count]
    ref.xyz = ref.xyz[:, :pose_count]
    if ref.quat is not None:
        ref.quat = ref.quat[:, :pose_count]

    return pose_count


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


def compute_rte(ts, est_xyz, ref_xyz, est_quat, ref_quat, i, j, delta=RTE_DELTA):
    assert j > i

    n = (j - i) // delta
    n += 1 if (j - i) % delta != 0 else 0  # Edge case when j-i is mult. of DELTA

    rmse = 0.0
    for k in range(0, n - 1):
        k_off = i + k
        k0 = k_off * delta
        k1 = (k_off + 1) * delta
        est0 = se3(est_xyz[:, k0], est_quat[:, k0])
        est1 = se3(est_xyz[:, k1], est_quat[:, k1])

        ref0 = se3(ref_xyz[:, k0], ref_quat[:, k0])
        ref1 = se3(ref_xyz[:, k1], ref_quat[:, k1])

        est_delta = se3_inv(est0) @ est1
        ref_delta = se3_inv(ref0) @ ref1

        estref_delta = se3_inv(est_delta) @ ref_delta

        rel_err = norm(estref_delta[:3, 3])
        rmse += rel_err**2

    return sqrt(rmse / (n - 1))


def compute_ates(ref_path: Path, est_paths: List[Path]) -> List[float]:
    REF = euroc_csv_to_trajectory(ref_path)
    ests = [euroc_csv_to_trajectory(est_path) for est_path in est_paths]

    ates = []
    for est in ests:  # TODO: Parallelize forloop for speed
        ref = REF.copy()
        pose_count = associate(est, ref)  # TODO: Use a single call to compute_ate_and_align_ref for speed
        T_ref_est = al.align_ref(est.xyz, ref.xyz, 0, pose_count)
        ate = al.compute_ate(est.xyz, ref.xyz, 0, pose_count, T_ref_est)
        ates.append(ate)

    return ates


def compute_rtes(ref_path: Path, est_paths: List[Path]) -> List[float]:
    REF = euroc_csv_to_trajectory(ref_path)
    ests = [euroc_csv_to_trajectory(est_path) for est_path in est_paths]

    rtes = []
    for est in ests:
        ref = REF.copy()
        pose_count = associate(est, ref, use_quats=True)
        rte = compute_rte(est.ts, est.xyz, ref.xyz, est.quat, ref.quat, 0, pose_count, RTE_DELTA)
        rtes.append(rte)

    return rtes


# if __name__ == "__main__":
#     ref = "/home/mateo/Documents/projects/xrtslam-metrics/test/data/targets/MOO02/gt.csv"
#     est = "/home/mateo/Desktop/delete/snakeslam/msdRT/MOO02/trajectory.causal.csv"
#     print(f"{compute_ates(Path(ref), [Path(est)])=}")
