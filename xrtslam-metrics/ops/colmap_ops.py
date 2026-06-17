#!/usr/bin/env python

import sys


from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable, Optional
import json

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import os

from euroc_ops import SensorPaths

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_gen_cameras = Command("gen_cameras", "Generate cameras.txt from calibration json", gen_cameras)
    cmd_gen_images = Command("gen_images", "Generate images.txt from EuRoC dataset", gen_images)
    # fmt: on

    parser = ArgumentParser(
        description="Helper commands to convert data between colmap and EuRoC formats: See https://colmap.github.io/format.html",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_gen_cameras.name, help=cmd_gen_cameras.desc)
    subparser.set_defaults(func=cmd_gen_cameras.func)
    subparser.add_argument("calibration_path", type=Path, help="Path to the calib json")
    subparser.add_argument("--cam", type=int, default=0, help="Camera index")
    subparser.add_argument("--output_txt", type=Path, default="cameras.txt")

    subparser = subparsers.add_parser(cmd_gen_images.name, help=cmd_gen_images.desc)
    subparser.set_defaults(func=cmd_gen_images.func)
    subparser.add_argument("dataset_path", type=Path, help="Path to the EuRoC dataset")
    subparser.add_argument("calibration_path", type=Path, help="Path to the calib json")
    subparser.add_argument("keyframes_path", type=Path, help="Path to keyframe list")
    subparser.add_argument("--cam", type=int, default=0, help="Camera index")
    subparser.add_argument("--output_txt", type=Path, default="images.txt")

    return parser.parse_args()


def gen_cameras(args: Namespace):
    "Generate cameras.txt from calibration json"
    calib_path = args.calibration_path
    output_txt = args.output_txt

    with open(calib_path) as f:
        j = json.load(f)
        j = j["value0"]

    output_txt = open(output_txt, "w")
    output_txt.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

    i = args.cam
    cam_id = i
    T_imu_cam = j["T_imu_cam"][i]
    intrinsics = j["intrinsics"][i]
    resolution = j["resolution"][i]
    width, height = resolution

    k = intrinsics["intrinsics"]
    fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]

    model_params = [fx, fy, cx, cy]
    if intrinsics["camera_type"] == "pinhole-radtan8":
        k1, k2, p1, p2 = k["k1"], k["k2"], k["p1"], k["p2"]
        k3, k4, k5, k6 = k["k3"], k["k4"], k["k5"], k["k6"]
        model = "FULL_OPENCV"
        model_params += [k1, k2, p1, p2, k3, k4, k5, k6]
    elif intrinsics["camera_type"] == "kb4":
        model = "OPENCV_FISHEYE"
        model_params += [k["k1"], k["k2"], k["k3"], k["k4"]]
    else:
        raise ValueError(f"Unknown camera type: {intrinsics['camera_type']}")
    l = f"{cam_id} {model} {width} {height} {' '.join(map(str, model_params))}\n"
    output_txt.write(l)

    output_txt.close()


# Type aliases
# TODO@mateosss: maybe move these to utils.py
SCALAR = np.float32
Timestamps = NDArray[np.int64]
Vector3 = NDArray[SCALAR]  # xyz
Quaternion = NDArray[SCALAR]  # xyzw
Positions = NDArray[Vector3]
Quaternions = NDArray[Quaternion]


@dataclass
class Trajectory:
    ts: Timestamps
    xyz: Positions
    quat: Optional[Quaternions]

    def copy(self):
        return Trajectory(self.ts.copy(), self.xyz.copy(), self.quat.copy())


def euroc_csv_to_trajectory(csv_path: Path) -> Trajectory:
    # TODO@mateosss: Unify with function in alignment.py

    # Check if file has header, every other comment in the csv will be an error
    has_title = open(csv_path, "r", encoding="utf-8").readline().startswith("#")
    skipfirst = 1 if has_title else 0

    csv_cols = ["ts", "x", "y", "z", "qw", "qx", "qy", "qz"]
    dtypes = {"ts": np.int64} | {f: SCALAR for f in ["x", "y", "z", "qw", "qx", "qy", "qz"]}

    df = pd.read_csv(
        csv_path,
        skiprows=skipfirst,
        names=csv_cols,
        dtype=dtypes,
        index_col=0,
        engine="c",
    )

    ts = df.index.to_numpy().reshape(-1, 1)
    xyz = df[["x", "y", "z"]].to_numpy().T
    quat = df[["qx", "qy", "qz", "qw"]].to_numpy().T

    return Trajectory(ts, xyz, quat)


def gen_images(args: Namespace):
    paths = SensorPaths(args.dataset_path)
    calib_path = args.calibration_path
    keyframes_path = args.keyframes_path
    output_txt = args.output_txt
    cam_id = args.cam

    j = json.load(open(calib_path))
    T_imu_cam_j = j["value0"]["T_imu_cam"][cam_id]
    px, py, pz = T_imu_cam_j["px"], T_imu_cam_j["py"], T_imu_cam_j["pz"]
    qx, qy, qz, qw = (
        T_imu_cam_j["qx"],
        T_imu_cam_j["qy"],
        T_imu_cam_j["qz"],
        T_imu_cam_j["qw"],
    )
    T_imu_cam = np.eye(4)
    T_imu_cam[0:3, 0:3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_imu_cam[0:3, 3] = [px, py, pz]

    output_txt = open(output_txt, "w")
    output_txt.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")

    # Load ground-truth poses
    gt_trajectory = euroc_csv_to_trajectory(paths.gt)

    # Gather keyframe timestamp interpolated poses
    cols = ["ts", "png"]
    dtypes = {"ts": np.int64, "png": str}
    keyframes_ts = pd.read_csv(keyframes_path, comment="#", names=cols, dtype=dtypes)
    after_idxs = np.searchsorted(gt_trajectory.ts[:, 0], keyframes_ts["ts"].to_numpy())  # smaller >=kf
    before_idxs = after_idxs - 1  # biggest <kf

    keyframe_poses = []

    for t, png, i0, i1 in zip(keyframes_ts["ts"], keyframes_ts["png"], before_idxs, after_idxs):
        t0, t1 = gt_trajectory.ts[i0, 0], gt_trajectory.ts[i1, 0]
        # Exact ground-truth for keyframe found, use it
        if t == t1:
            T_w_i1 = np.eye(4)
            T_w_i1[0:3, 0:3] = R.from_quat(gt_trajectory.quat[:, t1]).as_matrix()
            T_w_i1[0:3, 3] = gt_trajectory.xyz[:, t1]
            T_w_c1 = T_w_i1 @ T_imu_cam
            keyframe_poses.append(T_w_c1)
            continue

        # Not found, so interpolate poses
        alpha = (t - t0) / (t1 - t0)
        r0 = gt_trajectory.quat[:, i0]
        r1 = gt_trajectory.quat[:, i1]
        p0 = gt_trajectory.xyz[:, i0]
        p1 = gt_trajectory.xyz[:, i1]
        q = Slerp([0, 1], R.from_quat([r0, r1]))(alpha)
        p = p0 + alpha * (p1 - p0)
        T_w_kfi = np.eye(4)
        T_w_kfi[0:3, 0:3] = q.as_matrix()
        T_w_kfi[0:3, 3] = p
        T_w_kf = T_w_kfi @ T_imu_cam
        keyframe_poses.append(T_w_kf)

    for t, png, T_w_kf in zip(keyframes_ts["ts"], keyframes_ts["png"], keyframe_poses):
        q = R.from_matrix(T_w_kf[0:3, 0:3]).as_quat()
        p = T_w_kf[0:3, 3]
        l = f"{t} {q[3]} {q[0]} {q[1]} {q[2]} {p[0]} {p[1]} {p[2]} {cam_id} {png}\n"
        output_txt.write(l)

    output_txt.close()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
