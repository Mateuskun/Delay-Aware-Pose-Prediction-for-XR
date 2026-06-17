#!/usr/bin/env python

import sys

from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable
import json
import yaml
from collections import OrderedDict

from scipy.spatial.transform import Rotation as R
import numpy as np
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_bslt2okvis2_calib = Command("bslt2okvis2_calib", "Convert a Basalt calibration file into OKVIS2 calibration file", bslt2okvis2_calib)
    cmd_traj_okvis2euroc = Command("traj_okvis2euroc", "Convert a OKVIS2 trajectory file into EuRoC format", traj_okvis2euroc)
    # fmt: on

    parser = ArgumentParser(
        description="Helper commands to convert data between okvis2 and EuRoC formats",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_bslt2okvis2_calib.name, help=cmd_bslt2okvis2_calib.desc)
    subparser.set_defaults(func=cmd_bslt2okvis2_calib.func)
    subparser.add_argument("calib_path", type=Path, help="Path to the Basalt calibration file")

    subparser = subparsers.add_parser(cmd_traj_okvis2euroc.name, help=cmd_traj_okvis2euroc.desc)
    subparser.set_defaults(func=cmd_traj_okvis2euroc.func)
    subparser.add_argument("trajectory_csv", type=Path, help="Path to the okvis2 trajectory csv file")
    subparser.add_argument("output_csv", type=Path, help="Where to output the euroc csv")

    return parser.parse_args()


def bslt2okvis2_calib(args: Namespace):
    "Convert a Basalt calibration file into an OKVIS2-like calibration file"

    calib_path = args.calib_path
    with open(calib_path) as f:
        j = json.load(f)
        j = j["value0"]

    okvis2_calib = {"cameras": []}
    cam_count = len(j["resolution"])

    for i in range(cam_count):
        T_imu_cam = j["T_imu_cam"][i]
        intrinsics = j["intrinsics"][i]
        resolution = j["resolution"][i]

        px = T_imu_cam["px"]
        py = T_imu_cam["py"]
        pz = T_imu_cam["pz"]
        qx = T_imu_cam["qx"]
        qy = T_imu_cam["qy"]
        qz = T_imu_cam["qz"]
        qw = T_imu_cam["qw"]

        r = R.from_quat([qx, qy, qz, qw])
        t = np.array([px, py, pz])
        mat = np.eye(4)
        mat[0:3, 0:3] = r.as_matrix()
        mat[0:3, 3] = t

        T_SC = mat.flatten().tolist()
        image_dimension = resolution
        k = intrinsics["intrinsics"]
        fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
        if intrinsics["camera_type"] == "pinhole-radtan8":
            k1, k2, p1, p2 = k["k1"], k["k2"], k["p1"], k["p2"]
            k3, k4, k5, k6 = k["k3"], k["k4"], k["k5"], k["k6"]
            distortion_coefficients = [k1, k2, p1, p2, k3, k4, k5, k6]
            distortion_type = "radialtangential8"
        elif intrinsics["camera_type"] == "kb4":
            k1, k2, k3, k4 = k["k1"], k["k2"], k["k3"], k["k4"]
            distortion_coefficients = [k1, k2, k3, k4]
            distortion_type = "equidistant"
        else:
            distortion_coefficients = None
            distortion_type = None
            raise ValueError(f"Unknown camera type: {intrinsics['camera_type']}")

        focal_length = [fx, fy]
        principal_point = [cx, cy]

        okvis2_calib["cameras"].append(
            {
                "T_SC": T_SC,
                "image_dimension": image_dimension,
                "distortion_coefficients": distortion_coefficients,
                "distortion_type": distortion_type,
                "focal_length": focal_length,
                "principal_point": principal_point,
                "camera_type": "gray",
                "slam_use": "okvis",
            }
        )

    print(yaml.dump(okvis2_calib, default_flow_style=None))


def traj_okvis2euroc(args: Namespace):
    "Convert an okvis2 trajectory file into an euroc trajectory file"
    trajectory_csv = args.trajectory_csv
    output_csv = args.output_csv
    with open(trajectory_csv, "r") as infile, open(output_csv, "w") as outfile:
        lines = infile.readlines()
        assert lines[0].startswith("timestamp"), lines[0]
        lines = lines[1:]
        for line in lines:
            values = line.split(",")
            values = [v.strip() for v in values]
            ts, px, py, pz, qx, qy, qz, qw = values[:8]
            processed_line = f"{ts},{px},{py},{pz},{qw},{qx},{qy},{qz}"
            outfile.write(processed_line + "\n")


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
