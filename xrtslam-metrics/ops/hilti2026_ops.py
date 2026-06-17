#!/usr/bin/env python

# Copyright 2026, Mattis Krauch
# SPDX-License-Identifier: BSL-1.0

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from contextlib import ExitStack

import numpy as np
import cv2
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Imu

import yaml
import json
from scipy.spatial.transform import Rotation as Rotation

def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_ds = Command("ds", "Convert hilti rosbag to euroc dataset", main_ds)
    cmd_gt = Command("gt", "Convert hilti groundtruth file to euroc groundtruth csv", main_gt)
    cmd_cal = Command("cal", "Convert hilti calibration yaml to basalt calib json", main_cal)
    # fmt: on

    parser = ArgumentParser(
        description="Make EuRoC dataset from Hilti ROS bag.\n\n"
        "Usage example for dataset exp14 (get it from https://hilti-challenge.com/dataset-2022.html):\n"
        "$ ./utils/hilti_rosbag_to_euroc.py ds bags/exp14_basement_2.bag exp14\n"
        "$ ./utils/hilti_rosbag_to_euroc.py gt ~/Downloads/exp14_basement_2_imu.txt exp14/mav0/state_groundtruth_estimate0/data.csv\n"
        "$ ./utils/hilti_rosbag_to_euroc.py cal exp14/calib_cam.yaml exp14/calib_imu.yaml exp14/calib.json"
    )

    subparsers = parser.add_subparsers(
        help="Convert hilti rosbag or groundtruth file or calibration file to euroc format",
        dest="mode",
        required=True,
    )

    parser_ds = subparsers.add_parser(cmd_ds.name, help=cmd_ds.desc)
    parser_ds.set_defaults(func=cmd_ds.func)
    parser_ds.add_argument("bag_path", type=Path)
    parser_ds.add_argument("output_path", type=Path)

    parser_gt = subparsers.add_parser(cmd_gt.name, help=cmd_gt.desc)
    parser_gt.set_defaults(func=cmd_gt.func)
    parser_gt.add_argument("hilti_gt_path", type=Path, help="The hilti groundtruth input file")
    parser_gt.add_argument(
        "output_path",
        type=Path,
        help="The euroc groundtruth file to write",
        default="out.csv",
    )

    parser_cal = subparsers.add_parser(cmd_cal.name, help=cmd_cal.desc)
    parser_cal.set_defaults(func=cmd_cal.func)
    parser_cal.add_argument("hilti_cam_cal_path", type=Path)
    parser_cal.add_argument("hilti_imu_cal_path", type=Path)
    parser_cal.add_argument("output_path", type=Path)
    
    return parser.parse_args()


def main_ds(bag_path: Path, output_path: Path):
    print("Converting Hilti dataset")

    mav0_path = output_path / "mav0"
    mav0_path.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')
    
    reader.open(storage_options, converter_options)

    print("Exporting bag data")
    bridge = CvBridge()
    cam_data_paths = []
    cam_counts = [0, 0]

    with ExitStack() as stack:
        imu_path = mav0_path / "imu0"
        imu_path.mkdir(exist_ok=True)
        imu_csv_file = stack.enter_context(open(imu_path / "data.csv", "w", encoding="utf-8"))
        imu_csv_file.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\r\n"
        )

        cam_csv_files = []
        for i in range(2):
            cam_path = mav0_path / f"cam{i}"
            cam_path.mkdir(exist_ok=True)
            data_path = cam_path / "data"
            data_path.mkdir(exist_ok=True)
            
            csv_f = stack.enter_context(open(cam_path / "data.csv", "w", encoding="utf-8"))
            csv_f.write("#timestamp [ns],filename\r\n")
            
            cam_csv_files.append(csv_f)
            cam_data_paths.append(data_path)

        # Single pass through the bag
        reader.set_filter(rosbag2_py.StorageFilter(topics=["/imu/data_raw", "/cam0/image_raw/compressed", "/cam1/image_raw/compressed"]))

        while reader.has_next():
            topic, data, _ = reader.read_next()
            
            if topic == "/imu/data_raw":
                msg = deserialize_message(data, Imu)
                timestamp = int(f"{msg.header.stamp.sec}{msg.header.stamp.nanosec:09d}")
                w = msg.angular_velocity
                a = msg.linear_acceleration
                imu_csv_file.write(f"{timestamp},{w.x},{w.y},{w.z},{a.x},{a.y},{a.z}\r\n")
                
            elif topic in ["/cam0/image_raw/compressed", "/cam1/image_raw/compressed"]:
                cam_idx = 0 if "cam0" in topic else 1
                msg = deserialize_message(data, CompressedImage)
                cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
                timestamp = int(f"{msg.header.stamp.sec}{msg.header.stamp.nanosec:09d}")
                cv2.imwrite(str(cam_data_paths[cam_idx] / f"{timestamp}.png"), cv_img)
                cam_csv_files[cam_idx].write(f"{timestamp},{timestamp}.png\r\n")
                print(f"Wrote image {cam_counts[cam_idx]}\r", end="")
                cam_counts[cam_idx] += 1

    print("\nDone.")


def main_gt(hilti_gt_path: Path, output_path: Path):
    print("Converting Hilti groundtruth")
    with open(hilti_gt_path, "r") as f:
        contents = f.read()

    first_line = (
        "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\r\n"
    )
    lines = contents.split("\n")
    lines = lines[1:] # Remove header line
    lines = [l for l in lines if l]  # Remove empty lines
    lines = [l.split(" ") for l in lines]  # Make lines into lists
    lines = [l[0:4] + [l[7]] + l[4:7] for l in lines]  # Swap xyzw to wxyz
    lines = [[f"{l[0].split('.')[0]}{l[0].split('.')[1].ljust(9, '0')}"] + l[1:] for l in lines]
    lines = [",".join(l) for l in lines]  # Back to comma-separated strings
    lines = "\r\n".join(lines) + "\r\n"  # Back to a big string with CRLF EOL
    lines = first_line + lines  # Add header

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(lines)
    print("Finished converting Hilti groundtruth")


def main_cal(hilti_cam_cal_path: Path, hilti_imu_cal_path: Path, output_path: Path):
    json_string = ""
    cam_data = []
    intrinsics = []
    resolutions = []

    # Extract camera calibration data
    with open(hilti_cam_cal_path, "r") as file:
        file.readline() # Skip first line
        data = yaml.full_load(file)

        for camera in data.values():
            T_cam_imu = np.array(camera["T_cam_imu"])
            T_imu_cam = np.linalg.inv(T_cam_imu)
            R, T = T_imu_cam[:3, :3], T_imu_cam[:3, 3]
            Q = Rotation.from_matrix(R).as_quat()
            cam_data.append({"px": T[0], "py": T[1], "pz": T[2], "qx": Q[0], "qy": Q[1], "qz": Q[2], "qw": Q[3]})

            ins = np.array(camera["intrinsics"])
            dcs = np.array(camera["distortion_coeffs"])
            intrinsics.append({
                "camera_type": "kb4",
                "intrinsics": {
                    "fx": ins[0], "fy": ins[1], "cx": ins[2], "cy": ins[3], 
                    "k1": dcs[0], "k2": dcs[1], "k3": dcs[2], "k4": dcs[3]
                }
            }) 

            resolutions.append(camera["resolution"])

    # Extract IMU calibration data
    with open(hilti_imu_cal_path, "r") as file:
        file.readline() # Skip first line
        data = yaml.full_load(file)
        imu = data["imu0"]

        json_string = {
            "value0": {
                "T_imu_cam": cam_data,
                "intrinsics": intrinsics,
                "resolution": resolutions,
                "calib_accel_bias": [0] * 9,
                "calib_gyro_bias": [0] * 12,
                "imu_update_rate": int(imu["update_rate"]),
                "accel_noise_std": [imu["accelerometer_noise_density"]] * 3,
                "gyro_noise_std": [imu["gyroscope_noise_density"]] * 3,
                "accel_bias_std": [imu["accelerometer_random_walk"]] * 3,
                "gyro_bias_std": [imu["gyroscope_random_walk"]] * 3,
                "cam_time_offset_ns": 10000, # Accoriding to Hilti Website
                "vignette": []
            }
        }

    with open(output_path, "w") as file:
        json.dump(json_string, file, indent=4)


def main():
    args = parse_args()
    func_args = vars(args)
    func = func_args.pop("func")
    if "mode" in func_args: 
        func_args.pop("mode")
    func(**func_args)

if __name__ == "__main__":
    main()
