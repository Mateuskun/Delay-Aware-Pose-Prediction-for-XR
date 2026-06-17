#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import cv2
import rosbag
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main_ds(bag_path, euroc_path):
    print("Converting Hilti dataset")

    mav0_path = euroc_path / "mav0"
    mav0_path.mkdir(parents=True, exist_ok=True)

    bag = rosbag.Bag(bag_path, "r")

    print("Exporting IMU data")
    imu_path = mav0_path / f"imu0"
    imu_path.mkdir()

    csv_path = imu_path / "data.csv"
    csv_file = open(csv_path, "w")
    csv_file.write(
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\r\n"
    )
    for topic, msg, t in bag.read_messages(topics=["/alphasense/imu"]):
        t = int(f"{msg.header.stamp.secs}{msg.header.stamp.nsecs:09d}")
        w = msg.angular_velocity
        a = msg.linear_acceleration
        csv_file.write(f"{t},{w.x},{w.y},{w.z},{a.x},{a.y},{a.z}\r\n")
    csv_file.close()
    print("Done.")

    # Export camera data
    bridge = CvBridge()
    for i in range(5):
        print(f"Exporting cam{i} data")
        cam_path = mav0_path / f"cam{i}"
        cam_path.mkdir()

        data_path = cam_path / "data"
        data_path.mkdir()

        csv_path = cam_path / "data.csv"
        csv_file = open(csv_path, "w")
        csv_file.write("#timestamp [ns],filename\r\n")

        for c, (topic, msg, t) in enumerate(bag.read_messages(topics=[f"/alphasense/cam{i}/image_raw"])):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            timestamp = int(f"{msg.header.stamp.secs}{msg.header.stamp.nsecs:09d}")
            cv2.imwrite(str(data_path / f"{timestamp}.png"), cv_img)
            csv_file.write(f"{timestamp},{timestamp}.png\r\n")
            print(f"Wrote image {c}\r", end="")

        csv_file.close()
        print(f"Done.")

    bag.close()


def main_gt(hilti_gt_path, output_path):
    print("Converting Hilti groundtruth")
    with open(hilti_gt_path, "r") as f:
        contents = f.read()

    first_line = (
        "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\r\n"
    )
    lines = contents.split("\n")
    lines = [l for l in lines if l]  # Remove empty lines
    lines = [l.split(" ") for l in lines]  # Make lines into lists
    lines = [l[0:4] + [l[7]] + l[4:7] for l in lines]  # Swap xyzw to wxyz
    lines = [[f"{l[0].split('.')[0]}{l[0].split('.')[1].ljust(9, '0')}"] + l[1:] for l in lines]
    lines = [",".join(l) for l in lines]  # Back to comma-separated strings
    lines = "\r\n".join(lines) + "\r\n"  # Back to a big string with CRLF EOL
    lines = first_line + lines  # Add header

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Make EuRoC dataset from Hilti ROS bag.\n\n"
        "Usage example for dataset exp14 (get it from https://hilti-challenge.com/dataset-2022.html):\n"
        "$ ./utils/hilti_rosbag_to_euroc.py ds bags/exp14_basement_2.bag exp14\n"
        "$ ./utils/hilti_rosbag_to_euroc.py gt ~/Downloads/exp14_basement_2_imu.txt exp14/mav0/state_groundtruth_estimate0/data.csv\n"
    )

    subparsers = parser.add_subparsers(
        help="Convert hilti rosbag or groundtruth file to euroc format",
        dest="mode",
        required=True,
    )

    parser_ds = subparsers.add_parser("ds", help="Convert hilti rosbag to euroc dataset")
    parser_ds.add_argument("bag", type=Path)
    parser_ds.add_argument("output_path", type=Path)

    parser_gt = subparsers.add_parser("gt", help="Convert hilti groundtruth file to euroc groundtruth csv")
    parser_gt.add_argument("hilti_gt_path", type=Path, help="The hilti groundtruth input file")
    parser_gt.add_argument(
        "output_path",
        type=Path,
        help="The euroc groundtruth file to write",
        default="out.csv",
    )

    args = parser.parse_args()
    if args.mode == "ds":
        bag_path = args.bag
        euroc_path = args.output_path
        main_ds(bag_path, euroc_path)
    elif args.mode == "gt":
        hilti_gt_path = args.hilti_gt_path
        output_path = args.output_path
        main_gt(hilti_gt_path, output_path)
    else:
        raise Exception(f"Invalid mode={args.mode}")

    return 0


if __name__ == "__main__":
    main()
