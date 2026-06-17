#!/usr/bin/env python

# Lastly tested with a Ubuntu 22.04 - python3.10.12 environment

import shutil

from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable

import PIL.Image
import numpy as np
from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore
from euroc_ops import SensorPaths

typestore = None
IMU = None
Image = None
CompressedImage = None
Header = None
Time = None
Vector3 = None
Quaternion = None
QUAT_IDENTITY = None
VEC3_ZERO = None
NP9_ZEROS = None


def set_global_typestore(ros_version: str) -> Stores:
    global typestore
    global IMU
    global Image
    global CompressedImage
    global Header
    global Time
    global Vector3
    global Quaternion
    global QUAT_IDENTITY
    global VEC3_ZERO
    global NP9_ZEROS

    Writer = None
    serialize = None
    if ros_version == "ros1":
        typestore = get_typestore(Stores.ROS1_NOETIC)
        Writer = Writer1
        serialize = typestore.serialize_ros1
        Header = typestore.types["std_msgs/msg/Header"]
    elif ros_version == "ros2":
        typestore = get_typestore(Stores.LATEST)
        Writer = Writer2
        serialize = typestore.serialize_cdr
        Header_ = typestore.types["std_msgs/msg/Header"]
        Header = lambda seq=0, **kwargs: Header_(**kwargs)
    else:
        raise ValueError(f"Unknown rosbag target: {ros_version}")

    IMU = typestore.types["sensor_msgs/msg/Imu"]
    Image = typestore.types["sensor_msgs/msg/Image"]
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    QUAT_IDENTITY = Quaternion(x=0, y=0, z=0, w=1)
    VEC3_ZERO = Vector3(x=0, y=0, z=0)
    NP9_ZEROS = np.zeros((9,), dtype=np.float64)

    return Writer, serialize


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_euroc_to_ros = Command("euroc2ros", "Convert an dataset in EuRoC ASL format into a ROS 1 or 2 bag", euroc2ros)
    # fmt: on

    parser = ArgumentParser(
        description="Helper commands to convert datasets between rosbags and EuRoC formats",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_euroc_to_ros.name, help=cmd_euroc_to_ros.desc)
    subparser.set_defaults(func=cmd_euroc_to_ros.func)
    subparser.add_argument(
        "ros_version",
        type=str,
        choices=["ros1", "ros2"],
        help="Whether to do a ros1 or ros2 bag",
    )
    subparser.add_argument("euroc_path", type=Path, help="Path to the EuRoC dataset")
    subparser.add_argument("rosbag_path", type=Path, help="Path to output ros bag")

    return parser.parse_args()


def euroc2ros(args: Namespace):
    "Convert an EuRoC dataset into a ROS 1 or ROS 2 bag"
    # NOTE: For debugging bags use tools like:
    # rostopic echo -b V1_01_easy.ros1.bag -p /cam0/image_raw > generated_bag.txt
    # rosbag check V1_01_easy.ros1.bag
    # etc.

    ros_version = args.ros_version
    euroc_path = SensorPaths(args.euroc_path)
    rosbag_path: Path = args.rosbag_path

    Writer, serialize = set_global_typestore(ros_version)

    if rosbag_path.exists():
        if rosbag_path.is_dir():
            shutil.rmtree(rosbag_path)
        else:
            rosbag_path.unlink()

    with Writer(rosbag_path) as writer:
        sensor_name = "imu0"
        imu_topic = f"/{sensor_name}"
        imu_msgtype = IMU.__msgtype__
        connection = writer.add_connection(imu_topic, imu_msgtype, typestore=typestore)

        with open(euroc_path.imu, "r") as imucsv:
            for i, line in enumerate(imucsv):
                if line.startswith("#"):
                    continue
                split = line.split(",")[:7]
                ts = int(split[0])
                wx, wy, wz, ax, ay, az = map(float, split[1:])

                time = Time(sec=ts // int(1e9), nanosec=ts % int(1e9))
                header = Header(seq=i, stamp=time, frame_id=sensor_name)
                w = Vector3(x=wx, y=wy, z=wz)
                a = Vector3(x=ax, y=ay, z=az)

                imu_msg = IMU(
                    header=header,
                    orientation=QUAT_IDENTITY,
                    orientation_covariance=NP9_ZEROS,
                    angular_velocity=w,
                    angular_velocity_covariance=NP9_ZEROS,
                    linear_acceleration=a,
                    linear_acceleration_covariance=NP9_ZEROS,
                )
                serialized_msg = serialize(imu_msg, imu_msgtype)
                writer.write(connection, ts, serialized_msg)

        cam_msgtype = Image.__msgtype__
        for cam in euroc_path.cams:
            sensor_name = cam.parent.name
            cam_topic = f"/{sensor_name}/image_raw"
            connection = writer.add_connection(cam_topic, cam_msgtype, typestore=typestore)
            with open(cam, "r") as camcsv:
                for i, line in enumerate(camcsv):
                    if line.startswith("#"):
                        continue
                    split = line.split(",")[:2]
                    ts = int(split[0])
                    img_name = split[1].strip()

                    time = Time(sec=ts // int(1e9), nanosec=ts % int(1e9))
                    header = Header(seq=i, stamp=time, frame_id=sensor_name)
                    img_path = cam.parent / f"data/{img_name}"

                    img = np.array(PIL.Image.open(img_path))
                    height, width = img.shape
                    img = img.reshape(-1)
                    img_msg = Image(
                        header=header,
                        height=height,
                        width=width,
                        encoding="mono8",
                        is_bigendian=0,
                        step=width,
                        data=img,
                    )

                    serialized_msg = serialize(img_msg, cam_msgtype)
                    writer.write(connection, ts, serialized_msg)


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
