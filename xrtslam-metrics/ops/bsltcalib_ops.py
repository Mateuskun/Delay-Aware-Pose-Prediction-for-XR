#!/usr/bin/env python

import sys

from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable
from math import sqrt
from numpy.linalg import inv
import json

from scipy.spatial.transform import Rotation as R
import numpy as np
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

Mat3 = np.ndarray
Quat = np.ndarray


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_wmr2bslt_calib = Command("wmr2bslt_calib", "Convert a WMR factory json calibrationinto Basalt calibration json", wmr2bslt_calib)
    cmd_xreal2bslt_calib = Command("xreal2bslt_calib", "Convert an XREAL factory json calibration into Basalt calibration json", xreal2bslt_calib)
    # fmt: on

    parser = ArgumentParser(
        description="Helper commands to convert data between okvis2 and EuRoC formats",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_wmr2bslt_calib.name, help=cmd_wmr2bslt_calib.desc)
    subparser.set_defaults(func=cmd_wmr2bslt_calib.func)
    subparser.add_argument("wmr_json", type=Path, help="Path to the WMR firmware json")

    subparser = subparsers.add_parser(cmd_xreal2bslt_calib.name, help=cmd_xreal2bslt_calib.desc)
    subparser.set_defaults(func=cmd_xreal2bslt_calib.func)
    subparser.add_argument("xreal_json", type=Path, help="Path to the XREAL firmware json")

    return parser.parse_args()


class BasaltCalib:
    json: dict
    num_cams: int

    def T_imu_cam(self, i: int) -> dict:
        return {}

    def intrinsics(self, i: int) -> dict:
        return {}

    def resolution(self, i: int) -> list:
        return []

    def calib_accel_bias(self) -> list:
        return []

    def calib_gyro_bias(self) -> list:
        return []

    def imu_update_rate(self) -> int:
        return 0

    def accel_noise_std(self) -> list:
        return []

    def gyro_noise_std(self) -> list:
        return []

    def accel_bias_std(self) -> list:
        return []

    def gyro_bias_std(self) -> list:
        return []

    def cam_time_offset_ns(self) -> int:
        return 0

    def view_offset(self) -> list:
        return []

    def vignette(self) -> list:
        return []

    def get_basalt_calib(self) -> dict:
        out_calib = {
            "value0": {
                "T_imu_cam": [self.T_imu_cam(i) for i in range(self.num_cams)],
                "intrinsics": [self.intrinsics(i) for i in range(self.num_cams)],
                "resolution": [self.resolution(i) for i in range(self.num_cams)],
                "calib_accel_bias": self.calib_accel_bias(),
                "calib_gyro_bias": self.calib_gyro_bias(),
                "imu_update_rate": self.imu_update_rate(),
                "accel_noise_std": self.accel_noise_std(),
                "gyro_noise_std": self.gyro_noise_std(),
                "accel_bias_std": self.accel_bias_std(),
                "gyro_bias_std": self.gyro_bias_std(),
                "cam_time_offset_ns": self.cam_time_offset_ns(),
                # "view_offset": self.view_offset(),
                "vignette": self.vignette(),
            }
        }
        return out_calib


class WmrCalib(BasaltCalib):
    def __init__(self, json_path: Path):
        self.json = json.load(open(json_path, "r", encoding="utf-8"))
        PURPOSE = "CALIBRATION_CameraPurposeHeadTracking"
        self.num_cams = sum(1 for c in self.json["CalibrationInformation"]["Cameras"] if c["Purpose"] == PURPOSE)

        # We get 250 packets with 4 samples each per second, totalling 1000 samples per second.
        # But in monado we just average those 4 samples to reduce the noise. So we have 250hz.
        self.IMU_UPDATE_RATE = 250

    def get(self, name):
        assert name in ["HT0", "HT1", "Gyro", "Accelerometer"]
        is_imu = name in ["Gyro", "Accelerometer"]
        calib = self.json["CalibrationInformation"]
        sensors = calib["InertialSensors" if is_imu else "Cameras"]
        name_key = "SensorType" if is_imu else "Location"
        sensor = next(filter(lambda s: s[name_key].endswith(name), sensors))
        return sensor

    def rt2mat(self, rt):
        R33 = np.array(rt["Rotation"]).reshape(3, 3)
        t31 = np.array(rt["Translation"]).reshape(3, 1)
        T34 = np.hstack((R33, t31))
        T44 = np.vstack((T34, [0, 0, 0, 1]))
        return T44

    def rmat2quat(self, r: Mat3) -> Quat:
        w = sqrt(1 + r[0, 0] + r[1, 1] + r[2, 2]) / 2
        w4 = 4 * w
        x = (r[2, 1] - r[1, 2]) / w4
        y = (r[0, 2] - r[2, 0]) / w4
        z = (r[1, 0] - r[0, 1]) / w4
        return np.array([x, y, z, w])

    def noise_std(self, name: str) -> list:
        imu = self.get(name)
        return imu["Noise"][0:3]

    def bias_std(self, name: str) -> list:
        imu = self.get(name)
        return list(map(sqrt, imu["BiasUncertainty"]))

    def project(self, intrinsics: dict, x: float, y: float, z: float):
        fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2 = (
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["cx"],
            intrinsics["cy"],
            intrinsics["k1"],
            intrinsics["k2"],
            intrinsics["k3"],
            intrinsics["k4"],
            intrinsics["k5"],
            intrinsics["k6"],
            intrinsics["p1"],
            intrinsics["p2"],
        )

        xp = x / z
        yp = y / z
        r2 = xp * xp + yp * yp
        cdist = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        deltaX = 2 * p1 * xp * yp + p2 * (r2 + 2 * xp * xp)
        deltaY = 2 * p2 * xp * yp + p1 * (r2 + 2 * yp * yp)
        xpp = xp * cdist + deltaX
        ypp = yp * cdist + deltaY
        u = fx * xpp + cx
        v = fy * ypp + cy
        return u, v

    def T_imu_cam(self, i: int) -> dict:
        # NOTE: The `Rt` field seems to be a transform from the sensor to HT0 (i.e.,
        # from HT0 space to sensor space). For basalt we need the transforms
        # expressed w.r.t IMU origin.

        # NOTE: The gyro and magnetometer translations are 0, probably because an
        # HMD is a rigid body. Therefore the accelerometer is considered as the IMU
        # origin.

        imu = self.get("Accelerometer")
        T_i_c0 = self.rt2mat(imu["Rt"])

        T = None
        cam = f"HT{i}"
        if cam == "HT0":
            T = T_i_c0
        elif cam == "HT1":
            cam1 = self.get("HT1")
            T_c1_c0 = self.rt2mat(cam1["Rt"])
            T_c0_c1 = inv(T_c1_c0)
            T_i_c1 = T_i_c0 @ T_c0_c1
            T = T_i_c1
        else:
            assert False

        q = self.rmat2quat(T[0:3, 0:3])
        p = T[0:3, 3]
        return {
            "px": p[0],
            "py": p[1],
            "pz": p[2],
            "qx": q[0],
            "qy": q[1],
            "qz": q[2],
            "qw": q[3],
        }

    def intrinsics(self, i: int) -> dict:
        # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/2feb3425259bf803749065bb6d628c6c180f8e77/include/k4a/k4atypes.h#L1024-L1046
        cam = f"HT{i}"
        camera = self.get(cam)
        model_params = camera["Intrinsics"]["ModelParameters"]
        assert camera["Intrinsics"]["ModelType"] == "CALIBRATION_LensDistortionModelRational6KT"
        width = camera["SensorWidth"]
        height = camera["SensorHeight"]
        return {
            "camera_type": "pinhole-radtan8",
            "intrinsics": {
                "fx": model_params[2] * width,
                "fy": model_params[3] * height,
                "cx": model_params[0] * width,
                "cy": model_params[1] * height,
                "k1": model_params[4],
                "k2": model_params[5],
                "p1": model_params[13],
                "p2": model_params[12],
                "k3": model_params[6],
                "k4": model_params[7],
                "k5": model_params[8],
                "k6": model_params[9],
                "rpmax": model_params[14],
            },
        }

    def resolution(self, i: int) -> list:
        cam = f"HT{i}"
        camera = self.get(cam)
        width = camera["SensorWidth"]
        height = camera["SensorHeight"]
        return [width, height]

    def calib_accel_bias(self) -> list:
        # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/2feb3425259bf803749065bb6d628c6c180f8e77/include/k4ainternal/calibration.h#L48-L77
        # https://vladyslavusenko.gitlab.io/basalt-headers/classbasalt_1_1CalibAccelBias.html#details
        # https://gitlab.com/VladyslavUsenko/basalt-headers/-/issues/8
        accel = self.get("Accelerometer")
        bias = accel["BiasTemperatureModel"]
        align = accel["MixingMatrixTemperatureModel"]
        return [
            -bias[0 * 4],
            -bias[1 * 4],
            -bias[2 * 4],
            align[0 * 4] - 1,  # [0, 0]
            align[3 * 4],  # [1, 0]
            align[6 * 4],  # [2, 0]
            align[4 * 4] - 1,  # [1, 1]
            align[7 * 4],  # [2, 1]
            align[8 * 4] - 1,  # [2, 2]
        ]

    def calib_gyro_bias(self) -> list:
        # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/2feb3425259bf803749065bb6d628c6c180f8e77/include/k4ainternal/calibration.h#L48-L77
        # https://vladyslavusenko.gitlab.io/basalt-headers/classbasalt_1_1CalibGyroBias.html#details
        gyro = self.get("Gyro")
        bias = gyro["BiasTemperatureModel"]
        align = gyro["MixingMatrixTemperatureModel"]
        return [
            -bias[0 * 4],
            -bias[1 * 4],
            -bias[2 * 4],
            align[0 * 4] - 1,  # [0, 0]
            align[3 * 4],  # [1, 0]
            align[6 * 4],  # [2, 0]
            align[1 * 4],  # [0, 1]
            align[4 * 4] - 1,  # [1, 1]
            align[7 * 4],  # [2, 1]
            align[2 * 4],  # [0, 2]
            align[5 * 4],  # [1, 2]
            align[8 * 4] - 1,  # [2, 2]
        ]

    def imu_update_rate(self) -> int:
        return self.IMU_UPDATE_RATE

    def accel_noise_std(self) -> list:
        return self.noise_std("Accelerometer")

    def gyro_noise_std(self) -> list:
        return self.noise_std("Gyro")

    def accel_bias_std(self) -> list:
        return self.bias_std("Accelerometer")

    def gyro_bias_std(self) -> list:
        return self.bias_std("Gyro")

    def cam_time_offset_ns(self) -> int:
        return 0

    def view_offset(self) -> list:
        """
        This is a very rough offset in pixels between the two cameras. Originally we
        needed to manually estimate it like explained and shown here
        https://youtu.be/jyQKjyRVMS4?t=670.
        With this calculation we get a similar number without the need to open Gimp.

        In reality this offset changes based on distance to the point, nonetheless
        it helps to get some features tracked in the right camera.
        """

        # Rough approximation of how far from the cameras features will likely be in your room
        DISTANCE_TO_WALL = 2  # In meters

        cam1 = self.get("HT1")
        width = cam1["SensorWidth"]
        height = cam1["SensorHeight"]
        cam1_intrinsics = self.intrinsics(1)["intrinsics"]
        T_c1_c0 = self.rt2mat(cam1["Rt"])  # Maps a point in c0 space to c1 space
        p = np.array([0, 0, DISTANCE_TO_WALL, 1])  # Fron tof c0, in homogeneous coords
        p_in_c1 = T_c1_c0 @ p  # Point in c1 coordinates
        u, v = self.project(cam1_intrinsics, *p_in_c1[0:3])
        view_offset = [width / 2 - u, height / 2 - v]  # We used a point in the middle of c0
        return view_offset

    def vignette(self) -> list:
        return []


class XrealCalib(BasaltCalib):
    def __init__(self, json_path: Path):
        self.json = json.load(open(json_path, "r", encoding="utf-8"))
        self.num_cams = self.json["SLAM_camera"]["num_of_cameras"]
        self.IMU_UPDATE_RATE = 1000

    def T_imu_cam(self, i: int) -> dict:
        cam = self.json["SLAM_camera"][f"device_{i+1}"]
        p_imu_cam = cam["imu_p_cam"]
        q_imu_cam = cam["imu_q_cam"]

        # Convert to SE3 matrix
        T_imu_cam = np.eye(4)
        T_imu_cam[0:3, 3] = p_imu_cam
        R_imu_cam = R.from_quat([q_imu_cam[0], q_imu_cam[1], q_imu_cam[2], q_imu_cam[3]]).as_matrix()
        T_imu_cam[0:3, 0:3] = R_imu_cam

        # Change of basis between XREAL and Basalt
        # IMU samples in XREAL frame give +X right, +Y up, +Z forward
        T_A_B = np.diag([1, -1, -1, 1])
        T_imu_cam = T_imu_cam @ T_A_B

        # Back to quat + pos
        p1 = T_imu_cam[0:3, 3]
        R_imu_cam = T_imu_cam[0:3, 0:3]
        q1 = R.from_matrix(R_imu_cam).as_quat()  # x, y, z, w
        return {
            "px": p1[0],
            "py": p1[1],
            "pz": p1[2],
            "qx": q1[0],
            "qy": q1[1],
            "qz": q1[2],
            "qw": q1[3],
        }

    def intrinsics(self, i: int) -> dict:
        cam = self.json["SLAM_camera"][f"device_{i+1}"]
        model = cam["camera_model"]
        assert model == "fisheye624"
        fc = cam["fc"]
        cc = cam["cc"]
        kc = cam["kc"]
        return {
            "camera_type": "fisheye624",
            "intrinsics": {
                "fx": fc[0],
                "fy": fc[1],
                "cx": cc[0],
                "cy": cc[1],
                "k1": kc[0],
                "k2": kc[1],
                "k3": kc[2],
                "k4": kc[3],
                "k5": kc[4],
                "k6": kc[5],
                "p1": kc[6],
                "p2": kc[7],
                "s1": kc[8],
                "s2": kc[9],
                "s3": kc[10],
                "s4": kc[11],
            },
        }

    def resolution(self, i: int) -> list:
        cam = self.json["SLAM_camera"][f"device_{i+1}"]
        res = cam["resolution"]
        return [res[0], res[1]]

    def calib_accel_bias(self) -> list:
        imu = self.json["IMU"]["device_1"]
        bias = imu["imu_intrinsics"]["accl_bias"]
        align = imu["imu_intrinsics"]["accl_calib_mat"]
        return [
            -bias[0],  # TODO: Assuming bias is positive in xreal json
            -bias[1],  # TODO: Assuming bias is positive in xreal json
            -bias[2],  # TODO: Assuming bias is positive in xreal json
            align[0] - 1,  # [0, 0]
            align[3],  # [1, 0]
            align[6],  # [2, 0]
            # align[1],  # [0, 1] # TODO: See https://gitlab.com/VladyslavUsenko/basalt-headers/-/issues/8
            align[4] - 1,  # [1, 1]
            align[7],  # [2, 1]
            # align[2],  # [0, 2] # TODO: See https://gitlab.com/VladyslavUsenko/basalt-headers/-/issues/8
            # align[5],  # [1, 2] # TODO: See https://gitlab.com/VladyslavUsenko/basalt-headers/-/issues/8
            align[8] - 1,  # [2, 2]
        ]

    def calib_gyro_bias(self) -> list:
        imu = self.json["IMU"]["device_1"]
        bias = imu["imu_intrinsics"]["gyro_bias"]
        align = imu["imu_intrinsics"]["gyro_calib_mat"]
        return [
            -bias[0],  # TODO: Assuming bias is positive in xreal json
            -bias[1],  # TODO: Assuming bias is positive in xreal json
            -bias[2],  # TODO: Assuming bias is positive in xreal json
            align[0] - 1,  # [0, 0]
            align[3],  # [1, 0]
            align[6],  # [2, 0]
            align[1],  # [0, 1]
            align[4] - 1,  # [1, 1]
            align[7],  # [2, 1]
            align[2],  # [0, 2]
            align[5],  # [1, 2]
            align[8] - 1,  # [2, 2]
        ]

    def imu_update_rate(self) -> int:
        return self.IMU_UPDATE_RATE

    def accel_noise_std(self) -> list:
        imu = self.json["IMU"]["device_1"]
        noises = imu["imu_noises"]
        return [noises[2], noises[2], noises[2]]

    def gyro_noise_std(self) -> list:
        imu = self.json["IMU"]["device_1"]
        noises = imu["imu_noises"]
        return [noises[0], noises[0], noises[0]]

    def accel_bias_std(self) -> list:
        imu = self.json["IMU"]["device_1"]
        noises = imu["imu_noises"]
        return [noises[3], noises[3], noises[3]]

    def gyro_bias_std(self) -> list:
        imu = self.json["IMU"]["device_1"]
        noises = imu["imu_noises"]
        return [noises[1], noises[1], noises[1]]

    def cam_time_offset_ns(self) -> int:
        return 0

    def view_offset(self) -> list:
        return []

    def vignette(self) -> list:
        return []


def wmr2bslt_calib(args: Namespace):
    "Convert a WMR factory json calibrationinto Basalt calibration json"
    wmr_json = args.wmr_json
    wmr = WmrCalib(wmr_json)
    out_calib = wmr.get_basalt_calib()
    print(json.dumps(out_calib, indent=4))


def xreal2bslt_calib(args: Namespace):
    "Convert an XREAL factory json calibrationinto Basalt calibration json"
    xreal_json = args.xreal_json
    xreal = XrealCalib(xreal_json)
    out_calib = xreal.get_basalt_calib()
    print(json.dumps(out_calib, indent=4))


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
