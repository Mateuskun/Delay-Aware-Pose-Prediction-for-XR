#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable, Optional
from bisect import bisect_left
from scipy.spatial.transform import Rotation

import json
import shutil
import sys
import os
import numpy as np
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_csv_unsafe


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_imu2cam_ts = Command("imu2cam_ts", "Create a new IMU csv with timestamps modified (read code)", imu2cam_ts)
    cmd_cam_offset_ts = Command("cam_offset_ts", "Create a new camera csv (and its extra file) with timestamps modified by an offset", cam_offset_ts)
    cmd_get_duration = Command("get_duration", "Get duration of dataset", get_duration)
    cmd_get_csv_duration = Command("get_csv_duration", "Get duration of a sensor csv file", get_csv_duration)
    cmd_verify = Command("verify", "Perform many asserts on the dataset to check its integrity", verify)
    cmd_get_max_sensor_dt = Command("get_max_sensor_dt", "Get maximum sensor delta time", get_max_sensor_dt)
    cmd_trim = Command("trim", "Trim dataset", trim)
    cmd_apply_imu_calib = Command("apply_imu_calib", "Apply IMU calibration to dataset samples", apply_imu_calib)
    cmd_apply_transform = Command("apply_transform", "Get trajectory with SE3 transform applied", apply_transform)
    cmd_preview_video = Command("preview_video", "Generate a preview video of the entire dataset. Dataset path should be an absolute path", preview_video)
    # fmt: on

    subcommands_woargs = [
        cmd_imu2cam_ts,
        cmd_get_duration,
        cmd_verify,
    ]

    parser = ArgumentParser(
        description="Sanitize EuRoC datasets",
    )
    subparsers = parser.add_subparsers(help="What operation to perform")

    for cmd in subcommands_woargs:
        subparser = subparsers.add_parser(cmd.name, help=cmd.desc)
        subparser.set_defaults(func=cmd.func)

    get_max_sensor_dt_parser = subparsers.add_parser(cmd_get_max_sensor_dt.name, help=cmd_get_max_sensor_dt.desc)
    get_max_sensor_dt_parser.set_defaults(func=cmd_get_max_sensor_dt.func)
    get_max_sensor_dt_parser.add_argument(
        "sensor",
        type=str,
        default="imu",
        choices=["imu", "gt"] + [f"cam{i}" for i in range(5)],
        help="Sensor to get max dt from",
    )

    cam_offset_ts_parser = subparsers.add_parser(cmd_cam_offset_ts.name, help=cmd_cam_offset_ts.desc)
    cam_offset_ts_parser.set_defaults(func=cmd_cam_offset_ts.func)
    cam_offset_ts_parser.add_argument(
        "offset",
        type=int,
        help="Offset to add to all timestamps in data.csv and data.extra.csv",
    )

    trim_parser = subparsers.add_parser(cmd_trim.name, help=cmd_trim.desc)
    trim_parser.set_defaults(func=cmd_trim.func)
    trim_parser.add_argument(
        "output_path",
        type=Path,
        default=None,
        help="Output path of the trimmed dataset",
    )
    trim_parser.add_argument(
        "start_s",
        type=float,
        default=0.0,
        help="At which second of the dataset to start the trim",
    )
    trim_parser.add_argument(
        "end_s",
        type=float,
        default=30.0,
        help="At which second of the dataset to start the trim",
    )

    apply_imu_calib_parser = subparsers.add_parser(cmd_apply_imu_calib.name, help=cmd_apply_imu_calib.desc)
    apply_imu_calib_parser.set_defaults(func=cmd_apply_imu_calib.func)
    apply_imu_calib_parser.add_argument(
        "calibration_file",
        type=Path,
        help="Input calibration file in Basalt format: See https://vladyslavusenko.gitlab.io/basalt-headers/classbasalt_1_1CalibGyroBias.html#details and https://vladyslavusenko.gitlab.io/basalt-headers/classbasalt_1_1CalibAccelBias.html#details",
    )
    apply_imu_calib_parser.add_argument(
        "output_path",
        type=Path,
        help="Path where to save the calibrated CSV file",
    )

    apply_transform_parser = subparsers.add_parser(cmd_apply_transform.name, help=cmd_apply_transform.desc)
    apply_transform_parser.set_defaults(func=cmd_apply_transform.func)
    apply_transform_parser.add_argument("input_csv", type=Path, help="Trajectory csv file in EuRoC format")
    apply_transform_parser.add_argument("px", type=float)
    apply_transform_parser.add_argument("py", type=float)
    apply_transform_parser.add_argument("pz", type=float)
    apply_transform_parser.add_argument("qx", type=float)
    apply_transform_parser.add_argument("qy", type=float)
    apply_transform_parser.add_argument("qz", type=float)
    apply_transform_parser.add_argument("qw", type=float)
    apply_transform_parser.add_argument("output_csv", type=Path, help="Path where to save the transformed CSV file")

    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Dataset path (the path that contains the mav0 directory)",
    )

    get_csv_duration_parser = subparsers.add_parser(cmd_get_csv_duration.name, help=cmd_get_csv_duration.desc)
    get_csv_duration_parser.set_defaults(func=cmd_get_csv_duration.func)
    get_csv_duration_parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the csv file",
    )

    preview_video_parser = subparsers.add_parser(cmd_preview_video.name, help=cmd_preview_video.desc)
    preview_video_parser.set_defaults(func=cmd_preview_video.func)
    preview_video_parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="How many seconds should the video last, conflicts with --speedup",
    )
    preview_video_parser.add_argument(
        "--speedup",
        "-s",
        type=float,
        default=None,
        help="Speedup playback speed, conflicts with --duration",
    )
    preview_video_parser.add_argument(
        "--scale",
        "-rs",
        type=float,
        default=0.5,
        help="Resolution scale. 640x480 at -rs 0.5 becomes 320x240",
    )
    preview_video_parser.add_argument(
        "--rotate_90cw",
        "-rcw90",
        type=bool,
        default=False,
        help="Rotate 90 degrees clockwise",
    )
    preview_video_parser.add_argument(
        "--rotate_90ccw",
        "-rccw90",
        type=bool,
        default=False,
        help="Rotate 90 degrees counter clockwise",
    )
    preview_video_parser.add_argument(
        "--fps",
        "-f",
        type=int,
        default=24,
        help="Frames per second",
    )
    preview_video_parser.add_argument(
        "--tmp_dir",
        "-t",
        type=Path,
        default="tmp",
        help="Temp dir name",
    )
    preview_video_parser.add_argument(
        "--cam_idx",
        "-i",
        type=int,
        default=0,
        help="Camera index to make a video from, defaults to cam0",
    )
    preview_video_parser.add_argument(
        "--output_path",
        "-o",
        default="dataset_preview",
        type=Path,
        help="Name of the generated video file without extension",
    )

    return parser.parse_args()


class SensorPaths:
    gt: Path
    imu: Path
    cams: list[Path]

    imu_extra: Optional[Path]
    cams_extra: list[Optional[Path]]

    def __init__(self, dataset_path: Path):
        euroc_path = dataset_path / "mav0"
        assert euroc_path.exists(), f"Dataset path not found: {euroc_path}"

        imu_path = euroc_path / "imu0/data.csv"
        possible_gt_dirs = ["gt", "state_groundtruth_estimate0", "mocap0"]
        for gt_dir in possible_gt_dirs:
            gt_path = euroc_path / f"{gt_dir}/data.csv"
            if gt_path.exists():
                break
        assert gt_path.exists()
        cams_path = sorted([d / "data.csv" for d in euroc_path.iterdir() if d.name.startswith("cam")])

        csvs = [imu_path] + cams_path
        assert all(csv.exists() for csv in csvs)
        if not gt_path.exists():
            print("Warning: gt path doesn't exist")

        imu_extra_path = imu_path.with_suffix(".extra.csv")
        cams_extra_path = [cam_path.with_suffix(".extra.csv") for cam_path in cams_path]

        imu_extra_path = imu_extra_path if imu_extra_path.exists() else None
        cams_extra_path = [cam_extra if cam_extra.exists() else None for cam_extra in cams_extra_path]

        self.gt = gt_path
        self.imu = imu_path
        self.cams = cams_path
        self.imu_extra = imu_extra_path
        self.cams_extra = cams_extra_path

    def __getitem__(self, key: str) -> Path:
        if hasattr(self, key):
            return vars(self)[key]
        elif key.startswith("cam"):
            return self.cams[int(key.split("cam")[-1])]
        else:
            raise IndexError(f"Key {key} not found")

    @property
    def has_imu_extra(self) -> bool:
        return self.imu_extra is not None

    @property
    def has_cams_extra(self) -> bool:
        return all(c is not None for c in self.cams_extra)


def get_max_sensor_dt(args: Namespace) -> None:
    paths = SensorPaths(args.dataset_path)
    sensor = load_csv_unsafe(paths[args.sensor], np.int64)
    dts = sensor[:, 0][1:] - sensor[:, 0][:-1]  # type: ignore
    print(f"Max sensor deltatime={max(dts)} on line {np.argmax(dts) + 2} and next")
    print(f"Min sensor deltatime={min(dts)} on line {np.argmin(dts) + 2} and next")


def get_duration_of(file_path):
    with open(file_path, "r") as file:
        header = file.readline().strip()
        first_line = file.readline().strip()

        file.seek(0, os.SEEK_END)  # seek end of file
        pos = file.tell()  # cursor position
        last_line = ""

        while pos > 0:  # while we've not reached start of file
            pos -= 1
            file.seek(pos)
            if file.read(1) == "\n":
                last_line = file.readline().strip()
                if last_line != "":  # ignore empty lines
                    break

    first_ts = int(first_line.split(",")[0])
    last_ts = int(last_line.split(",")[0])
    return (last_ts - first_ts) / 1e9


def get_duration(args: Namespace):
    paths = SensorPaths(args.dataset_path)

    gt_duration = get_duration_of(paths.gt) / 60
    imu_duration = get_duration_of(paths.imu) / 60
    cams_duration = [get_duration_of(cam) / 60 for cam in paths.cams]

    print(f"Duration gt={gt_duration:.2f}min")
    print(f"Duration imu={imu_duration:.2f}min")
    for i, cam_duration in enumerate(cams_duration):
        print(f"Duration cam{i}={cam_duration:.2f}min")


def get_csv_duration(args: Namespace):
    path = args.csv_path

    duration = get_duration_of(path)
    print(f"Duration {path}={duration:.2f} seconds")


def verify(args: Namespace):
    paths = SensorPaths(args.dataset_path)

    typ = np.int64  # 32 unicode characters should be enough for all CSVs
    imu = load_csv_unsafe(paths.imu, typ)
    gt = load_csv_unsafe(paths.gt, typ)
    cams = [load_csv_unsafe(cam_path, "<U32") for cam_path in paths.cams]
    cams = [np.array([(int(ts), int(fn.split(".png")[0])) for ts, fn in cam], dtype=typ) for cam in cams]

    for i, sensor in enumerate([imu, gt] + cams):
        tss = sensor[:, 0]
        tss_diffs = tss[1:] - tss[:-1]  # type: ignore

        assert all(tss_diffs > 0), "Timestamps are not strictly increasing"

        estimated_dt = (tss[-1] - tss[0]) / tss.size
        if not all(tss_diffs < estimated_dt * 2):
            sensor_str = ["imu", "gt"] + [f"cam{i}" for i in range(len(cams))]
            print(
                f"Warning: delta between timestamps {max(tss_diffs)}, "
                f"for sensor {sensor_str[i]} at line={np.argmax(tss_diffs) + 2}, "
                f"which is {max(tss_diffs) / estimated_dt:.2f}x "
                f"the average delta={estimated_dt}"
            )

    if paths.has_imu_extra:
        imu_extras = load_csv_unsafe(paths.imu.with_stem("data.extra"), typ)
        assert all(imu_extras[:, 0] == imu[:, 0])

    if paths.has_cams_extra:
        cams_extras = [load_csv_unsafe(cam_path.with_stem("data.extra"), typ) for cam_path in paths.cams]
        for cam_extra, cam in zip(cams_extras, cams):
            assert all(cam_extra[:, 0] == cam[:, 0])

    cam0_tss = set()
    for i, cam in enumerate(cams):
        csv_tss = set(cam[:, 1])
        img_tss = {np.int64(img.stem) for img in (paths.cams[i].parent / "data").iterdir()}
        if (img_tss - csv_tss) != set():
            print(f"Warning: Images not in csv for cam{i}: {img_tss - csv_tss}")
        assert csv_tss - img_tss == set(), f"Stamps without images: {csv_tss - img_tss}"
        if i == 0:
            cam0_tss = csv_tss
        else:
            cami_tss = csv_tss
            assert cam0_tss - cami_tss == set(), f"cam0 images not in cam{i}: {cam0_tss - cami_tss}"
            assert cami_tss - cam0_tss == set(), f"cam{i} images not in cam0: {cami_tss - cam0_tss}"

    imu_dur = get_duration_of(paths.imu)
    TOLERANCE_S = 1
    for sensor_path in [paths.gt] + paths.cams:
        sensor_dur = get_duration_of(sensor_path)
        check = abs(sensor_dur - imu_dur) < TOLERANCE_S
        assert check, f"Duration of {sensor_path} ({sensor_dur:.2f}s) is too different from IMU data ({imu_dur:.2f}s)"


def trim(args: Namespace):
    paths = SensorPaths(args.dataset_path)
    start_s = args.start_s
    end_s = args.end_s
    output_path = args.output_path

    typ = "<U32"  # 32 unicode characters should be enough for all CSVs
    imu = load_csv_unsafe(paths.imu, typ)  # type: ignore

    gt = load_csv_unsafe(paths.gt, typ)  # type: ignore
    cams = [load_csv_unsafe(cam_path, typ) for cam_path in paths.cams]  # type: ignore

    first_ts = int(imu[0, 0])
    start_ts = first_ts + start_s * 1e9
    end_ts = start_ts + (end_s - start_s) * 1e9

    imu_start_idx = bisect_left(imu[:, 0], start_ts, key=float)
    imu_end_idx = bisect_left(imu[:, 0], end_ts, key=float)

    gt_start_idx = bisect_left(gt[:, 0], start_ts, key=float)
    gt_end_idx = bisect_left(gt[:, 0], end_ts, key=float)

    cams_start_idx = [bisect_left(cam[:, 0], start_ts, key=float) for cam in cams]
    cams_end_idx = [bisect_left(cam[:, 0], end_ts, key=float) for cam in cams]

    # Trim data
    imu_cut = imu[imu_start_idx:imu_end_idx]
    gt_cut = gt[gt_start_idx:gt_end_idx]
    cams_cut = [cam[i:j] for cam, i, j in zip(cams, cams_start_idx, cams_end_idx)]

    # Setup directory structure
    output_path.mkdir()
    mav0_path = output_path / "mav0"
    mav0_path.mkdir()
    imu0_path = mav0_path / "imu0"
    imu0_path.mkdir()
    gt_path = mav0_path / "gt"
    gt_path.mkdir()
    cams_path = [mav0_path / f"cam{i}" for i in range(len(cams))]
    for path in cams_path:
        path.mkdir()
    cams_data_path = [cam / "data" for cam in cams_path]
    for path in cams_data_path:
        path.mkdir()

    # Save csv files
    imu0_csv = imu0_path / "data.csv"
    gt_csv = gt_path / "data.csv"
    cams_csv = [path / "data.csv" for path in cams_path]
    imu_header = "timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
    gt_header = "timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []"
    cam_header = "timestamp [ns],filename"
    kwargs = {"delimiter": ",", "fmt": "%s", "comments": "#", "newline": "\r\n"}

    np.savetxt(imu0_csv, imu_cut, header=imu_header, **kwargs)
    np.savetxt(gt_csv, gt_cut, header=gt_header, **kwargs)
    for cam_csv, cam_cut in zip(cams_csv, cams_cut):
        np.savetxt(cam_csv, cam_cut, header=cam_header, **kwargs)

    # Copy frames
    for cam_path_src, dst, cam in zip(paths.cams, cams_data_path, cams_cut):
        src = cam_path_src.parent / "data"
        for _, fn in cam:
            shutil.copy(src / fn, dst / fn)

    if paths.has_imu_extra:
        imu_extras = load_csv_unsafe(paths.imu.with_stem("data.extra"), typ)  # type: ignore
        imu_extras_start_idx = bisect_left(imu_extras[:, 0], start_ts, key=float)
        imu_extras_end_idx = bisect_left(imu_extras[:, 0], end_ts, key=float)
        imu_extras_cut = imu_extras[imu_extras_start_idx:imu_extras_end_idx]
        imu0_extra_csv = imu0_path / "data.extra.csv"
        imu_extra_header = "timestamp [ns],host timestamp [ns]"
        np.savetxt(imu0_extra_csv, imu_extras_cut, header=imu_extra_header, **kwargs)

    if paths.has_cams_extra:
        cams_extras = [load_csv_unsafe(cam_path.with_stem("data.extra"), typ) for cam_path in paths.cams]  # type: ignore
        cams_extras_start_idx = [bisect_left(cam[:, 0], start_ts, key=float) for cam in cams_extras]
        cams_extras_end_idx = [bisect_left(cam[:, 0], end_ts, key=float) for cam in cams_extras]
        cams_extras_cut = [cam[i:j] for cam, i, j in zip(cams_extras, cams_extras_start_idx, cams_extras_end_idx)]
        cams_extra_csv = [path / "data.extra.csv" for path in cams_path]
        cam_extra_header = "timestamp [ns],host timestamp [ns]"
        for cam_extra_csv, cam_extra_cut in zip(cams_extra_csv, cams_extras_cut):
            np.savetxt(cam_extra_csv, cam_extra_cut, header=cam_extra_header, **kwargs)


def apply_imu_calib(args: Namespace):
    paths = SensorPaths(args.dataset_path)
    calibration_fn = args.calibration_file
    output_fn = args.output_path

    calibration = json.load(open(calibration_fn, "r"))
    gb_raw = calibration["value0"]["calib_gyro_bias"][0:3]
    gm_raw = calibration["value0"]["calib_gyro_bias"][3:12]
    ab_raw = calibration["value0"]["calib_accel_bias"][0:3]
    am_raw = calibration["value0"]["calib_accel_bias"][3:9]

    gb = np.array(gb_raw)  # Bias
    gm = np.array(gm_raw)  # Scale/alignment matrix
    ab = np.array(ab_raw)
    am = np.array(am_raw[0:3] + [0] + am_raw[3:5] + [0, 0] + am_raw[5:6])
    gm = gm.reshape((3, 3)).T + np.identity(3)
    am = am.reshape((3, 3)).T + np.identity(3)

    typ = "<U32"
    imu = load_csv_unsafe(paths.imu, typ)  # type: ignore

    new_imu = np.array(
        [
            np.concatenate(
                (
                    [t],
                    (gm @ np.array([wx, wy, wz]).astype(float) - gb).astype(typ),
                    (am @ np.array([ax, ay, az]).astype(float) - ab).astype(typ),
                )
            )
            for t, wx, wy, wz, ax, ay, az in imu
        ]
    )

    kwargs = {"delimiter": ",", "fmt": "%s", "comments": "#", "newline": "\r\n"}
    header = "timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
    np.savetxt(output_fn, new_imu, header=header, **kwargs)


def apply_transform(args: Namespace):
    input_csv = args.input_csv
    output_csv = args.output_csv
    R_i_c0 = Rotation.from_quat([args.qx, args.qy, args.qz, args.qw]).as_matrix()
    t_i_c0 = np.array([args.px, args.py, args.pz])
    T_i_c0 = np.eye(4)
    T_i_c0[:3, :3] = R_i_c0
    T_i_c0[:3, 3] = t_i_c0

    with (
        open(input_csv, "r", encoding="utf-8") as infile,
        open(output_csv, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if line.startswith("#"):
                outfile.write(line)
                continue
            ts, px, py, pz, qw, qx, qy, qz = line.strip().split(",")[:8]
            pos = np.array([float(px), float(py), float(pz)])
            rot = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
            T_w_c0 = np.eye(4)
            T_w_c0[:3, :3] = rot
            T_w_c0[:3, 3] = pos
            T_w_i = T_w_c0 @ np.linalg.inv(T_i_c0)
            px, py, pz = T_w_i[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(T_w_i[:3, :3]).as_quat()

            processed_line = f"{ts},{px},{py},{pz},{qw},{qx},{qy},{qz}\n"
            outfile.write(processed_line)


def cam_offset_ts(args: Namespace):
    paths = SensorPaths(args.dataset_path)
    offset = args.offset

    typ = "<U32"  # 32 unicode characters should be enough for all CSVs
    cams = [load_csv_unsafe(cam_path, typ) for cam_path in paths.cams]  # type: ignore
    cams_extras = [load_csv_unsafe(cam_path.with_stem("data.extra"), typ) for cam_path in paths.cams]  # type: ignore

    for cam in cams:
        cam[:, 0] = [str(int(i) + offset) for i in cam[:, 0]]

    for i, cam in enumerate(cams_extras):
        aligned_tss = np.array([str(int(i) + offset) for i in cam[:, 0]])
        aligned_tss = aligned_tss.reshape(-1, 1)
        cams_extras[i] = np.append(aligned_tss, cam, axis=1)

    kwargs = {"delimiter": ",", "fmt": "%s", "comments": "#", "newline": "\r\n"}
    for cam_path, cam in zip(paths.cams, cams):
        header = "timestamp [ns],filename"
        np.savetxt(cam_path, cam, header=header, **kwargs)

    for cam_path, cam_extra in zip(paths.cams, cams_extras):
        header = "aligned timestamp [ns],timestamp [ns],host timestamp [ns]"
        np.savetxt(cam_path.with_stem("data.extra"), cam_extra, header=header, **kwargs)


def imu2cam_ts(args: Namespace):
    paths = SensorPaths(args.dataset_path)

    # Load CSVs
    typ = "<U32"  # 32 unicode characters should be enough for all CSVs
    imu = load_csv_unsafe(paths.imu, typ)  # type: ignore
    imu_extras = load_csv_unsafe(paths.imu.with_stem("data.extra"), typ)  # type: ignore
    # gt = load_csv_unsafe(paths.gt, typ)  # type: ignore
    # cams = [load_csv_unsafe(cam_path, typ) for cam_path in paths.cams]  # type: ignore
    # Perform transformations

    # Edit all IMU timestamps
    # offset += int(imu[-1][0]) - int(cams[0][-1][0])
    offset = int(imu[0, 0]) - int(imu_extras[0, 1])  # Offset w.r.t. first arrival time
    imu[:, 0] = [str(int(i) - offset) for i in imu[:, 0]]

    # Save transformed files
    kwargs = {
        "delimiter": ",",
        "fmt": "%s",
        "header": "timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]",
        "comments": "#",
    }
    np.savetxt(paths.imu.with_stem("data_changed"), imu, **kwargs)


def preview_video(args: Namespace):
    paths = SensorPaths(args.dataset_path)
    duration = args.duration
    speedup = args.speedup
    scale = args.scale
    rotate_90cw = args.rotate_90cw
    rotate_90ccw = args.rotate_90ccw
    fps = args.fps
    output_fn = args.output_path
    tmp_dir = args.tmp_dir
    cam_idx = args.cam_idx

    no_conflict = (duration is None or speedup is None) and duration != speedup
    assert no_conflict, f"Conflicting options provided: {duration=}, {speedup=}"

    input_dir = paths.cams[cam_idx].parent / "data"
    frames = sorted([f for f in input_dir.iterdir() if f.suffix])

    if speedup and not duration:
        full_duration = get_duration_of(paths.cams[cam_idx])
        duration = full_duration / speedup
    else:
        raise ValueError(f"Provide exactly one of --{speedup=} or --{duration=}")

    used_frames = int(duration * fps)
    step = len(frames) / used_frames

    selected_frames = frames[:: round(step)]

    temp_dir = Path(tmp_dir)
    temp_dir.mkdir()
    symlinked_frames = [temp_dir / f.name for f in selected_frames]
    for f, l in zip(selected_frames, symlinked_frames):
        l.symlink_to(f)

    rotcmd = "transpose=1" if rotate_90cw else "transpose=2" if rotate_90ccw else ""

    # Small, half size, high compression, good for web (used in dataset preview)
    cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i {temp_dir}/*.png -vf scale=iw*{scale}:ih*{scale},{rotcmd} -c:v libvpx-vp9 -b:v 0.6M -crf 50 -pix_fmt yuv420p {output_fn.with_suffix('.webm')}"

    # Super fast, almost no compression, very good visual quality (used in trailer video)
    # cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i {temp_dir}/*.png -vf scale=iw*{scale}:ih*{scale},{rotcmd} -c:v libx264 -crf 0 -preset ultrafast -pix_fmt yuv420p {output_fn.with_suffix('.mp4')}"

    # Other commands:
    # cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i {temp_dir}/*.png -c:v libx264 -pix_fmt yuv420p -crf 38 {output_fn.with_suffix('.mp4')}"
    # cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i {temp_dir}/*.png -c:v libvpx-vp9 -b:v 0.6M -crf 50 -pix_fmt yuv420p {output_fn.with_suffix('.webm')}"
    # cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i {temp_dir}/*.png -c:v libx264 -pix_fmt yuv420p {output_fn.with_suffix('.mp4')}"

    print(cmd)
    subprocess.run(cmd.split(), check=True)

    for l in symlinked_frames:
        l.unlink()
    temp_dir.rmdir()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
