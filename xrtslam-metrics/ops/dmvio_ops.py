#!/usr/bin/env python

import re
import sys

from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable
from decimal import Decimal

import sys
import os
import numpy as np

from euroc_ops import SensorPaths

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_csv_unsafe


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_traj_dm2euroc = Command("traj_dm2euroc", "Convert a dm-vio output trajectory into euroc trajectory format", traj_dm2euroc)
    cmd_euroc2dm_files = Command("euroc2dm_files", "Export interpolated imu.txt and times.txt files as required by dm-vio from an euroc dataset", euroc2dm_files)
    # fmt: on

    parser = ArgumentParser(
        description="Helper commands to convert between DM-VIO and EuRoC formats",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_traj_dm2euroc.name, help=cmd_traj_dm2euroc.desc)
    subparser.set_defaults(func=cmd_traj_dm2euroc.func)
    subparser.add_argument(
        "input_txt",
        type=Path,
        help="Path to the result.txt trajectory file produced by DM-VIO (ts [s], px, py, pz, qx, qy, qz, qw)",
    )
    subparser.add_argument(
        "output_csv",
        type=Path,
        help="Path to the output trajectory CSV file in euroc format (ts [ns], px, py, pz, qw, qx, qy, qz)",
    )

    subparser = subparsers.add_parser(cmd_euroc2dm_files.name, help=cmd_euroc2dm_files.desc)
    subparser.set_defaults(func=cmd_euroc2dm_files.func)
    subparser.add_argument(
        "dataset_path",
        type=Path,
        help="Dataset path (the path that contains the mav0 directory)",
    )
    subparser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the output directory where imu.txt and times.txt files will be saved",
    )

    return parser.parse_args()


def traj_dm2euroc(args: Namespace):
    """
    Perform the following regex:
    s/(.*)\\.(.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)/$1$2 $3 $4 $5 $9 $6 $7 $8/g
    on a an input file that has lines like: 123.456 1.05 0.01 0.03 1e-5 -3e-10 0.0 1.0
    """

    input_txt = args.input_txt
    output_csv = args.output_csv

    with open(input_txt, "r") as infile, open(output_csv, "w") as outfile:
        for line in infile:
            pattern = re.compile(r"(.*) (.*) (.*) (.*) (.*) (.*) (.*) (.*)")
            match = pattern.match(line)
            assert match, line
            ts, px, py, pz, qx, qy, qz, qw = match.groups()
            ts = f"{Decimal(ts) * Decimal('1e9'):.0f}"
            processed_line = f"{ts},{px},{py},{pz},{qw},{qx},{qy},{qz}"
            outfile.write(processed_line + "\n")


def interpolate_imu_file(imu_input_filename, times_input_filename, imu_output_filename):
    """
    Inserts interpolated IMU measurements at all timestamps of images.
    Function heavily inspired from: https://github.com/lukasvst/dm-vio-python-tools/blob/master/interpolate_imu_file.py
    """

    def offset_times(array, offset):
        for line in array:
            line[0] = str(int(line[0]) + offset)

    with open(imu_input_filename) as imu_input_file:
        imu_lines = imu_input_file.readlines()
        imu_lines = [line.rstrip("\n").split(" ") for line in imu_lines]
        imu_time0 = int(imu_lines[0][0])
        # We want to use np.interp but it cannot handle the long timestamps. So we subtract the timestamp of the first
        # imu data from all timestamps and will add them back later.
        offset_times(imu_lines, -imu_time0)
        imu_data = np.array(imu_lines, dtype=float)

    with open(times_input_filename) as times_input_file:
        times_lines = times_input_file.readlines()
        times_lines = [line.rstrip("\n").split(" ") for line in times_lines]
        offset_times(times_lines, -imu_time0)
        times_data = np.array(times_lines, dtype=float)

    image_times = times_data[:, 0]
    imu_times = imu_data[:, 0]
    min_imu_time = imu_data[0, 0]
    max_imu_time = imu_data[imu_data.shape[0] - 1, 0]

    filtered_times = image_times[np.logical_and(image_times <= max_imu_time, image_times >= min_imu_time)]

    all_times = np.concatenate((filtered_times, imu_times), axis=0)
    all_times.sort()

    interpolated = [np.interp(all_times, imu_times, imu_data[:, i + 1]) for i in range(6)]
    interpolated.insert(0, all_times)
    interpolated_stacked = np.stack(interpolated).transpose()

    with open(imu_output_filename, "w") as out_file:
        outlist = interpolated_stacked.tolist()
        offset_times(outlist, imu_time0)  # add back the offset.
        outlines = [
            (" ".join([elem if type(elem) is str else "{:f}".format(elem) for elem in line]) + "\n") for line in outlist
        ]
        out_file.writelines(outlines)


def euroc2dm_files(args: Namespace):
    paths = SensorPaths(args.dataset_path)
    output_dir = args.output_dir

    # Generate times.txt
    cam_path = paths.cams[0]
    times_txt_path = output_dir / "times.txt"
    with open(cam_path, "r") as cam_csv, open(times_txt_path, "w") as times_txt:
        # Convert lines of the form 'ts[ns],filename.png' to 'filename ts[s]'
        pattern = re.compile(r"(.*), ?(.*)\.png")
        for line in cam_csv:
            if line.startswith("#"):
                continue
            match = pattern.match(line)
            assert match, line
            ts, filename = match.groups()
            ts = ts[:-9] + "." + ts[-9:]
            times_txt.write(f"{filename} {ts}\n")

    # Generate imu_tmp.txt to then pass to final interpolated imu.txt
    imu_path = paths.imu
    imu_tmp_txt_path = output_dir / "imu_tmp.txt"
    with open(imu_path, "r") as imu_csv, open(imu_tmp_txt_path, "w") as imu_txt:
        pattern = re.compile(r"(.*), ?(.*), ?(.*), ?(.*), ?(.*), ?(.*), ?(.*)")
        for line in imu_csv:
            if line.startswith("#"):
                continue
            match = pattern.match(line)
            ts, wx, wy, wz, ax, ay, az = match.groups()
            imu_txt.write(f"{ts} {wx} {wy} {wz} {ax} {ay} {az}\n")

    # Generate imu.txt
    imu_txt_path = output_dir / "imu.txt"
    interpolate_imu_file(imu_tmp_txt_path, times_txt_path, imu_txt_path)

    # Delete imu_tmp.txt
    imu_tmp_txt_path.unlink()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
