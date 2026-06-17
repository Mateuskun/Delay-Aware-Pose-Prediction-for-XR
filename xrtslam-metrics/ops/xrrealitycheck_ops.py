#!/usr/bin/env python

import sys

from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable
from decimal import Decimal
from argparse import RawTextHelpFormatter

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_xrrc2euroc = Command("xrrc2euroc", "Convert a \"XR Reality Check\" dataset into euroc format", xrrc2euroc)
    # fmt: on

    parser = ArgumentParser(
        description="""
        Helper commands to convert between XR Reality Check and EuRoC formats.

        - Project page: https://github.com/Duke-I3T-Lab/XR_Tracking_Evaluation
        - Paper: "XR Reality Check: What Commercial Devices Deliver for Spatial Tracking"
        - ArXiV: https://www.arxiv.org/abs/2508.08642
        """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_xrrc2euroc.name, help=cmd_xrrc2euroc.desc)
    subparser.set_defaults(func=cmd_xrrc2euroc.func)
    subparser.add_argument(
        "in_xrrc_path",
        type=Path,
        help="Path to the XR Reality Check dataset",
    )
    subparser.add_argument(
        "out_euroc_path",
        type=Path,
        help="Path to save the converted Euroc dataset",
    )

    return parser.parse_args()


def xrrc2euroc(args: Namespace):
    "Convert a XR Reality Check dataset into euroc trajectory format"

    xrrc_path: Path = args.in_xrrc_path
    euroc_path: Path = args.out_euroc_path

    # Create paths

    mav0_path = euroc_path / "mav0"
    cam0_path = mav0_path / "cam0" / "data"
    cam1_path = mav0_path / "cam1" / "data"
    imu0_path = mav0_path / "imu0"
    gt_path = mav0_path / "state_groundtruth_estimate0"

    euroc_path.mkdir(parents=True)
    mav0_path.mkdir(parents=True)
    cam0_path.mkdir(parents=True)
    cam1_path.mkdir(parents=True)
    imu0_path.mkdir(parents=True)
    gt_path.mkdir(parents=True)

    # Copy images

    for cam in ["cam0", "cam1"]:
        xrrc_cam_path = xrrc_path / "data" / "sensor" / cam
        euroc_cam_path = mav0_path / cam / "data"
        for img_file in xrrc_cam_path.glob("*.png"):
            ts = f"{Decimal(img_file.stem) * Decimal('1e9'):.0f}"
            dest_file = euroc_cam_path / f"{ts}.png"
            dest_file.write_bytes(img_file.read_bytes())

    # Convert cam csvs

    for cam in ["cam0", "cam1"]:
        xrrc_cam_csv = xrrc_path / "data" / "sensor" / f"{cam}.csv"
        # eg row: 1750031353.247485 ./data/cam0/1750031353.247485.png
        euroc_cam_csv = mav0_path / cam / "data.csv"
        # eg row: 1750031353247485000,1750031353.247485.png
        with open(xrrc_cam_csv, "r", encoding="utf-8") as infile, open(euroc_cam_csv, "w", encoding="utf-8") as outfile:
            outfile.write("# ts[ns], filename\n")
            for line in infile:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                ts_s, _ = line.split(" ")
                ts = f"{Decimal(ts_s) * Decimal('1e9'):.0f}"
                filename = f"{ts}.png"
                outfile.write(f"{ts},{filename}\n")

    # Convert IMU csv

    xrrc_imu_csv = xrrc_path / "data" / "sensor" / "imu" / "data.csv"
    # eg row: 1750031353.244186,0.104720,1.645845,0.162316,-2.020170,-9.659550,-2.304563
    euroc_imu_csv = imu0_path / "data.csv"
    # eg row: 1750031353244186000,0.104720,1.645845,0.162316,-2.020170,-9.659550,-2.304563
    with open(xrrc_imu_csv, "r", encoding="utf-8") as infile, open(euroc_imu_csv, "w", encoding="utf-8") as outfile:
        outfile.write("# ts[ns], wx[rad/s], wy[rad/s], wz[rad/s], ax[m/s^2], ay[m/s^2], az[m/s^2]\n")
        for line in infile:
            if line.startswith("#") or line.strip() == "":
                continue
            ts_s, wx, wy, wz, ax, ay, az = line[:-1].split(",")
            ts = f"{Decimal(ts_s) * Decimal('1e9'):.0f}"
            outfile.write(f"{ts},{wx},{wy},{wz},{ax},{ay},{az}\n")

    # Convert GT csv
    xrrc_gt_csv = xrrc_path / "data" / "gt" / "gt_ORB.csv"
    if not xrrc_gt_csv.exists():
        print(f"Warning: skipping groundtruth file since it does not exist: {xrrc_gt_csv}")
        gt_path.rmdir()
        return
    # eg row: 1.750031353203660965e+09 -3.613524620811395249e-01 -3.838964371767826511e-02 1.607203256083865828e+00 -4.576607623047912465e-01 5.741413967195462265e-01 -4.994074540736170853e-01 4.598918112287211368e-01
    euroc_gt_csv = gt_path / "data.csv"
    # eg row: 1750031353203660965,-0.3613524620811395249,-0.03838964371767826511,1.6072032560838658,-0.4576607623047912465,0.5741413967195462265,-0.4994074540736170853,0.4598918112287211368
    with open(xrrc_gt_csv, "r", encoding="utf-8") as infile, open(euroc_gt_csv, "w", encoding="utf-8") as outfile:
        outfile.write("# ts[ns], px[m], py[m], pz[m], qw[], qx[], qy[] ,qz[]\n")
        for line in infile:
            ts_s, px, py, pz, qx, qy, qz, qw = line[:-1].split(" ")
            ts = f"{Decimal(ts_s) * Decimal('1e9'):.0f}"
            outfile.write(f"{ts},{px},{py},{pz},{qw},{qx},{qy},{qz}\n")


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
