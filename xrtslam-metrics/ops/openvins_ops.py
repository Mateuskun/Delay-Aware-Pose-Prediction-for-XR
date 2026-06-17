#!/usr/bin/env python

import sys

from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable
from decimal import Decimal

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_csv_unsafe


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_traj_ov2euroc = Command("traj_ov2euroc", "Convert a OpenVINS output trajectory into euroc trajectory format", traj_ov2euroc)
    # fmt: on

    parser = ArgumentParser(
        description="Helper commands to convert between OpenVins and EuRoC formats",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_traj_ov2euroc.name, help=cmd_traj_ov2euroc.desc)
    subparser.set_defaults(func=cmd_traj_ov2euroc.func)
    subparser.add_argument(
        "input_txt",
        type=Path,
        help="Path to the result.txt trajectory file produced by OpenVINS (ts[s] q[w x y z] p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans ... imu_model dw da tg wtoI atoI etc)",
    )
    subparser.add_argument(
        "output_csv",
        type=Path,
        help="Path to the output trajectory CSV file in euroc format (ts[ns], px, py, pz, qw, qx, qy, qz)",
    )

    return parser.parse_args()


def traj_ov2euroc(args: Namespace):
    """
    Convert a OpenVINS output trajectory into euroc trajectory format

    From: ts[s] q[w x y z] p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans ... imu_model dw da tg wtoI atoI etc
    To: ts[ns] px py pz qw qx qy qz
    """

    input_txt = args.input_txt
    output_csv = args.output_csv

    with open(input_txt, "r") as infile, open(output_csv, "w") as outfile:
        outfile.write("# ts[ns], px[m], py[m], pz[m], qw[], qx[], qy[] ,qz[]\n")
        for line in infile:
            if line.startswith("#"):
                continue
            ts, qw, qx, qy, qz, px, py, pz = line.split(" ")[:8]
            ts = f"{Decimal(ts) * Decimal('1e9'):.0f}"
            processed_line = f"{ts},{px},{py},{pz},{qw},{qx},{qy},{qz}\n"
            outfile.write(processed_line)


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
