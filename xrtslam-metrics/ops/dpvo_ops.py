#!/usr/bin/env python

import os
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable
from decimal import Decimal
from euroc_ops import SensorPaths


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # List of operations for this script

    # fmt: off
    cmd_euroc2tumrgbd = Command("euroc2tumrgbd", "Convert an EuRoC dataset into TUM RGBD format used by DPVO", euroc2tumrgbd)
    cmd_traj_dpvo2euroc = Command("traj_dpvo2euroc", "Convert a DPVO trajectory to euroc format", traj_dpvo2euroc)
    # fmt: on

    parser = ArgumentParser(
        description="Helper commands to convert between DROID SLAM and EuRoC formats",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_euroc2tumrgbd.name, help=cmd_euroc2tumrgbd.desc)
    subparser.set_defaults(func=cmd_euroc2tumrgbd.func)
    subparser.add_argument(
        "dataset_path",
        type=Path,
        help="EuRoC dataset path to adapt to TUM RGB-D format",
    )

    subparser = subparsers.add_parser(cmd_traj_dpvo2euroc.name, help=cmd_traj_dpvo2euroc.desc)
    subparser.set_defaults(func=cmd_traj_dpvo2euroc.func)
    subparser.add_argument(
        "input_txt",
        type=Path,
        help="trajectory .txt file from DPVO",
    )
    subparser.add_argument(
        "output_csv",
        type=Path,
        help="output .csv file in euroc format",
    )

    return parser.parse_args()


def euroc2tumrgbd(args: Namespace):
    "Adapts an EuRoC dataset to the TUM RGB-D format accepted"

    dataset_path: Path = args.dataset_path
    sensor_paths = SensorPaths(dataset_path)

    cam0_data = sensor_paths.cams[0].parent / "data"
    rgb_dir = dataset_path / "rgb"

    os.system(f"cp -r {cam0_data} {rgb_dir}")

    rgb_txt = rgb_dir.parent / "rgb.txt"
    with open(rgb_txt, "w") as f:
        f.write("# color images\n")
        f.write(f"# file: '{dataset_path.name}'\n")
        f.write("# timestamp filename\n")
        for img in rgb_dir.iterdir():
            ns = int(img.stem)
            s = f"{ns / 1e9:.9f}"
            img.rename(rgb_dir / f"{s}.png")
            f.write(f"{s} rgb/{s}.png\n")


def traj_dpvo2euroc(args: Namespace):
    """
    Convert a DPVO trajectory .txt file into an euroc .csv

    From: ts[s] px py pz qx qy qz qw
    To: ts[ns] px py pz qw qx qy qz
    """

    input_txt = args.input_txt
    output_csv = args.output_csv

    with open(input_txt, "r") as infile, open(output_csv, "w") as outfile:
        outfile.write("#ts[ns],px[m],py[m],pz[m],qw[],qx[],qy[],qz[]\n")
        for line in infile:
            if line.startswith("#"):
                continue
            ts, px, py, pz, qx, qy, qz, qw = line.strip().split(" ")[:8]
            ts = f"{Decimal(ts) * Decimal('1e9'):.0f}"
            processed_line = f"{ts},{px},{py},{pz},{qw},{qx},{qy},{qz}\n"
            outfile.write(processed_line)


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
