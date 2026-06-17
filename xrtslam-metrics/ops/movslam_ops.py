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
    cmd_euroc2tartan_gt0 = Command("euroc2tartan_gt", "Convert an EuRoC cam0 groundtruth poses into Tartan pose_left.txt style file", euroc2tartan_gt)
    # fmt: on

    parser = ArgumentParser(
        description="Helper commands to convert between DROID SLAM and EuRoC formats",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    subparser = subparsers.add_parser(cmd_euroc2tartan_gt0.name, help=cmd_euroc2tartan_gt0.desc)
    subparser.set_defaults(func=cmd_euroc2tartan_gt0.func)
    subparser.add_argument(
        "euroc_gt_csv",
        type=Path,
        help="EuRoC groundtruth csv to convert to Tartan format",
    )
    subparser.add_argument(
        "output_txt",
        type=Path,
        help="output .txt file in Tartan pose_left.txt format",
    )

    return parser.parse_args()


def euroc2tartan_gt(args: Namespace):
    """
    Adapts an EuRoC groundtruth csv to the Tartan pose_left.txt format

    Euroc groundtruth csv has this format:

    ```csv
    #timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]
    1403715888379057920,0.898029,2.028208,0.955711,0.051153,0.827881,-0.050831,0.556249,0.005925,-0.011939,-0.007347,-0.002341,0.021815,0.076602,-0.022808,0.177689,0.090354
    1403715888384058112,0.898059,2.028148,0.955674,0.051146,0.827886,-0.050832,0.556242,0.005799,-0.011837,-0.007330,-0.002341,0.021815,0.076602,-0.022808,0.177689,0.090354
    ...
    ```

    Tartan pose_left.txt format has this format (no timestamp, px py pz qx qy qz qw):

    ```txt
    1.279419517517089844e+01 -7.122835636138916016e+00 1.570048213005065918e+00 -1.611406058073043823e-01 2.350628972053527832e-01 1.773252189159393311e-01 9.419845342636108398e-01
    1.266237258911132812e+01 -7.102912425994873047e+00 1.490393638610839844e+00 -1.516564488410949707e-01 2.352153509855270386e-01 1.884286850690841675e-01 9.413653016090393066e-01
    ```
    """

    gt_csv: Path = args.euroc_gt_csv
    output_txt: Path = args.output_txt

    with open(gt_csv, "r") as infile, open(output_txt, "w") as outfile:
        for line in infile:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            ts, px, py, pz, qw, qx, qy, qz = parts[:8]
            processed_line = f"{px} {py} {pz} {qx} {qy} {qz} {qw}\n"
            outfile.write(processed_line)


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
