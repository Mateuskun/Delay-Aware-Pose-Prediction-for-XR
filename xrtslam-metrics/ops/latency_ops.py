#!/usr/bin/env python

"""Latency / timing-injection harness operations.

Thin CLI in the `ops/` house style over the existing harness modules
(`align_timelines.py`, `replay.py`, `experiment.py`); it adds no logic of its own.

```bash
# 1) Move a recorded run's CSVs into the dataset clock domain (point 2).
$xrtmet/ops/latency_ops.py move-timeline --timing-dir timing \
    --dataset MOO01_hand_puncher_1 --out timing/aligned

# 2) Regenerate one pose per when_ns (predict + filter) with a chosen method.
$xrtmet/ops/latency_ops.py replay --timing-dir timing \
    --dataset MOO01_hand_puncher_1 --out timing/replay --method dead_reckoning

# 3) Sweep the prediction methods, scored ATE/RTE against GT.
$xrtmet/ops/latency_ops.py experiment --timing-dir timing \
    --dataset MOO01_hand_puncher_1 --out timing/experiment
```
"""

import os
import sys
import subprocess
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict import PredictionType  # noqa: E402
from replay import replay_run  # noqa: E402
from experiment import BASELINE, run_experiment  # noqa: E402

_METHODS = ["none", "pose_only", "gyro", "accel_gyro", "dead_reckoning"]


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # List of operations for this script

    # fmt: off
    cmd_move_timeline = Command("move-timeline", "Move a recorded run's timing CSVs into the dataset clock domain", move_timeline)
    cmd_replay = Command("replay", "Regenerate one pose per when_ns (predict + filter), choosing the prediction method", replay)
    cmd_experiment = Command("experiment", "Sweep the prediction methods and score ATE/RTE against dataset GT", experiment)
    # fmt: on

    parser = ArgumentParser(
        description="Offline latency / timing-injection harness operations",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers(help="What operation to perform")

    # move-timeline
    sp = subparsers.add_parser(cmd_move_timeline.name, help=cmd_move_timeline.desc)
    sp.set_defaults(func=cmd_move_timeline.func)
    sp.add_argument("--timing-dir", type=Path, required=True, help="dir with the recorded CSVs")
    sp.add_argument("--dataset", type=Path, required=True, help="EuRoC dataset dir (provides the clock anchor + GT)")
    sp.add_argument("--out", type=Path, required=True, help="output dir for the aligned CSVs")

    # replay
    sp = subparsers.add_parser(cmd_replay.name, help=cmd_replay.desc)
    sp.set_defaults(func=cmd_replay.func)
    sp.add_argument("--timing-dir", type=Path, required=True, help="dir with the recorded CSVs")
    sp.add_argument("--dataset", type=Path, required=True, help="EuRoC dataset dir (clock anchor + GT)")
    sp.add_argument("--out", type=Path, required=True, help="output dir for tracking/prediction/filtering.csv")
    sp.add_argument("--method", choices=_METHODS, default="dead_reckoning",
                    help="prediction method to use (predict to the recorded when_ns)")

    # experiment
    sp = subparsers.add_parser(cmd_experiment.name, help=cmd_experiment.desc)
    sp.set_defaults(func=cmd_experiment.func)
    sp.add_argument("--timing-dir", type=Path, required=True, help="dir with the recorded CSVs")
    sp.add_argument("--dataset", type=Path, required=True, help="EuRoC dataset dir (clock anchor + GT)")
    sp.add_argument("--out", type=Path, required=True, help="output dir for the per-method replay CSVs")

    return parser.parse_args()


def move_timeline(args: Namespace):
    """Move a recorded run's CSVs into the dataset clock domain.

    Thin wrapper over align_timelines.py (kept as the single source for the
    file-transform), so this re-uses it verbatim rather than duplicating it.
    """
    script = Path(__file__).resolve().parent.parent / "align_timelines.py"
    subprocess.run(
        [sys.executable, str(script),
         "--timing-dir", str(args.timing_dir),
         "--dataset", str(args.dataset),
         "--out", str(args.out)],
        check=True,
    )


def replay(args: Namespace):
    """Regenerate tracking/prediction/filtering.csv, one pose per when_ns."""
    pred_type = PredictionType[args.method.upper()]
    res = replay_run(args.timing_dir, args.out, pred_type=pred_type, dataset_dir=args.dataset)
    print(f"[{args.method}] wrote {res.n_predicted} poses ({res.n_skipped} skipped) to {args.out}")
    print(f"  {res.tracking_path}")
    print(f"  {res.prediction_path}")
    print(f"  {res.filtering_path}")


def experiment(args: Namespace):
    """Sweep the prediction methods and print ATE/RTE against the dataset GT."""
    res = run_experiment(args.timing_dir, args.dataset, args.out)
    print(f"{'method':>16} {'pred ATE':>10} {'pred RTE':>10} {'filt ATE':>10} {'filt RTE':>10}")
    print("-" * 62)
    for name in res:
        ap_, rp = res[name]["prediction"]
        af, rf = res[name]["filtering"]
        tag = "  <- baseline" if name == BASELINE else ""
        print(f"{name:>16} {ap_:>10.4f} {rp:>10.4f} {af:>10.4f} {rf:>10.4f}{tag}")


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
