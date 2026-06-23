from __future__ import annotations

import argparse
from pathlib import Path

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit

from filter import FilterConfig
from predict import PredictionType
from replay import replay_run
from tracking import get_sanitized_trajectories

BASELINE = "dead_reckoning"

DEFAULT_PREDICTIONS: list[tuple[str, PredictionType]] = [
    ("none", PredictionType.NONE),
    ("pose_only", PredictionType.POSE_ONLY),
    ("gyro", PredictionType.GYRO),
    ("accel_gyro", PredictionType.ACCEL_GYRO),
    (BASELINE, PredictionType.DEAD_RECKONING),
]


def score(est_csv: Path, gt_csv: Path) -> tuple[float, float]:
    traj_est, traj_ref = get_sanitized_trajectories(est_csv, gt_csv, silence=True)
    ape = main_ape.ape(
        traj_ref=traj_ref,
        traj_est=traj_est,
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=False,
    )
    rpe = main_rpe.rpe(
        traj_ref=traj_ref,
        traj_est=traj_est,
        pose_relation=PoseRelation.translation_part,
        delta=6,
        delta_unit=Unit.frames,
        rel_delta_tol=0.1,
        all_pairs=False,
        align=True,
        correct_scale=False,
        support_loop=False,
    )
    return float(ape.stats["rmse"]), float(rpe.stats["rmse"])


def run_experiment(
    timing_dir,
    dataset_dir,
    out_dir,
    predictions: list[tuple[str, PredictionType]] | None = None,
    filter_config: FilterConfig | None = None,
    slam_dir=None,
) -> dict:
    # slam_dir is None for a full replay (flavor A: SLAM/IMU live in timing_dir);
    # for WMR timing injection (flavor B) it points at the dataset replay that
    # carries the SLAM trajectory + IMU, while timing_dir carries the WMR timing.
    timing_dir = Path(timing_dir)
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    gt = dataset_dir / "mav0" / "gt" / "data.csv"

    predictions = predictions or DEFAULT_PREDICTIONS
    fc = filter_config or FilterConfig(use_one_euro_filter=True)

    results: dict = {}
    for name, pred_type in predictions:
        res = replay_run(
            timing_dir, out_dir / name,
            pred_type=pred_type, filter_config=fc, dataset_dir=dataset_dir,
            slam_dir=slam_dir,
        )
        results[name] = {
            "prediction": score(res.prediction_path, gt),
            "filtering": score(res.filtering_path, gt),
            "n_predicted": res.n_predicted,
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timing-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = run_experiment(args.timing_dir, args.dataset, args.out)

    print(f"{'method':>16} {'pred ATE':>10} {'pred RTE':>10} {'filt ATE':>10} {'filt RTE':>10}")
    print("-" * 62)
    for name in res:
        ap_, rp = res[name]["prediction"]
        af, rf = res[name]["filtering"]
        tag = "  <- baseline" if name == BASELINE else ""
        print(f"{name:>16} {ap_:>10.4f} {rp:>10.4f} {af:>10.4f} {rf:>10.4f}{tag}")

    # Best prediction RTE vs. the baseline.
    base_rte = res[BASELINE]["prediction"][1]
    best = min(res, key=lambda n: res[n]["prediction"][1])
    best_rte = res[best]["prediction"][1]
    print(
        f"\nbest prediction RTE: {best} ({best_rte:.4f})  vs baseline "
        f"{BASELINE} ({base_rte:.4f})  ->  dRTE={best_rte - base_rte:+.4f}"
    )


if __name__ == "__main__":
    main()
