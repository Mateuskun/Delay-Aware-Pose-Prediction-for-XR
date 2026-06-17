from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from filter import FilterConfig, PoseFilter
from math3d import ORIENTATION_VALID, POSITION_VALID, Pose, Quat, SpaceRelation, vec3

POS_MEAN_TOL_M = 5e-4
POS_P99_TOL_M = 2e-3
ROT_MEAN_TOL_DEG = 0.05
ROT_P99_TOL_DEG = 0.2


def find_timing_dir() -> Path | None:
    env = os.environ.get("MONADO_TIMING_DIR")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    candidate = Path(__file__).resolve().parent.parent.parent / "timing"
    return candidate if candidate.is_dir() else None


def load_pose_csv(path: Path) -> list[tuple[int, Pose, bool]]:
    rows: list[tuple[int, Pose, bool]] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            ts = int(float(row[0]))
            pos = vec3(float(row[1]), float(row[2]), float(row[3]))
            qw, qx, qy, qz = (float(row[4]), float(row[5]), float(row[6]), float(row[7]))
            valid = abs(qw * qw + qx * qx + qy * qy + qz * qz - 1.0) <= 0.1
            rows.append((ts, Pose(pos, Quat(qx, qy, qz, qw).normalized()), valid))
    return rows


def _pose_pair_errors(a: Pose, b: Pose) -> tuple[float, float]:
    pos = float(np.linalg.norm(a.position - b.position))
    d = min(1.0, abs(a.orientation.dot(b.orientation)))
    rot_deg = np.degrees(2.0 * np.arccos(d))
    return pos, float(rot_deg)


def run_golden_filter(timing_dir: Path) -> dict | None:
    pred = load_pose_csv(timing_dir / "prediction.csv")
    filt = load_pose_csv(timing_dir / "filtering.csv")

    if len(pred) != len(filt):
        raise AssertionError(
            f"prediction.csv ({len(pred)}) and filtering.csv ({len(filt)}) "
            "have different row counts"
        )

    identical = all(
        np.allclose(p.position, fp.position, atol=1e-9)
        and np.allclose(p.orientation.as_xyzw(), fp.orientation.as_xyzw(), atol=1e-9)
        for (_, p, _), (_, fp, _) in zip(pred, filt)
    )
    if identical:
        return None

    pose_filter = PoseFilter(FilterConfig(use_one_euro_filter=True))

    pre_anchor_cutoff_idx = 0
    for i in range(1, len(pred)):
        if pred[i][0] + 10**12 < pred[i - 1][0]:
            pre_anchor_cutoff_idx = i

    pos_errs: list[float] = []
    rot_errs: list[float] = []
    for i, ((ts, predicted_pose, valid), (_, expected_pose, _)) in enumerate(zip(pred, filt)):
        if i < pre_anchor_cutoff_idx:
            continue
        if not valid:
            continue
        rel = SpaceRelation()
        rel.pose = predicted_pose.copy()
        rel.relation_flags = POSITION_VALID | ORIENTATION_VALID
        out = pose_filter.run(ts, rel)
        pe, re = _pose_pair_errors(out.pose, expected_pose)
        pos_errs.append(pe)
        rot_errs.append(re)

    pos = np.array(pos_errs)
    rot = np.array(rot_errs)
    return {
        "count": len(pos),
        "pos_max_m": float(pos.max()),
        "pos_mean_m": float(pos.mean()),
        "pos_p99_m": float(np.percentile(pos, 99)),
        "rot_max_deg": float(rot.max()),
        "rot_mean_deg": float(rot.mean()),
        "rot_p99_deg": float(np.percentile(rot, 99)),
    }


def test_golden_filter_one_euro():
    timing_dir = find_timing_dir()
    if timing_dir is None:
        print("SKIP test_golden_filter_one_euro: no timing/ directory found")
        return

    metrics = run_golden_filter(timing_dir)
    if metrics is None:
        print(
            "SKIP test_golden_filter_one_euro: filtering.csv == prediction.csv "
            "(filter was disabled in this run)"
        )
        return

    print(
        f"golden filter: n={metrics['count']}\n"
        f"  pos  mean={metrics['pos_mean_m'] * 1000:.4f}mm  p99={metrics['pos_p99_m'] * 1000:.4f}mm  max={metrics['pos_max_m'] * 1000:.4f}mm\n"
        f"  rot  mean={metrics['rot_mean_deg']:.4f}deg  p99={metrics['rot_p99_deg']:.4f}deg  max={metrics['rot_max_deg']:.4f}deg"
    )
    assert metrics["pos_mean_m"] < POS_MEAN_TOL_M, (
        f"mean position diff {metrics['pos_mean_m'] * 1000:.4f}mm exceeds "
        f"{POS_MEAN_TOL_M * 1000:.4f}mm"
    )
    assert metrics["pos_p99_m"] < POS_P99_TOL_M, (
        f"p99 position diff {metrics['pos_p99_m'] * 1000:.4f}mm exceeds "
        f"{POS_P99_TOL_M * 1000:.4f}mm"
    )
    assert metrics["rot_mean_deg"] < ROT_MEAN_TOL_DEG, (
        f"mean rotation diff {metrics['rot_mean_deg']:.4f}deg exceeds "
        f"{ROT_MEAN_TOL_DEG:.4f}deg"
    )
    assert metrics["rot_p99_deg"] < ROT_P99_TOL_DEG, (
        f"p99 rotation diff {metrics['rot_p99_deg']:.4f}deg exceeds "
        f"{ROT_P99_TOL_DEG:.4f}deg"
    )


if __name__ == "__main__":
    test_golden_filter_one_euro()
