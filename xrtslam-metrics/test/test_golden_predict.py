from __future__ import annotations

import bisect
import csv
import os
from pathlib import Path

import numpy as np

from math3d import Pose, Quat, SpaceRelation, vec3
from predict import ImuHistory, PredictionType, RelationHistory, predict_pose

GRAVITY_CORRECTION = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

POS_MEAN_TOL_M = 5e-3
ROT_MEAN_TOL_DEG = 0.2

POS_P99_TOL_M = 2e-2
ROT_P99_TOL_DEG = 0.5


def find_timing_dir() -> Path | None:
    env = os.environ.get("MONADO_TIMING_DIR")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    candidate = Path(__file__).resolve().parent.parent.parent / "timing"
    return candidate if candidate.is_dir() else None


def _iter_rows(path: Path):
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            yield row


def load_prediction(path: Path) -> list[tuple[int, Pose, bool]]:
    out: list[tuple[int, Pose, bool]] = []
    for row in _iter_rows(path):
        ts = int(float(row[0]))
        pos = vec3(float(row[1]), float(row[2]), float(row[3]))
        qw, qx, qy, qz = (float(row[4]), float(row[5]), float(row[6]), float(row[7]))
        valid = abs(qw * qw + qx * qx + qy * qy + qz * qz - 1.0) <= 0.1
        out.append((ts, Pose(pos, Quat(qx, qy, qz, qw).normalized()), valid))
    return out


def load_slam_relations(path: Path) -> list[tuple[int, SpaceRelation]]:
    out: list[tuple[int, SpaceRelation]] = []
    for row in _iter_rows(path):
        ts = int(float(row[0]))
        rel = SpaceRelation()
        rel.pose.position = vec3(float(row[1]), float(row[2]), float(row[3]))
        rel.pose.orientation = Quat(
            float(row[5]), float(row[6]), float(row[7]), float(row[4])
        ).normalized()
        rel.linear_velocity = vec3(float(row[8]), float(row[9]), float(row[10]))
        rel.angular_velocity = vec3(float(row[11]), float(row[12]), float(row[13]))
        rel.relation_flags = 0x3F
        out.append((ts, rel))
    return out


def load_imu(path: Path) -> ImuHistory:
    imu = ImuHistory()
    for row in _iter_rows(path):
        ts = int(float(row[0]))
        accel = vec3(float(row[1]), float(row[2]), float(row[3]))
        gyro = vec3(float(row[4]), float(row[5]), float(row[6]))
        imu.push(ts, gyro, accel)
    return imu


def _pose_pair_errors(a: Pose, b: Pose) -> tuple[float, float]:
    pos = float(np.linalg.norm(a.position - b.position))
    d = min(1.0, abs(a.orientation.dot(b.orientation)))
    rot_deg = np.degrees(2.0 * np.arccos(d))
    return pos, float(rot_deg)


def run_golden_predict(timing_dir: Path) -> dict:
    pred = load_prediction(timing_dir / "prediction.csv")
    rels = load_slam_relations(timing_dir / "slam_relations.csv")
    imu_all = load_imu(timing_dir / "imu.csv")

    pre_anchor_cutoff_idx = 0
    for i in range(1, len(pred)):
        if pred[i][0] + 10**12 < pred[i - 1][0]:
            pre_anchor_cutoff_idx = i

    history = RelationHistory()
    next_rel = 0
    pos_errs: list[float] = []
    rot_errs: list[float] = []
    skipped = 0
    pre_anchor_skipped = 0
    invalid_skipped = 0
    for i, (when_ns, expected_pose, valid) in enumerate(pred):
        if i < pre_anchor_cutoff_idx:
            pre_anchor_skipped += 1
            continue
        if not valid:
            invalid_skipped += 1
            continue
        while next_rel < len(rels) and rels[next_rel][0] <= when_ns:
            history.push(rels[next_rel][1], rels[next_rel][0])
            next_rel += 1
        if len(history) == 0:
            skipped += 1
            continue

        out = predict_pose(
            history,
            when_ns,
            pred_type=PredictionType.DEAD_RECKONING,
            imu=imu_all,
            gravity_correction=GRAVITY_CORRECTION,
        )
        pe, re = _pose_pair_errors(out.pose, expected_pose)
        pos_errs.append(pe)
        rot_errs.append(re)

    pos = np.array(pos_errs)
    rot = np.array(rot_errs)
    return {
        "count": len(pos),
        "skipped": skipped,
        "pre_anchor_skipped": pre_anchor_skipped,
        "invalid_skipped": invalid_skipped,
        "pos_max_m": float(pos.max()) if pos.size else 0.0,
        "pos_mean_m": float(pos.mean()) if pos.size else 0.0,
        "pos_p99_m": float(np.percentile(pos, 99)) if pos.size else 0.0,
        "rot_max_deg": float(rot.max()) if rot.size else 0.0,
        "rot_mean_deg": float(rot.mean()) if rot.size else 0.0,
        "rot_p99_deg": float(np.percentile(rot, 99)) if rot.size else 0.0,
    }


def test_golden_predict_dead_reckoning():
    timing_dir = find_timing_dir()
    if timing_dir is None:
        print("SKIP test_golden_predict_dead_reckoning: no timing/ directory found")
        return
    for name in ("prediction.csv", "slam_relations.csv", "imu.csv"):
        if not (timing_dir / name).is_file():
            print(f"SKIP test_golden_predict_dead_reckoning: missing {name}")
            return

    m = run_golden_predict(timing_dir)
    print(
        f"golden predict: n={m['count']} skipped={m['skipped']} "
        f"pre_anchor_skipped={m['pre_anchor_skipped']} "
        f"invalid_skipped={m['invalid_skipped']}\n"
        f"  pos  mean={m['pos_mean_m'] * 1000:.4f}mm  p99={m['pos_p99_m'] * 1000:.4f}mm  max={m['pos_max_m'] * 1000:.4f}mm\n"
        f"  rot  mean={m['rot_mean_deg']:.4f}deg  p99={m['rot_p99_deg']:.4f}deg  max={m['rot_max_deg']:.4f}deg"
    )
    assert m["pos_mean_m"] < POS_MEAN_TOL_M, (
        f"mean position diff {m['pos_mean_m'] * 1000:.4f}mm exceeds {POS_MEAN_TOL_M * 1000:.4f}mm"
    )
    assert m["pos_p99_m"] < POS_P99_TOL_M, (
        f"p99 position diff {m['pos_p99_m'] * 1000:.4f}mm exceeds {POS_P99_TOL_M * 1000:.4f}mm"
    )
    assert m["rot_mean_deg"] < ROT_MEAN_TOL_DEG, (
        f"mean rotation diff {m['rot_mean_deg']:.4f}deg exceeds {ROT_MEAN_TOL_DEG:.4f}deg"
    )
    assert m["rot_p99_deg"] < ROT_P99_TOL_DEG, (
        f"p99 rotation diff {m['rot_p99_deg']:.4f}deg exceeds {ROT_P99_TOL_DEG:.4f}deg"
    )


if __name__ == "__main__":
    test_golden_predict_dead_reckoning()
