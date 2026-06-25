from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from csvio import SlamSource, dataset_clock_offset, load_display_events
from filter import FilterConfig, PoseFilter
from math3d import SpaceRelation
from predict import PredictionType, RelationHistory, predict_pose

GRAVITY_CORRECTION = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

_POSE_HEADER = (
    "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
    "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n"
)


def write_pose_csv(path: Path, rows: list[tuple[int, SpaceRelation]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(_POSE_HEADER)
        for ts, rel in rows:
            p = rel.pose.position
            q = rel.pose.orientation  # Quat is (x, y, z, w)
            f.write(
                f"{ts},{p[0]:.10f},{p[1]:.10f},{p[2]:.10f},"
                f"{q.w:.10f},{q.x:.10f},{q.y:.10f},{q.z:.10f}\n"
            )


@dataclass
class ReplayResult:
    tracking_path: Path
    prediction_path: Path
    filtering_path: Path
    n_predicted: int
    n_skipped: int


def replay_run(
    timing_dir: Path,
    out_dir: Path,
    pred_type: PredictionType = PredictionType.DEAD_RECKONING,
    filter_config: FilterConfig | None = None,
    dataset_dir: Path | None = None,
    slam_dir: Path | None = None,
    slam_source: SlamSource | None = None,
) -> ReplayResult:
    # timing_dir provides the display-path timing (display.csv + camera.csv anchor).
    # The SLAM trajectory + IMU come from either slam_dir (a Monado run:
    # slam_relations.csv + accel-first imu.csv, host clock → dataset_clock_offset)
    # or an explicit slam_source (e.g. a patched basalt_vio run, already in the
    # dataset clock; see csvio.basalt_slam_source). For a full Monado replay
    # (flavor A) slam_dir == timing_dir; for timing injection (flavor B) timing_dir
    # is the WMR/StereoKit run and the SLAM comes from the dataset run / Basalt.
    # Each source is rebased onto the dataset cam0 clock by its own offset, so the
    # two timelines join consistently.
    timing_dir = Path(timing_dir)
    out_dir = Path(out_dir)

    if dataset_dir is not None:
        display_offset = dataset_clock_offset(timing_dir, dataset_dir)
    else:
        display_offset = 0

    if slam_source is None:
        slam_dir = Path(slam_dir) if slam_dir is not None else timing_dir
        slam_offset = (
            dataset_clock_offset(slam_dir, dataset_dir) if dataset_dir is not None else 0
        )
        slam_source = SlamSource(
            relations_path=slam_dir / "slam_relations.csv",
            imu_path=slam_dir / "imu.csv",
            imu_euroc=False,
            clock_offset=slam_offset,
        )

    events = load_display_events(timing_dir / "display.csv", ts_offset=display_offset)
    rels = slam_source.load_relations()
    imu = slam_source.load_imu()

    history = RelationHistory()
    posefilter = PoseFilter(filter_config or FilterConfig(use_one_euro_filter=True))
    next_rel = 0

    tracking_rows: list[tuple[int, SpaceRelation]] = [(ts, r) for ts, r in rels]
    pred_rows: list[tuple[int, SpaceRelation]] = []
    filt_rows: list[tuple[int, SpaceRelation]] = []
    n_skipped = 0

    for ev in events:
        if ev.locate_views is None:
            n_skipped += 1
            continue
        while next_rel < len(rels) and rels[next_rel][0] <= ev.locate_views:
            ts, rel = rels[next_rel]
            history.push(rel, ts)
            next_rel += 1
        if len(history) == 0:
            n_skipped += 1
            continue

        when_ns = ev.display_time
        if when_ns is None:
            n_skipped += 1
            continue

        predicted = predict_pose(
            history,
            when_ns,
            pred_type=pred_type,
            imu=imu,
            gravity_correction=GRAVITY_CORRECTION,
        )
        filtered = posefilter.run(when_ns, predicted)

        pred_rows.append((when_ns, predicted))
        filt_rows.append((when_ns, filtered))

    tracking_path = out_dir / "tracking.csv"
    prediction_path = out_dir / "prediction.csv"
    filtering_path = out_dir / "filtering.csv"
    write_pose_csv(tracking_path, tracking_rows)
    write_pose_csv(prediction_path, pred_rows)
    write_pose_csv(filtering_path, filt_rows)

    return ReplayResult(
        tracking_path=tracking_path,
        prediction_path=prediction_path,
        filtering_path=filtering_path,
        n_predicted=len(pred_rows),
        n_skipped=n_skipped,
    )
