# xrtslam-metrics/csvio.py
from __future__ import annotations

import csv as _csv
from dataclasses import dataclass
from pathlib import Path

from math3d import Pose, Quat, SpaceRelation, vec3
from predict import ImuHistory


@dataclass
class DisplayEvent:
    frame_id: int | None
    display_time: int | None
    wait_frame: int | None
    begin_frame: int | None
    locate_views: int | None
    predict_filter: int | None
    present: int | None


def _iter_rows(path: Path):
    with open(path, newline="") as f:
        for row in _csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            yield row


def _opt_int(s: str) -> int | None:
    s = s.strip()
    return int(s) if s != "" else None


def _preanchor_cutoff(seq: list[int | None]) -> int:
    cutoff = 0
    for i in range(1, len(seq)):
        a, b = seq[i - 1], seq[i]
        if a is not None and b is not None and b + 10**12 < a:
            cutoff = i
    return cutoff


def load_display_events(path: Path, ts_offset: int = 0) -> list[DisplayEvent]:
    # columns: display_time,wait_frame,begin_frame,locate_views,predict_filter,present,frame_id
    raw = [[_opt_int(x) for x in row] for row in _iter_rows(path)]
    raw = raw[_preanchor_cutoff([r[0] for r in raw]):]  # on display_time, file order

    def sh(v: int | None) -> int | None:
        return None if v is None else v + ts_offset

    out = [
        DisplayEvent(
            display_time=sh(r[0]),
            wait_frame=sh(r[1]),
            begin_frame=sh(r[2]),
            locate_views=sh(r[3]),
            predict_filter=sh(r[4]),
            present=sh(r[5]),
            frame_id=r[6],
        )
        for r in raw
    ]
    # Sort by locate_views; None last (stable for rows without a call time).
    out.sort(key=lambda e: (e.locate_views is None, e.locate_views or 0))
    return out


def load_pose_csv(path: Path) -> list[tuple[int, Pose, bool]]:
    out: list[tuple[int, Pose, bool]] = []
    for row in _iter_rows(path):
        ts = int(float(row[0]))
        pos = vec3(float(row[1]), float(row[2]), float(row[3]))
        qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
        valid = abs(qw * qw + qx * qx + qy * qy + qz * qz - 1.0) <= 0.1
        out.append((ts, Pose(pos, Quat(qx, qy, qz, qw).normalized()), valid))
    return out


def load_slam_relations(path: Path, ts_offset: int = 0) -> list[tuple[int, SpaceRelation]]:
    out: list[tuple[int, SpaceRelation]] = []
    for row in _iter_rows(path):
        ts = int(float(row[0])) + ts_offset
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


def load_imu(path: Path, ts_offset: int = 0) -> ImuHistory:
    # Monado imu.csv column order: ts, ax, ay, az, wx, wy, wz (accel-first).
    imu = ImuHistory()
    for row in _iter_rows(path):
        ts = int(float(row[0])) + ts_offset
        accel = vec3(float(row[1]), float(row[2]), float(row[3]))
        gyro = vec3(float(row[4]), float(row[5]), float(row[6]))
        imu.push(ts, gyro, accel)
    return imu


def load_imu_euroc(path: Path, ts_offset: int = 0) -> ImuHistory:
    # EuRoC mav0/imu0/data.csv column order: ts, wx, wy, wz, ax, ay, az (gyro-first).
    imu = ImuHistory()
    for row in _iter_rows(path):
        ts = int(float(row[0])) + ts_offset
        gyro = vec3(float(row[1]), float(row[2]), float(row[3]))
        accel = vec3(float(row[4]), float(row[5]), float(row[6]))
        imu.push(ts, gyro, accel)
    return imu


def first_camera_exposure_ns(camera_csv: Path) -> int:
    for row in _iter_rows(camera_csv):
        return int(float(row[0]))
    raise ValueError(f"no data rows in {camera_csv}")


def dataset_cam0_first_ns(dataset_dir: Path) -> int:
    cam0 = Path(dataset_dir) / "mav0" / "cam0" / "data.csv"
    for row in _iter_rows(cam0):
        return int(float(row[0]))
    raise ValueError(f"no data rows in {cam0}")


def dataset_clock_offset(timing_dir: Path, dataset_dir: Path) -> int:
    """ We use this to calculate t"""
    timing_dir = Path(timing_dir)
    return dataset_cam0_first_ns(dataset_dir) - first_camera_exposure_ns(timing_dir / "camera.csv")


@dataclass
class SlamSource:
    """Where the SLAM trajectory + IMU come from for a replay.

    Abstracts the difference between a Monado run dir (slam_relations.csv +
    accel-first imu.csv, host clock → needs a dataset_clock_offset) and a
    standalone Basalt run (slam_relations.csv produced by the patched
    basalt_vio + gyro-first EuRoC imu0, already in the dataset clock → offset 0).
    """

    relations_path: Path
    imu_path: Path
    imu_euroc: bool = False  # gyro-first EuRoC imu0 order vs accel-first Monado imu.csv
    clock_offset: int = 0  # ns added to every SLAM/IMU timestamp (0 if already dataset-clock)

    def load_relations(self) -> list[tuple[int, SpaceRelation]]:
        return load_slam_relations(self.relations_path, ts_offset=self.clock_offset)

    def load_imu(self) -> ImuHistory:
        loader = load_imu_euroc if self.imu_euroc else load_imu
        return loader(self.imu_path, ts_offset=self.clock_offset)


def basalt_slam_source(basalt_dir: Path, dataset_dir: Path) -> SlamSource:
    """SLAM source for a standalone (patched) basalt_vio run on an EuRoC dataset.

    slam_relations.csv comes from basalt_vio --save-relations-fn; IMU comes from
    the dataset's mav0/imu0/data.csv. Both are already in the dataset clock.
    """
    basalt_dir = Path(basalt_dir)
    dataset_dir = Path(dataset_dir)
    return SlamSource(
        relations_path=basalt_dir / "slam_relations.csv",
        imu_path=dataset_dir / "mav0" / "imu0" / "data.csv",
        imu_euroc=True,
        clock_offset=0,
    )
