# t_tracker_slam.cpp::predict_pose, t_predict.c::m_predict_relation, t_dead_reckoning.c::t_apply_dead_reckoning
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from math3d import (
    ANGULAR_VELOCITY_VALID,
    BITMASK_NONE,
    LINEAR_VELOCITY_VALID,
    ORIENTATION_VALID,
    POSITION_VALID,
    Pose,
    Quat,
    SpaceRelation,
    ns_to_s,
    quat_integrate_velocity,
    space_relation_interpolate,
    vec3,
)


class PredictionType(IntEnum):
    NONE = 0           #: No prediction, always return the last SLAM tracked pose
    POSE_ONLY = 1      #: Predicts from last two SLAM poses only
    GYRO = 2           #: Predicts from last SLAM pose with angular velocity computed from IMU
    ACCEL_GYRO = 3     #: Predicts from last SLAM pose with angular and linear velocity computed from IMU
    DEAD_RECKONING = 4 #: Predicts from a pose that is the last SLAM pose with the IMU samples that came after it integrated on top; velocities from latest IMU sample.



# m_predict_relation
def _predict_orientation(rel: SpaceRelation, delta_s: float, out: SpaceRelation) -> None:
    # m_predict.c::do_orientation
    flags = rel.relation_flags
    if delta_s == 0.0:
        out.pose.orientation = rel.pose.orientation
        out.angular_velocity = rel.angular_velocity.copy()
        return

    accum = vec3()
    valid_orientation = bool(flags & ORIENTATION_VALID)
    valid_angular_velocity = bool(flags & ANGULAR_VELOCITY_VALID)

    if valid_angular_velocity:
        orientation_inv = rel.pose.orientation.invert()
        accum = orientation_inv.rotate_derivative(rel.angular_velocity)

    if valid_orientation:
        out.pose.orientation = quat_integrate_velocity(rel.pose.orientation, accum, delta_s)

    if valid_angular_velocity:
        out.angular_velocity = out.pose.orientation.rotate_derivative(accum)


def _predict_position(rel: SpaceRelation, delta_s: float, out: SpaceRelation) -> None:
    # m_predict.c::do_position
    flags = rel.relation_flags
    if delta_s == 0.0:
        out.pose.position = rel.pose.position.copy()
        out.linear_velocity = rel.linear_velocity.copy()
        return

    accum = vec3()
    valid_position = bool(flags & POSITION_VALID)
    valid_linear_velocity = bool(flags & LINEAR_VELOCITY_VALID)

    if valid_linear_velocity:
        accum = rel.linear_velocity.copy()

    if valid_position:
        out.pose.position = rel.pose.position + accum * delta_s

    if valid_linear_velocity:
        out.linear_velocity = accum


def predict_relation(rel: SpaceRelation, delta_s: float) -> SpaceRelation:
    out = SpaceRelation()
    _predict_orientation(rel, delta_s, out)
    _predict_position(rel, delta_s, out)
    out.relation_flags = rel.relation_flags
    return out


@dataclass
class RelationHistory:
    _ts: list[int] = field(default_factory=list)
    _rels: list[SpaceRelation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._ts)

    def push(self, rel: SpaceRelation, ts: int) -> None:
        idx = bisect.bisect_right(self._ts, ts)
        self._ts.insert(idx, ts)
        self._rels.insert(idx, rel.copy())

    def get_latest(self) -> tuple[int, SpaceRelation] | None:
        if not self._ts:
            return None
        return self._ts[-1], self._rels[-1].copy()

    def get(self, when_ns: int) -> SpaceRelation:
        if not self._ts:
            out = SpaceRelation()
            out.relation_flags = BITMASK_NONE
            return out

        if when_ns <= self._ts[0]:
            return self._rels[0].copy()

        if when_ns >= self._ts[-1]:
            last = self._rels[-1]
            return predict_relation(last, ns_to_s(when_ns - self._ts[-1]))

        hi = bisect.bisect_left(self._ts, when_ns)
        if self._ts[hi] == when_ns:
            return self._rels[hi].copy()
        lo = hi - 1
        span = self._ts[hi] - self._ts[lo]
        t = (when_ns - self._ts[lo]) / span if span else 0.0
        return space_relation_interpolate(self._rels[lo], self._rels[hi], t)


@dataclass
class ImuHistory:

    ts: list[int] = field(default_factory=list)
    gyro: list[np.ndarray] = field(default_factory=list)
    accel: list[np.ndarray] = field(default_factory=list)

    def push(self, ts: int, gyro: np.ndarray, accel: np.ndarray) -> None:
        self.ts.append(int(ts))
        self.gyro.append(np.asarray(gyro, dtype=np.float64))
        self.accel.append(np.asarray(accel, dtype=np.float64))

    @property
    def last_ts(self) -> int:
        return self.ts[-1] if self.ts else 0

    def average(self, start_ts: int, end_ts: int, channel: str) -> np.ndarray:
        samples = self.gyro if channel == "gyro" else self.accel
        lo = bisect.bisect_left(self.ts, start_ts)
        hi = bisect.bisect_right(self.ts, end_ts)
        if hi <= lo:
            return vec3()
        return np.mean(samples[lo:hi], axis=0)

    def window(self, start_ts: int, end_ts: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
        lo = bisect.bisect_right(self.ts, start_ts)
        hi = bisect.bisect_right(self.ts, end_ts)
        return [(self.ts[i], self.gyro[i], self.accel[i]) for i in range(lo, hi)]


# t_apply_dead_reckoning
def apply_dead_reckoning(
    imu: ImuHistory,
    gravity_correction: np.ndarray,
    when_ns: int,
    base_rel: SpaceRelation,
    base_rel_ts: int,
    use_accel: bool = True,
) -> SpaceRelation:
    integ = base_rel.copy()
    integ_ts = base_rel_ts

    samples = imu.window(base_rel_ts, when_ns + 10**12)

    for ts, gyro, accel in samples:
        clamped = ts > when_ns
        step_ts = when_ns if clamped else ts

        dt = ns_to_s(step_ts - integ_ts)
        integ_ts = step_ts

        integ.pose.orientation = quat_integrate_velocity(integ.pose.orientation, gyro, dt)
        integ.angular_velocity = integ.pose.orientation.rotate_derivative(gyro)

        if use_accel:
            world_accel = integ.pose.orientation.rotate(accel) + gravity_correction
            integ.linear_velocity = integ.linear_velocity + world_accel * dt
            integ.pose.position = integ.pose.position + (
                integ.linear_velocity * dt + world_accel * (dt * dt * 0.5)
            )

        if clamped:
            break

    last_imu_to_now = ns_to_s(when_ns - integ_ts)
    return predict_relation(integ, last_imu_to_now)


def predict_pose(
    rels: RelationHistory,
    when_ns: int,
    pred_type: PredictionType = PredictionType.DEAD_RECKONING,
    imu: ImuHistory | None = None,
    gravity_correction: np.ndarray | None = None,
) -> SpaceRelation:
    if gravity_correction is None:
        gravity_correction = vec3()

    latest = rels.get_latest()
    if latest is None:
        out = SpaceRelation()
        out.relation_flags = BITMASK_NONE
        return out
    rel_ts, rel = latest

    if pred_type == PredictionType.NONE:
        return rel

    if pred_type == PredictionType.POSE_ONLY or when_ns <= rel_ts:
        return rels.get(when_ns)

    if pred_type == PredictionType.DEAD_RECKONING:
        if imu is None:
            raise ValueError("DEAD_RECKONING prediction requires an ImuHistory")
        return apply_dead_reckoning(imu, gravity_correction, when_ns, rel, rel_ts)

    if imu is None:
        raise ValueError("GYRO/ACCEL_GYRO prediction requires an ImuHistory")

    if pred_type >= PredictionType.GYRO:
        avg_gyro = imu.average(rel_ts, when_ns, "gyro")
        rel.angular_velocity = rel.pose.orientation.rotate_derivative(avg_gyro)

    if pred_type >= PredictionType.ACCEL_GYRO:
        avg_accel = imu.average(rel_ts, when_ns, "accel")
        world_accel = rel.pose.orientation.rotate(avg_accel) + gravity_correction
        slam_to_imu_dt = ns_to_s(imu.last_ts - rel_ts)
        rel.linear_velocity = rel.linear_velocity + world_accel * slam_to_imu_dt

    return predict_relation(rel, ns_to_s(when_ns - rel_ts))
