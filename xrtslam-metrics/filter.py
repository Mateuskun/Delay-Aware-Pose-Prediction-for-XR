from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from math3d import (
    ORIENTATION_VALID,
    POSITION_VALID,
    Quat,
    SpaceRelation,
    quat_exp,
    quat_ln,
    quat_slerp,
    space_relation_interpolate,
    vec3,
)

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000

# One Euro Filter: m_filter_one_euro.c:

def _calc_smoothing_alpha(fc: float, dt: float) -> float:
    """``calc_smoothing_alpha`` -- alpha = 1 / (1 + tau/dt), tau = 1/(2 pi fc)."""
    r = 2.0 * math.pi * fc * dt
    return r / (r + 1.0)


def _exp_smooth_vec3(alpha: float, y: np.ndarray, prev_y: np.ndarray) -> np.ndarray:
    return alpha * y + (1.0 - alpha) * prev_y


def _exp_smooth_quat(alpha: float, y: Quat, prev_y: Quat) -> Quat:
    return quat_slerp(prev_y, y, alpha)

# m_filter_euro_vec3
@dataclass
class OneEuroVec3:
    fc_min: float
    fc_min_d: float
    beta: float
    prev_y: np.ndarray = field(default_factory=vec3)
    prev_dy: np.ndarray = field(default_factory=vec3)
    prev_ts: int = 0
    have_prev: bool = False

    def run(self, ts: int, value: np.ndarray) -> np.ndarray:
        if not self.have_prev:
            self.prev_dy = vec3()
            self.prev_y = np.asarray(value, dtype=np.float64).copy()
            self.prev_ts = ts
            self.have_prev = True
            return self.prev_y.copy()

        if ts == self.prev_ts:
            return self.prev_y.copy()

        dt = ((ts - self.prev_ts) % (1 << 64)) / NS_PER_S
        self.prev_ts = ts

        alpha_d = _calc_smoothing_alpha(self.fc_min_d, dt)
        dy = (value - self.prev_y) / dt
        self.prev_dy = _exp_smooth_vec3(alpha_d, dy, self.prev_dy)

        dy_mag = float(np.linalg.norm(self.prev_dy))
        alpha = _calc_smoothing_alpha(self.fc_min + self.beta * dy_mag, dt)
        self.prev_y = _exp_smooth_vec3(alpha, value, self.prev_y)
        return self.prev_y.copy()

# m_filter_euro_quat
@dataclass
class OneEuroQuat:
    fc_min: float
    fc_min_d: float
    beta: float
    prev_y: Quat = field(default_factory=Quat.identity)
    prev_dy: Quat = field(default_factory=Quat.identity)
    prev_ts: int = 0
    have_prev: bool = False

    def run(self, ts: int, value: Quat) -> Quat:
        if not self.have_prev:
            self.prev_dy = Quat.identity()
            self.prev_y = value
            self.prev_ts = ts
            self.have_prev = True
            return self.prev_y

        if ts == self.prev_ts:
            return self.prev_y

        dt = ((ts - self.prev_ts) % (1 << 64)) / NS_PER_S
        self.prev_ts = ts

        alpha_d = _calc_smoothing_alpha(self.fc_min_d, dt)

        dy = self.prev_y.unrotate(value)
        dy_aa = quat_ln(dy) / dt
        dy = quat_exp(dy_aa)
        self.prev_dy = _exp_smooth_quat(alpha_d, dy, self.prev_dy)

        smooth_dy_mag = float(np.linalg.norm(quat_ln(self.prev_dy)))
        alpha = _calc_smoothing_alpha(self.fc_min + self.beta * smooth_dy_mag, dt)
        self.prev_y = _exp_smooth_quat(alpha, value, self.prev_y)
        return self.prev_y


# filter_pose: t_tracker_slam.cpp
@dataclass
class FilterConfig:
    use_moving_average_filter: bool = False
    window_ms: float = 66.0  # Monado default moving-average window

    use_exponential_smoothing_filter: bool = False
    alpha: float = 0.4

    use_one_euro_filter: bool = False
    min_cutoff: float = math.pi  # M_PI
    min_dcutoff: float = 1.0
    beta: float = 0.16


class PoseFilter:
    def __init__(self, config: FilterConfig | None = None):
        self.config = config or FilterConfig()
        c = self.config

        self._pos_hist: deque[tuple[int, np.ndarray]] = deque()
        self._rot_hist: deque[tuple[int, np.ndarray]] = deque()

        self._es_last: SpaceRelation | None = None

        self._oe_pos = OneEuroVec3(c.min_cutoff, c.min_dcutoff, c.beta)
        self._oe_rot = OneEuroQuat(c.min_cutoff, c.min_dcutoff, c.beta)

    def _moving_average(self, when_ns: int, rel: SpaceRelation) -> None:
        flags = rel.relation_flags
        if flags & POSITION_VALID:
            self._pos_hist.append((when_ns, rel.pose.position.copy()))
        if flags & ORIENTATION_VALID:
            q = rel.pose.orientation
            self._rot_hist.append((when_ns, vec3(q.x, q.y, q.z)))

        window = int(self.config.window_ms * NS_PER_MS)
        lo = when_ns - window

        def avg(hist: deque[tuple[int, np.ndarray]]) -> np.ndarray | None:
            while hist and hist[0][0] < lo:
                hist.popleft()
            vals = [v for (ts, v) in hist if lo <= ts <= when_ns]
            return np.mean(vals, axis=0) if vals else None

        avg_pos = avg(self._pos_hist)
        avg_rot = avg(self._rot_hist)

        if avg_rot is not None:
            sq = float(avg_rot @ avg_rot)
            w = math.sqrt(max(0.0, 1.0 - sq))
            rel.pose.orientation = Quat(avg_rot[0], avg_rot[1], avg_rot[2], w)
        if avg_pos is not None:
            rel.pose.position = avg_pos

    def _exponential_smoothing(self, rel: SpaceRelation) -> None:
        target = rel.copy()
        if self._es_last is None:
            self._es_last = target
        else:
            self._es_last = space_relation_interpolate(
                self._es_last, target, self.config.alpha
            )
        blended = self._es_last
        rel.pose.position = blended.pose.position.copy()
        rel.pose.orientation = blended.pose.orientation
        rel.linear_velocity = blended.linear_velocity.copy()
        rel.angular_velocity = blended.angular_velocity.copy()

    def _one_euro(self, when_ns: int, rel: SpaceRelation) -> None:
        flags = rel.relation_flags
        if flags & POSITION_VALID:
            rel.pose.position = self._oe_pos.run(when_ns, rel.pose.position)
        if flags & ORIENTATION_VALID:
            rel.pose.orientation = self._oe_rot.run(when_ns, rel.pose.orientation)


    def run(self, when_ns: int, relation: SpaceRelation) -> SpaceRelation:
        """Filter ``relation`` in place-style, returning the filtered copy."""
        rel = relation.copy()
        if self.config.use_moving_average_filter:
            self._moving_average(when_ns, rel)
        if self.config.use_exponential_smoothing_filter:
            self._exponential_smoothing(rel)
        if self.config.use_one_euro_filter:
            self._one_euro(when_ns, rel)
        return rel
