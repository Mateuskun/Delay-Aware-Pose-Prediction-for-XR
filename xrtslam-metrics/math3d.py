# From aux/math/m_base.cpp
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

NANOSECONDS_PER_SECOND = 1_000_000_000


def ns_to_s(dt_ns: float) -> float:
    return dt_ns / NANOSECONDS_PER_SECOND


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)

@dataclass
class Quat:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


    @staticmethod
    def identity() -> "Quat":
        return Quat(0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def from_xyzw(a: np.ndarray) -> "Quat":
        return Quat(float(a[0]), float(a[1]), float(a[2]), float(a[3]))


    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def as_xyzw(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float64)


    def normalized(self) -> "Quat":
        n = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
        if n == 0.0:
            return Quat.identity()
        return Quat(self.x / n, self.y / n, self.z / n, self.w / n)

    def conjugate(self) -> "Quat":
        return Quat(-self.x, -self.y, -self.z, self.w)

    def invert(self) -> "Quat":
        return self.conjugate()

    def mul(self, other: "Quat") -> "Quat":
        a, b = self, other
        return Quat(
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        )

    def __matmul__(self, other: "Quat") -> "Quat":
        return self.mul(other)

    def rotate(self, v: np.ndarray) -> np.ndarray:
        u = self.xyz
        s = self.w
        return 2.0 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)

    def rotate_derivative(self, v: np.ndarray) -> np.ndarray:
        return self.rotate(v)

    def unrotate(self, other: "Quat") -> "Quat":
        return self.invert().mul(other)

    def dot(self, other: "Quat") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w


def quat_exp(v: np.ndarray) -> Quat:
    angle = float(np.linalg.norm(v))
    if angle < 1e-10:
        return Quat(float(v[0]), float(v[1]), float(v[2]), 1.0).normalized()
    s = math.sin(angle) / angle
    return Quat(float(v[0]) * s, float(v[1]) * s, float(v[2]) * s, math.cos(angle))


def quat_ln(q: Quat) -> np.ndarray:
    vec_len = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z)
    if vec_len < 1e-10:
        return vec3()
    angle = math.atan2(vec_len, q.w)
    return q.xyz * (angle / vec_len)


def quat_integrate_velocity(q: Quat, ang_vel: np.ndarray, dt: float) -> Quat:
    half_angle = ang_vel * (dt * 0.5)
    delta = quat_exp(half_angle)
    return q.mul(delta).normalized()


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    cos_half_theta = a.dot(b)
    bb = b
    if cos_half_theta < 0.0:  # take the shorter arc
        bb = Quat(-b.x, -b.y, -b.z, -b.w)
        cos_half_theta = -cos_half_theta

    if cos_half_theta >= 1.0:
        return a

    half_theta = math.acos(cos_half_theta)
    sin_half_theta = math.sqrt(1.0 - cos_half_theta * cos_half_theta)

    if abs(sin_half_theta) < 1e-6:
        return Quat(
            a.x * 0.5 + bb.x * 0.5,
            a.y * 0.5 + bb.y * 0.5,
            a.z * 0.5 + bb.z * 0.5,
            a.w * 0.5 + bb.w * 0.5,
        ).normalized()

    ratio_a = math.sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = math.sin(t * half_theta) / sin_half_theta
    return Quat(
        a.x * ratio_a + bb.x * ratio_b,
        a.y * ratio_a + bb.y * ratio_b,
        a.z * ratio_a + bb.z * ratio_b,
        a.w * ratio_a + bb.w * ratio_b,
    )


def quat_finite_difference(left: Quat, right: Quat, dt: float) -> np.ndarray:
    if dt <= 0.0:
        return vec3()
    rel = left.invert().mul(right)
    return quat_ln(rel) * (2.0 / dt)


# xrt_space_relation_flags bits we care about.
POSITION_VALID = 1 << 0
ORIENTATION_VALID = 1 << 1
LINEAR_VELOCITY_VALID = 1 << 2
ANGULAR_VELOCITY_VALID = 1 << 3
BITMASK_ALL = POSITION_VALID | ORIENTATION_VALID | LINEAR_VELOCITY_VALID | ANGULAR_VELOCITY_VALID
BITMASK_NONE = 0


@dataclass
class Pose:
    position: np.ndarray = field(default_factory=vec3)
    orientation: Quat = field(default_factory=Quat.identity)

    def copy(self) -> "Pose":
        return Pose(self.position.copy(), replace(self.orientation))


@dataclass
class SpaceRelation:

    pose: Pose = field(default_factory=Pose)
    linear_velocity: np.ndarray = field(default_factory=vec3)
    angular_velocity: np.ndarray = field(default_factory=vec3)
    relation_flags: int = BITMASK_NONE

    @staticmethod
    def zero() -> "SpaceRelation":
        return SpaceRelation()

    def copy(self) -> "SpaceRelation":
        return SpaceRelation(
            self.pose.copy(),
            self.linear_velocity.copy(),
            self.angular_velocity.copy(),
            self.relation_flags,
        )


def space_relation_interpolate(a: SpaceRelation, b: SpaceRelation, t: float) -> SpaceRelation:
    out = SpaceRelation()
    out.relation_flags = a.relation_flags & b.relation_flags
    out.pose.position = (1.0 - t) * a.pose.position + t * b.pose.position
    out.pose.orientation = quat_slerp(a.pose.orientation, b.pose.orientation, t)
    out.linear_velocity = (1.0 - t) * a.linear_velocity + t * b.linear_velocity
    out.angular_velocity = (1.0 - t) * a.angular_velocity + t * b.angular_velocity
    return out
