import math

import numpy as np

from math3d import (
    Quat,
    quat_exp,
    quat_finite_difference,
    quat_integrate_velocity,
    quat_ln,
    quat_slerp,
    vec3,
)


def test_quat_identity_multiply():
    q = Quat(0.1, 0.2, 0.3, 0.9).normalized()
    ident = Quat.identity()
    r = q.mul(ident)
    assert np.allclose(r.as_xyzw(), q.as_xyzw())


def test_quat_invert_roundtrip():
    q = Quat(0.1, -0.2, 0.3, 0.9).normalized()
    r = q.mul(q.invert())
    assert np.allclose(r.as_xyzw(), [0, 0, 0, 1], atol=1e-9)


def test_quat_rotate_known():
    q = Quat(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    rotated = q.rotate(vec3(1.0, 0.0, 0.0))
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_quat_exp_ln_roundtrip():
    v = vec3(0.3, -0.1, 0.2)
    back = quat_ln(quat_exp(v))
    assert np.allclose(back, v, atol=1e-9)


def test_quat_exp_zero_is_identity():
    q = quat_exp(vec3(0.0, 0.0, 0.0))
    assert np.allclose(q.as_xyzw(), [0, 0, 0, 1], atol=1e-9)


def test_integrate_velocity_about_z():
    w, dt = 1.5, 0.4
    q = quat_integrate_velocity(Quat.identity(), vec3(0.0, 0.0, w), dt)
    expected_angle = w * dt
    assert math.isclose(2.0 * math.atan2(q.z, q.w), expected_angle, abs_tol=1e-9)


def test_finite_difference_about_z():
    angle, dt = 0.6, 0.5
    right = Quat(0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
    ang_vel = quat_finite_difference(Quat.identity(), right, dt)
    assert np.allclose(ang_vel, [0.0, 0.0, angle / dt], atol=1e-9)


def test_slerp_endpoints():
    a = Quat.identity()
    b = Quat(0.0, 0.0, math.sin(0.5), math.cos(0.5))
    assert np.allclose(quat_slerp(a, b, 0.0).as_xyzw(), a.as_xyzw(), atol=1e-9)
    assert np.allclose(quat_slerp(a, b, 1.0).as_xyzw(), b.as_xyzw(), atol=1e-9)
