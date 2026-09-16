"""fit_joint: does solving in pixels actually beat solving in metres?"""

import numpy as np
import pytest

from meshfit.correspond import Correspondence
from meshfit.frames import Y_UP
from meshfit.pose import Pose
from meshfit.solver import (
    fit_anisotropic,
    fit_joint,
    fit_upright,
    reprojection_rms,
    rz,
    yaw_of,
)
from meshfit.views import View

RNG = np.random.default_rng(7)


def look_at(eye, target):
    """OpenCV-convention cam2world: camera looks down its own +Z."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], axis=1)
    c2w[:3, 3] = eye
    return c2w


def make_view(eye, target=None, hw=(480, 640), f=500.0):
    target = np.zeros(3) if target is None else target
    H, W = hw
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    rgb = np.zeros((H, W, 3), np.uint8)
    mask = np.ones((H, W), bool)
    return View(rgb=rgb, mask=mask, intrinsic=K, cam2world=look_at(np.asarray(eye, float), target))


def truth_pose(yaw=0.6, scale=(1.3, 0.8, 1.1), t=(0.2, -0.3, 0.5)):
    return Pose(R=rz(yaw) @ Y_UP.R_up, scale=np.array(scale), t=np.array(t))


def build(views, pose, n=150, depth_sigma=0.0, px_sigma=0.0, rng=RNG):
    """Correspondences for `pose`, with depth noise applied ALONG each ray --
    which is how monocular depth error actually behaves."""
    corrs = []
    for v in views:
        canonical = rng.normal(size=(n, 3)) * 0.5
        world = pose.apply(canonical)
        c = Correspondence(canonical=canonical, pixel=np.zeros((n, 2)), view=v)
        uv, ok = c.project(world)
        uv = uv + rng.normal(size=uv.shape) * px_sigma

        eye = v.cam2world[:3, 3]
        ray = world - eye
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        noisy = world + ray * rng.normal(size=(n, 1)) * depth_sigma

        corrs.append(Correspondence(canonical=canonical[ok], pixel=uv[ok],
                                    view=v, world=noisy[ok]))
    return corrs


def pose_error(a, b):
    return (abs(yaw_of(a.R) - yaw_of(b.R)),
            float(np.linalg.norm(a.t - b.t)),
            float(np.max(np.abs(a.scale / b.scale - 1.0))))


# ---------------------------------------------------------------------------

def test_recovers_the_true_pose_from_clean_multiview_data():
    truth = truth_pose()
    views = [make_view([3, 0, 1]), make_view([0, 3, 1]), make_view([-2, -2, 2])]
    corrs = build(views, truth)
    init = Pose(R=rz(0.0) @ Y_UP.R_up, scale=np.ones(3), t=np.zeros(3))

    got = fit_joint(corrs, init)
    d_yaw, d_t, d_s = pose_error(got, truth)
    assert d_yaw < 1e-4 and d_t < 1e-4 and d_s < 1e-4


def test_beats_a_3d_fit_when_depth_is_noisy_but_pixels_are_clean():
    """The reason fit_joint exists. Depth error is ~10cm, matcher error ~1px;
    a 3D residual weights them equally, reprojection does not."""
    truth = truth_pose()
    views = [make_view([3, 0, 1]), make_view([0, 3, 1])]
    corrs = build(views, truth, depth_sigma=0.10, px_sigma=0.5)

    x = np.concatenate([c.canonical for c in corrs])
    y = np.concatenate([c.world for c in corrs])
    from_3d = fit_anisotropic(x, y, prior=np.ones(3), reg=1e-6,
                              init_theta=yaw_of(fit_upright(x, y).R))
    joint = fit_joint(corrs, from_3d, sigma_px=1.0, sigma_depth=0.10)

    e3 = pose_error(from_3d, truth)
    ej = pose_error(joint, truth)
    print(f"\n  3D fit   yaw {e3[0]:.4f}  t {e3[1]:.4f}m  scale {e3[2]:.4f}")
    print(f"  fit_joint yaw {ej[0]:.4f}  t {ej[1]:.4f}m  scale {ej[2]:.4f}")
    assert ej[1] < e3[1]          # translation
    assert ej[2] < e3[2]          # scale


def test_single_view_needs_the_depth_prior_to_fix_scale():
    """Reprojection alone is gauge-degenerate under one camera: slide along the
    ray and rescale, and every pixel is identical."""
    truth = truth_pose()
    corrs = build([make_view([3, 0, 1])], truth)
    init = Pose(R=rz(0.5) @ Y_UP.R_up, scale=np.ones(3) * 1.5, t=np.array([0.5, 0, 0]))

    with_prior = fit_joint(corrs, init, sigma_depth=0.02)
    assert pose_error(with_prior, truth)[2] < 0.02

    stripped = [Correspondence(canonical=c.canonical, pixel=c.pixel, view=c.view)
                for c in corrs]
    without = fit_joint(stripped, init)
    # pixels still line up, but the pose is free to slide along the ray
    assert reprojection_rms(without, corrs) < 1.0
    assert pose_error(without, truth)[2] > pose_error(with_prior, truth)[2]


def test_huber_survives_gross_matcher_outliers():
    truth = truth_pose()
    corrs = build([make_view([3, 0, 1]), make_view([0, 3, 1])], truth, px_sigma=0.3)
    corrs[0].pixel[:25] += RNG.normal(size=(25, 2)) * 150.0      # ~17% garbage

    got = fit_joint(corrs, truth_pose(yaw=0.4, scale=(1.0, 1.0, 1.0), t=(0, 0, 0)))
    assert pose_error(got, truth)[1] < 0.05


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="no correspondences"):
        fit_joint([], Pose.identity())


def test_reprojection_rms_drops_after_fitting():
    truth = truth_pose()
    corrs = build([make_view([3, 0, 1]), make_view([0, 3, 1])], truth, px_sigma=0.4)
    init = Pose(R=rz(0.2) @ Y_UP.R_up, scale=np.ones(3), t=np.zeros(3))
    assert reprojection_rms(fit_joint(corrs, init), corrs) < reprojection_rms(init, corrs)
