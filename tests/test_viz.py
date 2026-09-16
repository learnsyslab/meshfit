"""Overlay geometry. The viser half needs a browser, so only the 2D path and
the quaternion conversion are covered here."""

import numpy as np
import pytest

from meshfit.frames import Y_UP
from meshfit.pose import Pose
from meshfit.solver import rz
from meshfit.views import View
from meshfit.viz import _wxyz, mask_iou, overlay, overlay_panel, silhouette


class FakeMesh:
    def __init__(self, vertices, faces=None):
        self.vertices = np.asarray(vertices, float)
        self.faces = np.zeros((0, 3), np.uint32) if faces is None else faces

    def copy(self):
        return FakeMesh(self.vertices.copy(), self.faces)


def make_view(hw=(120, 160), f=150.0, eye=(0.0, 0.0, -3.0)):
    H, W = hw
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, 3] = eye                      # identity rotation: camera looks +Z
    rgb = np.full((H, W, 3), 128, np.uint8)
    return View(rgb=rgb, mask=np.zeros((H, W), bool), intrinsic=K, cam2world=c2w)


def cube(scale=0.3, n=12):
    g = np.linspace(-scale, scale, n)
    return FakeMesh(np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3))


def test_silhouette_lands_where_the_mesh_is():
    view, mesh = make_view(), cube()
    mask = silhouette(view, mesh, Pose.identity())
    assert mask.any()
    ys, xs = np.nonzero(mask)
    # object at the world origin, camera on -Z looking at it -> centred
    assert abs(xs.mean() - view.image_hw[1] / 2) < 5
    assert abs(ys.mean() - view.image_hw[0] / 2) < 5


def test_silhouette_follows_translation():
    view, mesh = make_view(), cube()
    centre = np.nonzero(silhouette(view, mesh, Pose.identity()))[1].mean()
    shifted = Pose(R=np.eye(3), scale=np.ones(3), t=np.array([0.5, 0.0, 0.0]))
    assert np.nonzero(silhouette(view, mesh, shifted))[1].mean() > centre + 10


def test_points_behind_the_camera_are_dropped():
    view, mesh = make_view(), cube()
    behind = Pose(R=np.eye(3), scale=np.ones(3), t=np.array([0.0, 0.0, -10.0]))
    assert not silhouette(view, mesh, behind).any()


def test_overlay_marks_agreement_in_yellow():
    view, mesh = make_view(), cube()
    pose = Pose.identity()
    view.mask = silhouette(view, mesh, pose)          # perfect agreement
    out = overlay(view, mesh, pose)
    agreeing = out[view.mask]
    # yellow is high red, high green, low blue
    assert agreeing[:, 0].mean() > 180 and agreeing[:, 1].mean() > 160
    assert agreeing[:, 2].mean() < 110


def test_overlay_preserves_shape_and_leaves_background_alone():
    view, mesh = make_view(), cube()
    out = overlay(view, mesh, Pose.identity())
    assert out.shape == view.rgb.shape and out.dtype == np.uint8
    untouched = ~(silhouette(view, mesh, Pose.identity()) | view.mask)
    assert np.array_equal(out[untouched], view.rgb[untouched])


def test_mask_iou_is_one_when_they_coincide_and_zero_when_disjoint():
    view, mesh = make_view(), cube()
    pose = Pose.identity()
    view.mask = silhouette(view, mesh, pose)
    assert np.isclose(mask_iou(view, mesh, pose), 1.0)

    view.mask = np.zeros_like(view.mask)
    view.mask[:5, :5] = True
    assert mask_iou(view, mesh, pose) < 0.05


def test_panel_without_a_renderer_is_just_the_overlay():
    view, mesh = make_view(), cube()
    assert overlay_panel(view, mesh, Pose.identity()).shape == view.rgb.shape


def test_panel_with_a_renderer_is_side_by_side():
    view, mesh = make_view(), cube()

    def fake_render(placed, K, c2w, hw):
        from meshfit.refine import Render
        H, W = hw
        return Render(rgb=np.zeros((H, W, 3), np.uint8), mask=np.ones((H, W), bool))

    H, W = view.image_hw
    assert overlay_panel(view, mesh, Pose.identity(), fake_render).shape == (H, 2 * W, 3)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.5, np.pi])
def test_quaternion_round_trips(yaw):
    R = rz(yaw) @ Y_UP.R_up
    w, x, y, z = _wxyz(R)
    back = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(back, R, atol=1e-6)
