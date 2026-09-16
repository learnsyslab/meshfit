"""Initialisers, including the symmetry descriptor the yaw sweep produces."""

import numpy as np

from meshfit.frames import Y_UP
from meshfit.init import (
    _symmetry_of,
    from_bbox,
    from_generator,
    from_search,
)
from meshfit.pose import Pose
from meshfit.solver import rz, tilt_of, yaw_of
from meshfit.views import Observation, View

RNG = np.random.default_rng(11)


class FakeMesh:
    def __init__(self, v, f=None):
        self.vertices = np.asarray(v, float)
        self.faces = np.zeros((0, 3), np.uint32) if f is None else f

    def copy(self):
        return FakeMesh(self.vertices.copy(), self.faces)


def make_view(hw=(120, 160), f=150.0, eye=(0.0, 0.0, -4.0)):
    H, W = hw
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, 3] = eye
    return View(rgb=np.full((H, W, 3), 128, np.uint8), mask=np.ones((H, W), bool),
                intrinsic=K, cam2world=c2w)


def box_mesh(half=(0.2, 0.5, 0.3), n=9):
    """Canonical box, +Y-up, deliberately not square in plan so yaw is
    determinable from the outline."""
    axes = [np.linspace(-h, h, n) for h in half]
    return FakeMesh(np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3))


# -- from_generator ---------------------------------------------------------

def test_from_generator_adopts_the_supplied_pose():
    T = np.eye(4)
    T[:3, :3] = rz(0.6) @ Y_UP.R_up
    T[:3, 3] = [1.0, 2.0, 0.5]
    init = from_generator(T, 1.4)
    assert init.source == "generator"
    assert np.isclose(yaw_of(init.pose.R), 0.6)
    assert np.allclose(init.pose.scale, 1.4)
    assert np.allclose(init.pose.t, [1.0, 2.0, 0.5])


def test_from_generator_preserves_tilt_by_default():
    """Genuinely tilted objects exist; an initialiser cannot tell them from a
    sloppy generator, so it must not silently straighten either."""
    tilt = np.array([[1.0, 0, 0], [0, np.cos(0.2), -np.sin(0.2)],
                     [0, np.sin(0.2), np.cos(0.2)]])
    T = np.eye(4)
    T[:3, :3] = tilt @ rz(0.3) @ Y_UP.R_up
    assert tilt_of(from_generator(T, 1.0).pose.R) > 1.0


def test_from_generator_straightens_only_when_asked():
    tilt = np.array([[1.0, 0, 0], [0, np.cos(0.2), -np.sin(0.2)],
                     [0, np.sin(0.2), np.cos(0.2)]])
    T = np.eye(4)
    T[:3, :3] = tilt @ rz(0.3) @ Y_UP.R_up
    assert tilt_of(from_generator(T, 1.0, force_upright=True).pose.R) < 1e-9


def test_from_generator_accepts_anisotropic_scale():
    T = np.eye(4)
    T[:3, :3] = Y_UP.R_up
    assert np.allclose(from_generator(T, np.array([1.0, 2.0, 3.0])).pose.scale,
                       [1.0, 2.0, 3.0])


# -- from_bbox --------------------------------------------------------------

def test_from_bbox_recovers_scale_and_position():
    mesh = box_mesh()
    truth = Pose.similarity(R=Y_UP.R_up, s=2.0, t=np.array([1.0, -0.5, 0.25]))
    obs = Observation(views=[make_view()], points=truth.apply(mesh.vertices))

    pose = from_bbox(mesh, obs).pose
    assert np.isclose(pose.scale[0], 2.0, rtol=1e-6)
    assert np.allclose(pose.t, truth.t, atol=1e-6)


def test_from_bbox_is_upright():
    mesh = box_mesh()
    obs = Observation(views=[make_view()], points=mesh.vertices)
    assert tilt_of(from_bbox(mesh, obs, yaw=1.1).pose.R) < 1e-9


def test_from_bbox_uses_height_not_mean_extent():
    """A single-view cloud only covers the front, so horizontal extents are
    truncated; height is the one that survives."""
    mesh = box_mesh()
    truth = Pose.similarity(R=Y_UP.R_up, s=1.5, t=np.zeros(3))
    full = truth.apply(mesh.vertices)
    front = full[full[:, 1] < np.median(full[:, 1])]      # keep half the depth
    obs = Observation(views=[make_view()], points=front)
    assert np.isclose(from_bbox(mesh, obs).pose.scale[0], 1.5, rtol=1e-6)


# -- symmetry ---------------------------------------------------------------

def test_sharp_single_peak_is_unambiguous():
    yaws = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    scores = np.exp(-((yaws - np.pi) ** 2) / 0.1)
    s = _symmetry_of(scores, yaws)
    assert s.k_fold == 1 and s.margin > 0.5 and not s.is_ambiguous


def test_four_fold_symmetry_is_detected():
    yaws = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    scores = 0.5 + 0.5 * np.cos(4 * yaws)          # square table
    s = _symmetry_of(scores, yaws)
    assert s.k_fold == 4 and s.is_ambiguous


def test_flat_curve_is_ambiguous():
    yaws = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    s = _symmetry_of(np.full(36, 0.8), yaws)       # vase
    assert s.is_ambiguous and s.margin < 0.05


def test_all_zero_scores_do_not_divide_by_zero():
    yaws = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    s = _symmetry_of(np.zeros(12), yaws)
    assert s.is_ambiguous and s.margin == 0.0


# -- from_search ------------------------------------------------------------

def test_search_finds_the_true_yaw_from_the_outline():
    mesh = box_mesh(half=(0.15, 0.5, 0.4))          # clearly non-square in plan
    truth_yaw = np.pi / 2
    truth = Pose.similarity(R=rz(truth_yaw) @ Y_UP.R_up, s=1.0, t=np.zeros(3))
    view = make_view()
    view.mask = _splat(view, mesh, truth)
    obs = Observation(views=[view], points=truth.apply(mesh.vertices))

    init = from_search(mesh, obs, yaw_steps=36)
    err = abs(((yaw_of(init.pose.R) - truth_yaw + np.pi) % (2 * np.pi)) - np.pi)
    assert init.source == "search"
    assert min(err, abs(err - np.pi)) < 0.2          # up to the 180 deg tie


def test_search_reports_ambiguity_for_a_square_footprint():
    mesh = box_mesh(half=(0.3, 0.5, 0.3))           # square in plan
    truth = Pose.similarity(R=Y_UP.R_up, s=1.0, t=np.zeros(3))
    view = make_view()
    view.mask = _splat(view, mesh, truth)
    obs = Observation(views=[view], points=truth.apply(mesh.vertices))

    assert from_search(mesh, obs, yaw_steps=36).symmetry.is_ambiguous


def test_search_without_a_matcher_still_returns_a_pose():
    mesh = box_mesh()
    view = make_view()
    obs = Observation(views=[view], points=mesh.vertices)
    init = from_search(mesh, obs, match=None, yaw_steps=12)
    assert init.pose is not None and init.symmetry is not None


def _splat(view, mesh, pose):
    from meshfit.viz import silhouette
    return silhouette(view, mesh, pose)
