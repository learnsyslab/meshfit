"""The refine loop, exercised with a stub renderer and a stub matcher.

Neither bpy nor RoMa is involved: the loop's job is bookkeeping -- place,
render, match, lift, pool, solve, compose -- and that is all testable with
fakes. What the real backends add is pixels, not logic.
"""

import numpy as np
import pytest

from meshfit.frames import Y_UP
from meshfit.pose import Pose
from meshfit.refine import refine
from meshfit.solver import rz, tilt_of, yaw_of
from meshfit.views import View

RNG = np.random.default_rng(3)


def unit_sphere():
    """A real trimesh: the loop raycasts the placed mesh to lift keypoints,
    so a vertex cloud is no longer enough."""
    import trimesh
    return trimesh.creation.icosphere(subdivisions=3, radius=0.35)


def look_at(eye, target=None):
    target = np.zeros(3) if target is None else target
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    c2w[:3, 3] = eye
    return c2w


def project(points_world, K, c2w):
    w2c = np.linalg.inv(c2w)
    cam = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uv = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                   K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)
    return uv, cam, z


def scene(hw=(240, 320), f=300.0):
    """A mesh, its true pose, and views whose pointmaps encode that truth."""
    H, W = hw
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    mesh = unit_sphere()
    canonical = np.asarray(mesh.vertices, dtype=float)
    truth = Pose(R=rz(0.5) @ Y_UP.R_up, scale=np.full(3, 1.2),
                 t=np.array([0.1, -0.2, 0.3]))
    world = truth.apply(canonical)

    views = []
    for eye in ([2.5, 0.0, 1.0], [0.0, 2.5, 1.0]):
        c2w = look_at(np.asarray(eye, float))
        uv, cam, _ = project(world, K, c2w)
        uv_int = np.round(uv).astype(int)
        inside = ((uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
                  & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H))
        # integer keypoints make the bilinear pointmap sample exact
        pointmap = np.zeros((H, W, 3))
        pointmap[uv_int[inside, 1], uv_int[inside, 0]] = cam[inside]
        views.append(View(rgb=np.zeros((H, W, 3), np.uint8),
                          mask=np.ones((H, W), bool), intrinsic=K,
                          cam2world=c2w, pointmap=pointmap))
    return mesh, truth, views, canonical


def make_backends(truth, canonical, *, coverage=True, matches=True):
    """A perfect matcher: it reports where the mesh renders NOW, paired with
    where that same surface point is actually observed."""
    state = {}

    def render(placed, K, c2w, image_hw):
        from meshfit.refine import Render
        H, W = image_hw
        verts = np.asarray(placed.vertices)
        uv, _cam, z = project(verts, K, c2w)
        uv_int = np.round(uv).astype(int)
        inside = ((uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
                  & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H) & (z > 0))
        # front-facing only: a raycast at this pixel returns the near surface
        outward = verts - verts.mean(axis=0)
        towards_cam = c2w[:3, 3] - verts
        ok = inside & ((outward * towards_cam).sum(axis=1) > 0)

        mask = np.zeros((H, W), bool)
        if coverage:
            mask[uv_int[ok, 1], uv_int[ok, 0]] = True
        state["rendered_uv"] = uv[ok]
        state["K"], state["c2w"], state["hw"] = K, c2w, image_hw
        state["ok"] = ok
        return Render(rgb=np.zeros((H, W, 3), np.uint8), mask=mask)

    def match(rendered_rgb, observed_rgb):
        if not matches:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        ok = state["ok"]
        truth_uv, _, _ = project(truth.apply(canonical), state["K"], state["c2w"])
        H, W = state["hw"]
        obs = np.round(truth_uv[ok]).astype(float)
        keep = ((obs[:, 0] >= 0) & (obs[:, 0] < W)
                & (obs[:, 1] >= 0) & (obs[:, 1] < H))
        k_render = np.round(state["rendered_uv"][keep])
        return k_render, obs[keep], np.ones(keep.sum())

    return render, match


# ---------------------------------------------------------------------------

def test_converges_to_the_true_pose():
    mesh, truth, views, canonical = scene()
    render, match = make_backends(truth, canonical)
    start = Pose(R=rz(0.35) @ Y_UP.R_up, scale=np.full(3, 1.0), t=np.zeros(3))

    out = refine(mesh, start, views, render, match, iters=5)

    assert out.iterations >= 1
    assert abs(yaw_of(out.pose.R) - yaw_of(truth.R)) < 0.02
    assert np.linalg.norm(out.pose.t - truth.t) < 0.02
    assert abs(out.pose.scale[0] - truth.scale[0]) < 0.02


def test_stays_upright_no_matter_how_many_iterations():
    mesh, truth, views, canonical = scene()
    render, match = make_backends(truth, canonical)
    start = Pose(R=rz(1.2) @ Y_UP.R_up, scale=np.full(3, 0.7), t=np.zeros(3))
    assert tilt_of(refine(mesh, start, views, render, match, iters=8).pose.R) < 1e-9


def test_reports_doing_nothing_rather_than_faking_success():
    """Every view unusable must be distinguishable from 'already perfect'."""
    mesh, truth, views, canonical = scene()
    render, match = make_backends(truth, canonical, coverage=False)
    start = Pose(R=rz(0.35) @ Y_UP.R_up, scale=np.ones(3), t=np.zeros(3))

    out = refine(mesh, start, views, render, match)
    assert out.found_nothing
    assert out.iterations == 0
    assert out.skipped[0] == len(views)
    assert np.allclose(out.pose.R, start.R)      # unchanged, and says so


def test_too_few_matches_skips_the_view():
    mesh, truth, views, canonical = scene()
    render, match = make_backends(truth, canonical, matches=False)
    out = refine(mesh, Pose.identity(), views, render, match)
    assert out.found_nothing and out.skipped[0] == len(views)


def test_stops_early_once_the_delta_is_negligible():
    mesh, truth, views, canonical = scene()
    render, match = make_backends(truth, canonical)
    out = refine(mesh, truth.copy(), views, render, match, iters=9)
    assert out.converged and out.iterations < 9


def test_emits_correspondences_for_the_joint_solve():
    mesh, truth, views, canonical = scene()
    render, match = make_backends(truth, canonical)
    start = Pose(R=rz(0.4) @ Y_UP.R_up, scale=np.ones(3), t=np.zeros(3))

    out = refine(mesh, start, views, render, match, iters=4)
    assert out.correspondences
    for c in out.correspondences:
        c.validate()
        assert c.world is not None                 # depth prior available
        # canonical points should land back near the mesh's own extent
        assert np.abs(c.canonical).max() < 3.0
    # and they should reproject tightly under the recovered pose
    from meshfit.solver import reprojection_rms
    assert reprojection_rms(out.pose, out.correspondences) < 3.0


def test_requires_pointmaps():
    mesh, truth, views, canonical = scene()
    for v in views:
        v.pointmap = None
    render, match = make_backends(truth, canonical)
    with pytest.raises(ValueError, match="pointmaps"):
        refine(mesh, Pose.identity(), views, render, match)


# --- competing free-rotation hypothesis ------------------------------------

def tilted_scene(tilt_rad=0.5):
    """Same setup, but the truth pose is genuinely tilted -- the case a
    yaw-only delta provably cannot reach."""
    mesh, truth, views, canonical = scene()
    tilt = np.array([[1.0, 0, 0],
                     [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
                     [0, np.sin(tilt_rad), np.cos(tilt_rad)]])
    truth = Pose(R=tilt @ truth.R, scale=truth.scale, t=truth.t)

    H, W = views[0].image_hw
    world = truth.apply(canonical)
    for v in views:
        uv, cam, _z = project(world, v.intrinsic, v.cam2world)
        ij = np.round(uv).astype(int)
        inside = ((ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H))
        pm = np.zeros((H, W, 3))
        pm[ij[inside, 1], ij[inside, 0]] = cam[inside]
        v.pointmap = pm
    return mesh, truth, views, canonical


def test_upright_mode_pins_pitch_and_roll_to_zero():
    """The plausibility guarantee: whatever the init's lean, the result stands."""
    mesh, truth, views, canonical = tilted_scene()
    render, match = make_backends(truth, canonical)
    start = Pose(R=rz(0.3) @ Y_UP.R_up, scale=np.full(3, 1.0), t=np.zeros(3))

    out = refine(mesh, start, views, render, match, iters=4, tilt_mode="upright")
    assert tilt_of(out.pose.R) < 1e-9            # started upright, stayed upright
    assert not out.used_free_rotation


def test_free_rotation_recovers_a_genuinely_tilted_object():
    mesh, truth, views, canonical = tilted_scene()
    render, match = make_backends(truth, canonical)
    start = Pose(R=rz(0.3) @ Y_UP.R_up, scale=np.full(3, 1.0), t=np.zeros(3))

    out = refine(mesh, start, views, render, match, iters=4, tilt_mode="auto")
    assert out.used_free_rotation
    assert abs(tilt_of(out.pose.R) - tilt_of(truth.R)) < 3.0
    assert np.isclose(out.tilt_deg, tilt_of(out.pose.R))


def test_free_rotation_is_not_taken_when_the_object_is_upright():
    """Extra degrees of freedom always lower the residual, so an undemanding
    test would hand every upright object a tilted pose."""
    mesh, truth, views, canonical = scene()          # upright truth
    render, match = make_backends(truth, canonical)
    start = Pose(R=rz(0.35) @ Y_UP.R_up, scale=np.full(3, 1.0), t=np.zeros(3))

    out = refine(mesh, start, views, render, match, iters=4, tilt_mode="auto")
    assert tilt_of(out.pose.R) < 1.0


# --- the pooled cloud is not part of the solve path ------------------------

def test_refinement_runs_without_any_object_points():
    """`Observation.points` seeds a search; refinement never reads it.

    Guards the claim the docs make -- if this ever starts failing, the cloud
    has quietly become load-bearing and the API should say so again.
    """
    mesh, truth, views, canonical = scene()
    render, match = make_backends(truth, canonical)
    start = Pose(R=rz(0.35) @ Y_UP.R_up, scale=np.full(3, 1.0), t=np.zeros(3))

    out = refine(mesh, start, views, render, match, iters=4)
    assert out.iterations >= 1
    assert abs(yaw_of(out.pose.R) - yaw_of(truth.R)) < 0.02


def test_observation_without_points_is_valid():
    from meshfit.views import Observation

    _mesh, _truth, views, _canonical = scene()
    obs = Observation(views=views)
    obs.validate()
    assert obs.points is None


def test_asking_for_extent_without_points_says_what_to_do():
    import pytest

    from meshfit.views import Observation

    _mesh, _truth, views, _canonical = scene()
    with pytest.raises(ValueError, match="object_points_from_views"):
        _ = Observation(views=views).extent
