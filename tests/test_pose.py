import numpy as np
import pytest

from meshfit.pose import Pose
from meshfit.solver import rz


def test_scalar_scale_is_broadcast():
    assert np.allclose(Pose(R=np.eye(3), scale=2.0, t=np.zeros(3)).scale, [2, 2, 2])


def test_apply_scales_in_canonical_axes_before_rotating():
    """Order matters: scale acts in canonical axes, THEN rotation."""
    pose = Pose(R=rz(np.pi / 2), scale=np.array([2.0, 1.0, 1.0]), t=np.zeros(3))
    # +X canonical is stretched by 2, then rotated onto +Y
    assert np.allclose(pose.apply(np.array([[1.0, 0.0, 0.0]])), [[0.0, 2.0, 0.0]])


def test_matrix_excludes_scale():
    """T_world_canonical is the RIGID part; scale stays separate."""
    pose = Pose(R=np.eye(3), scale=np.array([2.0, 3.0, 4.0]), t=np.array([1.0, 0, 0]))
    T = pose.matrix()
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [1, 0, 0])


def test_scale_metric_is_the_geometric_mean():
    pose = Pose(R=np.eye(3), scale=np.array([1.0, 2.0, 4.0]), t=np.zeros(3))
    assert np.isclose(pose.scale_metric, 2.0)


def test_isotropic_compose_matches_direct_application():
    base = Pose.similarity(R=rz(0.3), s=2.0, t=np.array([1.0, 2.0, 3.0]))
    delta = Pose.similarity(R=rz(-0.7), s=0.5, t=np.array([-1.0, 0.5, 0.2]))
    pts = np.random.default_rng(0).normal(size=(50, 3))
    assert np.allclose(base.compose(delta).apply(pts), delta.apply(base.apply(pts)))


def test_anisotropic_delta_is_refused():
    """{R diag(S)} is not closed under composition -- refuse rather than lie."""
    base = Pose.similarity(R=np.eye(3), s=1.0, t=np.zeros(3))
    aniso = Pose(R=np.eye(3), scale=np.array([2.0, 1.0, 1.0]), t=np.zeros(3))
    with pytest.raises(ValueError, match="not closed under composition"):
        base.compose(aniso)


def test_anisotropic_base_survives_an_isotropic_delta():
    """The legal direction: similarity on the LEFT of an anisotropic pose."""
    base = Pose(R=rz(0.4), scale=np.array([2.0, 1.0, 0.5]), t=np.array([1.0, 0, 0]))
    delta = Pose.similarity(R=rz(0.2), s=1.5, t=np.array([0.0, 1.0, 0.0]))
    pts = np.random.default_rng(1).normal(size=(50, 3))
    assert np.allclose(base.compose(delta).apply(pts), delta.apply(base.apply(pts)))


def test_to_dict_omits_scale_aniso_when_isotropic():
    d = Pose.similarity(R=np.eye(3), s=2.0, t=np.zeros(3)).to_dict()
    assert "scale_aniso" not in d
    assert np.isclose(d["scale_metric"], 2.0)


def test_to_dict_emits_scale_aniso_when_anisotropic():
    pose = Pose(R=np.eye(3), scale=np.array([1.0, 2.0, 4.0]), t=np.zeros(3))
    d = pose.to_dict(projected=True, tilt_removed_deg=1.5)
    assert d["scale_aniso"] == [1.0, 2.0, 4.0]
    assert np.isclose(d["scale_metric"], 2.0)          # geometric mean fallback
    assert d["projected"] is True
