import numpy as np
import pytest

from meshfit.frames import Y_UP, Frame
from meshfit.solver import (
    fit_anisotropic,
    fit_similarity,
    fit_upright,
    project_upright,
    rz,
    tilt_of,
    yaw_of,
)

RNG = np.random.default_rng(0)


def upright_pose_points(n=200, yaw=0.7, scale=1.4, t=(1.0, -2.0, 0.5), frame=Y_UP):
    """Canonical points plus their image under a known upright pose."""
    x = RNG.normal(size=(n, 3))
    R = rz(yaw) @ frame.R_up
    return x, (scale * x) @ R.T + np.asarray(t)


# -- fit_upright ------------------------------------------------------------

def test_fit_upright_recovers_an_upright_pose_exactly():
    x, y = upright_pose_points()
    pose = fit_upright(x, y)
    assert np.isclose(yaw_of(pose.R), 0.7)
    assert np.allclose(pose.scale, 1.4)
    assert np.allclose(pose.t, [1.0, -2.0, 0.5])
    assert pose.rms < 1e-9


def test_fit_upright_is_upright_even_when_the_data_is_tilted():
    """The guarantee: tilt is not in the parameter space, so outliers or a
    genuinely fallen object cannot tip the result over."""
    x = RNG.normal(size=(200, 3))
    R_tilted = rz(0.3) @ Frame.from_up("+Z").R_up      # deliberately wrong up
    y = (x @ R_tilted.T) + np.array([0.5, 0.5, 0.5])
    assert tilt_of(fit_upright(x, y).R) < 1e-9


@pytest.mark.parametrize("spec", ["+X", "+Y", "+Z", "-Y"])
def test_fit_upright_honours_the_declared_canonical_up(spec):
    frame = Frame.from_up(spec)
    x, y = upright_pose_points(frame=frame)
    pose = fit_upright(x, y, frame)
    assert pose.rms < 1e-9
    assert tilt_of(pose.R, frame) < 1e-9


# -- fit_similarity ---------------------------------------------------------

def test_fit_similarity_recovers_an_arbitrary_rotation():
    x = RNG.normal(size=(200, 3))
    R = np.linalg.qr(RNG.normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))
    y = (2.5 * x) @ R.T + np.array([1.0, 2.0, 3.0])
    pose = fit_similarity(x, y)
    assert np.allclose(pose.R, R)
    assert np.allclose(pose.scale, 2.5)
    assert pose.rms < 1e-9


def test_fit_similarity_never_returns_a_reflection():
    x = RNG.normal(size=(100, 3))
    y = x * np.array([1.0, 1.0, -1.0])          # mirrored: not a rotation
    assert np.linalg.det(fit_similarity(x, y).R) > 0


def test_similarity_beats_upright_on_a_genuinely_tilted_object():
    """The competing-hypothesis signal used to detect fallen objects."""
    x = RNG.normal(size=(200, 3))
    R = rz(0.4) @ Frame.from_up("+Z").R_up
    y = x @ R.T
    assert fit_similarity(x, y).rms < fit_upright(x, y).rms


# -- fit_anisotropic --------------------------------------------------------

def test_fit_anisotropic_recovers_per_axis_scales():
    x = RNG.normal(size=(400, 3))
    scale = np.array([2.0, 0.5, 1.5])
    R = rz(0.35) @ Y_UP.R_up
    y = (x * scale) @ R.T + np.array([0.2, 0.3, 0.4])
    pose = fit_anisotropic(x, y, prior=np.ones(3), reg=1e-8, init_theta=0.35)
    assert np.allclose(pose.scale, scale, atol=1e-6)
    assert pose.rms < 1e-6


def test_unobserved_axis_falls_back_to_its_prior():
    """A degenerate axis must hold the prior rather than collapse -- this is
    what keeps single-view objects from imploding along unseen depth."""
    x = RNG.normal(size=(300, 3))
    x[:, 1] = 0.0                                # canonical +Y carries no signal
    R = rz(0.0) @ Y_UP.R_up
    y = (x * np.array([1.0, 1.0, 1.0])) @ R.T
    pose = fit_anisotropic(x, y, prior=np.array([1.0, 0.75, 1.0]), reg=0.1)
    assert np.isclose(pose.scale[1], 0.75, atol=1e-6)


def test_fit_anisotropic_stays_upright():
    x = RNG.normal(size=(200, 3))
    y = (x * np.array([1.0, 2.0, 0.5])) @ (rz(0.9) @ Y_UP.R_up).T
    assert tilt_of(fit_anisotropic(x, y, prior=np.ones(3)).R) < 1e-9


def test_fit_anisotropic_requires_axis_aligned_up():
    frame = Frame.from_up([0.0, 1.0, 1.0])       # diagonal: not a permutation
    with pytest.raises(ValueError, match="axis-aligned"):
        fit_anisotropic(RNG.normal(size=(10, 3)), RNG.normal(size=(10, 3)),
                        prior=np.ones(3), frame=frame)


# -- angles -----------------------------------------------------------------

def test_project_upright_keeps_yaw_and_drops_tilt():
    x = RNG.normal(size=(100, 3))
    R = np.linalg.qr(RNG.normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))
    tilted = fit_similarity(x, x @ R.T)
    projected = project_upright(tilted)
    assert tilt_of(projected.R) < 1e-9
    assert np.isclose(yaw_of(projected.R), yaw_of(tilted.R))


def test_tilt_of_measures_degrees_from_vertical():
    assert np.isclose(tilt_of(Y_UP.R_up), 0.0)
    # tip the object 90 deg about world +X: its up axis becomes horizontal
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]) @ Y_UP.R_up
    assert np.isclose(tilt_of(R), 90.0)
