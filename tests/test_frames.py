import numpy as np
import pytest

from meshfit.frames import WORLD_UP, Frame, axis_vector, rotation_between

AXES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


@pytest.mark.parametrize("spec", AXES)
def test_frame_carries_its_up_axis_to_world_up(spec):
    f = Frame.from_up(spec)
    assert np.allclose(f.R_up @ f.up, WORLD_UP)


@pytest.mark.parametrize("spec", AXES)
def test_frame_rotation_is_proper(spec):
    f = Frame.from_up(spec)
    assert np.allclose(f.R_up @ f.R_up.T, np.eye(3))
    assert np.isclose(np.linalg.det(f.R_up), 1.0)      # no reflections


@pytest.mark.parametrize("spec", AXES)
def test_axis_aligned_specs_permit_per_axis_scales(spec):
    assert Frame.from_up(spec).is_axis_aligned


def test_default_matches_posealign_constant():
    """The +Y default must reproduce posealign's hardcoded R_Y2Z exactly."""
    R_Y2Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert np.allclose(Frame.from_up("+Y").R_up, R_Y2Z)


def test_axis_vector_accepts_shorthand_and_vectors():
    assert np.allclose(axis_vector("y"), [0, 1, 0])
    assert np.allclose(axis_vector("+Y"), [0, 1, 0])
    assert np.allclose(axis_vector([0, 2, 0]), [0, 1, 0])      # normalised


def test_axis_vector_rejects_nonsense():
    with pytest.raises(ValueError):
        axis_vector("+Q")
    with pytest.raises(ValueError):
        axis_vector([0, 0, 0])


def test_antipodal_rotation_is_deterministic_and_correct():
    a, b = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    R1, R2 = rotation_between(a, b), rotation_between(a, b)
    assert np.allclose(R1 @ a, b)
    assert np.allclose(R1, R2)                                  # reproducible
    assert np.isclose(np.linalg.det(R1), 1.0)


def test_frame_rejects_inconsistent_construction():
    with pytest.raises(ValueError):
        Frame(up=np.array([0.0, 1.0, 0.0]), R_up=np.eye(3))
