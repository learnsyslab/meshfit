"""Canonical-frame conventions.

A generator emits its mesh in its own frame -- origin-centred, roughly unit
scale, some default orientation. The only thing meshfit needs from that frame
is which axis points up, because that is what "upright" is measured against.

glTF specifies +Y as up, so `DEFAULT_UP` is "+Y" and GLB input usually needs no
thought. But the format convention and the object's canonicalisation are
different claims: a generator can emit a chair lying on its side inside a
perfectly valid GLB, and OBJ/PLY carry no convention at all while USD stages
from Omniverse are authored Z-up. Hence the override.

A wrong up axis does not raise and does not inflate the residual -- it is a
different self-consistent parameterisation, so the solver returns a confident
fit with the object on its side. `meshfit.diagnostics` carries the cheap
plausibility check that catches it; this module just does the geometry.

The WORLD is assumed Z-up and metric. That is a precondition, not a parameter:
USD, IsaacLab, MuJoCo and Isaac Sim are all Z-up, and a differently-oriented
world is one change of basis the caller applies once at the boundary rather
than something threaded through every solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: World up. Fixed by precondition -- see the module docstring.
WORLD_UP = np.array([0.0, 0.0, 1.0])

#: glTF's convention, and therefore the right default for GLB input.
DEFAULT_UP = "+Y"

_AXES = {
    "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
}


def axis_vector(spec: str | np.ndarray) -> np.ndarray:
    """Accept '+Y', 'y', '-Z' or a raw 3-vector; return a unit vector."""
    if not isinstance(spec, str):
        v = np.asarray(spec, dtype=float).reshape(3)
        n = np.linalg.norm(v)
        if n < 1e-9:
            raise ValueError("axis vector is degenerate")
        return v / n

    key = spec.strip().upper()
    if len(key) == 1:
        key = "+" + key
    if key not in _AXES:
        raise ValueError(f"unknown axis {spec!r}; expected one of {sorted(_AXES)}")
    return np.array(_AXES[key])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation carrying unit vector `a` onto unit vector `b`.

    Rodrigues, with the antipodal case resolved to a fixed perpendicular axis
    so that repeated runs agree (any perpendicular would be geometrically
    valid, but a nondeterministic choice makes poses irreproducible).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.cross(a, b)
    cos = float(a @ b)
    sin = float(np.linalg.norm(v))

    if sin > 1e-12:
        K = _skew(v)
        return np.eye(3) + K + K @ K * ((1.0 - cos) / sin**2)
    if cos > 0:
        return np.eye(3)

    seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis = np.cross(a, seed)
    K = _skew(axis / np.linalg.norm(axis))
    return np.eye(3) + 2.0 * (K @ K)          # 180 deg about `axis`


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


@dataclass(frozen=True)
class Frame:
    """A generator's canonical frame, related to the Z-up world.

    up    the canonical up axis, in canonical coordinates
    R_up  rotation taking canonical coordinates to Z-up, so `R_up @ up == +Z`

    Constrained solvers only ever produce rotations of the form
    `Rz(yaw) @ R_up`, which is exactly why they cannot return a tilted object.
    """

    up: np.ndarray
    R_up: np.ndarray

    @classmethod
    def from_up(cls, canonical_up: str | np.ndarray = DEFAULT_UP) -> Frame:
        up = axis_vector(canonical_up)
        return cls(up=up, R_up=rotation_between(up, WORLD_UP))

    def __post_init__(self) -> None:
        if not np.allclose(self.R_up @ self.up, WORLD_UP, atol=1e-8):
            raise ValueError("Frame.R_up does not carry Frame.up onto world +Z")

    @property
    def is_axis_aligned(self) -> bool:
        """True when `R_up` is a signed permutation matrix.

        Per-axis quantities (anisotropic scales, their priors) are reordered
        between canonical and Z-up axes with `abs(R_up)`, which is only a
        permutation under this condition. Axis-aligned `canonical_up` values --
        every string in `_AXES` -- always satisfy it; an arbitrary vector need
        not.
        """
        M = np.abs(self.R_up)
        return bool(np.allclose(M.sum(axis=0), 1.0, atol=1e-8)
                    and np.allclose(M.max(axis=0), 1.0, atol=1e-8))


#: The common case: a glTF/TRELLIS-style +Y-up mesh in a Z-up world.
Y_UP = Frame.from_up("+Y")

#: The world frame as its own canonical frame, so `R_up` is the identity. Use
#: it for world -> world fits, where both point sets are already Z-up and no
#: canonical rotation is needed.
WORLD = Frame.from_up("+Z")
