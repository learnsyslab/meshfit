"""The pose type.

Scale is per-canonical-axis (3,) rather than a scalar. Isotropic is just the
special case [s, s, s]. This matters because the two cannot be juggled as
separate variables without them drifting apart -- see `compose`.

The composed map is

    p_world = R @ (scale * p_canonical) + t

i.e. scale acts FIRST, in canonical axes, where it has semantic meaning (the
object's own width/depth/height). `R` and `t` together are the rigid part that
digital-sister's contract calls `T_world_canonical`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    """A placement: rotation, per-axis metric scale, translation.

    The composed map is

        p_world = R @ (scale * p_canonical) + t

    so scale acts FIRST, in canonical axes, where it has semantic meaning --
    the object's own width, depth and height. `R` and `t` together are the
    rigid part that the published schema calls `T_world_canonical`; scale is
    kept separate so per-axis scale stays recoverable.

    Attributes:
        R: `(3, 3)` `float64`. World <- canonical rotation, right-handed,
            `det(R) == +1`.
        scale: `(3,)` `float64`. Metric scale per CANONICAL axis. A scalar is
            broadcast, so isotropic is just `[s, s, s]`.
        t: `(3,)` `float64`. Translation, metres, world frame.
        rms: Residual of the fit that produced this pose, in metres. `inf` for
            a pose that was constructed rather than fitted.
    """

    R: np.ndarray
    scale: np.ndarray
    t: np.ndarray
    rms: float = np.inf

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)
        s = np.asarray(self.scale, dtype=float)
        # a bare float is the isotropic case; broadcast it rather than making
        # callers remember to
        self.scale = np.full(3, float(s)) if s.ndim == 0 else s.reshape(3)

    # -- construction ----------------------------------------------------
    @classmethod
    def identity(cls) -> Pose:
        return cls(R=np.eye(3), scale=np.ones(3), t=np.zeros(3))

    @classmethod
    def similarity(cls, R: np.ndarray, s: float, t: np.ndarray, rms: float = np.inf) -> Pose:
        """Isotropic pose -- the only kind that is safe to `compose`."""
        return cls(R=R, scale=np.full(3, float(s)), t=t, rms=rms)

    # -- properties ------------------------------------------------------
    @property
    def is_isotropic(self) -> bool:
        return bool(np.allclose(self.scale, self.scale[0]))

    @property
    def scale_metric(self) -> float:
        """Geometric mean of the per-axis scales.

        Required by the scene contract: when `scale_aniso` is present,
        `scale_metric` must be its geometric mean so that consumers which do
        not handle anisotropy still get a sane isotropic fallback.
        """
        return float(np.cbrt(np.prod(self.scale)))

    # -- application -----------------------------------------------------
    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Canonical points -> world."""
        return (np.asarray(pts, dtype=float) * self.scale) @ self.R.T + self.t

    def inverse_apply(self, pts_world: np.ndarray) -> np.ndarray:
        """World points -> canonical. The inverse of `apply`.

        Used to recover which canonical point a lifted surface hit corresponds
        to, so image matches can be handed to a solver that re-estimates the
        pose from scratch.
        """
        centred = np.asarray(pts_world, dtype=float) - self.t
        return (centred @ self.R) / self.scale

    def matrix(self) -> np.ndarray:
        """The RIGID 4x4 (`T_world_canonical`). Scale is NOT baked in -- the
        contract keeps them separate so `scale_aniso` stays recoverable."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def copy(self) -> Pose:
        return Pose(R=self.R.copy(), scale=self.scale.copy(), t=self.t.copy(), rms=self.rms)

    # -- composition -----------------------------------------------------
    def compose(self, delta: Pose) -> Pose:
        """Apply an ISOTROPIC `delta` after self: world' = delta o self.

        Anisotropic deltas are rejected, and not out of caution -- the set
        {R diag(S)} is provably not closed under composition. Counterexample:
        diag(2,1,1) @ Rz(45) has non-orthogonal columns, so it equals no
        R' diag(S') whatsoever. Refinement therefore composes similarities on
        the LEFT, which IS closed:

            (s R') (R diag(S)) = (R'R) diag(s S)

        because a scalar commutes with everything. That is why anisotropy has
        to live innermost, in canonical axes, fixed across iterations.
        """
        if not delta.is_isotropic:
            raise ValueError(
                "cannot compose an anisotropic delta: {R diag(S)} is not closed "
                "under composition. Solve anisotropy jointly (see solve.polish) "
                "instead of iterating it."
            )
        s_d = float(delta.scale[0])
        return Pose(
            R=delta.R @ self.R,
            scale=s_d * self.scale,
            t=s_d * (delta.R @ self.t) + delta.t,
            rms=delta.rms,
        )

    # -- serialisation ---------------------------------------------------
    def to_dict(self, *, projected: bool | None = None,
                tilt_removed_deg: float | None = None) -> dict:
        """Serialise to the published pose schema (see README).

        `scale_aniso` appears only when the scale really is anisotropic -- an
        isotropic pose should not push consumers down the anisotropic path.
        """
        out: dict = {
            "T_world_canonical": self.matrix().tolist(),
            "scale_metric": self.scale_metric,
        }
        if not self.is_isotropic:
            out["scale_aniso"] = [float(v) for v in self.scale]
        if projected is not None:
            out["projected"] = bool(projected)
        if tilt_removed_deg is not None:
            out["tilt_removed_deg"] = float(tilt_removed_deg)
        return out
