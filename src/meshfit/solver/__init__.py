"""Turning matched points into a pose.

Three kinds of solve, in the order the pipeline uses them:

    closed_form    direct fits under a constraint -- upright, free, anisotropic
    robust         RANSAC around any of them, which closed form makes affordable
    least_squares  the joint nonlinear solve, minimising pixels rather than metres

The split is not arbitrary. A closed-form fit is global given its
correspondences and costs microseconds, so a thousand RANSAC hypotheses are
free; the nonlinear solve can use the cost we actually want but is local and
needs correspondences that are already close. Neither does the other's job.

Names map onto digital-sister's posealign as:
    fit_upright      <- fit_constrained
    fit_similarity   <- fit_free
    fit_anisotropic  <- fit_aniso_z
"""

from .closed_form import fit_anisotropic, fit_similarity, fit_upright
from .least_squares import fit_joint, reprojection_rms
from .robust import ransac
from .rotations import project_upright, rz, tilt_of, yaw_of

__all__ = [
    "fit_anisotropic",
    "fit_joint",
    "fit_similarity",
    "fit_upright",
    "project_upright",
    "ransac",
    "reprojection_rms",
    "rz",
    "tilt_of",
    "yaw_of",
]
