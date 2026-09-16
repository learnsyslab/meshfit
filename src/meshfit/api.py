"""The entry point: mesh in, pose out.

`fit` runs the three stages in order and records what each one did.

    initialise   adopt the generator's pose, or search for one
    refine       render, match, lift, solve a delta -- repeat
    polish       one joint least-squares pass in pixels

Every stage is optional and every stage is replaceable, because they fail
independently: a pose-aware backend makes the first redundant, no GPU makes the
second impossible, and the third only helps once correspondences exist.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from .correspond import Correspondence, stack_world_pairs
from .frames import DEFAULT_UP, Frame
from .init import Initialization, Symmetry, from_generator, from_search
from .metrics import mean_mask_iou
from .pose import Pose
from .refine import RefineResult, refine
from .solver import (
    fit_joint,
    fit_similarity,
    project_upright,
    reprojection_rms,
    tilt_of,
)
from .views import Observation


@dataclass
class Fit:
    """A pose, and enough context to judge whether to trust it."""

    pose: Pose
    canonical_up: str = DEFAULT_UP
    init_source: str = "unknown"
    tilt_mode: str = "auto"
    used_free_rotation: bool = False
    symmetry: Symmetry | None = None
    projected: bool = False
    tilt_removed_deg: float = 0.0
    matches: int = 0
    inliers: int = 0
    refine_iterations: int = 0
    reprojection_rms_px: float = float("nan")
    stages: list[dict] = field(default_factory=list)
    #: Conditions the caller should know about. Not errors -- the fit ran, and
    #: these say what it could and could not determine from the data given.
    warnings: list[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool | None:
        """True / False when symmetry was assessed, None when it was not.

        None is not False. A pose-aware generator hands over an orientation
        without evidence about whether the image could have determined it, and
        reporting that as "not ambiguous" tells a caller the opposite of the
        truth. Callers that must act on orientation should treat None as
        "unknown" and check, not as permission.
        """
        return None if self.symmetry is None else self.symmetry.is_ambiguous

    @property
    def trustworthy(self) -> bool:
        """Only when ambiguity was actually assessed AND came back low."""
        return self.symmetry is not None and not self.symmetry.is_ambiguous

    def to_dict(self) -> dict:
        """The published schema: the pose block, plus meshfit's own namespaces.

        `pose` matches digital-sister's scene contract exactly so consumers
        need no translation layer; everything meshfit adds lives under separate
        keys where it cannot collide.
        """
        payload = {
            "pose": self.pose.to_dict(projected=self.projected,
                                      tilt_removed_deg=self.tilt_removed_deg),
            "confidence": {
                "symmetry_margin": None if self.symmetry is None
                                   else float(self.symmetry.margin),
                "k_fold": None if self.symmetry is None else int(self.symmetry.k_fold),
                "ambiguous": self.ambiguous,          # None = never assessed
                "symmetry_assessed": self.symmetry is not None,
                "matches": int(self.matches),
                "inliers": int(self.inliers),
                "reprojection_rms_px": float(self.reprojection_rms_px),
            },
            "provenance": {
                "canonical_up": self.canonical_up,
                "init_source": self.init_source,
                "tilt_mode": self.tilt_mode,
                "free_rotation": bool(self.used_free_rotation),
                "refine_iterations": int(self.refine_iterations),
                "stages": self.stages,
                "warnings": list(self.warnings),
            },
        }
        return payload


def fit(
    mesh,
    observation: Observation,
    *,
    canonical_up: str = DEFAULT_UP,
    init: Pose | None = None,
    render=None,
    match=None,
    do_refine: bool = True,
    do_polish: bool = True,
    force_upright: bool = False,
    refine_iters: int = 3,
    yaw_steps: int = 36,
    gate_on_iou: bool = True,
    tilt_mode: str = "auto",
    min_tilt_deg: float = 20.0,
    debug=None,
    sigma_px: float = 1.0,
    sigma_depth: float = 0.05,
) -> Fit:
    """Place `mesh` into `observation`'s world frame.

    `init` is a pose-aware generator's estimate; without one, a yaw search runs
    instead, which needs `render`. `refine` and `polish` both need `render` and
    `match`, and are skipped with a recorded reason when either is missing --
    meshfit returns the best pose it could justify rather than pretending.

    `force_upright` projects tilt out of the FINAL pose. Off by default: some
    objects really are tilted, and the free-rotation hypothesis is reported in
    `tilt_removed_deg` so the caller can decide with the number in hand.

    `tilt_mode` chooses how rotation is handled: "upright" pins pitch and roll
    to zero, "free" re-estimates them, "auto" fits both and takes the free one
    only when it leans more than `min_tilt_deg` and renders better.

    `gate_on_iou` makes every stage prove itself: a stage's output is kept only
    if it renders to a better silhouette overlap than its input. Refinement is
    not monotonic -- on a textureless or repetitive object a matcher can be
    confidently, consistently wrong, and the resulting pose is worse than the
    one it started from. Measuring is cheap; assuming improvement is not.
    """
    frame = Frame.from_up(canonical_up)
    observation.validate()

    initialisation = _initialise(mesh, observation, init, render, match, frame,
                                 yaw_steps, debug)
    result = Fit(pose=initialisation.pose, canonical_up=canonical_up,
                 init_source=initialisation.source, tilt_mode=tilt_mode,
                 symmetry=initialisation.symmetry)
    result.stages.append({"stage": "init", "source": initialisation.source})

    gate = _Gate(mesh, observation, render) if gate_on_iou else None
    if gate is not None:
        result.stages[-1]["mask_iou"] = round(gate.score(result.pose), 4)

    if debug is not None and render is not None:
        debug.overlay("init", observation.views[0], mesh, result.pose, render)

    correspondences: list[Correspondence] = []
    if do_refine:
        correspondences = _refine_stage(mesh, observation, render, match, frame,
                                        refine_iters, result, gate, debug,
                                        tilt_mode, min_tilt_deg)
        if debug is not None and render is not None:
            debug.overlay("refine", observation.views[0], mesh, result.pose, render)

    if do_polish and correspondences:
        _warn_polish_limits(observation, result, frame)
        _polish_stage(correspondences, frame, sigma_px, sigma_depth, result, gate,
                      free_rotation=result.used_free_rotation)
        if debug is not None and render is not None:
            debug.overlay("polish", observation.views[0], mesh, result.pose, render)

    if correspondences:
        result.reprojection_rms_px = reprojection_rms(result.pose, correspondences)

    if force_upright:
        tilt = tilt_of(result.pose.R, frame)
        if tilt > 1e-9:
            result.pose = project_upright(result.pose, frame)
            result.projected = True
            result.tilt_removed_deg = tilt
            result.stages.append({"stage": "project_upright", "tilt_deg": tilt})
    elif not result.projected:
        # nothing was projected away; report the tilt that REMAINS, which is
        # what a caller needs to decide whether to trust the pose
        result.tilt_removed_deg = tilt_of(result.pose.R, frame)

    return result


def _initialise(mesh, observation, init, render, match, frame, yaw_steps,
                debug=None) -> Initialization:
    if init is not None:
        return from_generator(init.matrix(), init.scale, frame)
    if render is None:
        raise ValueError("no init pose and no renderer: a pose-blind generator "
                         "needs `render` so the yaw search can score hypotheses")
    return from_search(mesh, observation, render, match, frame,
                       yaw_steps=yaw_steps, debug=debug)


def _warn_polish_limits(observation: Observation, result: Fit, frame: Frame) -> None:
    """Say what the joint solve can and cannot determine here.

    Neither condition is an error and neither disables the stage -- the gate
    already rejects a polish that makes things worse. They are reported so the
    caller can decide whether the stage is worth its time on their data.
    """
    if len(observation.views) == 1:
        result.warnings.append(
            "single view: reprojection alone is gauge-degenerate along the "
            "viewing ray, so scale and depth rest entirely on the point cloud "
            "prior and the polish is only as good as that depth. "
            "Re-run with --no-polish to skip it.")

    tilt = tilt_of(result.pose.R, frame)
    if tilt > 1.0 and not result.used_free_rotation:
        result.warnings.append(
            f"pose leans {tilt:.1f} deg but refinement settled on the upright "
            f"hypothesis, so polish runs in its 7-parameter form and will not "
            f"keep that lean. Use --tilt free if the lean is real.")


class _Gate:
    """Scores a pose by rendered silhouette overlap, averaged over views.

    Silhouette IoU is blind to depth -- an object at twice the distance and
    twice the size scores identically -- so this cannot certify a pose. It is
    used only to REJECT a stage that made the overlap visibly worse, which is a
    failure it does detect reliably.
    """

    def __init__(self, mesh, observation, render):
        self.mesh, self.observation, self.render = mesh, observation, render

    def score(self, pose: Pose) -> float:
        return mean_mask_iou(self.observation.views, self.mesh, pose, self.render)

    def accepts(self, before: Pose, after: Pose, tol: float = 1e-4
                ) -> tuple[bool, float, float]:
        a, b = self.score(before), self.score(after)
        return b >= a - tol, a, b


def _refine_stage(mesh, observation, render, match, frame, iters, result,
                  gate=None, debug=None, tilt_mode: str = "auto",
                  min_tilt_deg: float = 20.0) -> list[Correspondence]:
    if render is None or match is None:
        result.stages.append({"stage": "refine", "skipped": "needs render and match"})
        return []

    out: RefineResult = refine(mesh, result.pose, observation.views, render, match,
                               frame, iters=iters, debug=debug,
                               tilt_mode=tilt_mode, min_tilt_deg=min_tilt_deg)
    if out.found_nothing:
        result.stages.append({"stage": "refine", "skipped": "no usable matches"})
        return []

    record = {"stage": "refine", "iterations": out.iterations,
              "matches": out.matches, "inliers": out.inliers,
              "converged": out.converged,
              "free_rotation": bool(out.used_free_rotation),
              "tilt_deg": round(float(out.tilt_deg), 3)}

    accepted = True
    if gate is not None:
        accepted, before, after = gate.accepts(result.pose, out.pose)
        record |= {"accepted": accepted, "mask_iou_before": round(before, 4),
                   "mask_iou_after": round(after, 4)}

    if accepted:
        result.pose = out.pose
        result.matches, result.inliers = out.matches, out.inliers
        result.refine_iterations = out.iterations
        result.used_free_rotation = out.used_free_rotation
    result.stages.append(record)

    # Correspondences are returned either way: they were found at the refined
    # pose, but they are real image matches, and the polish stage is gated too.
    return out.correspondences


def _polish_stage(correspondences, frame, sigma_px, sigma_depth, result,
                  gate=None, free_rotation: bool = False) -> None:
    """Joint solve, parameterised to match what refinement concluded.

    If refinement settled on an upright pose, polish keeps pitch and roll at
    zero (7 parameters). If it found a real lean, polish keeps and refines it
    (9 parameters). Choosing per object avoids the failure where a stage
    dedicated to improving the pose instead undoes the previous stage's
    finding.
    """
    before = reprojection_rms(result.pose, correspondences)
    with warnings.catch_warnings():
        # `fit` has already reported this in result.warnings; emitting it again
        # on stderr just makes the same point twice.
        warnings.filterwarnings("ignore", message="fit_joint is in its")
        polished = fit_joint(correspondences, result.pose, frame,
                             sigma_px=sigma_px, sigma_depth=sigma_depth,
                             free_rotation=free_rotation)
    after = reprojection_rms(polished, correspondences)

    # Two independent checks, because they catch different failures: the
    # optimiser may not reduce this particular metric, and reducing it against
    # wrong correspondences still moves the object off the object.
    record = {"stage": "polish", "params": 9 if free_rotation else 7,
              "reproj_px_before": round(float(before), 3),
              "reproj_px_after": round(float(after), 3)}
    accepted = after <= before
    if accepted and gate is not None:
        accepted, iou_before, iou_after = gate.accepts(result.pose, polished)
        record |= {"mask_iou_before": round(iou_before, 4),
                   "mask_iou_after": round(iou_after, 4)}

    if accepted:
        # fit_joint parameterises yaw only, so accepting it straightens the
        # object. That is a projection whether or not anyone asked for one, and
        # the record has to say so -- a `projected: false` on a pose that was
        # in fact straightened is worse than no field at all.
        removed = tilt_of(result.pose.R, frame) - tilt_of(polished.R, frame)
        if removed > 1e-6:
            result.projected = True
            result.tilt_removed_deg = float(removed)
            record["tilt_removed_deg"] = round(float(removed), 3)
        result.pose = polished
    record["accepted"] = bool(accepted)
    result.stages.append(record)


def free_rotation_hypothesis(correspondences: list[Correspondence]) -> tuple[Pose, float]:
    """Fit with rotation unconstrained, and report the tilt it wants.

    The competing hypothesis: when this fits decisively better than the upright
    answer, the object is probably genuinely tilted -- or `canonical_up` is
    wrong. Both are worth surfacing, and the residual cannot tell them apart.
    """

    x, y = stack_world_pairs(correspondences)
    pose = fit_similarity(x, y)
    return pose, tilt_of(pose.R)
