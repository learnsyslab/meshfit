"""Seeing whether an alignment is actually right.

Two views of the same answer, because they fail differently:

`overlay` composites the placed mesh back onto the photograph. This is the one
that catches convention bugs -- a mirrored camera axis or a swapped up vector
produces a pose with a perfectly respectable residual and an overlay that is
obviously, visibly wrong.

`show` puts the mesh, the observed points and the cameras in one interactive
3D scene. This is the one that catches depth errors, which an overlay cannot
show at all: an object at twice the distance and twice the size reprojects
onto exactly the same pixels.

viser and PIL are imported lazily -- they are dev extras, and nothing in the
solve path should need them.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .io import scene_points_from_views
from .metrics import mask_iou, silhouette
from .pose import Pose
from .views import Observation, View

__all__ = ["DebugSink", "mask_iou", "overlay", "overlay_panel",
           "save_overlay", "show", "silhouette"]

# green = where the mesh landed, red = where the object actually is,
# yellow = agreement. Matching digital-sister's posealign overlay.
PREDICTED = np.array([40, 255, 80], dtype=np.float32)
OBSERVED = np.array([255, 40, 80], dtype=np.float32)
AGREEMENT = np.array([255, 230, 40], dtype=np.float32)


# ---------------------------------------------------------------------------
# 2D: mesh composited back onto the photograph
# ---------------------------------------------------------------------------


def overlay(view: View, mesh, pose: Pose, render=None, alpha: float = 0.55) -> np.ndarray:
    """RGB with the predicted and observed silhouettes tinted over it.

    Read it by looking for red and green: yellow means the two agree, a green
    fringe means the mesh spills past the object, a red fringe means it falls
    short. Large disjoint patches of each mean the pose is simply wrong.
    """
    predicted = silhouette(view, mesh, pose, render)
    observed = view.mask
    both = predicted & observed

    out = view.rgb.astype(np.float32)
    for region, colour, a in ((predicted & ~both, PREDICTED, alpha),
                              (observed & ~both, OBSERVED, alpha),
                              (both, AGREEMENT, 0.75)):
        out[region] = out[region] * (1.0 - a) + colour * a
    return out.clip(0, 255).astype(np.uint8)


def overlay_panel(view: View, mesh, pose: Pose, render=None) -> np.ndarray:
    """`overlay` beside the bare render, when a renderer is available.

    Seeing the render alone matters: a black or garbled panel says the RENDERER
    is broken, which otherwise looks identical to a bad pose.
    """
    composited = overlay(view, mesh, pose, render)
    if render is None:
        return composited
    drawn = render(_placed(mesh, pose), view.intrinsic, view.cam2world,
                   view.image_hw)
    return np.concatenate([drawn.rgb, composited], axis=1)


def save_overlay(path, view: View, mesh, pose: Pose, render=None) -> None:


    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_panel(view, mesh, pose, render)).save(path, quality=95)


# ---------------------------------------------------------------------------
# 3D: the interactive scene
# ---------------------------------------------------------------------------


def show(
    observation: Observation,
    mesh,
    pose: Pose,
    *,
    before: Pose | None = None,
    context: bool = True,
    port: int = 8080,
    point_size: float = 0.004,
    frustum_scale: float = 0.15,
    block: bool = True,
):
    """Open a viser scene showing where the mesh ended up.

    Points carry their real pixel colours, and by default the WHOLE frame is
    drawn, not just the object. Both matter for reading the picture: a cloud of
    uniformly tinted object points floating in space tells you nothing about
    whether the object is sitting on the desk or hovering above it, which is
    exactly the error an overlay cannot show either.

    Pass `before` to draw the initialisation alongside the result -- each layer
    gets its own checkbox, so "did refinement help" becomes something you look
    at rather than infer from two residuals.
    """
    import viser

    server = viser.ViserServer(port=port)
    # label -> the scene handles it owns. viser's SceneApi cannot be indexed by
    # name, so visibility has to go through the handle each `add_*` returns.
    layers: dict[str, list] = {}

    if context:
        try:
            pts, cols = scene_points_from_views(observation.views)
            layers["scene"] = [server.scene.add_point_cloud(
                "/scene", points=pts.astype(np.float32), colors=cols,
                point_size=point_size * 0.75)]
        except ValueError:
            pass                       # no pointmaps; object points still show

    if observation.points is not None:
        layers["object"] = [server.scene.add_point_cloud(
            "/object",
            points=observation.points.astype(np.float32),
            colors=(observation.colors if observation.colors is not None
                    else np.full((len(observation.points), 3), OBSERVED, dtype=np.uint8)),
            point_size=point_size * 1.5,   # bigger: object, not backdrop
        )]

    layers["aligned"] = [_add_mesh(server, "/aligned", mesh, pose, colour=(80, 220, 120))]
    if before is not None:
        layers["initial"] = [_add_mesh(server, "/initial", mesh, before,
                                       colour=(150, 150, 165))]

    frusta = []
    for i, view in enumerate(observation.views):
        H, W = view.image_hw
        frusta.append(server.scene.add_camera_frustum(
            f"/cameras/view_{i:03d}",
            fov=float(2 * np.arctan2(H / 2, view.intrinsic[1, 1])),
            aspect=W / H,
            scale=frustum_scale,
            wxyz=_wxyz(view.cam2world[:3, :3]),
            position=view.cam2world[:3, 3].astype(np.float32),
            image=view.rgb,
        ))
    layers["cameras"] = frusta

    _add_controls(server, layers)
    if block:
        print(f"viser running at http://localhost:{port}  (ctrl-c to stop)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    return server


def _add_mesh(server, name: str, mesh, pose: Pose, colour):
    return server.scene.add_mesh_simple(
        name,
        vertices=pose.apply(np.asarray(mesh.vertices, dtype=float)).astype(np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=colour,
        flat_shading=False,
    )


def _add_controls(server, layers: dict[str, list]) -> None:
    """One checkbox per layer, toggling the handles that layer owns.

    Handles, not names: `server.scene["/aligned"]` raises, because viser's
    SceneApi has no lookup by path -- the object each `add_*` call returns is
    the only way to reach a node again.

    `initial` starts hidden so the first thing on screen is the answer, with
    the before-state one click away.
    """
    for label, handles in layers.items():
        visible = label != "initial"
        for handle in handles:
            handle.visible = visible
        box = server.gui.add_checkbox(label, initial_value=visible)

        @box.on_update
        def _(event, handles=handles) -> None:
            for handle in handles:
                handle.visible = event.target.value


def _placed(mesh, pose: Pose):
    placed = mesh.copy()
    placed.vertices = pose.apply(np.asarray(mesh.vertices, dtype=float))
    return placed


def _wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w,x,y,z), which is viser's convention.

    Quaternions are banned from meshfit's JSON output precisely because wxyz
    and xyzw get confused; this is a viser-facing detail, kept local.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
             (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return np.array(q, dtype=np.float32)


# ---------------------------------------------------------------------------
# debug sink
# ---------------------------------------------------------------------------


class DebugSink:
    """Writes every intermediate image a fit produces, to a directory.

    Almost every failure in this pipeline is only diagnosable by eye: a mesh
    rendering flat grey, a matcher firing on the background, a yaw sweep whose
    peaks are all equal. Numbers report that a stage was rejected; these
    pictures say why.

    Files are named `<NN>_<stage>_<what>.jpg` so an alphabetical listing is
    chronological.
    """

    def __init__(self, root, quality: int = 88):

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self._n = 0

    def _path(self, stage: str, what: str, ext: str = "jpg"):
        self._n += 1
        return self.root / f"{self._n:03d}_{stage}_{what}.{ext}"

    def image(self, stage: str, what: str, rgb: np.ndarray) -> None:

        path = self._path(stage, what)
        img = Image.fromarray(np.asarray(rgb).astype(np.uint8))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(path, quality=self.quality)

    def render(self, stage: str, view_index: int, drawn) -> None:
        """The render, and its silhouette beside it."""
        rgb = np.asarray(drawn.rgb)
        mask = np.repeat(np.asarray(drawn.mask)[:, :, None], 3, axis=2) * np.uint8(255)
        self.image(stage, f"view{view_index:02d}_render", np.concatenate([rgb, mask], axis=1))

    def matches(self, stage: str, view_index: int, rendered: np.ndarray,
                observed: np.ndarray, k0: np.ndarray, k1: np.ndarray,
                max_lines: int = 120) -> None:
        """Render and observation side by side, with correspondence lines.

        The single most informative debug image: it shows at once whether the
        render looks like the object, whether the matcher fired at all, and
        whether the lines are coherent or a random spray.
        """
        import cv2

        H, W = rendered.shape[:2]
        canvas = np.concatenate([rendered, _fit_to(observed, (H, W))], axis=1)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        if len(k0):
            step = max(1, len(k0) // max_lines)
            for i in range(0, len(k0), step):
                p0 = (int(k0[i, 0]), int(k0[i, 1]))
                p1 = (int(k1[i, 0]) + W, int(k1[i, 1]))
                cv2.line(canvas, p0, p1, (0, 220, 0), 1, cv2.LINE_AA)
                cv2.circle(canvas, p0, 2, (0, 0, 255), -1)
                cv2.circle(canvas, p1, 2, (255, 0, 0), -1)
        cv2.putText(canvas, f"{len(k0)} matches", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(self._path(stage, f"view{view_index:02d}_matches")), canvas)

    def overlay(self, stage: str, view, mesh, pose, render=None) -> None:
        self.image(stage, "overlay", overlay(view, mesh, pose, render))

    def sweep(self, stage: str, scores: np.ndarray, yaws: np.ndarray,
              thumbnails: list | None = None) -> None:
        """The yaw sweep's score curve, and optionally the hypotheses.

        A flat or many-peaked curve here IS the symmetry warning, seen directly.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=120)
        ax.plot(np.degrees(yaws), scores, marker="o", ms=3, lw=1.2)
        best = int(np.argmax(scores))
        ax.axvline(np.degrees(yaws[best]), color="tab:red", lw=1, ls="--")
        ax.set_xlabel("yaw [deg]")
        ax.set_ylabel("silhouette IoU")
        ax.set_title(f"best {np.degrees(yaws[best]):.0f} deg  IoU {scores[best]:.3f}")
        fig.tight_layout()
        fig.savefig(self._path(stage, "yaw_sweep", ext="png"))
        plt.close(fig)

        if thumbnails:
            self.image(stage, "yaw_hypotheses", _contact_sheet(thumbnails))


def _fit_to(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if img.shape[:2] == tuple(hw):
        return img
    return np.asarray(Image.fromarray(img).resize((hw[1], hw[0])))


def _contact_sheet(images: list, columns: int = 6) -> np.ndarray:
    rows = []
    for start in range(0, len(images), columns):
        chunk = list(images[start:start + columns])
        while len(chunk) < columns:
            chunk.append(np.zeros_like(chunk[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)
