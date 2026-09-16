"""Blender rendering.

The renderer's only job is to draw the mesh as it would appear from a given
camera. It does not produce depth: the 3D that refinement needs is recovered
exactly by intersecting the mesh (`lift_by_raycast`), which is both more
accurate than a depth buffer and keeps Blender's fast-moving compositor API out
of the critical path entirely.

What Blender buys over a bare rasteriser is appearance. Matching a properly lit
render against a photograph gives a feature matcher more to work with than
matching flat vertex colours against one. The cost is speed, which is why
`silhouette_only` exists -- the yaw sweep compares outlines, and an outline
needs no materials, no lighting and no samples.

Camera conventions, the classic source of silent errors here:

    meshfit / OpenCV    +X right, +Y down, +Z forward
    Blender             +X right, +Y up,   -Z forward

so the rotation is conjugated by diag(1, -1, -1) on the way in. A mistake here
does not raise -- it produces a plausible render that is subtly mirrored, which
is exactly what `meshfit.viz.overlay` is for.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image

from ..views import Render
from .appearance import apply_appearance, enable_cycles_gpu

#: Blender's sensor width is arbitrary -- only the lens/sensor ratio matters,
#: so any value works as long as `lens` is derived from the same one.
SENSOR_MM = 36.0


class BlenderRenderer:
    """Draw meshes with Blender. Satisfies the `Renderer` protocol.

    The scene is built once and reused: adding and removing a mesh is cheap,
    while `read_factory_settings` is not, and a yaw sweep re-renders the same
    object dozens of times. Call `close()` when finished, or use it as a
    context manager.
    """

    def __init__(self, *, engine: str = "BLENDER_EEVEE", samples: int = 16,
                 background: float = 0.05, light_energy: float = 1.0,
                 key_energy: float = 2.5, key_angle_deg: float = 25.0,
                 shading: bool = False, use_gpu: bool = True):
        import bpy

        self._bpy = bpy
        self.engine = engine
        self.samples = samples
        self._object = None

        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.render.engine = engine
        scene.render.film_transparent = True          # alpha IS the silhouette
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        if engine == "CYCLES":
            scene.cycles.samples = samples
            if use_gpu:
                enable_cycles_gpu(bpy)

        # Uniform environment, and `shading` OFF by default. Under uniform
        # hemispherical light a Lambertian surface receives the same irradiance
        # whatever its normal, so this renders flat albedo: no shape cue, and
        # measurably less contrast than the photograph (42 against 61, with no
        # pixel below 65 where the photograph reaches 31).
        #
        # That looks like a defect and is not. We added a key light and
        # ray-traced occlusion to close that gap and measured what it did to
        # RoMa's on-object matches:
        #
        #     object          uniform   mild    harsh
        #     paper roll         3776   2953     2329
        #     drawer             4842   4797     4660
        #     controller         3446   3452     3726
        #
        # Two of three get monotonically WORSE. A dense matcher trained on real
        # image pairs is already illumination-robust, so shading gives it no
        # cue it wanted -- it adds view-dependent structure that does not
        # correspond, because our light is not the scene's light. Albedo is the
        # thing that genuinely corresponds between a render and a photograph.
        #
        # `shading=True` remains available: it helped the controller, whose
        # near-black surface has little albedo variation to offer. But it is
        # not the default, and a low-texture object is not fixed by lighting.
        world = bpy.data.worlds.new("w")
        world.use_nodes = True
        bg = world.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        # ~1.0 keeps mid-tones near a typical photo exposure; brighter washes
        # out texture contrast, which is precisely what the matcher keys on.
        bg.inputs[1].default_value = light_energy
        scene.world = world
        scene.view_settings.view_transform = "Standard"   # no filmic tone curve

        if shading and engine == "BLENDER_EEVEE":
            scene.eevee.use_raytracing = True      # occlusion in the recesses
            scene.eevee.use_shadows = True
            scene.eevee.use_fast_gi = True
        self._add_key_light(bpy, scene, key_energy, key_angle_deg) if shading else None

        # One-entry caches. A refinement loop or yaw sweep renders the SAME
        # mesh and texture at many poses, and re-uploading either costs seconds:
        # encoding a 4096px texture is ~3.3s, rebuilding a 600k-vertex mesh
        # ~1.1s, while moving the vertices it already has is ~0.1s.
        self._key = None
        self._topology = None          # (n_vertices, n_faces) currently uploaded
        self._texture_key = None       # identifies the texture already loaded
        self._bl_image = None

        self._camera_data = bpy.data.cameras.new("meshfit_cam")
        self._camera_data.sensor_fit = "HORIZONTAL"
        self._camera_data.sensor_width = SENSOR_MM
        self._camera = bpy.data.objects.new("meshfit_cam", self._camera_data)
        scene.collection.objects.link(self._camera)
        scene.camera = self._camera
        self._scene = scene
        self._tmpdir = tempfile.mkdtemp(prefix="meshfit_render_")

    def _add_key_light(self, bpy, scene, energy: float, angle_deg: float) -> None:
        """A broad sun, offset slightly from the camera axis.

        Near-frontal so it cannot carve shadows the photograph does not have,
        and wide-angled so its shadows are soft. Its job is a gradient across
        curved surfaces -- the cue that conveys shape on an untextured object,
        and the reason a white drawer front is not a featureless blob.
        """
        data = bpy.data.lights.new("meshfit_key", type="SUN")
        data.energy = energy
        data.angle = np.radians(angle_deg)          # angular diameter: soft
        light = bpy.data.objects.new("meshfit_key", data)
        scene.collection.objects.link(light)
        self._key = light

    def _aim_key_light(self, c2w: np.ndarray) -> None:
        """Point the key along the viewing direction, tilted up and to the left.

        Re-aimed per view so a multi-view sequence is lit consistently relative
        to each camera rather than to the world.
        """
        if getattr(self, "_key", None) is None:
            return
        forward = c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])
        up = c2w[:3, :3] @ np.array([0.0, -1.0, 0.0])
        right = np.cross(forward, up)
        direction = forward + 0.45 * up + 0.35 * right
        direction /= np.linalg.norm(direction)
        # a SUN points down its local -Z
        z = -direction
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        M = np.eye(4)
        M[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
        M[:3, 3] = c2w[:3, 3]
        self._key.matrix_world = M.T.tolist()

    # -- protocol ---------------------------------------------------------
    def __call__(self, mesh, intrinsic: np.ndarray, cam2world: np.ndarray,
                 image_hw: tuple[int, int]) -> Render:
        rgba = self._render_rgba(mesh, intrinsic, cam2world, image_hw)
        return Render(rgb=rgba[..., :3].copy(), mask=rgba[..., 3] > 127)

    def silhouette_only(self, mesh, intrinsic: np.ndarray, cam2world: np.ndarray,
                        image_hw: tuple[int, int]) -> np.ndarray:
        """Just the outline, rendered as cheaply as Blender can manage.

        Workbench skips materials, lighting and sampling entirely. The yaw
        sweep scores outline overlap, so that is all it needs -- and it is the
        stage that runs dozens of times per object.
        """
        previous, self.engine = self.engine, "BLENDER_WORKBENCH"
        self._scene.render.engine = "BLENDER_WORKBENCH"
        try:
            return self._render_rgba(mesh, intrinsic, cam2world, image_hw)[..., 3] > 127
        finally:
            self.engine = previous
            self._scene.render.engine = previous

    # -- internals --------------------------------------------------------
    def _render_rgba(self, mesh, intrinsic, cam2world, image_hw) -> np.ndarray:
        self._set_camera(np.asarray(intrinsic, float), np.asarray(cam2world, float),
                         image_hw)
        self._aim_key_light(np.asarray(cam2world, float))
        self._set_mesh(mesh)

        path = os.path.join(self._tmpdir, "frame.png")
        self._scene.render.filepath = path
        self._bpy.ops.render.render(write_still=True)

        with Image.open(path) as im:
            return np.asarray(im.convert("RGBA"))

    def _set_camera(self, K: np.ndarray, c2w: np.ndarray,
                    image_hw: tuple[int, int]) -> None:
        H, W = image_hw
        render = self._scene.render
        render.resolution_x, render.resolution_y = W, H

        cam = self._camera_data
        cam.lens = float(K[0, 0]) * SENSOR_MM / W
        # principal-point offset, in units of sensor width for HORIZONTAL fit
        cam.shift_x = float(W / 2 - K[0, 2]) / W
        cam.shift_y = float(K[1, 2] - H / 2) / W
        # non-square pixels: Blender has one lens, so fy/fx becomes pixel aspect
        render.pixel_aspect_x = 1.0
        render.pixel_aspect_y = float(K[0, 0] / K[1, 1])

        M = np.eye(4)
        M[:3, :3] = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])   # OpenCV -> Blender
        M[:3, 3] = c2w[:3, 3]
        self._camera.matrix_world = M.T.tolist()               # bpy wants row-major

    def _set_mesh(self, mesh) -> None:
        bpy = self._bpy
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        topology = (len(vertices), len(faces))

        # Same topology means the caller is re-posing one object, so only the
        # coordinates have changed. Everything else -- faces, UVs, material,
        # texture -- is still valid and re-uploading it is pure waste.
        if self._object is not None and self._topology == topology:
            self._object.data.vertices.foreach_set("co", vertices.ravel().tolist())
            self._object.data.update()
            return

        self._clear_mesh()
        data = bpy.data.meshes.new("meshfit_mesh")
        data.from_pydata(vertices.tolist(), [], faces.tolist())
        data.update()
        apply_appearance(bpy, data, mesh, self)

        self._object = bpy.data.objects.new("meshfit_obj", data)
        self._scene.collection.objects.link(self._object)
        self._topology = topology

    def _blender_image(self, image):
        """Load a PIL texture into Blender, reusing the last one if unchanged.

        Keyed on size, mode and a prefix of the raw bytes: full-image hashing
        would cost as much as the encode we are avoiding, and a prefix already
        separates the textures a single fit ever sees.
        """
        key = (image.size, image.mode, hash(image.tobytes()[:65536]))
        if key == self._texture_key and self._bl_image is not None:
            return self._bl_image

        path = os.path.join(self._tmpdir, "basecolor.png")
        image.convert("RGB").save(path)
        self._bl_image = self._bpy.data.images.load(path)
        self._texture_key = key
        return self._bl_image

    def _clear_mesh(self) -> None:
        bpy = self._bpy
        if self._object is not None:
            data = self._object.data
            bpy.data.objects.remove(self._object, do_unlink=True)
            bpy.data.meshes.remove(data)
            self._object = None
            self._topology = None

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        self._clear_mesh()

    def __enter__(self) -> BlenderRenderer:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
