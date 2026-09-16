"""Getting a mesh's colours into Blender.

Nothing here is about rendering mechanics -- it is the question "what does this
mesh look like", which every mesh answers differently depending on what its
generator exported.

The order matters and is not cosmetic. UV textures first, vertex colours
second, flat grey last: a feature matcher works on what it can see, and baking
a 4096px texture down to per-vertex colours discards exactly the
high-frequency detail it keys on. On a real TRELLIS.2 mesh that difference was
two orders of magnitude in match count.
"""

from __future__ import annotations

import numpy as np


def apply_appearance(bpy, data, mesh, renderer) -> None:
    """Give the Blender mesh the best appearance the source mesh carries.

    UV textures first, vertex colours second, plain grey last. The order is not
    cosmetic: a feature matcher works on what it can see, and baking a 4096px
    texture down to per-vertex colours throws away exactly the high-frequency
    detail it keys on. On a real TRELLIS.2 mesh that difference was two orders of
    magnitude in match count.
    """
    material = bpy.data.materials.new("meshfit_mat")
    material.use_nodes = True
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    # Matte, and specular OFF entirely.
    #
    # A dielectric reflects a few percent of white across its whole surface,
    # and under a white environment that is a constant wash toward grey. It
    # costs saturation the photograph has: measured on a windmill, 0.212
    # against the photograph's 0.256, recovered to 0.261 by zeroing it.
    #
    # It is worth real matches on dark objects, where a white wash swamps what
    # little colour signal exists. RoMa on-object matches, specular on -> off:
    #
    #     game controller   3441 -> 4612   (+34%, saturation 0.029 -> 0.166)
    #     paper roll        3830 -> 3799
    #     toy figurine      2142 -> 2137
    #     drawer            4846 -> 4829   (saturation 0.004 either way)
    #     windmill          4952 -> 4952
    #
    # Note the contrast with adding a key light, which HURT (see the world
    # setup above). Both are the same question: does the render show what
    # CORRESPONDS to the photograph? A key light adds structure that depends on
    # where we put the light, and that structure is in one image only. A
    # specular wash is likewise absent from the object's albedo -- so removing
    # it helps for exactly the reason adding shading hurt.
    bsdf.inputs["Roughness"].default_value = 0.85
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0

    if not (_attach_uv_texture(bpy, data, mesh, tree, bsdf, renderer)
            or _attach_vertex_colors(bpy, data, mesh, tree, bsdf)):
        bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)
    data.materials.append(material)


def _attach_uv_texture(bpy, data, mesh, tree, bsdf, renderer) -> bool:
    """Wire the mesh's base-colour texture through its UVs."""
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if uv is None or image is None:
        return False

    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) != len(mesh.vertices):
        return False

    # NO v-flip here. glTF stores UVs top-left-origin and Blender is
    # bottom-left, but trimesh already normalises on load, so flipping again
    # sends every triangle to a different island of the atlas -- which on a
    # fragmented atlas renders as colourful speckle rather than an obvious
    # mirror, and is easy to mistake for a matcher problem.

    layer = data.uv_layers.new(name="UVMap")
    loop_vertex = np.empty(len(data.loops), dtype=np.int64)
    data.loops.foreach_get("vertex_index", loop_vertex)
    layer.uv.foreach_set("vector", uv[loop_vertex].ravel())

    # Via a file rather than `Image.pixels`: a 4096x4096 texture is 67M floats
    # and the round-trip through disk is far faster. Cached, because the encode
    # dominates everything else in a render.
    bl_image = renderer._blender_image(image)

    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = bl_image
    node.interpolation = "Linear"
    tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    return True


def _attach_vertex_colors(bpy, data, mesh, tree, bsdf) -> bool:
    colors = _vertex_colors_array(mesh)
    if colors is None:
        return False
    layer = data.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    flat = np.concatenate([colors, np.ones((len(colors), 1))], axis=1).ravel()
    layer.data.foreach_set("color", flat)

    node = tree.nodes.new("ShaderNodeAttribute")
    node.attribute_name = "Col"
    tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    return True


def _vertex_colors_array(mesh) -> np.ndarray | None:
    """Per-vertex RGB in [0,1], if the mesh carries any."""
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    try:
        if hasattr(visual, "to_color"):
            visual = visual.to_color()
        colors = np.asarray(visual.vertex_colors, dtype=np.float64)
    except Exception:
        return None
    if colors.ndim != 2 or len(colors) != len(mesh.vertices):
        return None
    return colors[:, :3] / 255.0


def enable_cycles_gpu(bpy) -> None:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for device in prefs.devices:
                device.use = device.type in (backend, "CPU")
            bpy.context.scene.cycles.device = "GPU"
            return
