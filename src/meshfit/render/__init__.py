"""Rendering a placed mesh into a view.

    blender      the renderer itself: scene, camera, mesh upload, caching
    appearance   how a mesh's colours reach Blender -- textures, vertex colours

Split because they answer different questions. The camera conversion and the
upload caching are about mechanics and change when Blender changes; the
appearance code is about what a generator happened to export and changes when
generators change.
"""

from .appearance import apply_appearance, enable_cycles_gpu
from .blender import SENSOR_MM, BlenderRenderer

__all__ = ["SENSOR_MM", "BlenderRenderer", "apply_appearance", "enable_cycles_gpu"]
