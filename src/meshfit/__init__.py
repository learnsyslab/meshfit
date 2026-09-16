"""meshfit -- place a generated mesh into an observed scene.

Given a mesh in a generator's canonical frame, one or more RGB views with
object masks and calibrated cameras, and the gravity-aligned metric point
cloud those views came from, meshfit returns the pose that puts the mesh where
the object actually is: correct position, correct metric scale, upright.

It does not care which generator produced the mesh. Pose-aware backends
(SAM3D, recgen) can hand over their pose estimate as an initialisation; for
pose-blind ones (TRELLIS and most image-to-3D models) meshfit finds its own.
"""

from .frames import DEFAULT_UP, Y_UP, Frame
from .pose import Pose
from .views import Observation, View

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_UP",
    "Y_UP",
    "Frame",
    "Observation",
    "Pose",
    "View",
    "__version__",
]
