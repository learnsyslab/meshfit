"""RoMa feature matching.

Correspondences between the render and the photograph. RoMa is dense, so it
returns matches over the whole overlap rather than at sparse keypoints, which
matters here: a generated mesh's texture is approximate, and a detector looking
for repeatable corners finds far fewer of them on a render than on a photo.

The CUDA `local_corr` extension is OFF by default. romatch treats it as
optional -- it falls back to a pure-torch implementation, and switches to that
itself on non-Linux -- so leaving it off keeps meshfit installable with no build
toolchain. Turn it on for speed once you have one.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class RoMaMatcher:
    """Callable satisfying the `Matcher` protocol.

    The model is loaded on first use rather than in `__init__`, so constructing
    a matcher stays cheap and importing meshfit never pulls a checkpoint.
    """

    def __init__(self, device: str = "cuda", conf_threshold: float = 0.0,
                 use_custom_corr: bool = False, max_matches: int | None = 5000,
                 seed: int = 0):
        self.device = device
        self.conf_threshold = conf_threshold
        self.use_custom_corr = use_custom_corr
        self.max_matches = max_matches
        self.seed = seed
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from romatch import roma_outdoor
            self._model = roma_outdoor(device=self.device,
                                       use_custom_corr=self.use_custom_corr)
        return self._model

    def __call__(self, rendered_rgb: np.ndarray, observed_rgb: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        H0, W0 = rendered_rgb.shape[:2]
        H1, W1 = observed_rgb.shape[:2]
        with torch.inference_mode():
            warp, certainty = self.model.match(Image.fromarray(rendered_rgb),
                                               Image.fromarray(observed_rgb),
                                               device=self.device)
            matches, certainty = self.model.sample(warp, certainty)
            k0, k1 = self.model.to_pixel_coordinates(matches, H0, W0, H1, W1)

        k0 = k0.cpu().numpy()
        k1 = k1.cpu().numpy()
        conf = certainty.cpu().numpy()

        # No absolute confidence cut by default. RoMa's certainty is not
        # calibrated across image pairs -- matching an untextured render
        # against a photograph, we have measured a maximum of 0.1 over the
        # whole field, so a 0.5 threshold discards every good match and keeps
        # only stragglers. Geometry filters far better than a magic number:
        # `refine` drops matches that miss the rendered silhouette, and RANSAC
        # removes what survives that.
        if self.conf_threshold > 0:
            keep = conf >= self.conf_threshold
            k0, k1, conf = k0[keep], k1[keep], conf[keep]

        # Downsample rather than truncate: RoMa returns matches in no
        # meaningful order, but every one costs a ray in the lift, and a few
        # thousand already over-determines seven parameters.
        if self.max_matches and len(k0) > self.max_matches:
            idx = np.random.default_rng(self.seed).choice(
                len(k0), self.max_matches, replace=False)
            k0, k1, conf = k0[idx], k1[idx], conf[idx]
        return k0, k1, conf
