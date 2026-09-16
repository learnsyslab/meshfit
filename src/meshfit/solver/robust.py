"""Robust estimation around a closed-form fit."""

from __future__ import annotations

import numpy as np

from ..pose import Pose


def ransac(
    x: np.ndarray,
    y: np.ndarray,
    fit,
    min_samples: int,
    inlier_thresh: float = 0.05,
    iters: int = 1000,
    seed: int = 0,
) -> tuple[Pose, np.ndarray]:
    """RANSAC around any paired-point fit. Returns (pose on inliers, mask).

    `min_samples` is the fit's DoF in points: 2 for `fit_upright` (yaw,
    isotropic scale and translation are determined by two pairs), 3 for
    `fit_similarity`. Smaller minimal sets need far fewer iterations to hit a
    clean sample, which is a practical reason to prefer the constrained fit
    even before considering plausibility.

    `inlier_thresh` is a distance IN METRES -- the world is metric by
    precondition, so this is an absolute tolerance, not a relative one.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < min_samples:
        raise ValueError(f"need at least {min_samples} pairs, got {n}")

    best_mask, best_count = np.ones(n, bool), 0
    for _ in range(iters):
        idx = rng.choice(n, min_samples, replace=False)
        try:
            candidate = fit(x[idx], y[idx])
        except Exception:
            continue                      # degenerate sample; draw another
        mask = np.linalg.norm(candidate.apply(x) - y, axis=1) < inlier_thresh
        if mask.sum() > best_count:
            best_count, best_mask = int(mask.sum()), mask

    if best_count < min_samples:
        best_mask = np.ones(n, bool)      # nothing beat chance; fit everything
    return fit(x[best_mask], y[best_mask]), best_mask
