# Output schema

```json
{
  "pose": {
    "T_world_canonical": [[...4x4...]],
    "scale_metric": 0.42,
    "scale_aniso": [0.41, 0.43, 0.42],
    "projected": false,
    "tilt_removed_deg": 0.0
  },
  "confidence": {
    "symmetry_margin": 0.31, "k_fold": 1, "ambiguous": false,
    "matches": 4390, "inliers": 4193, "reprojection_rms_px": 1.9
  },
  "provenance": {
    "canonical_up": "+Y", "init_source": "generator",
    "tilt_mode": "auto", "free_rotation": false,
    "refine_iterations": 3, "stages": [...], "warnings": [...]
  }
}
```

## pose

$$p_{world} = T_{world\,canonical}\cdot(\text{scale}\odot p_{canonical})$$

`T_world_canonical` is the **rigid** part only, rotation and translation. Scale
is kept separate so `scale_aniso` stays recoverable.

`scale_aniso` is per-**canonical**-axis, and appears only when the scale really
is anisotropic. `scale_metric` is its geometric mean, so a consumer that does
not handle anisotropy still gets a sane fallback.

!!! note "No bare quaternions"
    Poses are 4×4 matrices. Quaternion conventions (wxyz vs xyzw) get confused
    silently and the resulting error looks like a bad fit rather than a parsing
    bug.

`projected` and `tilt_removed_deg` record whether tilt was removed, including
when that happened as a side effect of the 7-parameter polish rather than
because it was asked for.

## confidence

`ambiguous` is the field to check, see [Symmetry](symmetry.md).

`reprojection_rms_px` is the quantity polish minimises. It is a 2D measurement
and says nothing about depth, so read it beside a 3D residual rather than
instead of one.

## provenance

`stages` is the per-stage record: what ran, what it produced, and whether it
was `accepted`. `warnings` names conditions meshfit could not resolve, such as a
single view or a lean it cannot refine, each with the flag that addresses it.

!!! warning "mask IoU is blind to depth"
    An object at twice the distance and twice the size scores a perfect 1.0.
    Always read it next to a 3D residual.
