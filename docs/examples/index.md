# Examples

Three generators, three conventions, one code path.

Each example ends with the result you can turn around yourself: the observed
frame as a coloured point cloud, with the fitted mesh sitting in it at the size
and orientation meshfit recovered. Drag to orbit, scroll to zoom, and uncheck
**mesh** to see the points it was fitted to. Everything is in metres, so the
mesh is at its recovered size rather than scaled to fit the box.

## Pose-aware: SAM 3D

SAM 3D emits a mesh **and** a pose estimate. `meta.json` declares both the
convention and the estimate:

```json
{"backend": "sam3d", "pose_aware": true, "up_axis": "+Y",
 "T_world_canonical_init": [[...]], "scale_metric_init": 0.2035}
```

```bash
meshfit fit test_data/complex_tabletop_4_obj_001 --overlay out/controller.png
```

```
init      : source=generator, mask_iou=0.2012
refine    : free_rotation=True, tilt_deg=61.479, accepted=True, iou 0.2012 -> 0.9394
polish    : params=9, reproj 12.44 -> 9.02 px, accepted=True, iou -> 0.9478
```

<div class="scene-viewer" data-src="../scenes/complex_tabletop_4_obj_001.glb"></div>

The controller leans against a paper roll at about 55°, so the free hypothesis
wins and polish runs in its 9-parameter form. SAM 3D's own estimate had 56.7°,
and an independent pipeline on the same scene arrived at 62.06°.

## Pose-aware, different convention: recgen

recgen reports its pose in a **normalised camera frame**, so it composes
through two more transforms before meshfit can use it:

```
canonical --(pose)--> ncam --(cam2ncam⁻¹)--> camera --(cam2world)--> world
```

Since `cam2ncam` is a uniform scale plus translation, the chain collapses into
a single similarity. recgen's textured export is also written in glTF's +Y-up
convention while the pose was estimated against a +Z-up mesh. That is recorded
as `canonical_up` rather than fixed by rewriting vertices.

```
init      : source=generator, mask_iou=0.8627
refine    : accepted=True, iou 0.8627 -> 0.8819
polish    : params=7, reproj 4.88 -> 2.72 px, accepted=True, iou -> 0.9227
```

<div class="scene-viewer" data-src="../scenes/recgen_drawer_1.glb"></div>

## Pose-blind: TRELLIS.2

TRELLIS.2 returns a canonical mesh and nothing else. There is no `init.json`,
so meshfit searches.

```bash
meshfit fit test_data/trellis_windmill --overlay out/windmill.png --debug
```

```
init      : source=search, mask_iou=0.8191
refine    : free_rotation=False, accepted=False
polish    : params=7, reproj 26.70 -> 8.58 px, accepted=True, iou -> 0.8695
```

<div class="scene-viewer" data-src="../scenes/trellis_windmill.glb"></div>

The yaw ring did the hard part, going from no pose at all to 0.819, and polish
took it to 0.870. Refinement was rejected by the silhouette gate.

!!! tip "The mask must be the whole object"
    Segmenting this souvenir with the prompt `"windmill"` returned only the
    sail blades, 27k pixels. `"miniature windmill house"` returned the whole
    figurine, 94k. The first gave a final IoU of 0.378; the second, 0.870. The
    mask defines what meshfit is fitting to.
