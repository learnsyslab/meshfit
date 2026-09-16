# Visualisation

Two views of the same answer. They catch different mistakes: the overlay catches
a mesh in the wrong **place**, viser catches one at the wrong **depth**, which an
overlay cannot show because an object twice as far away at twice the size covers
the same pixels.

## Overlay

```bash
meshfit fit <case> --overlay out/fit.png
```

The mesh composited onto the photograph. **Green** where the mesh is, **red**
where the observed mask is, **yellow** where they agree. A green fringe means the
mesh spills past the object, red means it falls short, large separate patches
mean the pose is wrong.

## viser

```bash
python scripts/show.py viser <case>
```

The scene in 3D: the frame as a colour point cloud, the object's points larger,
the fitted mesh in green and the initialisation in grey, plus camera frusta. Each
layer has a checkbox, so "did refinement help" is something you look at.

## Every intermediate

```bash
meshfit fit <case> --debug
```

Writes every render, match visualisation and per-stage overlay to
`out/debug/<case>/`, numbered in order. Most failures here are only diagnosable
by eye: a render with no texture, or match lines that spray at random rather than
running parallel, both look like solver problems in the numbers.
