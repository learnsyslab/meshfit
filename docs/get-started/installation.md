# Installation

```bash
pip install meshfit
```

That pulls numpy, scipy, trimesh, embreex, romatch (which brings torch) and
bpy. No compiler and no source builds: RoMa's CUDA `local_corr` extension is
optional and off by default, and Blender ships as a wheel.

!!! note "Python 3.11"
    `bpy` embeds its own CPython, so meshfit pins `bpy<5.1` and needs Python
    3.11.

## Development

[pixi](https://pixi.sh/) provides the interpreter; pip resolves the rest.

```bash
git clone https://github.com/learnsyslab/meshfit.git
cd meshfit
pixi run install-dev
pixi run test
pixi run lint
```

## Optional: faster matching

RoMa runs on pure torch by default. If you have CUDA and a compiler, its fused
correlation kernel is faster:

```bash
pip install "romatch[fused-local-corr]"
```

Then pass `use_custom_corr=True` to `RoMaMatcher`. Results are unchanged; only
throughput differs.

## Checking the install

```bash
python -c "import meshfit; print(meshfit.__version__)"
meshfit fit --help
```

A GPU is needed for RoMa in practice. Without one, `--cpu-only` still runs
initialisation, which is enough to verify the data contract.
