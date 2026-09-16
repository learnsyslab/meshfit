"""Contact sheets over every case: final overlays, and the debug renders.

    python scripts/summary.py overlays   -> out/summary_overlays.jpg
    python scripts/summary.py renders    -> out/summary_renders.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "test_data"
OUT = ROOT / "out"


def _label(img: np.ndarray, text: str, scale: float = 1.0) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    pad = max(2, int(6 * scale))
    d.rectangle([0, 0, im.width, int(22 * scale)], fill=(0, 0, 0))
    d.text((pad, int(4 * scale)), text, fill=(255, 255, 255))
    return np.asarray(im)


def _row(images: list[np.ndarray], height: int) -> np.ndarray:
    out = []
    for img in images:
        im = Image.fromarray(img)
        out.append(np.asarray(im.resize((round(im.width * height / im.height), height))))
    return np.concatenate(out, axis=1)


def _pad_to(rows: list[np.ndarray]) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    return np.concatenate(
        [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows], axis=0)


def overlays(height: int = 240) -> Path:
    rows = []
    for case in sorted(CASES.iterdir()):
        path = OUT / f"{case.name}.png"
        if not path.exists():
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        rows.append(_label(_row([img], height), case.name))
    out = OUT / "summary_overlays.jpg"
    Image.fromarray(_pad_to(rows)).save(out, quality=90)
    return out


def renders(height: int = 200) -> Path:
    """First render and first match view from each case's debug directory."""
    rows = []
    for case in sorted(CASES.iterdir()):
        d = OUT / "debug" / case.name
        if not d.is_dir():
            continue
        picks = []
        for pattern in ("*_render.jpg", "*_matches.jpg", "*polish_overlay.jpg",
                        "*refine_overlay.jpg"):
            hits = sorted(d.glob(pattern))
            if hits:
                picks.append(np.asarray(Image.open(hits[0]).convert("RGB")))
        if picks:
            rows.append(_label(_row(picks, height), case.name))
    out = OUT / "summary_renders.jpg"
    Image.fromarray(_pad_to(rows)).save(out, quality=88)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "overlays"
    print("wrote", {"overlays": overlays, "renders": renders}[what]())
