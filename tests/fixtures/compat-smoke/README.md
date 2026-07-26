# compat-smoke fixture dataset

Four 32x32 PNG swatches plus ai-toolkit-style `.txt` captions, ~230 bytes per image. They are
the training set for the dispatch-gated GPU compat smoke (`deploy/compat_smoke.py`, issue #29):
enough signal for a denoising loss to move LoRA weights, small enough that the repo stays a
source repo and not a data repo.

They are NOT a quality fixture. Nothing here says anything about Wan output; the smoke only
answers "does this dependency set install, import, and complete a real train step on a GPU".

Regenerate (deterministic, stdlib only, no Pillow):

```python
import struct, zlib
from pathlib import Path

OUT, SIZE = Path("tests/fixtures/compat-smoke"), 32
SWATCHES = {"swatch-01": (200, 60, 60), "swatch-02": (60, 160, 200),
            "swatch-03": (240, 210, 90), "swatch-04": (90, 200, 120)}

def chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

for name, (r, g, b) in SWATCHES.items():
    raw = bytearray()
    for y in range(SIZE):
        raw.append(0)
        for x in range(SIZE):
            k = (x + y) // 8
            raw += bytes((max(0, r - 12 * k), max(0, g - 12 * k), max(0, b - 12 * k)))
    (OUT / f"{name}.png").write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b""))
```
