#!/usr/bin/env python3
"""Download the HumanoidBench prop textures and license into ``models/props/``.

    python scripts/fetch_props.py

The prop MJCFs in ``models/props/*.xml`` are transcribed from HumanoidBench's
``tasks/room.xml``, ``push.xml`` and ``package.xml`` and reference these
textures. The files are committed, so this only needs to run to refresh them.
The 35 MB ``plywood-4k.png`` is deliberately not fetched.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

COMMIT = "cb1189039151c8aadaaa987b442da54383c87fab"
BASE = f"https://raw.githubusercontent.com/carlosferrazza/humanoid-bench/{COMMIT}"
TEXTURES = (
    "light-wood.png",
    "dark-wood.png",
    "package_plain.png",
    "metal.png",
    "steel-scratched.png",
)
PROPS_DIR = Path(__file__).resolve().parents[1] / "models" / "props"


def fetch(url: str, dest: Path) -> int:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    dest.write_bytes(payload)
    return len(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROPS_DIR)
    args = parser.parse_args(argv)
    textures = args.out / "textures"
    textures.mkdir(parents=True, exist_ok=True)
    for name in TEXTURES:
        size = fetch(f"{BASE}/humanoid_bench/assets/tasks/assets/{name}", textures / name)
        print(f"{name}: {size / 1e6:.2f} MB")
    fetch(f"{BASE}/LICENSE", args.out / "LICENSE.humanoidbench")
    print(f"HumanoidBench {COMMIT[:12]} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
