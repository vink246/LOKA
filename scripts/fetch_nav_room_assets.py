#!/usr/bin/env python3
"""Download MP3D-matched CC0 room assets from Poly Haven into ``models/room/``.

    python scripts/fetch_nav_room_assets.py

Requires: Pillow, trimesh, network. Writes textures (1k/512 PNG) and chair/shelf
OBJ meshes sized to LOKA's navigation prop footprints. See
``models/room/LICENSE.polyhaven``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROOM = ROOT / "models" / "room"
TEX = ROOM / "textures"
MESH = ROOM / "meshes"
UA = {"User-Agent": "LOKA-research/1.0 (academic)"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def api_files(asset_id: str) -> dict:
    return json.loads(get(f"https://api.polyhaven.com/files/{asset_id}"))


def save_diffuse(asset_id: str, out: Path, res: str = "1k") -> None:
    meta = api_files(asset_id)
    url = meta["Diffuse"][res]["png"]["url"]
    out.write_bytes(get(url))
    print(f"{out.name}: {out.stat().st_size / 1e6:.2f} MB")


def resize_png(src: Path, dst: Path, size: int) -> None:
    from PIL import Image

    Image.open(src).convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(
        dst, optimize=True
    )


def cardboard(path: Path, size: int = 512) -> None:
    from PIL import Image

    rng = np.random.default_rng(0)
    base = np.array([0.76, 0.62, 0.42])
    noise = rng.normal(0, 0.04, (size, size, 1))
    yy = np.linspace(0, 40 * np.pi, size)[:, None]
    corr = 0.03 * np.sin(yy)
    rgb = np.clip(base + noise + corr[..., None], 0, 1)
    rgb[size // 2 - 16 : size // 2 + 16] = np.array([0.85, 0.78, 0.55])
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def upright(mesh):
    import trimesh

    extents = mesh.extents
    up = int(np.argmax(extents))
    if up == 2:
        return mesh
    axes = [0, 1, 2]
    axes.remove(up)
    new_r = np.zeros((3, 3))
    new_r[2, up] = 1.0
    new_r[0, axes[0]] = 1.0
    new_r[1, axes[1]] = 1.0
    if np.linalg.det(new_r) < 0:
        new_r[1] *= -1
    mesh.apply_transform(
        np.vstack([np.hstack([new_r, np.zeros((3, 1))]), [0, 0, 0, 1]])
    )
    return mesh


def fetch_mesh(asset_id: str, stem: str, target, *, nonuniform: bool) -> None:
    import trimesh
    from PIL import Image

    meta = api_files(asset_id)
    entry = meta["gltf"]["1k"]["gltf"]
    work = MESH / f"_{stem}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    gltf_path = work / f"{asset_id}_1k.gltf"
    gltf_path.write_bytes(get(entry["url"]))
    for rel, info in entry["include"].items():
        dest = work / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(get(info["url"]))
    scene = trimesh.load(gltf_path, force="scene")
    geoms = list(scene.geometry.values())
    mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0].copy()
    mesh = upright(mesh)
    # Fit each axis independently so tall skinny scans still fill the prop footprint.
    mesh.vertices = mesh.vertices * np.array(
        [float(target[i]) / max(float(mesh.extents[i]), 1e-9) for i in range(3)]
    )
    if stem == "shelf":
        # Poly Haven bookshelf opens along -Y; LOKA shelf collisions open along -X.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        )
        mesh.vertices = mesh.vertices * np.array(
            [float(target[i]) / max(float(mesh.extents[i]), 1e-9) for i in range(3)]
        )
    center = mesh.bounds.mean(axis=0)
    mesh.vertices -= [center[0], center[1], mesh.bounds[0, 2]]
    mesh.export(MESH / f"{stem}.obj")
    for path in (work / "textures").glob("*diff*"):
        Image.open(path).convert("RGB").save(TEX / f"{stem}_diff_1k.png", optimize=True)
        break
    shutil.rmtree(work)
    print(f"{stem}.obj extents={mesh.extents}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    TEX.mkdir(parents=True, exist_ok=True)
    MESH.mkdir(parents=True, exist_ok=True)

    save_diffuse("wooden_floor_01", TEX / "wooden_floor_01_diff_1k.png")
    save_diffuse("plastered_wall", TEX / "plastered_wall_diff_1k.png")
    save_diffuse("wood_table_worn", TEX / "wood_table_worn_diff_1k.png")
    save_diffuse("dark_wood", TEX / "dark_wood_diff_1k.png")
    resize_png(TEX / "wood_table_worn_diff_1k.png", TEX / "wood_table_worn_diff_512.png", 512)
    resize_png(TEX / "dark_wood_diff_1k.png", TEX / "dark_wood_diff_512.png", 512)
    (TEX / "wood_table_worn_diff_1k.png").unlink(missing_ok=True)
    (TEX / "dark_wood_diff_1k.png").unlink(missing_ok=True)
    cardboard(TEX / "cardboard_diff_512.png")

    fetch_mesh("WoodenChair_01", "chair", (0.42, 0.42, 0.85), nonuniform=False)
    fetch_mesh("wooden_bookshelf_worn", "shelf", (0.65, 0.90, 1.85), nonuniform=True)

    license_src = ROOM / "LICENSE.polyhaven"
    if not license_src.is_file():
        print("warning: LICENSE.polyhaven missing", file=__import__("sys").stderr)
    print(f"assets -> {ROOM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
