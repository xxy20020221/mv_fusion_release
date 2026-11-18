#!/usr/bin/env python3
"""Tint a 64^3 voxel edit region onto a textured GLB for inspection.

Every texel of the base-color texture is mapped to its 3D surface point by
rasterizing the mesh in UV space (nvdiffrast); texels whose point lies in a
region voxel are blended with a highlight color. The region is in Easy3E's
internal frame (Blender Z-up, normalized to [-0.5, 0.5]^3), e.g. the
edit_region_round2.npy written by the round-2 driver; the GLB is mapped there
with glTF Y-up -> Z-up and the optional (scale, offset) of a transforms.json.

    python export_edit_region_on_asset.py --glb <round-1 out>/target_1_rgba.glb \
        --region_npy <round-2 work>/3Dmasks/edit_region_round2.npy --out omega2_on_asset.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from PIL import Image


def texel_positions(mesh: trimesh.Trimesh, verts_internal: np.ndarray, size: tuple[int, int]):
    """(H, W, 3) surface point per texel in image-row order and the (H, W) coverage mask."""
    H, W = size
    uv = np.asarray(mesh.visual.uv, dtype=np.float32)
    clip = np.concatenate([uv * 2.0 - 1.0, np.zeros((len(uv), 1)), np.ones((len(uv), 1))], axis=1)
    ctx = dr.RasterizeCudaContext()
    tri = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int32, device="cuda")
    rast, _ = dr.rasterize(ctx, torch.as_tensor(clip, dtype=torch.float32, device="cuda")[None], tri, (H, W))
    pos, _ = dr.interpolate(torch.as_tensor(verts_internal, dtype=torch.float32, device="cuda")[None], rast, tri)
    # Raster row 0 is v = 0 (bottom); texture image row 0 is v = 1 (top).
    pos = pos[0].flip(0).cpu().numpy()
    covered = (rast[0, ..., 3] > 0).flip(0).cpu().numpy()
    return pos, covered


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", type=Path, required=True)
    ap.add_argument("--region_npy", type=Path, required=True, help="(G,G,G) boolean grid in the internal frame.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--transform_json", type=Path, default=None,
                    help="transforms.json whose scale/offset map the GLB into the internal frame "
                         "(omit for assets that are already normalized, e.g. Easy3E outputs).")
    ap.add_argument("--color", default="255,40,200", help="Highlight color R,G,B.")
    ap.add_argument("--alpha", type=float, default=0.6, help="Highlight opacity.")
    ap.add_argument("--gutter", type=int, default=3, help="Texels to extend the tint into UV gutters (avoids seams).")
    args = ap.parse_args()

    region = np.load(args.region_npy).astype(bool)
    G = region.shape[0]
    scale, offset = 1.0, np.zeros(3)
    if args.transform_json is not None:
        meta = json.loads(args.transform_json.read_text())
        scale, offset = float(meta["scale"]), np.asarray(meta["offset"], dtype=np.float64)
    color = np.array([float(c) for c in args.color.split(",")], dtype=np.float32)

    scene = trimesh.load(str(args.glb), force="scene")
    stats = []
    for node in scene.graph.nodes_geometry:
        T, name = scene.graph[node]
        mesh = scene.geometry[name]
        mat = getattr(mesh.visual, "material", None)
        tex = getattr(mat, "baseColorTexture", None)
        if tex is None or getattr(mesh.visual, "uv", None) is None:
            continue
        v = trimesh.transform_points(np.asarray(mesh.vertices), T)
        v_int = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1) * scale + offset[None, :]
        img = np.asarray(tex.convert("RGBA")).astype(np.float32)
        pos, covered = texel_positions(mesh, v_int, img.shape[:2])
        idx = np.clip(np.floor((pos + 0.5) * G).astype(int), 0, G - 1)
        inside = covered & region[idx[..., 0], idx[..., 1], idx[..., 2]]
        if args.gutter > 0:
            grown = cv2.dilate(inside.astype(np.uint8), np.ones((2 * args.gutter + 1,) * 2, np.uint8)) > 0
            inside |= grown & ~covered
        img[inside, :3] = img[inside, :3] * (1.0 - args.alpha) + color * args.alpha
        mat.baseColorTexture = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGBA")
        stats.append({"geometry": name, "texels_covered": int(covered.sum()), "texels_tinted": int(inside.sum())})

    if not stats:
        raise RuntimeError(f"No base-color textured geometry in {args.glb}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(args.out))
    print(json.dumps({"out": str(args.out), "region_voxels": int(region.sum()), "geometries": stats}, indent=2))


if __name__ == "__main__":
    main()
