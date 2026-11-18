#!/usr/bin/env python3
"""Rough 3D edit mask for a composition case (source = anchor renders, target = composed views).

Writes into <item_dir>:
  object.glb               copy of the anchor GLB (the asset Easy3E edits)
  mask.glb                 scene with node `mask_all`: watertight mesh covering the interaction
                           region, in the anchor GLB's own (world) frame
  transforms_compose.json  Easy3E source normalization (scale, offset). TRELLIS normalizes what the
                           target shows (anchor + added parts), so the composite bbox (anchor + mask)
                           is centered and scaled into the [-0.5, 0.5]^3 grid instead of the anchor
                           alone (added parts such as a hanging basket stick out of the anchor bbox)
  view_cameras_compose.json  the views' cameras in that normalized frame (ViewCamera fields; the
                           principal point is offset because the source framing is kept)
  _mask_build/             2D edit regions, overlays and a tinted preview of the mask

Mask = occlusion-aware visual hull of the 2D edit regions B_v (source vs target,
compute_view_support.edit_support_2d) on a grid around the anchor:
  * every kept cell projects inside the (dilated) source-or-target silhouette in every view
    (all composed parts and all removed anchor parts lie there, occluded or not);
  * a view that sees the cell (not hidden behind the anchor) without an edit there rejects it;
  * at least one view sees the cell inside an edit.
It is then cleaned, dilated and meshed with marching cubes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import scipy.ndimage as ndi
import trimesh
from PIL import Image
from skimage import measure

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
import compute_view_support as cvs  # noqa: E402
from process_easy3e_rgba_target import write_easy3e_rgba_target  # noqa: E402


def anchor_in_compose_frame(glb: Path):
    """Anchor mesh in the compose frame: Blender Z-up, unit longest side, centered (raw -> v*s0 + o0)."""
    loaded = trimesh.load(str(glb), force="scene")
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    v = cvs.gltf_to_internal(np.asarray(mesh.vertices))
    bmin, bmax = v.min(0), v.max(0)
    s0 = 1.0 / float((bmax - bmin).max())
    o0 = -(bmin + bmax) / 2.0 * s0
    return trimesh.Trimesh(vertices=v * s0 + o0, faces=np.asarray(mesh.faces), process=False), s0, o0


def carve(anchor: trimesh.Trimesh, cams: dict, edits: dict, silhouettes: dict, half: float, res: int):
    lin = (np.arange(res) + 0.5) / res * 2 * half - half
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    scene = cvs.raycasting_scene(anchor)
    seen_edit = np.zeros(len(grid), dtype=bool)
    rejected = np.zeros(len(grid), dtype=bool)
    voxel = 2 * half / res
    for v, cam in cams.items():
        px, py, _ = cam.project(grid)
        in_frame = (px >= 0) & (px < cam.width) & (py >= 0) & (py < cam.height)
        d = cam.dir_to_camera(grid)
        rays = np.concatenate([grid + d * voxel, d], 1).astype(np.float32)
        hit = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())
        visible = in_frame & ~hit
        e = np.zeros(len(grid), dtype=bool)
        sil = np.zeros(len(grid), dtype=bool)
        e[in_frame] = edits[v][py[in_frame].astype(int), px[in_frame].astype(int)]
        sil[in_frame] = silhouettes[v][py[in_frame].astype(int), px[in_frame].astype(int)]
        seen_edit |= visible & e
        rejected |= ~sil | (visible & ~e)
    return (seen_edit & ~rejected).reshape(res, res, res), lin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item_dir", type=Path, required=True)
    ap.add_argument("--views", default=None, help="Comma-separated view labels to carve with (default: all).")
    ap.add_argument("--edit_threshold", type=float, default=0.3, help="B_v level counted as edited.")
    ap.add_argument("--dilate_2d", type=float, default=0.02, help="2D edit-region dilation (fraction of image size).")
    ap.add_argument("--dilate_silhouette", type=float, default=0.015, help="Silhouette dilation (fraction of image size).")
    ap.add_argument("--half_extent", type=float, default=1.0, help="Carving grid covers [-h, h]^3 in the compose frame.")
    ap.add_argument("--res", type=int, default=128, help="Carving grid resolution.")
    ap.add_argument("--dilate_3d", type=int, default=3, help="3D dilation of the hull (carving voxels).")
    ap.add_argument("--fit", type=float, default=0.95, help="Longest side of the centered composite bbox (anchor + mask).")
    args = ap.parse_args()

    item = args.item_dir.resolve()
    meta = json.loads((item / "compose_views.json").read_text())
    anchor_glb = Path(meta["objects"][meta["anchor"]]["glb"])
    W = int(meta["camera"]["resolution"])
    ortho = float(meta["camera"]["ortho_scale"])
    views = [v for v in meta["views"] if args.views is None or v["label"] in args.views.split(",")]
    build = item / "_mask_build"
    build.mkdir(exist_ok=True)

    anchor, s0, o0 = anchor_in_compose_frame(anchor_glb)
    cams, edits, silhouettes = {}, {}, {}
    for v in views:
        lab = v["label"]
        cams[lab] = cvs.ViewCamera(float(v["azimuth"]), float(v["elevation"]), None, W / ortho, W / 2, W / 2, W, W)
        tgt_rgba = build / f"target_{lab}_rgba.png"
        if not tgt_rgba.exists():
            write_easy3e_rgba_target(input_path=item / f"target_{lab}.png", output_rgba_path=tgt_rgba,
                                     output_mask_path=build / f"target_{lab}_alpha.png",
                                     output_mask_512_path=build / f"target_{lab}_alpha_512.png")
        # Targets may be resized / re-framed: warp them into the source frame first.
        tgt_aligned = build / f"target_{lab}_rgba_aligned.png"
        align = cvs.warp_target_to_source(item / f"source_{lab}.png", tgt_rgba, tgt_aligned)
        print(f"[mask] view {lab}: target alignment {align}")
        B, _ = cvs.edit_support_2d(item / f"source_{lab}.png", tgt_aligned, aligned=True)
        r = max(1, int(round(args.dilate_2d * W)))
        edits[lab] = cv2.dilate((B > args.edit_threshold).astype(np.uint8), cvs._disk(r)) > 0
        # Source-or-target silhouette (all composed parts and removed anchor parts project inside).
        src_alpha = np.asarray(Image.open(item / f"source_{lab}.png").convert("RGBA"))[..., 3] > 127
        tgt_in_src = np.asarray(Image.open(tgt_aligned).convert("RGBA"))[..., 3] > 127
        rs = max(1, int(round(args.dilate_silhouette * W)))
        silhouettes[lab] = cv2.dilate((src_alpha | tgt_in_src).astype(np.uint8), cvs._disk(rs)) > 0
        src = cvs._rgba_on_white(np.asarray(Image.open(item / f"source_{lab}.png").convert("RGBA")))
        Image.fromarray(cvs.colorize(edits[lab].astype(np.float32), src, cv2.COLORMAP_JET, 0.45, min_value=0.5)
                        ).save(build / f"edit_region_2d_{lab}.png")

    hull, lin = carve(anchor, cams, edits, silhouettes, args.half_extent, args.res)
    lab3, n = ndi.label(hull)
    if n == 0:
        raise RuntimeError("Visual hull of the edit regions is empty")
    sizes = ndi.sum(hull, lab3, range(1, n + 1))
    hull = np.isin(lab3, 1 + np.flatnonzero(sizes >= 0.05 * sizes.max()))
    mask = ndi.binary_fill_holes(ndi.binary_dilation(hull, iterations=args.dilate_3d))

    # Normalize the composite (anchor + mask bbox) the way TRELLIS normalizes what the target
    # shows: centered, longest side `fit`. internal = compose * s1 + o1; the cameras keep the
    # source images' framing, so they get a principal-point offset.
    step = lin[1] - lin[0]
    pts = np.argwhere(mask) * step + lin[0]
    lo = np.minimum(anchor.vertices.min(0), pts.min(0) - step / 2)
    hi = np.maximum(anchor.vertices.max(0), pts.max(0) + step / 2)
    s1 = args.fit / float((hi - lo).max())
    o1 = -(lo + hi) / 2.0 * s1
    transform = {"scale": s0 * s1, "offset": (o0 * s1 + o1).tolist(), "aabb": [[-0.5] * 3, [0.5] * 3], "frames": [],
                 "note": "Easy3E source normalization (v_internal = v_blender_raw * scale + offset): the composite "
                         f"bbox (anchor + mask) is centered with longest side {args.fit}; compose frame scaled by "
                         f"{s1:.4f} and shifted by {np.round(o1, 4).tolist()}."}
    (item / "transforms_compose.json").write_text(json.dumps(transform, indent=2))
    cams_int = {}
    for lab, c in cams.items():
        _, right, up, _ = cvs._basis(c.azimuth, c.elevation)
        scale = c.scale / s1
        cams_int[lab] = cvs.ViewCamera(c.azimuth, c.elevation, None, scale, c.tx - scale * float(o1 @ right),
                                       c.ty + scale * float(o1 @ up), c.width, c.height)
    (item / "view_cameras_compose.json").write_text(json.dumps({k: asdict(c) for k, c in cams_int.items()}, indent=2))

    # Watertight mesh of the mask, written in the anchor GLB's raw (glTF Y-up) frame.
    verts, faces, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5)
    v_compose = (verts - 1) * step + lin[0]
    v_blender = (v_compose - o0) / s0
    v_gltf = np.stack([v_blender[:, 0], v_blender[:, 2], -v_blender[:, 1]], 1)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.Trimesh(v_gltf, faces[:, ::-1], process=True), node_name="mask_all", geom_name="mask_all")
    scene.export(str(item / "mask.glb"))
    if not (item / "object.glb").exists():
        shutil.copy2(anchor_glb, item / "object.glb")

    # Checks: Easy3E's voxelization of the mask (internal frame) and its projection on every view.
    m_int = trimesh.Trimesh(v_compose * s1 + o1, faces[:, ::-1], process=True)
    omega = cvs.solid_mask_from_mesh(m_int)
    np.save(build / "mask_all_64.npy", omega)
    cells = np.argwhere(omega)
    tiles = []
    for lab, cam in cams_int.items():
        tgt = cvs._rgba_on_white(np.asarray(Image.open(build / f"target_{lab}_rgba_aligned.png").convert("RGBA")))
        proj = cvs.project_field_on_view(cam, np.ones(len(cells), np.float32), cells, 384)
        tiles.append(cvs._label(cvs.colorize(proj, cv2.resize(tgt, (384, 384)), cv2.COLORMAP_AUTUMN, 0.35), f"mask on target {lab}"))
    Image.fromarray(np.concatenate(tiles, 1)).save(build / "mask_on_targets.png")
    summary = {"anchor_glb": str(anchor_glb), "views": list(cams), "compose_scale_s1": s1, "compose_offset_o1": o1.tolist(),
               "hull_cells": int(hull.sum()), "mask_cells_carving_grid": int(mask.sum()),
               "mask_voxels_64": int(omega.sum()), "mask_bbox_internal": [cells.min(0).tolist(), cells.max(0).tolist()]}
    (build / "mask_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
