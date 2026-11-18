#!/usr/bin/env python3
"""Geometry-derived per-view 3D support fields for Easy3E mv_fusion.

Implements the "geometry-derived view support" of Route-Diagnose-Refine
(Sec. 3.2, Eqs. 4-7): given a rough 3D edit mask Omega and N source/target
view pairs, estimate for every view i a soft 3D field chi_i(p) on the 64^3
TRELLIS grid saying where that view's edit evidence is spatially applicable.

    r_i(p)  = V_i(p) * B_i(pi_i(p)) * (max(n(p).d_i(p), 0) + eps)^gamma * psi(d_Omega(p))   (4)
    u_i     = argmin_u sum_p (u - r_i)^2 + mu sum_{(p,q)} w_pq (u_p - u_q)^2                  (5)
    chi_i   = u_i / (sum_j u_j + eta)                                                         (6)

Ownership (Eq. 7) is computed downstream by the mv_fusion sampler from chi in
acquisition order (base view first), exactly like the existing
"later trust view wins" rule; it is also exported here for inspection.

Terms:
  V_i  soft visibility: ray from the cell toward camera i through the source
       mesh; depth below the first occluder decays as exp(-depth / tau).
  B_i  soft 2D edit support in source-image pixels, from the aligned
       source-target difference (detect_image_edit_regions.py), including
       source silhouette that disappears in the target.
  n.d  facing term; n is the nearest-surface normal close to the surface and
       the distance-field gradient in free space (where new geometry appears).
  psi  rises from psi_min at the Omega boundary to 1 at `psi_width` voxels inside.

Cameras of user-supplied source renders are unknown, so `fit_camera` recovers
(azimuth, elevation, perspective distance, 2D similarity) by silhouette IoU on
the normalized source mesh, using texture colors to break left/right mirror
ambiguities. Known cameras can be given with --camera_json instead.

All geometry lives in the pipeline-internal frame (Blender Z-up, normalized to
[-0.5, 0.5]^3), i.e. the frame of <render_dir>/mesh.ply and the 64^3 raw_mask.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import open3d as o3d
import scipy.ndimage as ndi
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_image_edit_regions as edit_regions  # noqa: E402

GRID = 64
VOXEL = 1.0 / GRID


# --------------------------------------------------------------------------- #
# Frames and voxelization (must match Easy3E/pipeline_edit/utils.py)
# --------------------------------------------------------------------------- #
def gltf_to_internal(v: np.ndarray) -> np.ndarray:
    """glTF Y-up -> Blender Z-up: (x, y, z) -> (x, -z, y)."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=-1)


def load_render_transform(render_dir: Path) -> tuple[float, np.ndarray]:
    meta = json.loads((render_dir / "transforms.json").read_text())
    return float(meta["scale"]), np.asarray(meta["offset"], dtype=np.float64)


def load_mesh_internal(glb_path: Path, scale: float, offset: np.ndarray) -> trimesh.Trimesh:
    """Load a GLB in its world frame and apply Easy3E's mask normalization."""
    loaded = trimesh.load(str(glb_path), force="scene")
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    verts = gltf_to_internal(np.asarray(mesh.vertices)) * scale + offset[None, :]
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces), process=False)


def grid_centers() -> np.ndarray:
    lin = (np.arange(GRID, dtype=np.float32) + 0.5) / GRID - 0.5
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


def solid_mask_from_mesh(mesh: trimesh.Trimesh) -> np.ndarray:
    """Same inside test as pipeline_edit.utils._solid_mask_via_sdf."""
    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    tmesh.triangle.indices = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    pts = o3d.core.Tensor(grid_centers().reshape(-1, 3))
    sdf = scene.compute_signed_distance(pts).numpy().reshape(GRID, GRID, GRID)
    return sdf <= 0


def surface_voxels_from_mesh(mesh_ply: Path) -> np.ndarray:
    """Same surface voxelization as pipeline_edit.utils.voxelize (source coords)."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    v = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=VOXEL, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    idx = np.array([vx.grid_index for vx in vg.get_voxels()], dtype=np.int64)
    occ = np.zeros((GRID, GRID, GRID), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def raycasting_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    tmesh.triangle.indices = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def _basis(azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blender track-to camera (see blender_scripts/render_glb_mv.py).

    Returns (to_cam, right, up, forward); to_cam is the unit vector from the
    origin toward the camera, image x follows `right`, image y follows -`up`.
    """
    az, el = math.radians(azimuth), math.radians(elevation)
    to_cam = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    forward = -to_cam
    right = np.cross(forward, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-8:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return to_cam, right, up, forward


@dataclass
class ViewCamera:
    """Ortho (distance=None) or perspective camera looking at the origin.

    pixel_x = scale * u + tx, pixel_y = -scale * v + ty where (u, v) are the
    ortho plane coordinates or the perspective ratios (x/z, y/z).
    """

    azimuth: float
    elevation: float
    distance: float | None
    scale: float
    tx: float
    ty: float
    width: int
    height: int
    iou: float = 0.0
    color_score: float = 0.0

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        to_cam, right, up, forward = _basis(self.azimuth, self.elevation)
        pts = np.asarray(pts, dtype=np.float64)
        if self.distance is None:
            u, v = pts @ right, pts @ up
            depth = pts @ forward
        else:
            rel = pts - to_cam[None, :] * self.distance
            depth = rel @ forward
            safe = np.maximum(depth, 1e-6)
            u, v = (rel @ right) / safe, (rel @ up) / safe
        return self.scale * u + self.tx, -self.scale * v + self.ty, depth

    def dir_to_camera(self, pts: np.ndarray) -> np.ndarray:
        to_cam, *_ = _basis(self.azimuth, self.elevation)
        if self.distance is None:
            return np.broadcast_to(to_cam, np.shape(pts)).astype(np.float64)
        d = to_cam[None, :] * self.distance - np.asarray(pts, dtype=np.float64)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def dist_to_camera(self, pts: np.ndarray) -> np.ndarray:
        if self.distance is None:
            return np.full(len(pts), np.inf)
        to_cam, *_ = _basis(self.azimuth, self.elevation)
        return np.linalg.norm(to_cam[None, :] * self.distance - pts, axis=-1)


def _sample_mesh_colors(m: trimesh.Trimesh, pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """RGB in [0, 255] at surface samples: PBR/simple base-color texture via UV, else vertex/face colors."""
    vis = m.visual
    if isinstance(vis, trimesh.visual.texture.TextureVisuals) and vis.uv is not None:
        mat = vis.material
        tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        factor = getattr(mat, "baseColorFactor", None)
        if factor is None:
            factor = getattr(mat, "diffuse", None)
        factor = np.ones(4) if factor is None else np.asarray(factor, dtype=np.float64).reshape(-1)[:4]
        factor = factor / 255.0 if factor.max() > 1.0 else factor
        if tex is not None:
            bary = trimesh.triangles.points_to_barycentric(m.triangles[faces], pts)
            uv = np.einsum("nk,nkj->nj", bary, vis.uv[m.faces[faces]])
            img = np.asarray(tex.convert("RGB"), dtype=np.float32)
            h, w = img.shape[:2]
            x = (uv[:, 0] % 1.0) * (w - 1)
            y = (1.0 - (uv[:, 1] % 1.0)) * (h - 1)
            col = img[np.round(y).astype(int), np.round(x).astype(int)]
            return col * factor[None, :3]
        return np.tile(factor[None, :3] * 255.0, (len(pts), 1))
    try:
        return np.asarray(vis.face_colors, dtype=np.float32)[faces, :3]
    except Exception:
        return np.full((len(pts), 3), 160.0, dtype=np.float32)


def sample_colored_points(glb_path: Path, count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample surface points (internal frame) with base colors."""
    loaded = trimesh.load(str(glb_path), force="scene")
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]
    meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    areas = np.array([m.area for m in meshes], dtype=np.float64)
    counts = np.maximum(1, np.round(count * areas / areas.sum())).astype(int)
    pts_all, col_all = [], []
    for m, n in zip(meshes, counts):
        pts, faces = trimesh.sample.sample_surface(m, int(n), seed=seed)
        pts_all.append(pts)
        col_all.append(_sample_mesh_colors(m, pts, faces))
    return gltf_to_internal(np.concatenate(pts_all)), np.concatenate(col_all).astype(np.float32) / 255.0


def _target_mask(rgba: np.ndarray, res: int) -> np.ndarray:
    m = (rgba[..., 3] > 127).astype(np.uint8) * 255
    return cv2.resize(m, (res, res), interpolation=cv2.INTER_AREA) > 127


class _Splatter:
    """GPU point splatting of silhouettes / z-buffered colors for camera fitting."""

    def __init__(self, pts: np.ndarray, colors: np.ndarray, device: str):
        self.device = device
        self.pts = torch.as_tensor(pts, dtype=torch.float32, device=device)
        self.col = torch.as_tensor(colors, dtype=torch.float32, device=device)

    def _uvd(self, azimuth: float, elevation: float, distance: float | None):
        to_cam, right, up, forward = (torch.as_tensor(a, dtype=torch.float32, device=self.device)
                                      for a in _basis(azimuth, elevation))
        p = self.pts if distance is None else self.pts - to_cam * distance
        depth = p @ forward
        u, v = p @ right, p @ up
        if distance is not None:
            u, v = u / depth.clamp(min=1e-4), v / depth.clamp(min=1e-4)
        return u, v, depth

    @staticmethod
    def init_similarity(u, v, tgt_bbox, res):
        """Scale/translation mapping the (u, -v) point bbox onto the target bbox."""
        uq = torch.quantile(u[:: max(1, len(u) // 200000)], torch.tensor([0.001, 0.999], device=u.device))
        vq = torch.quantile(v[:: max(1, len(v) // 200000)], torch.tensor([0.001, 0.999], device=v.device))
        x0, y0, x1, y1 = tgt_bbox
        su = (x1 - x0) / max(float(uq[1] - uq[0]), 1e-6)
        sv = (y1 - y0) / max(float(vq[1] - vq[0]), 1e-6)
        s = 0.5 * (su + sv)
        tx = 0.5 * (x0 + x1) - s * 0.5 * float(uq[0] + uq[1])
        ty = 0.5 * (y0 + y1) + s * 0.5 * float(vq[0] + vq[1])
        return s, tx, ty

    def silhouette(self, u, v, s, tx, ty, res):
        px = torch.floor(s * u + tx).long()
        py = torch.floor(-s * v + ty).long()
        ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
        img = torch.zeros(res * res, device=self.device)
        img[py[ok] * res + px[ok]] = 1.0
        img = img.view(1, 1, res, res)
        img = F.max_pool2d(img, 3, 1, 1)
        img = -F.max_pool2d(-img, 3, 1, 1)  # closing fills splat gaps
        return img[0, 0] > 0.5

    def color_image(self, u, v, depth, s, tx, ty, res):
        px = torch.floor(s * u + tx).long()
        py = torch.floor(-s * v + ty).long()
        ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
        key = (py * res + px)[ok]
        d = depth[ok]
        zmin = torch.full((res * res,), float("inf"), device=self.device)
        zmin.scatter_reduce_(0, key, d, reduce="amin")
        front = d <= zmin[key] + 2.0 * VOXEL
        acc = torch.zeros(res * res, 3, device=self.device)
        cnt = torch.zeros(res * res, device=self.device)
        acc.index_add_(0, key[front], self.col[ok][front])
        cnt.index_add_(0, key[front], torch.ones_like(d[front]))
        img = acc / cnt.clamp(min=1)[:, None]
        return img.view(res, res, 3), (cnt > 0).view(res, res)


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / max(union, 1)


def _chroma_similarity(render_rgb: np.ndarray, render_mask: np.ndarray, src_rgb: np.ndarray, src_mask: np.ndarray) -> float:
    """Shading-robust color agreement: compare blurred LAB chroma (a, b) on common pixels."""
    common = render_mask & src_mask
    if common.sum() < 32:
        return 0.0
    def chroma(img):
        lab = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        return cv2.GaussianBlur(lab[..., 1:], (0, 0), 1.5)
    diff = np.linalg.norm(chroma(render_rgb) - chroma(src_rgb), axis=-1)[common]
    return float(np.exp(-np.median(diff) / 12.0))


def fit_camera(
    splat: _Splatter,
    image_rgba: np.ndarray,
    distances: Sequence[float | None] = (None, 6.0, 3.5, 2.5, 1.8),
    az_step: float = 6.0,
    el_range: tuple[float, float] = (-30.0, 60.0),
    el_step: float = 6.0,
    coarse_res: int = 128,
    fine_res: int = 256,
    top_k: int = 12,
    color_weight: float = 0.25,
    snap_deg: float = 15.0,
    snap_ortho_scales: Sequence[float] = (1.0, 1.2),
    snap_tolerance: float = 0.01,
    log=print,
) -> ViewCamera:
    """Fit an object-centric camera to a source rendering by silhouette IoU.

    Candidates on an (azimuth, elevation, distance) grid are scored by the IoU
    between the splatted mesh silhouette (bbox-aligned 2D similarity) and the
    image alpha. The best few are refined by coordinate descent; the final
    choice adds a texture-chroma term so mirror-symmetric silhouettes resolve
    to the side whose colors actually match.
    """
    H, W = image_rgba.shape[:2]
    assert H == W, "fit_camera expects square renders"

    def tgt_at(res):
        m = _target_mask(image_rgba, res)
        ys, xs = np.nonzero(m)
        return torch.as_tensor(m, device=splat.device), (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    tgt_c, bbox_c = tgt_at(coarse_res)
    cands = []
    for dist in distances:
        for el in np.arange(el_range[0], el_range[1] + 1e-6, el_step):
            for az in np.arange(0.0, 360.0, az_step):
                u, v, _ = splat._uvd(float(az), float(el), dist)
                s, tx, ty = splat.init_similarity(u, v, bbox_c, coarse_res)
                iou = _iou(splat.silhouette(u, v, s, tx, ty, coarse_res), tgt_c)
                cands.append((iou, float(az), float(el), dist, s, tx, ty))
    cands.sort(key=lambda c: -c[0])

    # Keep diverse seeds (different az/el or projection) for refinement.
    seeds = []
    for c in cands:
        if all(abs(((c[1] - o[1] + 180) % 360) - 180) > 2 * az_step or abs(c[2] - o[2]) > 2 * el_step
               or c[3] != o[3] for o in seeds):
            seeds.append(c)
        if len(seeds) >= top_k:
            break

    tgt_f, bbox_f = tgt_at(fine_res)
    src_rgb = cv2.resize(image_rgba[..., :3], (fine_res, fine_res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    src_mask = tgt_f.cpu().numpy()

    def score(az, el, dist, s, tx, ty):
        u, v, _ = splat._uvd(az, el, dist)
        return _iou(splat.silhouette(u, v, s, tx, ty, fine_res), tgt_f)

    refined = []
    for _, az, el, dist, *_ in seeds:
        u, v, _ = splat._uvd(az, el, dist)
        s, tx, ty = splat.init_similarity(u, v, bbox_f, fine_res)
        best = score(az, el, dist, s, tx, ty)
        params = [az, el, s, tx, ty]
        steps = [az_step / 2, el_step / 2, 0.02 * s, 2.0, 2.0]
        for _ in range(6):
            improved = False
            for k in range(5):
                for sgn in (1, -1):
                    trial = list(params)
                    trial[k] += sgn * steps[k]
                    if k == 2 or k == 3 or k == 4:
                        val = score(trial[0], trial[1], dist, trial[2], trial[3], trial[4])
                    else:
                        u, v, _ = splat._uvd(trial[0], trial[1], dist)
                        trial[2], trial[3], trial[4] = splat.init_similarity(u, v, bbox_f, fine_res)
                        val = score(*trial[:2], dist, *trial[2:])
                    if val > best + 1e-5:
                        best, params, improved = val, trial, True
            if not improved:
                steps = [x * 0.5 for x in steps]
        az, el, s, tx, ty = params
        u, v, depth = splat._uvd(az, el, dist)
        rgb, mask = splat.color_image(u, v, depth, s, tx, ty, fine_res)
        csim = _chroma_similarity(rgb.cpu().numpy(), mask.cpu().numpy(), src_rgb, src_mask)
        refined.append((best + color_weight * csim, best, csim, az % 360.0, el, dist, s, tx, ty))
    refined.sort(key=lambda r: -r[0])
    for r in refined[:5]:
        log(f"    cand total={r[0]:.4f} iou={r[1]:.4f} chroma={r[2]:.3f} az={r[3]:.1f} el={r[4]:.1f} dist={r[5]}")
    _, iou, csim, az, el, dist, s, tx, ty = refined[0]

    # Renders in this pipeline come from regular camera grids (e.g. 30-degree
    # steps, ortho_scale 1.0/1.2, object centered). Snap to the nearest such
    # camera when it explains the silhouette as well as the free fit.
    if snap_deg > 0 and dist is None:
        saz = (round(az / snap_deg) * snap_deg) % 360.0
        sel = round(el / snap_deg) * snap_deg
        ss = min((fine_res / o for o in snap_ortho_scales), key=lambda c: abs(c - s))
        siou = score(saz, sel, None, ss, fine_res / 2.0, fine_res / 2.0)
        log(f"    snap az={saz:.1f} el={sel:.1f} ortho_scale={fine_res / ss:.2f} iou={siou:.4f} (free fit {iou:.4f})")
        if siou >= iou - snap_tolerance:
            az, el, s, tx, ty, iou = saz, sel, ss, fine_res / 2.0, fine_res / 2.0, siou
    f = W / float(fine_res)
    return ViewCamera(azimuth=float(az), elevation=float(el), distance=dist, scale=float(s * f),
                      tx=float(tx * f), ty=float(ty * f), width=W, height=H, iou=float(iou), color_score=float(csim))


def render_camera_overlay(cam: ViewCamera, splat: _Splatter, image_rgba: np.ndarray, out_path: Path, res: int = 512):
    """Debug overlay: fitted silhouette (magenta edges) and splatted colors next to the source."""
    k = res / float(cam.width)
    u, v, depth = splat._uvd(cam.azimuth, cam.elevation, cam.distance)
    sil = splat.silhouette(u, v, cam.scale * k, cam.tx * k, cam.ty * k, res).cpu().numpy()
    rgb, mask = splat.color_image(u, v, depth, cam.scale * k, cam.tx * k, cam.ty * k, res)
    src = _rgba_on_white(cv2.resize(image_rgba, (res, res), interpolation=cv2.INTER_AREA))
    edges = cv2.Canny(sil.astype(np.uint8) * 255, 50, 150) > 0
    over = src.copy()
    over[cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (255, 0, 255)
    splat_img = np.where(mask.cpu().numpy()[..., None], (rgb.cpu().numpy() * 255).astype(np.uint8), 255)
    Image.fromarray(np.concatenate([over, splat_img.astype(np.uint8)], axis=1)).save(out_path)


def _rgba_on_white(rgba: np.ndarray) -> np.ndarray:
    if rgba.shape[-1] == 3:
        return rgba.astype(np.uint8)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# B_i: soft 2D edit support from the source-target difference
# --------------------------------------------------------------------------- #
def target_alignment(s_rgb: np.ndarray, t_rgb: np.ndarray, s_mask: np.ndarray, t_mask: np.ndarray,
                     min_inliers: int = 25) -> tuple[np.ndarray, dict]:
    """Source -> target similarity for generated edit targets.

    Features of the shared object (SIFT + RANSAC) when well supported, otherwise a pure
    resize: image editors keep the framing, while bbox-based fallbacks are wrong as soon
    as the target adds parts outside the source silhouette.
    """
    A, feat = edit_regions.estimate_affine_with_features(s_rgb, t_rgb, s_mask, t_mask, "auto")
    if A is not None and feat.get("inliers", 0) >= min_inliers:
        return A.astype(np.float32), {"method": "features", "inliers": int(feat["inliers"])}
    kx, ky = t_rgb.shape[1] / s_rgb.shape[1], t_rgb.shape[0] / s_rgb.shape[0]
    return np.array([[kx, 0, 0], [0, ky, 0]], dtype=np.float32), {"method": "resize", "inliers": int(feat.get("inliers", 0))}


def warp_target_to_source(source_path: Path, target_path: Path, out_path: Path) -> dict:
    """Write the RGBA target warped into the source image's pixel frame (see target_alignment)."""
    s_rgb, s_alpha, _ = edit_regions.load_rgba(source_path)
    t_rgb, t_alpha, _ = edit_regions.load_rgba(target_path)
    s_mask = edit_regions.estimate_foreground_mask(s_rgb, s_alpha, 8, 35.0)
    t_mask = edit_regions.estimate_foreground_mask(t_rgb, t_alpha, 8, 35.0)
    A, info = target_alignment(s_rgb, t_rgb, s_mask, t_mask)
    rgba = np.asarray(Image.open(target_path).convert("RGBA"))
    h, w = s_rgb.shape[:2]
    aligned = cv2.warpAffine(rgba, cv2.invertAffineTransform(A), (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    info["source_coverage"] = float((s_mask & (aligned[..., 3] > 127)).sum() / max(s_mask.sum(), 1))
    Image.fromarray(aligned).save(out_path)
    return info


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def edit_support_2d(source_path: Path, target_path: Path, delta_lo: float = 20.0, delta_hi: float = 45.0,
                    open_frac: float = 0.006, min_area_frac: float = 0.0008,
                    blur_frac: float = 0.008, aligned: bool = False) -> tuple[np.ndarray, dict]:
    """Soft instruction-induced edit region B_i in SOURCE pixel coordinates.

    Targets come from a generative editor that re-renders the whole image, so
    a single LAB threshold fires along every slightly misaligned edge. We use:
      * similarity alignment + local-tolerant LAB delta (detect_image_edit_regions),
      * a soft ramp delta_lo..delta_hi instead of a hard threshold,
      * silhouette changes: new target shape and removed source shape (outside a
        misalignment tolerance band),
      * grayscale opening (~open_frac of the image) of the color channel to delete
        thin misalignment slivers, a light opening of the silhouette channel so
        small removals (leftover fragments) survive, and removal of tiny components,
    then warp back to the source frame and blur.
    """
    s_rgb, s_alpha, _ = edit_regions.load_rgba(source_path)
    t_rgb, t_alpha, _ = edit_regions.load_rgba(target_path)
    s_mask = edit_regions.estimate_foreground_mask(s_rgb, s_alpha, 8, 35.0)
    t_mask = edit_regions.estimate_foreground_mask(t_rgb, t_alpha, 8, 35.0)
    th, tw = t_rgb.shape[:2]
    if aligned:  # target already warped into the source frame (only a resize may remain)
        A = np.array([[tw / s_rgb.shape[1], 0, 0], [0, th / s_rgb.shape[0], 0]], dtype=np.float32)
        align_info = {"method": "given"}
    else:
        A, align_info = edit_regions.estimate_alignment(s_rgb, t_rgb, s_mask, t_mask, "auto", 25.0)
    warped_rgb = cv2.warpAffine(s_rgb, A, (tw, th), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    warped_mask = edit_regions.warp_mask(s_mask, A, (th, tw))
    k = max(th, tw) / 512.0

    delta, near = edit_regions.local_min_lab_delta(warped_rgb, t_rgb, warped_mask, max(2, int(round(4 * k))))
    delta = cv2.GaussianBlur(np.where(np.isfinite(delta), delta, 0.0).astype(np.float32), (0, 0), k)
    color = np.clip((delta - delta_lo) / (delta_hi - delta_lo), 0.0, 1.0) * (t_mask & near)
    tol = _disk(max(2, int(round(4 * k))))
    new_shape = t_mask & ~(cv2.dilate(warped_mask.astype(np.uint8), tol) > 0)
    removed = warped_mask & ~(cv2.dilate(t_mask.astype(np.uint8), tol) > 0)

    # Color slivers come from misaligned edges and need a strong opening; silhouette
    # changes already exclude the misalignment band (tol), so a light opening keeps
    # small removals such as leftover fragments.
    r = max(1, int(round(open_frac * max(th, tw))))
    color = cv2.morphologyEx(color.astype(np.float32), cv2.MORPH_OPEN, _disk(r))
    shape = cv2.morphologyEx((new_shape | removed).astype(np.float32), cv2.MORPH_OPEN, _disk(max(1, r // 3)))

    def keep_components(channel: np.ndarray, min_area: float) -> np.ndarray:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((channel > 0.5).astype(np.uint8), 8)
        keep = np.zeros(channel.shape, dtype=np.uint8)
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == c] = 1
        return channel * (cv2.dilate(keep, _disk(2 * r)) > 0)

    soft = np.maximum(keep_components(color, min_area_frac * th * tw),
                      keep_components(shape, min_area_frac / 8.0 * th * tw))
    soft = cv2.GaussianBlur(soft, (0, 0), blur_frac * max(th, tw))
    soft = np.clip(soft / max(float(soft.max()), 1e-6), 0.0, 1.0)

    sh, sw = s_rgb.shape[:2]
    soft_s = cv2.warpAffine(soft, cv2.invertAffineTransform(A), (sw, sh), flags=cv2.INTER_LINEAR)
    info = {"alignment": align_info["method"], "changed_ratio": float((soft_s > 0.5).mean()),
            "color_change_px": int((color > 0.5).sum()), "new_target_shape_px": int(new_shape.sum()),
            "removed_source_shape_px": int(removed.sum()), "matrix_source_to_target": A.astype(float).tolist()}
    return np.clip(soft_s, 0.0, 1.0).astype(np.float32), info


# --------------------------------------------------------------------------- #
# Support field (Eqs. 4-6)
# --------------------------------------------------------------------------- #
@dataclass
class SupportParams:
    gamma: float = 1.0           # angular selectivity (Eq. 4)
    eps: float = 0.15            # facing stabilizer (Eq. 4)
    vis_band: float = 1.0        # voxels: ray origin offset so surface cells see themselves
    vis_tau: float = 1.5         # voxels: soft z-buffer decay below the first occluder
    psi_min: float = 0.2         # psi at the Omega boundary
    psi_width: float = 3.0       # voxels inside Omega where psi reaches 1
    near_surface: float = 1.5    # voxels: blend from triangle normal to distance gradient
    mu: float = 4.0              # smoothness (Eq. 5)
    affinity_power: float = 2.0  # w_pq = (0.5 (1 + g_p.g_q))^power, floored
    affinity_floor: float = 0.05
    eta: float = 0.05            # normalization stabilizer (Eq. 6)


def _surface_queries(mesh: trimesh.Trimesh, pts: np.ndarray):
    scene = raycasting_scene(mesh)
    q = scene.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
    closest = q["points"].numpy().astype(np.float64)
    prim = q["primitive_ids"].numpy().astype(np.int64)
    tri_n = np.asarray(mesh.face_normals)[np.clip(prim, 0, len(mesh.faces) - 1)]
    return scene, closest, tri_n


def visibility(scene, cam: ViewCamera, pts: np.ndarray, p: SupportParams) -> np.ndarray:
    d = cam.dir_to_camera(pts)
    origins = pts + d * (p.vis_band * VOXEL)
    rays = np.concatenate([origins, d], axis=1).astype(np.float32)
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
    t_hit[t_hit >= cam.dist_to_camera(pts)] = np.inf  # occluders behind the camera do not count
    return np.where(np.isfinite(t_hit), np.exp(-t_hit / (p.vis_tau * VOXEL)), 1.0)


def sample_image(img: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    out = ndi.map_coordinates(img.astype(np.float32), [np.clip(py, 0, H - 1), np.clip(px, 0, W - 1)],
                              order=1, mode="nearest")
    inside = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
    return np.where(inside, out, 0.0)


def build_graph_laplacian(domain: np.ndarray, gvec: np.ndarray, p: SupportParams):
    """6-neighborhood Laplacian on Omega cells with edge-aware weights (Eq. 5)."""
    idx = -np.ones(domain.shape, dtype=np.int64)
    cells = np.argwhere(domain)
    idx[tuple(cells.T)] = np.arange(len(cells))
    rows, cols, vals = [], [], []
    for axis in range(3):
        a = np.take(domain, range(0, GRID - 1), axis=axis) & np.take(domain, range(1, GRID), axis=axis)
        src = np.argwhere(a)
        dst = src.copy()
        dst[:, axis] += 1
        i, j = idx[tuple(src.T)], idx[tuple(dst.T)]
        cos = np.sum(gvec[i] * gvec[j], axis=1)
        w = np.maximum((0.5 * (1.0 + cos)) ** p.affinity_power, p.affinity_floor)
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    n = len(cells)
    Wm = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    L = sps.diags(np.asarray(Wm.sum(axis=1)).ravel()) - Wm
    return cells, L


def compute_support(
    mesh: trimesh.Trimesh,
    domain: np.ndarray,
    cams: Sequence[ViewCamera],
    B_maps: Sequence[np.ndarray],
    p: SupportParams,
    log=print,
) -> dict:
    cells = np.argwhere(domain)
    pts = (cells.astype(np.float64) + 0.5) * VOXEL - 0.5
    scene, closest, tri_n = _surface_queries(mesh, pts)
    off = pts - closest
    dist = np.linalg.norm(off, axis=1)
    gvec = off / np.maximum(dist, 1e-9)[:, None]
    w_far = np.clip(dist / (p.near_surface * VOXEL), 0.0, 1.0)
    w_far = w_far * w_far * (3 - 2 * w_far)

    # psi: signed distance (in voxels) to the Omega boundary, positive inside.
    d_in = ndi.distance_transform_edt(domain)[tuple(cells.T)] - 0.5
    x = np.clip(d_in / p.psi_width, 0.0, 1.0)
    psi = p.psi_min + (1.0 - p.psi_min) * x * x * (3 - 2 * x)

    _, L = build_graph_laplacian(domain, gvec, p)
    A = (sps.identity(len(cells), format="csc") + p.mu * L.tocsc()).tocsc()
    solve = spla.factorized(A)

    terms, r_all, u_all = [], [], []
    for cam, B in zip(cams, B_maps):
        V = visibility(scene, cam, pts, p)
        px, py, _ = cam.project(pts)
        Bv = sample_image(B, px, py)
        dvec = cam.dir_to_camera(pts)
        facing_tri = np.abs(np.sum(tri_n * dvec, axis=1))  # visibility already rejects back faces
        facing_free = np.maximum(np.sum(gvec * dvec, axis=1), 0.0)
        facing = (1 - w_far) * facing_tri + w_far * facing_free
        F_term = (facing + p.eps) ** p.gamma / (1.0 + p.eps) ** p.gamma
        r = V * Bv * F_term * psi
        u = np.clip(solve(r), 0.0, None)
        terms.append({"V": V, "B": Bv, "F": F_term})
        r_all.append(r)
        u_all.append(u)
    u_all = np.stack(u_all)
    chi = u_all / (u_all.sum(axis=0, keepdims=True) + p.eta)

    def to_grid(vals):
        g = np.zeros((len(vals), GRID, GRID, GRID), dtype=np.float32)
        g[(slice(None),) + tuple(cells.T)] = vals
        return g

    return {
        "cells": cells, "psi": psi, "dist": dist,
        "r": to_grid(np.stack(r_all)), "u": to_grid(u_all), "chi": to_grid(chi),
        "V": to_grid(np.stack([t["V"] for t in terms])),
        "B": to_grid(np.stack([t["B"] for t in terms])),
        "F": to_grid(np.stack([t["F"] for t in terms])),
    }


def priority_ownership(chi: np.ndarray) -> np.ndarray:
    """Eq. (7): chi[0] is the base view; later views win where supported."""
    alpha = np.zeros_like(chi)
    remaining = np.ones_like(chi[0])
    for i in range(len(chi) - 1, 0, -1):
        alpha[i] = chi[i] * remaining
        remaining = remaining * (1.0 - chi[i])
    alpha[0] = remaining
    return alpha


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
VIEW_COLORS = np.array([[150, 150, 150], [230, 60, 60], [60, 120, 230], [60, 190, 90], [230, 160, 40],
                        [170, 70, 200]], dtype=np.float32)


def _cubes(cells: np.ndarray, rgba: np.ndarray, path: Path):
    if len(cells) == 0:
        return
    base = trimesh.creation.box(extents=[VOXEL] * 3)
    centers = (cells.astype(np.float64) + 0.5) * VOXEL - 0.5
    mesh = trimesh.util.concatenate([base.copy().apply_translation(c) for c in centers])
    mesh.visual.vertex_colors = np.repeat(rgba.astype(np.uint8), len(base.vertices), axis=0)
    mesh.export(str(path))


def export_heatmap_glb(cells: np.ndarray, values: np.ndarray, path: Path):
    low = np.array([235, 235, 235], dtype=np.float32)
    high = np.array([200, 20, 40], dtype=np.float32)
    v = np.clip(values, 0, 1)[:, None]
    rgb = low * (1 - v) + high * v
    _cubes(cells, np.concatenate([rgb, np.full((len(cells), 1), 255)], axis=1), path)


def export_ownership_glb(cells: np.ndarray, alpha_cells: np.ndarray, path: Path):
    """Color = alpha-weighted mix of per-view colors (view 0 = base, gray)."""
    cols = VIEW_COLORS[np.arange(alpha_cells.shape[0]) % len(VIEW_COLORS)]
    rgb = alpha_cells.T @ cols
    _cubes(cells, np.concatenate([rgb, np.full((len(cells), 1), 255)], axis=1), path)


def project_field_on_view(cam: ViewCamera, field_cells: np.ndarray, cells: np.ndarray, size: int) -> np.ndarray:
    """Max-intensity projection of a per-cell scalar onto the view image (NaN = no cell)."""
    pts = (cells.astype(np.float64) + 0.5) * VOXEL - 0.5
    px, py, _ = cam.project(pts)
    k = size / float(cam.width)
    px, py = np.round(px * k).astype(int), np.round(py * k).astype(int)
    img = np.full((size, size), -1.0, dtype=np.float32)
    rad = max(1, int(round(cam.scale * k * VOXEL * 0.6)))
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            x, y = px + dx, py + dy
            ok = (x >= 0) & (x < size) & (y >= 0) & (y < size)
            np.maximum.at(img, (y[ok], x[ok]), field_cells[ok].astype(np.float32))
    img[img < 0] = np.nan
    return img


def colorize(field: np.ndarray, base_rgb: np.ndarray, cmap=cv2.COLORMAP_INFERNO, alpha: float = 0.65,
             min_value: float | None = None) -> np.ndarray:
    valid = np.isfinite(field)
    if min_value is not None:
        valid &= np.nan_to_num(field) > min_value
    f = np.nan_to_num(np.clip(field, 0, 1))
    col = cv2.cvtColor(cv2.applyColorMap((f * 255).astype(np.uint8), cmap), cv2.COLOR_BGR2RGB)
    out = base_rgb.astype(np.float32).copy()
    out[valid] = out[valid] * (1 - alpha) + col[valid] * alpha
    return out.astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Downstream helpers (used by Easy3E sampler via the npz)
# --------------------------------------------------------------------------- #
def extend_field(chi: np.ndarray, domain: np.ndarray, band: int = 6) -> np.ndarray:
    """Nearest-value extension of chi outside Omega (within `band` voxels).

    mv_fusion feathers the latent mask beyond Omega, so ownership must be
    defined a little outside it; we copy the nearest Omega cell's value.
    """
    dist, (ii, jj, kk) = ndi.distance_transform_edt(~domain, return_indices=True)
    ext = chi[:, ii, jj, kk]
    ext[:, dist > band] = 0.0
    return ext.astype(np.float32)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render_dir", type=Path, required=True,
                    help="Easy3E preprocess dir with mesh.ply, normalized.glb, transforms.json.")
    ap.add_argument("--mask_glb", type=Path, default=None, help="Rough 3D edit mask (mask_all) GLB in the asset's world frame.")
    ap.add_argument("--mask_is_normalized", action="store_true",
                    help="mask_glb is already in the normalized internal frame (e.g. round-2 conflict masks).")
    ap.add_argument("--mask_npy", type=Path, default=None,
                    help="Edit region as a 64^3 boolean .npy in the internal frame (later rounds); overrides --mask_glb.")
    ap.add_argument("--views", required=True, help="Comma-separated view names in acquisition order; the first is the base view.")
    ap.add_argument("--source_images", required=True, help="Comma-separated source renders, one per view.")
    ap.add_argument("--target_images", default=None, help="Comma-separated edited targets (RGBA preferred), one per view.")
    ap.add_argument("--camera_json", type=Path, default=None,
                    help="Optional {view: {azimuth, elevation, distance, scale, tx, ty, width, height}}; missing views are fitted.")
    ap.add_argument("--targets_aligned", action="store_true",
                    help="Targets are already in their source image's pixel frame; skip the 2D alignment.")
    ap.add_argument("--fit_cameras_only", action="store_true",
                    help="Only fit/collect the view cameras and write view_cameras.json.")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fit_points", type=int, default=600000)
    for f in SupportParams.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name}", type=float, default=f.default)
    return ap.parse_args()


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    params = SupportParams(**{k: getattr(args, k) for k in SupportParams.__dataclass_fields__})
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    sources = [Path(p.strip()) for p in args.source_images.split(",") if p.strip()]
    targets = [Path(p.strip()) for p in (args.target_images or "").split(",") if p.strip()]
    assert len(views) == len(sources), "views/sources length mismatch"
    assert args.fit_cameras_only or len(targets) == len(views), "views/targets length mismatch"

    # ---- cameras ----
    known = json.loads(args.camera_json.read_text()) if args.camera_json and args.camera_json.exists() else {}
    pts_c, col_c = sample_colored_points(args.render_dir / "normalized.glb", args.fit_points)
    splat = _Splatter(pts_c, col_c, args.device)
    cams: list[ViewCamera] = []
    for v, src in zip(views, sources):
        rgba = np.array(Image.open(src).convert("RGBA"))
        if v in known:
            cam = ViewCamera(**known[v])
            print(f"[support] view {v}: using provided camera az={cam.azimuth:.1f} el={cam.elevation:.1f}")
        else:
            print(f"[support] view {v}: fitting camera to {src.name} ...")
            cam = fit_camera(splat, rgba)
            print(f"[support] view {v}: az={cam.azimuth:.1f} el={cam.elevation:.1f} dist={cam.distance} "
                  f"iou={cam.iou:.3f} chroma={cam.color_score:.3f}")
        render_camera_overlay(cam, splat, rgba, out / f"camera_fit_{v}.png")
        cams.append(cam)
    (out / "view_cameras.json").write_text(json.dumps({v: asdict(c) for v, c in zip(views, cams)}, indent=2))
    if args.fit_cameras_only:
        print(f"[support] wrote {out / 'view_cameras.json'}")
        return

    # ---- geometry + rough edit mask Omega on the 64^3 grid ----
    mesh = trimesh.load(str(args.render_dir / "mesh.ply"), process=False)
    if args.mask_npy is not None:
        domain = np.load(args.mask_npy).astype(bool)
    elif args.mask_glb is not None:
        scale, offset = (1.0, np.zeros(3)) if args.mask_is_normalized else load_render_transform(args.render_dir)
        domain = solid_mask_from_mesh(load_mesh_internal(args.mask_glb, scale, offset))
    else:
        raise ValueError("Provide --mask_glb or --mask_npy")
    occ = surface_voxels_from_mesh(args.render_dir / "mesh.ply")
    print(f"[support] Omega cells={int(domain.sum())}, source surface voxels={int(occ.sum())}, "
          f"inside Omega={int((occ & domain).sum())}")

    # ---- B_i ----
    B_maps, B_info = [], {}
    for v, src, tgt in zip(views, sources, targets):
        B, info = edit_support_2d(src, tgt, aligned=args.targets_aligned)
        B_maps.append(B)
        B_info[v] = info
        src_rgb = _rgba_on_white(np.array(Image.open(src).convert("RGBA")))
        Image.fromarray(_label(colorize(B, src_rgb, cv2.COLORMAP_JET, 0.5, min_value=0.02),
                               f"B_{v} (2D edit support)")).save(out / f"edit_support_2d_{v}.png")
        print(f"[support] view {v}: B changed_ratio={info['changed_ratio']:.3f} ({info['alignment']})")

    # ---- Eqs. 4-6 ----
    res = compute_support(mesh, domain, cams, B_maps, params)
    chi, cells = res["chi"], res["cells"]
    alpha = priority_ownership(chi)
    chi_ext = extend_field(chi, domain)

    np.savez_compressed(
        out / "view_support.npz",
        views=np.array(views), base_view=np.array(views[0]), domain=domain, source_occupancy=occ,
        chi=chi.astype(np.float32), chi_extended=chi_ext, alpha=alpha.astype(np.float32),
        r=res["r"], u=res["u"], V=res["V"], B=res["B"], F=res["F"],
    )

    # ---- per-view outputs ----
    sel = occ[tuple(cells.T)]  # visualize on the source surface voxels inside Omega
    occ_cells = cells[sel]
    summary = {"views": views, "base_view": views[0], "params": asdict(params),
               "omega_cells": int(len(cells)), "omega_surface_voxels": int(sel.sum()), "per_view": {}}
    panels = []
    for i, (v, cam, src) in enumerate(zip(views, cams, sources)):
        chi_c = chi[(i,) + tuple(cells.T)]
        a_c = alpha[(i,) + tuple(cells.T)]
        export_heatmap_glb(occ_cells, chi_c[sel], out / f"support_chi_{v}_surface.glb")
        export_heatmap_glb(cells[chi_c > 0.5], chi_c[chi_c > 0.5], out / f"support_region_{v}_chi_gt_0.5.glb")
        summary["per_view"][v] = {
            "camera": asdict(cam), "B": B_info[v],
            "chi_mean_omega": float(chi_c.mean()), "chi_p90_omega": float(np.quantile(chi_c, 0.9)),
            "cells_chi_gt_0.5": int((chi_c > 0.5).sum()), "frac_chi_gt_0.5": float((chi_c > 0.5).mean()),
            "ownership_mean_omega": float(a_c.mean()), "cells_owned_gt_0.5": int((a_c > 0.5).sum()),
            "visible_frac_omega": float((res["V"][(i,) + tuple(cells.T)] > 0.5).mean()),
        }
        # 2D: max-intensity projection of every chi_j onto the source image of view i
        src_rgb = cv2.resize(_rgba_on_white(np.array(Image.open(src).convert("RGBA"))), (384, 384))
        row = [_label(colorize(cv2.resize(B_maps[i], (384, 384)), src_rgb, cv2.COLORMAP_JET, 0.5, min_value=0.02),
                      f"view {v}: B_{v}")]
        for j, vj in enumerate(views):
            proj = project_field_on_view(cam, chi[(j,) + tuple(cells.T)], cells, 384)
            row.append(_label(colorize(proj, src_rgb, min_value=0.0), f"chi_{vj} (MIP) on view {v}"))
        own = src_rgb.astype(np.float32).copy()
        for j in range(1, len(views)):
            proj = np.nan_to_num(project_field_on_view(cam, alpha[(j,) + tuple(cells.T)], cells, 384))[..., None]
            own = own * (1 - 0.7 * proj) + VIEW_COLORS[j % len(VIEW_COLORS)] * 0.7 * proj
        row.append(_label(own.astype(np.uint8), "ownership of later views"))
        panels.append(np.concatenate(row, axis=1))
    Image.fromarray(np.concatenate(panels, axis=0)).save(out / "support_projections.png")
    export_ownership_glb(cells, alpha[(slice(None),) + tuple(cells.T)], out / "ownership_omega.glb")
    export_ownership_glb(occ_cells, alpha[(slice(None),) + tuple(occ_cells.T)], out / "ownership_surface.glb")
    (out / "view_support_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({v: {k: summary["per_view"][v][k] for k in ("chi_mean_omega", "frac_chi_gt_0.5", "ownership_mean_omega")}
                      for v in views}, indent=2))
    print(f"[support] wrote {out / 'view_support.npz'}")


if __name__ == "__main__":
    main()
