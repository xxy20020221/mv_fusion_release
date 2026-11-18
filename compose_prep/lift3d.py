"""3D-lifted layout guides for multi-view composition.

Image editors (and VLM judges) place an added object by 2D habit, e.g. headphones always get
their band across the image and the earcups at the image edges, which is only right in a front
view. To give every secondary view a geometrically consistent target, the accepted primary
composite is lifted to 3D with TRELLIS.2, aligned to the anchor's normalized frame, and
rendered from each selected camera. The renders serve as layout guides for Nano Banana and as
a VLM-free geometric check.

Alignment: multi-start (yaw) similarity ICP of the anchor's normalized mesh onto the
reconstruction (the anchor is a subset of it), disambiguated by silhouette agreement with the
primary composite at the known primary camera.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRELLIS2_RUNNER = Path(__file__).resolve().parent / "run_trellis2_image_to_3d.py"
# Python of a separate TRELLIS.2 environment (https://github.com/microsoft/TRELLIS.2).
DEFAULT_TRELLIS2_PYTHON = Path(os.getenv("TRELLIS2_PYTHON", "python"))


# --------------------------------------------------------------------------- #
# External tools
# --------------------------------------------------------------------------- #
def run_trellis2(image: Path, out_dir: Path, python: Path, quality: str, log_path: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), str(TRELLIS2_RUNNER), "--image-path", str(image), "--out-dir", str(out_dir),
           "--quality", quality, "--suffix", "lift", "--skip-preview"]
    with log_path.open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT)).returncode
    glb = out_dir / f"{image.stem}_trellis2_lift.glb"
    if rc != 0 or not glb.exists():
        raise RuntimeError(f"TRELLIS.2 lift failed (rc={rc}); see {log_path}")
    return glb


def blender_export_mesh(blender: Path, script: Path, glb: Path, npz: Path, normalize: bool, log_path: Path) -> None:
    cmd = [str(blender), "-b", "--factory-startup", "-P", str(script), "--",
           "--object", str(glb), "--output_dir", str(npz.parent), "--export_mesh_npz", str(npz)]
    if not normalize:
        cmd.append("--no_normalize")
    _run_blender(cmd, log_path)
    if not npz.exists():
        raise RuntimeError(f"Blender mesh export failed; see {log_path}")


def blender_render_aligned(blender: Path, script: Path, glb: Path, transform: Path, views: dict[str, tuple[float, float]],
                           out_dir: Path, resolution: int, ortho_scale: float, samples: int, gpu: bool, log_path: Path) -> None:
    cmd = [str(blender), "-b", "--factory-startup", "-P", str(script), "--",
           "--object", str(glb), "--output_dir", str(out_dir), "--no_normalize", "--object_transform", str(transform),
           "--views", ";".join(f"{az},{el}" for az, el in views.values()), "--view_ids", ",".join(views),
           "--resolution", str(resolution), "--ortho_scale", str(ortho_scale), "--samples", str(samples)]
    if gpu:
        cmd.append("--gpu")
    _run_blender(cmd, log_path)


def _run_blender(cmd: list[str], log_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"Blender failed (rc={rc}); see {log_path}")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def sample_surface(npz: Path, count: int, seed: int = 0) -> np.ndarray:
    data = np.load(npz)
    mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float64)


def umeyama(src: np.ndarray, ref: np.ndarray, scale_bounds: tuple[float, float] | None = None) -> np.ndarray:
    """Least-squares similarity T with ref ~= T(src)."""
    src_mu, ref_mu = src.mean(0), ref.mean(0)
    src_c, ref_c = src - src_mu, ref - ref_mu
    u, d, vt = np.linalg.svd(ref_c.T @ src_c / len(src))
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rot = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / (np.sum(src_c ** 2) / len(src)))
    if scale_bounds is not None:
        scale = float(np.clip(scale, *scale_bounds))
    mat = np.eye(4)
    mat[:3, :3] = scale * rot
    mat[:3, 3] = ref_mu - scale * rot @ src_mu
    return mat


def apply(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return points @ mat[:3, :3].T + mat[:3, 3]


def rot_z(degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def project(points: np.ndarray, camera_npy: Path, size: int) -> np.ndarray:
    """Pixel coordinates (u, v) on a size x size image for the ortho camera saved by the renderer."""
    cam = np.load(camera_npy, allow_pickle=True).item()
    rt, k = np.asarray(cam["extrinsic"]), np.asarray(cam["intrinsic"])
    cam_pts = points @ rt[:, :3].T + rt[:, 3]
    uv = cam_pts[:, :2] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]
    return uv * (size / float(cam["resolution"]))


def splat_mask(points: np.ndarray, camera_npy: Path, size: int) -> np.ndarray:
    uv = np.round(project(points, camera_npy, size)).astype(int)
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
    mask = np.zeros((size, size), dtype=bool)
    mask[uv[keep, 1], uv[keep, 0]] = True
    mask = ndimage.binary_dilation(mask, iterations=2)
    # No hole filling: thin loops (straps, handles) legitimately enclose background.
    return ndimage.binary_closing(mask, iterations=2)


def foreground_mask(image_path: Path, size: int) -> np.ndarray:
    image = Image.open(image_path)
    if image.mode == "RGBA":
        return np.asarray(image.getchannel("A").resize((size, size), Image.Resampling.BILINEAR)) > 127
    pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).astype(np.int16)
    return (pixels < 235).any(axis=-1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def bbox_extent(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.max() - xs.min() + 1, ys.max() - ys.min() + 1], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def align_reconstruction(recon: np.ndarray, anchor: np.ndarray, primary_camera: Path, primary_composite: Path,
                         anchor_render: Path, yaw_step: float = 30.0, iters: int = 40, trim: float = 0.85,
                         size: int = 256) -> tuple[np.ndarray, dict[str, Any]]:
    """Similarity T mapping reconstruction points into the anchor's normalized frame."""
    composite_fg = foreground_mask(primary_composite, size)
    anchor_fg = foreground_mask(anchor_render, size)
    # Sanity check of the camera model: the projected anchor must match its own render.
    projection_check = iou(splat_mask(anchor, primary_camera, size), anchor_fg)

    # Initial scale: the composite is larger than the anchor alone by roughly the 2D bbox ratio.
    ratio = float(np.max(bbox_extent(composite_fg) / bbox_extent(anchor_fg)))
    recon_longest = float(np.max(np.ptp(recon, axis=0)))
    anchor_longest = float(np.max(np.ptp(anchor, axis=0)))
    s0 = anchor_longest * ratio / recon_longest
    bounds = (0.6 * s0, 1.6 * s0)

    # Correspondences go anchor -> moving reconstruction (the anchor is a subset of it).
    candidates = []
    for yaw in np.arange(0.0, 360.0, yaw_step):
        T = np.eye(4)
        T[:3, :3] = s0 * rot_z(yaw)
        T[:3, 3] = anchor.mean(0) - T[:3, :3] @ recon.mean(0)
        dist = np.inf
        for _ in range(iters):
            moved = apply(recon, T)
            d, idx = cKDTree(moved).query(anchor, k=1)
            keep = d <= np.quantile(d, trim)
            T = umeyama(recon[idx[keep]], anchor[keep], bounds)
            new_dist = float(d[keep].mean())
            if abs(dist - new_dist) < 1e-6:
                break
            dist = new_dist
        silhouette = iou(splat_mask(apply(recon, T), primary_camera, size), composite_fg)
        candidates.append({"yaw_init": float(yaw), "dist": dist, "silhouette_iou": silhouette, "T": T})

    best_dist = min(c["dist"] for c in candidates)
    plausible = [c for c in candidates if c["dist"] <= 1.5 * best_dist]
    best = max(plausible, key=lambda c: (c["silhouette_iou"], -c["dist"]))
    report = {
        "projection_check_iou": projection_check,
        "initial_scale": s0,
        "scale": float(np.cbrt(abs(np.linalg.det(best["T"][:3, :3])))),
        "yaw_init": best["yaw_init"],
        "anchor_to_recon_dist": best["dist"],
        "primary_silhouette_iou": best["silhouette_iou"],
        "candidates": [{k: v for k, v in c.items() if k != "T"} for c in candidates],
    }
    return best["T"], report


def other_region(image_path: Path, anchor_render: Path, size: int = 256, dilate: int = 3) -> np.ndarray:
    """Foreground that is not covered by the (slightly dilated) anchor silhouette, i.e. the added object."""
    anchor = ndimage.binary_dilation(foreground_mask(anchor_render, size), iterations=dilate)
    return foreground_mask(image_path, size) & ~anchor


def guide_visual(guide: Path, anchor_render: Path, out: Path, tint: tuple[int, int, int] = (0, 190, 255),
                 strength: float = 0.45) -> Path:
    """Grayscale 'clay' version of a guide render with the added object tinted.

    Editors copy whatever appearance a guide shows (the reconstruction's simplified materials), and
    VLM judges mistake the guide's surfaces for the composite's; removing color keeps only shape,
    placement and occlusion, and the tint marks which region is the added object.
    """
    rgba = Image.open(guide).convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    gray = np.asarray(canvas.convert("L").convert("RGB")).astype(np.float32)
    size = rgba.size[0]
    region = other_region(guide, anchor_render, size=size, dilate=max(1, size // 128))
    gray[region] = (1 - strength) * gray[region] + strength * np.asarray(tint, dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray.clip(0, 255).astype(np.uint8)).save(out)
    return out


def lift_primary(primary_composite: Path, anchor_render_dir: Path, anchor_glb: Path, primary_cid: str,
                 views: dict[str, tuple[float, float]], work: Path, blender: Path, render_script: Path,
                 resolution: int, ortho_scale: float, gpu: bool, trellis2_python: Path, quality: str) -> dict[str, Any]:
    """Reconstruct the primary composite in 3D, align it to the anchor frame and render guides.

    Returns {"guides": {cid: png}, "visuals": {cid: png}, "glb", "transform", "report"}.
    """
    work.mkdir(parents=True, exist_ok=True)
    glb = run_trellis2(primary_composite, work / "trellis2", trellis2_python, quality, work / "trellis2.log")

    anchor_npz = anchor_render_dir / "mesh_world.npz"
    if not anchor_npz.exists():
        blender_export_mesh(blender, render_script, anchor_glb, anchor_npz, True, work / "export_anchor.log")
    recon_npz = work / "recon_world.npz"
    blender_export_mesh(blender, render_script, glb, recon_npz, False, work / "export_recon.log")

    T, report = align_reconstruction(
        sample_surface(recon_npz, 30000), sample_surface(anchor_npz, 8000),
        anchor_render_dir / "camera" / f"{primary_cid}.npy", primary_composite, anchor_render_dir / f"{primary_cid}.png")
    transform = work / "recon_to_anchor.npy"
    np.save(transform, T)
    (work / "alignment.json").write_text(json.dumps(report, indent=2))
    print(f"[lift] alignment: projection_check_iou={report['projection_check_iou']:.3f} "
          f"silhouette_iou={report['primary_silhouette_iou']:.3f} dist={report['anchor_to_recon_dist']:.4f} "
          f"scale={report['scale']:.3f} yaw_init={report['yaw_init']:.0f}", flush=True)

    guide_dir = work / "guides"
    blender_render_aligned(blender, render_script, glb, transform, views, guide_dir, resolution, ortho_scale,
                           64, gpu, work / "render_guides.log")
    guides = {cid: guide_dir / f"{cid}.png" for cid in views}
    missing = [str(p) for p in guides.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"guide renders missing: {missing}")
    visuals = {cid: guide_visual(path, anchor_render_dir / f"{cid}.png", work / "guides_vis" / f"{cid}.png")
               for cid, path in guides.items()}
    return {"guides": guides, "visuals": visuals, "glb": glb, "transform": transform, "report": report}


def load_lift(prev: Path, work: Path, cids: list[str]) -> dict[str, Any] | None:
    """Copy a finished lift of an earlier run into `work` and return it like lift_primary, or None if
    it does not cover every requested camera."""
    import shutil
    glbs = sorted((prev / "trellis2").glob("*_trellis2_lift.glb"))
    if not glbs or not all((prev / d / f"{cid}.png").is_file() for d in ("guides", "guides_vis") for cid in cids):
        return None
    shutil.copytree(prev, work, dirs_exist_ok=True)
    return {"guides": {cid: work / "guides" / f"{cid}.png" for cid in cids},
            "visuals": {cid: work / "guides_vis" / f"{cid}.png" for cid in cids},
            "glb": work / "trellis2" / glbs[0].name, "transform": work / "recon_to_anchor.npy",
            "report": json.loads((work / "alignment.json").read_text()), "reused_from": str(prev)}
