#!/usr/bin/env python3
"""Round r -> r+1 transition for Easy3E mv_fusion (Route-Diagnose-Refine Sec. 3.3-3.4).

* Diagnosis states of round r on its active voxels (quiescent_voxels.glb palette):
  C = conflict / unresolved (red), U = quiescent / source-stable (blue),
  S = edited-stable = (active & Omega^r) minus (C | U).
* Next active region (Eq. 16), snapped to the latent cells it touches and keeping
  the preserved set U out:
      Omega^{r+1} = Omega^r & Snap(Close(Dilate(C^r))) & ~U^r
* Source renders I^{s,r+1}_i of every view from the round-r output (Alg. 1, line 3),
  tone-matched to the user's original source renders so that the source/target
  contrast of old views does not pick up a renderer difference.
* Target synthesis for the newly acquired view with Nano Banana (api/magic666_client.py,
  OpenAI-compatible gateway), conditioned on the primary-view source/target pair
  (edit intent) and the other edited views (multi-view consistency).

CLI: `python mv_fusion_rounds.py synthesize ...` runs only the target synthesis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from paths import API_DIR  # noqa: E402
import compute_view_support as cvs  # noqa: E402
import plan_mv_fusion_next_view as planner  # noqa: E402

GRID = 64


# --------------------------------------------------------------------------- #
# Diagnosis states and the next active region
# --------------------------------------------------------------------------- #
def _voxel_grid(centers: np.ndarray, select: np.ndarray) -> np.ndarray:
    idx = np.clip(np.round((centers + 0.5) * GRID - 0.5).astype(int), 0, GRID - 1)
    grid = np.zeros((GRID,) * 3, dtype=bool)
    grid[tuple(idx[select].T)] = True
    return grid


def load_diagnosis_states(quiescent_glb: Path) -> dict[str, np.ndarray]:
    """Active / conflict / quiescent voxel grids from flow_euler's quiescent_voxels.glb.

    Palette of _export_coords_category_glb: label 2 (conflict) = red [255,80,80],
    label 1 (quiescent) = blue [80,180,255], label 0 (neutral) = gray.
    """
    centers, colors = planner.voxel_centers_and_colors(quiescent_glb)
    rgb = colors[:, :3].astype(int)
    return {
        "active": _voxel_grid(centers, np.ones(len(centers), dtype=bool)),
        "conflict": _voxel_grid(centers, (rgb[:, 0] > 220) & (rgb[:, 1] < 130) & (rgb[:, 2] < 130)),
        "quiescent": _voxel_grid(centers, (rgb[:, 2] > 220) & (rgb[:, 0] < 130)),
    }


def round_edit_region(mask_glb: Path, render_dir: Path) -> np.ndarray:
    """Omega^r on the 64^3 grid exactly as Easy3E voxelized mask_all in round r."""
    scale, offset = cvs.load_render_transform(render_dir)
    return cvs.solid_mask_from_mesh(cvs.load_mesh_internal(mask_glb, scale, offset))


def _ball(r: int) -> np.ndarray:
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return np.linalg.norm(g, axis=0) <= r + 1e-6


def snap_to_cells(mask: np.ndarray, cell: int) -> np.ndarray:
    """Every cell x cell x cell block touched by `mask`, at full resolution."""
    n = GRID // cell
    touched = mask.reshape(n, cell, n, cell, n, cell).any(axis=(1, 3, 5))
    return touched.repeat(cell, 0).repeat(cell, 1).repeat(cell, 2)


def next_active_region(omega: np.ndarray, states: dict[str, np.ndarray], dilate: int = 2, close: int = 2,
                       exclude_quiescent: bool = True, latent_cell: int = 4) -> tuple[np.ndarray, dict]:
    """Omega^{r+1} = Omega^r & Snap(Close(Dilate(C^r))) & ~U^r.

    Snap: the sparse-structure stage edits whole latent cells (4^3 voxels for the
    16^3 latent), and Easy3E freezes cells that are at most half covered by the
    mask. Voxels in such frozen cells were never editable in round r, so their
    low disagreement is no evidence that they are resolved (e.g. arm remnants left
    at the Omega boundary). Growing the region to the latent cells it touches (and
    making them editable) lets round r+1 reach them; it never leaves Omega^r.
    """
    C, U = states["conflict"] & omega, states["quiescent"] & omega
    grown = ndi.binary_dilation(C, _ball(dilate)) if dilate > 0 else C
    if close > 0:
        grown = ndi.binary_closing(np.pad(grown, close), _ball(close))[close:-close, close:-close, close:-close]
    unsnapped = omega & grown
    if latent_cell > 1:
        grown = snap_to_cells(grown, latent_cell)
    nxt = omega & grown
    if exclude_quiescent:
        nxt &= ~U
    S = states["active"] & omega & ~C & ~U
    info = {
        "omega_r_cells": int(omega.sum()), "active_in_omega_r": int((states["active"] & omega).sum()),
        "conflict_C": int(C.sum()), "quiescent_U": int(U.sum()), "edited_stable_S": int(S.sum()),
        "unresolved_fraction_C_over_active": float(C.sum() / max((states["active"] & omega).sum(), 1)),
        "omega_next_cells": int(nxt.sum()), "omega_next_active": int((nxt & states["active"]).sum()),
        "omega_next_cells_before_snap": int((unsnapped & ~U if exclude_quiescent else unsnapped).sum()),
        "dilate": dilate, "close": close, "latent_cell": latent_cell, "exclude_quiescent": exclude_quiescent,
    }
    return nxt, info


def export_state_glbs(states: dict[str, np.ndarray], omega: np.ndarray, omega_next: np.ndarray, out_dir: Path):
    """round_states.glb: active voxels of Omega^r (red C, blue U, gray S);
    next_edit_region.glb: all cells of Omega^{r+1} (yellow = free space, orange = occupied)."""
    act = states["active"] & omega
    cells = np.argwhere(act)
    lab = np.full(len(cells), 0)
    lab[states["conflict"][tuple(cells.T)]] = 2
    lab[states["quiescent"][tuple(cells.T)]] = 1
    pal = np.array([[170, 170, 170, 255], [80, 180, 255, 255], [255, 80, 80, 255]])
    cvs._cubes(cells, pal[lab], out_dir / "round_states.glb")
    cells = np.argwhere(omega_next)
    occ = states["active"][tuple(cells.T)]
    cvs._cubes(cells, np.where(occ[:, None], [[255, 140, 0, 255]], [[255, 230, 90, 160]]), out_dir / "next_edit_region.glb")


# --------------------------------------------------------------------------- #
# Source renders of the current asset
# --------------------------------------------------------------------------- #
def render_views(glb: Path, angles: dict[str, tuple[float, float]], out_dir: Path, *, resolution: int = 1024,
                 samples: int = 128, shrink: float = 0.0, ortho_scale: float = 1.2) -> dict[str, Path]:
    """Orthographic CYCLES renders (blender_scripts/render_glb_mv_high_detail.py) of `glb` at {view: (az, el)}."""
    sel = {v: {"azimuth": float(a), "elevation": float(e)} for v, (a, e) in angles.items()}
    paths = planner.render_selected_views(
        glb=glb, output_dir=out_dir, selections=sel, object_label="render", resolution=resolution,
        samples=samples, shrink=shrink, ortho_scale=ortho_scale, distance=2.0, engine="CYCLES",
        cycles_device="CPU",
        runtime_prefix=planner.DEFAULT_RUNTIME_PREFIX if planner.DEFAULT_RUNTIME_PREFIX.exists() else None,
        use_xvfb=False, xvfb_display=":0")
    return {v: Path(p) for v, p in paths.items()}


def recentered_glb(glb: Path, look_at: np.ndarray, out: Path) -> Path:
    """Copy of `glb` translated so `look_at` (internal Z-up frame) is at the origin.

    The render script's cameras look at the origin; this reproduces cameras that look
    at `look_at` instead. internal (x, y, z) = glTF (x, -z, y).
    """
    import trimesh

    scene = trimesh.load(str(glb), force="scene")
    c = np.asarray(look_at, dtype=np.float64)
    scene.apply_translation([-c[0], -c[2], c[1]])
    scene.export(str(out))
    return out


def fit_tone_lut(pairs: list[tuple[Path, Path]]) -> np.ndarray:
    """Per-channel monotone LUT (3, 256) mapping our renders to the reference renders.

    Quantile matching over the foreground pixels shared by each (render, reference)
    pair; both images show the same geometry from the same camera, so only the
    renderer's tone differs.
    """
    ours, refs = [], []
    for render, ref in pairs:
        a = np.asarray(Image.open(render).convert("RGBA"))
        b = np.asarray(Image.open(ref).convert("RGBA").resize(a.shape[1::-1], Image.BILINEAR))
        m = (a[..., 3] > 200) & (b[..., 3] > 200)
        ours.append(a[..., :3][m])
        refs.append(b[..., :3][m])
    ours, refs = np.concatenate(ours), np.concatenate(refs)
    q = np.linspace(0, 1, 257)
    lut = np.zeros((3, 256), dtype=np.float32)
    for c in range(3):
        lut[c] = np.interp(np.arange(256), np.quantile(ours[:, c], q), np.quantile(refs[:, c], q))
    return lut


def apply_tone_lut(src: Path, dst: Path, lut: np.ndarray) -> None:
    rgba = np.asarray(Image.open(src).convert("RGBA")).copy()
    for c in range(3):
        rgba[..., c] = np.clip(lut[c][rgba[..., c]], 0, 255).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(dst)


# --------------------------------------------------------------------------- #
# Target synthesis for the new view (nano banana)
# --------------------------------------------------------------------------- #
def _magic666():
    """Image-editing API client (api/magic666_client.py); needs MAGIC666_API_KEY."""
    sys.path.insert(0, str(API_DIR))
    import magic666_client  # noqa: E402
    return magic666_client


SYNTH_PROMPT = """You are editing a rendering of a 3D asset so that it shows an edit that has already been designed in other views of the same asset.

Image 1 is the CURRENT rendering of the asset from a NEW camera ({view_desc}). This is the image you must edit and return.
Image 2 is a reference sheet of the SAME asset seen from OTHER cameras. Its first panel ("BEFORE") shows the original asset; the other panels ("AFTER") show the asset after the desired edit. Use the sheet only to understand the edit; never copy its viewpoints.

Edit instruction: {instruction}

Requirements:
- Return exactly one image with exactly the same camera as Image 1: {view_keep}. Same framing, scale and position; do not rotate, zoom, crop or move the object.
- Show the edited asset of the AFTER panels as it would look from Image 1's camera: same body pose, arm and hand placement, clothing, colors and accessories.
- Image 1 may already be partially edited or contain leftover fragments; remove or correct them so the result is consistent with the AFTER panels.
- Change only what the edit requires; keep every other part (body, limbs, hair, textures, props) as in Image 1.
- Keep the clean rendering style and lighting of Image 1 on a plain pure white background. No text, borders, ground shadows or extra objects."""


def _reference_sheet(primary_pair: tuple[Path, Path], extra_targets: list[Path], height: int = 512) -> Image.Image:
    panels = []
    for path, label in [(primary_pair[0], "BEFORE"), (primary_pair[1], "AFTER")] + [(p, "AFTER") for p in extra_targets]:
        img = Image.open(path)
        rgb = cvs._rgba_on_white(np.asarray(img.convert("RGBA"))) if img.mode in ("RGBA", "LA") else np.asarray(img.convert("RGB"))
        panel = Image.fromarray(rgb).resize((height, height), Image.BILINEAR)
        panels.append(Image.fromarray(cvs._label(np.asarray(panel), label)))
    sheet = Image.new("RGB", (height * len(panels), height), "white")
    for i, p in enumerate(panels):
        sheet.paste(p, (i * height, 0))
    return sheet


def synthesize_view_target(source_png: Path, primary_pair: tuple[Path, Path], extra_targets: list[Path],
                           instruction: str, view_desc: str, view_keep: str, out_dir: Path,
                           model: str | None = None, n_candidates: int = 2, max_attempts: int = 6,
                           min_inliers: int = 12, max_scale_error: float = 0.1) -> dict:
    """Generate target candidates until `n_candidates` keep Image 1's framing; pick the best aligned.

    Framing is checked by similarity-aligning the candidate to the source render
    (SIFT + RANSAC): a candidate that switched viewpoint has almost no inliers or
    a wrong scale and is rejected.
    """
    client = _magic666()
    model = model or os.environ.get("MAGIC666_IMAGE_MODEL", client.IMAGE_MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = SYNTH_PROMPT.format(view_desc=view_desc, view_keep=view_keep, instruction=instruction)
    src_on_white = Image.fromarray(cvs._rgba_on_white(np.asarray(Image.open(source_png).convert("RGBA"))))
    sheet = _reference_sheet(primary_pair, extra_targets)
    sheet.save(out_dir / "reference_sheet.png")
    (out_dir / "prompt.txt").write_text(prompt)

    s_rgb, s_alpha, _ = cvs.edit_regions.load_rgba(source_png)
    s_mask = cvs.edit_regions.estimate_foreground_mask(s_rgb, s_alpha, 8, 35.0)
    candidates = []
    for k in range(max_attempts):
        if sum(c["accepted"] for c in candidates) >= n_candidates:
            break
        path = out_dir / f"candidate_{k}.png"
        if not path.exists():
            print(f"[synth] requesting candidate {k} from {model} ...", flush=True)
            client.generate_image(prompt, [src_on_white, sheet], model=model).save(path)
        c_rgb, c_alpha, _ = cvs.edit_regions.load_rgba(path)
        c_mask = cvs.edit_regions.estimate_foreground_mask(c_rgb, c_alpha, 8, 35.0)
        _, info = cvs.edit_regions.estimate_alignment(s_rgb, c_rgb, s_mask, c_mask, "auto", 25.0)
        feat = info["feature"]
        expected = c_rgb.shape[1] / float(s_rgb.shape[1])
        scale = float(np.sqrt(abs(np.linalg.det(np.asarray(feat["matrix"])[:, :2])))) if "matrix" in feat else 0.0
        cand = {"path": str(path), "inliers": int(feat.get("inliers", 0)),
                "inlier_ratio": float(feat.get("inlier_ratio", 0.0)),
                "framing_scale_error": abs(scale / expected - 1.0)}
        cand["accepted"] = cand["inliers"] >= min_inliers and cand["framing_scale_error"] <= max_scale_error
        print(f"[synth] candidate {k}: {cand}", flush=True)
        candidates.append(cand)
    accepted = [c for c in candidates if c["accepted"]]
    if not accepted:
        raise RuntimeError(f"No candidate kept the new view's framing after {len(candidates)} attempts; see {out_dir}")
    best = max(accepted, key=lambda c: (c["inlier_ratio"], c["inliers"]))
    result = {"model": model, "prompt_file": str(out_dir / "prompt.txt"), "source": str(source_png),
              "reference_sheet": str(out_dir / "reference_sheet.png"),
              "references": [str(p) for p in (*primary_pair, *extra_targets)], "candidates": candidates,
              "selected": best["path"]}
    (out_dir / "synthesis.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthesize", help="Generate the edited target of a new view with nano banana.")
    s.add_argument("--source", type=Path, required=True, help="Current-asset render of the new view.")
    s.add_argument("--primary_source", type=Path, required=True)
    s.add_argument("--primary_target", type=Path, required=True)
    s.add_argument("--extra_targets", default="", help="Comma-separated edited targets of other views.")
    s.add_argument("--instruction", required=True)
    s.add_argument("--view_desc", default="a new viewpoint")
    s.add_argument("--view_keep", default="the same viewpoint as Image 1")
    s.add_argument("--out_dir", type=Path, required=True)
    s.add_argument("--model", default=None)
    s.add_argument("--n_candidates", type=int, default=2)
    s.add_argument("--max_attempts", type=int, default=6)
    args = ap.parse_args()
    extra = [Path(p) for p in args.extra_targets.split(",") if p.strip()]
    res = synthesize_view_target(args.source, (args.primary_source, args.primary_target), extra, args.instruction,
                                 args.view_desc, args.view_keep, args.out_dir, args.model, args.n_candidates,
                                 args.max_attempts)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
