#!/usr/bin/env python3
"""Does an unseen view inflate the differential-flow disagreement (Eq. 10) where the edit is correct?

Controlled test on a finished edit S -> T (round-1 result). Source/target images of every view are
rendered from S and T with the same renderer and cameras, so the targets agree with the 3D state by
construction and only visibility differs between views. The sparse-structure state is clamped to the
correct edit x_T, and along the FlowEdit schedule (shared noise) we compute, per view,
    Delta v_i = v(z_tar, t, c^t_i) - lambda v(z_src, t, c^s_i)                  (Eq. 3)
    x0_i      = denoised prediction of the edited structure from v(z_tar, t, c^t_i)
and, per view pair, the disagreement D_ij(p) = mean_t ||Delta v_i - Delta v_j||_2 (Eq. 10) on the
16^3 latent cells.

Cells of the edit mask whose geometry changed (S vs T) are grouped per view pair by visibility
(ray casting against T): seen by both views, or by one view only. For one-view cells the unseen
view's guess is scored independently of D, against the ground truth x_T: it "guessed right" when its
prediction error mean_t ||x0_j - x_T|| is no larger than the seeing view's. The table reports D and
the share of cells above the paper's conflict threshold tau_C = mean + std of D over the mask (Eq. 12).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "mv_fusion"))
sys.path.insert(0, str(PROJECT_ROOT / "Easy3E"))
import compute_view_support as cvs  # noqa: E402
import mv_fusion_rounds as rounds  # noqa: E402
from evaluate_view_support_runs import camera_look_at  # noqa: E402
from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402

GRID = 64
GROUPS = ("both", "one_right", "one_wrong")


def render_views(glb: Path, cams: dict, out_dir: Path) -> dict:
    """RGBA renders of `glb` (internal frame) at cameras sharing a look-at point."""
    out_dir.mkdir(parents=True, exist_ok=True)
    look_at = camera_look_at(cams)
    if look_at.any():
        glb = rounds.recentered_glb(glb, look_at, out_dir / "recentered.glb")
    c0 = next(iter(cams.values()))
    return rounds.render_views(glb, {v: (c["azimuth"], c["elevation"]) for v, c in cams.items()}, out_dir,
                               resolution=int(c0["width"]), samples=64, ortho_scale=c0["width"] / c0["scale"])


def mesh_internal(glb: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(glb), force="scene")
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    return trimesh.Trimesh(cvs.gltf_to_internal(np.asarray(mesh.vertices)), np.asarray(mesh.faces), process=False)


def camera_json(item: Path, run: str) -> Path:
    """Composition cases store their cameras next to the item; other cases use the run's fitted cameras."""
    for path in (item / "view_cameras_compose.json", item / run / "view_support" / "view_cameras.json"):
        if path.exists():
            return path
    raise FileNotFoundError(f"No view cameras for {item} / {run}")


def edit_region(item: Path, run_work: Path) -> np.ndarray:
    """The round's 64^3 edit mask: from build_rough_edit_mask.py, else voxelized from the run's mask_all.glb."""
    if (item / "_mask_build" / "mask_all_64.npy").exists():
        return np.load(item / "_mask_build" / "mask_all_64.npy")
    return rounds.round_edit_region(run_work / "3Dmasks" / "mask_all.glb", run_work / "render")


@torch.no_grad()
def compute(args) -> Path:
    item = args.item_dir.resolve()
    run_out = item / args.run
    run_work = item / args.run.replace("_out_", "_work_", 1)
    out = run_out / "unseen_view_experiment"
    out.mkdir(parents=True, exist_ok=True)
    views = [v.strip() for v in args.views.split(",")]
    cams = {v: c for v, c in json.loads(camera_json(item, args.run).read_text()).items() if v in views}
    glb_T = sorted((run_out / "out").glob("*_rgba.glb"))[-1]
    src_png = render_views(run_work / "render" / "normalized.glb", cams, out / "render_S")
    tgt_png = render_views(glb_T, cams, out / "render_T")

    pipe = TrellisImageTo3DPipeline.from_pretrained(os.environ["TRELLIS_CHECKPOINT"])
    pipe.cuda()
    flow, dec, sampler = pipe.models["sparse_structure_flow_model"], pipe.models["sparse_structure_decoder"], pipe.sparse_structure_sampler
    params = pipe.sparse_structure_sampler_params
    conds = {}
    for v in views:
        s_img, t_img = pipe.preprocess_image_pair(Image.open(src_png[v]), Image.open(tgt_png[v]))
        conds[v] = (pipe.get_cond([s_img]), pipe.get_cond([t_img]))

    def velocity(x, t, cond):
        return sampler._inference_model(flow, x, t, cond["cond"], neg_cond=cond["neg_cond"],
                                        cfg_strength=params["cfg_strength"], cfg_interval=params["cfg_interval"])

    x_S = torch.load(run_work / "render" / "ss_latent.pt", map_location="cuda").float()
    if (run_out / "features" / "ss_latent_edited.pt").exists():
        x_T = torch.load(run_out / "features" / "ss_latent_edited.pt", map_location="cuda").float()
    else:  # older runs: encode the occupancy of the final SLAT, like render/ss_latent.pt encodes S
        from trellis import models
        enc = models.from_pretrained(f"{os.environ['TRELLIS_CHECKPOINT']}/ckpts/ss_enc_conv3d_16l8_fp16").eval().cuda()
        coords = torch.load(run_out / "features" / "slat_edited.pt", map_location="cuda")["coords"][:, 1:4].long()
        occ = torch.zeros((1, 1, GRID, GRID, GRID), device="cuda")
        occ[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        x_T = enc(occ, sample_posterior=False).float()
        del enc
    occ_S = (dec(x_S) > 0)[0, 0].cpu().numpy()
    occ_T = (dec(x_T) > 0)[0, 0].cpu().numpy()

    # Noise only inside the round's edit mask (latent cells, outward feather), as in the run.
    omega = torch.from_numpy(edit_region(item, run_work).astype(np.float32))[None, None].cuda()
    m = (F.interpolate(omega, size=x_S.shape[-3:], mode="trilinear", align_corners=False) > 0.5).float()
    m = torch.maximum(m, F.avg_pool3d(m, 5, 1, 2))

    steps, s_min, lam = int(params["steps"]), sampler.sigma_min, args.flowedit_lambda
    ts = np.linspace(1, 0, steps + 1)
    ts = params["rescale_t"] * ts / (1 + (params["rescale_t"] - 1) * ts)
    pairs = list(combinations(views, 2))
    lat = tuple(x_S.shape[-3:])
    D = {p: torch.zeros(lat, device="cuda") for p in pairs}
    guess_err = {v: torch.zeros(lat, device="cuda") for v in views}
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    n = 0
    for k in range(steps):
        t = float(ts[k])
        for _ in range(args.noise_draws):
            eps = torch.randn(x_S.shape, generator=gen, device="cuda")
            z_src = m * ((1 - t) * x_S + (s_min + (1 - s_min) * t) * eps) + (1 - m) * x_S
            z_tar = x_T + (z_src - x_S)
            dv = {}
            for v in views:
                v_tar = velocity(z_tar, t, conds[v][1])
                dv[v] = v_tar - lam * velocity(z_src, t, conds[v][0])
                x0, _ = sampler._v_to_xstart_eps(x_t=z_tar, t=t, v=v_tar)
                guess_err[v] += torch.linalg.norm(x0 - x_T, dim=1)[0]
            for (i, j) in pairs:
                D[(i, j)] += torch.linalg.norm(dv[i] - dv[j], dim=1)[0]
            n += 1

    # Visibility of the changed voxels in every view (ray casting against T), per latent cell.
    cell = GRID // lat[0]
    vox = np.argwhere(occ_T ^ occ_S)
    scene = cvs.raycasting_scene(mesh_internal(glb_T))
    lin = np.ravel_multi_index(tuple((vox // cell).T), lat)
    n_changed = np.bincount(lin, minlength=int(np.prod(lat))).reshape(lat)
    vis_frac = {}
    for v in views:
        vis = cvs.visibility(scene, cvs.ViewCamera(**cams[v]), (vox + 0.5) / GRID - 0.5, cvs.SupportParams()) > 0.5
        vis_frac[v] = np.bincount(lin, weights=vis.astype(float), minlength=int(np.prod(lat))).reshape(lat) / np.maximum(n_changed, 1)
    raw = out / "raw.npz"
    np.savez_compressed(
        raw, views=np.array(views), pairs=np.array([f"{i}-{j}" for i, j in pairs]),
        in_mask=(m[0, 0] >= 1.0).cpu().numpy(), n_changed=n_changed,
        **{f"vis_{v}": vis_frac[v] for v in views},
        **{f"guess_err_{v}": (guess_err[v] / n).cpu().numpy() for v in views},
        **{f"D_{i}-{j}": (D[(i, j)] / n).cpu().numpy() for i, j in pairs})
    return raw


def pair_groups(d, pair: str, min_changed: int = 4) -> dict:
    """Latent cells of one view pair: seen by both / by one view with the unseen view's guess right or wrong.

    A view sees a cell when >= 50% of the cell's changed voxels are visible in it; only cells inside
    the edit mask with >= `min_changed` changed voxels are used. Also returns which view sees a
    one-view cell ("only_i" / "only_j").
    """
    i, j = pair.split("-")
    changed = d["in_mask"] & (d["n_changed"] >= min_changed)
    vi, vj = d[f"vis_{i}"] >= 0.5, d[f"vis_{j}"] >= 0.5
    ratio = np.where(vi, d[f"guess_err_{j}"] / d[f"guess_err_{i}"], d[f"guess_err_{i}"] / d[f"guess_err_{j}"])
    return {"both": changed & vi & vj,
            "one_right": changed & (vi ^ vj) & (ratio <= 1.0),
            "one_wrong": changed & (vi ^ vj) & (ratio > 1.0),
            "only_i": changed & vi & ~vj, "only_j": changed & vj & ~vi}


def analyze(raw_path: Path, min_changed: int = 4) -> dict:
    d = np.load(raw_path)
    in_mask = d["in_mask"]
    vals = {g: [] for g in GROUPS}
    flags = {g: [] for g in GROUPS}
    for pair in d["pairs"]:
        Dm = d[f"D_{pair}"]
        tau = Dm[in_mask].mean() + Dm[in_mask].std()          # Eq. 12 (relative part)
        sel = pair_groups(d, str(pair), min_changed)
        for g in GROUPS:
            s = sel[g]
            vals[g].append(Dm[s])
            flags[g].append(Dm[s] > tau)
    ref = np.concatenate(vals["both"]).mean()
    n_one = sum(len(np.concatenate(vals[g])) for g in ("one_right", "one_wrong"))
    table = []
    for g in GROUPS:
        v = np.concatenate(vals[g])
        f = np.concatenate(flags[g])
        table.append({"group": g, "cells": int(len(v)),
                      "share_of_one_view_cells": (float(len(v) / max(n_one, 1)) if g != "both" else None),
                      "D_rel_to_both": float(v.mean() / ref) if len(v) else None,
                      "flagged_conflict_rate": float(f.mean()) if len(f) else None})
    return {"table": table, "min_changed_voxels_per_cell": min_changed,
            "note": "cells x view-pairs; D relative to mean D of cells seen by both views; "
                    "flagged = D above tau_C = mean + std over the edit mask (Eq. 12)"}


def visualize(item_dir: Path, run: str, min_changed: int = 4, size: int = 512) -> Path:
    """Per view pair, the cells of each group drawn on both views' renders of T.

    Fill: green = seen by both views, orange = seen only by the left view, blue = seen only by
    the right view; red outline = one-view cell whose unseen view guessed wrong. Cells are drawn
    far-to-near as their projected 4^3-voxel boxes, so cells hidden in a view still show (under
    nearer ones) where they project.
    """
    import cv2

    exp = item_dir.resolve() / run / "unseen_view_experiment"
    d = np.load(exp / "raw.npz")
    cams = json.loads(camera_json(item_dir.resolve(), run).read_text())
    lat = d["in_mask"].shape[0]
    step = 1.0 / lat
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.float64)
    colors = {"both": (60, 190, 90), "only_i": (255, 150, 30), "only_j": (60, 120, 230)}
    rows = []
    for pair in [str(p) for p in d["pairs"]]:
        i, j = pair.split("-")
        g = pair_groups(d, pair, min_changed)
        panels = []
        for v in (i, j):
            cam = cvs.ViewCamera(**cams[v])
            img = np.asarray(Image.open(exp / "render_T" / "renders_by_method" / "render" / f"{v}.png").convert("RGBA"))
            base = cvs._rgba_on_white(img).astype(np.float32)
            k = size / img.shape[1]
            base = cv2.resize(base, (size, size))
            overlay, edges = base.copy(), []
            todo = [(c, name) for name in ("both", "only_i", "only_j") for c in np.argwhere(g[name])]
            depth = [cam.project(((c + 0.5) * step - 0.5)[None])[2][0] for c, _ in todo]
            for idx in np.argsort(depth)[::-1]:                       # far first
                c, name = todo[idx]
                px, py, _ = cam.project((c + corners) * step - 0.5)
                hull = cv2.convexHull(np.stack([px * k, py * k], 1).astype(np.int32))
                cv2.fillConvexPoly(overlay, hull, colors[name])
                if name != "both" and g["one_wrong"][tuple(c)]:
                    edges.append(hull)
            out = (0.45 * overlay + 0.55 * base).astype(np.uint8)
            cv2.polylines(out, edges, True, (220, 30, 30), 1, cv2.LINE_AA)
            panels.append(cvs._label(out, f"pair {pair}: view {v}"))
        rows.append(np.concatenate(panels, 1))
    legend = np.full((40, rows[0].shape[1], 3), 255, np.uint8)
    for x, (text, col) in zip((10, 260, 560, 820), (("seen by both", colors["both"]), ("seen only by left view", colors["only_i"]),
                                                     ("seen only by right view", colors["only_j"]), ("unseen view guessed wrong", (220, 30, 30)))):
        cv2.rectangle(legend, (x, 12), (x + 18, 30), col, -1 if x != 820 else 2)
        cv2.putText(legend, text, (x + 24, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    path = exp / "visibility_groups.png"
    Image.fromarray(np.concatenate(rows + [legend], 0)).save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item_dir", type=Path, required=True)
    ap.add_argument("--run", default="_easy3e_out_round1_auto_support_feather", help="Round output dir (T).")
    ap.add_argument("--views", default="0,1,2")
    ap.add_argument("--noise_draws", type=int, default=2, help="Noise samples per step (shared by all views).")
    ap.add_argument("--flowedit_lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--analyze_only", action="store_true", help="Re-run the analysis on an existing raw.npz.")
    ap.add_argument("--visualize", action="store_true", help="Only draw the visibility groups on the views (needs raw.npz).")
    args = ap.parse_args()
    if args.visualize:
        print(visualize(args.item_dir, args.run))
        return
    raw = (args.item_dir.resolve() / args.run / "unseen_view_experiment" / "raw.npz") if args.analyze_only else compute(args)
    result = analyze(raw)
    result.update({"item": str(args.item_dir), "run": args.run})
    (raw.parent / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
