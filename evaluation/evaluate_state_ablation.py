#!/usr/bin/env python3
"""Three-state vs two-state diagnosis ablation (round 1, run_state_ablation.sh outputs).

Variants (Easy3E --mv_fusion_state_mode), all from the same seed, so the sparse-structure stage
and its diagnosis are shared and only the lock-back before the SLAT stage differs:
  three_state           conflict C / edited-stable S / quiescent U, U locked back to the source
  three_state_srcocc    as three_state, but only U voxels also occupied in the source are locked back
  no_lockback           two states C / stable, nothing locked back
  lockback_nonconflict  two states C / source, every non-conflict cell of the edit mask locked back

Per view (inputs 0,1 and the held-out view 2 when it exists) the final GLBs and the source are
rendered with the same Blender setup at the source cameras, and scored on 512x512 images:
  * full image vs the target (aligned to the source frame): LPIPS, DINO-I, PSNR
  * edit region E_v  = dilated 2D edit support (B_v > 0.3) of source vs target: PSNR / LPIPS vs target
  * keep region K_v  = object pixels outside E_v, where target = source: PSNR / LPIPS vs the source
    reconstruction (the source SLAT decoded and exported like the edits), so the TRELLIS decoder's own
    error cancels and only edit-induced drift is measured
In 3D, the final SLAT is compared with the source SLAT on the shared diagnosis partition of the
edit mask (C, S, U, and N = mask cells not occupied after the sparse-structure edit): voxels whose
occupancy or features differ from the source.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "mv_fusion"))
import compute_view_support as cvs  # noqa: E402
import mv_fusion_rounds as rounds  # noqa: E402
from evaluate_view_support_runs import Metrics, label, on_gray, render_run  # noqa: E402
from process_easy3e_rgba_target import write_easy3e_rgba_target  # noqa: E402

PY = os.environ.get("TRELLIS_PYTHON", sys.executable)
MODES = ("three_state", "three_state_srcocc", "no_lockback", "lockback_nonconflict")
SIZE = 512


SUFFIX = ""  # run-tag suffix of a seed replicate, e.g. "_s2" (set from --suffix)


def run_dirs(item: Path, mode: str) -> tuple[Path, Path]:
    tag = f"abl_{mode}{SUFFIX}"
    return item / f"_easy3e_out_round1_auto_support_{tag}", item / f"_easy3e_work_round1_auto_support_{tag}"


def cameras(item: Path, views: list[str], eval_dir: Path) -> dict:
    """Composition cases have known cameras; the others reuse the run's fitted cameras and fit the held-out view."""
    for known in ("view_cameras_compose.json", "view_cameras_blender.json"):
        if (item / known).exists():
            return json.loads((item / known).read_text())
    out, work = run_dirs(item, "three_state")
    known_path = out / "view_support" / "view_cameras.json"
    known = json.loads(known_path.read_text())
    if all(v in known for v in views):
        return known
    fit_dir = eval_dir / "cameras"
    if not (fit_dir / "view_cameras.json").exists():
        subprocess.run([PY, "-u", str(PROJECT_ROOT / "mv_fusion" / "compute_view_support.py"), "--render_dir", str(work / "render"),
                        "--views", ",".join(views), "--camera_json", str(known_path), "--fit_cameras_only",
                        "--source_images", ",".join(str(item / f"source_{v}.png") for v in views),
                        "--output_dir", str(fit_dir)], check=True)
    return json.loads((fit_dir / "view_cameras.json").read_text())


def aligned_target(item: Path, v: str, eval_dir: Path) -> Path:
    prebuilt = item / "_mask_build" / f"target_{v}_rgba_aligned.png"
    if prebuilt.exists():
        return prebuilt
    rgba, aligned = eval_dir / f"target_{v}_rgba.png", eval_dir / f"target_{v}_rgba_aligned.png"
    if not aligned.exists():
        write_easy3e_rgba_target(input_path=item / f"target_{v}.png", output_rgba_path=rgba,
                                 output_mask_path=eval_dir / f"target_{v}_alpha.png",
                                 output_mask_512_path=eval_dir / f"target_{v}_alpha_512.png")
        print(f"[abl] view {v}: target alignment {cvs.warp_target_to_source(item / f'source_{v}.png', rgba, aligned)}")
    return aligned


def source_reconstruction(work: Path) -> Path:
    """GLB of the source SLAT (render/slat_latent.pt) through the decoder and to_glb settings of the edits."""
    out = work / "render" / "source_recon_rgba.glb"
    if out.exists():
        return out
    import os
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    sys.path.insert(0, str(PROJECT_ROOT / "Easy3E"))
    from trellis.modules import sparse as sp
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipe = TrellisImageTo3DPipeline.from_pretrained(os.environ["TRELLIS_CHECKPOINT"])
    pipe.cuda()
    d = torch.load(work / "render" / "slat_latent.pt", map_location="cuda", weights_only=False)
    slat = sp.SparseTensor(feats=d["feats"].float().cuda(), coords=d["coords"].int().cuda())
    with torch.no_grad():
        dec = pipe.decode_slat(slat, ["mesh", "gaussian"])
    # to_glb bakes the texture by gradient descent, so it runs with grad enabled
    postprocessing_utils.to_glb(dec["gaussian"][0], dec["mesh"][0], simplify=0, texture_size=2048).export(str(out))
    del pipe
    torch.cuda.empty_cache()
    return out


def alpha_512(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
    return cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA) > 127


def masked_scores(metrics: Metrics, pred: Image.Image, ref: Image.Image, mask: np.ndarray) -> dict:
    """PSNR over the mask pixels and LPIPS on the mask's bounding box with everything else set to gray."""
    if mask.sum() < 16:
        return {"psnr": float("nan"), "lpips": float("nan"), "pixels": int(mask.sum())}
    p, r = np.asarray(pred).astype(np.float32), np.asarray(ref).astype(np.float32)
    mse = float(((p - r) ** 2)[mask].mean())
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - 16, 0), min(ys.max() + 17, SIZE)
    x0, x1 = max(xs.min() - 16, 0), min(xs.max() + 17, SIZE)
    m = mask[..., None]
    crop = lambda img: Image.fromarray(np.where(m, img, 127).astype(np.uint8)[y0:y1, x0:x1]).resize((256, 256), Image.BILINEAR)
    pt, rt = (torch.from_numpy(np.asarray(crop(x))).permute(2, 0, 1)[None].float().div(255).to(metrics.device) for x in (p, r))
    with torch.no_grad():
        lp = float(metrics.lpips(pt, rt).item())
    return {"psnr": float(10 * np.log10(255.0 ** 2 / max(mse, 1e-6))), "lpips": lp, "pixels": int(mask.sum())}


def voxel_stats(item: Path) -> dict:
    """Changed voxels (occupancy or SLAT features vs the source) per diagnosis region, for every variant."""
    _, work = run_dirs(item, "three_state")
    out3, _ = run_dirs(item, "three_state")
    states = rounds.load_diagnosis_states(out3 / "features" / "mv_fusion_voxel_diff" / "quiescent_voxels.glb")
    omega = rounds.round_edit_region(work / "3Dmasks" / "mask_all.glb", work / "render")
    C, U = states["conflict"] & omega, states["quiescent"] & omega
    S = states["active"] & omega & ~C & ~U
    N = omega & ~states["active"]
    regions = {"C": C, "S": S, "U": U, "N": N, "outside": ~omega}

    def dense(path: Path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        c = d["coords"][:, 1:].long().numpy()
        occ = np.zeros((64,) * 3, bool)
        feat = np.zeros((64,) * 3 + (d["feats"].shape[1],), np.float32)
        occ[tuple(c.T)] = True
        feat[tuple(c.T)] = d["feats"].float().numpy()
        return occ, feat

    occ_s, feat_s = dense(work / "render" / "slat_latent.pt")
    ss = {m: torch.load(run_dirs(item, m)[0] / "features" / "ss_latent_edited.pt", map_location="cpu", weights_only=False)
          for m in MODES}
    res = {"region_cells": {k: int(v.sum()) for k, v in regions.items()},
           "region_source_voxels": {k: int((v & occ_s).sum()) for k, v in regions.items()},
           "ss_latent_max_abs_diff_vs_three_state": {m: float((ss[m] - ss["three_state"]).abs().max()) for m in MODES},
           "changed_voxels": {}, "occupied_voxels": {}}
    for m in MODES:
        occ, feat = dense(run_dirs(item, m)[0] / "features" / "slat_edited.pt")
        changed = (occ ^ occ_s) | (occ & occ_s & (np.linalg.norm(feat - feat_s, axis=-1) > 1e-3))
        res["changed_voxels"][m] = {k: int((changed & v).sum()) for k, v in regions.items()}
        res["occupied_voxels"][m] = {k: int((occ & v).sum()) for k, v in regions.items()}
        lockback = json.loads((run_dirs(item, m)[0] / "features" / "mv_fusion_voxel_diff" / "quiescent_stats.json").read_text())
        res.setdefault("lockback_cells", {})[m] = lockback.get("num_lockback_cells")
    return res


def evaluate(item: Path, metrics: Metrics, samples: int) -> dict:
    eval_dir = item / f"_state_ablation_eval{SUFFIX}"
    eval_dir.mkdir(exist_ok=True)
    views = [v for v in ("0", "1", "2") if (item / f"source_{v}.png").exists() and (item / f"target_{v}.png").exists()]
    cams = cameras(item, views, eval_dir)
    rargs = SimpleNamespace(resolution=1024, samples=samples, shrink=0, force_render=False)
    _, work3 = run_dirs(item, "three_state")
    glbs = {"source": work3 / "render" / "normalized.glb", "source_recon": source_reconstruction(work3)}
    for m in MODES:
        glbs[m] = sorted((run_dirs(item, m)[0] / "out").glob("*_rgba.glb"))[-1]
    renders = {k: render_run(g, eval_dir / f"render_{k}", views, cams, rargs) for k, g in glbs.items()}

    rows, sheet = [], []
    for vi, v in enumerate(views):
        tgt_path = aligned_target(item, v, eval_dir)
        target = on_gray(tgt_path)
        src = on_gray(renders["source"][vi])
        recon = on_gray(renders["source_recon"][vi])
        B, _ = cvs.edit_support_2d(item / f"source_{v}.png", tgt_path, aligned=True)
        B = cv2.resize(B.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        E = cv2.dilate((B > 0.3).astype(np.uint8), cvs._disk(round(0.02 * SIZE))) > 0
        fg = alpha_512(renders["source"][vi]) | alpha_512(tgt_path)
        for m in MODES + ("source_recon",):
            fg |= alpha_512(renders[m][vi])
        K = (cv2.dilate(fg.astype(np.uint8), cvs._disk(4)) > 0) & ~E
        tiles = [label(src.resize((384, 384)), f"source {v}"), label(target.resize((384, 384)), f"target {v}")]
        for m in MODES + ("source_recon",):
            pred = on_gray(renders[m][vi])
            full = metrics(pred, target)
            edit = masked_scores(metrics, pred, target, E)
            keep = masked_scores(metrics, pred, recon, K)
            keep_src = masked_scores(metrics, pred, src, K)
            rows.append({"view": v, "held_out": v not in ("0", "1"), "mode": m,
                         "lpips": full["lpips"], "dino_i": full["dino_i"], "psnr": full["psnr"],
                         "edit_psnr": edit["psnr"], "edit_lpips": edit["lpips"], "edit_pixels": edit["pixels"],
                         "keep_psnr": keep["psnr"], "keep_lpips": keep["lpips"], "keep_pixels": keep["pixels"],
                         "keep_psnr_vs_asset": keep_src["psnr"], "keep_lpips_vs_asset": keep_src["lpips"]})
            if m == "source_recon":
                continue
            tiles.append(label(pred.resize((384, 384)), f"{m} | LPIPS {full['lpips']:.3f}"))
        overlay = np.asarray(target).copy()
        overlay[E] = (0.5 * overlay[E] + 0.5 * np.array([230, 60, 40])).astype(np.uint8)
        overlay[K] = (0.6 * overlay[K] + 0.4 * np.array([40, 120, 230])).astype(np.uint8)
        tiles.append(label(Image.fromarray(overlay).resize((384, 384)), "red: edit E, blue: keep K"))
        sheet.append(np.concatenate([np.asarray(t) for t in tiles], 1))
    Image.fromarray(np.concatenate(sheet, 0)).save(eval_dir / "comparison_sheet.png")

    summary = {}
    for m in MODES + ("source_recon",):
        for split, keep in (("inputs", False), ("held_out", True)):
            rr = [r for r in rows if r["mode"] == m and r["held_out"] == keep]
            if rr:
                summary.setdefault(m, {})[split] = {k: float(np.nanmean([r[k] for r in rr]))
                                                    for k in ("lpips", "dino_i", "psnr", "edit_psnr", "edit_lpips",
                                                              "keep_psnr", "keep_lpips", "keep_psnr_vs_asset",
                                                              "keep_lpips_vs_asset")}
    result = {"item": str(item), "views": views, "rows": rows, "mean": summary, "voxels": voxel_stats(item)}
    (eval_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="Comma-separated item dirs.")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--suffix", default="", help="Seed-replicate suffix of the run tags (e.g. _s2).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = args.suffix
    metrics = Metrics(args.device)
    for item in args.items.split(","):
        r = evaluate(Path(item).resolve(), metrics, args.samples)
        print(json.dumps({"item": r["item"], "mean": r["mean"], "voxels": r["voxels"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
