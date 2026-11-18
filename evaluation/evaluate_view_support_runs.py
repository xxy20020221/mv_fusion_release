#!/usr/bin/env python3
"""Compare mv_fusion runs (e.g. manual trust boxes vs. auto view support) on one item.

Every run's final GLB is rendered with the same Blender setup at the source
cameras (from <run>/view_support/view_cameras.json or --cameras), then scored
against target_<v>.png with the preprocessing of the per-case metric scripts
(alpha over gray 127, 512x512): PSNR, SSIM, LPIPS(alex), DINO-I (dinov2-base
mean token), CLIP-I (clip-vit-base-patch32) and, with --prompt (or <item_dir>/prompt.txt),
CLIP-T (render vs edit text). A side-by-side sheet is written too.

    python evaluation/evaluate_view_support_runs.py --item_dir demo/rider \
        --runs _easy3e_out_round1,_easy3e_out_round1_auto_support \
        --camera_json demo/rider/_easy3e_out_round1_auto_support/view_support/view_cameras.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mv_fusion"))
from paths import BLENDER_BIN as BLENDER  # noqa: E402
from paths import RENDER_GLB_MV_HIGH_DETAIL_SCRIPT as RENDER_SCRIPT  # noqa: E402


def camera_look_at(cams: dict) -> np.ndarray:
    """Common look-at point of orthographic cameras with principal-point offsets (least squares)."""
    import compute_view_support as cvs

    rows, rhs = [], []
    for c in cams.values():
        _, right, up, _ = cvs._basis(c["azimuth"], c["elevation"])
        rows += [right, up]
        rhs += [-(c["tx"] - c["width"] / 2.0) / c["scale"], (c["ty"] - c["height"] / 2.0) / c["scale"]]
    look_at = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0]
    return look_at if np.linalg.norm(look_at) > 1e-6 else np.zeros(3)


def render_run(glb: Path, out_dir: Path, view_ids, cams, args) -> list[Path]:
    ortho = {round(c["width"] / c["scale"], 3) for c in cams.values()}
    if len(ortho) != 1 or any(c.get("distance") is not None for c in cams.values()):
        raise ValueError(f"Expected one shared orthographic scale for all views, got {ortho}")
    ortho_scale = ortho.pop()
    image_dir = out_dir / "image_ori"
    expected = [image_dir / f"{i:03d}.png" for i in range(len(view_ids))]
    if not args.force_render and all(p.exists() for p in expected):
        return expected
    out_dir.mkdir(parents=True, exist_ok=True)
    look_at = camera_look_at({v: cams[v] for v in view_ids})
    if look_at.any():  # the render script's cameras look at the origin
        if args.shrink != 0:
            raise ValueError("Cameras with a principal-point offset need --shrink 0")
        import mv_fusion_rounds as rounds
        glb = rounds.recentered_glb(glb, look_at, out_dir / "recentered.glb")
    cmd = [
        str(BLENDER), "-b", "-P", str(RENDER_SCRIPT), "--",
        "--object_path", str(glb), "--output_dir", str(out_dir),
        "--resolution", str(args.resolution), "--shrink", str(args.shrink), "--distance", "2.0",
        "--ortho_scale", str(ortho_scale), "--engine", "CYCLES", "--cycles_device", "CPU",
        "--samples", str(args.samples),
        "--view_azimuths", ",".join(str(cams[v]["azimuth"]) for v in view_ids),
        "--view_elevations", ",".join(str(cams[v]["elevation"]) for v in view_ids),
        "--view_ids", ",".join(str(i) for i in range(len(view_ids))),
    ]
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    with (out_dir / "render_blender.log").open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return expected


def on_gray(path: Path, size: int = 512) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (127, 127, 127, 255))
        bg.alpha_composite(img)
        img = bg.convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


class Metrics:
    def __init__(self, device: str):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.device = device
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        # Hub ids or local directories (CLIP_MODEL / DINO_MODEL), e.g. for offline machines.
        dino_id = os.environ.get("DINO_MODEL", "facebook/dinov2-base")
        clip_id = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_id)
        self.dino = AutoModel.from_pretrained(dino_id).to(device).eval()
        try:
            self.clip_proc = CLIPProcessor.from_pretrained(clip_id)
            self.clip = CLIPModel.from_pretrained(clip_id).to(device).eval()
        except OSError as exc:  # weights not cached and no network
            print(f"[eval][WARN] CLIP-I disabled: {exc}")
            self.clip = None

    @torch.no_grad()
    def clip_t(self, pred: Image.Image, prompt: str | None) -> float:
        if self.clip is None or not prompt:
            return float("nan")
        text = self.clip.text_model(**self.clip_proc(text=[prompt], return_tensors="pt", padding=True,
                                                     truncation=True).to(self.device))
        t = self.clip.text_projection(text.pooler_output)
        img = self.clip.vision_model(**self.clip_proc(images=[pred], return_tensors="pt").to(self.device))
        f = self.clip.visual_projection(img.pooler_output)
        return float(((f / f.norm(dim=-1, keepdim=True)) * (t / t.norm(dim=-1, keepdim=True))).sum().item())

    @torch.no_grad()
    def __call__(self, pred: Image.Image, target: Image.Image) -> dict:
        p, t = np.asarray(pred), np.asarray(target)
        pt = torch.from_numpy(p).permute(2, 0, 1)[None].float().div(255).to(self.device)
        tt = torch.from_numpy(t).permute(2, 0, 1)[None].float().div(255).to(self.device)

        def dino_feat(img):
            out = self.dino(**self.dino_proc(images=[img], return_tensors="pt").to(self.device))
            f = out.last_hidden_state.mean(dim=1)
            return f / f.norm(dim=-1, keepdim=True)

        def clip_feat(img):
            out = self.clip.vision_model(**self.clip_proc(images=[img], return_tensors="pt").to(self.device))
            f = self.clip.visual_projection(out.pooler_output)
            return f / f.norm(dim=-1, keepdim=True)

        return {
            "psnr": float(peak_signal_noise_ratio(t, p, data_range=255)),
            "ssim": float(structural_similarity(t, p, channel_axis=2, data_range=255)),
            "lpips": float(self.lpips(pt, tt).item()),
            "dino_i": float((dino_feat(pred) * dino_feat(target)).sum().item()),
            "clip_i": float((clip_feat(pred) * clip_feat(target)).sum().item()) if self.clip is not None else float("nan"),
        }


def label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 * len(text) + 10, 22], fill=(0, 0, 0))
    d.text((5, 5), text, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item_dir", type=Path, required=True)
    ap.add_argument("--runs", required=True, help="Comma-separated run dir names under item_dir.")
    ap.add_argument("--views", default=None, help="Comma-separated view ids (default: all in the camera json).")
    ap.add_argument("--camera_json", type=Path, required=True, help="view_cameras.json from compute_view_support.py")
    ap.add_argument("--target_rgba_dir", type=Path, default=None,
                    help="Dir with target_<v>_rgba.png (background-removed targets staged by the driver). "
                         "Default: <item_dir>/_easy3e_work_round1/images; raw target_<v>.png is used if missing.")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--prompt", default=None, help="Edit text for CLIP-T (default: <item_dir>/prompt.txt if present).")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--shrink", type=float, default=1.0,
                    help="1 = renormalize each GLB to its own bbox (old protocol); 0 = render in the shared "
                         "normalized frame, needed when comparing rounds whose bboxes differ.")
    ap.add_argument("--render_tag", default="render_compare",
                    help="Name of the per-run render folder (use a new tag when changing views or --shrink).")
    ap.add_argument("--force_render", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    item = args.item_dir.resolve()
    cams = json.loads(args.camera_json.read_text())
    view_ids = [v.strip() for v in args.views.split(",")] if args.views else list(cams)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    out_dir = args.output_dir or item / "view_support_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    renders = {}
    for run in runs:
        glbs = sorted((item / run / "out").glob("*_rgba.glb"))
        if not glbs:
            raise FileNotFoundError(f"No *_rgba.glb in {item / run / 'out'}")
        print(f"[eval] rendering {run}: {glbs[-1].name}", flush=True)
        renders[run] = render_run(glbs[-1], item / run / args.render_tag, view_ids, cams, args)

    metrics = Metrics(args.device)
    prompt = args.prompt or ((item / "prompt.txt").read_text().strip() if (item / "prompt.txt").exists() else None)
    rgba_dir = args.target_rgba_dir or item / "_easy3e_work_round1" / "images"
    rows = []
    sheet_rows = []
    for vi, v in enumerate(view_ids):
        target_path = rgba_dir / f"target_{v}_rgba.png"
        if not target_path.exists():
            target_path = item / f"target_{v}.png"
        target = on_gray(target_path)
        source_path = item / f"source_{v}.png" if (item / f"source_{v}.png").exists() else rgba_dir / f"source_{v}.png"
        tiles = [label(on_gray(source_path, 384), f"source {v}"), label(target.resize((384, 384)), f"target {v}")]
        for run in runs:
            pred = on_gray(renders[run][vi])
            m = metrics(pred, target)
            m["clip_t"] = metrics.clip_t(pred, prompt)
            rows.append({"run": run, "view": v, "target": str(target_path), **m})
            tiles.append(label(pred.resize((384, 384)), f"{run} | LPIPS {m['lpips']:.3f} DINO {m['dino_i']:.3f}"))
        sheet = Image.new("RGB", (384 * len(tiles), 384), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, (384 * i, 0))
        sheet_rows.append(sheet)

    summary = {}
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        summary[run] = {k: float(np.mean([r[k] for r in rr])) for k in ("psnr", "ssim", "lpips", "dino_i", "clip_i", "clip_t")}
        summary[run]["worst_lpips"] = float(max(r["lpips"] for r in rr))
        summary[run]["worst_dino_i"] = float(min(r["dino_i"] for r in rr))

    full = Image.new("RGB", (sheet_rows[0].width, 384 * len(sheet_rows)), "white")
    for i, s in enumerate(sheet_rows):
        full.paste(s, (0, 384 * i))
    full.save(out_dir / "comparison_sheet.png")
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "metrics.json").write_text(json.dumps({"views": view_ids, "cameras": {v: cams[v] for v in view_ids},
                                                      "rows": rows, "mean": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
