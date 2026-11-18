#!/usr/bin/env python3
"""Export an Easy3E final target GLB from a saved edited SLAT latent.

This is intended for mv_fusion post-processing: when Easy3E's final
`target_*_rgba.glb` export failed or is missing, we can reuse the already saved
`features/slat_edited.pt` and decode it to mesh/gaussian, then run the standard
Easy3E postprocessing exporter. This does not rerun sparse-structure or SLAT
sampling.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EASY3E_ROOT = PROJECT_ROOT / "Easy3E"
if str(EASY3E_ROOT) not in sys.path:
    sys.path.insert(0, str(EASY3E_ROOT))

import torch

import trellis.modules.sparse as sp
from pipeline_edit.utils import load_pipeline
from trellis.utils import postprocessing_utils


def _load_slat(path: Path, device: str) -> sp.SparseTensor:
    cache = torch.load(str(path), map_location=device)
    if not isinstance(cache, dict) or "feats" not in cache or "coords" not in cache:
        raise ValueError(f"Unexpected SLAT cache format: {path}")
    return sp.SparseTensor(
        feats=cache["feats"].to(device),
        coords=cache["coords"].to(device),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode Easy3E slat_edited.pt to final target GLB.")
    ap.add_argument("--slat", type=Path, required=True)
    ap.add_argument("--output-glb", type=Path, required=True)
    ap.add_argument("--checkpoint", default="microsoft/TRELLIS-image-large")
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    slat_path = args.slat.expanduser().resolve()
    output_glb = args.output_glb.expanduser().resolve()
    if not slat_path.exists() or slat_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing non-empty SLAT file: {slat_path}")
    if output_glb.exists() and output_glb.stat().st_size > 0 and not args.force:
        print(f"[export] Reuse existing final target GLB: {output_glb}", flush=True)
        return

    device = args.device
    if device != "cuda":
        raise ValueError("Easy3E load_pipeline() currently requires CUDA; please use --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to decode Easy3E slat_edited.pt into final target GLB.")

    class _Args:
        pass

    pipeline_args = _Args()
    pipeline_args.checkpoint = args.checkpoint
    pipeline_args.no_gpu = device != "cuda"

    print(f"[export] Loading Easy3E pipeline: {args.checkpoint}", flush=True)
    pipeline = load_pipeline(pipeline_args)
    if device == "cuda":
        pipeline.cuda()

    print(f"[export] Loading SLAT: {slat_path}", flush=True)
    slat = _load_slat(slat_path, device)
    print(
        f"[export] Decoding SLAT feats={tuple(slat.feats.shape)} coords={tuple(slat.coords.shape)}",
        flush=True,
    )
    outputs = pipeline.decode_slat(slat, formats=["mesh", "gaussian"])
    mesh = outputs["mesh"][0]
    gaussian = outputs["gaussian"][0]
    for attr in ("vertices", "faces", "vertex_attrs", "face_normal", "vertex_normal"):
        value = getattr(mesh, attr, None)
        if torch.is_tensor(value):
            setattr(mesh, attr, value.detach())
    for attr, value in list(vars(gaussian).items()):
        if torch.is_tensor(value):
            setattr(gaussian, attr, value.detach())

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] Exporting final target GLB: {output_glb}", flush=True)
    glb = postprocessing_utils.to_glb(
        gaussian,
        mesh,
        simplify=0,
        texture_size=int(args.texture_size),
    )
    glb.export(str(output_glb))
    print(f"[export] Done: {output_glb} ({output_glb.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
