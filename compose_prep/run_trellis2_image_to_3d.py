#!/usr/bin/env python3
"""Unified TRELLIS.2 image-to-3D runner.

This script replaces per-image temporary runners such as
``tmp_run_trellis2_*_highest.py``. It keeps the same highest-quality defaults
while exposing the image path, output location, quality preset, and detailed
sampling/export parameters as command-line arguments.
"""

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Clone of https://github.com/microsoft/TRELLIS.2 (only needed for the composition layout guides).
REPO_ROOT = Path(os.environ.get("TRELLIS2_REPO", PROJECT_ROOT / "third_party" / "TRELLIS.2"))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_NAME = "microsoft/TRELLIS.2-4B"

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(REPO_ROOT.resolve()))

QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "pipeline_type": "512",
        "max_num_tokens": 24576,
        "decimation_target": 300_000,
        "texture_size": 1024,
        "nvdiffrast_face_limit": 16_777_216,
        "sampler_params": {
            "sparse_structure": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            "shape_slat": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            "tex_slat": {
                "steps": 12,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
        },
    },
    "standard": {
        "pipeline_type": "1024_cascade",
        "max_num_tokens": 49152,
        "decimation_target": 500_000,
        "texture_size": 2048,
        "nvdiffrast_face_limit": 16_777_216,
        "sampler_params": {
            "sparse_structure": {
                "steps": 20,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            "shape_slat": {
                "steps": 20,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            "tex_slat": {
                "steps": 20,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
        },
    },
    "high": {
        "pipeline_type": "1024_cascade",
        "max_num_tokens": 49152,
        "decimation_target": 750_000,
        "texture_size": 4096,
        "nvdiffrast_face_limit": 16_777_216,
        "sampler_params": {
            "sparse_structure": {
                "steps": 35,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            "shape_slat": {
                "steps": 35,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            "tex_slat": {
                "steps": 35,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
        },
    },
    "highest": {
        "pipeline_type": "1536_cascade",
        "max_num_tokens": 49152,
        "decimation_target": 1_000_000,
        "texture_size": 4096,
        "nvdiffrast_face_limit": 16_777_216,
        "sampler_params": {
            "sparse_structure": {
                "steps": 50,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            "shape_slat": {
                "steps": 50,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            "tex_slat": {
                "steps": 50,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRELLIS.2 image-to-3D generation from any local image."
    )
    parser.add_argument(
        "--image-path",
        required=True,
        type=Path,
        help="Input image path, for example path/to/image.png.",
    )
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_PRESETS),
        default="highest",
        help="Quality preset. 'highest' matches the old tmp_run_trellis2_*_highest.py scripts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Exact output directory. If omitted, use <out-root>/trellis2_<image_stem>_<suffix>.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output folder name under --out-root. If omitted, use trellis2_<image_stem>_<suffix>.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Suffix for default output folder and output files. Defaults to --quality.",
    )
    parser.add_argument(
        "--pipeline-type",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        default=None,
    )
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--decimation-target", type=int, default=None)
    parser.add_argument("--texture-size", type=int, default=None)
    parser.add_argument("--nvdiffrast-face-limit", type=int, default=None)
    parser.add_argument("--ss-steps", type=int, default=None)
    parser.add_argument("--shape-steps", type=int, default=None)
    parser.add_argument("--tex-steps", type=int, default=None)
    parser.add_argument("--ss-guidance-strength", type=float, default=None)
    parser.add_argument("--shape-guidance-strength", type=float, default=None)
    parser.add_argument("--tex-guidance-strength", type=float, default=None)
    parser.add_argument("--ss-guidance-rescale", type=float, default=None)
    parser.add_argument("--shape-guidance-rescale", type=float, default=None)
    parser.add_argument("--tex-guidance-rescale", type=float, default=None)
    parser.add_argument("--ss-rescale-t", type=float, default=None)
    parser.add_argument("--shape-rescale-t", type=float, default=None)
    parser.add_argument("--tex-rescale-t", type=float, default=None)
    parser.add_argument(
        "--hdri",
        default="forest",
        help="HDRI name under TRELLIS.2/assets/hdri, or an explicit .exr path.",
    )
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip rendering the MP4 preview and only export GLB plus summary.",
    )
    return parser.parse_args()


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "image"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_hdri_path(value: str) -> Path:
    path = Path(value)
    if path.suffix:
        return resolve_path(path)
    return REPO_ROOT / "assets" / "hdri" / f"{value}.exr"


def build_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings = deepcopy(QUALITY_PRESETS[args.quality])
    sampler_params = settings["sampler_params"]

    override_map = {
        "pipeline_type": args.pipeline_type,
        "max_num_tokens": args.max_num_tokens,
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
        "nvdiffrast_face_limit": args.nvdiffrast_face_limit,
    }
    for key, value in override_map.items():
        if value is not None:
            settings[key] = value

    sampler_overrides = {
        ("sparse_structure", "steps"): args.ss_steps,
        ("shape_slat", "steps"): args.shape_steps,
        ("tex_slat", "steps"): args.tex_steps,
        ("sparse_structure", "guidance_strength"): args.ss_guidance_strength,
        ("shape_slat", "guidance_strength"): args.shape_guidance_strength,
        ("tex_slat", "guidance_strength"): args.tex_guidance_strength,
        ("sparse_structure", "guidance_rescale"): args.ss_guidance_rescale,
        ("shape_slat", "guidance_rescale"): args.shape_guidance_rescale,
        ("tex_slat", "guidance_rescale"): args.tex_guidance_rescale,
        ("sparse_structure", "rescale_t"): args.ss_rescale_t,
        ("shape_slat", "rescale_t"): args.shape_rescale_t,
        ("tex_slat", "rescale_t"): args.tex_rescale_t,
    }
    for (stage, key), value in sampler_overrides.items():
        if value is not None:
            sampler_params[stage][key] = value

    return settings


def import_runtime_modules() -> Dict[str, Any]:
    import cv2
    import imageio
    import numpy as np
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    import o_voxel

    try:
        import OpenEXR
    except Exception:
        OpenEXR = None

    return {
        "cv2": cv2,
        "imageio": imageio,
        "np": np,
        "torch": torch,
        "Image": Image,
        "Trellis2ImageTo3DPipeline": Trellis2ImageTo3DPipeline,
        "EnvMap": EnvMap,
        "render_utils": render_utils,
        "o_voxel": o_voxel,
        "OpenEXR": OpenEXR,
    }


def load_exr_rgb(path: Path, runtime: Dict[str, Any]) -> Any:
    cv2 = runtime["cv2"]
    np = runtime["np"]
    OpenEXR = runtime["OpenEXR"]

    if OpenEXR is not None:
        try:
            exr = OpenEXR.File(str(path))
            part = exr.parts[0]
            channels = part.channels
            if "RGB" in channels:
                return np.asarray(channels["RGB"].pixels, dtype=np.float32)
            if {"R", "G", "B"}.issubset(channels.keys()):
                return np.stack(
                    [
                        np.asarray(channels["R"].pixels, dtype=np.float32),
                        np.asarray(channels["G"].pixels, dtype=np.float32),
                        np.asarray(channels["B"].pixels, dtype=np.float32),
                    ],
                    axis=-1,
                )
        except Exception as exc:
            print(f"[runner] OpenEXR loader failed for {path}: {exc}; falling back to cv2", flush=True)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to load EXR file: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)


def tensor_size_mb(tensor: Any) -> float:
    return tensor.numel() * tensor.element_size() / 1024 / 1024


def main() -> None:
    args = parse_args()
    image_path = resolve_path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    settings = build_settings(args)
    image_stem = slugify_name(image_path.stem)
    suffix = slugify_name(args.suffix or args.quality)
    output_name = slugify_name(args.output_name or f"trellis2_{image_stem}_{suffix}")
    out_dir = resolve_path(args.out_dir) if args.out_dir else resolve_path(args.out_root) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_path = out_dir / f"{image_stem}_preprocessed.png"
    glb_path = out_dir / f"{image_stem}_trellis2_{suffix}.glb"
    mp4_path: Optional[Path] = None if args.skip_preview else out_dir / f"{image_stem}_trellis2_{suffix}.mp4"
    summary_path = out_dir / "summary.json"
    hdri_path = resolve_hdri_path(args.hdri)

    runtime = import_runtime_modules()
    torch = runtime["torch"]
    Image = runtime["Image"]
    Trellis2ImageTo3DPipeline = runtime["Trellis2ImageTo3DPipeline"]
    EnvMap = runtime["EnvMap"]
    render_utils = runtime["render_utils"]
    imageio = runtime["imageio"]
    o_voxel = runtime["o_voxel"]

    print(f"[runner] Input image: {image_path}", flush=True)
    print(f"[runner] Output dir: {out_dir}", flush=True)
    print(f"[runner] Quality preset: {args.quality}", flush=True)
    print(f"[runner] Settings: {json.dumps(settings, indent=2)}", flush=True)
    print(f"[runner] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    envmap = None
    if not args.skip_preview:
        print(f"[runner] Loading envmap: {hdri_path}", flush=True)
        envmap_np = load_exr_rgb(hdri_path, runtime)
        envmap = EnvMap(torch.tensor(envmap_np, dtype=torch.float32, device="cuda"))

    print(f"[runner] Loading TRELLIS.2 pipeline: {args.model_name}", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_name)
    pipeline.cuda()
    print(f"[runner] Pipeline default type: {pipeline.default_pipeline_type}", flush=True)

    raw_image = Image.open(image_path)
    preprocessed = pipeline.preprocess_image(raw_image)
    preprocessed.save(preprocessed_path)
    print(f"[runner] Saved preprocessed image: {preprocessed_path}", flush=True)

    sampler_params = settings["sampler_params"]
    print(
        f"[runner] Running pipeline_type={settings['pipeline_type']}, seed={args.seed}, "
        f"max_num_tokens={settings['max_num_tokens']}, "
        f"sampler_steps={sampler_params['sparse_structure']['steps']}/"
        f"{sampler_params['shape_slat']['steps']}/"
        f"{sampler_params['tex_slat']['steps']}",
        flush=True,
    )
    meshes, latents = pipeline.run(
        raw_image,
        seed=args.seed,
        sparse_structure_sampler_params=sampler_params["sparse_structure"],
        shape_slat_sampler_params=sampler_params["shape_slat"],
        tex_slat_sampler_params=sampler_params["tex_slat"],
        preprocess_image=True,
        return_latent=True,
        pipeline_type=settings["pipeline_type"],
        max_num_tokens=settings["max_num_tokens"],
    )
    mesh = meshes[0]
    shape_slat, tex_slat, actual_resolution = latents
    print(f"[runner] Actual decoded resolution: {actual_resolution}", flush=True)
    print(f"[runner] Mesh before simplify: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}", flush=True)
    mesh.simplify(settings["nvdiffrast_face_limit"])
    print(f"[runner] Mesh after simplify: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}", flush=True)

    if mp4_path is not None:
        print(f"[runner] Rendering preview video: {mp4_path}", flush=True)
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave(mp4_path, video, fps=args.preview_fps)

    print(f"[runner] Exporting GLB: {glb_path}", flush=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=actual_resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=settings["decimation_target"],
        texture_size=settings["texture_size"],
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    glb.export(str(glb_path), extension_webp=True)

    summary = {
        "input_image": str(image_path),
        "output_dir": str(out_dir),
        "preprocessed_image": str(preprocessed_path),
        "model": args.model_name,
        "seed": args.seed,
        "quality": args.quality,
        "requested_pipeline_type": settings["pipeline_type"],
        "actual_resolution": int(actual_resolution),
        "max_num_tokens": settings["max_num_tokens"],
        "sampler_params": sampler_params,
        "decimation_target": settings["decimation_target"],
        "texture_size": settings["texture_size"],
        "nvdiffrast_face_limit": settings["nvdiffrast_face_limit"],
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_voxel_size": float(mesh.voxel_size),
        "shape_slat_tokens": int(shape_slat.coords.shape[0]),
        "tex_slat_tokens": int(tex_slat.coords.shape[0]),
        "shape_slat_mb": tensor_size_mb(shape_slat.feats),
        "tex_slat_mb": tensor_size_mb(tex_slat.feats),
        "glb": str(glb_path),
        "preview_video": str(mp4_path) if mp4_path is not None else None,
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    torch.cuda.empty_cache()
    print("[runner] Done", flush=True)


if __name__ == "__main__":
    main()
