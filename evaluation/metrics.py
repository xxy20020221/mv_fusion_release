#!/usr/bin/env python3
"""Evaluate mv_fusion outputs with CLIP-T, DINO-I, LPIPS, and FID.

The script is intentionally self-contained and works in two modes:

1. GLB mode: render each mv_fusion output GLB to a small set of aligned views,
   then evaluate the rendered images against target_<view>.png/RGBA references.
2. Image mode: pass --pred_dir to evaluate an existing directory of prediction
   images directly. This is useful when Blender/system libraries are missing.

Metric definitions follow the common Easy3E-style usage:
- CLIP-T: CLIP image-text cosine similarity between generated renders and an
  edit text prompt. Higher is better.
- DINO-I: DINOv2 image-image cosine similarity between generated renders and
  target edited reference images. Higher is better.
- LPIPS: Learned perceptual distance between generated renders and target
  edited reference images. Lower is better.
- FID: Fréchet distance between generated render distribution and target image
  distribution. Lower is better.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import models, transforms

try:
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel
except Exception as exc:  # pragma: no cover - reported at runtime
    AutoImageProcessor = None
    AutoModel = None
    AutoProcessor = None
    CLIPModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mv_fusion"))
from paths import BLENDER_BIN as DEFAULT_BLENDER  # noqa: E402
from paths import RENDER_GLB_MV_SCRIPT as DEFAULT_RENDER_SCRIPT  # noqa: E402

PROJECT_ROOT = REPO_ROOT
# Optional separate Python for the metric models (--prefer_test_env); default: not used.
DEFAULT_TEST_ENV_PYTHON = Path(os.environ.get("METRICS_PYTHON", "/nonexistent"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class ImagePair:
    view_id: str
    pred: Path
    target: Path


@dataclass
class MetricResult:
    name: str
    clip_t: float | None
    dino_i: float | None
    lpips: float | None
    fid: float | None
    n_pairs: int
    pred_dir: Path
    target_dir: Path
    glb_path: Path | None
    error: str | None = None


def _parse_csv_floats(value: str | None, default: Sequence[float]) -> list[float]:
    if value is None or not value.strip():
        return [float(x) for x in default]
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str | None, default: Sequence[str]) -> list[str]:
    if value is None or not value.strip():
        return [str(x) for x in default]
    return [x.strip() for x in value.split(",") if x.strip()]


def _natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _rgb_on_background(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """Convert any image to RGB, compositing alpha on a white background."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (*background, 255))
        out = Image.alpha_composite(bg, rgba)
        return out.convert("RGB")
    return image.convert("RGB")


def load_rgb(path: Path, size: int | None = None) -> Image.Image:
    img = _rgb_on_background(Image.open(path))
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def image_to_uint8_tensor(paths: Sequence[Path], size: int) -> torch.Tensor:
    arrs = []
    for p in paths:
        img = load_rgb(p, size=size)
        arr = np.asarray(img, dtype=np.uint8)
        arrs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(arrs, dim=0)


def image_to_float_tensor(paths: Sequence[Path], size: int, normalize_lpips: bool = False) -> torch.Tensor:
    arrs = []
    for p in paths:
        img = load_rgb(p, size=size)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if normalize_lpips:
            arr = arr * 2.0 - 1.0
        arrs.append(arr)
    return torch.stack(arrs, dim=0)


def _safe_mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def _write_json(path: Path, data: dict) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_output_dirs(item_dir: Path, explicit: Sequence[str] | None = None) -> list[Path]:
    if explicit:
        return [Path(p).expanduser().resolve() for p in explicit]
    candidates = sorted(item_dir.glob("_easy3e_out_round*"), key=_natural_key)
    candidates += sorted(item_dir.glob("_easy3e_out"), key=_natural_key)
    uniq = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen and (p / "run_summary.json").exists():
            uniq.append(p)
            seen.add(rp)
    return uniq


def discover_glbs(output_dir: Path) -> list[Path]:
    out_dir = output_dir / "out"
    if out_dir.exists():
        glbs = sorted(out_dir.glob("*_rgba.glb"), key=_natural_key)
        if glbs:
            return glbs
        glbs = sorted(out_dir.glob("*.glb"), key=_natural_key)
        glbs = [g for g in glbs if "velocity_conflict" not in g.name and "quiescent" not in g.name]
        if glbs:
            return glbs
    feat = output_dir / "features" / "vox_edit.glb"
    return [feat] if feat.exists() else []


def load_run_summary(output_dir: Path) -> dict:
    path = output_dir / "run_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def target_dir_from_summary(output_dir: Path, item_dir: Path) -> Path:
    summary = load_run_summary(output_dir)
    primary = summary.get("primary_target")
    if primary:
        p = Path(primary)
        if p.exists():
            return p.parent
    round_id = summary.get("round")
    if round_id:
        p = item_dir / f"_easy3e_work_round{round_id}" / "images"
        if p.exists():
            return p
    for cand in [item_dir / "_easy3e_work_round2" / "images", item_dir / "_easy3e_work_round1" / "images", item_dir]:
        if cand.exists() and list(cand.glob("target_*.png")):
            return cand
    return item_dir


def list_target_images(target_dir: Path, view_ids: Sequence[str] | None = None) -> dict[str, Path]:
    targets: dict[str, Path] = {}
    patterns = ["target_*_rgba.png", "target_*.png"]
    for pat in patterns:
        for p in sorted(target_dir.glob(pat), key=_natural_key):
            stem = p.stem
            if stem.endswith("_alpha_mask") or stem.endswith("_alpha_mask_512"):
                continue
            m = re.match(r"target_(.+?)(?:_rgba)?$", stem)
            if not m:
                continue
            vid = m.group(1)
            if view_ids is not None and vid not in view_ids:
                continue
            targets.setdefault(vid, p)
    return targets


def list_prediction_images(pred_dir: Path, view_ids: Sequence[str] | None = None) -> dict[str, Path]:
    search_dirs = [pred_dir, pred_dir / "image_ori", pred_dir / "images", pred_dir / "renders"]
    preds: dict[str, Path] = {}
    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.iterdir(), key=_natural_key):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            if p.name.endswith("_alpha_mask.png") or p.name.endswith("_alpha_mask_512.png"):
                continue
            stem = p.stem
            vid: str | None = None
            m = re.match(r"(?:pred|render|source|target)_(.+?)(?:_rgba)?$", stem)
            if m:
                vid = m.group(1)
            elif stem.isdigit():
                vid = str(int(stem))
            if vid is None:
                continue
            if view_ids is not None and vid not in view_ids:
                continue
            preds.setdefault(vid, p)
    return preds


def pair_images(pred_dir: Path, target_dir: Path, view_ids: Sequence[str] | None = None) -> list[ImagePair]:
    targets = list_target_images(target_dir, view_ids=view_ids)
    preds = list_prediction_images(pred_dir, view_ids=view_ids)
    pairs = []
    for vid in sorted(set(targets) & set(preds), key=lambda x: int(x) if x.isdigit() else x):
        pairs.append(ImagePair(view_id=vid, pred=preds[vid], target=targets[vid]))
    return pairs


def make_blender_stub_libs(lib_dir: Path) -> Path | None:
    """Create a tiny libXxf86vm.so.1 shim if gcc is available.

    Some headless containers miss libXxf86vm but Blender only needs the symbols
    for optional display-mode queries. This shim is enough for background CPU
    rendering on those machines. Other missing X11/GL libs must still be
    provided by the system/conda environment.
    """
    gcc = shutil.which("gcc")
    if gcc is None:
        return None
    _ensure_dir(lib_dir)
    src = lib_dir / "libxxf86vm_stub.c"
    src.write_text(
        "\n".join(
            [
                "int XF86VidModeQueryVersion(void *dpy, int *major, int *minor) { if (major) *major = 2; if (minor) *minor = 2; return 1; }",
                "int XF86VidModeGetAllModeLines(void *dpy, int screen, int *modecount_return, void *modesinfo) { if (modecount_return) *modecount_return = 0; return 0; }",
                "int XF86VidModeSwitchToMode(void *dpy, int screen, void *modeline) { return 0; }",
                "int XF86VidModeSetViewPort(void *dpy, int screen, int x, int y) { return 0; }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = lib_dir / "libXxf86vm.so.1"
    cmd = [gcc, "-shared", "-fPIC", "-Wl,-soname,libXxf86vm.so.1", "-o", str(out), str(src)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        symlink = lib_dir / "libXxf86vm.so"
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(out.name)
        return lib_dir
    except Exception:
        return None


def blender_library_dirs(
    blender: Path,
    local_lib_dir: Path | None = None,
    runtime_prefix: Path | None = None,
) -> list[Path]:
    dirs: list[Path] = []
    if runtime_prefix is not None:
        for cand in [runtime_prefix / "lib", runtime_prefix / "x86_64-conda-linux-gnu" / "sysroot" / "usr" / "lib64"]:
            if cand.exists():
                dirs.append(cand)
    if local_lib_dir is not None:
        dirs.append(local_lib_dir)
    blender_dir = blender.parent
    for cand in [blender_dir / "lib", blender_dir / "lib" / "mesa"]:
        if cand.exists():
            dirs.append(cand)
    return dirs


def blender_missing_libraries(blender: Path, extra_ld: Sequence[Path] = ()) -> list[str]:
    env = os.environ.copy()
    ld_parts = [str(p) for p in extra_ld if p is not None]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    try:
        proc = subprocess.run(
            ["ldd", str(blender)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=20,
        )
    except Exception:
        return []
    missing = []
    for line in proc.stdout.splitlines():
        if "not found" in line:
            missing.append(line.strip().split()[0])
    return missing


def render_glb(
    glb: Path,
    output_dir: Path,
    view_ids: Sequence[str],
    azimuths: Sequence[float],
    elevations: Sequence[float],
    blender: Path,
    render_script: Path,
    resolution: int,
    samples: int,
    shrink: float,
    local_lib_dir: Path | None,
    runtime_prefix: Path | None,
    use_xvfb: bool = True,
    xvfb_display: str = ":99",
    force: bool = False,
) -> None:
    image_dir = output_dir / "image_ori"
    expected = [image_dir / f"{int(v) if str(v).isdigit() else i:03d}.png" for i, v in enumerate(view_ids)]
    if not force and all(p.exists() and p.stat().st_size > 0 for p in expected):
        return
    _ensure_dir(output_dir)
    if not blender.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    if not render_script.exists():
        raise FileNotFoundError(f"Render script not found: {render_script}")
    if len(view_ids) != len(azimuths):
        raise ValueError("view_ids and azimuths must have the same length")
    if len(elevations) == 1 and len(azimuths) > 1:
        elevations = list(elevations) * len(azimuths)
    if len(elevations) != len(azimuths):
        raise ValueError("view_elevations must have length 1 or match view_azimuths")

    extra_ld = blender_library_dirs(blender, local_lib_dir=local_lib_dir, runtime_prefix=runtime_prefix)
    missing = blender_missing_libraries(blender, extra_ld=extra_ld)
    if missing:
        raise RuntimeError(
            "Blender is missing dynamic libraries: "
            + ", ".join(missing)
            + ". Install them in the environment or run with --pred_dir to evaluate existing render images."
        )

    cmd = [
        str(blender),
        "-b",
        "-P",
        str(render_script),
        "--",
        "--object_path",
        str(glb),
        "--output_dir",
        str(output_dir),
        "--resolution",
        str(resolution),
        "--shrink",
        str(shrink),
        "--distance",
        "2.0",
        "--ortho_scale",
        "1.0",
        "--engine",
        "CYCLES",
        "--cycles_device",
        "CPU",
        "--samples",
        str(samples),
        "--view_azimuths",
        ",".join(str(x) for x in azimuths),
        "--view_elevations",
        ",".join(str(x) for x in elevations),
        "--view_ids",
        ",".join(str(v) for v in view_ids),
    ]
    env = os.environ.copy()
    env.pop("WAYLAND_DISPLAY", None)
    ld_parts = [str(p) for p in extra_ld]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    if runtime_prefix is not None:
        env["PATH"] = f"{runtime_prefix / 'bin'}:{env.get('PATH', '')}"
    env.setdefault("MESA_GL_VERSION_OVERRIDE", "4.3")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    xvfb_proc = None
    xvfb_log_path = output_dir / "render_xvfb.log"
    if use_xvfb:
        xvfb_bin = shutil.which("Xvfb", path=env.get("PATH"))
        if xvfb_bin is None:
            raise RuntimeError("Xvfb requested but not found. Install xorg-xvfb-server or pass --no_xvfb.")
        env["DISPLAY"] = xvfb_display
        with xvfb_log_path.open("w", encoding="utf-8") as xvfb_log:
            xvfb_proc = subprocess.Popen(
                [xvfb_bin, xvfb_display, "-screen", "0", "1024x768x24"],
                env=env,
                stdout=xvfb_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        time.sleep(3.0)
    else:
        env.pop("DISPLAY", None)

    log_path = output_dir / "render_blender.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    finally:
        if xvfb_proc is not None:
            xvfb_proc.terminate()
            try:
                xvfb_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                xvfb_proc.kill()
    if proc.returncode != 0:
        raise RuntimeError(f"Blender render failed with code {proc.returncode}. See {log_path}")

    # Copy canonical view names pred_<view>.png into output_dir/images.
    named_dir = output_dir / "images"
    _ensure_dir(named_dir)
    for i, vid in enumerate(view_ids):
        numeric_name = f"{int(vid) if str(vid).isdigit() else i:03d}.png"
        src = image_dir / numeric_name
        if src.exists():
            shutil.copy2(src, named_dir / f"pred_{vid}.png")


class MetricComputer:
    def __init__(self, device: torch.device, batch_size: int = 8, no_download: bool = False):
        self.device = device
        self.batch_size = batch_size
        self.no_download = no_download
        self.clip_model = None
        self.clip_processor = None
        self.dino_model = None
        self.dino_processor = None
        self.lpips_model = None
        self.inception = None

    def _check_transformers(self) -> None:
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise RuntimeError(f"transformers import failed: {_TRANSFORMERS_IMPORT_ERROR}")
        if any(x is None for x in [AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel]):
            raise RuntimeError("transformers CLIP/DINO classes are unavailable")

    def load_clip(self):
        if self.clip_model is not None:
            return
        self._check_transformers()
        kwargs = {"local_files_only": self.no_download}
        model_id = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
        print(f"[metrics] loading CLIP model: {model_id}", flush=True)
        self.clip_processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        self.clip_model = CLIPModel.from_pretrained(model_id, use_safetensors=True, **kwargs).to(self.device).eval()
        print("[metrics] CLIP ready", flush=True)

    def load_dino(self):
        if self.dino_model is not None:
            return
        self._check_transformers()
        kwargs = {"local_files_only": self.no_download}
        model_id = os.environ.get("DINO_MODEL", "facebook/dinov2-base")
        print(f"[metrics] loading DINO model: {model_id}", flush=True)
        self.dino_processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
        self.dino_model = AutoModel.from_pretrained(model_id, use_safetensors=True, **kwargs).to(self.device).eval()
        print("[metrics] DINO ready", flush=True)

    def load_lpips(self):
        if self.lpips_model is not None:
            return
        # Torchmetrics' LPIPS implementation downloads AlexNet weights via torchvision.
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except Exception as exc:
            raise RuntimeError(f"LPIPS metric unavailable: {exc}") from exc
        print("[metrics] loading LPIPS/AlexNet", flush=True)
        self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(self.device).eval()
        print("[metrics] LPIPS ready", flush=True)

    def load_inception(self):
        if self.inception is not None:
            return
        print("[metrics] loading InceptionV3 for FID", flush=True)
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, transform_input=False, aux_logits=True)
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        self.inception = model
        print("[metrics] InceptionV3 ready", flush=True)

    @torch.no_grad()
    def clip_t(self, pred_paths: Sequence[Path], prompt: str) -> float | None:
        if not prompt:
            return None
        self.load_clip()
        scores = []
        text_inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        text_outputs = self.clip_model.text_model(**text_inputs)
        text_features = self.clip_model.text_projection(text_outputs.pooler_output)
        text_features = F.normalize(text_features, dim=-1)
        for start in range(0, len(pred_paths), self.batch_size):
            batch_paths = pred_paths[start : start + self.batch_size]
            images = [load_rgb(p) for p in batch_paths]
            inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
            image_outputs = self.clip_model.vision_model(**inputs)
            image_features = self.clip_model.visual_projection(image_outputs.pooler_output)
            image_features = F.normalize(image_features, dim=-1)
            batch_scores = (image_features @ text_features.T).squeeze(1).detach().cpu().tolist()
            scores.extend(batch_scores)
        return _safe_mean(scores)

    @torch.no_grad()
    def dino_i(self, pred_paths: Sequence[Path], target_paths: Sequence[Path]) -> float | None:
        self.load_dino()
        scores = []
        for start in range(0, len(pred_paths), self.batch_size):
            p_batch = pred_paths[start : start + self.batch_size]
            t_batch = target_paths[start : start + self.batch_size]
            p_images = [load_rgb(p) for p in p_batch]
            t_images = [load_rgb(p) for p in t_batch]
            p_inputs = self.dino_processor(images=p_images, return_tensors="pt").to(self.device)
            t_inputs = self.dino_processor(images=t_images, return_tensors="pt").to(self.device)
            p_out = self.dino_model(**p_inputs)
            t_out = self.dino_model(**t_inputs)
            p_feat = getattr(p_out, "pooler_output", None)
            t_feat = getattr(t_out, "pooler_output", None)
            if p_feat is None:
                p_feat = p_out.last_hidden_state[:, 0]
            if t_feat is None:
                t_feat = t_out.last_hidden_state[:, 0]
            p_feat = F.normalize(p_feat, dim=-1)
            t_feat = F.normalize(t_feat, dim=-1)
            scores.extend((p_feat * t_feat).sum(dim=-1).detach().cpu().tolist())
        return _safe_mean(scores)

    @torch.no_grad()
    def lpips(self, pred_paths: Sequence[Path], target_paths: Sequence[Path], size: int = 256) -> float | None:
        self.load_lpips()
        vals = []
        for start in range(0, len(pred_paths), self.batch_size):
            p_batch = pred_paths[start : start + self.batch_size]
            t_batch = target_paths[start : start + self.batch_size]
            p = image_to_float_tensor(p_batch, size=size, normalize_lpips=True).to(self.device)
            t = image_to_float_tensor(t_batch, size=size, normalize_lpips=True).to(self.device)
            v = self.lpips_model(p, t)
            vals.append(float(v.detach().cpu().item()))
        return _safe_mean(vals)

    @torch.no_grad()
    def _inception_features(self, paths: Sequence[Path], size: int = 299) -> np.ndarray:
        self.load_inception()
        preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        feats = []
        for start in range(0, len(paths), self.batch_size):
            batch = image_to_float_tensor(paths[start : start + self.batch_size], size=size, normalize_lpips=False)
            batch = preprocess(batch).to(self.device)
            out = self.inception(batch)
            if isinstance(out, tuple):
                out = out[0]
            feats.append(out.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(feats, axis=0)

    @staticmethod
    def _frechet(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))

    @torch.no_grad()
    def fid(self, pred_paths: Sequence[Path], target_paths: Sequence[Path]) -> float | None:
        # FID with very few images is noisy/singular. We use shrinkage for
        # numerical stability so that the metric is still useful for comparing
        # runs on the same small view set.
        if len(pred_paths) < 2 or len(target_paths) < 2:
            return None
        p = self._inception_features(pred_paths)
        t = self._inception_features(target_paths)
        mu_p = p.mean(axis=0)
        mu_t = t.mean(axis=0)
        sigma_p = np.cov(p, rowvar=False)
        sigma_t = np.cov(t, rowvar=False)
        eye = np.eye(sigma_p.shape[0], dtype=np.float64)
        sigma_p = sigma_p + eye * 1e-6
        sigma_t = sigma_t + eye * 1e-6
        return self._frechet(mu_p, sigma_p, mu_t, sigma_t)


def evaluate_pairs(
    name: str,
    pairs: Sequence[ImagePair],
    prompt: str,
    computer: MetricComputer,
    pred_dir: Path,
    target_dir: Path,
    glb_path: Path | None,
    skip_clip: bool = False,
    skip_dino: bool = False,
    skip_lpips: bool = False,
    skip_fid: bool = False,
) -> tuple[MetricResult, list[dict]]:
    if not pairs:
        return (
            MetricResult(name, None, None, None, None, 0, pred_dir, target_dir, glb_path, error="No matched prediction/target image pairs"),
            [],
        )
    pred_paths = [p.pred for p in pairs]
    target_paths = [p.target for p in pairs]
    details: list[dict] = []
    result = MetricResult(name, None, None, None, None, len(pairs), pred_dir, target_dir, glb_path)
    try:
        if not skip_clip:
            result.clip_t = computer.clip_t(pred_paths, prompt)
    except Exception as exc:
        result.error = f"CLIP-T failed: {exc}"
    try:
        if not skip_dino:
            result.dino_i = computer.dino_i(pred_paths, target_paths)
    except Exception as exc:
        result.error = (result.error + "; " if result.error else "") + f"DINO-I failed: {exc}"
    try:
        if not skip_lpips:
            result.lpips = computer.lpips(pred_paths, target_paths)
    except Exception as exc:
        result.error = (result.error + "; " if result.error else "") + f"LPIPS failed: {exc}"
    try:
        if not skip_fid:
            result.fid = computer.fid(pred_paths, target_paths)
    except Exception as exc:
        result.error = (result.error + "; " if result.error else "") + f"FID failed: {exc}"

    for pair in pairs:
        details.append(
            {
                "view_id": pair.view_id,
                "pred": str(pair.pred),
                "target": str(pair.target),
            }
        )
    return result, details


def result_to_dict(r: MetricResult) -> dict:
    return {
        "name": r.name,
        "CLIP-T": r.clip_t,
        "DINO-I": r.dino_i,
        "LPIPS": r.lpips,
        "FID": r.fid,
        "n_pairs": r.n_pairs,
        "pred_dir": str(r.pred_dir),
        "target_dir": str(r.target_dir),
        "glb_path": str(r.glb_path) if r.glb_path else None,
        "error": r.error,
    }


def write_csv(path: Path, rows: Sequence[MetricResult]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "CLIP-T", "DINO-I", "LPIPS", "FID", "n_pairs", "pred_dir", "target_dir", "glb_path", "error"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(result_to_dict(r))


def maybe_reexec_in_test_env() -> None:
    current = Path(sys.executable).resolve()
    if DEFAULT_TEST_ENV_PYTHON.exists() and current != DEFAULT_TEST_ENV_PYTHON.resolve():
        os.execv(str(DEFAULT_TEST_ENV_PYTHON), [str(DEFAULT_TEST_ENV_PYTHON), *sys.argv])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item_dir", type=Path, default=REPO_ROOT / "demo" / "ninja")
    parser.add_argument("--output_dir", type=Path, default=None, help="Metric output folder. Default: <item_dir>/test")
    parser.add_argument("--easy3e_out", action="append", default=None, help="Specific mv_fusion output dir. Can be repeated.")
    parser.add_argument("--glb", action="append", default=None, help="Specific GLB to evaluate. Can be repeated.")
    parser.add_argument("--pred_dir", type=Path, default=None, help="Evaluate existing prediction images instead of rendering GLBs.")
    parser.add_argument("--target_dir", type=Path, default=None, help="Directory containing target_<view>.png or target_<view>_rgba.png")
    parser.add_argument("--prompt", type=str, default="a ninja character wearing a golden belt", help="Text prompt for CLIP-T")
    parser.add_argument("--view_ids", type=str, default="0,1,2")
    parser.add_argument("--view_azimuths", type=str, default="0,120,270")
    parser.add_argument("--view_elevations", type=str, default="0,0,0")
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--render_samples", type=int, default=32)
    parser.add_argument("--render_shrink", type=float, default=0.0, help="Use 0 for already-normalized Easy3E outputs; 1 normalizes source GLBs")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--render_script", type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--blender_runtime_prefix", type=Path, default=None, help="Optional prefix containing X11/OpenGL runtime libraries and Xvfb")
    parser.add_argument("--no_xvfb", action="store_true", help="Do not launch Blender under Xvfb")
    parser.add_argument("--xvfb_display", type=str, default=":99")
    parser.add_argument("--force_render", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_download", action="store_true", help="Only use already-cached model weights")
    parser.add_argument("--skip_clip", action="store_true")
    parser.add_argument("--skip_dino", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--prefer_test_env", action="store_true", help="Re-exec with $METRICS_PYTHON if it is set")
    parser.add_argument("--make_blender_stub_libs", action="store_true", default=True)
    args = parser.parse_args()

    if args.prefer_test_env:
        maybe_reexec_in_test_env()

    item_dir = args.item_dir.expanduser().resolve()
    output_dir = args.output_dir or (item_dir / "test")
    output_dir = output_dir.expanduser().resolve()
    _ensure_dir(output_dir)

    view_ids = _parse_csv_strings(args.view_ids, ["0", "1", "2"])
    azimuths = _parse_csv_floats(args.view_azimuths, [0.0, 120.0, 270.0])
    elevations = _parse_csv_floats(args.view_elevations, [0.0] * len(azimuths))
    if len(elevations) == 1 and len(azimuths) > 1:
        elevations = elevations * len(azimuths)

    runtime_prefix = args.blender_runtime_prefix.expanduser().resolve() if args.blender_runtime_prefix else None
    manifest = {
        "item_dir": str(item_dir),
        "output_dir": str(output_dir),
        "prompt": args.prompt,
        "view_ids": view_ids,
        "view_azimuths": azimuths,
        "view_elevations": elevations,
        "python": sys.executable,
        "torch": torch.__version__,
        "device": args.device,
        "blender_runtime_prefix": str(runtime_prefix) if runtime_prefix else None,
        "use_xvfb": not args.no_xvfb,
        "xvfb_display": args.xvfb_display,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    local_lib_dir = None
    if args.make_blender_stub_libs:
        local_lib_dir = make_blender_stub_libs(output_dir / "_local_libs")
        if local_lib_dir is not None:
            manifest["local_blender_stub_libs"] = str(local_lib_dir)

    rows: list[MetricResult] = []
    all_details: dict[str, list[dict]] = {}

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    computer = MetricComputer(device=device, batch_size=args.batch_size, no_download=args.no_download)

    if args.pred_dir is not None:
        pred_dir = args.pred_dir.expanduser().resolve()
        target_dir = (args.target_dir.expanduser().resolve() if args.target_dir else item_dir)
        pairs = pair_images(pred_dir, target_dir, view_ids=view_ids)
        result, details = evaluate_pairs(
            pred_dir.name,
            pairs,
            args.prompt,
            computer,
            pred_dir,
            target_dir,
            None,
            skip_clip=args.skip_clip,
            skip_dino=args.skip_dino,
            skip_lpips=args.skip_lpips,
            skip_fid=args.skip_fid,
        )
        rows.append(result)
        all_details[result.name] = details
    else:
        glbs: list[tuple[str, Path, Path]] = []
        if args.glb:
            target_dir = args.target_dir.expanduser().resolve() if args.target_dir else item_dir
            for g in args.glb:
                gp = Path(g).expanduser().resolve()
                glbs.append((gp.stem, gp, target_dir))
        else:
            out_dirs = discover_output_dirs(item_dir, explicit=args.easy3e_out)
            for out in out_dirs:
                target_dir = args.target_dir.expanduser().resolve() if args.target_dir else target_dir_from_summary(out, item_dir)
                for gp in discover_glbs(out):
                    glbs.append((out.name, gp, target_dir))

        if not glbs:
            raise FileNotFoundError("No GLBs found. Pass --glb or --pred_dir, or check --item_dir/--easy3e_out.")

        for name, glb, target_dir in glbs:
            render_dir = output_dir / "renders" / name
            try:
                render_glb(
                    glb,
                    render_dir,
                    view_ids=view_ids,
                    azimuths=azimuths,
                    elevations=elevations,
                    blender=args.blender.expanduser().resolve(),
                    render_script=args.render_script.expanduser().resolve(),
                    resolution=args.render_resolution,
                    samples=args.render_samples,
                    shrink=args.render_shrink,
                    local_lib_dir=local_lib_dir,
                    runtime_prefix=runtime_prefix,
                    use_xvfb=not args.no_xvfb,
                    xvfb_display=args.xvfb_display,
                    force=args.force_render,
                )
                pred_dir = render_dir / "images"
                pairs = pair_images(pred_dir, target_dir, view_ids=view_ids)
                result, details = evaluate_pairs(
                    name,
                    pairs,
                    args.prompt,
                    computer,
                    pred_dir,
                    target_dir,
                    glb,
                    skip_clip=args.skip_clip,
                    skip_dino=args.skip_dino,
                    skip_lpips=args.skip_lpips,
                    skip_fid=args.skip_fid,
                )
            except Exception as exc:
                pred_dir = render_dir / "images"
                result = MetricResult(name, None, None, None, None, 0, pred_dir, target_dir, glb, error=str(exc))
                details = []
            rows.append(result)
            all_details[name] = details

    manifest["results"] = [result_to_dict(r) for r in rows]
    manifest["pairs"] = all_details
    manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    _write_json(output_dir / "metrics.json", manifest)
    write_csv(output_dir / "metrics.csv", rows)

    print(f"[metrics] wrote {output_dir / 'metrics.json'}")
    print(f"[metrics] wrote {output_dir / 'metrics.csv'}")
    for r in rows:
        print(
            f"{r.name}: CLIP-T={r.clip_t} DINO-I={r.dino_i} "
            f"LPIPS={r.lpips} FID={r.fid} n={r.n_pairs}"
            + (f" ERROR={r.error}" if r.error else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
