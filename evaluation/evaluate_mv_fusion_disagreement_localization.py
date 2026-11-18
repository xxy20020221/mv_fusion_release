#!/usr/bin/env python3
"""Evaluate mv_fusion disagreement against image-derived 3D residuals."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import requests
import trimesh
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "mv_fusion"))
import metrics as render_eval  # noqa: E402
from paths import RENDER_GLB_MV_HIGH_DETAIL_SCRIPT as DEFAULT_HIGH_DETAIL_RENDER_SCRIPT  # noqa: E402

DEFAULT_ITEM_DIR = PROJECT_ROOT / "demo" / "rider"
DEFAULT_RUN_DIR = DEFAULT_ITEM_DIR / "_easy3e_out_round1"
DEFAULT_WORK_DIR = DEFAULT_ITEM_DIR / "_easy3e_work_round1"
DEFAULT_RUNTIME_PREFIX = Path(os.environ.get("BLENDER_RUNTIME_PREFIX", PROJECT_ROOT / "blender_runtime_libs"))
DEFAULT_VLM_MODEL = "gemini-3-pro-preview"
DEFAULT_VLM_FALLBACK_MODELS = "gemini-3-pro-preview,claude-sonnet-4-5-20250929,gpt-4o,gpt-4o-mini"
DEFAULT_MAGIC666_BASE_URL = os.environ.get("MAGIC666_BASE_URL", "https://magic666.top/v1")
DEFAULT_MAGIC666_API_KEY = os.environ.get("MAGIC666_API_KEY", "")
DEFAULT_VIEWS = [
    (30.0, 30.0),
    (120.0, 30.0),
    (210.0, 30.0),
    (300.0, 30.0),
    (60.0, -30.0),
    (150.0, -30.0),
    (240.0, -30.0),
    (330.0, -30.0),
]
VLM_SYSTEM_PROMPT = """You are a strict visual evaluator for 3D asset editing.
Compare a rendered image of the current 3D result with a target generated/reference image.
Your task is to mark TARGET_REFERENCE regions that still require editing in the 3D asset because important semantic content, shape, geometry, material pattern, color identity, or distinctive object features are clearly different.
Prioritize high-impact editable differences over minor details. For characters or creatures, explicitly inspect and prioritize hands, fingers, arms, face/head, clothing pieces, accessories, shoes/feet, weapons/tools, and distinctive color or texture blocks. If a hand, finger pose, missing hand, or wrong hand shape is visibly different, it should usually be marked as an important region.
Do not spend region slots on low-impact details while more obvious body-part, clothing, accessory, silhouette, or object-identity errors remain unmarked.
Ignore global scale changes, translation, mild pose/camera mismatch, tiny pixel misalignment, anti-aliasing, lighting, shadows, background transparency, renderer style, and insignificant texture noise.
Return machine-readable JSON only."""
LOW_COLOR = np.array([255.0, 238.0, 225.0], dtype=np.float32)
HIGH_COLOR = np.array([165.0, 0.0, 38.0], dtype=np.float32)
CONFLICT_COLOR = np.array([255, 80, 80, 240], dtype=np.uint8)
NON_CONFLICT_COLOR = np.array([160, 160, 160, 120], dtype=np.uint8)


@dataclass(frozen=True)
class ViewPair:
    target_id: str
    render_view_id: int


def parse_pairs(value: str) -> list[ViewPair]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        target_id, render_id = item.split(":", 1)
        out.append(ViewPair(target_id.strip(), int(render_id)))
    if not out:
        raise ValueError("No view pairs were provided")
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_camera(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return data.item() if isinstance(data, np.ndarray) and data.shape == () else dict(data)


def load_scene_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = tuple(loaded.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry in {path}")
        return trimesh.util.concatenate(geoms)
    return loaded


def voxel_centers_and_colors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_scene_mesh(path)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if len(verts) % 8 != 0:
        raise ValueError(f"Expected cube-concatenated voxel GLB: {path}")
    centers = verts.reshape(-1, 8, 3).mean(axis=1)
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).reshape(-1, 8, 4)[:, 0, :]
    return centers, colors


def centers_to_indices(centers: np.ndarray, grid_size: int) -> np.ndarray:
    return np.clip(np.rint((centers + 0.5) * grid_size - 0.5).astype(np.int32), 0, grid_size - 1)


def decode_heatmap_values(colors: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    rgb = colors[:, :3].astype(np.float32)
    denom = HIGH_COLOR - LOW_COLOR
    valid = np.abs(denom) > 1e-6
    norm = ((rgb[:, valid] - LOW_COLOR[valid]) / denom[valid]).mean(axis=1)
    norm = np.clip(norm, 0.0, 1.0)
    return (v_min + norm * max(v_max - v_min, 1e-6)).astype(np.float32)


def load_scalar_mask_signal(
    run_dir: Path,
    grid_size: int,
    stats_name: str,
    min_key: str,
    max_key: str,
    center_reference: np.ndarray | None = None,
    mismatch_label: str = "signal",
) -> dict:
    stats_path = run_dir / "features" / "mv_fusion_voxel_diff" / stats_name
    if not stats_path.exists():
        stats_path = run_dir / "out" / stats_name
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    heatmap_path = Path(stats["heatmap_glb"])
    mask_path = Path(stats["conflict_mask_glb"])
    if not heatmap_path.exists():
        heatmap_path = stats_path.parent / heatmap_path.name
    if not mask_path.exists():
        mask_path = stats_path.parent / mask_path.name
    centers, heat_colors = voxel_centers_and_colors(heatmap_path)
    mask_centers, mask_colors = voxel_centers_and_colors(mask_path)
    if centers.shape != mask_centers.shape or np.max(np.abs(centers - mask_centers)) > 1e-5:
        raise ValueError(f"{mismatch_label} heatmap and mask voxel centers differ")
    if center_reference is not None and (centers.shape != center_reference.shape or np.max(np.abs(centers - center_reference)) > 1e-5):
        raise ValueError(f"{mismatch_label} voxel centers differ from disagreement centers")
    scores = decode_heatmap_values(
        heat_colors,
        float(stats[min_key]),
        float(stats[max_key]),
    )
    conflict = np.linalg.norm(mask_colors.astype(np.int16) - CONFLICT_COLOR.astype(np.int16), axis=1) < 40.0
    return {
        "stats": stats,
        "stats_path": stats_path,
        "heatmap_path": heatmap_path,
        "mask_path": mask_path,
        "centers": centers,
        "indices": centers_to_indices(centers, grid_size),
        "scores": scores,
        "conflict": conflict,
    }


def load_disagreement(run_dir: Path, grid_size: int) -> dict:
    return load_scalar_mask_signal(
        run_dir=run_dir,
        grid_size=grid_size,
        stats_name="velocity_conflict_1_vs_0_stats.json",
        min_key="velocity_diff_min_in_mask",
        max_key="velocity_diff_max_in_mask",
        mismatch_label="Disagreement",
    )


def load_edit_magnitude_conflict(run_dir: Path, grid_size: int, center_reference: np.ndarray) -> dict | None:
    stats_path = run_dir / "features" / "mv_fusion_voxel_diff" / "edit_magnitude_conflict_stats.json"
    if not stats_path.exists():
        stats_path = run_dir / "out" / "edit_magnitude_conflict_stats.json"
    if not stats_path.exists():
        return None
    return load_scalar_mask_signal(
        run_dir=run_dir,
        grid_size=grid_size,
        stats_name="edit_magnitude_conflict_stats.json",
        min_key="edit_magnitude_min_in_mask",
        max_key="edit_magnitude_max_in_mask",
        center_reference=center_reference,
        mismatch_label="Edit magnitude conflict",
    )


def rgb_on_white(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        img = img.convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def alpha_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray | None:
    img = Image.open(path)
    if img.mode not in ("RGBA", "LA") and not (img.mode == "P" and "transparency" in img.info):
        return None
    rgba = img.convert("RGBA")
    if size is not None and rgba.size != size:
        rgba = rgba.resize(size, Image.Resampling.NEAREST)
    return np.asarray(rgba, dtype=np.uint8)[..., 3] > 8


def find_target(target_dir: Path, fallback_dir: Path, target_id: str) -> Path:
    candidates = [
        target_dir / f"target_{target_id}.png",
        target_dir / f"target_{target_id}_rgba.png",
        fallback_dir / f"target_{target_id}_rgba.png",
        fallback_dir / f"target_{target_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing target for {target_id}: {candidates}")


def compute_pixel_residual_maps(pairs: Sequence[ViewPair], render_dir: Path, target_dir: Path, fallback_dir: Path, out_dir: Path) -> dict[int, dict]:
    ensure_dir(out_dir)
    out = {}
    for pair in pairs:
        pred_path = render_dir / "image_ori" / f"{pair.render_view_id:03d}.png"
        target_path = find_target(target_dir, fallback_dir, pair.target_id)
        render_size = Image.open(pred_path).size
        target_size = Image.open(target_path).size
        pred = rgb_on_white(pred_path, target_size)
        target = rgb_on_white(target_path, target_size)
        mask = alpha_mask(target_path, target_size)
        if mask is None:
            pred_alpha = alpha_mask(pred_path, target_size)
            mask = pred_alpha if pred_alpha is not None else np.ones(target.shape[:2], dtype=bool)
        residual = (np.linalg.norm(pred - target, axis=2) / math.sqrt(3.0)).astype(np.float32)
        residual[~mask] = 0.0
        Image.fromarray((np.clip(residual, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(
            out_dir / f"residual_target_{pair.target_id}_view_{pair.render_view_id:03d}.png"
        )
        out[pair.render_view_id] = {
            "target_id": pair.target_id,
            "pred_path": str(pred_path),
            "target_path": str(target_path),
            "render_size": render_size,
            "target_size": target_size,
            "residual": residual,
            "valid_mask": mask,
            "source": "pixel",
        }
    return out


def _image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _prepare_vlm_image(path: Path, out_path: Path, max_side: int = 768, quality: int = 88) -> Path:
    ensure_dir(out_path.parent)
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        img = img.convert("RGB")
    scale = min(1.0, float(max_side) / float(max(img.size)))
    if scale < 1.0:
        new_size = (max(1, int(round(img.size[0] * scale))), max(1, int(round(img.size[1] * scale))))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    img.save(out_path, format="JPEG", quality=quality, optimize=True)
    return out_path


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in VLM response: {text[:500]}")
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(cleaned[start : idx + 1])
                if not isinstance(data, dict):
                    raise ValueError("VLM response JSON root is not an object")
                return data
    raise ValueError(f"Could not parse complete JSON object from VLM response: {text[:500]}")


def _region_list(data: dict[str, Any]) -> list[Any]:
    for key in ("regions", "different_regions", "edit_regions", "boxes", "differences"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_vlm_regions(data: dict[str, Any]) -> list[dict[str, Any]]:
    regions = []
    for raw in _region_list(data):
        if not isinstance(raw, dict):
            continue
        box = raw.get("box") or raw.get("bbox") or raw.get("bounding_box")
        if isinstance(box, dict):
            if all(k in box for k in ("x", "y", "width", "height")):
                x = _as_float(box.get("x"), math.nan)
                y = _as_float(box.get("y"), math.nan)
                box = [x, y, x + _as_float(box.get("width"), math.nan), y + _as_float(box.get("height"), math.nan)]
            else:
                box = [
                    box.get("x1", box.get("xmin", box.get("left"))),
                    box.get("y1", box.get("ymin", box.get("top"))),
                    box.get("x2", box.get("xmax", box.get("right"))),
                    box.get("y2", box.get("ymax", box.get("bottom"))),
                ]
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [_as_float(v, math.nan) for v in box]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        severity = max(0.0, min(1.0, _as_float(raw.get("severity", raw.get("score", 1.0)), 1.0)))
        confidence = max(0.0, min(1.0, _as_float(raw.get("confidence", 1.0), 1.0)))
        # Treat VLM-selected regions as GT; confidence is retained only for auditing.
        regions.append(
            {
                "box": [x1, y1, x2, y2],
                "severity": severity,
                "confidence": confidence,
                "score": severity,
                "reason": str(raw.get("reason", raw.get("description", "")))[:300],
            }
        )
    return regions


def _vlm_question(target_size: tuple[int, int]) -> str:
    width, height = target_size
    return f"""You will receive two images:
Image 1 is CURRENT_RENDER, a Blender rendering of the current edited 3D asset.
Image 2 is TARGET_REFERENCE, the desired generated/reference image.

Find the most important regions in TARGET_REFERENCE that still need editing in the 3D asset.
Before choosing boxes, systematically compare these high-priority areas when visible: hands/fingers, arms, face/head/hair, torso clothing, belts/straps/bags, shoes/feet, held objects/tools/weapons, major silhouette parts, and distinctive material/color/texture blocks.
Prioritize regions by editing importance:
1. Missing or wrong body parts, especially hands/fingers/arms, face/head, feet/shoes.
2. Missing or wrong salient accessories or held objects, such as straps, bags, belts, weapons, tools, armor, or badges.
3. Wrong clothing pieces, large color blocks, material patterns, or distinctive textures.
4. Only then mark smaller local details if no higher-priority differences remain.
Do not mark minor rendering/style differences, shadows, lighting, antialiasing, tiny boundary offsets, or features that are only globally shifted/rescaled by camera or alignment. If the same feature exists and is merely shifted or slightly resized, do not mark it.
Use tight but complete boxes around the target-side feature that should be edited. If two nearby missing/wrong features are part of one semantic object, group them; if they are separate important objects, use separate boxes. Prefer missing obvious regions over subtle decorative details.

Return JSON only, with this schema:
{{
  "summary": "short comparison summary",
  "regions": [
    {{"box": [x1, y1, x2, y2], "severity": 0.0-1.0, "confidence": 0.0-1.0, "reason": "why this region needs editing; mention the target-side feature"}}
  ]
}}
Coordinates must be normalized to the TARGET_REFERENCE image only: x/y range 0..1000, origin at top-left.
The actual target image size is {width}x{height}. Return the top 3-8 important editable regions. Use an empty list only if there is no important editable difference."""


def _call_vlm_api(
    pred_path: Path,
    target_path: Path,
    target_size: tuple[int, int],
    models: Sequence[str],
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[dict[str, Any], str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    errors = []
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _vlm_question(target_size)},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(pred_path)}},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(target_path)}},
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            response = requests.post(f"{api_base.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
            if response.status_code != 200:
                errors.append(f"{model}: HTTP {response.status_code} {response.text[:500]}")
                continue
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return _extract_json_object(text), text, model
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            continue
    raise RuntimeError("All VLM API models failed:\n" + "\n".join(errors))


def _regions_to_heatmap(regions: Sequence[dict[str, Any]], size: tuple[int, int], valid_mask: np.ndarray) -> np.ndarray:
    width, height = size
    heat = np.zeros((height, width), dtype=np.float32)
    for region in regions:
        x1, y1, x2, y2 = region["box"]
        px1 = int(math.floor(x1 / 1000.0 * width))
        py1 = int(math.floor(y1 / 1000.0 * height))
        px2 = int(math.ceil(x2 / 1000.0 * width))
        py2 = int(math.ceil(y2 / 1000.0 * height))
        px1, px2 = max(0, min(width - 1, px1)), max(1, min(width, px2))
        py1, py2 = max(0, min(height - 1, py1)), max(1, min(height, py2))
        if px2 <= px1 or py2 <= py1:
            continue
        heat[py1:py2, px1:px2] = np.maximum(heat[py1:py2, px1:px2], float(region.get("score", 1.0)))
    heat[~valid_mask] = 0.0
    return heat


def _save_vlm_visualizations(
    pred_path: Path,
    target_path: Path,
    target_size: tuple[int, int],
    residual: np.ndarray,
    regions: Sequence[dict[str, Any]],
    out_prefix: Path,
) -> dict[str, str]:
    ensure_dir(out_prefix.parent)
    width, height = target_size
    pred_img = Image.fromarray((rgb_on_white(pred_path, target_size) * 255).clip(0, 255).astype(np.uint8), mode="RGB")
    target_img = Image.fromarray((rgb_on_white(target_path, target_size) * 255).clip(0, 255).astype(np.uint8), mode="RGB")
    heat_img = Image.fromarray((np.clip(residual, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    heat_path = out_prefix.with_name(out_prefix.name + "_heatmap.png")
    heat_img.save(heat_path)

    overlay = target_img.convert("RGBA")
    fill = Image.new("RGBA", overlay.size, (255, 0, 0, 0))
    fill_alpha = (np.clip(residual, 0.0, 1.0) * 150).astype(np.uint8)
    fill_np = np.array(fill)
    fill_np[..., 3] = fill_alpha
    fill = Image.fromarray(fill_np, mode="RGBA")
    overlay = Image.alpha_composite(overlay, fill)
    draw = ImageDraw.Draw(overlay)
    for idx, region in enumerate(regions, start=1):
        x1, y1, x2, y2 = region["box"]
        box = [x1 / 1000.0 * width, y1 / 1000.0 * height, x2 / 1000.0 * width, y2 / 1000.0 * height]
        draw.rectangle(box, outline=(255, 0, 0, 255), width=4)
        draw.text((box[0] + 4, box[1] + 4), f"{idx}:{float(region.get('score', 1.0)):.2f}", fill=(255, 255, 255, 255))
    overlay_path = out_prefix.with_name(out_prefix.name + "_overlay.png")
    overlay.convert("RGB").save(overlay_path)

    canvas = Image.new("RGB", (width * 2, height + 34), (255, 255, 255))
    canvas.paste(pred_img, (0, 34))
    canvas.paste(overlay.convert("RGB"), (width, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), "CURRENT_RENDER", fill=(0, 0, 0))
    draw.text((width + 10, 9), "TARGET_REFERENCE + VLM edit regions", fill=(0, 0, 0))
    compare_path = out_prefix.with_name(out_prefix.name + "_compare.png")
    canvas.save(compare_path)
    return {"heatmap": str(heat_path), "overlay": str(overlay_path), "compare": str(compare_path)}


def compute_vlm_residual_maps(
    pairs: Sequence[ViewPair],
    render_dir: Path,
    target_dir: Path,
    fallback_dir: Path,
    out_dir: Path,
    models: Sequence[str],
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    force: bool,
) -> dict[int, dict]:
    ensure_dir(out_dir)
    decision_dir = out_dir / "decisions"
    api_image_dir = out_dir / "api_images"
    ensure_dir(decision_dir)
    ensure_dir(api_image_dir)
    out = {}
    for pair in pairs:
        pred_path = render_dir / "image_ori" / f"{pair.render_view_id:03d}.png"
        target_path = find_target(target_dir, fallback_dir, pair.target_id)
        render_size = Image.open(pred_path).size
        target_size = Image.open(target_path).size
        mask = alpha_mask(target_path, target_size)
        if mask is None:
            pred_alpha = alpha_mask(pred_path, target_size)
            mask = pred_alpha if pred_alpha is not None else np.ones((target_size[1], target_size[0]), dtype=bool)
        decision_base = decision_dir / f"vlm_target_{pair.target_id}_view_{pair.render_view_id:03d}"
        json_path = decision_base.with_suffix(".json")
        raw_path = decision_base.with_suffix(".txt")
        used_model = models[0] if models else ""
        if json_path.exists() and not force:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            raw_text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else json.dumps(data, ensure_ascii=False)
            used_model = str(data.get("model", used_model))
        else:
            api_pred_path = _prepare_vlm_image(pred_path, api_image_dir / f"render_view_{pair.render_view_id:03d}.jpg")
            api_target_path = _prepare_vlm_image(target_path, api_image_dir / f"target_{pair.target_id}_view_{pair.render_view_id:03d}.jpg")
            data, raw_text, used_model = _call_vlm_api(
                pred_path=api_pred_path,
                target_path=api_target_path,
                target_size=target_size,
                models=models,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            raw_path.write_text(raw_text, encoding="utf-8")
        regions = _parse_vlm_regions(data)
        data["regions"] = regions
        data["target_id"] = pair.target_id
        data["render_view_id"] = pair.render_view_id
        data["pred_path"] = str(pred_path)
        data["target_path"] = str(target_path)
        data["coordinate_system"] = "normalized_target_image_0_1000_top_left"
        data["model"] = used_model
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        residual = _regions_to_heatmap(regions, target_size, mask)
        vis = _save_vlm_visualizations(pred_path, target_path, target_size, residual, regions, out_dir / f"vlm_target_{pair.target_id}_view_{pair.render_view_id:03d}")
        out[pair.render_view_id] = {
            "target_id": pair.target_id,
            "pred_path": str(pred_path),
            "target_path": str(target_path),
            "render_size": render_size,
            "target_size": target_size,
            "residual": residual,
            "valid_mask": mask,
            "source": "vlm",
            "model": used_model,
            "vlm_decision_json": str(json_path),
            "vlm_decision_raw": str(raw_path),
            "vlm_regions": regions,
            "vlm_visualizations": vis,
        }
    return out


def compute_residual_maps(
    pairs: Sequence[ViewPair],
    render_dir: Path,
    target_dir: Path,
    fallback_dir: Path,
    out_dir: Path,
    residual_source: str,
    vlm_models: Sequence[str],
    vlm_api_base: str,
    vlm_api_key: str,
    vlm_temperature: float,
    vlm_max_tokens: int,
    vlm_timeout: int,
    force_vlm: bool,
) -> dict[int, dict]:
    if residual_source == "pixel":
        return compute_pixel_residual_maps(pairs, render_dir, target_dir, fallback_dir, out_dir)
    if residual_source == "vlm":
        if not vlm_api_key:
            raise RuntimeError("Missing VLM API key. Set MAGIC666_API_KEY or pass --vlm_api_key.")
        return compute_vlm_residual_maps(
            pairs=pairs,
            render_dir=render_dir,
            target_dir=target_dir,
            fallback_dir=fallback_dir,
            out_dir=out_dir,
            models=vlm_models,
            api_base=vlm_api_base,
            api_key=vlm_api_key,
            temperature=vlm_temperature,
            max_tokens=vlm_max_tokens,
            timeout=vlm_timeout,
            force=force_vlm,
        )
    raise ValueError(f"Unknown residual source: {residual_source}")


def render_transform_from_meta(render_dir: Path) -> tuple[float, np.ndarray]:
    meta_path = render_dir / "meta.npy"
    if not meta_path.exists():
        return 1.0, np.zeros(3, dtype=np.float32)
    meta = np.asarray(np.load(meta_path), dtype=np.float32).reshape(-1)
    if len(meta) >= 4:
        return float(meta[0]), meta[1:4].astype(np.float32)
    return 1.0, np.zeros(3, dtype=np.float32)


def project_points_ortho(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.asarray(camera["intrinsic"], dtype=np.float32)
    rt = np.asarray(camera["extrinsic"], dtype=np.float32)
    pts_h = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    cam_pts = pts_h @ rt.T
    x = k[0, 0] * cam_pts[:, 0] + k[0, 2]
    y = k[1, 1] * cam_pts[:, 1] + k[1, 2]
    return x, y, cam_pts[:, 2]


def residual_to_voxels(centers: np.ndarray, pairs: Sequence[ViewPair], render_dir: Path, residual_maps: dict[int, dict], depth_eps: float) -> tuple[np.ndarray, np.ndarray, dict]:
    scale, offset = render_transform_from_meta(render_dir)
    points = centers * scale + offset[None, :]
    residual_3d = np.zeros(len(centers), dtype=np.float32)
    visible_count = np.zeros(len(centers), dtype=np.int32)
    per_view = []
    for pair in pairs:
        cam = load_camera(render_dir / "camera" / f"{pair.render_view_id:03d}.npy")
        info = residual_maps[pair.render_view_id]
        residual = info["residual"]
        valid_mask = info["valid_mask"]
        target_w, target_h = info["target_size"]
        render_w, render_h = info["render_size"]
        x, y, z = project_points_ortho(points, cam)
        px = np.rint(x * float(target_w) / float(render_w)).astype(np.int32)
        py = np.rint(y * float(target_h) / float(render_h)).astype(np.int32)
        in_frame = (px >= 0) & (px < target_w) & (py >= 0) & (py < target_h)
        fg = np.zeros_like(in_frame)
        fg[in_frame] = valid_mask[py[in_frame], px[in_frame]]
        visible = np.zeros_like(in_frame)
        if np.any(fg):
            pix_key = py[fg].astype(np.int64) * int(target_w) + px[fg].astype(np.int64)
            z_fg = z[fg]
            order = np.lexsort((z_fg, pix_key))
            sorted_key = pix_key[order]
            sorted_z = z_fg[order]
            _, first = np.unique(sorted_key, return_index=True)
            starts = np.r_[first, len(sorted_z)]
            min_z = np.empty_like(sorted_z)
            for a, b in zip(starts[:-1], starts[1:]):
                min_z[a:b] = sorted_z[a]
            front_sorted = sorted_z <= min_z + depth_eps
            fg_indices = np.flatnonzero(fg)
            visible[fg_indices[order[front_sorted]]] = True
        vals = np.zeros_like(residual_3d)
        vals[visible] = residual[py[visible], px[visible]]
        residual_3d = np.maximum(residual_3d, vals)
        visible_count += visible.astype(np.int32)
        per_view.append({
            "target_id": pair.target_id,
            "render_view_id": pair.render_view_id,
            "pred_path": info["pred_path"],
            "target_path": info["target_path"],
            "visible_voxels": int(visible.sum()),
            "mean_visible_residual": float(vals[visible].mean()) if np.any(visible) else 0.0,
            "max_visible_residual": float(vals[visible].max()) if np.any(visible) else 0.0,
        })
    return residual_3d, visible_count, {"scale": scale, "offset": offset.tolist(), "per_view": per_view}


def _score_positive_mask(scores: np.ndarray, domain_mask: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    valid = np.asarray(domain_mask, dtype=bool) & np.isfinite(scores)
    return valid & (scores > float(threshold))


def compute_front_visibility_by_view(
    centers: np.ndarray,
    pairs: Sequence[ViewPair],
    render_dir: Path,
    depth_eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return binary front visibility V_i(p) for each selected current render view."""
    scale, offset = render_transform_from_meta(render_dir)
    points = centers * scale + offset[None, :]
    visibility = np.zeros((len(pairs), len(centers)), dtype=np.float32)
    per_view = []
    for view_idx, pair in enumerate(pairs):
        cam = load_camera(render_dir / "camera" / f"{pair.render_view_id:03d}.npy")
        pred_path = render_dir / "image_ori" / f"{pair.render_view_id:03d}.png"
        render_w, render_h = Image.open(pred_path).size
        valid_mask = alpha_mask(pred_path, (render_w, render_h))
        if valid_mask is None:
            valid_mask = np.ones((render_h, render_w), dtype=bool)
        x, y, z = project_points_ortho(points, cam)
        px = np.rint(x).astype(np.int32)
        py = np.rint(y).astype(np.int32)
        in_frame = (px >= 0) & (px < render_w) & (py >= 0) & (py < render_h)
        fg = np.zeros_like(in_frame)
        fg[in_frame] = valid_mask[py[in_frame], px[in_frame]]
        visible = np.zeros_like(in_frame)
        if np.any(fg):
            pix_key = py[fg].astype(np.int64) * int(render_w) + px[fg].astype(np.int64)
            z_fg = z[fg]
            order = np.lexsort((z_fg, pix_key))
            sorted_key = pix_key[order]
            sorted_z = z_fg[order]
            _, first = np.unique(sorted_key, return_index=True)
            starts = np.r_[first, len(sorted_z)]
            min_z = np.empty_like(sorted_z)
            for a, b in zip(starts[:-1], starts[1:]):
                min_z[a:b] = sorted_z[a]
            front_sorted = sorted_z <= min_z + depth_eps
            fg_indices = np.flatnonzero(fg)
            visible[fg_indices[order[front_sorted]]] = True
        visibility[view_idx, visible] = 1.0
        per_view.append({
            "target_id": pair.target_id,
            "render_view_id": int(pair.render_view_id),
            "visible_voxels": int(visible.sum()),
        })
    return visibility, {"scale": scale, "offset": offset.tolist(), "per_view": per_view}


def compute_visibility_baseline(visibility_by_view: np.ndarray, domain_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    max_visibility = visibility_by_view.max(axis=0) if visibility_by_view.size else np.zeros_like(domain_mask, dtype=np.float32)
    visibility_shortage = 1.0 - max_visibility.astype(np.float32)
    visibility_shortage[~domain_mask.astype(bool)] = 0.0
    edit_mask = _score_positive_mask(visibility_shortage, domain_mask.astype(bool), threshold=0.5)
    meta = {
        "definition": "S_vis(p)=1-max_i V_i(p), where V_i is binary front visibility in the selected current render view i; edit mask is S_vis>0.5.",
        "n_views": int(visibility_by_view.shape[0]) if visibility_by_view.ndim == 2 else 0,
        "edit_mask_threshold": 0.5,
        "n_edit_voxels": int(edit_mask.sum()),
    }
    return visibility_shortage.astype(np.float32), edit_mask, meta


def load_support_chi(path: Path, key: str, n_voxels: int) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if key not in data.files:
            raise KeyError(f"Missing key {key!r} in {path}; available keys: {data.files}")
        chi = np.asarray(data[key], dtype=np.float32)
    else:
        chi = np.asarray(data, dtype=np.float32)
    if chi.ndim == 1:
        chi = chi[None, :]
    if chi.ndim != 2 or chi.shape[1] != n_voxels:
        raise ValueError(f"Expected chi array shape (n_views, {n_voxels}) or ({n_voxels},), got {chi.shape} from {path}")
    return np.clip(chi.astype(np.float32), 0.0, 1.0)


def compute_support_shortage_baseline(
    chi_by_view: np.ndarray,
    domain_mask: np.ndarray,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    aggregation = aggregation.lower().strip()
    if aggregation == "max":
        support = chi_by_view.max(axis=0) if chi_by_view.size else np.zeros_like(domain_mask, dtype=np.float32)
        definition = "S_support(p)=1-max_i chi_i(p)."
    elif aggregation == "mean":
        support = chi_by_view.mean(axis=0) if chi_by_view.size else np.zeros_like(domain_mask, dtype=np.float32)
        definition = "S_support(p)=1-mean_i chi_i(p)."
    else:
        raise ValueError(f"Unsupported support aggregation: {aggregation}")
    support_shortage = 1.0 - support.astype(np.float32)
    support_shortage[~domain_mask.astype(bool)] = 0.0
    edit_mask = _score_positive_mask(support_shortage, domain_mask.astype(bool), threshold=1e-6)
    meta = {
        "definition": definition + " Here chi_i(p) is implemented as binary front-visible geometric support in current render view i; edit mask is S_support>0.",
        "aggregation": aggregation,
        "n_views": int(chi_by_view.shape[0]) if chi_by_view.ndim == 2 else 0,
        "edit_mask_threshold": 1e-6,
        "n_edit_voxels": int(edit_mask.sum()),
    }
    return support_shortage.astype(np.float32), edit_mask, meta


def export_scalar_heatmap_glb(centers: np.ndarray, values: np.ndarray, path: Path, value_range: tuple[float, float] | None = None) -> None:
    ensure_dir(path.parent)
    voxel_size = 1.0 / 64.0
    base = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    mesh = trimesh.util.concatenate([base.copy().apply_translation(p) for p in centers])
    vals = values.astype(np.float32)
    v_min = float(vals.min()) if value_range is None else value_range[0]
    v_max = float(vals.max()) if value_range is None else value_range[1]
    norm = np.clip((vals - v_min) / max(v_max - v_min, 1e-6), 0.0, 1.0)
    rgb = (LOW_COLOR[None, :] * (1.0 - norm[:, None]) + HIGH_COLOR[None, :] * norm[:, None]).clip(0, 255).astype(np.uint8)
    colors = np.concatenate([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)], axis=1)
    mesh.visual.vertex_colors = np.repeat(colors, len(base.vertices), axis=0)
    mesh.export(str(path))


def export_binary_glb(centers: np.ndarray, mask: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    voxel_size = 1.0 / 64.0
    base = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    mesh = trimesh.util.concatenate([base.copy().apply_translation(p) for p in centers])
    colors = np.where(mask[:, None], CONFLICT_COLOR[None, :], NON_CONFLICT_COLOR[None, :]).astype(np.uint8)
    mesh.visual.vertex_colors = np.repeat(colors, len(base.vertices), axis=0)
    mesh.export(str(path))


def metric_values(scores: np.ndarray, labels: np.ndarray, residual: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(scores) & np.isfinite(residual)
    scores = scores[finite].astype(np.float64)
    residual = residual[finite].astype(np.float64)
    labels = labels[finite].astype(np.int32)
    out = {"spearman": None, "auroc": None, "auprc": None, "top10_recall": None}
    if len(scores) >= 2 and np.std(scores) > 0 and np.std(residual) > 0:
        corr = spearmanr(scores, residual).correlation
        out["spearman"] = float(corr) if corr is not None and math.isfinite(float(corr)) else None
    if len(labels) and labels.min() != labels.max():
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        k = max(1, int(math.ceil(0.10 * len(scores))))
        top_idx = np.argsort(-scores)[:k]
        out["top10_recall"] = float(labels[top_idx].sum() / max(int(labels.sum()), 1))
    return out


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_glb_high_detail(
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
    ortho_scale: float,
    local_lib_dir: Path | None,
    runtime_prefix: Path | None,
    use_xvfb: bool,
    xvfb_display: str,
    force: bool,
) -> None:
    image_dir = output_dir / "image_ori"
    expected = [image_dir / f"{int(v) if str(v).isdigit() else i:03d}.png" for i, v in enumerate(view_ids)]
    if not force and all(p.exists() and p.stat().st_size > 0 for p in expected):
        return
    ensure_dir(output_dir)
    if not blender.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    if not render_script.exists():
        raise FileNotFoundError(f"Render script not found: {render_script}")
    extra_ld = render_eval.blender_library_dirs(blender, local_lib_dir=local_lib_dir, runtime_prefix=runtime_prefix)
    missing = render_eval.blender_missing_libraries(blender, extra_ld=extra_ld)
    if missing:
        raise RuntimeError("Blender is missing dynamic libraries: " + ", ".join(missing))
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
        str(ortho_scale),
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
    if use_xvfb:
        xvfb_bin = shutil.which("Xvfb", path=env.get("PATH"))
        if xvfb_bin is None:
            raise RuntimeError("Xvfb requested but not found. Install xorg-xvfb-server or pass --no_xvfb.")
        env["DISPLAY"] = xvfb_display
        with (output_dir / "render_xvfb.log").open("w", encoding="utf-8") as xvfb_log:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_dir", type=Path, default=DEFAULT_ITEM_DIR)
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--output_glb", type=Path, default=None)
    ap.add_argument("--target_dir", type=Path, default=None)
    ap.add_argument("--view_pairs", type=str, default="0:3,1:0,2:2")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--render_samples", type=int, default=128)
    ap.add_argument("--render_shrink", type=float, default=1.0)
    ap.add_argument("--render_ortho_scale", type=float, default=1.2)
    ap.add_argument("--grid_size", type=int, default=64)
    ap.add_argument("--residual_source", type=str, default="vlm", choices=["vlm", "pixel"])
    ap.add_argument("--residual_threshold", type=str, default="q90")
    ap.add_argument("--support_aggregation", type=str, default="mean", choices=["max", "mean"])
    ap.add_argument("--support_chi_npz", type=Path, default=None, help="Optional NPZ/NPY containing per-view chi_i(p) support values aligned to active voxels. If omitted, front visibility is used as geometric chi_i.")
    ap.add_argument("--support_chi_key", type=str, default="chi_by_view", help="Array key used when --support_chi_npz points to an NPZ file.")
    ap.add_argument("--depth_eps", type=float, default=1.0 / 64.0)
    ap.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_MODEL)
    ap.add_argument("--vlm_fallback_models", type=str, default=DEFAULT_VLM_FALLBACK_MODELS)
    ap.add_argument("--vlm_api_base", type=str, default=DEFAULT_MAGIC666_BASE_URL)
    ap.add_argument("--vlm_api_key", type=str, default=DEFAULT_MAGIC666_API_KEY)
    ap.add_argument("--vlm_temperature", type=float, default=0.0)
    ap.add_argument("--vlm_max_tokens", type=int, default=2048)
    ap.add_argument("--vlm_timeout", type=int, default=180)
    ap.add_argument("--force_vlm", action="store_true")
    ap.add_argument("--blender", type=Path, default=render_eval.DEFAULT_BLENDER)
    ap.add_argument("--render_script", type=Path, default=DEFAULT_HIGH_DETAIL_RENDER_SCRIPT)
    ap.add_argument("--blender_runtime_prefix", type=Path, default=DEFAULT_RUNTIME_PREFIX if DEFAULT_RUNTIME_PREFIX.exists() else None)
    ap.add_argument("--xvfb_display", type=str, default=":119")
    ap.add_argument("--no_xvfb", action="store_true")
    ap.add_argument("--force_render", action="store_true")
    args = ap.parse_args()

    item_dir = args.item_dir.resolve()
    run_dir = args.run_dir.resolve()
    work_dir = args.work_dir.resolve()
    default_out_name = "diagnosis_disagreement_eval_rider_high_detail8_vlm" if args.residual_source == "vlm" else "diagnosis_disagreement_eval_rider_high_detail8_pixel"
    out_dir = (args.output_dir or (run_dir / default_out_name)).resolve()
    ensure_dir(out_dir)
    output_glb = (args.output_glb or (run_dir / "out" / "target_1_rgba.glb")).resolve()
    if not output_glb.exists():
        output_glb = (run_dir / "features" / "vox_edit.glb").resolve()
    if not output_glb.exists():
        raise FileNotFoundError(f"Could not locate output GLB under {run_dir}")
    target_dir = (args.target_dir or item_dir).resolve()
    fallback_target_dir = work_dir / "images"
    pairs = parse_pairs(args.view_pairs)
    vlm_models = []
    for name in [args.vlm_model] + [x.strip() for x in args.vlm_fallback_models.split(",") if x.strip()]:
        if name and name not in vlm_models:
            vlm_models.append(name)
    view_ids = [str(p.render_view_id) for p in pairs]
    azimuths, elevations = [], []
    for p in pairs:
        try:
            azimuth, elevation = DEFAULT_VIEWS[p.render_view_id]
        except IndexError as exc:
            raise ValueError(f"render_view_id={p.render_view_id} is outside the 8-view preset") from exc
        azimuths.append(float(azimuth))
        elevations.append(float(elevation))

    render_dir = out_dir / "renders"
    render_glb_high_detail(
        glb=output_glb,
        output_dir=render_dir,
        view_ids=view_ids,
        azimuths=azimuths,
        elevations=elevations,
        blender=args.blender,
        render_script=args.render_script,
        resolution=args.resolution,
        samples=args.render_samples,
        shrink=args.render_shrink,
        ortho_scale=args.render_ortho_scale,
        local_lib_dir=out_dir / "_blender_stub_libs",
        runtime_prefix=args.blender_runtime_prefix,
        use_xvfb=not args.no_xvfb,
        xvfb_display=args.xvfb_display,
        force=args.force_render,
    )

    residual_maps = compute_residual_maps(
        pairs=pairs,
        render_dir=render_dir,
        target_dir=target_dir,
        fallback_dir=fallback_target_dir,
        out_dir=out_dir / "residual_2d_vlm" if args.residual_source == "vlm" else out_dir / "residual_2d_pixel",
        residual_source=args.residual_source,
        vlm_models=vlm_models,
        vlm_api_base=args.vlm_api_base,
        vlm_api_key=args.vlm_api_key,
        vlm_temperature=args.vlm_temperature,
        vlm_max_tokens=args.vlm_max_tokens,
        vlm_timeout=args.vlm_timeout,
        force_vlm=args.force_vlm,
    )
    disagreement = load_disagreement(run_dir, args.grid_size)
    centers = disagreement["centers"].astype(np.float32)
    scores = disagreement["scores"].astype(np.float32)
    conflict = disagreement["conflict"].astype(bool)
    edit_magnitude_signal = load_edit_magnitude_conflict(run_dir, args.grid_size, centers)
    edit_magnitude_score = edit_magnitude_signal["scores"].astype(np.float32) if edit_magnitude_signal is not None else None
    edit_magnitude_mask = edit_magnitude_signal["conflict"].astype(bool) if edit_magnitude_signal is not None else None
    residual_3d, visible_count, proj_meta = residual_to_voxels(centers, pairs, render_dir, residual_maps, args.depth_eps)
    eval_mask = visible_count > 0
    baseline_eval_mask = np.ones(len(centers), dtype=bool)
    if args.residual_threshold.lower().startswith("q"):
        q = float(args.residual_threshold[1:]) / 100.0
        threshold = float(np.quantile(residual_3d[eval_mask], q)) if np.any(eval_mask) else float("inf")
        threshold_mode = args.residual_threshold
    else:
        threshold = float(args.residual_threshold)
        threshold_mode = "manual"
    labels = (residual_3d > threshold) & eval_mask

    visibility_by_view, visibility_proj_meta = compute_front_visibility_by_view(centers, pairs, render_dir, args.depth_eps)
    if args.support_chi_npz is not None:
        chi_by_view = load_support_chi(args.support_chi_npz.resolve(), args.support_chi_key, len(centers))
        support_chi_source = str(args.support_chi_npz.resolve())
    else:
        chi_by_view = visibility_by_view
        support_chi_source = "front_visibility_by_view_fallback"

    metrics_score = metric_values(scores[eval_mask], labels[eval_mask], residual_3d[eval_mask])
    metrics_conflict = metric_values(conflict[eval_mask].astype(np.float32), labels[eval_mask], residual_3d[eval_mask])
    metrics_edit_magnitude = None
    metrics_edit_magnitude_mask = None
    if edit_magnitude_score is not None and edit_magnitude_mask is not None:
        metrics_edit_magnitude = metric_values(edit_magnitude_score[eval_mask], labels[eval_mask], residual_3d[eval_mask])
        metrics_edit_magnitude_mask = metric_values(edit_magnitude_mask[eval_mask].astype(np.float32), labels[eval_mask], residual_3d[eval_mask])
    visibility_score, visibility_edit_mask, visibility_meta = compute_visibility_baseline(
        visibility_by_view=visibility_by_view,
        domain_mask=baseline_eval_mask,
    )
    support_score, support_edit_mask, support_meta = compute_support_shortage_baseline(
        chi_by_view=chi_by_view,
        domain_mask=baseline_eval_mask,
        aggregation=args.support_aggregation,
    )
    support_meta["chi_source"] = support_chi_source
    metrics_visibility = metric_values(visibility_score[baseline_eval_mask], labels[baseline_eval_mask], residual_3d[baseline_eval_mask])
    metrics_support = metric_values(support_score[baseline_eval_mask], labels[baseline_eval_mask], residual_3d[baseline_eval_mask])
    metrics_visibility_mask = metric_values(visibility_edit_mask[baseline_eval_mask].astype(np.float32), labels[baseline_eval_mask], residual_3d[baseline_eval_mask])
    metrics_support_mask = metric_values(support_edit_mask[baseline_eval_mask].astype(np.float32), labels[baseline_eval_mask], residual_3d[baseline_eval_mask])
    residual_label_name = "VLM edit-region score" if args.residual_source == "vlm" else "pixel RGB residual"
    rows = [
        {"signal": "Visibility shortage score", "residual_source": residual_label_name, **metrics_visibility, "n_eval_voxels": int(baseline_eval_mask.sum()), "n_positive_residual_voxels": int(labels.sum())},
        {"signal": "Visibility shortage edit mask", "residual_source": residual_label_name, **metrics_visibility_mask, "n_eval_voxels": int(baseline_eval_mask.sum()), "n_positive_residual_voxels": int(labels.sum())},
        {"signal": f"View support shortage score ({args.support_aggregation})", "residual_source": residual_label_name, **metrics_support, "n_eval_voxels": int(baseline_eval_mask.sum()), "n_positive_residual_voxels": int(labels.sum())},
        {"signal": f"View support shortage edit mask ({args.support_aggregation})", "residual_source": residual_label_name, **metrics_support_mask, "n_eval_voxels": int(baseline_eval_mask.sum()), "n_positive_residual_voxels": int(labels.sum())},
    ]
    if metrics_edit_magnitude is not None and metrics_edit_magnitude_mask is not None:
        rows.extend([
            {"signal": "Edit magnitude heatmap score", "residual_source": residual_label_name, **metrics_edit_magnitude, "n_eval_voxels": int(eval_mask.sum()), "n_positive_residual_voxels": int(labels.sum())},
            {"signal": "Edit magnitude-only conflict mask", "residual_source": residual_label_name, **metrics_edit_magnitude_mask, "n_eval_voxels": int(eval_mask.sum()), "n_positive_residual_voxels": int(labels.sum())},
        ])
    rows.extend([
        {"signal": "Disagreement D^r heatmap score", "residual_source": residual_label_name, **metrics_score, "n_eval_voxels": int(eval_mask.sum()), "n_positive_residual_voxels": int(labels.sum())},
        {"signal": "Conflict mask binary score", "residual_source": residual_label_name, **metrics_conflict, "n_eval_voxels": int(eval_mask.sum()), "n_positive_residual_voxels": int(labels.sum())},
    ])
    write_csv(out_dir / "metrics.csv", rows)
    npz_path = out_dir / "voxel_residual_and_disagreement.npz"
    np.savez_compressed(
        npz_path,
        centers=centers,
        indices=disagreement["indices"].astype(np.int32),
        disagreement_score=scores,
        conflict_flag=conflict.astype(np.uint8),
        residual_3d=residual_3d,
        residual_label=labels.astype(np.uint8),
        eval_mask=eval_mask.astype(np.uint8),
        baseline_eval_mask=baseline_eval_mask.astype(np.uint8),
        visible_count=visible_count,
        visibility_by_view=visibility_by_view.astype(np.float32),
        support_chi_by_view=chi_by_view.astype(np.float32),
        visibility_shortage_score=visibility_score,
        visibility_edit_mask=visibility_edit_mask.astype(np.uint8),
        support_shortage_score=support_score,
        support_edit_mask=support_edit_mask.astype(np.uint8),
        edit_magnitude_score=(edit_magnitude_score if edit_magnitude_score is not None else np.zeros_like(scores, dtype=np.float32)),
        edit_magnitude_conflict_flag=(edit_magnitude_mask.astype(np.uint8) if edit_magnitude_mask is not None else np.zeros_like(conflict, dtype=np.uint8)),
    )
    residual_vis = residual_3d.copy()
    residual_vis[~eval_mask] = 0.0
    export_scalar_heatmap_glb(centers, residual_vis, out_dir / "residual_3d_heatmap.glb", (0.0, max(float(residual_vis.max()), 1e-6)))
    export_binary_glb(centers, labels, out_dir / "residual_3d_mask.glb")
    export_binary_glb(centers, eval_mask, out_dir / "eval_visible_voxels.glb")
    export_scalar_heatmap_glb(centers, visibility_score, out_dir / "baseline_visibility_shortage_heatmap.glb", (0.0, 1.0))
    export_binary_glb(centers, visibility_edit_mask, out_dir / "baseline_visibility_shortage_edit_mask.glb")
    export_scalar_heatmap_glb(centers, support_score, out_dir / "baseline_view_support_shortage_heatmap.glb", (0.0, 1.0))
    export_binary_glb(centers, support_edit_mask, out_dir / "baseline_view_support_shortage_edit_mask.glb")
    if edit_magnitude_score is not None and edit_magnitude_mask is not None:
        edit_mag_vis = edit_magnitude_score.copy()
        edit_mag_vis[~eval_mask] = 0.0
        edit_mag_vmax = max(float(edit_mag_vis[eval_mask].max()) if np.any(eval_mask) else 0.0, 1e-6)
        export_scalar_heatmap_glb(centers, edit_mag_vis, out_dir / "baseline_edit_magnitude_heatmap.glb", (0.0, edit_mag_vmax))
        export_binary_glb(centers, edit_magnitude_mask, out_dir / "baseline_edit_magnitude_conflict_mask.glb")
    summary = {
        "item_dir": str(item_dir),
        "run_dir": str(run_dir),
        "work_dir": str(work_dir),
        "output_glb": str(output_glb),
        "target_dir": str(target_dir),
        "render_dir": str(render_dir),
        "view_pairs": [p.__dict__ for p in pairs],
        "azimuths": azimuths,
        "elevations": elevations,
        "render_script": str(args.render_script),
        "render_resolution": int(args.resolution),
        "render_samples": int(args.render_samples),
        "render_ortho_scale": float(args.render_ortho_scale),
        "blender_runtime_prefix": str(args.blender_runtime_prefix) if args.blender_runtime_prefix else None,
        "residual_source": args.residual_source,
        "residual_label_name": residual_label_name,
        "vlm_model": args.vlm_model if args.residual_source == "vlm" else None,
        "vlm_model_fallbacks": vlm_models if args.residual_source == "vlm" else None,
        "vlm_used_models": sorted({str(info.get("model", "")) for info in residual_maps.values() if info.get("source") == "vlm" and info.get("model", "")}),
        "vlm_api_base": args.vlm_api_base if args.residual_source == "vlm" else None,
        "vlm_prompt_focus": "Prioritize high-impact editable semantic differences, especially hands/fingers, arms, face/head, clothing, accessories, shoes/feet, held objects, silhouette, and distinctive material/color/texture blocks; ignore alignment and rendering-style noise.",
        "vlm_decisions": [
            {
                "target_id": info.get("target_id"),
                "render_view_id": int(view_id),
                "decision_json": info.get("vlm_decision_json"),
                "decision_raw": info.get("vlm_decision_raw"),
                "regions": info.get("vlm_regions", []),
                "visualizations": info.get("vlm_visualizations", {}),
            }
            for view_id, info in residual_maps.items()
            if info.get("source") == "vlm"
        ],
        "residual_threshold": threshold,
        "residual_threshold_mode": threshold_mode,
        "support_aggregation": args.support_aggregation,
        "support_chi_source": support_chi_source,
        "baseline_definitions": {
            "visibility": visibility_meta,
            "view_support": support_meta,
        },
        "n_active_voxels_total": int(len(centers)),
        "n_original_mask_all_voxels_from_stats": int(disagreement["stats"].get("num_active_voxels_in_mask_all", 0)),
        "n_eval_visible_voxels": int(eval_mask.sum()),
        "n_baseline_eval_voxels": int(baseline_eval_mask.sum()),
        "n_positive_residual_voxels": int(labels.sum()),
        "n_conflict_voxels_total": int(conflict.sum()),
        "n_conflict_voxels_eval": int((conflict & eval_mask).sum()),
        "n_edit_magnitude_conflict_voxels_total": int(edit_magnitude_mask.sum()) if edit_magnitude_mask is not None else None,
        "n_edit_magnitude_conflict_voxels_eval": int((edit_magnitude_mask & eval_mask).sum()) if edit_magnitude_mask is not None else None,
        "residual_3d_stats_eval": {
            "min": float(residual_3d[eval_mask].min()) if np.any(eval_mask) else 0.0,
            "mean": float(residual_3d[eval_mask].mean()) if np.any(eval_mask) else 0.0,
            "max": float(residual_3d[eval_mask].max()) if np.any(eval_mask) else 0.0,
            "p50": float(np.quantile(residual_3d[eval_mask], 0.50)) if np.any(eval_mask) else 0.0,
            "p90": float(np.quantile(residual_3d[eval_mask], 0.90)) if np.any(eval_mask) else 0.0,
            "p95": float(np.quantile(residual_3d[eval_mask], 0.95)) if np.any(eval_mask) else 0.0,
        },
        "metrics": rows,
        "projection": proj_meta,
        "disagreement_stats_path": str(disagreement["stats_path"]),
        "disagreement_heatmap_path": str(disagreement["heatmap_path"]),
        "conflict_mask_path": str(disagreement["mask_path"]),
        "edit_magnitude_stats_path": str(edit_magnitude_signal["stats_path"]) if edit_magnitude_signal is not None else None,
        "edit_magnitude_heatmap_path": str(edit_magnitude_signal["heatmap_path"]) if edit_magnitude_signal is not None else None,
        "edit_magnitude_conflict_mask_path": str(edit_magnitude_signal["mask_path"]) if edit_magnitude_signal is not None else None,
        "artifacts": {
            "metrics_csv": str(out_dir / "metrics.csv"),
            "voxel_npz": str(npz_path),
            "residual_2d_dir": str(out_dir / ("residual_2d_vlm" if args.residual_source == "vlm" else "residual_2d_pixel")),
            "residual_3d_heatmap_glb": str(out_dir / "residual_3d_heatmap.glb"),
            "residual_3d_mask_glb": str(out_dir / "residual_3d_mask.glb"),
            "eval_visible_voxels_glb": str(out_dir / "eval_visible_voxels.glb"),
            "baseline_visibility_shortage_heatmap_glb": str(out_dir / "baseline_visibility_shortage_heatmap.glb"),
            "baseline_visibility_shortage_edit_mask_glb": str(out_dir / "baseline_visibility_shortage_edit_mask.glb"),
            "baseline_view_support_shortage_heatmap_glb": str(out_dir / "baseline_view_support_shortage_heatmap.glb"),
            "baseline_view_support_shortage_edit_mask_glb": str(out_dir / "baseline_view_support_shortage_edit_mask.glb"),
            "baseline_edit_magnitude_heatmap_glb": str(out_dir / "baseline_edit_magnitude_heatmap.glb") if edit_magnitude_score is not None else None,
            "baseline_edit_magnitude_conflict_mask_glb": str(out_dir / "baseline_edit_magnitude_conflict_mask.glb") if edit_magnitude_mask is not None else None,
        },
        "notes": [
            "Disagreement scores are decoded from exported heatmap GLB colors using the stats JSON value range, so values are quantized approximations of D^r.",
            "The 3D residual target is max-pooled from the selected 2D residual source over the rendered/GT views with z-buffer visibility over active voxels.",
            "For --residual_source vlm, 2D residual maps are VLM-selected edit regions on the target image, intentionally ignoring small alignment, scale, render-style, lighting, and shadow differences.",
            "The exact mask_all membership is not stored in the exported heatmap GLB; D^r/conflict rows evaluate visible active voxels and report the original mask_all count from stats.",
            "Row 1 uses S_vis(p)=1-max_i V_i(p), where V_i is binary front visibility in the selected current render views; its edit mask is S_vis>0.5.",
            "Row 2 uses S_support(p)=1-aggregate_i chi_i(p), with aggregate selected by --support_aggregation. If --support_chi_npz is omitted, chi_i falls back to the same binary front-visible geometric support as V_i; its edit mask is S_support>0.",
            "Row 1 and Row 2 are evaluated over all exported active voxels so unobserved or weakly supported voxels are not removed before scoring.",
            "Row 3 uses S_mag(p)=mean_t ||v_tar_fused,t(p)||_2, and its edit mask is inside mask_all with S_mag above the edit-magnitude-only threshold; multi-view velocity disagreement is intentionally ignored.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()