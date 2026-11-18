#!/usr/bin/env python3
"""Plan and render the next view for mv_fusion rounds.

This implements the paper-style next-view acquisition step after a completed
mv_fusion round.  Given a round output directory, it reads the velocity-conflict
mask as the unresolved region, scores candidate cameras, and renders the next
view of the current edited GLB.

Implemented policies:
  - ours: unresolved-region coverage minus angular redundancy
  - random: deterministic random candidate
  - farthest_camera: maximum angular distance from existing views
  - whole_object_coverage: maximum visible active-object voxel coverage
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BLENDER_BIN, CHECKPOINT, MV_FUSION_DIR, REPO_ROOT  # noqa: E402
from paths import RENDER_GLB_MV_HIGH_DETAIL_SCRIPT as RENDER_GLB_MV_SCRIPT  # noqa: E402

EXPORT_FINAL_GLB_SCRIPT = MV_FUSION_DIR / "export_easy3e_final_glb_from_slat.py"
# Optional prefix with X11/OpenGL runtime libraries for Blender on bare servers (see README).
DEFAULT_RUNTIME_PREFIX = Path(os.environ.get("BLENDER_RUNTIME_PREFIX", REPO_ROOT / "blender_runtime_libs"))

DEFAULT_METHODS = ("ours", "random", "farthest_camera", "whole_object_coverage")
CONFLICT_COLOR = np.array([255, 80, 80], dtype=np.int16)


@dataclass(frozen=True)
class ViewCandidate:
    view_id: int
    azimuth: float
    elevation: float

    @property
    def key(self) -> str:
        return f"{self.azimuth:.6g},{self.elevation:.6g}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def blender_library_dirs(blender: Path, runtime_prefix: Path | None = None) -> list[Path]:
    dirs: list[Path] = []
    if runtime_prefix is not None:
        for cand in (
            runtime_prefix / "lib",
            runtime_prefix / "x86_64-conda-linux-gnu" / "sysroot" / "usr" / "lib64",
        ):
            if cand.exists():
                dirs.append(cand)
    for cand in (blender.parent / "lib", blender.parent / "lib" / "mesa"):
        if cand.exists():
            dirs.append(cand)
    return dirs


def blender_missing_libraries(blender: Path, extra_ld: Sequence[Path]) -> list[str]:
    env = os.environ.copy()
    ld_parts = [str(p) for p in extra_ld]
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
    return [line.strip().split()[0] for line in proc.stdout.splitlines() if "not found" in line]


def normalize_angle(angle: float) -> float:
    return float(angle) % 360.0


def direction_from_angles(azimuth: float, elevation: float) -> np.ndarray:
    az = math.radians(float(azimuth))
    el = math.radians(float(elevation))
    vec = np.array([
        math.cos(az) * math.cos(el),
        math.sin(az) * math.cos(el),
        math.sin(el),
    ], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-12)


def angular_distance_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    da = direction_from_angles(a[0], a[1])
    db = direction_from_angles(b[0], b[1])
    dot = float(np.clip(np.dot(da, db), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def parse_angle_list(value: str | None) -> list[tuple[float, float]]:
    if value is None or not value.strip():
        return []
    out: list[tuple[float, float]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        az, el = item.split(",", 1)
        out.append((normalize_angle(float(az)), float(el)))
    return out


def parse_candidates(value: str | None) -> list[ViewCandidate]:
    if value is None or not value.strip():
        return []
    out: list[ViewCandidate] = []
    for idx, item in enumerate(value.split(";")):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(",")]
        if len(parts) == 2:
            view_id = idx
            az, el = float(parts[0]), float(parts[1])
        elif len(parts) == 3:
            view_id = int(parts[0])
            az, el = float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"Invalid candidate view spec: {item!r}")
        out.append(ViewCandidate(view_id=view_id, azimuth=normalize_angle(az), elevation=float(el)))
    return out


def load_camera(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return data.item() if isinstance(data, np.ndarray) and data.shape == () else dict(data)


def default_candidate_views(work_dir: Path) -> list[ViewCandidate]:
    cam_dir = work_dir / "edit_views" / "camera"
    candidates: list[ViewCandidate] = []
    if cam_dir.exists():
        for path in sorted(cam_dir.glob("*.npy")):
            cam = load_camera(path)
            if "azimuth" not in cam or "elevation" not in cam:
                continue
            candidates.append(ViewCandidate(
                view_id=int(path.stem),
                azimuth=normalize_angle(float(cam["azimuth"])),
                elevation=float(cam["elevation"]),
            ))
    if candidates:
        return candidates
    return [ViewCandidate(i, normalize_angle(i * 22.5), 0.0) for i in range(16)]


def frange_inclusive(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    values: list[float] = []
    x = float(start)
    stop = float(stop)
    eps = abs(float(step)) * 1e-6
    while x <= stop + eps:
        values.append(round(float(x), 10))
        x += float(step)
    return values


def make_free_angle_grid(
    azimuth_min: float,
    azimuth_max: float,
    azimuth_step: float,
    elevation_min: float,
    elevation_max: float,
    elevation_step: float,
) -> list[ViewCandidate]:
    azimuths = frange_inclusive(azimuth_min, azimuth_max, azimuth_step)
    elevations = frange_inclusive(elevation_min, elevation_max, elevation_step)
    candidates: list[ViewCandidate] = []
    view_id = 0
    for elevation in elevations:
        for azimuth in azimuths:
            candidates.append(ViewCandidate(
                view_id=view_id,
                azimuth=normalize_angle(azimuth),
                elevation=float(elevation),
            ))
            view_id += 1
    return candidates


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def infer_existing_angles(summary: dict[str, Any], candidates: list[ViewCandidate], override: str | None) -> list[tuple[float, float]]:
    explicit = parse_angle_list(override)
    if explicit:
        return explicit
    by_id = {str(c.view_id): c for c in candidates}
    existing: list[tuple[float, float]] = []
    for view_name in summary.get("views", []) or []:
        cand = by_id.get(str(view_name))
        if cand is not None:
            existing.append((cand.azimuth, cand.elevation))
    if summary.get("new_view_angle"):
        try:
            az, el = str(summary["new_view_angle"]).split(",", 1)
            existing.append((normalize_angle(float(az)), float(el)))
        except Exception:
            pass
    # Preserve order while removing near-duplicates.
    dedup: list[tuple[float, float]] = []
    for angle in existing:
        if all(angular_distance_deg(angle, old) > 1e-3 for old in dedup):
            dedup.append(angle)
    return dedup


def load_scene_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = tuple(loaded.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry in {path}")
        return trimesh.util.concatenate(geoms)
    return loaded


def voxel_centers_and_colors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_scene_mesh(path)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if len(verts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.uint8)
    if len(verts) % 8 != 0:
        # Fallback for non-cube meshes: use vertices as points and colors if present.
        colors = getattr(mesh.visual, "vertex_colors", np.zeros((len(verts), 4), dtype=np.uint8))
        return verts, np.asarray(colors, dtype=np.uint8)
    centers = verts.reshape(-1, 8, 3).mean(axis=1)
    colors = np.asarray(getattr(mesh.visual, "vertex_colors", np.zeros((len(verts), 4), dtype=np.uint8)), dtype=np.uint8)
    if colors.shape[0] == len(verts):
        colors = colors.reshape(-1, 8, colors.shape[-1])[:, 0, :]
    else:
        colors = np.zeros((len(centers), 4), dtype=np.uint8)
    return centers.astype(np.float32), colors.astype(np.uint8)


def color_is_conflict(colors: np.ndarray) -> np.ndarray:
    if colors.size == 0 or colors.shape[-1] < 3:
        return np.zeros((colors.shape[0],), dtype=bool)
    rgb = colors[:, :3].astype(np.int16)
    return (rgb[:, 0] >= 220) & (rgb[:, 1] <= 130) & (rgb[:, 2] <= 130)


def grid_key(points: np.ndarray, grid_size: int = 64) -> np.ndarray:
    idx = np.round((points.astype(np.float64) + 0.5) * grid_size - 0.5).astype(np.int64)
    idx = np.clip(idx, 0, grid_size - 1)
    return idx[:, 0] * (grid_size * grid_size) + idx[:, 1] * grid_size + idx[:, 2]


def subtract_quiescent(unresolved: np.ndarray, quiescent_glb: Path | None) -> np.ndarray:
    if quiescent_glb is None or not quiescent_glb.exists() or unresolved.size == 0:
        return unresolved
    q_centers, _ = voxel_centers_and_colors(quiescent_glb)
    if q_centers.size == 0:
        return unresolved
    q_keys = set(grid_key(q_centers).tolist())
    keep = np.array([k not in q_keys for k in grid_key(unresolved).tolist()], dtype=bool)
    return unresolved[keep]


def find_colored_sparse_voxel_glb(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists() or explicit.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing non-empty colored sparse voxel GLB: {explicit}")
        return explicit
    preferred = [
        run_dir / "features" / "vox_edit.glb",
        run_dir / "out" / "vox_edit.glb",
    ]
    for path in preferred:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"Could not find colored sparse voxel GLB under {run_dir}")


def _infer_target_glb_name(summary: dict[str, Any]) -> str:
    primary = summary.get("primary_target")
    if primary:
        return f"{Path(str(primary)).stem}.glb"
    command = summary.get("command") or []
    if "--input_image" in command:
        idx = command.index("--input_image")
        if idx + 1 < len(command):
            return f"{Path(str(command[idx + 1])).stem}.glb"
    return "target_final.glb"


def find_or_export_target_glb(
    run_dir: Path,
    summary: dict[str, Any],
    explicit: Path | None,
    checkpoint: str,
    texture_size: int,
    force_export: bool,
) -> Path:
    if explicit is not None:
        if explicit.exists() and explicit.stat().st_size > 0 and not force_export:
            return explicit
        output_glb = explicit
    else:
        out_dir = run_dir / "out"
        candidates = sorted(out_dir.glob("target*_rgba.glb"), key=lambda p: p.stat().st_mtime)
        candidates += sorted(out_dir.glob("*_rgba.glb"), key=lambda p: p.stat().st_mtime)
        candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
        if candidates and not force_export:
            return candidates[-1]
        output_glb = out_dir / _infer_target_glb_name(summary)

    if output_glb.exists() and output_glb.stat().st_size > 0 and not force_export:
        return output_glb

    slat_path = run_dir / "features" / "slat_edited.pt"
    if not slat_path.exists() or slat_path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Could not find a non-empty final target GLB under {run_dir / 'out'}, "
            f"and missing SLAT fallback: {slat_path}"
        )
    if not EXPORT_FINAL_GLB_SCRIPT.exists():
        raise FileNotFoundError(f"Missing final target exporter: {EXPORT_FINAL_GLB_SCRIPT}")
    ensure_dir(output_glb.parent)
    cmd = [
        sys.executable,
        "-u",
        str(EXPORT_FINAL_GLB_SCRIPT),
        "--slat",
        str(slat_path),
        "--output-glb",
        str(output_glb),
        "--checkpoint",
        str(checkpoint),
        "--texture-size",
        str(texture_size),
        "--force",
    ]
    export_env = os.environ.copy()
    export_env.setdefault("SPCONV_ALGO", "native")
    export_env.setdefault("ATTN_BACKEND", "sdpa")
    export_env.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    log_path = output_glb.parent / f"{output_glb.stem}_export_from_slat.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=export_env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not output_glb.exists() or output_glb.stat().st_size <= 0:
        raise RuntimeError(f"Failed to export final target GLB from SLAT; see {log_path}")
    return output_glb


def find_conflict_mask(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Missing conflict mask: {explicit}")
        return explicit
    out_dir = run_dir / "out"
    masks = sorted(out_dir.glob("velocity_conflict_*_mask.glb"), key=lambda p: p.stat().st_mtime)
    if masks:
        return masks[-1]
    feature_dir = run_dir / "features" / "mv_fusion_voxel_diff"
    masks = sorted(feature_dir.glob("velocity_conflict_*_mask.glb"), key=lambda p: p.stat().st_mtime)
    if masks:
        return masks[-1]
    raise FileNotFoundError(f"Could not find velocity_conflict_*_mask.glb under {run_dir}")


def find_quiescent_glb(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for path in (
        run_dir / "out" / "quiescent_voxels_only.glb",
        run_dir / "features" / "mv_fusion_voxel_diff" / "quiescent_voxels_only.glb",
    ):
        if path.exists():
            return path
    return None


def camera_visibility(points: np.ndarray, candidate: ViewCandidate, resolution: int, ortho_scale: float, depth_eps: float,
                      footprint: float = 1.0 / 64.0) -> np.ndarray:
    """Front visibility of voxel centers under an orthographic camera.

    Each voxel occludes its whole projected footprint (side `footprint`), not
    just the pixel of its center; otherwise at resolution 256 / ortho 1.2 a
    voxel covers ~3x3 pixels and voxels behind it would leak through.
    """
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    cam_dir = direction_from_angles(candidate.azimuth, candidate.elevation)
    right = np.array([-cam_dir[1], cam_dir[0], 0.0], dtype=np.float64)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, cam_dir)
    up /= max(float(np.linalg.norm(up)), 1e-12)

    pts = points.astype(np.float64)
    x = pts @ right
    y = pts @ up
    z = pts @ cam_dir
    half = float(ortho_scale) * 0.5
    in_frame = (np.abs(x) <= half) & (np.abs(y) <= half)
    visible = np.zeros((len(points),), dtype=bool)
    if not np.any(in_frame):
        return visible
    px = np.floor((x[in_frame] / float(ortho_scale) + 0.5) * (resolution - 1)).astype(np.int64)
    py = np.floor((y[in_frame] / float(ortho_scale) + 0.5) * (resolution - 1)).astype(np.int64)
    z_fg = z[in_frame]
    rad = int(np.ceil(0.5 * float(footprint) / float(ortho_scale) * (resolution - 1)))
    zbuf = np.full(int(resolution) * int(resolution), -np.inf)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            qx, qy = px + dx, py + dy
            ok = (qx >= 0) & (qx < resolution) & (qy >= 0) & (qy < resolution)
            np.maximum.at(zbuf, qy[ok] * int(resolution) + qx[ok], z_fg[ok])
    front = z_fg >= zbuf[py * int(resolution) + px] - float(depth_eps)
    visible[np.flatnonzero(in_frame)[front]] = True
    return visible


def score_candidates(
    candidates: list[ViewCandidate],
    existing_angles: list[tuple[float, float]],
    object_points: np.ndarray,
    unresolved_points: np.ndarray,
    resolution: int,
    ortho_scale: float,
    depth_eps: float,
    rho: float,
    exclude_existing_eps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        min_dist = min((angular_distance_deg((cand.azimuth, cand.elevation), old) for old in existing_angles), default=180.0)
        max_overlap = 0.0
        if existing_angles:
            cdir = direction_from_angles(cand.azimuth, cand.elevation)
            max_overlap = max(max(0.0, float(np.dot(cdir, direction_from_angles(a, e)))) for a, e in existing_angles)
        is_existing = min_dist <= float(exclude_existing_eps)
        # Unresolved voxels must be occluded by the whole object, so z-buffer both sets together.
        vis_all = camera_visibility(np.concatenate([object_points, unresolved_points]), cand, resolution, ortho_scale, depth_eps)
        vis_obj = vis_all[:len(object_points)]
        vis_unres = vis_all[len(object_points):]
        object_visible = int(vis_obj.sum())
        unresolved_visible = int(vis_unres.sum())
        object_ratio = float(object_visible / max(len(object_points), 1))
        unresolved_ratio = float(unresolved_visible / max(len(unresolved_points), 1))
        rows.append({
            "view_id": int(cand.view_id),
            "azimuth": float(cand.azimuth),
            "elevation": float(cand.elevation),
            "is_existing": bool(is_existing),
            "min_angular_distance_to_existing_deg": float(min_dist),
            "max_angular_overlap_with_existing": float(max_overlap),
            "object_visible_voxels": object_visible,
            "object_visible_ratio": object_ratio,
            "unresolved_visible_voxels": unresolved_visible,
            "unresolved_visible_ratio": unresolved_ratio,
            "ours_score": float(unresolved_ratio - float(rho) * max_overlap),
            "farthest_score": float(min_dist),
            "whole_object_coverage_score": object_ratio,
        })
    return rows


def choose_best(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    valid = [r for r in rows if not r["is_existing"]]
    if not valid:
        valid = rows
    return max(valid, key=lambda r: (float(r[score_key]), float(r["min_angular_distance_to_existing_deg"]), -int(r["view_id"])))


def select_views(rows: list[dict[str, Any]], methods: Iterable[str], seed: int) -> dict[str, dict[str, Any]]:
    valid = [r for r in rows if not r["is_existing"]]
    if not valid:
        valid = rows
    rng = random.Random(int(seed))
    selections: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method == "ours":
            choice = choose_best(rows, "ours_score")
        elif method == "random":
            choice = rng.choice(valid)
        elif method == "farthest_camera":
            choice = choose_best(rows, "farthest_score")
        elif method == "whole_object_coverage":
            choice = choose_best(rows, "whole_object_coverage_score")
        else:
            raise ValueError(f"Unknown method: {method}")
        selections[method] = dict(choice)
    return selections


def render_selected_views(
    glb: Path,
    output_dir: Path,
    selections: dict[str, dict[str, Any]],
    object_label: str,
    resolution: int,
    samples: int,
    shrink: float,
    ortho_scale: float,
    distance: float,
    engine: str,
    cycles_device: str,
    runtime_prefix: Path | None,
    use_xvfb: bool,
    xvfb_display: str,
) -> dict[str, str]:
    ensure_dir(output_dir)
    if not BLENDER_BIN.exists():
        raise FileNotFoundError(f"Missing Blender binary: {BLENDER_BIN}")
    if not RENDER_GLB_MV_SCRIPT.exists():
        raise FileNotFoundError(f"Missing render script: {RENDER_GLB_MV_SCRIPT}")
    extra_ld = blender_library_dirs(BLENDER_BIN, runtime_prefix=runtime_prefix)
    missing = blender_missing_libraries(BLENDER_BIN, extra_ld=extra_ld)
    if missing:
        raise RuntimeError("Blender is missing dynamic libraries: " + ", ".join(missing))

    unique: list[tuple[float, float]] = []
    method_to_render_id: dict[str, int] = {}
    for method, row in selections.items():
        angle = (float(row["azimuth"]), float(row["elevation"]))
        found = None
        for idx, old in enumerate(unique):
            if angular_distance_deg(angle, old) < 1e-4:
                found = idx
                break
        if found is None:
            found = len(unique)
            unique.append(angle)
        method_to_render_id[method] = found

    render_dir = output_dir / "blender" / object_label
    ensure_dir(render_dir)
    cmd = [
        str(BLENDER_BIN),
        "-b",
        "-P",
        str(RENDER_GLB_MV_SCRIPT),
        "--",
        "--object_path",
        str(glb),
        "--output_dir",
        str(render_dir),
        "--resolution",
        str(resolution),
        "--shrink",
        str(shrink),
        "--distance",
        str(distance),
        "--ortho_scale",
        str(ortho_scale),
        "--engine",
        str(engine),
        "--cycles_device",
        str(cycles_device),
        "--samples",
        str(samples),
        "--view_azimuths",
        ",".join(str(a) for a, _ in unique),
        "--view_elevations",
        ",".join(str(e) for _, e in unique),
        "--view_ids",
        ",".join(str(i) for i in range(len(unique))),
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
            raise RuntimeError("Xvfb requested but not found. Pass --no-xvfb or provide --blender-runtime-prefix with Xvfb.")
        env["DISPLAY"] = xvfb_display
        with (output_dir / f"render_{object_label}_xvfb.log").open("w", encoding="utf-8") as xvfb_log:
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
    log_path = output_dir / f"render_{object_label}_blender.log"
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
        raise RuntimeError(f"Blender render failed with code {proc.returncode}; see {log_path}")

    method_paths: dict[str, str] = {}
    named_dir = output_dir / "renders_by_method" / object_label
    ensure_dir(named_dir)
    for method, render_id in method_to_render_id.items():
        src = render_dir / "image_ori" / f"{render_id:03d}.png"
        dst = named_dir / f"{method}.png"
        if not src.exists():
            raise FileNotFoundError(f"Expected rendered view missing: {src}")
        shutil.copy2(src, dst)
        method_paths[method] = str(dst)
    return method_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan and render mv_fusion next-view acquisition policies.")
    ap.add_argument("--item-dir", type=Path, default=REPO_ROOT / "demo" / "rider")
    ap.add_argument("--run-dir", type=Path, default=None, help="Completed round output root, e.g. <item>/_easy3e_out_round1")
    ap.add_argument("--work-dir", type=Path, default=None, help="Completed round work root, e.g. <item>/_easy3e_work_round1")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--output-glb", type=Path, default=None,
                    help="Deprecated alias for --target-glb.")
    ap.add_argument("--sparse-voxel-glb", type=Path, default=None,
                    help="Colored sparse voxel GLB to render, default: <run>/features/vox_edit.glb.")
    ap.add_argument("--target-glb", type=Path, default=None,
                    help="Final generated target GLB to render, default: <run>/out/target*_rgba.glb or export from slat_edited.pt.")
    ap.add_argument("--target-export-checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--target-export-texture-size", type=int, default=2048)
    ap.add_argument("--force-target-export", action="store_true")
    ap.add_argument("--conflict-mask", type=Path, default=None)
    ap.add_argument("--quiescent-glb", type=Path, default=None)
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--candidate-views", default=None, help="Optional manual semicolon-separated candidate views: 'az,el;az,el' or 'id,az,el;...'. If omitted, use the free-angle grid below.")
    ap.add_argument("--candidate-azimuth-min", type=float, default=0.0,
                    help="Minimum azimuth in degrees for the default free-angle candidate grid.")
    ap.add_argument("--candidate-azimuth-max", type=float, default=355.0,
                    help="Maximum azimuth in degrees for the default free-angle candidate grid.")
    ap.add_argument("--candidate-azimuth-step", type=float, default=5.0,
                    help="Azimuth step in degrees for the default free-angle candidate grid.")
    ap.add_argument("--candidate-elevation-min", type=float, default=-30.0,
                    help="Minimum elevation in degrees for the default free-angle candidate grid.")
    ap.add_argument("--candidate-elevation-max", type=float, default=30.0,
                    help="Maximum elevation in degrees for the default free-angle candidate grid.")
    ap.add_argument("--candidate-elevation-step", type=float, default=5.0,
                    help="Elevation step in degrees for the default free-angle candidate grid.")
    ap.add_argument("--existing-view-angles", default=None, help="Semicolon-separated az,el list. If omitted, infer from run_summary view ids.")
    ap.add_argument("--random-seed", type=int, default=1)
    ap.add_argument("--rho", type=float, default=0.15, help="Redundancy penalty for ours: score=unresolved_visibility-rho*max_overlap.")
    ap.add_argument("--visibility-resolution", type=int, default=256)
    ap.add_argument("--depth-eps", type=float, default=1.0 / 64.0)
    ap.add_argument("--exclude-existing-eps-deg", type=float, default=1.0)
    ap.add_argument("--render-resolution", type=int, default=1024)
    ap.add_argument("--render-samples", type=int, default=128)
    ap.add_argument("--render-shrink", type=float, default=1.0)
    ap.add_argument("--render-ortho-scale", type=float, default=1.2)
    ap.add_argument("--render-distance", type=float, default=2.0)
    ap.add_argument("--render-engine", default="CYCLES")
    ap.add_argument("--cycles-device", default="CPU", choices=("CPU", "GPU"))
    ap.add_argument("--blender-runtime-prefix", type=Path,
                    default=DEFAULT_RUNTIME_PREFIX if DEFAULT_RUNTIME_PREFIX.exists() else None,
                    help="Optional prefix containing X11/OpenGL runtime libraries and Xvfb.")
    ap.add_argument("--no-xvfb", action="store_true", help="Do not launch Blender under Xvfb.")
    ap.add_argument("--xvfb-display", default=":123")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    item_dir = args.item_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else item_dir / f"_easy3e_out_round{args.round}"
    work_dir = args.work_dir.expanduser().resolve() if args.work_dir else item_dir / f"_easy3e_work_round{args.round}"
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / f"next_view_acquisition_round{args.round}"
    runtime_prefix = args.blender_runtime_prefix.expanduser().resolve() if args.blender_runtime_prefix else None
    ensure_dir(output_dir)

    summary = load_run_summary(run_dir)
    manual_candidates = parse_candidates(args.candidate_views)
    if manual_candidates:
        candidates = manual_candidates
        candidate_source: dict[str, Any] = {"mode": "manual", "spec": str(args.candidate_views)}
    else:
        candidates = make_free_angle_grid(
            azimuth_min=args.candidate_azimuth_min,
            azimuth_max=args.candidate_azimuth_max,
            azimuth_step=args.candidate_azimuth_step,
            elevation_min=args.candidate_elevation_min,
            elevation_max=args.candidate_elevation_max,
            elevation_step=args.candidate_elevation_step,
        )
        candidate_source = {
            "mode": "free_angle_grid",
            "azimuth_min": float(args.candidate_azimuth_min),
            "azimuth_max": float(args.candidate_azimuth_max),
            "azimuth_step": float(args.candidate_azimuth_step),
            "elevation_min": float(args.candidate_elevation_min),
            "elevation_max": float(args.candidate_elevation_max),
            "elevation_step": float(args.candidate_elevation_step),
        }
    if not candidates:
        raise RuntimeError("No candidate views available.")
    existing_angle_candidates = default_candidate_views(work_dir)
    existing_angles = infer_existing_angles(summary, existing_angle_candidates, args.existing_view_angles)

    conflict_mask = find_conflict_mask(run_dir, args.conflict_mask)
    quiescent_glb = find_quiescent_glb(run_dir, args.quiescent_glb)
    sparse_voxel_glb = find_colored_sparse_voxel_glb(run_dir, args.sparse_voxel_glb)
    target_glb_arg = args.target_glb or args.output_glb
    target_glb = find_or_export_target_glb(
        run_dir=run_dir,
        summary=summary,
        explicit=target_glb_arg,
        checkpoint=args.target_export_checkpoint,
        texture_size=args.target_export_texture_size,
        force_export=args.force_target_export,
    )

    object_points, colors = voxel_centers_and_colors(conflict_mask)
    conflict_flags = color_is_conflict(colors)
    unresolved_points = object_points[conflict_flags]
    unresolved_before = int(len(unresolved_points))
    unresolved_points = subtract_quiescent(unresolved_points, quiescent_glb)
    if len(unresolved_points) == 0:
        unresolved_points = object_points[conflict_flags]
    if len(object_points) == 0:
        raise RuntimeError(f"No active voxel centers found in {conflict_mask}")
    if len(unresolved_points) == 0:
        # Graceful fallback: no conflict left, so ours degenerates to object coverage.
        unresolved_points = object_points

    rows = score_candidates(
        candidates=candidates,
        existing_angles=existing_angles,
        object_points=object_points,
        unresolved_points=unresolved_points,
        resolution=args.visibility_resolution,
        ortho_scale=args.render_ortho_scale,
        depth_eps=args.depth_eps,
        rho=args.rho,
        exclude_existing_eps=args.exclude_existing_eps_deg,
    )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    selections = select_views(rows, methods, seed=args.random_seed)

    sparse_voxel_render_paths: dict[str, str] = {}
    target_render_paths: dict[str, str] = {}
    if not args.skip_render:
        sparse_voxel_render_paths = render_selected_views(
            glb=sparse_voxel_glb,
            output_dir=output_dir,
            selections=selections,
            object_label="sparse_voxel",
            resolution=args.render_resolution,
            samples=args.render_samples,
            shrink=args.render_shrink,
            ortho_scale=args.render_ortho_scale,
            distance=args.render_distance,
            engine=args.render_engine,
            cycles_device=args.cycles_device,
            runtime_prefix=runtime_prefix,
            use_xvfb=not args.no_xvfb,
            xvfb_display=args.xvfb_display,
        )
        target_render_paths = render_selected_views(
            glb=target_glb,
            output_dir=output_dir,
            selections=selections,
            object_label="target",
            resolution=args.render_resolution,
            samples=args.render_samples,
            shrink=args.render_shrink,
            ortho_scale=args.render_ortho_scale,
            distance=args.render_distance,
            engine=args.render_engine,
            cycles_device=args.cycles_device,
            runtime_prefix=runtime_prefix,
            use_xvfb=not args.no_xvfb,
            xvfb_display=args.xvfb_display,
        )
        for method in selections:
            if method in sparse_voxel_render_paths:
                selections[method]["sparse_voxel_render_path"] = sparse_voxel_render_paths[method]
            if method in target_render_paths:
                selections[method]["target_render_path"] = target_render_paths[method]

    csv_path = output_dir / "candidate_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plan = {
        "item_dir": str(item_dir),
        "run_dir": str(run_dir),
        "work_dir": str(work_dir),
        "round": int(args.round),
        "policy_note": "ours selects argmax unresolved_visible_ratio - rho * angular_overlap over a free azimuth/elevation candidate grid; Table-4 baselines are random, farthest_camera, and whole_object_coverage. For each selected next view, the planner renders both the feature-colored sparse voxel GLB and the final generated target GLB.",
        "sparse_voxel_glb": str(sparse_voxel_glb),
        "target_glb": str(target_glb),
        "conflict_mask": str(conflict_mask),
        "quiescent_glb": str(quiescent_glb) if quiescent_glb else None,
        "num_object_voxels_for_visibility": int(len(object_points)),
        "num_unresolved_voxels_before_quiescent_subtract": unresolved_before,
        "num_unresolved_voxels_used": int(len(unresolved_points)),
        "candidate_count": int(len(candidates)),
        "candidate_source": candidate_source,
        "existing_view_angles": [{"azimuth": float(a), "elevation": float(e)} for a, e in existing_angles],
        "rho": float(args.rho),
        "render_script": str(RENDER_GLB_MV_SCRIPT),
        "render_settings": {
            "resolution": int(args.render_resolution),
            "samples": int(args.render_samples),
            "shrink": float(args.render_shrink),
            "ortho_scale": float(args.render_ortho_scale),
            "distance": float(args.render_distance),
            "engine": str(args.render_engine),
            "cycles_device": str(args.cycles_device),
        },
        "blender_runtime_prefix": str(runtime_prefix) if runtime_prefix else None,
        "use_xvfb": not args.no_xvfb,
        "xvfb_display": args.xvfb_display,
        "methods": selections,
        "candidate_scores_csv": str(csv_path),
    }
    plan_path = output_dir / "next_view_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "next_view_plan": str(plan_path),
        "methods": {
            k: {
                "view_id": v["view_id"],
                "azimuth": v["azimuth"],
                "elevation": v["elevation"],
                "sparse_voxel_render_path": v.get("sparse_voxel_render_path"),
                "target_render_path": v.get("target_render_path"),
            }
            for k, v in selections.items()
        },
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
