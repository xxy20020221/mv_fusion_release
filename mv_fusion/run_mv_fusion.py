"""Run Easy3E mv_fusion for a single "item directory".

Expected item layout:
    <item_dir>/
        object.glb          # source 3D asset
        mask.glb            # scene containing named nodes:
                            #   mask_all         -> whole editable region
                            #   mask_region_<v>  -> trust-region for view <v>
        mask_2.glb          # (round 2 only) mask.glb-like scene for the newly-added view
                            #   mask_region_new  -> trust-region of the new view
        source_<v>.png      # rendered source image for view <v>
                            # (source_<new_view_id>.png is auto-rendered in round 2 if missing)
        target_<v>.png      # user-edited image for view <v> (raw, un-RGBA)
        ...

Round 1 (default): runs the standard 2-view mv_fusion on the item and writes
outputs to <item_dir>/_easy3e_out_round1/.

Round 2: takes round-1 outputs (target_X_rgba.glb + velocity_conflict_*_mask.glb)
as inputs, adds a new view, re-runs mv_fusion with N (>=3) views. Round 2
outputs go to <item_dir>/_easy3e_out_round2/.

Usage (see README.md for the full option list):
    # Round 1 with the hand-made trust regions mask_region_<v> of mask.glb
    python mv_fusion/run_mv_fusion.py --item_dir demo/ninja --views 0,1 --no_guidance

    # Round 2 of the manual variant: mask_all = round-1 conflict voxels, new view's
    # trust region = mask_2.glb::mask_region_new
    python mv_fusion/run_mv_fusion.py --item_dir demo/ninja --round 2 --new_view_id 2 \
        --new_view_angle 270.0,0.0 --no_guidance

    # Geometry-derived per-view support instead of hand-made mask_region_<v>
    # boxes (only mask_all is needed); outputs go to *_auto_support dirs.
    python mv_fusion/run_mv_fusion.py --item_dir demo/rider --views 0,1 --view_support auto --no_guidance

    # Round 2 of Route-Diagnose-Refine on top of a finished round 1: the round-1
    # asset/SLAT become the source state, the edit region is the round-1 conflict
    # set (dilated, closed, snapped to latent cells, minus quiescent voxels), all
    # views are re-rendered, the new view is planned ('ours') and, without a
    # target_<new>.png, its target is synthesized with Nano Banana (API);
    # see _resolve_round2_auto_config.
    python mv_fusion/run_mv_fusion.py --item_dir demo/rider --round 2 --view_support auto \
        --new_view_angle auto --edit_instruction "..." --no_guidance
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

# Reuse the Easy3E-RGBA post-processing already used by the pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (BLENDER_BIN, CHECKPOINT, EASY3E_ROOT, ENC_PRETRAINED, MV_FUSION_DIR,  # noqa: E402
                   PYTHON_BIN, RENDER_GLB_MV_SCRIPT)
from process_easy3e_rgba_target import write_easy3e_rgba_target  # noqa: E402


# --------------------------------------------------------------------------- #
# Constants (paths: mv_fusion/paths.py, overridable with environment variables)
# --------------------------------------------------------------------------- #
NEXT_VIEW_PLANNER = MV_FUSION_DIR / "plan_mv_fusion_next_view.py"
VIEW_SUPPORT_SCRIPT = MV_FUSION_DIR / "compute_view_support.py"

# mv_fusion / flowedit hyper-parameters.
SEED = 1
CFG_SRC = 5.0
CFG_TAR = 5.0
FLOWEDIT_LAMBDA = 1.0
MV_FUSION_ALPHA_INSIDE = 1.0
MV_FUSION_ALPHA_OUTSIDE = 0.0
MV_FUSION_TRUST_REGION_STEP = 10
MV_FUSION_STEPS = "1,3,5,10,20"
MV_FUSION_SOFT_EDGE_WIDTH = 0.035
MV_FUSION_SOFT_CORE_WIDTH = 0.04
MV_FUSION_SOFT_OUTSIDE_WIDTH = 0.04
MV_FUSION_SOFT_INSIDE_PLATEAU = 0.9


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _find_view_pairs(item_dir: Path):
    """Return sorted list of view ids v for which both source_<v>.png and
    target_<v>.png exist."""
    view_ids = []
    for src in sorted(item_dir.glob("source_*.png")):
        stem = src.stem  # source_<v>
        v = stem[len("source_"):]
        tgt = item_dir / f"target_{v}.png"
        if tgt.exists():
            view_ids.append(v)
    return view_ids


def _extract_named_geometry(mask_glb: Path, node_name: str, out_glb: Path,
                            prenormalize: tuple | None = None) -> bool:
    """Extract the sub-mesh whose scene node is called `node_name` and export
    it as its own single-mesh GLB. Returns True on success.

    If `prenormalize` is not None, it must be a tuple ``(scale, offset_xyz)``
    (matching Easy3E's `_normalize_mask` semantics: v_norm = (v + offset) * scale)
    and the extracted mesh is transformed into the normalized frame before
    export. This is used by round-2 so the normalized round-1 world-space
    masks stay aligned with the identity-transform work root.
    """
    scene = trimesh.load(str(mask_glb), force="scene")
    if not isinstance(scene, trimesh.Scene):
        # single mesh loaded as Trimesh -> only usable if node_name matches
        # the file stem.
        if node_name in ("mask_all", mask_glb.stem):
            out_glb.parent.mkdir(parents=True, exist_ok=True)
            m = scene.copy() if prenormalize is None else _apply_prenormalize(scene.copy(), prenormalize)
            m.export(str(out_glb))
            return True
        return False

    if node_name not in scene.graph.nodes_geometry:
        # not present
        return False

    T, geom_name = scene.graph[node_name]
    mesh = scene.geometry[geom_name].copy()
    mesh.apply_transform(T)
    if prenormalize is not None:
        mesh = _apply_prenormalize(mesh, prenormalize)

    sub_scene = trimesh.Scene()
    sub_scene.add_geometry(mesh, node_name=node_name, geom_name=node_name)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    sub_scene.export(str(out_glb))
    return True


def _apply_prenormalize(mesh, prenormalize: tuple):
    """Apply Easy3E's blender-space normalization to a trimesh mesh in-place.

    Easy3E's `_normalize_mask` runs inside blender, whose scene frame is Z-up
    while the GLB (glTF 2.0) file frame is Y-up (blender applies
    ``(x, y, z) -> (x, -z, y)`` on import and the inverse on export). The
    (scale, offset) recorded in ``transforms.json`` are defined in that
    blender Z-up frame, and the actual internal normalized point cloud is

        v_internal = R_YZ(v_world_glb) * scale + offset

    where ``R_YZ: (x, y, z) -> (x, -z, y)``. When we later hand the mesh
    back to blender via a GLB, blender will re-apply ``R_YZ`` on import,
    so to end up with ``v_internal`` inside blender the GLB must contain
    the *inverse-Y-up* of ``v_internal``. Composing the two operations:

        v_glb_out = R_YZ^{-1}(R_YZ(v) * scale + offset)
                  = v * scale + R_YZ^{-1}(offset)

    which is exactly what we write below. This matches the mesh.ply
    produced by round-1 (X in [-0.5, 0.5], Y = short axis, Z = long axis for
    a standing character) when the same transforms.json is re-applied.
    """
    scale, offset_xyz = prenormalize
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    scale_f = float(scale)
    off = np.asarray(offset_xyz, dtype=np.float64).reshape(3)
    # R_YZ^{-1}(offset) with R_YZ: (x, y, z) -> (x, -z, y):
    # inverse maps (x, y, z) -> (x, z, -y).
    off_inv = np.asarray([off[0], off[2], -off[1]], dtype=np.float64).reshape(1, 3)
    verts = verts * scale_f + off_inv
    mesh.vertices = verts
    return mesh


def _list_mask_nodes(mask_glb: Path):
    scene = trimesh.load(str(mask_glb), force="scene")
    if not isinstance(scene, trimesh.Scene):
        return [mask_glb.stem]
    return list(scene.graph.nodes_geometry)


def _write_easy3e_rgba(src_png: Path, dst_png: Path) -> None:
    """Convert an edited target image into an Easy3E-compatible RGBA image
    (transparent background, foreground alpha=255). We write only the RGBA
    output next to `dst_png`."""
    dst_mask = dst_png.with_name(dst_png.stem + "_alpha_mask.png")
    dst_mask_512 = dst_png.with_name(dst_png.stem + "_alpha_mask_512.png")
    summary = write_easy3e_rgba_target(
        input_path=src_png,
        output_rgba_path=dst_png,
        output_mask_path=dst_mask,
        output_mask_512_path=dst_mask_512,
    )
    print(f"[rgba] {src_png.name} -> {dst_png.name} "
          f"(fg_ratio={summary['foreground_ratio']:.3f})")


def _align_target_to_source(source_png: Path, target_rgba_png: Path) -> dict:
    """Warp an RGBA target into its source image's pixel frame, in place (unaligned copy kept).

    Uses compute_view_support.target_alignment: features of the shared object when well
    supported, otherwise a pure resize (generated targets keep the source framing).
    """
    import compute_view_support as cvs

    unaligned = target_rgba_png.with_name(f"{target_rgba_png.stem}_unaligned.png")
    shutil.copy2(target_rgba_png, unaligned)
    info = cvs.warp_target_to_source(source_png, unaligned, target_rgba_png)
    print(f"[align] {target_rgba_png.name} -> frame of {source_png.name} ({info['method']}, "
          f"inliers={info['inliers']}, source silhouette covered {info['source_coverage']:.3f})")
    return info


def _render_new_view(
    object_glb: Path,
    out_png: Path,
    azimuth_deg: float,
    elevation_deg: float,
    resolution: int = 512,
    shrink: float = 0.0,
) -> None:
    """Headless-render a single ortho view of `object_glb` into `out_png` using
    blender_scripts/render_glb_mv.py. Writes the image directly to `out_png`
    (bypassing the script's default `image_ori/<vid>.png` layout by copying
    afterward).

    NOTE: default shrink=0 means "do not renormalize the mesh"; this is what
    we want for round-1 output glb which is already in [-0.5, 0.5]^3.
    """
    if not RENDER_GLB_MV_SCRIPT.exists():
        raise FileNotFoundError(f"Missing renderer: {RENDER_GLB_MV_SCRIPT}")
    if not BLENDER_BIN.exists():
        raise FileNotFoundError(f"Missing blender binary: {BLENDER_BIN}")

    render_tmp_dir = out_png.parent / f"_render_tmp_{out_png.stem}"
    render_tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(BLENDER_BIN), "-b",
        "-P", str(RENDER_GLB_MV_SCRIPT), "--",
        "--object_path", str(object_glb),
        "--output_dir", str(render_tmp_dir),
        "--resolution", str(resolution),
        "--shrink", str(shrink),
        "--distance", "2.0",
        "--ortho_scale", "1.0",
        "--engine", "CYCLES",
        "--cycles_device", "CPU",
        "--samples", "32",
        "--view_azimuths", f"{azimuth_deg}",
        "--view_elevations", f"{elevation_deg}",
        "--view_ids", "0",
    ]
    print(f"[render_new_view] {' '.join(cmd)}")
    env = os.environ.copy()
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("DISPLAY", None)
    subprocess.run(cmd, env=env, check=True)

    rendered_png = render_tmp_dir / "image_ori" / "000.png"
    if not rendered_png.exists():
        raise FileNotFoundError(f"Blender did not produce expected image: {rendered_png}")
    shutil.copy2(rendered_png, out_png)
    print(f"[render_new_view] {rendered_png} -> {out_png}")


def _conflict_mask_to_mask_all_glb(
    conflict_mask_glb: Path,
    out_mask_all_glb: Path,
    voxel_size: float = 1.0 / 64.0,
) -> int:
    """Read a velocity_conflict_*_mask.glb (categorical GLB where red-labeled
    voxels represent conflict) and export a new GLB containing only those
    red cubes. The resulting GLB is meant to be used as the next-round
    `mask_all`. Returns the number of conflict voxels kept."""
    scene = trimesh.load(str(conflict_mask_glb), force="mesh")
    if scene is None or (hasattr(scene, "vertices") and len(scene.vertices) == 0):
        raise RuntimeError(f"Empty conflict-mask glb: {conflict_mask_glb}")

    mesh = scene  # trimesh.load(..., force="mesh") concatenates all geometry into one mesh
    if not hasattr(mesh, "visual") or not hasattr(mesh.visual, "vertex_colors"):
        raise RuntimeError(
            f"Conflict-mask glb has no per-vertex colors: {conflict_mask_glb}"
        )
    colors = np.asarray(mesh.visual.vertex_colors)  # (V,4) uint8
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise RuntimeError(f"Unexpected vertex_colors shape: {colors.shape}")

    # Palette used by _export_coords_category_glb: label 2 -> [255, 80, 80, 240]
    # We select vertices with strong red, weak green, weak blue.
    is_red = (
        (colors[:, 0] >= 220)
        & (colors[:, 1] <= 130)
        & (colors[:, 2] <= 130)
    )
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    # keep only faces whose 3 vertices are all red -> corresponds to red cubes
    face_red = is_red[faces].all(axis=1)
    if not face_red.any():
        raise RuntimeError(
            f"No red (conflict) voxels found in {conflict_mask_glb}"
        )
    kept_faces = faces[face_red]
    kept_vert_idx = np.unique(kept_faces.reshape(-1))
    remap = -np.ones(verts.shape[0], dtype=np.int64)
    remap[kept_vert_idx] = np.arange(kept_vert_idx.shape[0])
    new_verts = verts[kept_vert_idx]
    new_faces = remap[kept_faces]

    # The conflict-mask GLB was written by flow_euler with vertex positions
    # equal to ``(idx + 0.5) / 64 - 0.5`` in the pipeline-internal (blender
    # Z-up, normalized) frame -- i.e. NOT the Y-up frame the GLB spec
    # expects. When Easy3E hands this GLB back to blender it will apply the
    # standard glTF Y-up -> Z-up conversion ``(x, y, z) -> (x, -z, y)``,
    # scrambling the mask. To cancel that we pre-apply the *inverse*
    # rotation ``(x, y, z) -> (x, z, -y)`` before export.
    new_verts_gltf = np.stack(
        [new_verts[:, 0], new_verts[:, 2], -new_verts[:, 1]], axis=-1
    )

    out_mesh = trimesh.Trimesh(vertices=new_verts_gltf, faces=new_faces, process=False)
    # Estimate how many voxels: red cubes have 24 vertices each (box has 8 unique verts
    # duplicated 3 times by trimesh.box). Fall back to reporting kept vertex count / 8.
    approx_voxels = int(round(kept_vert_idx.shape[0] / 8.0))

    sub_scene = trimesh.Scene()
    sub_scene.add_geometry(out_mesh, node_name="mask_all", geom_name="mask_all")
    out_mask_all_glb.parent.mkdir(parents=True, exist_ok=True)
    sub_scene.export(str(out_mask_all_glb))
    print(f"[conflict->mask_all] {conflict_mask_glb} -> {out_mask_all_glb} "
          f"(kept {approx_voxels} conflict voxels, applied Y-up<->Z-up inverse)")
    return approx_voxels


def _read_red_voxel_centers_from_category_glb(glb_path: Path, apply_inv_yup=True,
                                              only_red: bool = True):
    """Read a voxel-cube GLB (as produced by ``_export_coords_category_glb`` or
    ``_conflict_mask_to_mask_all_glb``) and return an (N, 3) float array of
    unique voxel-center coordinates in the pipeline's internal Z-up normalized
    frame ([-0.5, 0.5]^3).

    Approach: each voxel is exported as a cube whose 8 corner vertices are
    consecutive in the vertex array (``trimesh.util.concatenate`` preserves
    order, and ``_conflict_mask_to_mask_all_glb`` only remaps face indices
    which is later handled by grouping through faces).

    - When ``only_red=True``: keep only cubes whose faces are all
      red-labeled ([255,80,80,240] palette) -- used e.g. for quiescent /
      conflict subset extraction. NOTE: quiescent GLBs currently contain a
      single (red) color for every voxel so this is equivalent to the
      ``only_red=False`` case.
    - When ``only_red=False``: keep every cube -- used for the mask_all.glb
      written by ``_conflict_mask_to_mask_all_glb`` (which strips per-vertex
      colors during Trimesh reconstruction).

    Both cases fall back to the same recovery rule: reshape the vertex list
    into groups of 8 and average each group to get the cube center.

    When ``apply_inv_yup=True`` (default), we invert the Y-up<->Z-up transform
    that ``_conflict_mask_to_mask_all_glb`` applied on export.
    """
    scene = trimesh.load(str(glb_path), force="mesh")
    if scene is None or not hasattr(scene, "vertices") or len(scene.vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    verts = np.asarray(scene.vertices)
    faces = np.asarray(scene.faces)

    if only_red:
        if not hasattr(scene, "visual") or not hasattr(scene.visual, "vertex_colors"):
            return np.zeros((0, 3), dtype=np.float32)
        colors = np.asarray(scene.visual.vertex_colors)
        if colors.ndim != 2 or colors.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        is_red = (colors[:, 0] >= 220) & (colors[:, 1] <= 130) & (colors[:, 2] <= 130)
        # If ALL vertices are red, skip filtering to keep the cube-8-verts
        # grouping intact (this is the common case for our quiescent glbs).
        if not is_red.all():
            face_red = is_red[faces].all(axis=1)
            if not face_red.any():
                return np.zeros((0, 3), dtype=np.float32)
            # Build a per-cube red-mask: a cube is red iff all 12 of its faces are red.
            # Faces are laid out in the same 12-per-cube order as vertices are in
            # 8-per-cube order. We compute how many red faces exist per cube.
            n_faces_per_cube = 12
            if faces.shape[0] % n_faces_per_cube != 0:
                # Fallback: just keep vertices touched by any red face.
                touched = np.unique(faces[face_red].reshape(-1))
                verts = verts[touched]
                faces = np.array([], dtype=np.int64).reshape(0, 3)
            else:
                cube_red = face_red.reshape(-1, n_faces_per_cube).all(axis=1)
                # Corresponding vertex slice: 8-per-cube.
                verts = verts.reshape(-1, 8, 3)[cube_red].reshape(-1, 3)

    if verts.shape[0] == 0 or verts.shape[0] % 8 != 0:
        # Should not happen for our GLBs; but be safe.
        return np.zeros((0, 3), dtype=np.float32)

    # Cube centers = mean of each group of 8 corner vertices.
    centers = verts.reshape(-1, 8, 3).mean(axis=1)
    if apply_inv_yup:
        centers = np.stack([centers[:, 0], -centers[:, 2], centers[:, 1]], axis=-1)

    # Dedup by rounding to the 64^3 center grid.
    voxel = 1.0 / 64.0
    idx = np.round((centers + 0.5) / voxel - 0.5).astype(np.int64)
    idx = np.clip(idx, 0, 63)
    idx_flat = idx[:, 0] * (64 * 64) + idx[:, 1] * 64 + idx[:, 2]
    _, uniq_first = np.unique(idx_flat, return_index=True)
    uniq_idx = idx[uniq_first]
    return (uniq_idx.astype(np.float32) + 0.5) * voxel - 0.5


def _subtract_quiescent_from_mask_all_glb(
    mask_all_glb: Path,
    quiescent_only_glb: Path,
    out_mask_all_glb: Path,
    voxel_size: float = 1.0 / 64.0,
) -> tuple:
    """Load an existing mask_all glb (as written by _conflict_mask_to_mask_all_glb)
    and produce a new mask_all glb with all voxels in ``quiescent_only_glb``
    removed. Both inputs are treated as *voxel-center* sets in the pipeline's
    ``(x, y, z) -> (x, z, -y)`` before export.

    Returns ``(voxels_before, voxels_after)``.
    """
    # mask_all_glb was written with the Y-up<->Z-up inverse already applied and
    # does NOT preserve the red color palette (trimesh strips per-vertex colors
    # when the mesh is rebuilt), so we treat every vertex as an in-mask voxel.
    mask_centers = _read_red_voxel_centers_from_category_glb(
        mask_all_glb, apply_inv_yup=True, only_red=False
    )
    # quiescent_voxels_only.glb was written directly by flow_euler (Z-up internal)
    # with the red palette intact.
    q_centers = _read_red_voxel_centers_from_category_glb(
        quiescent_only_glb, apply_inv_yup=False, only_red=True
    )
    if mask_centers.shape[0] == 0:
        print(f"[mask_all-quiescent] {mask_all_glb.name}: 0 voxels detected in mask_all "
              f"(nothing to subtract).")
        return 0, 0
    if q_centers.shape[0] == 0:
        print(f"[mask_all-quiescent] {mask_all_glb.name}: quiescent set empty "
              f"({mask_centers.shape[0]} mask voxels unchanged).")
        return int(mask_centers.shape[0]), int(mask_centers.shape[0])
    # Discretize both to 64^3 grid indices for exact set difference.
    def _to_idx_flat(c):
        idx = np.round((c + 0.5) / voxel_size - 0.5).astype(np.int64)
        idx = np.clip(idx, 0, 63)
        return idx[:, 0] * (64 * 64) + idx[:, 1] * 64 + idx[:, 2]

    m_flat = _to_idx_flat(mask_centers)
    q_flat_set = set(_to_idx_flat(q_centers).tolist())
    keep = np.array([f not in q_flat_set for f in m_flat.tolist()], dtype=bool)
    kept_centers = mask_centers[keep]
    n_before = int(mask_centers.shape[0])
    n_after = int(kept_centers.shape[0])
    if n_after == 0:
        raise RuntimeError(
            f"After subtracting quiescent voxels, mask_all is empty. "
            f"before={n_before}, quiescent={q_centers.shape[0]}"
        )

    # Rebuild a category GLB with the same red palette, matching the
    # Y-up<->Z-up inverse convention used by _conflict_mask_to_mask_all_glb.
    voxel = voxel_size
    base_cube = trimesh.creation.box(extents=[voxel, voxel, voxel])
    cubes = []
    for c in kept_centers:
        # Apply Y-up<->Z-up inverse on export: (x, y, z) -> (x, z, -y)
        c_gltf = np.array([c[0], c[2], -c[1]])
        cubes.append(base_cube.copy().apply_translation(c_gltf))
    merged = trimesh.util.concatenate(cubes)
    # Paint red (label=2)
    colors = np.tile(np.array([255, 80, 80, 240], dtype=np.uint8), (merged.vertices.shape[0], 1))
    merged.visual = trimesh.visual.ColorVisuals(mesh=merged, vertex_colors=colors)

    sub_scene = trimesh.Scene()
    sub_scene.add_geometry(merged, node_name="mask_all", geom_name="mask_all")
    out_mask_all_glb.parent.mkdir(parents=True, exist_ok=True)
    sub_scene.export(str(out_mask_all_glb))
    print(f"[mask_all-quiescent] {mask_all_glb.name}: {n_before} -> {n_after} voxels "
          f"(subtracted {n_before - n_after} quiescent voxels)")
    return n_before, n_after


def _run_and_log(command, log_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("EASY3E_BLENDER_CPU", "1")
    env.setdefault("EASY3E_BLENDER_SAMPLES", "16")
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("ATTN_BACKEND", "sdpa")
    env.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    env["PYTHONPATH"] = str(EASY3E_ROOT)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=str(EASY3E_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command)


# --------------------------------------------------------------------------- #
# Round-1 vs Round-2 config resolution
# --------------------------------------------------------------------------- #
def _resolve_round1_config(item_dir: Path, views: list[str] | None = None,
                           source_transform: Path | None = None):
    """Original round-1 layout: source is object.glb, mask_all comes from
    mask.glb, views are all source_/target_ pairs (or the requested subset).
    `source_transform` replaces preprocess's auto normalization (e.g. to leave
    room for parts added outside the source bbox)."""
    object_glb = item_dir / "object.glb"
    mask_glb = item_dir / "mask.glb"
    _require(object_glb, "object.glb")
    _require(mask_glb, "mask.glb")

    view_ids = _find_view_pairs(item_dir)
    if views:
        missing = [v for v in views if v not in view_ids]
        if missing:
            raise ValueError(f"--views {missing} have no source_/target_ pair in {item_dir}")
        view_ids = [v for v in view_ids if v in views]
    if len(view_ids) < (1 if views else 2):
        raise ValueError(
            f"Round 1 requires at least 2 views (or an explicit --views), found source/target "
            f"pairs for views {view_ids} in {item_dir}"
        )
    mask_nodes = _list_mask_nodes(mask_glb)
    if "mask_all" not in mask_nodes:
        raise RuntimeError(
            f"mask.glb must contain a node named 'mask_all', got {mask_nodes}"
        )
    return {
        "source_glb": object_glb,
        "mask_all_source_glb": mask_glb,
        "mask_all_node": "mask_all",
        "region_glbs": {v: (mask_glb, f"mask_region_{v}", None) for v in view_ids
                        if f"mask_region_{v}" in mask_nodes},
        "view_ids": view_ids,
        "mask_nodes": mask_nodes,
        "output_root": item_dir / "_easy3e_out_round1",
        "work_root": item_dir / "_easy3e_work_round1",
        "source_transform_path": source_transform,
    }


def _resolve_round2_config(item_dir: Path, args) -> dict:
    """Round-2 layout: source becomes the round-1 target glb, mask_all comes
    from the round-1 velocity_conflict mask, an extra view (new_view_id) is
    added with source_<new>.png auto-rendered from the new source_glb.

    Reuses round-1's transforms.json so that the new source_glb, mask_all
    (already living in [-0.5,0.5]^3) and the newly-rendered view stay
    aligned."""
    # Round-1 outputs may live under _easy3e_out_round1/ (new layout) or
    # _easy3e_out/ (legacy layout). Prefer the round-1 run of the same variant
    # (e.g. _easy3e_out_round1_auto_support), then the default, then legacy.
    suffix = _run_suffix(args)
    layouts = [("_easy3e_out_round1", "_easy3e_work_round1"), ("_easy3e_out", "_easy3e_work")]
    if suffix:
        layouts.insert(0, (f"_easy3e_out_round1{suffix}", f"_easy3e_work_round1{suffix}"))
    for out_name, work_name in layouts:
        candidate_out = item_dir / out_name
        candidate_work = item_dir / work_name
        if candidate_out.exists() and (candidate_work / "render" / "transforms.json").exists():
            round1_out = candidate_out
            round1_render_dir = candidate_work / "render"
            print(f"[round2] Using round-1 outputs from {round1_out} "
                  f"(render dir: {round1_render_dir})")
            break
    else:
        raise FileNotFoundError(
            f"Could not locate round-1 outputs under {item_dir} "
            f"(looked for _easy3e_out_round1/ and _easy3e_out/)."
        )

    round1_transform = round1_render_dir / "transforms.json"
    if not round1_transform.exists():
        raise FileNotFoundError(
            f"Round 1 transforms.json not found: {round1_transform}. "
            f"Please run round 1 first."
        )

    # Round-1 final glb (source for round 2). We pick the newest *_rgba.glb.
    candidates = sorted((round1_out / "out").glob("*_rgba.glb"))
    if not candidates:
        raise FileNotFoundError(
            f"No round-1 output *_rgba.glb found under {round1_out / 'out'}"
        )
    round1_final_glb = candidates[-1]

    # Round-1 conflict mask -> new mask_all for round 2.
    conflict_mask_glb = _round_diagnosis_file(round1_out, "velocity_conflict_*_mask.glb")
    if conflict_mask_glb is None:
        raise FileNotFoundError(f"No velocity_conflict_*_mask.glb under {round1_out}")

    # Round-1 quiescent-only mask (voxels the sampler flagged as velocity-stable
    # AND low-magnitude inside mask_all -> should be frozen to source in round 2).
    # This is optional: rounds run before the quiescent feature was added won't have it.
    quiescent_only_glb = _round_diagnosis_file(round1_out, "quiescent_voxels_only.glb")

    # New view id / angle.
    new_view_id = args.new_view_id or "2"
    new_view_angle = args.new_view_angle or "270.0,0.0"
    try:
        az_str, el_str = new_view_angle.split(",")
        new_view_az = float(az_str.strip())
        new_view_el = float(el_str.strip())
    except Exception as exc:
        raise ValueError(
            f"--new_view_angle must be 'az,el' (e.g. '270.0,0.0'), got "
            f"{new_view_angle!r}"
        ) from exc

    # target_<new_view_id>.png must exist (user-provided edited image).
    new_target_png = item_dir / f"target_{new_view_id}.png"
    _require(new_target_png, f"target_{new_view_id}.png")

    # source_<new_view_id>.png: auto-render from round1_final_glb if missing.
    new_source_png = item_dir / f"source_{new_view_id}.png"
    if not new_source_png.exists() or args.force_render_new_view:
        print(f"[round2] Rendering new view {new_view_id} "
              f"(az={new_view_az}, el={new_view_el}) from {round1_final_glb}")
        _render_new_view(
            round1_final_glb,
            new_source_png,
            azimuth_deg=new_view_az,
            elevation_deg=new_view_el,
        )
    else:
        print(f"[round2] Reusing existing {new_source_png}")

    # Base views: everything from round 1 that already has source_/target_.
    base_view_ids = _find_view_pairs(item_dir)
    # ensure new_view_id is included at the END so it has highest priority in fusion.
    view_ids = [v for v in base_view_ids if v != new_view_id] + [new_view_id]

    # mask_2.glb: contains mask_region_new for the newly-added view.
    mask_2_glb = item_dir / "mask_2.glb"
    _require(mask_2_glb, "mask_2.glb")
    mask_2_nodes = _list_mask_nodes(mask_2_glb)
    if "mask_region_new" not in mask_2_nodes:
        raise RuntimeError(
            f"mask_2.glb must contain node 'mask_region_new', got {mask_2_nodes}"
        )
    # Also allow the ORIGINAL mask.glb region masks for the base views (so the
    # base views' trust regions still apply in round 2). Round-1 mask.glb
    # lives in blender world space, so we must apply round-1's (scale, offset)
    # to bring those meshes into [-0.5, 0.5]^3 (matching mask_region_new).
    mask_glb = item_dir / "mask.glb"
    orig_mask_nodes = _list_mask_nodes(mask_glb) if mask_glb.exists() else []

    # Load round-1 transforms.json (scale + offset) once.
    with round1_transform.open("r", encoding="utf-8") as f:
        r1t = json.load(f)
    r1_scale = float(r1t.get("scale", 1.0))
    r1_offset = list(r1t.get("offset", [0.0, 0.0, 0.0]))
    prenorm = (r1_scale, r1_offset)
    print(f"[round2] Using round-1 (scale={r1_scale}, offset={r1_offset}) to "
          f"pre-normalize world-space masks from {mask_glb}.")

    # region_glbs entry format: {view_id: (source_glb, node_name, prenormalize_or_None)}
    region_glbs = {}
    for v in base_view_ids:
        if v == new_view_id:
            continue
        node = f"mask_region_{v}"
        if node in orig_mask_nodes:
            region_glbs[v] = (mask_glb, node, prenorm)
    # New-view region comes from mask_2.glb; it already lives in [-0.5, 0.5]^3
    # (built by the user directly in the normalized frame), so no prenormalize.
    region_glbs[new_view_id] = (mask_2_glb, "mask_region_new", None)

    return {
        "source_glb": round1_final_glb,
        # For round 2, mask_all comes from the conflict mask (converted).
        "conflict_mask_glb": conflict_mask_glb,
        "quiescent_only_glb": quiescent_only_glb,
        "region_glbs": region_glbs,
        "view_ids": view_ids,
        "new_view_id": new_view_id,
        "output_root": item_dir / "_easy3e_out_round2",
        "work_root": item_dir / "_easy3e_work_round2",
        # Round-1 output glb is already normalized to [-0.5, 0.5]^3, so we
        # supply an IDENTITY transforms.json (scale=1, offset=0) to prevent
        # preprocess from re-normalizing the glb (which would break the
        # alignment with conflict_mask_glb and mask_region_new). This will be
        # materialized to disk by main() before preprocess is called.
        "source_transform_path": "IDENTITY",
        "round1_transform": round1_transform,  # kept only for logging
        # Method D: reuse round-1's final edited SLAT as round-2's z_src.
        # This file is written by pipelines/trellis_image_to_3d.py::flowedit
        # at the end of round 1 (features/slat_edited.pt under the round-1
        # output_root — same convention as driver's feature_dir = output_root/features).
        "round1_slat_edited": (round1_out / "features" / "slat_edited.pt"),
    }


def _round_diagnosis_file(round_out: Path, pattern: str) -> Path | None:
    """Newest diagnosis file of a finished round.

    features/mv_fusion_voxel_diff/ is written by the same edit run as
    features/slat_edited.pt; the out/ copies are only refreshed when the driver
    finishes, so they can be stale (e.g. after a re-export of the final GLB).
    """
    for d in (round_out / "features" / "mv_fusion_voxel_diff", round_out / "out"):
        hits = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime)
        if hits:
            return hits[-1]
    return None


def _resolve_round2_auto_config(item_dir: Path, args) -> dict:
    """Round r -> r+1 transition of Route-Diagnose-Refine (Sec. 3.3-3.4), --view_support auto.

    * state continuation: the previous round's final GLB / SLAT (and structure
      latent) become the new source state; preprocess uses an identity transform;
    * next edit region (Eq. 16): Omega^1 & Close(Dilate(C^1)) minus the quiescent
      set U^1, which joins the preserved region; edited-stable voxels stay as edited;
    * every view's source image is re-rendered from the current asset (Alg. 1,
      line 3) and tone-matched to the original source renders;
    * the new view comes from the 'ours' next-view policy, its target is synthesized
      with nano banana from the primary-view source/target pair and the other edited
      views, and it is appended last so it takes priority in ownership (Eq. 7).
    """
    import mv_fusion_rounds as rounds

    suffix = _run_suffix(args)
    output_root = item_dir / f"_easy3e_out_round2{suffix}"
    work_root = item_dir / f"_easy3e_work_round2{suffix}"
    log_dir = output_root / "logs"
    for d in (output_root, work_root, log_dir, work_root / "3Dmasks"):
        d.mkdir(parents=True, exist_ok=True)

    # ---- previous round ----
    if args.round1_out_dir is not None:
        r1_out = args.round1_out_dir.expanduser().resolve()
    else:
        r1_out = next((item_dir / n for n in (f"_easy3e_out_round1{suffix}", "_easy3e_out_round1")
                       if (item_dir / n).exists()), None)
        if r1_out is None:
            raise FileNotFoundError(f"No round-1 output root under {item_dir}; pass --round1_out_dir")
    r1_work = (args.round1_work_dir.expanduser().resolve() if args.round1_work_dir is not None
               else r1_out.parent / r1_out.name.replace("_out_", "_work_", 1))
    glbs = sorted((r1_out / "out").glob("*_rgba.glb"), key=lambda p: p.stat().st_mtime)
    if not glbs:
        raise FileNotFoundError(f"No round-1 output *_rgba.glb under {r1_out / 'out'}")
    r1_glb = glbs[-1]
    r1_slat = r1_out / "features" / "slat_edited.pt"
    r1_ss = r1_out / "features" / "ss_latent_edited.pt"
    states_glb = _round_diagnosis_file(r1_out, "quiescent_voxels.glb")
    for path, label in ((r1_slat, "round-1 slat_edited.pt"), (r1_work / "render" / "transforms.json", "round-1 transforms.json"),
                        (r1_work / "3Dmasks" / "mask_all.glb", "round-1 mask_all.glb")):
        _require(path, label)
    if states_glb is None:
        raise FileNotFoundError(f"No quiescent_voxels.glb (round-1 diagnosis states) under {r1_out}")
    r1_summary_path = r1_out / "run_summary.json"
    r1_summary = json.loads(r1_summary_path.read_text()) if r1_summary_path.exists() else {}
    old_views = [str(v) for v in (r1_summary.get("views") or _find_view_pairs(item_dir))]
    print(f"[round2] previous round {r1_out.name}: glb={r1_glb.name}, views={old_views}, states={states_glb}")

    # ---- cameras of the previous views ----
    cams: dict = {}
    for cam_file in (r1_out / "view_support" / "view_cameras.json", args.support_camera_json):
        if cam_file is not None and Path(cam_file).exists():
            cams.update(json.loads(Path(cam_file).read_text()))
    missing = [v for v in old_views if v not in cams]
    if missing:
        fit_dir = work_root / "camera_fit"
        _run_and_log([
            str(PYTHON_BIN), "-u", str(VIEW_SUPPORT_SCRIPT), "--fit_cameras_only",
            "--render_dir", str(r1_work / "render"), "--views", ",".join(missing),
            "--source_images", ",".join(str(item_dir / f"source_{v}.png") for v in missing),
            "--output_dir", str(fit_dir),
        ], log_dir / "camera_fit.log")
        cams.update(json.loads((fit_dir / "view_cameras.json").read_text()))
    if any(cams[v].get("distance") is not None for v in old_views):
        raise ValueError("Round-2 re-rendering supports orthographic view cameras only")
    ortho_scales = {round(cams[v]["width"] / cams[v]["scale"], 3) for v in old_views}
    if len(ortho_scales) != 1:
        raise ValueError(f"Previous views use different ortho scales {ortho_scales}")
    ortho_scale = ortho_scales.pop()
    res = int(cams[old_views[0]]["width"])
    # The cameras look at a common point of the normalized frame; it is not the origin when
    # the source framing was kept around a re-centered composite (build_rough_edit_mask.py).
    rows, rhs = [], []
    for v in old_views:
        c = cams[v]
        _, right, up, _ = rounds.cvs._basis(c["azimuth"], c["elevation"])
        rows += [right, up]
        rhs += [-(c["tx"] - c["width"] / 2.0) / c["scale"], (c["ty"] - c["height"] / 2.0) / c["scale"]]
    look_at = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0]
    if np.linalg.norm(look_at) < 1e-6:
        look_at = np.zeros(3)

    # ---- newly acquired view (Eq. 17, 'ours') ----
    new_id = args.new_view_id or str(max(int(v) for v in old_views if v.isdigit()) + 1)
    angle = args.new_view_angle or "auto"
    if angle == "auto":
        plan_dir = output_root / "next_view_acquisition"
        plan_path = plan_dir / "next_view_plan.json"
        if not plan_path.exists():
            _run_and_log([
                str(PYTHON_BIN), "-u", str(NEXT_VIEW_PLANNER),
                "--item-dir", str(item_dir), "--run-dir", str(r1_out), "--work-dir", str(r1_work),
                "--round", "1", "--output-dir", str(plan_dir), "--methods", "ours", "--skip-render",
                "--conflict-mask", str(_round_diagnosis_file(r1_out, "velocity_conflict_*_mask.glb")),
                "--quiescent-glb", str(_round_diagnosis_file(r1_out, "quiescent_voxels_only.glb")),
                "--sparse-voxel-glb", str(r1_out / "features" / "vox_edit.glb"), "--target-glb", str(r1_glb),
                "--existing-view-angles", ";".join(f"{cams[v]['azimuth']},{cams[v]['elevation']}" for v in old_views),
                "--render-ortho-scale", str(ortho_scale),
            ], log_dir / "next_view_acquisition.log")
        ours = json.loads(plan_path.read_text())["methods"]["ours"]
        new_az, new_el = float(ours["azimuth"]), float(ours["elevation"])
    else:
        new_az, new_el = (float(x) for x in angle.split(","))
    _, right, up, _ = rounds.cvs._basis(new_az, new_el)
    cams[new_id] = {"azimuth": new_az, "elevation": new_el, "distance": None, "scale": res / ortho_scale,
                    "tx": res / 2.0 - res / ortho_scale * float(look_at @ right),
                    "ty": res / 2.0 + res / ortho_scale * float(look_at @ up), "width": res, "height": res}
    view_ids = [v for v in old_views if v != new_id] + [new_id]
    camera_json = work_root / "view_cameras.json"
    camera_json.write_text(json.dumps({v: cams[v] for v in view_ids}, indent=2))
    print(f"[round2] new view {new_id}: az={new_az}, el={new_el}")

    # ---- source renders of the current asset, tone-matched to the original sources ----
    render_root = work_root / "source_renders"
    sources = {v: render_root / f"source_{v}.png" for v in view_ids}
    if not all(p.exists() for p in sources.values()):
        angles = {v: (cams[v]["azimuth"], cams[v]["elevation"]) for v in view_ids}
        current_glb, previous_glb = r1_glb, r1_work / "render" / "normalized.glb"
        if look_at.any():
            render_root.mkdir(parents=True, exist_ok=True)
            current_glb = rounds.recentered_glb(current_glb, look_at, render_root / "current_recentered.glb")
            previous_glb = rounds.recentered_glb(previous_glb, look_at, render_root / "previous_recentered.glb")
        current = rounds.render_views(current_glb, angles, render_root / "current", resolution=res, ortho_scale=ortho_scale)
        previous = rounds.render_views(previous_glb, {v: angles[v] for v in old_views},
                                       render_root / "previous_source", resolution=res, ortho_scale=ortho_scale)
        lut = rounds.fit_tone_lut([(previous[v], item_dir / f"source_{v}.png") for v in old_views])
        np.save(render_root / "tone_lut.npy", lut)
        for v in view_ids:
            rounds.apply_tone_lut(current[v], sources[v], lut)

    # ---- targets: previous targets + synthesized target of the new view ----
    targets = {v: item_dir / f"target_{v}.png" for v in old_views}
    synth_dir = output_root / "next_view_target"
    new_target = next((p for p in (args.new_view_target, item_dir / f"target_{new_id}.png",
                                   synth_dir / f"target_{new_id}.png") if p is not None and Path(p).exists()), None)
    if new_target is None:
        if not args.edit_instruction:
            raise ValueError("Synthesizing the new view's target needs --edit_instruction (or pass --new_view_target)")
        base = args.base_view or old_views[0]
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_source = synth_dir / f"source_{new_id}.png"
        if not synth_source.exists():
            shutil.copy2(sources[new_id], synth_source)
        result = rounds.synthesize_view_target(
            synth_source, (item_dir / f"source_{base}.png", item_dir / f"target_{base}.png"),
            [item_dir / f"target_{v}.png" for v in old_views if v != base], args.edit_instruction,
            args.new_view_desc, args.new_view_keep, synth_dir, model=args.synthesis_model,
            n_candidates=args.synthesis_candidates)
        new_target = synth_dir / f"target_{new_id}.png"
        shutil.copy2(result["selected"], new_target)
    targets[new_id] = Path(new_target)

    # ---- three-state diagnosis and the next edit region ----
    omega = rounds.round_edit_region(r1_work / "3Dmasks" / "mask_all.glb", r1_work / "render")
    states = rounds.load_diagnosis_states(states_glb)
    omega_next, state_info = rounds.next_active_region(
        omega, states, dilate=args.next_mask_dilate, close=args.next_mask_close,
        exclude_quiescent=not args.keep_quiescent_editable, latent_cell=args.next_mask_latent_cell)
    if not omega_next.any():
        raise RuntimeError("Previous round left no unresolved voxels (C empty); stopping as in Alg. 1.")
    mask_voxels = work_root / "3Dmasks" / "edit_region_round2.npy"
    np.save(mask_voxels, omega_next)
    rounds.export_state_glbs(states, omega, omega_next, output_root)
    state_info.update({"previous_round_out": str(r1_out), "states_glb": str(states_glb),
                       "edit_region_npy": str(mask_voxels)})
    (output_root / "round_state.json").write_text(json.dumps(state_info, indent=2))
    print(f"[round2] states: {state_info}")

    return {
        "source_glb": r1_glb,
        "source_transform_path": "IDENTITY",
        "view_ids": view_ids,
        "new_view_id": new_id,
        "new_view_angle": f"{new_az},{new_el}",
        "source_paths": sources,
        "target_paths": targets,
        "alignment_sources": {v: item_dir / f"source_{v}.png" for v in view_ids if (item_dir / f"source_{v}.png").exists()},
        "mask_voxels": mask_voxels,
        "region_glbs": {},
        "output_root": output_root,
        "work_root": work_root,
        "round1_slat_edited": r1_slat,
        "round1_ss_latent": r1_ss if r1_ss.exists() else None,
        "camera_json": camera_json,
        "round_state": state_info,
    }


def _run_suffix(args) -> str:
    """Output-dir suffix that keeps ablation variants apart from the default run."""
    suffix = ""
    if args.fusion_mode != "trust_region":
        suffix += f"_{args.fusion_mode}"
    if args.view_support == "auto":
        suffix += "_auto_support"
    if args.run_tag:
        suffix += f"_{args.run_tag}"
    return suffix


def _render_cache_complete(render_dir: Path) -> bool:
    return all(
        (render_dir / name).exists() and (render_dir / name).stat().st_size > 0
        for name in ("transforms.json", "mesh.ply", "feature.npz", "ss_latent.pt",
                     "slat_latent.pt", "normalized.glb")
    )


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_dir", required=True, type=Path,
                        help="Directory containing object.glb / mask.glb / "
                             "source_<v>.png / target_<v>.png.")
    parser.add_argument("--round", type=int, choices=[1, 2], default=1,
                        help="Which iteration to run. 1 = original 2-view flow; "
                             "2 = use round-1 outputs (target glb + conflict mask) "
                             "and add a new view.")
    parser.add_argument("--new_view_id", default=None,
                        help="View id label (a token used in source_<v>.png / "
                             "target_<v>.png filenames) for the newly-added view "
                             "in round 2. Default: '2' (manual) or the next free "
                             "integer id (--view_support auto).")
    parser.add_argument("--new_view_angle", default=None,
                        help="Camera 'azimuth_deg,elevation_deg' (blender frame) "
                             "for the newly-added round-2 view. Default: "
                             "'270.0,0.0' (ninja back view) for manual; 'auto' for "
                             "--view_support auto, which runs the 'ours' next-view "
                             "policy of plan_mv_fusion_next_view.py on round 1.")
    parser.add_argument("--force_render_new_view", action="store_true",
                        help="Re-render source_<new_view_id>.png even if it "
                             "already exists on disk.")
    parser.add_argument("--trust_views", default=None,
                        help="Comma-separated list of view ids to use as trust "
                             "views. Defaults to every view that has a "
                             "matching mask_region_<v> (or mask_region_new for "
                             "the new view in round 2).")
    parser.add_argument("--base_view", default=None,
                        help="View id to use as base view. Defaults to the "
                             "remaining non-trust view.")
    parser.add_argument("--fusion_mode", choices=("trust_region", "view_average"), default="trust_region",
                        help="mv_fusion velocity fusion mode. 'trust_region' keeps the current "
                             "confidence-region weighted fusion; 'view_average' ignores "
                             "confidence regions and averages all views equally.")
    parser.add_argument("--view_support", choices=("manual", "auto"), default="manual",
                        help="Where per-view trust regions come from. 'manual' (default): the "
                             "mask_region_<v> boxes in mask.glb / mask_2.glb. 'auto': geometry-derived "
                             "soft support chi_v computed by compute_view_support.py from the rough "
                             "mask_all, the fitted view cameras and the source/target images "
                             "(Route-Diagnose-Refine Eqs. 4-7). Every non-base view becomes a trust "
                             "view in acquisition order; outputs go to *_auto_support dirs.")
    parser.add_argument("--support_camera_json", type=Path, default=None,
                        help="Optional known cameras for --view_support auto "
                             "({view: ViewCamera fields}); views not listed are fitted to their source image.")
    # ---- round 2 with --view_support auto (Route-Diagnose-Refine round transition) ----
    parser.add_argument("--round1_out_dir", type=Path, default=None,
                        help="Round-2 auto: previous round's output root (default: the same variant's "
                             "_easy3e_out_round1*, else _easy3e_out_round1).")
    parser.add_argument("--round1_work_dir", type=Path, default=None,
                        help="Round-2 auto: previous round's work root (default: derived from --round1_out_dir).")
    parser.add_argument("--next_mask_dilate", type=int, default=2,
                        help="Round-2 auto: voxel dilation of the conflict set C before closing (Eq. 16).")
    parser.add_argument("--next_mask_close", type=int, default=2,
                        help="Round-2 auto: voxel closing radius applied after the dilation (Eq. 16).")
    parser.add_argument("--next_mask_latent_cell", type=int, default=4,
                        help="Round-2 auto: snap the next edit region to the latent cells it touches (4 = 16^3 latent "
                             "of a 64^3 grid; 1 = no snapping, i.e. the paper's voxel-level Eq. 16).")
    parser.add_argument("--next_latent_threshold", type=float, default=0.0,
                        help="Round-2 auto: latent-cell coverage above which the voxel edit region is editable in the "
                             "sparse-structure stage (0 = any covered cell; Easy3E's GLB masks use 0.5).")
    parser.add_argument("--keep_quiescent_editable", action="store_true",
                        help="Round-2 auto: do not remove the previous round's quiescent voxels U from the next "
                             "edit region (by default U joins the preserved set).")
    parser.add_argument("--edit_instruction", default=None,
                        help="Round-2 auto: text edit instruction used when synthesizing the new view's target.")
    parser.add_argument("--new_view_desc", default="a new viewpoint",
                        help="Round-2 auto: short description of the new camera for the synthesis prompt.")
    parser.add_argument("--new_view_keep", default="the same viewpoint as Image 1",
                        help="Round-2 auto: what must stay fixed about the new camera, for the synthesis prompt.")
    parser.add_argument("--new_view_target", type=Path, default=None,
                        help="Round-2 auto: use this edited image as the new view's target instead of synthesizing one.")
    parser.add_argument("--synthesis_model", default="gemini-3-pro-image-preview",
                        help="Round-2 auto: magic666 image model for new-view target synthesis (nano banana pro by "
                             "default; gemini-2.5-flash-image does not follow the multi-image reference prompt).")
    parser.add_argument("--synthesis_candidates", type=int, default=2,
                        help="Round-2 auto: number of framing-consistent candidates to generate before picking one.")
    parser.add_argument("--views", default=None,
                        help="Round 1: comma-separated subset of view ids to use (default: every "
                             "source_<v>/target_<v> pair).")
    parser.add_argument("--source_transform", type=Path, default=None,
                        help="Round 1: transforms.json (scale, offset) replacing the automatic normalization of "
                             "object.glb, e.g. transforms_compose.json from build_rough_edit_mask.py.")
    parser.add_argument("--pair_crop", action="store_true",
                        help="Align every target to its source's pixel frame and crop each source/target pair "
                             "with one shared box for the image conditions (needed when targets add parts "
                             "outside the source silhouette, e.g. object composition).")
    parser.add_argument("--feather_outward", action="store_true",
                        help="Feather the latent edit mask outward only in the sparse-structure FlowEdit; Easy3E's "
                             "default feather attenuates small edit regions (round-2 regions, added parts) to ~0.3.")
    parser.add_argument("--run_tag", default="",
                        help="Optional extra suffix for the output/work dirs (e.g. 'views01') to keep "
                             "several runs of the same variant apart.")
    parser.add_argument("--no_guidance", action="store_true", default=False,
                        help="Disable orthographic silhouette guidance. Guidance is "
                             "ENABLED by default; pass --no_guidance to turn it off "
                             "(e.g. when custom user views are not aligned to the "
                             "preprocess hammersley sequence).")
    parser.add_argument("--enable_guidance", action="store_true",
                        help="Force-enable guidance (redundant now that guidance is on "
                             "by default, kept for backwards compatibility). Overrides "
                             "--no_guidance if both are passed.")
    parser.add_argument("--force_preprocess", action="store_true",
                        help="Rerun preprocess even if render cache exists.")
    parser.add_argument("--velocity_conflict_threshold", default="auto",
                        help="Threshold for the trust-vs-base velocity conflict glb. "
                             "'auto' (default) uses mean+std; pass a positive float for a manual threshold.")
    parser.add_argument("--velocity_conflict_min_ratio", type=float, default=0.0,
                        help="If >0, at least this fraction of mask_all voxels will be flagged as conflict.")
    parser.add_argument("--export_edit_magnitude_conflict", action="store_true",
                        help="Export row-3 diagnostic heatmap/mask using only large fused edit magnitude, ignoring multi-view velocity disagreement.")
    parser.add_argument("--edit_magnitude_conflict_threshold", default="auto",
                        help="Threshold for the edit-magnitude-only conflict glb. 'auto' uses mean+std; pass a positive float for a manual threshold.")
    parser.add_argument("--edit_magnitude_conflict_min_ratio", type=float, default=0.0,
                        help="If >0, at least this fraction of mask_all voxels will be flagged in the edit-magnitude-only conflict mask.")
    # ---------------- Quiescent hard on/off switch (works for BOTH rounds) ----------------
    # Default: enabled (backwards-compatible). Use --disable_quiescent to fully turn off
    # quiescent detection/shrinkage in the current run, regardless of round or thresholds.
    parser.add_argument("--enable_quiescent", dest="quiescent_enabled",
                        action="store_true", default=True,
                        help="Enable quiescent detection & shrinkage for THIS round. "
                             "Default. Overrides any earlier --disable_quiescent on the same line.")
    parser.add_argument("--disable_quiescent", dest="quiescent_enabled",
                        action="store_false",
                        help="Disable quiescent entirely for THIS round: raw_mask stays intact "
                             "for the SLAT stage, no quiescent_* files are produced, and round-2 "
                             "will NOT subtract round-1 quiescent voxels from mask_all.")
    parser.add_argument("--seed", type=int, default=SEED, help="Easy3E seed (FlowEdit noise of both stages).")
    parser.add_argument("--state_mode", default="three_state",
                        choices=["three_state", "three_state_srcocc", "no_lockback", "lockback_nonconflict"],
                        help="Ablation of the diagnosis states locked back to the source before the SLAT stage: "
                             "three_state = quiescent only (default); three_state_srcocc = quiescent voxels also occupied in the "
                             "source; no_lockback = nothing (conflict / stable); "
                             "lockback_nonconflict = every non-conflict cell of mask_all (conflict / source).")
    parser.add_argument("--quiescent_ratio", type=float, default=0.0,
                        help="Upper-bound fraction of mask_all voxels to freeze as 'quiescent' "
                             "(velocity-stable + low-magnitude -> reuse source mesh). "
                             "0 (default) disables the ratio cap; then the two quiescent thresholds decide.")
    parser.add_argument("--quiescent_disagreement_threshold", default="auto",
                        help="Upper bound on multi-view velocity disagreement for quiescent voxels. "
                             "'auto' (default) uses the ~30th-percentile inside mask_all.")
    parser.add_argument("--quiescent_edit_mag_threshold", default="auto",
                        help="Upper bound on fused edit-magnitude for quiescent voxels. "
                             "'auto' (default) uses the ~30th-percentile inside mask_all.")
    parser.add_argument("--round2_shrink_mask_all_by_quiescent", action="store_true", default=True,
                        help="Round-2 only: after building mask_all from round-1 conflict voxels, "
                             "additionally subtract round-1 quiescent voxels from the source glb's "
                             "editable region. Enabled by default; use --no_round2_shrink to disable.")
    parser.add_argument("--no_round2_shrink", dest="round2_shrink_mask_all_by_quiescent",
                        action="store_false", help="Disable round-2 quiescent-based shrinkage of mask_all.")
    parser.add_argument("--skip_next_view_acquisition", action="store_true",
                        help="Do not run the post-round next-view acquisition planner.")
    parser.add_argument("--next_view_methods", default="ours,random,farthest_camera,whole_object_coverage",
                        help="Comma-separated next-view policies to evaluate after this round.")
    parser.add_argument("--next_view_candidate_views", default=None,
                        help="Optional semicolon-separated candidate views: 'az,el;az,el' or 'id,az,el;...'. If omitted, use the free-angle grid below.")
    parser.add_argument("--next_view_candidate_azimuth_min", type=float, default=0.0,
                        help="Minimum azimuth in degrees for next-view free-angle grid.")
    parser.add_argument("--next_view_candidate_azimuth_max", type=float, default=355.0,
                        help="Maximum azimuth in degrees for next-view free-angle grid.")
    parser.add_argument("--next_view_candidate_azimuth_step", type=float, default=5.0,
                        help="Azimuth step in degrees for next-view free-angle grid.")
    parser.add_argument("--next_view_candidate_elevation_min", type=float, default=-30.0,
                        help="Minimum elevation in degrees for next-view free-angle grid.")
    parser.add_argument("--next_view_candidate_elevation_max", type=float, default=30.0,
                        help="Maximum elevation in degrees for next-view free-angle grid.")
    parser.add_argument("--next_view_candidate_elevation_step", type=float, default=5.0,
                        help="Elevation step in degrees for next-view free-angle grid.")
    parser.add_argument("--next_view_existing_angles", default=None,
                        help="Optional semicolon-separated existing view angles 'az,el;az,el'. If omitted, infer from run_summary and edit_views/camera.")
    parser.add_argument("--next_view_random_seed", type=int, default=1,
                        help="Random seed for the Table-4 random next-view baseline.")
    parser.add_argument("--next_view_rho", type=float, default=0.15,
                        help="Redundancy penalty for ours: unresolved coverage - rho * overlap.")
    parser.add_argument("--next_view_render_resolution", type=int, default=1024,
                        help="Blender render resolution for selected next views.")
    parser.add_argument("--next_view_render_samples", type=int, default=128,
                        help="Cycles samples for selected next-view renders.")
    parser.add_argument("--next_view_render_shrink", type=float, default=1.0,
                        help="Normalize rendered mesh so the longest bbox side equals this value.")
    parser.add_argument("--next_view_render_ortho_scale", type=float, default=1.2,
                        help="Orthographic camera scale for selected next-view renders.")
    parser.add_argument("--next_view_skip_render", action="store_true",
                        help="Only write next-view scores/angles; do not run Blender renders.")
    args = parser.parse_args()

    item_dir: Path = args.item_dir.expanduser().resolve()
    _require(item_dir, "item directory")

    # ---------------- resolve per-round input layout ----------------
    if args.round == 1:
        cfg = _resolve_round1_config(
            item_dir, [v.strip() for v in args.views.split(",")] if args.views else None,
            args.source_transform.expanduser().resolve() if args.source_transform else None)
    elif args.view_support == "auto":
        cfg = _resolve_round2_auto_config(item_dir, args)
    else:
        cfg = _resolve_round2_config(item_dir, args)

    view_ids = cfg["view_ids"]
    output_root: Path = cfg["output_root"]
    work_root: Path = cfg["work_root"]
    suffix = _run_suffix(args)
    if suffix:
        output_root = item_dir / f"_easy3e_out_round{args.round}{suffix}"
        work_root = item_dir / f"_easy3e_work_round{args.round}{suffix}"
    source_glb: Path = cfg["source_glb"]
    source_transform_path = cfg.get("source_transform_path")

    # If the round-2 config asked for an identity transforms.json, materialize
    # it under work_root (mask normalization + preprocess both read this file).
    if source_transform_path == "IDENTITY":
        work_root.mkdir(parents=True, exist_ok=True)
        identity_path = work_root / "identity_transforms.json"
        identity_path.write_text(json.dumps({
            "scale": 1.0,
            "offset": [0.0, 0.0, 0.0],
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "frames": [],
        }, indent=2))
        source_transform_path = identity_path
        print(f"[driver] Round 2: using IDENTITY transform at {identity_path}")

    print(f"[driver] Round {args.round}: views={view_ids}, source_glb={source_glb}, "
          f"output_root={output_root}")

    # ---------------- resolve trust / base views ----------------
    # A single view runs plain single-view Easy3E FlowEdit (no mv_fusion, no trust regions).
    single_view = len(view_ids) == 1
    if single_view:
        trust_views, base_view = [], view_ids[0]
    elif args.trust_views is not None:
        trust_views = [t.strip() for t in args.trust_views.split(",") if t.strip()]
    elif args.view_support == "auto":
        # Acquisition order: base (primary) view first, every other view is a trust
        # view routed by its geometry-derived support; later views win on overlap.
        base_first = args.base_view if args.base_view is not None else view_ids[0]
        trust_views = [v for v in view_ids if v != base_first]
    else:
        trust_views = [v for v in view_ids if v in cfg["region_glbs"]]
        if not trust_views:
            trust_views = [view_ids[0]]
    if single_view:
        pass
    elif args.base_view is not None:
        base_view = args.base_view
    else:
        remaining = [v for v in view_ids if v not in trust_views]
        if not remaining:
            raise ValueError(
                f"No candidate base view left after picking trust={trust_views}; "
                f"views={view_ids}"
            )
        base_view = remaining[0]
    for tv in trust_views:
        if tv not in view_ids:
            raise ValueError(f"trust view '{tv}' not in views {view_ids}")
    if base_view not in view_ids or base_view in trust_views:
        raise ValueError(
            f"Invalid base_view='{base_view}' (trust_views={trust_views}, views={view_ids})"
        )
    print(f"[driver] trust_views={trust_views}, base_view={base_view}")

    # ---------------- prepare working directories ----------------
    render_dir = work_root / "render"
    feature_dir = output_root / "features"
    output_dir = output_root / "out"
    log_dir = output_root / "logs"
    images_dir = work_root / "images"
    masks_dir = work_root / "3Dmasks"
    ori_dir = work_root / "ori"
    for d in [work_root, output_root, render_dir, feature_dir, output_dir,
              log_dir, images_dir, masks_dir, ori_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- stage source + target images ----------------
    staged_sources = {}
    staged_targets_rgba = {}
    for v in view_ids:
        # Round-2 auto provides re-rendered sources and the synthesized new-view target.
        src_png_in = cfg.get("source_paths", {}).get(v, item_dir / f"source_{v}.png")
        tgt_png_in = cfg.get("target_paths", {}).get(v, item_dir / f"target_{v}.png")
        _require(src_png_in, f"source_{v}.png")
        _require(tgt_png_in, f"target_{v}.png")
        src_png_out = images_dir / f"source_{v}.png"
        tgt_rgba_out = images_dir / f"target_{v}_rgba.png"
        shutil.copy2(src_png_in, src_png_out)
        _write_easy3e_rgba(tgt_png_in, tgt_rgba_out)
        if args.pair_crop:
            # Round 2 aligns to the original source render: it has the same camera framing as the
            # re-rendered source but the asset's original appearance, which the target matches.
            _align_target_to_source(cfg.get("alignment_sources", {}).get(v, src_png_out), tgt_rgba_out)
        staged_sources[v] = src_png_out
        staged_targets_rgba[v] = tgt_rgba_out

    # ---------------- prepare mask_all + per-trust-view region GLBs ----------------
    mask_voxels = cfg.get("mask_voxels")
    if mask_voxels is not None:
        mask_all_glb = None  # the edit region is already a voxel set (round-2 auto)
    elif args.round == 1:
        mask_all_glb = masks_dir / "mask_all.glb"
        ok = _extract_named_geometry(cfg["mask_all_source_glb"], "mask_all", mask_all_glb)
        if not ok:
            raise RuntimeError("Could not extract 'mask_all' from round-1 mask.glb")
    else:
        # Round 2: turn velocity_conflict_*_mask.glb into a new mask_all.glb.
        mask_all_glb = masks_dir / "mask_all.glb"
        kept = _conflict_mask_to_mask_all_glb(
            cfg["conflict_mask_glb"], mask_all_glb,
        )
        if kept <= 0:
            raise RuntimeError(
                f"Round 2 mask_all is empty (no conflict voxels in "
                f"{cfg['conflict_mask_glb']}). Aborting."
            )
        # Optional: subtract round-1 quiescent voxels so quiescent regions are
        # frozen in round 2 as well (the source glb for round 2 already contains
        # the correct mesh there; we just don't want to edit those voxels again).
        if not args.quiescent_enabled:
            print("[driver] quiescent disabled for this round -> "
                  "skipping mask_all shrinkage by round-1 quiescent voxels.")
        elif args.round2_shrink_mask_all_by_quiescent:
            r1_quiescent_glb = cfg.get("quiescent_only_glb")
            if r1_quiescent_glb is not None and Path(r1_quiescent_glb).exists():
                try:
                    _subtract_quiescent_from_mask_all_glb(
                        mask_all_glb, Path(r1_quiescent_glb), mask_all_glb,
                    )
                except Exception as e:
                    print(f"[driver][WARN] Failed to subtract quiescent voxels from mask_all: {e}")
            else:
                print("[driver] Round-1 quiescent_voxels_only.glb not found; "
                      "skipping mask_all shrinkage. (Round 1 may predate quiescent detection.)")

    # per-trust-view region GLBs, in the SAME ORDER as trust_views (this order
    # matters: later trust views win on overlap). Auto view support replaces them.
    region_glbs_for_cmd = []
    trust_views_for_cmd = []
    for tv in trust_views:
        if args.view_support == "auto":
            trust_views_for_cmd.append(tv)
            continue
        entry = cfg["region_glbs"].get(tv)
        if entry is None:
            print(f"[driver][WARN] trust view '{tv}' has no confidence region; skipping.")
            continue
        src_glb, node, prenorm = entry
        out_glb = masks_dir / f"mask_region_{tv}.glb"
        ok = _extract_named_geometry(src_glb, node, out_glb, prenormalize=prenorm)
        if not ok:
            raise RuntimeError(
                f"Could not extract node '{node}' from {src_glb} for trust view {tv}"
            )
        if prenorm is not None:
            print(f"[driver] Pre-normalized {node} from {src_glb.name} -> "
                  f"{out_glb.name} (scale={prenorm[0]}, offset={prenorm[1]})")
        region_glbs_for_cmd.append(out_glb)
        trust_views_for_cmd.append(tv)
    if not trust_views_for_cmd and not single_view:
        raise RuntimeError(
            "No trust view has a confidence region. Please provide mask_region_<v> "
            "nodes in mask.glb / mask_2.glb, or pass --trust_views explicitly."
        )

    # ---------------- primary target image (drives azimuth heuristic) ----------------
    # In Easy3E, azimuth for guidance is derived from `--original_image`'s stem
    # (must be int). We alias the primary trust source to `<primary_trust>.png`.
    primary_trust = trust_views_for_cmd[0] if trust_views_for_cmd else base_view
    primary_target = staged_targets_rgba[primary_trust]
    try:
        int(primary_trust)
        ori_name = f"{primary_trust}.png"
    except ValueError:
        ori_name = "0.png"
    original_image = ori_dir / ori_name
    shutil.copy2(staged_sources[primary_trust], original_image)

    # ---------------- build inference command ----------------
    # Guidance is ON by default. --no_guidance disables it; --enable_guidance
    # force-enables it (overrides --no_guidance if both are passed).
    use_guidance = (not args.no_guidance) or args.enable_guidance
    src_csv = ",".join(str(staged_sources[v]) for v in view_ids)
    tgt_csv = ",".join(str(staged_targets_rgba[v]) for v in view_ids)
    views_csv = ",".join(view_ids)
    trust_views_csv = ",".join(trust_views_for_cmd)
    region_glbs_csv = ",".join(str(p) for p in region_glbs_for_cmd)

    command = [
        str(PYTHON_BIN), "-u", str(EASY3E_ROOT / "inference.py"),
        "--input_image", str(primary_target),
        "--output_dir", str(output_dir),
        "--root_dir", str(work_root),
        "--render_dir", str(render_dir),
        "--feature_dir", str(feature_dir),
        "--obj_path", str(source_glb),
        *(["--mask_voxels", str(mask_voxels), "--mask_voxels_latent_threshold", str(args.next_latent_threshold)]
          if mask_voxels is not None else ["--mask_path", str(mask_all_glb)]),
        "--mode", "edit",
        "--original_image", str(original_image),
        "--checkpoint", str(CHECKPOINT),
        "--enc_pretrained", str(ENC_PRETRAINED),
        "--seed", str(args.seed),
        "--cfg_src", str(CFG_SRC),
        "--cfg_tar", str(CFG_TAR),
        "--flowedit_lambda", str(FLOWEDIT_LAMBDA),
    ]
    mv_args = [
        "--mv_fusion",
        "--mv_fusion_source_images", src_csv,
        "--mv_fusion_target_images", tgt_csv,
        "--mv_fusion_view_names", views_csv,
        "--mv_fusion_trust_views", trust_views_csv,
        "--mv_fusion_base_view", base_view,
        "--mv_fusion_fusion_mode", args.fusion_mode,
        "--mv_fusion_conf_box_glbs", region_glbs_csv,
        "--mv_fusion_alpha_inside", str(MV_FUSION_ALPHA_INSIDE),
        "--mv_fusion_alpha_outside", str(MV_FUSION_ALPHA_OUTSIDE),
        "--mv_fusion_soft_blend",
        "--mv_fusion_soft_edge_width", str(MV_FUSION_SOFT_EDGE_WIDTH),
        "--mv_fusion_soft_core_width", str(MV_FUSION_SOFT_CORE_WIDTH),
        "--mv_fusion_soft_outside_width", str(MV_FUSION_SOFT_OUTSIDE_WIDTH),
        "--mv_fusion_soft_inside_plateau", str(MV_FUSION_SOFT_INSIDE_PLATEAU),
        "--mv_fusion_trust_region_step", str(MV_FUSION_TRUST_REGION_STEP),
        "--mv_fusion_steps", MV_FUSION_STEPS,
        "--mv_fusion_velocity_conflict_threshold", str(args.velocity_conflict_threshold),
        "--mv_fusion_velocity_conflict_min_ratio", str(args.velocity_conflict_min_ratio),
        "--mv_fusion_edit_magnitude_conflict_threshold", str(args.edit_magnitude_conflict_threshold),
        "--mv_fusion_edit_magnitude_conflict_min_ratio", str(args.edit_magnitude_conflict_min_ratio),
        "--mv_fusion_quiescent_ratio", str(args.quiescent_ratio),
        "--mv_fusion_state_mode", args.state_mode,
        "--mv_fusion_quiescent_disagreement_threshold", str(args.quiescent_disagreement_threshold),
        "--mv_fusion_quiescent_edit_mag_threshold", str(args.quiescent_edit_mag_threshold),
    ]
    if args.export_edit_magnitude_conflict:
        mv_args += ["--mv_fusion_export_edit_magnitude_conflict"]
    # Hard on/off switch for quiescent (default: on). Always append explicitly so
    # the inference-side default cannot silently override the driver's choice.
    if args.quiescent_enabled:
        mv_args += ["--mv_fusion_enable_quiescent"]
    else:
        mv_args += ["--mv_fusion_disable_quiescent"]
    if not single_view:
        command += mv_args
    if source_transform_path is not None:
        command += ["--source_transform_path", str(source_transform_path)]
    if not use_guidance:
        command += ["--no_guidance"]
    view_support_dir = output_root / "view_support"
    if args.view_support == "auto" and not single_view:
        command += ["--mv_fusion_support_npz", str(view_support_dir / "view_support.npz")]
    if args.pair_crop:
        command += ["--mv_fusion_pair_crop"]
    if args.feather_outward:
        command += ["--mask_feather_outward"]

    # ---- Method D: round-2 reuses round-1's final edited SLAT as z_src ----
    # This skips the lossy encoder path (glb -> render 150 views -> dinov2 ->
    # sparse-conv encoder), so mask-outside SLAT stays byte-identical to what
    # round-1's decoder saw. Round-1 still runs the encoder normally.
    if args.round == 2:
        r1_slat = cfg.get("round1_slat_edited")
        if r1_slat is not None and Path(r1_slat).exists():
            command += ["--slat_latent_override", str(r1_slat)]
            print(f"[driver] Round-2 will REUSE round-1 final SLAT: {r1_slat} "
                  f"(bypasses re-encoder).")
        else:
            print(f"[driver][WARN] round-1 slat_edited.pt not found at {r1_slat}; "
                  f"round-2 will fall back to encoder-derived SLAT. "
                  f"(Round-1 may predate the slat-reuse feature; rerun round-1 "
                  f"to generate it.)")
        # Structure state continuation (x_s^2 <- x_edit^1): the saved round-1 latent if it
        # exists, otherwise re-encode the occupancy of the reused SLAT coords.
        if args.view_support == "auto":
            if cfg.get("round1_ss_latent") is not None:
                command += ["--ss_latent_override", str(cfg["round1_ss_latent"])]
            else:
                command += ["--ss_latent_from_slat_override"]

    # ---------------- preprocess if render cache is missing ----------------
    required_cache = [
        render_dir / "transforms.json",
        render_dir / "mesh.ply",
        render_dir / "feature.npz",
        render_dir / "ss_latent.pt",
        render_dir / "slat_latent.pt",
    ]
    need_preprocess = args.force_preprocess or any(
        (not p.exists()) or p.stat().st_size == 0 for p in required_cache
    )
    # Round-1 variants (e.g. *_auto_support) edit the same object.glb with the same
    # normalization as the default run, so its preprocess cache can be copied.
    canonical_render_dir = cfg["work_root"] / "render"
    if (need_preprocess and not args.force_preprocess and args.round == 1
            and source_transform_path is None
            and canonical_render_dir != render_dir and _render_cache_complete(canonical_render_dir)):
        print(f"[driver] Reusing preprocess cache from {canonical_render_dir}")
        for src_f in canonical_render_dir.iterdir():
            if src_f.is_file() and src_f.suffix.lower() != ".png":
                shutil.copy2(src_f, render_dir / src_f.name)
        need_preprocess = False
    if need_preprocess:
        pre_command = [
            str(PYTHON_BIN), "-u", str(EASY3E_ROOT / "inference.py"),
            "--preprocess",
            "--root_dir", str(work_root),
            "--render_dir", str(render_dir),
            "--feature_dir", str(feature_dir),
            "--obj_path", str(source_glb),
            "--original_image", str(original_image),
            "--checkpoint", str(CHECKPOINT),
            "--enc_pretrained", str(ENC_PRETRAINED),
        ]
        if source_transform_path is not None:
            pre_command += ["--source_transform_path", str(source_transform_path)]
        print(f"[driver] Running preprocess: {' '.join(pre_command)}")
        pre_log = log_dir / "preprocess.log"
        _run_and_log(pre_command, pre_log)

    for p in required_cache:
        _require(p, f"render cache {p.name}")

    # ---------------- geometry-derived per-view support (auto) ----------------
    if args.view_support == "auto" and not single_view:
        support_views = [base_view] + trust_views_for_cmd
        support_cmd = [
            str(PYTHON_BIN), "-u", str(VIEW_SUPPORT_SCRIPT),
            "--render_dir", str(render_dir),
            *(["--mask_npy", str(mask_voxels)] if mask_voxels is not None else ["--mask_glb", str(mask_all_glb)]),
            "--views", ",".join(support_views),
            "--source_images", ",".join(str(staged_sources[v]) for v in support_views),
            "--target_images", ",".join(str(staged_targets_rgba[v]) for v in support_views),
            "--output_dir", str(view_support_dir),
        ]
        if args.pair_crop:
            support_cmd += ["--targets_aligned"]
        camera_json = args.support_camera_json or cfg.get("camera_json")
        if camera_json is not None:
            support_cmd += ["--camera_json", str(camera_json)]
        print(f"[driver] Computing view support: {' '.join(support_cmd)}")
        _run_and_log(support_cmd, log_dir / "view_support.log")

    # ---------------- write run summary ----------------
    summary = {
        "round": args.round,
        "item_dir": str(item_dir),
        "views": view_ids,
        "trust_views": trust_views_for_cmd,
        "base_view": base_view,
        "fusion_mode": args.fusion_mode,
        "view_support": args.view_support,
        "view_support_npz": str(view_support_dir / "view_support.npz") if args.view_support == "auto" else None,
        "source_glb": str(source_glb),
        "mask_all_glb": str(mask_all_glb) if mask_all_glb is not None else None,
        "mask_voxels": str(mask_voxels) if mask_voxels is not None else None,
        "round_state": cfg.get("round_state"),
        "region_glbs": [str(p) for p in region_glbs_for_cmd],
        "source_transform_path": str(source_transform_path) if source_transform_path else None,
        "primary_target": str(primary_target),
        "original_image": str(original_image),
        "use_guidance": use_guidance,
        "new_view_id": cfg.get("new_view_id") if args.round == 2 else None,
        "new_view_angle": cfg.get("new_view_angle", args.new_view_angle) if args.round == 2 else None,
        "command": command,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- run edit ----------------
    edit_log = log_dir / "edit.log"
    print(f"[driver] Running edit: log -> {edit_log}")
    _run_and_log(command, edit_log)

    # ---------------- copy velocity conflict glb next to the final result ----------------
    # Copy every velocity_conflict_* {heatmap.glb, mask.glb, stats.json} triple:
    #   * The aggregate multi-view heatmap (backward-compatible name for round 1).
    #   * The per-pair heatmaps that only appear when N_views >= 3 (round 2+).
    velconf_src_dir = feature_dir / "mv_fusion_voxel_diff"
    if velconf_src_dir.exists():
        velconf_files = sorted(velconf_src_dir.glob("velocity_conflict_*"))
        if velconf_files:
            for src_f in velconf_files:
                dst_f = output_dir / src_f.name
                try:
                    shutil.copy2(src_f, dst_f)
                    print(f"[driver] Copied {src_f.name} -> {dst_f}")
                except Exception as e:
                    print(f"[driver][WARN] Failed to copy {src_f}: {e}")
        else:
            print(f"[driver][WARN] No velocity_conflict_* under {velconf_src_dir}")

        # Also copy edit_magnitude_conflict_* files for the row-3 diagnostic setting.
        editmag_files = sorted(velconf_src_dir.glob("edit_magnitude_conflict_*"))
        if editmag_files:
            for src_f in editmag_files:
                dst_f = output_dir / src_f.name
                try:
                    shutil.copy2(src_f, dst_f)
                    print(f"[driver] Copied {src_f.name} -> {dst_f}")
                except Exception as e:
                    print(f"[driver][WARN] Failed to copy {src_f}: {e}")
        else:
            print(f"[driver][WARN] No edit_magnitude_conflict_* under {velconf_src_dir}")

        # Also copy quiescent_* files (voxels.glb, voxels_only.glb, stats.json).
        quiescent_files = sorted(velconf_src_dir.glob("quiescent_*"))
        if quiescent_files:
            for src_f in quiescent_files:
                dst_f = output_dir / src_f.name
                try:
                    shutil.copy2(src_f, dst_f)
                    print(f"[driver] Copied {src_f.name} -> {dst_f}")
                except Exception as e:
                    print(f"[driver][WARN] Failed to copy {src_f}: {e}")
        else:
            print(f"[driver][WARN] No quiescent_* under {velconf_src_dir}")

    # ---------------- next-view acquisition after the completed round ----------------
    # This implements the paper's active view-acquisition step and also the
    # Table-4 baselines. It only plans/renders candidate next views; it does not
    # launch the next editing round or require a new target image.
    if not args.skip_next_view_acquisition:
        planner_out = output_root / f"next_view_acquisition_round{args.round}"
        planner_cmd = [
            str(PYTHON_BIN), "-u", str(NEXT_VIEW_PLANNER),
            "--item-dir", str(item_dir),
            "--run-dir", str(output_root),
            "--work-dir", str(work_root),
            "--round", str(args.round),
            "--output-dir", str(planner_out),
            "--methods", str(args.next_view_methods),
            "--random-seed", str(args.next_view_random_seed),
            "--rho", str(args.next_view_rho),
            "--candidate-azimuth-min", str(args.next_view_candidate_azimuth_min),
            "--candidate-azimuth-max", str(args.next_view_candidate_azimuth_max),
            "--candidate-azimuth-step", str(args.next_view_candidate_azimuth_step),
            "--candidate-elevation-min", str(args.next_view_candidate_elevation_min),
            "--candidate-elevation-max", str(args.next_view_candidate_elevation_max),
            "--candidate-elevation-step", str(args.next_view_candidate_elevation_step),
            "--render-resolution", str(args.next_view_render_resolution),
            "--render-samples", str(args.next_view_render_samples),
            "--render-shrink", str(args.next_view_render_shrink),
            "--render-ortho-scale", str(args.next_view_render_ortho_scale),
        ]
        if args.next_view_candidate_views:
            planner_cmd += ["--candidate-views", str(args.next_view_candidate_views)]
        if args.next_view_existing_angles:
            planner_cmd += ["--existing-view-angles", str(args.next_view_existing_angles)]
        if args.next_view_skip_render:
            planner_cmd += ["--skip-render"]
        print(f"[driver] Running next-view acquisition: {' '.join(planner_cmd)}")
        _run_and_log(planner_cmd, log_dir / "next_view_acquisition.log")
        print(f"[driver] Next-view plan: {planner_out / 'next_view_plan.json'}")

    print(f"[driver] Done. Outputs under: {output_root}")


if __name__ == "__main__":
    main()
