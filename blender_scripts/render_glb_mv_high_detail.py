"""High-detail GLB renderer for TRELLIS sparse-structure / voxel meshes.

This script keeps the camera convention from the user-provided Blender script:
8 orthographic character views, camera parameter export with _K_ortho/_RT_3x4,
and high-detail lighting / voxel bevel controls.  It is CLI-compatible with
blender_scripts/render_glb_mv.py so evaluation scripts can call it via
Blender in background mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Generator, Optional, Tuple

import bpy
import numpy as np
from mathutils import Vector


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

ENV_STRENGTH = 0.35
KEY_ENERGY = 420.0
FILL_ENERGY = 70.0
RIM_ENERGY = 120.0
KEY_SIZE = 1.2
FILL_SIZE = 2.5
RIM_SIZE = 1.5
VOXEL_BEVEL_RATIO = 0.07
VOXEL_BEVEL_SEGMENTS = 1
VOXEL_BEVEL_ANGLE_DEG = 5.0
MIN_ROUGHNESS = 0.35
PIXEL_FILTER_WIDTH = 0.50
EXPOSURE = -0.6


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_views", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--ortho_scale", type=float, default=1.2)
    ap.add_argument("--engine", type=str, default="CYCLES")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--cycles_device", type=str, default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--view_azimuths", type=str, default="")
    ap.add_argument("--view_elevations", type=str, default="")
    ap.add_argument("--view_ids", type=str, default="")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return ap.parse_args(argv)


ARGS = _parse_args()


def _reset_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for tex in list(bpy.data.textures):
        bpy.data.textures.remove(tex, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)


def _load_object(path: str) -> None:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True, import_shading="NORMALS")
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == "obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    elif ext == "ply":
        bpy.ops.import_mesh.ply(filepath=path)
    elif ext == "stl":
        bpy.ops.import_mesh.stl(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh extension: {path}")


def _scene_meshes() -> Generator:
    for obj in bpy.context.scene.objects.values():
        if isinstance(getattr(obj, "data", None), bpy.types.Mesh):
            yield obj


def _scene_bbox(single: Optional[bpy.types.Object] = None, ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    bmin = Vector((math.inf, math.inf, math.inf))
    bmax = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    items = list(_scene_meshes()) if single is None else [single]
    for obj in items:
        found = True
        for c in obj.bound_box:
            p = Vector(c)
            if not ignore_matrix:
                p = obj.matrix_world @ p
            bmin = Vector(tuple(min(a, b) for a, b in zip(bmin, p)))
            bmax = Vector(tuple(max(a, b) for a, b in zip(bmax, p)))
    if not found:
        raise RuntimeError("No mesh found in current scene.")
    return bmin, bmax


def _normalize_scene(shrink: float) -> tuple[float, Vector]:
    if shrink <= 0:
        return 1.0, Vector((0.0, 0.0, 0.0))
    roots = [o for o in bpy.context.scene.objects.values() if not o.parent]
    if not roots:
        raise RuntimeError("Empty scene.")
    if len(roots) > 1:
        parent = bpy.data.objects.new("_NormalizeRoot", None)
        bpy.context.scene.collection.objects.link(parent)
        for obj in roots:
            obj.parent = parent
    else:
        parent = roots[0]

    bmin, bmax = _scene_bbox()
    longest = max(bmax - bmin)
    if longest <= 0:
        raise RuntimeError("Degenerate bbox.")
    scale = float(shrink) / float(longest)
    parent.scale = parent.scale * scale
    bpy.context.view_layer.update()

    bmin, bmax = _scene_bbox()
    offset = -(bmin + bmax) / 2.0
    parent.matrix_world.translation += offset
    bpy.context.view_layer.update()
    return scale, offset


def _force_flat_shading() -> None:
    for obj in _scene_meshes():
        for poly in obj.data.polygons:
            poly.use_smooth = False
    bpy.context.view_layer.update()


def _estimate_voxel_edge_local(obj) -> float | None:
    mesh = obj.data
    if len(mesh.edges) == 0:
        return None
    axis_lengths = []
    all_lengths = []
    max_edges = 250000
    for i, edge in enumerate(mesh.edges):
        if i >= max_edges:
            break
        p0 = mesh.vertices[edge.vertices[0]].co
        p1 = mesh.vertices[edge.vertices[1]].co
        d = p1 - p0
        length = d.length
        if length <= 1e-10:
            continue
        all_lengths.append(length)
        comps = [abs(d.x), abs(d.y), abs(d.z)]
        major = max(comps)
        minor = sorted(comps)[:2]
        if major > 0 and minor[1] <= major * 1e-4:
            axis_lengths.append(length)
    lengths = axis_lengths if len(axis_lengths) >= 8 else all_lengths
    if not lengths:
        return None
    arr = np.asarray(lengths, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 1e-10)]
    if len(arr) == 0:
        return None
    arr.sort()
    n = max(1, int(len(arr) * 0.25))
    return float(np.median(arr[:n]))


def _add_voxel_bevels() -> None:
    for obj in _scene_meshes():
        old = obj.modifiers.get("_VoxelDetailBevel")
        if old is not None:
            obj.modifiers.remove(old)
        voxel_edge = _estimate_voxel_edge_local(obj)
        if voxel_edge is None:
            print(f"[voxel] {obj.name}: could not estimate voxel edge; skip bevel")
            continue
        width = voxel_edge * VOXEL_BEVEL_RATIO
        mod = obj.modifiers.new("_VoxelDetailBevel", "BEVEL")
        mod.width = width
        mod.segments = VOXEL_BEVEL_SEGMENTS
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(VOXEL_BEVEL_ANGLE_DEG)
        if hasattr(mod, "affect"):
            mod.affect = "EDGES"
        if hasattr(mod, "use_clamp_overlap"):
            mod.use_clamp_overlap = True
        if hasattr(mod, "harden_normals"):
            mod.harden_normals = False
        print(f"[voxel] {obj.name}: edge~{voxel_edge:.6g}, bevel={width:.6g}")


def _material_uses_image_texture(mat) -> bool:
    if mat is None or not mat.use_nodes:
        return False
    return any(node.type == "TEX_IMAGE" for node in mat.node_tree.nodes)


def _ensure_vertex_color_materials() -> None:
    """Use mesh vertex colors for colored sparse voxel GLBs when no texture exists."""
    for obj in _scene_meshes():
        color_attrs = getattr(obj.data, "color_attributes", None)
        legacy_colors = getattr(obj.data, "vertex_colors", None)
        attr_name = None
        if color_attrs and len(color_attrs) > 0:
            attr_name = color_attrs[0].name
        elif legacy_colors and len(legacy_colors) > 0:
            attr_name = legacy_colors[0].name
        if not attr_name:
            continue

        has_textured_material = any(_material_uses_image_texture(slot.material) for slot in obj.material_slots)
        if has_textured_material:
            continue

        mat = bpy.data.materials.new(f"{obj.name}_VertexColor")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        attr = nodes.new(type="ShaderNodeAttribute")
        attr.attribute_name = attr_name
        color_output = attr.outputs.get("Color")
        base_color = bsdf.inputs.get("Base Color")
        if color_output is not None and base_color is not None:
            mat.node_tree.links.new(color_output, base_color)
        rough = bsdf.inputs.get("Roughness")
        if rough is not None and not rough.is_linked:
            rough.default_value = max(float(rough.default_value), MIN_ROUGHNESS)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        print(f"[material] {obj.name}: using vertex color attribute '{attr_name}'")


def _tune_materials_without_touching_textures() -> None:
    seen = set()
    for obj in _scene_meshes():
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen or not mat.use_nodes:
                continue
            seen.add(mat.name)
            for node in mat.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                rough = node.inputs.get("Roughness")
                if rough is not None and not rough.is_linked:
                    rough.default_value = max(float(rough.default_value), MIN_ROUGHNESS)


def _init_world_light() -> None:
    for obj in [x for x in bpy.context.scene.objects if x.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = ENV_STRENGTH


def _init_render_engine() -> str:
    scene = bpy.context.scene
    scene.view_settings.look = "AgX - Base Contrast"
    scene.view_settings.exposure = EXPOSURE
    scene.view_settings.gamma = 1.0
    render = scene.render
    available = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    engine = ARGS.engine if ARGS.engine in available else ("CYCLES" if "CYCLES" in available else next(iter(available)))
    render.engine = engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = ARGS.resolution
    render.resolution_y = ARGS.resolution
    render.resolution_percentage = 100
    render.film_transparent = True
    render.threads_mode = "FIXED"
    render.threads = ARGS.threads
    if engine == "CYCLES":
        cycles = scene.cycles
        cycles.samples = ARGS.samples
        cycles.use_denoising = False
        cycles.filter_type = "BOX"
        cycles.filter_width = PIXEL_FILTER_WIDTH
        cycles.diffuse_bounces = 2
        cycles.glossy_bounces = 2
        cycles.transparent_max_bounces = 3
        cycles.transmission_bounces = 3
        cycles.adaptive_threshold = 0.0
        if ARGS.cycles_device == "GPU":
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.compute_device_type = "CUDA"
                prefs.get_devices()
                cycles.device = "GPU"
            except Exception as exc:
                print(f"[render] GPU setup failed ({exc}); using CPU.")
                cycles.device = "CPU"
        else:
            cycles.device = "CPU"
    return engine


def _make_ortho_camera():
    for obj in [x for x in bpy.context.scene.objects if x.type == "CAMERA"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    cam_data = bpy.data.cameras.new("RenderCam")
    cam = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ARGS.ortho_scale
    cam.data.lens = 35
    cam.data.sensor_height = 32
    cam.data.sensor_width = 32
    empty = bpy.data.objects.new("_CamTarget", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = (0.0, 0.0, 0.0)
    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    constraint.target = empty
    return cam


def _place_camera(cam, azimuth_deg: float, elevation_deg: float, distance: float) -> None:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cam.location = (
        distance * math.cos(azimuth) * math.cos(elevation),
        distance * math.sin(azimuth) * math.cos(elevation),
        distance * math.sin(elevation),
    )


def _new_area_light(name: str, energy: float, size: float):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _look_at(obj, target=Vector((0.0, 0.0, 0.0))) -> None:
    direction = target - obj.location
    if direction.length > 1e-8:
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _make_light_rig() -> dict:
    return {
        "key": _new_area_light("_VoxelKey", KEY_ENERGY, KEY_SIZE),
        "fill": _new_area_light("_VoxelFill", FILL_ENERGY, FILL_SIZE),
        "rim": _new_area_light("_VoxelRim", RIM_ENERGY, RIM_SIZE),
    }


def _update_light_rig(cam, rig: dict) -> None:
    bpy.context.view_layer.update()
    matrix = cam.matrix_world.to_3x3()
    right = (matrix @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (matrix @ Vector((0.0, 1.0, 0.0))).normalized()
    forward = (matrix @ Vector((0.0, 0.0, -1.0))).normalized()
    cam_pos = cam.matrix_world.translation.copy()
    rig["key"].location = cam_pos - 0.85 * right + 0.95 * up + 0.55 * forward
    rig["fill"].location = cam_pos + 1.05 * right + 0.15 * up + 0.45 * forward
    rig["rim"].location = 1.35 * forward + 0.8 * up - 0.25 * right
    for light in rig.values():
        _look_at(light)
    bpy.context.view_layer.update()


def _K_ortho(ortho_scale: float) -> np.ndarray:
    scene = bpy.context.scene
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    par = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    fx = rx / ortho_scale
    fy = ry / ortho_scale / par
    return np.asarray([[fx, 0, rx / 2], [0, fy, ry / 2], [0, 0, 1]], dtype=np.float32)


def _RT_3x4(cam) -> np.ndarray:
    bpy.context.view_layer.update()
    loc, rot = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rot.to_matrix())
    t = np.asarray(loc)
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    R = R.T
    t = -R @ t
    return np.concatenate([cam_rec @ R, (cam_rec @ t)[:, None]], 1).astype(np.float32)


def _selected_views() -> tuple[list[int], list[float], list[float]]:
    if ARGS.view_azimuths.strip():
        azimuths = [float(x) for x in ARGS.view_azimuths.split(",") if x.strip()]
        if ARGS.view_ids.strip():
            ids = [int(x) for x in ARGS.view_ids.split(",") if x.strip()]
        else:
            ids = list(range(len(azimuths)))
        if len(ids) != len(azimuths):
            raise ValueError("--view_ids length must match --view_azimuths")
        if ARGS.view_elevations.strip():
            elevations = [float(x) for x in ARGS.view_elevations.split(",") if x.strip()]
            if len(elevations) != len(azimuths):
                raise ValueError("--view_elevations length must match --view_azimuths")
        else:
            elevations = [0.0] * len(azimuths)
        return ids, azimuths, elevations
    views = DEFAULT_VIEWS[: ARGS.num_views]
    return list(range(len(views))), [x[0] for x in views], [x[1] for x in views]


def render_all(object_file: str) -> None:
    os.makedirs(ARGS.output_dir, exist_ok=True)
    os.makedirs(os.path.join(ARGS.output_dir, "image_ori"), exist_ok=True)
    os.makedirs(os.path.join(ARGS.output_dir, "camera"), exist_ok=True)
    _reset_scene()
    _load_object(object_file)
    engine = _init_render_engine()
    _init_world_light()
    scale, offset = _normalize_scene(ARGS.shrink)
    _force_flat_shading()
    _add_voxel_bevels()
    _ensure_vertex_color_materials()
    _tune_materials_without_touching_textures()
    ids, azimuths, elevations = _selected_views()
    np.save(os.path.join(ARGS.output_dir, "meta.npy"), np.asarray([scale, offset[0], offset[1], offset[2]], np.float32))
    with open(os.path.join(ARGS.output_dir, "shrink.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "shrink": ARGS.shrink,
                "scale": float(scale),
                "offset": [float(offset[0]), float(offset[1]), float(offset[2])],
                "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                "distance": ARGS.distance,
                "ortho_scale": ARGS.ortho_scale,
                "resolution": ARGS.resolution,
                "engine": engine,
                "views": [[float(a), float(e)] for a, e in zip(azimuths, elevations)],
                "view_ids": ids,
                "voxel_detail": {
                    "flat_shade": True,
                    "bevel": True,
                    "bevel_ratio": VOXEL_BEVEL_RATIO,
                    "bevel_segments": VOXEL_BEVEL_SEGMENTS,
                    "denoising": False,
                    "pixel_filter_width": PIXEL_FILTER_WIDTH,
                    "exposure": EXPOSURE,
                },
            },
            f,
            indent=2,
        )
    cam = _make_ortho_camera()
    lights = _make_light_rig()
    for vid, azimuth, elevation in zip(ids, azimuths, elevations):
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ARGS.ortho_scale
        _place_camera(cam, azimuth, elevation, ARGS.distance)
        bpy.context.view_layer.update()
        _update_light_rig(cam, lights)
        out_png = os.path.join(ARGS.output_dir, "image_ori", f"{vid:03d}.png")
        bpy.context.scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        paras = {
            "intrinsic": _K_ortho(cam.data.ortho_scale),
            "extrinsic": _RT_3x4(cam),
            "azimuth": float(azimuth),
            "elevation": float(elevation),
            "distance": float(ARGS.distance),
            "ortho_scale": float(cam.data.ortho_scale),
            "focal": float(cam.data.lens),
            "sensor_width": float(cam.data.sensor_width),
            "near": float(ARGS.distance - 1),
            "far": float(ARGS.distance + 1),
            "camera": "ortho",
            "resolution": int(ARGS.resolution),
        }
        np.save(os.path.join(ARGS.output_dir, "camera", f"{vid:03d}.npy"), paras)
        print(f"[render] view {vid:03d} az={azimuth:6.1f} el={elevation:6.1f} -> {out_png}")
    print(f"[render] done -> {ARGS.output_dir}")


if __name__ == "__main__":
    try:
        render_all(ARGS.object_path)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
