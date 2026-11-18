"""Headless multi-view renderer for a single GLB, ported from
a reference Blender example (same ortho camera, camera-relative 3-point
light rig, AgX look, exposure and Cycles settings, same camera-parameter export).

Differences from the reference script:
  * CLI driven (no EDIT-ME constants) and imports the GLB itself.
  * Voxel-specific tweaks (flat shading, voxel bevels) are OFF by default because
    the inputs here are smooth textured meshes; enable with --flat_shade/--voxel_bevel.

Usage:
    blender -b --factory-startup -P render_views_blender.py -- \
        --object a.glb --output_dir out/a --views "30,30;120,30" [--view_ids 0,1]

Writes <output_dir>/<vid:03d>.png, <output_dir>/camera/<vid:03d>.npy and
<output_dir>/shrink.json (normalization + camera settings).
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--views", default="", help="'az,el;az,el;...' in degrees (empty: no rendering)")
    ap.add_argument("--view_ids", default="", help="comma-separated ids; default 0..N-1")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--ortho_scale", type=float, default=1.2)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--pixel_filter_width", type=float, default=0.50)
    ap.add_argument("--exposure", type=float, default=-0.6)
    ap.add_argument("--look", default="AgX - Base Contrast")
    ap.add_argument("--env_strength", type=float, default=0.35)
    ap.add_argument("--key_energy", type=float, default=420.0)
    ap.add_argument("--fill_energy", type=float, default=70.0)
    ap.add_argument("--rim_energy", type=float, default=120.0)
    ap.add_argument("--key_size", type=float, default=1.2)
    ap.add_argument("--fill_size", type=float, default=2.5)
    ap.add_argument("--rim_size", type=float, default=1.5)
    ap.add_argument("--flat_shade", action="store_true")
    ap.add_argument("--voxel_bevel", action="store_true")
    ap.add_argument("--voxel_bevel_ratio", type=float, default=0.07)
    ap.add_argument("--min_roughness", type=float, default=0.35)
    ap.add_argument("--no_tune_roughness", action="store_true")
    ap.add_argument("--object_transform", default="",
                    help=".npy 4x4 matrix applied to the imported object (Blender world frame) before normalization")
    ap.add_argument("--export_mesh_npz", default="",
                    help="write world-space vertices/faces (after normalization) to this .npz")
    return ap.parse_args(argv)


ARGS = parse_args()


# ---------- scene helpers ----------
def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)


def _scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(getattr(obj, "data", None), bpy.types.Mesh):
            yield obj


def _scene_bbox():
    bmin = Vector((math.inf,) * 3)
    bmax = Vector((-math.inf,) * 3)
    found = False
    for obj in _scene_meshes():
        found = True
        for c in obj.bound_box:
            w = obj.matrix_world @ Vector(c)
            bmin = Vector(tuple(min(a, b) for a, b in zip(bmin, w)))
            bmax = Vector(tuple(max(a, b) for a, b in zip(bmax, w)))
    if not found:
        raise RuntimeError("No mesh found in current scene.")
    return bmin, bmax


def _normalize_scene():
    """Rescale + recenter imported scene so longest bbox side is exactly 1."""
    roots = [o for o in bpy.context.scene.objects.values() if not o.parent]
    if len(roots) == 0:
        raise RuntimeError("Empty scene.")

    if len(roots) > 1:
        parent = bpy.data.objects.new("_NormalizeRoot", None)
        bpy.context.scene.collection.objects.link(parent)
        for o in roots:
            o.parent = parent
    else:
        parent = roots[0]

    bmin, bmax = _scene_bbox()
    longest = max(bmax - bmin)
    if longest <= 0:
        raise RuntimeError("Degenerate bbox (longest edge = 0).")

    scale = 1.0 / longest
    parent.scale = parent.scale * scale
    bpy.context.view_layer.update()

    bmin, bmax = _scene_bbox()
    offset = -(bmin + bmax) / 2.0
    parent.matrix_world.translation += offset
    bpy.context.view_layer.update()
    return scale, offset


def _apply_object_transform(path):
    """Parent the imported roots to an empty carrying the given world matrix."""
    T = np.load(path)
    roots = [o for o in bpy.context.scene.objects.values() if not o.parent]
    parent = bpy.data.objects.new("_AlignRoot", None)
    bpy.context.scene.collection.objects.link(parent)
    for o in roots:
        o.parent = parent
    parent.matrix_world = Matrix(T.tolist())
    bpy.context.view_layer.update()


def _export_mesh_npz(path):
    """World-space triangle mesh of everything imported (modifiers applied)."""
    deps = bpy.context.evaluated_depsgraph_get()
    verts, faces, offset = [], [], 0
    for obj in _scene_meshes():
        eval_obj = obj.evaluated_get(deps)
        mesh = eval_obj.to_mesh()
        mesh.calc_loop_triangles()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        M = np.array(obj.matrix_world)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        tri = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tri)
        verts.append(co)
        faces.append(tri.reshape(-1, 3) + offset)
        offset += len(co)
        eval_obj.to_mesh_clear()
    np.savez(path, vertices=np.concatenate(verts), faces=np.concatenate(faces))
    print(f"[render] exported mesh ({offset} verts) -> {path}", flush=True)


# ---------- optional voxel appearance ----------
def _force_flat_shading():
    for obj in _scene_meshes():
        for poly in obj.data.polygons:
            poly.use_smooth = False
    bpy.context.view_layer.update()


def _estimate_voxel_edge_local(obj):
    mesh = obj.data
    if len(mesh.edges) == 0:
        return None
    axis_lengths, all_lengths = [], []
    for i, e in enumerate(mesh.edges):
        if i >= 250000:
            break
        d = mesh.vertices[e.vertices[1]].co - mesh.vertices[e.vertices[0]].co
        L = d.length
        if L <= 1e-10:
            continue
        all_lengths.append(L)
        ad = [abs(d.x), abs(d.y), abs(d.z)]
        major = max(ad)
        minor = sorted(ad)[:2]
        if major > 0 and minor[1] <= major * 1e-4:
            axis_lengths.append(L)
    lengths = np.asarray(axis_lengths if len(axis_lengths) >= 8 else all_lengths, dtype=np.float64)
    lengths = lengths[np.isfinite(lengths) & (lengths > 1e-10)]
    if len(lengths) == 0:
        return None
    lengths.sort()
    return float(np.median(lengths[:max(1, int(len(lengths) * 0.25))]))


def _add_voxel_bevels():
    for obj in _scene_meshes():
        old = obj.modifiers.get("_VoxelDetailBevel")
        if old is not None:
            obj.modifiers.remove(old)
        voxel_edge = _estimate_voxel_edge_local(obj)
        if voxel_edge is None:
            continue
        mod = obj.modifiers.new("_VoxelDetailBevel", "BEVEL")
        mod.width = voxel_edge * ARGS.voxel_bevel_ratio
        mod.segments = 1
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(5.0)
        if hasattr(mod, "affect"):
            mod.affect = "EDGES"
        if hasattr(mod, "use_clamp_overlap"):
            mod.use_clamp_overlap = True


def _tune_materials_without_touching_textures():
    """Keep Base Color / image texture graph intact; only soften raw roughness."""
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
                    rough.default_value = max(float(rough.default_value), ARGS.min_roughness)


# ---------- world / renderer ----------
def _init_world_light():
    for o in [x for x in bpy.context.scene.objects if x.type == "LIGHT"]:
        bpy.data.objects.remove(o, do_unlink=True)

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True

    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = ARGS.env_strength


def _init_render_engine():
    scene = bpy.context.scene
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = ARGS.look
    except TypeError as e:
        print(f"[render] look {ARGS.look!r} unavailable ({e}); keeping default.")
    r = scene.render

    avail = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    engine = "CYCLES" if "CYCLES" in avail else next(iter(avail))
    r.engine = engine

    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.resolution_x = ARGS.resolution
    r.resolution_y = ARGS.resolution
    r.resolution_percentage = 100
    r.film_transparent = True

    scene.view_settings.exposure = ARGS.exposure
    scene.view_settings.gamma = 1.0

    if engine == "CYCLES":
        c = scene.cycles
        c.samples = ARGS.samples
        c.use_denoising = ARGS.denoise
        c.filter_type = "BOX"
        c.filter_width = ARGS.pixel_filter_width
        c.diffuse_bounces = 2
        c.glossy_bounces = 2
        c.transparent_max_bounces = 3
        c.transmission_bounces = 3
        c.adaptive_threshold = 0.0

        if ARGS.gpu:
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.compute_device_type = "CUDA"
                prefs.get_devices()
                c.device = "GPU"
            except Exception as e:
                print(f"[render] GPU setup failed ({e}); using CPU.")
                c.device = "CPU"
        else:
            c.device = "CPU"

    return engine


# ---------- camera ----------
def _make_ortho_camera():
    for o in [x for x in bpy.context.scene.objects if x.type == "CAMERA"]:
        bpy.data.objects.remove(o, do_unlink=True)

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

    cc = cam.constraints.new(type="TRACK_TO")
    cc.track_axis = "TRACK_NEGATIVE_Z"
    cc.up_axis = "UP_Y"
    cc.target = empty
    return cam


def _place_camera(cam, azimuth_deg, elevation_deg, distance):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cam.location = (
        distance * math.cos(az) * math.cos(el),
        distance * math.sin(az) * math.cos(el),
        distance * math.sin(el),
    )


# ---------- camera-relative 3-point light rig ----------
def _new_area_light(name, energy, size):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _look_at(obj, target=Vector((0.0, 0.0, 0.0))):
    direction = target - obj.location
    if direction.length > 1e-8:
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _make_light_rig():
    return {
        "key": _new_area_light("_Key", ARGS.key_energy, ARGS.key_size),
        "fill": _new_area_light("_Fill", ARGS.fill_energy, ARGS.fill_size),
        "rim": _new_area_light("_Rim", ARGS.rim_energy, ARGS.rim_size),
    }


def _update_light_rig(cam, rig):
    """Move lights with the camera so every view gets equally readable relief."""
    bpy.context.view_layer.update()  # evaluate TRACK_TO first

    R = cam.matrix_world.to_3x3()
    right = (R @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (R @ Vector((0.0, 1.0, 0.0))).normalized()
    forward = (R @ Vector((0.0, 0.0, -1.0))).normalized()
    cam_pos = cam.matrix_world.translation.copy()

    rig["key"].location = cam_pos - 0.85 * right + 0.95 * up + 0.55 * forward
    rig["fill"].location = cam_pos + 1.05 * right + 0.15 * up + 0.45 * forward
    rig["rim"].location = 1.35 * forward + 0.8 * up - 0.25 * right

    for light in rig.values():
        _look_at(light)

    bpy.context.view_layer.update()


# ---------- camera parameter export (reference convention) ----------
def _K_ortho(ortho_scale):
    scene = bpy.context.scene
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    par = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    fx = rx / ortho_scale
    fy = ry / ortho_scale / par
    return np.asarray([[fx, 0, rx / 2], [0, fy, ry / 2], [0, 0, 1]], dtype=np.float32)


def _RT_3x4(cam):
    bpy.context.view_layer.update()
    loc, rot = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rot.to_matrix())
    t = np.asarray(loc)
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    R = R.T
    t = -R @ t
    return np.concatenate([cam_rec @ R, (cam_rec @ t)[:, None]], 1).astype(np.float32)


def _parse_views(text):
    views = []
    for item in text.split(";"):
        item = item.strip()
        if item:
            az, el = (float(v) for v in item.split(","))
            views.append((az, el))
    return views


def main():
    views = _parse_views(ARGS.views)
    view_ids = [v.strip() for v in ARGS.view_ids.split(",") if v.strip()] or [f"{i:03d}" for i in range(len(views))]
    if len(view_ids) != len(views):
        raise ValueError(f"--view_ids ({len(view_ids)}) must match --views ({len(views)})")

    out_dir = ARGS.output_dir
    os.makedirs(os.path.join(out_dir, "camera"), exist_ok=True)

    _clear_scene()
    _import_glb(ARGS.object)
    if ARGS.object_transform:
        _apply_object_transform(ARGS.object_transform)

    engine = _init_render_engine()
    _init_world_light()

    # Normalize BEFORE adding camera/lights so only the imported asset is transformed.
    if ARGS.no_normalize:
        scale, offset = 1.0, Vector((0.0, 0.0, 0.0))
    else:
        scale, offset = _normalize_scene()

    if ARGS.flat_shade:
        _force_flat_shading()
    if ARGS.voxel_bevel:
        _add_voxel_bevels()
    if not ARGS.no_tune_roughness:
        _tune_materials_without_touching_textures()

    if ARGS.export_mesh_npz:
        _export_mesh_npz(ARGS.export_mesh_npz)
    if not views:
        return

    with open(os.path.join(out_dir, "shrink.json"), "w") as f:
        json.dump(
            {
                "object": os.path.abspath(ARGS.object),
                "scale": float(scale),
                "offset": [offset[0], offset[1], offset[2]],
                "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                "distance": ARGS.distance,
                "ortho_scale": ARGS.ortho_scale,
                "resolution": ARGS.resolution,
                "engine": engine,
                "views": [{"id": vid, "azimuth": az, "elevation": el} for vid, (az, el) in zip(view_ids, views)],
            },
            f,
            indent=2,
        )

    cam = _make_ortho_camera()
    lights = _make_light_rig()

    for vid, (az, el) in zip(view_ids, views):
        _place_camera(cam, az, el, ARGS.distance)
        bpy.context.view_layer.update()
        _update_light_rig(cam, lights)

        out_png = os.path.join(out_dir, f"{vid}.png")
        bpy.context.scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)

        paras = {
            "intrinsic": _K_ortho(cam.data.ortho_scale),
            "extrinsic": _RT_3x4(cam),
            "azimuth": float(az),
            "elevation": float(el),
            "distance": float(ARGS.distance),
            "ortho_scale": float(cam.data.ortho_scale),
            "focal": float(cam.data.lens),
            "sensor_width": float(cam.data.sensor_width),
            "near": float(ARGS.distance - 1),
            "far": float(ARGS.distance + 1),
            "camera": "ortho",
            "resolution": int(ARGS.resolution),
        }
        np.save(os.path.join(out_dir, "camera", f"{vid}.npy"), paras)
        print(f"[render] view {vid}  az={az:6.1f}  el={el:6.1f} -> {out_png}", flush=True)

    print(f"[render] done -> {out_dir}", flush=True)


main()
