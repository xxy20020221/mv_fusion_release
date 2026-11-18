"""Multi-view GLB renderer (orthographic, used for round-2 source renders and metrics.py).

参考 Easy3E/blender/render.py 的 _ortho_render_loop 精简而成:
    - trellis 生成的 glb 已在 [-0.5, 0.5]^3 (64^3 voxel 空间), 所以
      我们直接用 ORTHO 相机 + ortho_scale=1 + distance=2 + elevation=0
      环绕一圈拍 num_views 张;
    - 引擎默认 CYCLES + CPU: headless 服务器 + sm_90 GPU 上唯一可稳定
      跑通的组合 (EEVEE 需要 X display, Cycles GPU 在 Blender 3.x
      编不出 sm_90 内核; 我们的 config 已切到 Blender 4.0.0);
    - 输出:
        <output_dir>/image_ori/000.png ...
        <output_dir>/camera/000.npy   ...   (intrinsic/extrinsic/...)
        <output_dir>/meta.npy               [scale, off_x, off_y, off_z]
        <output_dir>/shrink.json

Usage (called by step1_render.py, do NOT import python-side):

    blender -b -P render_glb_mv.py -- \
        --object_path /path/to/x.glb \
        --output_dir /path/to/step1/ObjX \
        --num_views 12 \
        --resolution 512
"""
import argparse
import json
import math
import os
import sys
from typing import Generator, Optional, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector


# ---------- CLI ----------
def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_views", type=int, default=12)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--shrink", type=float, default=1.0,
                    help="normalize scene so longest edge = shrink. "
                         "trellis glb 已经归一化, 传 <=0 则跳过 normalize.")
    ap.add_argument("--distance", type=float, default=2.0,
                    help="camera distance from origin (ortho)")
    ap.add_argument("--ortho_scale", type=float, default=1.0,
                    help="ortho subject width (view size)")
    ap.add_argument("--elevation", type=float, default=0.0,
                    help="elevation in degrees")
    ap.add_argument("--engine", type=str, default="CYCLES",
                    choices=["CYCLES", "BLENDER_EEVEE",
                             "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--cycles_device", type=str, default="CPU",
                    choices=["CPU", "GPU"])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--view_azimuths", type=str, default="",
                    help="Optional comma-separated azimuth list (deg). If "
                         "given, --num_views is ignored and files are named "
                         "with the corresponding view id / azimuth.")
    ap.add_argument("--view_elevations", type=str, default="",
                    help="Optional comma-separated elevation list (deg), "
                         "one per --view_azimuths entry. If empty, the "
                         "global --elevation is used for every view.")
    ap.add_argument("--view_ids", type=str, default="",
                    help="Optional comma-separated view ids (int) matching "
                         "--view_azimuths one-to-one. Defaults to "
                         "range(len(view_azimuths)).")
    argv = sys.argv[sys.argv.index("--") + 1:]
    return ap.parse_args(argv)


ARGS = _parse()


# ---------- render setup ----------
def _init_render_engine():
    scene = bpy.context.scene
    r = scene.render

    # engine fallback (name might not exist in this blender build)
    avail = {e.identifier for e in
             bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    engine = ARGS.engine
    if engine not in avail:
        print(f"[render] engine '{engine}' unavailable ({sorted(avail)}); "
              f"falling back.")
        for cand in ("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT",
                     "BLENDER_WORKBENCH"):
            if cand in avail:
                engine = cand
                break
    r.engine = engine

    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.resolution_x = ARGS.resolution
    r.resolution_y = ARGS.resolution
    r.resolution_percentage = 100
    r.film_transparent = True
    r.threads_mode = "FIXED"
    r.threads = ARGS.threads

    if engine == "CYCLES":
        c = scene.cycles
        c.samples = ARGS.samples
        c.use_denoising = True
        c.filter_type = "BOX"
        c.filter_width = 1
        c.diffuse_bounces = 1
        c.glossy_bounces = 1
        c.transparent_max_bounces = 3
        c.transmission_bounces = 3
        c.adaptive_threshold = 0
        if ARGS.cycles_device == "GPU":
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.compute_device_type = "CUDA"
                prefs.get_devices()
                c.device = "GPU"
            except Exception as e:
                print(f"[render] GPU setup failed ({e}); fallback CPU.")
                c.device = "CPU"
        else:
            c.device = "CPU"
    return engine


ENGINE = _init_render_engine()


# ---------- scene helpers ----------
def reset_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for tex in list(bpy.data.textures):
        bpy.data.textures.remove(tex, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)


def load_object(p: str):
    ext = p.rsplit(".", 1)[-1].lower()
    if ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=p, merge_vertices=True,
                                  import_shading="NORMALS")
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=p)
    elif ext == "obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=p)
        else:
            bpy.ops.import_scene.obj(filepath=p)
    elif ext == "ply":
        bpy.ops.import_mesh.ply(filepath=p)
    elif ext == "stl":
        bpy.ops.import_mesh.stl(filepath=p)
    else:
        raise ValueError(f"unsupported ext: {p}")


def scene_meshes() -> Generator:
    for obj in bpy.context.scene.objects.values():
        if isinstance(getattr(obj, "data", None), bpy.types.Mesh):
            yield obj


def scene_bbox(single: Optional[bpy.types.Object] = None,
               ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    it = list(scene_meshes()) if single is None else [single]
    for obj in it:
        found = True
        for c in obj.bound_box:
            c = Vector(c)
            if not ignore_matrix:
                c = obj.matrix_world @ c
            bbox_min = tuple(min(a, b) for a, b in zip(bbox_min, c))
            bbox_max = tuple(max(a, b) for a, b in zip(bbox_max, c))
    if not found:
        raise RuntimeError("no mesh in scene")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene(shrink: float) -> Tuple[float, Vector]:
    """将场景 normalize 到最长边 = shrink, 中心在原点. shrink<=0 表示跳过。"""
    if shrink <= 0:
        return 1.0, Vector((0.0, 0.0, 0.0))

    roots = [o for o in bpy.context.scene.objects.values() if not o.parent]
    if len(roots) > 1:
        parent = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent)
        for o in roots:
            o.parent = parent
    else:
        parent = roots[0]

    bmin, bmax = scene_bbox()
    longest = max(bmax - bmin)
    if longest <= 0:
        raise RuntimeError("degenerate bbox")
    scale = (1.0 / longest) * float(shrink)
    parent.scale = parent.scale * scale
    bpy.context.view_layer.update()

    bmin, bmax = scene_bbox()
    offset = -(bmin + bmax) / 2.0
    parent.matrix_world.translation += offset
    bpy.context.view_layer.update()
    return scale, offset


def strip_bsdf_normal_links():
    """去掉 principled BSDF 的 normal 输入 (纹理不再扰动法线)."""
    for obj in scene_meshes():
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED" and \
                        node.inputs["Normal"].is_linked:
                    mat.node_tree.links.remove(
                        node.inputs["Normal"].links[0])
                if node.type == "TRANSPARENT_BSDF":
                    node.inputs[0].default_value = (1, 1, 1, 1)


def init_environment_light(env: float = 1.0, strength: float = 1.3):
    """删除所有灯, 用世界背景做柔和环境光."""
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
    bg.inputs["Color"].default_value = (env, env, env, 1.0)
    bg.inputs["Strength"].default_value = strength


# ---------- camera math (与 Easy3E 保持一致) ----------
def set_camera_mvdream(cam, azimuth_deg: float, elevation_deg: float,
                       distance: float):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cam.location = (distance * math.cos(az) * math.cos(el),
                    distance * math.sin(az) * math.cos(el),
                    distance * math.sin(el))
    d = -cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def get_K_ortho(camd, ortho_scale: float) -> Matrix:
    scene = bpy.context.scene
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    par = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    fx = rx / ortho_scale
    fy = ry / ortho_scale / par
    return Matrix(((fx, 0, rx / 2), (0, fy, ry / 2), (0, 0, 1)))


def get_RT_3x4(cam) -> np.ndarray:
    bpy.context.view_layer.update()
    loc, rot = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rot.to_matrix())
    t = np.asarray(loc)
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    return np.concatenate([cam_rec @ R, (cam_rec @ t)[:, None]], 1)


# ---------- main ----------
def render_all(object_file: str):
    os.makedirs(ARGS.output_dir, exist_ok=True)
    os.makedirs(os.path.join(ARGS.output_dir, "image_ori"), exist_ok=True)
    os.makedirs(os.path.join(ARGS.output_dir, "camera"), exist_ok=True)

    # 1) 干净场景
    reset_scene()

    # 2) 加载物体 + 环境光
    load_object(object_file)
    init_environment_light()
    try:
        strip_bsdf_normal_links()
    except Exception as e:
        print(f"[render] strip_bsdf_normal_links failed: {e}")

    # 3) normalize (trellis glb 已 normalize, shrink=1 时对 [-0.5,0.5]^3 是 no-op)
    scale, offset = normalize_scene(ARGS.shrink)
    np.save(os.path.join(ARGS.output_dir, "meta.npy"),
            np.asarray([scale, offset[0], offset[1], offset[2]], np.float32))
    with open(os.path.join(ARGS.output_dir, "shrink.json"), "w") as f:
        json.dump({"shrink": ARGS.shrink, "scale": scale,
                   "offset": [offset[0], offset[1], offset[2]],
                   "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                   "distance": ARGS.distance,
                   "ortho_scale": ARGS.ortho_scale,
                   "elevation": ARGS.elevation,
                   "num_views": ARGS.num_views,
                   "resolution": ARGS.resolution,
                   "engine": ENGINE}, f, indent=2)

    # 4) 创建 ortho 相机 (在 normalize 之后, 避免旧 Camera 状态残留)
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ARGS.ortho_scale
    cam.data.lens = 35
    cam.data.sensor_height = 32
    cam.data.sensor_width = 32
    cc = cam.constraints.new(type="TRACK_TO")
    cc.track_axis = "TRACK_NEGATIVE_Z"
    cc.up_axis = "UP_Y"

    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    cc.target = empty

    # 5) 环拍
    scene = bpy.context.scene
    # 决定要渲染哪些视角:
    if ARGS.view_azimuths.strip():
        azimuths = [float(a) for a in ARGS.view_azimuths.split(",") if a.strip()]
        if ARGS.view_ids.strip():
            ids = [int(v) for v in ARGS.view_ids.split(",") if v.strip()]
            if len(ids) != len(azimuths):
                raise ValueError("--view_ids length must match --view_azimuths")
        else:
            ids = list(range(len(azimuths)))
        if ARGS.view_elevations.strip():
            elevations = [float(e) for e in ARGS.view_elevations.split(",")
                          if e.strip()]
            if len(elevations) != len(azimuths):
                raise ValueError(
                    "--view_elevations length must match --view_azimuths")
        else:
            elevations = [ARGS.elevation] * len(azimuths)
    else:
        azimuths = [i * 360.0 / ARGS.num_views for i in range(ARGS.num_views)]
        ids = list(range(ARGS.num_views))
        elevations = [ARGS.elevation] * len(azimuths)

    for vid, azimuth, elev in zip(ids, azimuths, elevations):
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ARGS.ortho_scale
        bpy.context.view_layer.update()
        set_camera_mvdream(cam, azimuth, elev, ARGS.distance)

        render_path = os.path.join(ARGS.output_dir, "image_ori",
                                   f"{vid:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        K = get_K_ortho(cam.data, ortho_scale=cam.data.ortho_scale)
        RT = get_RT_3x4(cam)
        paras = {
            "intrinsic": np.asarray(K, np.float32),
            "extrinsic": np.asarray(RT, np.float32),
            "fov": float(cam.data.angle),
            "azimuth": float(azimuth),
            "elevation": float(elev),
            "distance": float(ARGS.distance),
            "ortho_scale": float(cam.data.ortho_scale),
            "focal": float(cam.data.lens),
            "sensor_width": float(cam.data.sensor_width),
            "near": float(ARGS.distance - 1),
            "far":  float(ARGS.distance + 1),
            "camera": "ortho",
            "resolution": int(ARGS.resolution),
        }
        np.save(os.path.join(ARGS.output_dir, "camera", f"{vid:03d}.npy"),
                paras)
        print(f"[render] view {vid:03d} az={azimuth:6.1f} el={elev:6.1f} "
              f"-> {render_path}")


if __name__ == "__main__":
    try:
        render_all(ARGS.object_path)
        print("[render] done ->", ARGS.output_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
