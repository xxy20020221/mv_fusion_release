import os
import traceback

import numpy as np
import open3d as o3d
import torch
from PIL import Image

import trellis.modules.sparse as sp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from trellis.utils.general_utils import get_view_azimuth_from_ori

from .utils import (
    _build_masks_from_mask_glb,
    _build_masks_from_voxels,
    _normalize_mask,
    encode_slat_latent_from_feature_tensor,
    get_voxels,
    load_encoder,
    voxelize,
)


def load_edit_mask_from_rgba(
    image_path: str,
    target_size=None,
    return_type="torch",
    threshold=0,
    soft=False,
    invert=False,
    save_path=None,
    device=None
):
    img = Image.open(image_path).convert("RGBA")

    if target_size is not None:
        resample = Image.BILINEAR if soft else Image.NEAREST
        img = img.resize((target_size[1], target_size[0]), resample=resample)

    alpha = np.array(img, dtype=np.uint8)[..., 3].astype(np.float32) / 255.0

    if soft:
        mask = alpha
    else:
        thr = float(threshold) / 255.0
        mask = (alpha > thr).astype(np.float32)

    if invert:
        mask = 1.0 - mask

    if save_path is not None:
        m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(m8, mode="L").save(save_path)

    if return_type == "numpy":
        return mask
    elif return_type == "pil":
        m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(m8, mode="L")
    else:
        t = torch.from_numpy(mask)[None, None, ...]
        t = t.to(dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t


def _get_azimuth_from_ori(args) -> float:
    try:
        if args.original_image:
            view_idx = int(os.path.splitext(os.path.basename(args.original_image))[0])
            return view_idx * 360.0 / 16.0
        ori_dir = os.path.join(args.root_dir, "ori")
        _, azimuth_deg = get_view_azimuth_from_ori(ori_dir)
        return azimuth_deg
    except Exception as e:
        print(f"[WARN] Could not extract azimuth from original image: {e}. Using default 270°.")
        return 270.0


def _split_csv_paths(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_steps(value, arg_name):
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"--{arg_name} must be a comma-separated list of integers") from exc


def _parse_float_bbox(value, arg_name):
    try:
        bbox = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"--{arg_name} must be x0,y0,x1,y1") from exc
    if len(bbox) != 4:
        raise ValueError(f"--{arg_name} must contain exactly four values: x0,y0,x1,y1")
    x0, y0, x1, y1 = bbox
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"--{arg_name} must be normalized and satisfy 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1")
    return bbox


def _load_mesh_bounds(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"[mv_fusion] Invalid confidence box mesh: {path}")
    bounds = np.asarray(mesh.get_axis_aligned_bounding_box().get_box_points(), dtype=np.float32)
    return np.stack([bounds.min(axis=0), bounds.max(axis=0)], axis=0)


def _normalize_mv_fusion_box(args, conf_box_glb):
    if not conf_box_glb:
        return None, None
    render_dir = getattr(args, "render_dir", None) or os.path.join(args.root_dir, "render")
    transform_path = os.path.join(render_dir, "transforms.json")
    if not os.path.exists(transform_path):
        print(f"[mv_fusion][WARN] Missing stage1 transform; using raw box bounds: {transform_path}")
        return conf_box_glb, None
    cache_dir = os.path.join(args.root_dir, ".mv_fusion_box_cache", os.path.splitext(os.path.basename(conf_box_glb))[0])
    os.makedirs(cache_dir, exist_ok=True)
    _normalize_mask(args, conf_box_glb, cache_dir)
    normalized_ply = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(conf_box_glb))[0]}.ply")
    if not os.path.exists(normalized_ply):
        raise FileNotFoundError(f"[mv_fusion] Missing normalized confidence box PLY: {normalized_ply}")
    return normalized_ply, _load_mesh_bounds(normalized_ply)


def _build_mv_velocity_debug(args):
    if not getattr(args, "debug_mv_velocity_diff", False):
        return None

    source_paths = _split_csv_paths(getattr(args, "debug_mv_source_images", ""))
    target_paths = _split_csv_paths(getattr(args, "debug_mv_target_images", ""))
    view_names = _split_csv_paths(getattr(args, "debug_mv_view_names", ""))
    if len(source_paths) != 2 or len(target_paths) != 2:
        raise ValueError("--debug_mv_velocity_diff requires exactly two source images and two target images")
    if view_names and len(view_names) != 2:
        raise ValueError("--debug_mv_view_names must be empty or contain exactly two names")
    if not view_names:
        view_names = [os.path.splitext(os.path.basename(path))[0] for path in source_paths]

    for label, paths in (("source", source_paths), ("target", target_paths)):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[debug_mv] Missing {label} image: {path}")

    steps = _parse_int_steps(getattr(args, "debug_mv_steps", "1,3,5,10,20"), "debug_mv_steps")

    output_dir = getattr(args, "debug_mv_output_dir", None)
    if not output_dir:
        feature_dir = getattr(args, "feature_dir", None) or os.path.join(args.root_dir, "feature")
        output_dir = os.path.join(feature_dir, "mv_velocity_diff_heatmaps")

    print(f"[debug_mv] Enabled velocity-difference heatmap export for views: {view_names}")
    return {
        "source_images": [Image.open(path) for path in source_paths],
        "target_images": [Image.open(path) for path in target_paths],
        "view_names": view_names,
        "steps": steps,
        "output_dir": output_dir,
    }


def _build_mv_fusion(args):
    if not getattr(args, "mv_fusion", False):
        return None

    source_paths = _split_csv_paths(getattr(args, "mv_fusion_source_images", ""))
    target_paths = _split_csv_paths(getattr(args, "mv_fusion_target_images", ""))
    view_names = _split_csv_paths(getattr(args, "mv_fusion_view_names", ""))
    if len(source_paths) < 2 or len(target_paths) < 2:
        raise ValueError("--mv_fusion requires at least two source images and two target images")
    if len(source_paths) != len(target_paths):
        raise ValueError(
            f"--mv_fusion_source_images ({len(source_paths)}) and --mv_fusion_target_images "
            f"({len(target_paths)}) must have the same number of entries"
        )
    if view_names and len(view_names) != len(source_paths):
        raise ValueError(
            f"--mv_fusion_view_names must be empty or match --mv_fusion_source_images "
            f"({len(source_paths)})"
        )
    if not view_names:
        view_names = [os.path.splitext(os.path.basename(path))[0] for path in source_paths]

    n_views = len(view_names)

    # -------- resolve trust / base views --------
    trust_views_csv = _split_csv_paths(getattr(args, "mv_fusion_trust_views", ""))
    if trust_views_csv:
        trust_views = list(trust_views_csv)
    else:
        trust_views = [getattr(args, "mv_fusion_trust_view", view_names[0])]
    base_view = getattr(args, "mv_fusion_base_view", view_names[-1])

    for tv in trust_views:
        if tv not in view_names:
            raise ValueError(f"mv_fusion trust view '{tv}' must be one of {view_names}")
    if base_view not in view_names:
        raise ValueError(f"--mv_fusion_base_view '{base_view}' must be one of {view_names}")
    if base_view in trust_views:
        raise ValueError(
            f"--mv_fusion_base_view '{base_view}' must be different from every trust view {trust_views}"
        )
    if len(set(trust_views)) != len(trust_views):
        raise ValueError(f"--mv_fusion_trust_views must be unique, got {trust_views}")

    fusion_mode = str(getattr(args, "mv_fusion_fusion_mode", "trust_region")).strip().lower()
    if fusion_mode not in {"trust_region", "view_average"}:
        raise ValueError(
            f"--mv_fusion_fusion_mode must be 'trust_region' or 'view_average', got {fusion_mode!r}"
        )

    for label, paths in (("source", source_paths), ("target", target_paths)):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[mv_fusion] Missing {label} image: {path}")

    steps = _parse_int_steps(getattr(args, "mv_fusion_steps", "1,3,5,10,20"), "mv_fusion_steps")

    # -------- optional geometry-derived per-view support (replaces confidence boxes) --------
    # compute_view_support.py writes chi_v on the 64^3 grid for every view; the sampler
    # uses chi of each trust view as its raw ownership and applies the same
    # "later trust view wins" rule (Route-Diagnose-Refine Eq. 7).
    support_chi64 = None
    support_npz = getattr(args, "mv_fusion_support_npz", None)
    if support_npz:
        if not os.path.exists(support_npz):
            raise FileNotFoundError(f"[mv_fusion] Missing view support npz: {support_npz}")
        data = np.load(support_npz)
        support_views = [str(v) for v in data["views"]]
        chi_key = "chi_extended" if "chi_extended" in data.files else "chi"
        missing = [v for v in trust_views + [base_view] if v not in support_views]
        if missing:
            raise ValueError(f"[mv_fusion] views {missing} missing from support npz views {support_views}")
        chi_all = data[chi_key].astype(np.float32)
        support_chi64 = {v: chi_all[support_views.index(v)] for v in support_views}
        print(f"[mv_fusion] Using geometry-derived view support '{chi_key}' from {support_npz} "
              f"(support views={support_views}); confidence boxes are ignored.")

    # -------- resolve per-trust-view confidence region GLBs --------
    # Priority:
    #   0) --mv_fusion_support_npz   (geometry-derived soft support; no boxes needed)
    #   1) --mv_fusion_conf_box_glbs (CSV, one per trust view; explicit, wins if given)
    #   2) --mv_fusion_conf_box_glb  (single glb; only valid when there is exactly one trust view)
    #   3) --mv_fusion_conf_bbox     (image-space fallback; only valid when there is exactly one trust view)
    conf_box_glbs_csv = _split_csv_paths(getattr(args, "mv_fusion_conf_box_glbs", ""))
    if support_chi64 is not None:
        conf_box_glbs_list = [None] * len(trust_views)
        conf_bbox = None
    elif conf_box_glbs_csv:
        if len(conf_box_glbs_csv) != len(trust_views):
            raise ValueError(
                f"--mv_fusion_conf_box_glbs ({len(conf_box_glbs_csv)}) must match "
                f"--mv_fusion_trust_views ({len(trust_views)})"
            )
        for gpath in conf_box_glbs_csv:
            if not os.path.exists(gpath):
                raise FileNotFoundError(f"[mv_fusion] Missing confidence box GLB: {gpath}")
        conf_box_glbs_list = list(conf_box_glbs_csv)
        conf_bbox = None
    else:
        # legacy single-trust-view path
        if len(trust_views) != 1:
            raise ValueError(
                "--mv_fusion_conf_box_glbs must be provided when --mv_fusion_trust_views "
                "has more than one entry"
            )
        single_glb = getattr(args, "mv_fusion_conf_box_glb", None)
        if single_glb:
            if not os.path.exists(single_glb):
                raise FileNotFoundError(f"[mv_fusion] Missing confidence box GLB: {single_glb}")
            conf_box_glbs_list = [single_glb]
            conf_bbox = None
        else:
            conf_box_glbs_list = [None]
            conf_bbox = _parse_float_bbox(
                getattr(args, "mv_fusion_conf_bbox", "0.36,0.30,0.64,0.58"),
                "mv_fusion_conf_bbox",
            )

    # -------- normalize per-trust-view confidence boxes to the shared render frame --------
    normalized_conf_boxes = []       # list[(normalized_ply_or_None, bounds_np_or_None)] per trust view
    for glb_path in conf_box_glbs_list:
        n_path, n_bounds = _normalize_mv_fusion_box(args, glb_path)
        normalized_conf_boxes.append((n_path, n_bounds))

    output_dir = getattr(args, "mv_fusion_output_dir", None)
    if not output_dir:
        feature_dir = getattr(args, "feature_dir", None) or os.path.join(args.root_dir, "feature")
        output_dir = os.path.join(feature_dir, "mv_fusion_voxel_diff")

    soft_blend = bool(getattr(args, "mv_fusion_soft_blend", False))
    soft_edge_width = float(getattr(args, "mv_fusion_soft_edge_width", 0.035))
    soft_core_width = float(getattr(args, "mv_fusion_soft_core_width", 0.04))
    soft_outside_width = float(getattr(args, "mv_fusion_soft_outside_width", 0.04))
    soft_inside_plateau = float(getattr(args, "mv_fusion_soft_inside_plateau", 0.9))

    print(
        f"[mv_fusion] Enabled velocity fusion: trust_views={trust_views}, base={base_view}, "
        f"n_views={n_views}, fusion_mode={fusion_mode}, conf_box_glbs={conf_box_glbs_list}, conf_bbox={conf_bbox}, "
        f"soft_blend={soft_blend}"
    )
    if soft_blend:
        print(
            f"[mv_fusion] Soft blend params: edge_width={soft_edge_width}, core_width={soft_core_width}, "
            f"outside_width={soft_outside_width}, inside_plateau={soft_inside_plateau}"
        )
    for tv, (n_path, n_bounds) in zip(trust_views, normalized_conf_boxes):
        if n_bounds is not None:
            print(f"[mv_fusion] Normalized confidence box for trust='{tv}': {n_path}, "
                  f"bounds={n_bounds.tolist()}")

    return {
        "source_images": [Image.open(path) for path in source_paths],
        "target_images": [Image.open(path) for path in target_paths],
        "view_names": view_names,
        "trust_view": trust_views[0],       # legacy field: primary trust view (last one in overlap)
        "trust_views": trust_views,          # NEW: full ordered list; later ones win on overlap
        "base_view": base_view,
        "fusion_mode": fusion_mode,
        "conf_bbox": conf_bbox,
        "conf_box_glb": conf_box_glbs_list[0] if conf_box_glbs_list else None,
        "conf_box_glbs": conf_box_glbs_list,  # NEW: one entry per trust view (may contain None)
        "conf_box_normalized_path": normalized_conf_boxes[0][0] if normalized_conf_boxes else None,
        "conf_box_bounds": (
            normalized_conf_boxes[0][1].tolist()
            if normalized_conf_boxes and normalized_conf_boxes[0][1] is not None
            else None
        ),
        "conf_box_bounds_list": [
            (b.tolist() if b is not None else None) for _, b in normalized_conf_boxes
        ],  # NEW: per-trust-view bounds in the same order as trust_views
        "support_chi64": support_chi64,  # {view: (64,64,64) chi} or None -> use confidence boxes
        "pair_crop": bool(getattr(args, "mv_fusion_pair_crop", False)),
        "alpha_inside": float(getattr(args, "mv_fusion_alpha_inside", 1.0)),
        "alpha_outside": float(getattr(args, "mv_fusion_alpha_outside", 0.0)),
        "soft_blend": soft_blend,
        "soft_edge_width": soft_edge_width,
        "soft_core_width": soft_core_width,
        "soft_outside_width": soft_outside_width,
        "soft_inside_plateau": soft_inside_plateau,
        "trust_region_step": int(getattr(args, "mv_fusion_trust_region_step", 10)),
        "steps": steps,
        "output_dir": output_dir,
        "velocity_conflict_threshold": getattr(args, "mv_fusion_velocity_conflict_threshold", "auto"),
        "velocity_conflict_min_ratio": float(getattr(args, "mv_fusion_velocity_conflict_min_ratio", 0.0)),
        "export_edit_magnitude_conflict": bool(getattr(args, "mv_fusion_export_edit_magnitude_conflict", False)),
        "edit_magnitude_conflict_threshold": getattr(args, "mv_fusion_edit_magnitude_conflict_threshold", "auto"),
        "edit_magnitude_conflict_min_ratio": float(getattr(args, "mv_fusion_edit_magnitude_conflict_min_ratio", 0.0)),
        "quiescent_enabled": bool(getattr(args, "mv_fusion_quiescent_enabled", True)),
        "state_mode": getattr(args, "mv_fusion_state_mode", "three_state"),
        "quiescent_ratio": float(getattr(args, "mv_fusion_quiescent_ratio", 0.0)),
        "quiescent_disagreement_threshold": getattr(args, "mv_fusion_quiescent_disagreement_threshold", "auto"),
        "quiescent_edit_mag_threshold": getattr(args, "mv_fusion_quiescent_edit_mag_threshold", "auto"),
    }


def process_single_sample(args, pipeline=None, encoder=None, image_path=None):
    filename = os.path.basename(image_path)
    output_root = args.output_dir
    render_path = getattr(args, "render_dir", None) or os.path.join(args.root_dir, "render")
    feature_path = getattr(args, "feature_dir", None) or os.path.join(args.root_dir, "feature")
    mesh_path = os.path.join(render_path, "mesh.ply")
    voxels_path = os.path.join(render_path, "voxels.ply")
    slat_feature_path = os.path.join(render_path, "feature.npz")

    try:

        if args.mode == "edit":
            image = Image.open(image_path)
            ori_image_path = args.original_image
            edit_mask = load_edit_mask_from_rgba(image_path, target_size=(512, 512))
            mesh = o3d.io.read_triangle_mesh(mesh_path) if os.path.exists(mesh_path) else None
            ori_image = Image.open(ori_image_path)
            voxels, positions = voxelize(mesh, voxels_path=voxels_path)

            ss_latent_path = os.path.join(render_path, "ss_latent.pt")
            slat_latent_path = os.path.join(render_path, "slat_latent.pt")

            if os.path.exists(ss_latent_path):
                latent = torch.load(ss_latent_path, map_location="cuda")
            else:
                encoder = load_encoder(args)
                ss = get_voxels(voxels)[None].float().to("cuda")
                latent = encoder(ss, sample_posterior=False)
                del encoder
                torch.cuda.empty_cache()

            if os.path.exists(slat_latent_path):
                slat_cache = torch.load(slat_latent_path, map_location="cuda")
                slat_latent_tensor = sp.SparseTensor(
                    feats=slat_cache["feats"].cuda(),
                    coords=slat_cache["coords"].cuda(),
                )
            else:
                slat_latent_tensor = encode_slat_latent_from_feature_tensor(
                    feature_npz_path=slat_feature_path,
                    enc_pretrained_path=args.enc_pretrained,
                )

            # ---- Method D override: replace SLAT with a previous round's final edited SLAT ----
            slat_override_path = getattr(args, "slat_latent_override", None)
            if slat_override_path and os.path.isfile(slat_override_path):
                _prev_shape = tuple(slat_latent_tensor.feats.shape)
                slat_cache = torch.load(slat_override_path, map_location="cuda")
                slat_latent_tensor = sp.SparseTensor(
                    feats=slat_cache["feats"].cuda(),
                    coords=slat_cache["coords"].cuda(),
                )
                print(
                    f"[SLAT_REUSE] Overrode latent_slat with {slat_override_path} "
                    f"(feats: {_prev_shape} -> {tuple(slat_latent_tensor.feats.shape)}). "
                    f"Encoder-derived slat_latent.pt was discarded."
                )
            elif slat_override_path:
                print(
                    f"[SLAT_REUSE][WARN] slat_latent_override path does not exist: "
                    f"{slat_override_path}; falling back to encoder-derived SLAT."
                )

            # ---- Round continuation of the sparse-structure state (x_s^{r+1} <- x_edit^r) ----
            # Prefer the previous round's final structure latent; otherwise re-encode the
            # occupancy of the reused SLAT coords, which is the previous round's exact
            # structure (the preprocess path voxelizes the decoded mesh instead).
            ss_override_path = getattr(args, "ss_latent_override", None)
            if ss_override_path:
                if not os.path.isfile(ss_override_path):
                    raise FileNotFoundError(f"[SS_REUSE] Missing ss_latent_override: {ss_override_path}")
                latent = torch.load(ss_override_path, map_location="cuda")
                print(f"[SS_REUSE] Overrode sparse-structure latent with {ss_override_path}")
            elif getattr(args, "ss_latent_from_slat_override", False) and slat_override_path \
                    and os.path.isfile(slat_override_path):
                encoder = load_encoder(args)
                occupancy = get_voxels(slat_latent_tensor.coords[:, 1:4])[None].float()
                latent = encoder(occupancy, sample_posterior=False)
                del encoder
                torch.cuda.empty_cache()
                print(f"[SS_REUSE] Re-encoded sparse-structure latent from the {slat_latent_tensor.coords.shape[0]} "
                      f"coords of {slat_override_path}")

            edit_name = os.path.splitext(filename)[0]
            if getattr(args, "full_edit_mask", False):
                raw_mask = torch.ones((1, 1, 64, 64, 64), dtype=torch.float32, device="cuda")
                latent_mask = torch.ones((1, 8, 16, 16, 16), dtype=torch.float32, device="cuda")
                print("[INFO] Using full 64^3 edit mask; no 3D mask glb will be loaded.")
            elif getattr(args, "mask_voxels", None):
                raw_mask, latent_mask = _build_masks_from_voxels(
                    args.mask_voxels, latent_threshold=float(getattr(args, "mask_voxels_latent_threshold", 0.5)))
                print(f"[INFO] Using voxel edit mask {args.mask_voxels}: {int(raw_mask.sum())} voxels, "
                      f"{int(latent_mask[:, :1].sum())} latent cells.")
            else:
                if getattr(args, "mask_path", None):
                    mask_path = args.mask_path
                    if not os.path.exists(mask_path):
                        raise FileNotFoundError(f"[mask] Not found mask glb: {mask_path}")
                else:
                    if not args.mask_dir:
                        raise ValueError("Either --mask_path or --mask_dir must be provided unless --full_edit_mask is set")
                    mask_path = os.path.join(args.mask_dir, f"{edit_name}.glb")
                    if not os.path.exists(mask_path):
                        glb_files = [f for f in os.listdir(args.mask_dir) if f.endswith('.glb')]
                        if not glb_files:
                            raise FileNotFoundError(f"[mask] No .glb file found in {args.mask_dir}")
                        mask_path = os.path.join(args.mask_dir, glb_files[0])
                        print(f"[INFO] Mask '{edit_name}.glb' not found, using fallback: {glb_files[0]}")
                mask_cache_dir = os.path.join(args.root_dir, ".mask_cache", edit_name)
                os.makedirs(mask_cache_dir, exist_ok=True)
                _normalize_mask(args, mask_path, mask_cache_dir)
                mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
                normalized_ply = os.path.join(mask_cache_dir, f"{mask_stem}.ply")
                raw_mask, latent_mask = _build_masks_from_mask_glb(args, normalized_ply)

            outputs = pipeline.flowedit(
                ori_image,
                image,
                seed=args.seed,
                sparse_structure_sampler_params={
                    "steps": int(getattr(args, "sparse_structure_steps", None)
                                 or getattr(args, "sampler_steps", 25)),
                },
                slat_sampler_params={
                    "steps": int(getattr(args, "slat_steps", None)
                                 or getattr(args, "sampler_steps", 25)),
                },
                latent=latent,
                latent_slat=slat_latent_tensor,
                coords=voxels,
                latent_mask=latent_mask,
                raw_mask=raw_mask,
                feature_path=feature_path,
                edit_mask=edit_mask,
                cfg_src_strength=args.cfg_src,
                cfg_tar_strength=args.cfg_tar,
                flowedit_lambda=args.flowedit_lambda,
                mask_feather_outward=bool(getattr(args, "mask_feather_outward", False)),
                pair_crop=bool(getattr(args, "mv_fusion_pair_crop", False)),
                enable_step_viz=getattr(args, "enable_step_viz", False),
                step_viz_every=getattr(args, "step_viz_every", 5),
                use_guidance=not getattr(args, "no_guidance", False),
                azimuth_deg=_get_azimuth_from_ori(args),
                mv_velocity_debug=_build_mv_velocity_debug(args),
                mv_fusion=_build_mv_fusion(args),
            )

        else:
            image = Image.open(image_path)
            outputs = pipeline.run(
                image,
                seed=args.seed,
            )

        name_without_ext = os.path.splitext(filename)[0]
        os.makedirs(output_root, exist_ok=True)
        glb_path = os.path.join(output_root, f"{name_without_ext}.glb")

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0,
            texture_size=int(getattr(args, "texture_size", 2048)),
        )
        glb.export(glb_path)

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        traceback.print_exc()

    return True
