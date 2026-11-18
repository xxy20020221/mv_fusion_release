from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from torch import Tensor
from ...utils.general_utils import combine,build_coord_index_map,replace_feats,_normalize_boxes_zyx,inside_any_zyx_mask,_resolve_viz_base
import numpy as np
import torch
import torch.nn.functional as F
import os
from ...utils.general_utils import _to_mask_tensor, _silhouette_from_sigma, project_ortho_no_center,apply_orient_2d
from ...utils.general_utils import (
    _align_spatial, _avg_pool_nd, _max_pool_nd,
    apply_mask_blend, combine_masks,
    build_soft_masks, make_soft_mask,
    binarize, mask_metrics,
    reduce_vector_norm, build_auto_mask_from_map,
    _to_np_2d, _save_grid_2x2, HAS_PIL, _save_panel_png,_stat,
    blend_feats_ring1_outside_rawmask
)
from ...utils.general_utils import grad_chain_diagnostics,blend_feats_ring1_outside
from ...utils.general_utils import silhouette_at_azimuth, save_guidance_debug_viz
import json
import trimesh


def _export_coords_scalar_heatmap_glb(
    coords: torch.Tensor,
    values: torch.Tensor,
    output_filename: str,
    grid_size: int = 64,
    value_range: Optional[Tuple[float, float]] = None,
):
    coords_cpu = coords.detach().cpu()
    values_cpu = values.detach().float().cpu().view(-1)
    xyz_indices_np = coords_cpu[:, 1:4].numpy()
    if xyz_indices_np.shape[0] == 0:
        return

    voxel_size = 1.0 / float(grid_size)
    base_cube = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    num_vertices_per_cube = len(base_cube.vertices)
    world_centers_np = (xyz_indices_np + 0.5) * voxel_size - 0.5
    final_mesh = trimesh.util.concatenate([base_cube.copy().apply_translation(p) for p in world_centers_np])

    if value_range is None:
        v_min = float(values_cpu.min().item())
        v_max = float(values_cpu.max().item())
    else:
        v_min, v_max = float(value_range[0]), float(value_range[1])
    norm = ((values_cpu - v_min) / max(v_max - v_min, 1e-6)).clamp(0.0, 1.0)
    norm_np = norm.numpy()
    low = np.array([255, 238, 225], dtype=np.float32)
    high = np.array([165, 0, 38], dtype=np.float32)
    rgb = (low[None, :] * (1.0 - norm_np[:, None]) + high[None, :] * norm_np[:, None]).clip(0, 255).astype(np.uint8)
    alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    voxel_colors = np.concatenate([rgb, alpha], axis=1)
    final_mesh.visual.vertex_colors = np.repeat(voxel_colors, num_vertices_per_cube, axis=0)
    final_mesh.export(output_filename)


def _export_coords_category_glb(coords: torch.Tensor, labels: torch.Tensor, output_filename: str, grid_size: int = 64):
    coords_cpu = coords.detach().cpu()
    labels_cpu = labels.detach().long().cpu().view(-1)
    xyz_indices_np = coords_cpu[:, 1:4].numpy()
    if xyz_indices_np.shape[0] == 0:
        return

    voxel_size = 1.0 / float(grid_size)
    base_cube = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    num_vertices_per_cube = len(base_cube.vertices)
    world_centers_np = (xyz_indices_np + 0.5) * voxel_size - 0.5
    final_mesh = trimesh.util.concatenate([base_cube.copy().apply_translation(p) for p in world_centers_np])
    palette = np.array([
        [160, 160, 160, 120],
        [80, 180, 255, 220],
        [255, 80, 80, 240],
        [170, 170, 170, 180],
    ], dtype=np.uint8)
    voxel_colors = palette[np.clip(labels_cpu.numpy(), 0, len(palette) - 1)]
    final_mesh.visual.vertex_colors = np.repeat(voxel_colors, num_vertices_per_cube, axis=0)
    final_mesh.export(output_filename)


def _load_scene_bounds(path: str) -> np.ndarray:
    loaded = trimesh.load(path, force="scene")
    bounds = np.asarray(loaded.bounds, dtype=np.float32)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        mesh = loaded.dump(concatenate=True) if hasattr(loaded, "dump") else loaded
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"Invalid confidence box bounds: {path}")
    return bounds


def _build_alpha_mask_from_box_bounds(bounds: np.ndarray, sample: torch.Tensor) -> torch.Tensor:
    D, H, W = sample.shape[-3:]
    box_min = np.clip(bounds[0], -0.5, 0.5)
    box_max = np.clip(bounds[1], -0.5, 0.5)
    if np.any(box_max <= box_min):
        raise ValueError(f"Degenerate confidence box bounds after clipping: min={box_min.tolist()}, max={box_max.tolist()}")

    xs = (torch.arange(D, device=sample.device, dtype=sample.dtype) + 0.5) / float(D) - 0.5
    ys = (torch.arange(H, device=sample.device, dtype=sample.dtype) + 0.5) / float(H) - 0.5
    zs = (torch.arange(W, device=sample.device, dtype=sample.dtype) + 0.5) / float(W) - 0.5
    mx = (xs >= float(box_min[0])) & (xs <= float(box_max[0]))
    my = (ys >= float(box_min[1])) & (ys <= float(box_max[1]))
    mz = (zs >= float(box_min[2])) & (zs <= float(box_max[2]))
    mask = mx[:, None, None] & my[None, :, None] & mz[None, None, :]
    return mask[None, None].expand(sample.shape[0], 1, D, H, W).contiguous()


def _build_fixed_alpha_mask(region_mask: torch.Tensor, alpha_inside: float, alpha_outside: float, sample: torch.Tensor) -> torch.Tensor:
    inside = max(0.0, min(1.0, float(alpha_inside)))
    outside = max(0.0, min(1.0, float(alpha_outside)))
    return torch.where(
        region_mask.to(device=sample.device),
        torch.full_like(sample[:, :1], inside),
        torch.full_like(sample[:, :1], outside),
    )


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _box_soft_alpha_from_xyz(
    xyz: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
    edge_width: float,
    core_width: float,
    outside_width: float,
    inside_plateau: float,
) -> torch.Tensor:
    eps = 1e-6
    box_min = torch.minimum(box_min, box_max - eps)
    edge_width = max(float(edge_width), eps)
    core_width = max(float(core_width), eps)
    outside_width = max(float(outside_width), eps)
    inside_plateau = max(0.5, min(1.0, float(inside_plateau)))

    inside_axis = (xyz >= box_min) & (xyz <= box_max)
    inside = inside_axis.all(dim=-1)

    dist_to_faces = torch.minimum(xyz - box_min, box_max - xyz)
    inner_dist = dist_to_faces.min(dim=-1).values.clamp(min=0.0)
    edge_r = _smoothstep01(inner_dist / edge_width)
    core_r = _smoothstep01((inner_dist - edge_width) / core_width)
    edge_to_plateau = 0.5 + (inside_plateau - 0.5) * edge_r
    plateau_to_core = inside_plateau + (1.0 - inside_plateau) * core_r
    inside_alpha = torch.where(inner_dist < edge_width, edge_to_plateau, plateau_to_core)

    outside_delta = torch.maximum(torch.maximum(box_min - xyz, xyz - box_max), torch.zeros_like(xyz))
    outside_dist = outside_delta.norm(dim=-1)
    outside_alpha = 0.5 * (1.0 - _smoothstep01(outside_dist / outside_width))

    return torch.where(inside, inside_alpha, outside_alpha).clamp(0.0, 1.0)


def _build_soft_alpha_mask_from_box_bounds(
    bounds: np.ndarray,
    sample: torch.Tensor,
    edge_width: float,
    core_width: float,
    outside_width: float,
    inside_plateau: float,
) -> torch.Tensor:
    D, H, W = sample.shape[-3:]
    bounds_t = torch.as_tensor(bounds, device=sample.device, dtype=sample.dtype)
    box_min = bounds_t[0].clamp(-0.5, 0.5)
    box_max = bounds_t[1].clamp(-0.5, 0.5)
    if torch.any(box_max <= box_min):
        raise ValueError(f"Degenerate confidence box bounds after clipping: min={box_min.detach().cpu().tolist()}, max={box_max.detach().cpu().tolist()}")

    xs = (torch.arange(D, device=sample.device, dtype=sample.dtype) + 0.5) / float(D) - 0.5
    ys = (torch.arange(H, device=sample.device, dtype=sample.dtype) + 0.5) / float(H) - 0.5
    zs = (torch.arange(W, device=sample.device, dtype=sample.dtype) + 0.5) / float(W) - 0.5
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    alpha = _box_soft_alpha_from_xyz(
        grid,
        box_min,
        box_max,
        edge_width=edge_width,
        core_width=core_width,
        outside_width=outside_width,
        inside_plateau=inside_plateau,
    )
    return alpha[None, None].expand(sample.shape[0], 1, D, H, W).contiguous()


def _fusion_strength_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    return (1.0 - (2.0 * alpha - 1.0).abs()).clamp(0.0, 1.0)


class FlowEulerSampler(Sampler):
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
     
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 28,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        mode = kwargs.get("mode","baseline")
        if mode == "flowedit":
            return self.sample_flowedit(model,noise,cond,steps,rescale_t,verbose,**kwargs)
        elif mode == "repaint":
            return self.sample_repaint(model,noise,cond,steps,rescale_t,verbose,**kwargs)
        else:
            return self.sample_ori(model,noise,cond,steps,rescale_t,verbose,**kwargs)
    
   
    _FE_N_MIN        = 2     
    _FE_N_AVG        = 4     
    _FE_FEATHER      = 2     
    _FE_GUARD        = 1    
    _FE_GUID_START   = 6    

    _FE_ETA          = 0.1   
    _FE_GAMMA        = 0.1   
    _FE_BETA_NORM    = 1.0   
    _FE_LAM_OUT      = 0.5   
    _FE_LAM_IN       = 0.1   

    _RP_BETA         = 0.8   
    _RP_ANCHOR_EPS   = True  
    _RP_INV_STEPS    = 4     

    @torch.no_grad()
    def sample_flowedit(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 25,
        rescale_t: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        
        cond_src         = kwargs["ori_cond"]
        cond_tar         = cond
        mv_debug         = kwargs.pop("mv_velocity_debug", None)
        mv_fusion        = kwargs.pop("mv_fusion", None)
        manual_mask      = kwargs.get("mask", None)
        cfg_src_strength = float(kwargs.get("cfg_src_strength", 5.0))
        cfg_tar_strength = float(kwargs.get("cfg_tar_strength", 5.0))
        use_guidance     = bool(kwargs.get("use_guidance", True))
        azimuth_deg      = float(kwargs.get("azimuth_deg", 270.0))
        debug_guid       = bool(kwargs.get("debug_guidance_viz", False))
        debug_viz_dir    = kwargs.get("debug_viz_dir", None)
        decode_voxel_fn  = kwargs.get("decode_voxel_fn", None)
        flowedit_lambda  = float(kwargs.get("flowedit_lambda", 0.5))

        lock_sched = lambda t: min(0.95, 0.40 + (1.0 - t) * 0.45)

        def forward_noise(x0, t_scalar, eps):
            s = 1e-5
            tv = float(t_scalar)
            return (1.0 - tv) * x0 + (s + (1 - s) * tv) * eps

        def _mul_mask(v):
            return v if combined_soft is None else v * combined_soft

        sample = noise
        x_src0 = kwargs.get("src_latent", None)
        if x_src0 is None:
            x_src0 = sample.clone()

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

        _edict = lambda d: type("edict", (object,), d)
        ret = _edict({
            "samples": None, "pred_x_t": [], "pred_x_0": [],
            "viz_grid_paths": [], "auto_masks": [], "mask_metrics": []
        })

        combined_soft = None
        if manual_mask is not None:
            combined_soft, _ = build_soft_masks(
                manual_mask, sample,
                feather=self._FE_FEATHER, guard=self._FE_GUARD
            )
            if bool(kwargs.get("mask_feather_outward", False)):
                # The average-pool feather also attenuates the interior: a region only a few latent
                # cells thick never exceeds ~0.35, which damps the edit and only partly noises the
                # source there. Keep the interior at 1 and feather outward only.
                combined_soft = torch.maximum(combined_soft, manual_mask.to(combined_soft.dtype))

        edit_mask = _to_mask_tensor(kwargs.get("edit_mask", None), sample)
        if edit_mask.shape[0] != sample.shape[0]:
            if edit_mask.shape[0] == 1:
                edit_mask = edit_mask.expand(sample.shape[0], -1, -1, -1)
            else:
                raise ValueError(
                    f"edit_mask batch mismatch with sample: {edit_mask.shape[0]} vs {sample.shape[0]}"
                )
        H_mask, W_mask = int(edit_mask.shape[-2]), int(edit_mask.shape[-1])

        
        from ...utils.general_utils import rotate_volume_z as _rot_vol_z

        def _build_3D_mask(edit_mask_2d, sigma_perm_shape):
            B, _, Y_dim, Z_dim, X_dim = sigma_perm_shape
            m = F.interpolate(edit_mask_2d, size=(X_dim, Z_dim), mode='bilinear', align_corners=True)
            m = m.flip(-2).transpose(-2, -1)
            m = m.unsqueeze(2).expand(B, 1, Y_dim, Z_dim, X_dim)
            return m.contiguous()

        
        def _guide(state, t_val, dt, step_i, prefix, v_norm_ref, xi_traj=None):
            if not use_guidance:
                return state
            try:
                with torch.enable_grad():
                    x_t   = state.detach().requires_grad_(True)
                    sigma = decode_voxel_fn(x_t)
                    tau   = 0.6 if t_val > 0.5 else 0.5

                    delta_rad = float(np.radians(azimuth_deg - 270.0))
                    rotated = sigma if abs((azimuth_deg - 270.0) % 360.0) < 1e-3 \
                              else _rot_vol_z(sigma, delta_rad)
                    sigma_perm = rotated.permute(0, 1, 3, 2, 4).contiguous()

                    M_3D = _build_3D_mask(edit_mask, sigma_perm.shape)

                    s = sigma_perm / tau
                    L_out = (F.softplus(s) * (1.0 - M_3D)).sum() \
                            / ((1.0 - M_3D).sum().clamp(min=1.0))
                    M_2D_ray = M_3D.amax(dim=2)
                    lse_y    = torch.logsumexp(s, dim=2)
                    L_in     = (F.softplus(-lse_y) * M_2D_ray).sum() \
                               / (M_2D_ray.sum().clamp(min=1.0))

                    L = self._FE_LAM_OUT * L_out + self._FE_LAM_IN * L_in
                    g = torch.autograd.grad(L, x_t)[0]
                    ret.mask_metrics.append({
                        "step": int(step_i),
                        "L_out": float(L_out.detach().cpu()),
                        "L_in":  float(L_in.detach().cpu()),
                    })

                if debug_guid and debug_viz_dir is not None:
                    with torch.no_grad():
                        sil_raw = silhouette_at_azimuth(sigma.detach(), azimuth_deg=azimuth_deg, tau=tau)
                        sil_img = project_ortho_no_center(sil_raw, out_hw=(H_mask, W_mask), ortho_scale=1.0)
                        save_guidance_debug_viz(
                            sil_img, edit_mask, step_i, azimuth_deg,
                            debug_viz_dir, prefix=prefix
                        )

                g = _mul_mask(g)
                g_norm = g.norm() + 1e-8
                scale  = self._FE_BETA_NORM * (v_norm_ref / g_norm)
                tilde_g = scale * g

                state = state - dt * (self._FE_ETA * tilde_g)
                if xi_traj is not None:
                    state = state - dt * (self._FE_GAMMA * _mul_mask(xi_traj))
            except Exception as e:
                ret.mask_metrics.append({"step": int(step_i), "err": str(e)})
            return state

        def _lock_back(x, t_val):
            if manual_mask is None:
                return x
            lk = lock_sched(t_val)
            if lk <= 0:
                return x
            keep = lk * (1 - manual_mask)
            return (1 - keep) * x + keep * x_src0

        def _cond_at(cond_dict, view_i):
            return {
                key: (value[view_i:view_i + 1] if torch.is_tensor(value) and value.shape[0] > view_i else value)
                for key, value in cond_dict.items()
            }

        def _predict_v_with_cond(cond_dict, x_t, t_val, cfg_strength):
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in {
                "ori_cond", "cfg_src_strength", "cfg_tar_strength", "flowedit_lambda", "mask_feather_outward", "use_guidance",
                "enable_step_viz", "step_viz_every", "azimuth_deg",
                "debug_guidance_viz", "debug_viz_dir", "decode_voxel_fn",
            }}
            clean_kwargs["cfg_strength"] = cfg_strength
            if "neg_cond" in cond_dict:
                clean_kwargs["neg_cond"] = cond_dict["neg_cond"]
            if "cfg_interval" in kwargs:
                clean_kwargs["cfg_interval"] = kwargs["cfg_interval"]
            return self._get_model_prediction(model, x_t, t_val, cond_dict.get("cond"), **clean_kwargs)[2]

        mv_fusion_enabled = (
            mv_fusion is not None
            and decode_voxel_fn is not None
            and len(mv_fusion.get("view_names", [])) >= 2
        )
        mv_fusion_steps = set(int(s) for s in mv_fusion.get("steps", [1, 3, 5, 10, 20])) if mv_fusion_enabled else set()
        if mv_fusion_enabled:
            os.makedirs(mv_fusion["output_dir"], exist_ok=True)
            view_names_fuse = mv_fusion["view_names"]
            fusion_mode = str(mv_fusion.get("fusion_mode", "trust_region")).strip().lower()
            if fusion_mode not in {"trust_region", "view_average"}:
                raise ValueError(f"mv_fusion.fusion_mode must be 'trust_region' or 'view_average', got {fusion_mode!r}")
            all_view_indices = list(range(len(view_names_fuse)))
            # ---- multi-trust-view setup ----
            # Resolve the ordered list of trust view indices. Later entries win on overlap
            # (i.e. "the newly-added view has priority"). Also keep legacy trust_i/base_i so
            # every downstream helper that still references them keeps working.
            trust_views_list = list(mv_fusion.get("trust_views", []) or [mv_fusion.get("trust_view", view_names_fuse[0])])
            trust_indices = [view_names_fuse.index(v) for v in trust_views_list]
            base_i = view_names_fuse.index(mv_fusion.get("base_view", view_names_fuse[-1]))
            trust_i = trust_indices[0]      # legacy single-trust index (first trust view)

            # ---- per-trust-view confidence regions (list, aligned with trust_indices) ----
            conf_box_glbs = list(mv_fusion.get("conf_box_glbs", []) or [mv_fusion.get("conf_box_glb", None)])
            conf_box_bounds_list_raw = mv_fusion.get("conf_box_bounds_list", None)
            if conf_box_bounds_list_raw is None:
                # legacy single-bounds fallback
                _cb = mv_fusion.get("conf_box_bounds", None)
                conf_box_bounds_list_raw = [_cb]
            support_chi64 = mv_fusion.get("support_chi64", None)

            trust_region_masks = []   # list of bool tensors [B,1,D,H,W]
            trust_region_bounds_all = []  # list of np.ndarray or None
            support_alpha_raw = []    # geometry-derived chi on the latent grid, per trust view
            if support_chi64 is not None:
                # Geometry-derived support (compute_view_support.py): average chi_v over the
                # 4^3 voxels of each latent cell; it is the raw alpha of that trust view.
                for tv_name in trust_views_list:
                    chi64 = torch.as_tensor(support_chi64[tv_name], device=sample.device, dtype=sample.dtype)
                    chi_lat = F.adaptive_avg_pool3d(chi64[None, None], tuple(sample.shape[-3:])).clamp(0.0, 1.0)
                    chi_lat = chi_lat.expand(sample.shape[0], 1, *sample.shape[-3:]).contiguous()
                    support_alpha_raw.append(chi_lat)
                    trust_region_masks.append(chi_lat > 0.5)
                    trust_region_bounds_all.append(None)
            box_entries = [] if support_chi64 is not None else zip(trust_views_list, conf_box_glbs, conf_box_bounds_list_raw)
            for tv_i, (tv_name, gpath, provided_bounds) in enumerate(box_entries):
                if gpath:
                    if provided_bounds is not None:
                        tv_bounds = np.asarray(provided_bounds, dtype=np.float32)
                    else:
                        tv_bounds = _load_scene_bounds(gpath)
                    tv_region = _build_alpha_mask_from_box_bounds(tv_bounds, sample)
                else:
                    # image-space bbox fallback (only meaningful when there is exactly one trust view)
                    tv_bounds = None
                    x0, y0, x1, y1 = [float(v) for v in mv_fusion.get("conf_bbox", [0.36, 0.30, 0.64, 0.58])]
                    tv_region = torch.zeros(
                        (sample.shape[0], 1, sample.shape[-3], sample.shape[-2], sample.shape[-1]),
                        device=sample.device, dtype=torch.bool,
                    )
                    z0 = max(0, min(sample.shape[-3], int(round(y0 * sample.shape[-3]))))
                    z1 = max(z0 + 1, min(sample.shape[-3], int(round(y1 * sample.shape[-3]))))
                    x0i = max(0, min(sample.shape[-1], int(round(x0 * sample.shape[-1]))))
                    x1i = max(x0i + 1, min(sample.shape[-1], int(round(x1 * sample.shape[-1]))))
                    tv_region[:, :, :, z0:z1, x0i:x1i] = True
                trust_region_masks.append(tv_region)
                trust_region_bounds_all.append(tv_bounds)

            # Union of all per-trust-view regions (for legacy trust_region_mask users like
            # _export_mv_fusion_trust_region_voxels).
            trust_region_mask = trust_region_masks[0].clone()
            for m in trust_region_masks[1:]:
                trust_region_mask = trust_region_mask | m
            # Also keep the legacy scalar bounds for logging (from the primary trust view).
            conf_box_bounds = trust_region_bounds_all[0] if trust_region_bounds_all else None
            conf_box_glb = conf_box_glbs[0] if conf_box_glbs else None

            alpha_inside = float(mv_fusion.get("alpha_inside", 1.0))
            alpha_outside = float(mv_fusion.get("alpha_outside", 0.0))
            soft_blend = bool(mv_fusion.get("soft_blend", False))
            soft_edge_width = float(mv_fusion.get("soft_edge_width", 0.035))
            soft_core_width = float(mv_fusion.get("soft_core_width", 0.04))
            soft_outside_width = float(mv_fusion.get("soft_outside_width", 0.04))
            soft_inside_plateau = float(mv_fusion.get("soft_inside_plateau", 0.9))

            # ---- build per-trust-view alpha masks ----
            # For each trust view k, alpha_masks_per_trust[k] is the per-voxel weight assigned
            # to trust view k's velocity. Later trust views win on overlap: we allocate the
            # remaining un-claimed weight ("owner slot") to each successive trust view.
            per_trust_alpha_raw = []   # raw (unclipped) alpha per trust view, from its own region
            for tv_i, tv_bounds in enumerate(trust_region_bounds_all):
                if support_alpha_raw:
                    a = support_alpha_raw[tv_i]
                elif soft_blend and tv_bounds is not None:
                    a = _build_soft_alpha_mask_from_box_bounds(
                        tv_bounds,
                        sample,
                        edge_width=soft_edge_width,
                        core_width=soft_core_width,
                        outside_width=soft_outside_width,
                        inside_plateau=soft_inside_plateau,
                    )
                else:
                    a = _build_fixed_alpha_mask(
                        trust_region_masks[tv_i], alpha_inside, alpha_outside, sample,
                    )
                per_trust_alpha_raw.append(a)

            if fusion_mode == "view_average":
                # Ablation mode: ignore confidence regions for the actual velocity fusion.
                # Keep uniform diagnostic alpha masks so exported stats/heatmaps reflect
                # equal weighting rather than the trust-region layout.
                uniform_alpha = torch.full_like(
                    per_trust_alpha_raw[0],
                    1.0 / float(len(view_names_fuse)),
                )
                alpha_masks_per_trust = [uniform_alpha.clone() for _ in range(len(trust_indices))]
                alpha_base_mask = uniform_alpha.clone()
            else:
                # Resolve overlaps with "new views win": iterate trust_views_list in order and
                # give each successive trust view priority over previously-assigned weight.
                # Concretely, if a voxel is claimed by an earlier trust view with alpha_prev and
                # by the current trust view with alpha_cur, we replace the earlier claim.
                # Simpler / more robust rule for N views:
                #   remaining = 1
                #   for k in reversed(order):        # last (newest) trust view first
                #       alpha_k = per_trust_alpha_raw[k] * remaining
                #       remaining = remaining * (1 - per_trust_alpha_raw[k])
                #   base weight = remaining
                alpha_masks_per_trust = [None] * len(trust_indices)
                remaining = torch.ones_like(per_trust_alpha_raw[0])
                for k in reversed(range(len(trust_indices))):
                    a_k = per_trust_alpha_raw[k].clamp(0.0, 1.0)
                    alpha_k = a_k * remaining
                    alpha_masks_per_trust[k] = alpha_k
                    remaining = remaining * (1.0 - a_k)
                alpha_base_mask = remaining
            # legacy scalar alpha_mask for the primary trust view (used by trust-region
            # export and blend-heatmap visualisation). It represents "alpha assigned to the
            # primary trust view" and is <= per_trust_alpha_raw[0] in trust_region mode;
            # in view_average mode it is the uniform per-view weight.
            alpha_mask = alpha_masks_per_trust[0]
            fusion_strength_mask = _fusion_strength_from_alpha(alpha_mask)
            alpha_stats = {
                "mode": fusion_mode,
                "alpha_mode": (
                    "view_support" if support_alpha_raw
                    else "soft" if soft_blend and conf_box_bounds is not None else "hard"
                ),
                "n_trust_views": len(trust_indices),
                "trust_alpha_stats": [
                    {
                        "trust_view": trust_views_list[k],
                        "min": float(alpha_masks_per_trust[k].min().item()),
                        "mean": float(alpha_masks_per_trust[k].mean().item()),
                        "max": float(alpha_masks_per_trust[k].max().item()),
                        "nonzero": int((alpha_masks_per_trust[k] > 1e-6).sum().item()),
                        "trust_region_cells": int(trust_region_masks[k].sum().item()),
                    }
                    for k in range(len(trust_indices))
                ],
                "base_alpha_min": float(alpha_base_mask.min().item()),
                "base_alpha_mean": float(alpha_base_mask.mean().item()),
                "base_alpha_max": float(alpha_base_mask.max().item()),
                # legacy fields (for backward compatibility with downstream logging):
                "min": float(alpha_mask.min().item()),
                "mean": float(alpha_mask.mean().item()),
                "max": float(alpha_mask.max().item()),
                "nonzero": int((alpha_mask > 1e-6).sum().item()),
                "trust_region_cells": int(trust_region_mask.sum().item()),
                "alpha_inside": float(max(0.0, min(1.0, alpha_inside))),
                "alpha_outside": float(max(0.0, min(1.0, alpha_outside))),
                "soft_edge_width": soft_edge_width,
                "soft_core_width": soft_core_width,
                "soft_outside_width": soft_outside_width,
                "soft_inside_plateau": soft_inside_plateau,
                "fusion_strength_min": float(fusion_strength_mask.min().item()),
                "fusion_strength_mean": float(fusion_strength_mask.mean().item()),
                "fusion_strength_max": float(fusion_strength_mask.max().item()),
            }
            mv_fusion["alpha_stats"] = alpha_stats
            if conf_box_bounds is not None:
                mv_fusion["conf_box_bounds"] = conf_box_bounds.tolist()
            print(
                f"[MV_FUSION] trust_views={trust_views_list}, base={view_names_fuse[base_i]}, "
                f"n_views={len(view_names_fuse)}, fusion_mode={fusion_mode}, conf_box_glbs={conf_box_glbs}, "
                f"alpha_stats={alpha_stats}"
            )
        else:
            alpha_mask = None
            alpha_masks_per_trust = None
            alpha_base_mask = None
            trust_region_mask = None
            trust_region_masks = None
            trust_indices = None
            trust_i = base_i = None
            trust_views_list = None

        # ---- Velocity-conflict accumulator (mv_fusion only) ----
        # We accumulate |v_tar_trust - v_tar_base| every FlowEdit step (per-voxel
        # scalar) and finally export a heatmap + a thresholded "conflict" glb
        # restricted to the mask_all region.
        #
        # Two levels of accumulators are maintained:
        #   * velocity_conflict_accum: multi-view mean pairwise disagreement,
        #     kept for backward compatibility with the round-1 (N=2) output.
        #   * pairwise_velocity_conflict_accums: dict keyed by (view_a, view_b)
        #     -> per-pair scalar accumulator, so for N>2 views we can export a
        #     dedicated heatmap for each pair (e.g. new_view vs each original
        #     view, without them being mixed together).
        velocity_conflict_accum = None
        velocity_conflict_count = 0
        pairwise_velocity_conflict_accums = {}   # (view_a, view_b) -> tensor
        pairwise_velocity_conflict_count = 0
        # Edit-magnitude accumulator (mv_fusion only): accumulate |v_tar_fused| per FlowEdit
        # step (per-voxel scalar). Combined with velocity_conflict_accum at the end to
        # classify quiescent voxels (multi-view consistent AND low edit magnitude).
        edit_magnitude_accum = None
        edit_magnitude_count = 0
        ori_mask_full = kwargs.get("ori_mask", None)
        velocity_conflict_threshold_cfg = None
        velocity_conflict_min_ratio = 0.0
        edit_magnitude_conflict_enabled_cfg = False
        edit_magnitude_conflict_threshold_cfg = "auto"
        edit_magnitude_conflict_min_ratio = 0.0
        # Quiescent-region config (mv_fusion only). See _compute_and_export_quiescent().
        # `quiescent_enabled` is a hard on/off switch (default True for backward
        # compatibility). When False, quiescent detection / export / shrinkage
        # are ALL skipped, regardless of the threshold / ratio settings below.
        quiescent_enabled_cfg = True
        quiescent_ratio_cfg = 0.0
        quiescent_disagreement_thr_cfg = "auto"
        quiescent_edit_mag_thr_cfg = "auto"
        # Which diagnosed voxels are locked back to the source before the SLAT stage:
        #   three_state          -- quiescent U only (default; C and edited-stable S keep the edit)
        #   three_state_srcocc   -- U voxels also occupied in the source only (flowedit() applies the
        #                           source-occupancy filter), so locking back never deletes new geometry
        #   no_lockback          -- two states (C / stable), nothing is locked back
        #   lockback_nonconflict -- two states (C / source), every non-conflict cell of mask_all
        #                           (decoded or not) is locked back, so S is reverted as well
        state_mode_cfg = "three_state"
        if mv_fusion_enabled:
            velocity_conflict_threshold_cfg = mv_fusion.get("velocity_conflict_threshold", "auto")
            velocity_conflict_min_ratio = float(mv_fusion.get("velocity_conflict_min_ratio", 0.0))
            edit_magnitude_conflict_enabled_cfg = bool(mv_fusion.get("export_edit_magnitude_conflict", False))
            edit_magnitude_conflict_threshold_cfg = mv_fusion.get("edit_magnitude_conflict_threshold", "auto")
            edit_magnitude_conflict_min_ratio = float(mv_fusion.get("edit_magnitude_conflict_min_ratio", 0.0))
            quiescent_enabled_cfg = bool(mv_fusion.get("quiescent_enabled", True))
            quiescent_ratio_cfg = float(mv_fusion.get("quiescent_ratio", 0.0))
            quiescent_disagreement_thr_cfg = mv_fusion.get("quiescent_disagreement_threshold", "auto")
            quiescent_edit_mag_thr_cfg = mv_fusion.get("quiescent_edit_mag_threshold", "auto")
            state_mode_cfg = str(mv_fusion.get("state_mode", "three_state"))
            if state_mode_cfg not in ("three_state", "three_state_srcocc", "no_lockback", "lockback_nonconflict"):
                raise ValueError(f"Unknown mv_fusion state_mode {state_mode_cfg!r}")

        def _accumulate_velocity_conflict(v_trust, v_base):
            """Accumulate reduce_vector_norm(v_trust - v_base) over FlowEdit steps."""
            nonlocal velocity_conflict_accum, velocity_conflict_count
            if not mv_fusion_enabled:
                return
            try:
                diff = reduce_vector_norm(v_trust.float() - v_base.float()).detach()
                if velocity_conflict_accum is None:
                    velocity_conflict_accum = torch.zeros_like(diff)
                velocity_conflict_accum = velocity_conflict_accum + diff
                velocity_conflict_count += 1
            except Exception as e:
                ret.mask_metrics.append({"step": -1, "mv_fusion_velocity_conflict_accum_err": str(e)})

        def _accumulate_velocity_conflict_multi(v_list, view_names=None):
            """Accumulate multi-view velocity disagreement over FlowEdit steps.

            For a list of N velocity tensors (all trust views + base view), we compute
            the mean per-voxel pairwise disagreement:
                diff = (1 / C(N,2)) * sum_{i<j} reduce_vector_norm(v_i - v_j)
            For N=2 this reduces exactly to the previous |v_trust - v_base| definition.

            If ``view_names`` is provided (same length as v_list), we ALSO maintain
            a per-pair accumulator in ``pairwise_velocity_conflict_accums`` so each
            pair (view_a, view_b) can later produce its own dedicated heatmap.
            """
            nonlocal velocity_conflict_accum, velocity_conflict_count
            nonlocal pairwise_velocity_conflict_count
            if not mv_fusion_enabled:
                return
            if v_list is None or len(v_list) < 2:
                return
            try:
                n = len(v_list)
                num_pairs = n * (n - 1) // 2
                acc = None
                # Compute each pair's diff once, feed into both the mean accumulator
                # and the per-pair accumulator (if view names available).
                for i in range(n):
                    for j in range(i + 1, n):
                        d = reduce_vector_norm(v_list[i].float() - v_list[j].float()).detach()
                        acc = d if acc is None else acc + d
                        if view_names is not None and len(view_names) == n:
                            key = (str(view_names[i]), str(view_names[j]))
                            if key not in pairwise_velocity_conflict_accums:
                                pairwise_velocity_conflict_accums[key] = torch.zeros_like(d)
                            pairwise_velocity_conflict_accums[key] = pairwise_velocity_conflict_accums[key] + d
                diff = acc / float(num_pairs)
                if velocity_conflict_accum is None:
                    velocity_conflict_accum = torch.zeros_like(diff)
                velocity_conflict_accum = velocity_conflict_accum + diff
                velocity_conflict_count += 1
                if view_names is not None and len(view_names) == n:
                    pairwise_velocity_conflict_count += 1
            except Exception as e:
                ret.mask_metrics.append({"step": -1, "mv_fusion_velocity_conflict_accum_multi_err": str(e)})

        def _accumulate_edit_magnitude(v_tar_fused):
            """Accumulate |v_tar_fused| (per-voxel scalar) over FlowEdit steps.

            Later combined with the velocity-conflict accumulator to classify quiescent
            voxels (a voxel is quiescent iff both disagreement and edit magnitude are low).
            """
            nonlocal edit_magnitude_accum, edit_magnitude_count
            if not mv_fusion_enabled:
                return
            try:
                mag = reduce_vector_norm(v_tar_fused.float()).detach()
                if edit_magnitude_accum is None:
                    edit_magnitude_accum = torch.zeros_like(mag)
                edit_magnitude_accum = edit_magnitude_accum + mag
                edit_magnitude_count += 1
            except Exception as e:
                ret.mask_metrics.append({"step": -1, "mv_fusion_edit_magnitude_accum_err": str(e)})

        def _export_mv_fusion_velocity_conflict():
            """Export a heatmap + thresholded mask of trust-vs-base velocity
            disagreement inside mask_all.

            When more than one trust view is present (round 2+), we additionally
            export one heatmap/mask/stats per (view_i, view_j) pair so the
            user can inspect e.g. the new view's disagreement with each original
            view separately (without averaging).
            """
            if (not mv_fusion_enabled) or velocity_conflict_accum is None or velocity_conflict_count <= 0:
                return

            trust_views_export = list(mv_fusion.get("trust_views", []) or [mv_fusion.get("trust_view", "trust")])
            trust_name = "_".join(trust_views_export)
            base_name = mv_fusion.get("base_view", "base")
            out_dir = mv_fusion["output_dir"]
            os.makedirs(out_dir, exist_ok=True)

            # (1) Aggregate multi-view mean-pairwise heatmap (backward compatible).
            mean_diff = (velocity_conflict_accum / float(velocity_conflict_count)).detach().float()
            _export_single_velocity_conflict_glb(
                mean_diff_16=mean_diff,
                stem=f"velocity_conflict_{trust_name}_vs_{base_name}",
                out_dir=out_dir,
                pair_a=trust_name,
                pair_b=base_name,
                accumulated_steps=int(velocity_conflict_count),
                extra_note="Multi-view mean pairwise disagreement (averaged over all C(N,2) pairs of trust views + base view).",
            )

            # (2) Per-pair heatmaps: only meaningful when we have >= 3 views.
            if pairwise_velocity_conflict_count > 0 and len(pairwise_velocity_conflict_accums) > 1:
                for (view_a, view_b), pair_accum in pairwise_velocity_conflict_accums.items():
                    pair_mean = (pair_accum / float(pairwise_velocity_conflict_count)).detach().float()
                    _export_single_velocity_conflict_glb(
                        mean_diff_16=pair_mean,
                        stem=f"velocity_conflict_{view_a}_vs_{view_b}",
                        out_dir=out_dir,
                        pair_a=view_a,
                        pair_b=view_b,
                        accumulated_steps=int(pairwise_velocity_conflict_count),
                        extra_note=f"Per-pair velocity disagreement between view '{view_a}' and view '{view_b}' (not averaged with other pairs).",
                    )

        def _export_single_velocity_conflict_glb(
            mean_diff_16, stem, out_dir, pair_a, pair_b, accumulated_steps, extra_note=""
        ):
            """Actually decode + upsample + threshold + export a single heatmap/mask/stats
            triple. Split out so we can also emit per-pair heatmaps in the multi-view case.
            ``mean_diff_16`` is the step-averaged per-voxel scalar tensor at 16^3 latent
            resolution; the rest is metadata for naming and provenance."""
            try:
                decoded = decode_voxel_fn(sample.detach()) > 0
                target_shape = decoded.shape[-3:]

                # Upsample mean_diff (in 16^3 latent space) to voxel resolution (64^3).
                mean_diff_hi = F.interpolate(
                    mean_diff_16,
                    size=target_shape,
                    mode="trilinear",
                    align_corners=False,
                )

                # mask_all (raw_mask, native 64^3). If not provided, fall back to full volume.
                if ori_mask_full is not None:
                    mask_all_hi = F.interpolate(
                        ori_mask_full.to(dtype=mean_diff_hi.dtype),
                        size=target_shape,
                        mode="nearest",
                    ) > 0.5
                else:
                    mask_all_hi = torch.ones_like(mean_diff_hi, dtype=torch.bool)

                # Export ALL active voxels of the generated glb (not restricted to mask_all).
                # mask_all is only used to decide *which* voxels are eligible to be marked as
                # "conflict"; voxels outside mask_all are always rendered as non-conflict (gray).
                active_all = decoded
                coords_all = torch.argwhere(active_all)[:, [0, 2, 3, 4]].int()
                if coords_all.numel() == 0:
                    print(f"[MV_FUSION_VELCONF][WARN] Decoded voxel volume is empty; skip export for {stem}.")
                    return
                b_all, x_all, y_all, z_all = coords_all.unbind(dim=1)
                voxel_values_all = mean_diff_hi[b_all, 0, x_all, y_all, z_all].detach().float().cpu()
                in_mask_all = mask_all_hi[b_all, 0, x_all, y_all, z_all].detach().cpu()

                # For threshold statistics, only use voxels inside mask_all (that's where the
                # trust-vs-base disagreement is meaningful; outside mask_all both branches share
                # the same source velocity so the diff is ~0 by construction).
                voxel_values_in = voxel_values_all[in_mask_all]
                if voxel_values_in.numel() == 0:
                    print(f"[MV_FUSION_VELCONF][WARN] {stem}: no active voxel inside mask_all; "
                          "will still export whole-glb visualisation with zero conflicts.")

                # Decide threshold based on inside-mask distribution.
                def _auto_thr(vals):
                    m = float(vals.mean().item())
                    s = float(vals.std().item())
                    return m + s

                if isinstance(velocity_conflict_threshold_cfg, str) and velocity_conflict_threshold_cfg.lower() == "auto":
                    threshold = _auto_thr(voxel_values_in) if voxel_values_in.numel() > 0 else float("inf")
                    threshold_mode = "auto(mean+std)"
                else:
                    try:
                        threshold = float(velocity_conflict_threshold_cfg)
                        threshold_mode = "manual"
                    except (TypeError, ValueError):
                        threshold = _auto_thr(voxel_values_in) if voxel_values_in.numel() > 0 else float("inf")
                        threshold_mode = f"auto_fallback(unparsable={velocity_conflict_threshold_cfg!r})"
                if velocity_conflict_min_ratio > 0.0 and voxel_values_in.numel() > 0:
                    # never mark more than (1 - min_ratio) fraction unconflicted below-threshold
                    q = float(torch.quantile(voxel_values_in, 1.0 - velocity_conflict_min_ratio).item())
                    threshold = min(threshold, q)
                    threshold_mode = f"{threshold_mode}+min_ratio({velocity_conflict_min_ratio})"

                # Heatmap over the whole glb: voxels outside mask_all get value 0 so they render
                # as the "cold" end of the colormap; inside mask_all keep the real diff.
                heatmap_values = voxel_values_all.clone()
                heatmap_values[~in_mask_all] = 0.0
                heatmap_path = os.path.join(out_dir, f"{stem}_heatmap.glb")
                v_min_in = float(voxel_values_in.min().item()) if voxel_values_in.numel() > 0 else 0.0
                v_max_in = float(voxel_values_in.max().item()) if voxel_values_in.numel() > 0 else 1.0
                _export_coords_scalar_heatmap_glb(
                    coords_all,
                    heatmap_values,
                    heatmap_path,
                    value_range=(v_min_in, max(v_max_in, v_min_in + 1e-6)),
                )

                # Conflict mask over the whole glb: only voxels inside mask_all *and* above
                # threshold are marked as conflict (red); everything else is gray.
                conflict_flag_all = (voxel_values_all > threshold) & in_mask_all
                mask_path = os.path.join(out_dir, f"{stem}_mask.glb")
                labels_all = torch.where(
                    conflict_flag_all,
                    torch.full_like(x_all, 2, dtype=torch.long).cpu(),
                    torch.full_like(x_all, 0, dtype=torch.long).cpu(),
                )
                _export_coords_category_glb(coords_all, labels_all, mask_path)

                stats = {
                    "threshold": float(threshold),
                    "threshold_mode": threshold_mode,
                    "num_active_voxels_total": int(coords_all.shape[0]),
                    "num_active_voxels_in_mask_all": int(in_mask_all.sum().item()),
                    "num_conflict_voxels": int(conflict_flag_all.sum().item()),
                    "conflict_ratio_over_total": float(conflict_flag_all.float().mean().item()),
                    "conflict_ratio_over_mask_all": (
                        float(conflict_flag_all.sum().item()) / float(in_mask_all.sum().item())
                        if int(in_mask_all.sum().item()) > 0 else 0.0
                    ),
                    "velocity_diff_min_in_mask": v_min_in,
                    "velocity_diff_mean_in_mask": (
                        float(voxel_values_in.mean().item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "velocity_diff_max_in_mask": v_max_in,
                    "velocity_diff_p50_in_mask": (
                        float(torch.quantile(voxel_values_in, 0.50).item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "velocity_diff_p90_in_mask": (
                        float(torch.quantile(voxel_values_in, 0.90).item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "velocity_diff_p95_in_mask": (
                        float(torch.quantile(voxel_values_in, 0.95).item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "velocity_diff_p99_in_mask": (
                        float(torch.quantile(voxel_values_in, 0.99).item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "accumulated_steps": int(accumulated_steps),
                    "pair_a": str(pair_a),
                    "pair_b": str(pair_b),
                    "heatmap_glb": heatmap_path,
                    "conflict_mask_glb": mask_path,
                    "note": (
                        "Exports cover the WHOLE generated glb voxel volume. Heatmap: continuous "
                        "|v_a - v_b| averaged over FlowEdit steps (voxels outside mask_all set to 0). "
                        "Conflict mask: red voxels are voxels INSIDE mask_all whose averaged "
                        "velocity disagreement exceeds threshold; all other voxels (including "
                        "outside mask_all) are gray. " + extra_note
                    ),
                }
                with open(os.path.join(out_dir, f"{stem}_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(
                    f"[MV_FUSION_VELCONF][{pair_a} vs {pair_b}] threshold={threshold:.4g} "
                    f"({threshold_mode}) | conflict {int(conflict_flag_all.sum().item())}/"
                    f"{int(coords_all.shape[0])} voxels "
                    f"({float(conflict_flag_all.float().mean().item()):.2%}) -> {mask_path}"
                )
            except Exception as e:
                ret.mask_metrics.append({"step": -1, "mv_fusion_velocity_conflict_err": f"{stem}: {e}"})
                print(f"[MV_FUSION_VELCONF][WARN] Export failed for {stem}: {e}")

        def _export_mv_fusion_edit_magnitude_conflict():
            """Export the row-3 ablation: high edit-magnitude voxels only.

            This is a diagnostic-only export. It reuses the accumulated fused target
            velocity magnitude and does not affect sample updates, quiescent detection,
            raw_mask shrinkage, or SLAT repainting.
            """
            if not mv_fusion_enabled or not edit_magnitude_conflict_enabled_cfg:
                return
            if edit_magnitude_accum is None or edit_magnitude_count <= 0:
                return
            try:
                out_dir = mv_fusion["output_dir"]
                os.makedirs(out_dir, exist_ok=True)

                mean_mag_16 = (edit_magnitude_accum / float(edit_magnitude_count)).detach().float()
                decoded = decode_voxel_fn(sample.detach()) > 0
                target_shape = decoded.shape[-3:]
                mean_mag_hi = F.interpolate(
                    mean_mag_16,
                    size=target_shape,
                    mode="trilinear",
                    align_corners=False,
                )

                if ori_mask_full is not None:
                    mask_all_hi = F.interpolate(
                        ori_mask_full.to(dtype=mean_mag_hi.dtype),
                        size=target_shape,
                        mode="nearest",
                    ) > 0.5
                else:
                    mask_all_hi = torch.ones_like(mean_mag_hi, dtype=torch.bool)

                coords_all = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
                if coords_all.numel() == 0:
                    print("[MV_FUSION_EDITMAG][WARN] Decoded voxel volume is empty; skip export.")
                    return
                b_all, x_all, y_all, z_all = coords_all.unbind(dim=1)
                voxel_values_all = mean_mag_hi[b_all, 0, x_all, y_all, z_all].detach().float().cpu()
                in_mask_all = mask_all_hi[b_all, 0, x_all, y_all, z_all].detach().cpu()
                voxel_values_in = voxel_values_all[in_mask_all]
                if voxel_values_in.numel() == 0:
                    print("[MV_FUSION_EDITMAG][WARN] No active voxel inside mask_all; "
                          "will still export whole-glb visualisation with zero conflicts.")

                def _auto_thr(vals):
                    m = float(vals.mean().item())
                    s = float(vals.std().item())
                    return m + s

                if isinstance(edit_magnitude_conflict_threshold_cfg, str) and edit_magnitude_conflict_threshold_cfg.lower() == "auto":
                    threshold = _auto_thr(voxel_values_in) if voxel_values_in.numel() > 0 else float("inf")
                    threshold_mode = "auto(mean+std)"
                else:
                    try:
                        threshold = float(edit_magnitude_conflict_threshold_cfg)
                        threshold_mode = "manual"
                    except (TypeError, ValueError):
                        threshold = _auto_thr(voxel_values_in) if voxel_values_in.numel() > 0 else float("inf")
                        threshold_mode = f"auto_fallback(unparsable={edit_magnitude_conflict_threshold_cfg!r})"
                if edit_magnitude_conflict_min_ratio > 0.0 and voxel_values_in.numel() > 0:
                    q = float(torch.quantile(voxel_values_in, 1.0 - edit_magnitude_conflict_min_ratio).item())
                    threshold = min(threshold, q)
                    threshold_mode = f"{threshold_mode}+min_ratio({edit_magnitude_conflict_min_ratio})"

                heatmap_values = voxel_values_all.clone()
                heatmap_values[~in_mask_all] = 0.0
                heatmap_path = os.path.join(out_dir, "edit_magnitude_conflict_heatmap.glb")
                v_min_in = float(voxel_values_in.min().item()) if voxel_values_in.numel() > 0 else 0.0
                v_max_in = float(voxel_values_in.max().item()) if voxel_values_in.numel() > 0 else 1.0
                _export_coords_scalar_heatmap_glb(
                    coords_all,
                    heatmap_values,
                    heatmap_path,
                    value_range=(v_min_in, max(v_max_in, v_min_in + 1e-6)),
                )

                conflict_flag_all = (voxel_values_all > threshold) & in_mask_all
                mask_path = os.path.join(out_dir, "edit_magnitude_conflict_mask.glb")
                labels_all = torch.where(
                    conflict_flag_all,
                    torch.full_like(x_all, 2, dtype=torch.long).cpu(),
                    torch.full_like(x_all, 0, dtype=torch.long).cpu(),
                )
                _export_coords_category_glb(coords_all, labels_all, mask_path)

                stats = {
                    "threshold": float(threshold),
                    "threshold_mode": threshold_mode,
                    "num_active_voxels_total": int(coords_all.shape[0]),
                    "num_active_voxels_in_mask_all": int(in_mask_all.sum().item()),
                    "num_conflict_voxels": int(conflict_flag_all.sum().item()),
                    "conflict_ratio_over_total": float(conflict_flag_all.float().mean().item()),
                    "conflict_ratio_over_mask_all": (
                        float(conflict_flag_all.sum().item()) / float(in_mask_all.sum().item())
                        if int(in_mask_all.sum().item()) > 0 else 0.0
                    ),
                    "edit_magnitude_min_in_mask": v_min_in,
                    "edit_magnitude_mean_in_mask": (
                        float(voxel_values_in.mean().item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "edit_magnitude_max_in_mask": v_max_in,
                    "edit_magnitude_p50_in_mask": (
                        float(torch.quantile(voxel_values_in, 0.50).item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "edit_magnitude_p90_in_mask": (
                        float(torch.quantile(voxel_values_in, 0.90).item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "edit_magnitude_p95_in_mask": (
                        float(torch.quantile(voxel_values_in, 0.95).item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "edit_magnitude_p99_in_mask": (
                        float(torch.quantile(voxel_values_in, 0.99).item()) if voxel_values_in.numel() > 0 else 0.0
                    ),
                    "accumulated_steps": int(edit_magnitude_count),
                    "heatmap_glb": heatmap_path,
                    "conflict_mask_glb": mask_path,
                    "note": (
                        "Row-3 diagnostic-only ablation. Heatmap: continuous averaged fused target "
                        "velocity magnitude |v_tar_fused| over FlowEdit steps (voxels outside mask_all set to 0). "
                        "Conflict mask: red voxels are voxels INSIDE mask_all whose averaged edit magnitude "
                        "exceeds threshold; multi-view velocity disagreement is intentionally ignored."
                    ),
                }
                with open(os.path.join(out_dir, "edit_magnitude_conflict_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(
                    f"[MV_FUSION_EDITMAG] threshold={threshold:.4g} ({threshold_mode}) | "
                    f"conflict {int(conflict_flag_all.sum().item())}/{int(coords_all.shape[0])} voxels "
                    f"({float(conflict_flag_all.float().mean().item()):.2%}) -> {mask_path}"
                )
            except Exception as e:
                ret.mask_metrics.append({"step": -1, "mv_fusion_edit_magnitude_conflict_err": str(e)})
                print(f"[MV_FUSION_EDITMAG][WARN] Export failed: {e}")

        def _compute_and_export_quiescent():
            """Detect the *quiescent* voxel set inside mask_all, export it as GLBs, and
            attach it to ``ret`` so the outer flowedit() can shrink raw_mask for the SLAT
            stage (quiescent voxels are then treated as outside mask_all -> use source).

            A voxel is considered quiescent iff ALL of:
              (1) It lies inside mask_all (raw_mask=1).
              (2) It is decoded active in the current fused sparse-structure sample.
              (3) Its multi-view velocity disagreement is LOW  (below quiescent_disagreement_threshold).
              (4) Its fused edit-magnitude is LOW              (below quiescent_edit_mag_threshold).

            Thresholds can be manually specified as floats, or set to "auto" (default) to
            use inside-mask distributional quantiles. Additionally, ``quiescent_ratio`` (0..1)
            gives an *upper bound* on the fraction of mask_all voxels that can be marked
            quiescent; if 0 the ratio cap is disabled and pure threshold-based selection
            is used. When ``quiescent_ratio > 0`` we shrink both thresholds to the
            corresponding quantile so that at most that fraction is chosen.

            Outputs (all inside ``out_dir``):
              * quiescent_voxels.glb          -- 3-color viz over the WHOLE glb:
                    blue   = quiescent (mask_all & decoded & low disagreement & low mag)
                    red    = conflict  (mask_all & decoded & high disagreement)
                    gray   = neutral   (everything else)
              * quiescent_voxels_only.glb     -- quiescent voxels ONLY, so it can be
                    directly consumed as a "shrunk mask_all" GLB for round-N+1 driver.
              * quiescent_stats.json          -- counts, thresholds, ratios.
            """
            if not mv_fusion_enabled:
                return None
            if not quiescent_enabled_cfg:
                # Hard OFF switch: skip detection, export, and ret.quiescent_mask
                # attachment entirely. The pipeline (trellis_image_to_3d.py) will
                # see no `ret.quiescent_mask`, so raw_mask stays intact and the
                # SLAT stage edits the full mask_all (no quiescent shrinkage).
                print("[MV_FUSION_QUIESCENT] disabled via mv_fusion.quiescent_enabled=False; skip.")
                return None
            if velocity_conflict_accum is None or velocity_conflict_count <= 0:
                return None
            if edit_magnitude_accum is None or edit_magnitude_count <= 0:
                return None
            try:
                out_dir = mv_fusion["output_dir"]
                os.makedirs(out_dir, exist_ok=True)

                # step-averaged 16^3 signals
                disagree_16 = (velocity_conflict_accum / float(velocity_conflict_count)).detach().float()
                edit_mag_16 = (edit_magnitude_accum   / float(edit_magnitude_count  )).detach().float()

                # Decode current fused sparse structure -> active voxel mask (64^3)
                decoded = decode_voxel_fn(sample.detach()) > 0     # [B,1,64,64,64] bool
                target_shape = decoded.shape[-3:]

                # Upsample signals to 64^3
                disagree_hi = F.interpolate(disagree_16, size=target_shape, mode="trilinear", align_corners=False)
                edit_mag_hi = F.interpolate(edit_mag_16, size=target_shape, mode="trilinear", align_corners=False)

                # mask_all (raw_mask). If missing, quiescent detection is not meaningful.
                if ori_mask_full is None:
                    print("[MV_FUSION_QUIESCENT][WARN] ori_mask (raw_mask) not provided; skip quiescent detection.")
                    return None
                mask_all_hi = F.interpolate(
                    ori_mask_full.to(dtype=disagree_hi.dtype),
                    size=target_shape,
                    mode="nearest",
                ) > 0.5

                # Coordinates of ALL active voxels (for whole-glb viz).
                active_all = decoded
                coords_all = torch.argwhere(active_all)[:, [0, 2, 3, 4]].int()
                if coords_all.numel() == 0:
                    print("[MV_FUSION_QUIESCENT][WARN] Decoded voxel volume is empty; skip export.")
                    return None
                b_all, x_all, y_all, z_all = coords_all.unbind(dim=1)

                disagree_vals = disagree_hi[b_all, 0, x_all, y_all, z_all].detach().float().cpu()
                edit_mag_vals = edit_mag_hi[b_all, 0, x_all, y_all, z_all].detach().float().cpu()
                in_mask_all   = mask_all_hi[b_all, 0, x_all, y_all, z_all].detach().cpu()

                # Only voxels inside mask_all are eligible to be quiescent (or conflict);
                # everything else is neutral.
                dis_in = disagree_vals[in_mask_all]
                mag_in = edit_mag_vals[in_mask_all]
                if dis_in.numel() == 0:
                    print("[MV_FUSION_QUIESCENT][WARN] No active voxel inside mask_all; skip export.")
                    return None

                # -------- (a) Disagreement threshold (conflict boundary) --------
                # Reuse the same auto rule as _export_mv_fusion_velocity_conflict
                # so "conflict" here is consistent with the conflict GLB.
                def _auto_thr(vals):
                    return float(vals.mean().item()) + float(vals.std().item())

                if isinstance(velocity_conflict_threshold_cfg, str) and velocity_conflict_threshold_cfg.lower() == "auto":
                    conflict_thr = _auto_thr(dis_in)
                    conflict_thr_mode = "auto(mean+std)"
                else:
                    try:
                        conflict_thr = float(velocity_conflict_threshold_cfg)
                        conflict_thr_mode = "manual"
                    except (TypeError, ValueError):
                        conflict_thr = _auto_thr(dis_in)
                        conflict_thr_mode = f"auto_fallback(unparsable={velocity_conflict_threshold_cfg!r})"
                if velocity_conflict_min_ratio > 0.0:
                    q = float(torch.quantile(dis_in, 1.0 - velocity_conflict_min_ratio).item())
                    conflict_thr = min(conflict_thr, q)
                    conflict_thr_mode = f"{conflict_thr_mode}+min_ratio({velocity_conflict_min_ratio})"

                # -------- (b) Quiescent thresholds (low-disagreement + low-mag) --------
                def _resolve_thr(vals, cfg, quantile_default):
                    if isinstance(cfg, str) and cfg.lower() == "auto":
                        return float(torch.quantile(vals, quantile_default).item()), f"auto(q{quantile_default:.2f})"
                    try:
                        return float(cfg), "manual"
                    except (TypeError, ValueError):
                        return (
                            float(torch.quantile(vals, quantile_default).item()),
                            f"auto_fallback(unparsable={cfg!r})",
                        )

                quiet_dis_thr, quiet_dis_mode = _resolve_thr(dis_in, quiescent_disagreement_thr_cfg, 0.30)
                quiet_mag_thr, quiet_mag_mode = _resolve_thr(mag_in, quiescent_edit_mag_thr_cfg,   0.30)

                # Optional ratio cap: never mark more than quiescent_ratio_cfg of mask_all as quiescent.
                if quiescent_ratio_cfg > 0.0:
                    q_dis = float(torch.quantile(dis_in, quiescent_ratio_cfg).item())
                    q_mag = float(torch.quantile(mag_in, quiescent_ratio_cfg).item())
                    quiet_dis_thr = min(quiet_dis_thr, q_dis)
                    quiet_mag_thr = min(quiet_mag_thr, q_mag)
                    quiet_dis_mode = f"{quiet_dis_mode}+ratio({quiescent_ratio_cfg})"
                    quiet_mag_mode = f"{quiet_mag_mode}+ratio({quiescent_ratio_cfg})"

                # -------- (c) Classify every active voxel --------
                is_conflict = (disagree_vals > conflict_thr) & in_mask_all
                is_quiescent = (
                    (disagree_vals <= quiet_dis_thr)
                    & (edit_mag_vals <= quiet_mag_thr)
                    & in_mask_all
                    & ~is_conflict  # never overlap with conflict; conflict wins if both flagged
                )

                # -------- (d) Whole-glb 3-color visualization --------
                # 0=neutral(gray), 1=quiescent(blue), 2=conflict(red)
                labels_viz = torch.zeros_like(x_all, dtype=torch.long).cpu()
                labels_viz[is_quiescent] = 1
                labels_viz[is_conflict]  = 2
                viz_path = os.path.join(out_dir, "quiescent_voxels.glb")
                _export_coords_category_glb(coords_all, labels_viz, viz_path)

                # -------- (e) Quiescent-only glb (for downstream driver consumption) --------
                q_only_path = os.path.join(out_dir, "quiescent_voxels_only.glb")
                if bool(is_quiescent.any()):
                    coords_q = coords_all[is_quiescent]
                    labels_q = torch.full((coords_q.shape[0],), 2, dtype=torch.long)  # red
                    _export_coords_category_glb(coords_q, labels_q, q_only_path)
                else:
                    # Emit an empty-ish placeholder so downstream code can detect "no quiescent".
                    print("[MV_FUSION_QUIESCENT] No voxel qualified as quiescent; skipping quiescent_voxels_only.glb.")

                # -------- (f) Build 64^3 boolean quiescent tensor and stash in ret --------
                quiescent_mask_64 = torch.zeros_like(mask_all_hi, dtype=torch.bool)
                if bool(is_quiescent.any()):
                    q_b = b_all[is_quiescent].long()
                    q_x = x_all[is_quiescent].long()
                    q_y = y_all[is_quiescent].long()
                    q_z = z_all[is_quiescent].long()
                    quiescent_mask_64[q_b, 0, q_x, q_y, q_z] = True
                # Attach the lock-back set to ret so trellis_image_to_3d.py can shrink raw_mask
                # for the slat stage (the GLBs / stats above always describe the three states).
                if state_mode_cfg in ("three_state", "three_state_srcocc"):
                    lockback_mask_64 = quiescent_mask_64
                elif state_mode_cfg == "no_lockback":
                    lockback_mask_64 = torch.zeros_like(quiescent_mask_64)
                else:
                    lockback_mask_64 = mask_all_hi & (disagree_hi <= conflict_thr)
                ret.quiescent_mask = lockback_mask_64.detach()

                # -------- (g) Stats --------
                num_in_mask = int(in_mask_all.sum().item())
                stats = {
                    "conflict_threshold":              float(conflict_thr),
                    "conflict_threshold_mode":         conflict_thr_mode,
                    "quiescent_disagreement_threshold": float(quiet_dis_thr),
                    "quiescent_disagreement_mode":     quiet_dis_mode,
                    "quiescent_edit_mag_threshold":    float(quiet_mag_thr),
                    "quiescent_edit_mag_mode":         quiet_mag_mode,
                    "quiescent_ratio_cap":             float(quiescent_ratio_cfg),
                    "num_active_voxels_total":         int(coords_all.shape[0]),
                    "num_active_voxels_in_mask_all":   num_in_mask,
                    "num_conflict_voxels":             int(is_conflict.sum().item()),
                    "num_quiescent_voxels":            int(is_quiescent.sum().item()),
                    "state_mode":                      state_mode_cfg,
                    "num_lockback_cells":              int(lockback_mask_64.sum().item()),
                    "num_lockback_active_voxels":      int(lockback_mask_64[b_all.long(), 0, x_all.long(), y_all.long(), z_all.long()].sum().item()),
                    "num_neutral_in_mask_all":         int((in_mask_all.sum().item()
                                                            - is_conflict.sum().item()
                                                            - is_quiescent.sum().item())),
                    "conflict_ratio_over_mask_all":    (float(is_conflict.sum().item())  / num_in_mask) if num_in_mask > 0 else 0.0,
                    "quiescent_ratio_over_mask_all":   (float(is_quiescent.sum().item()) / num_in_mask) if num_in_mask > 0 else 0.0,
                    "disagreement_stats_in_mask": {
                        "min":  float(dis_in.min().item()),
                        "mean": float(dis_in.mean().item()),
                        "p50":  float(torch.quantile(dis_in, 0.5).item()),
                        "p90":  float(torch.quantile(dis_in, 0.9).item()),
                        "max":  float(dis_in.max().item()),
                    },
                    "edit_mag_stats_in_mask": {
                        "min":  float(mag_in.min().item()),
                        "mean": float(mag_in.mean().item()),
                        "p50":  float(torch.quantile(mag_in, 0.5).item()),
                        "p90":  float(torch.quantile(mag_in, 0.9).item()),
                        "max":  float(mag_in.max().item()),
                    },
                    "accumulated_steps_disagreement":  int(velocity_conflict_count),
                    "accumulated_steps_edit_mag":      int(edit_magnitude_count),
                    "voxels_glb":                      viz_path,
                    "quiescent_only_glb":              q_only_path,
                    "note": (
                        "quiescent = mask_all AND decoded AND low disagreement AND low edit-magnitude AND NOT conflict. "
                        "In quiescent_voxels.glb the palette is blue=quiescent, red=conflict, gray=neutral. "
                        "The outer flowedit() will remove quiescent voxels from raw_mask before SLAT repaint, so "
                        "these voxels are effectively frozen to the source SLAT (mesh comes from source)."
                    ),
                }
                with open(os.path.join(out_dir, "quiescent_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(
                    f"[MV_FUSION_QUIESCENT] quiescent={int(is_quiescent.sum().item())}/"
                    f"{num_in_mask} of mask_all "
                    f"({(float(is_quiescent.sum().item())/num_in_mask if num_in_mask>0 else 0.0):.2%}) "
                    f"| conflict={int(is_conflict.sum().item())} | "
                    f"dis_thr={quiet_dis_thr:.4g} ({quiet_dis_mode}) | "
                    f"mag_thr={quiet_mag_thr:.4g} ({quiet_mag_mode}) -> {viz_path}"
                )
                print(f"[MV_FUSION_QUIESCENT] state_mode={state_mode_cfg}: "
                      f"{int(lockback_mask_64.sum().item())} cells locked back to source")
                return quiescent_mask_64
            except Exception as e:
                ret.mask_metrics.append({"step": -1, "mv_fusion_quiescent_err": str(e)})
                print(f"[MV_FUSION_QUIESCENT][WARN] Detection failed: {e}")
                return None

        def _export_mv_fusion_trust_region_voxels(step_number, t_val, fused_state):
            if (not mv_fusion_enabled) or int(mv_fusion.get("trust_region_step", 10)) != step_number:
                return
            try:
                decoded = decode_voxel_fn(fused_state.detach()) > 0
                coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
                if coords.numel() == 0:
                    return
                region_64 = F.interpolate(
                    trust_region_mask.float(),
                    size=decoded.shape[-3:],
                    mode="nearest",
                ).bool()
                b, x, y, z = coords.unbind(dim=1)
                inside = region_64[b, 0, x, y, z]
                labels = torch.where(
                    inside,
                    torch.full_like(x, 2, dtype=torch.long),
                    torch.full_like(x, 3, dtype=torch.long),
                ).detach().cpu()
                out_dir = mv_fusion["output_dir"]
                stem = f"step_{step_number:03d}_{mv_fusion['trust_view']}_trust_region_generated_voxels"
                glb_path = os.path.join(out_dir, f"{stem}.glb")
                _export_coords_category_glb(coords, labels, glb_path)
                stats = {
                    "step": int(step_number),
                    "t": float(t_val),
                    "trust_view": mv_fusion["trust_view"],
                    "base_view": mv_fusion["base_view"],
                    "conf_box_glb": mv_fusion.get("conf_box_glb", None),
                    "conf_box_bounds": mv_fusion.get("conf_box_bounds", None),
                    "alpha_stats": mv_fusion.get("alpha_stats", None),
                    "num_generated_voxels": int(coords.shape[0]),
                    "num_generated_inside_trust_region": int(inside.sum().item()),
                    "num_generated_outside_trust_region": int((~inside).sum().item()),
                    "glb": glb_path,
                    "note": "Red voxels are generated voxels inside the trusted-view confidence region; gray voxels are generated voxels outside it.",
                }
                with open(os.path.join(out_dir, f"{stem}_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(f"[MV_FUSION] Exported trust-region voxels {glb_path}")
            except Exception as e:
                ret.mask_metrics.append({"step": int(step_number), "mv_fusion_trust_region_err": str(e)})
                print(f"[MV_FUSION][WARN] Failed trust-region voxel export at step {step_number}: {e}")

        def _export_mv_fusion_blend_heatmap(step_number, t_val, fused_state):
            if (not mv_fusion_enabled) or (step_number not in mv_fusion_steps):
                return
            try:
                decoded = decode_voxel_fn(fused_state.detach()) > 0
                coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
                if coords.numel() == 0:
                    return
                strength_64 = F.interpolate(
                    fusion_strength_mask.detach().float(),
                    size=decoded.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                alpha_64 = F.interpolate(
                    alpha_mask.detach().float(),
                    size=decoded.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                b, x, y, z = coords.unbind(dim=1)
                voxel_values = strength_64[b, 0, x, y, z].detach().float().cpu()
                voxel_alpha = alpha_64[b, 0, x, y, z].detach().float().cpu()
                out_dir = mv_fusion["output_dir"]
                stem = f"step_{step_number:03d}_{mv_fusion['trust_view']}_vs_{mv_fusion['base_view']}_blend_strength"
                glb_path = os.path.join(out_dir, f"{stem}_heatmap.glb")
                _export_coords_scalar_heatmap_glb(coords, voxel_values, glb_path, value_range=(0.0, 1.0))
                stats = {
                    "step": int(step_number),
                    "t": float(t_val),
                    "trust_view": mv_fusion["trust_view"],
                    "base_view": mv_fusion["base_view"],
                    "alpha_stats": mv_fusion.get("alpha_stats", None),
                    "num_active_voxels": int(coords.shape[0]),
                    "blend_strength_min": float(voxel_values.min().item()),
                    "blend_strength_mean": float(voxel_values.mean().item()),
                    "blend_strength_max": float(voxel_values.max().item()),
                    "blend_strength_p50": float(torch.quantile(voxel_values, 0.50).item()),
                    "blend_strength_p90": float(torch.quantile(voxel_values, 0.90).item()),
                    "alpha_min_on_active": float(voxel_alpha.min().item()),
                    "alpha_mean_on_active": float(voxel_alpha.mean().item()),
                    "alpha_max_on_active": float(voxel_alpha.max().item()),
                    "glb": glb_path,
                    "note": "Color uses a fixed 0-1 scale. Pale voxels are near pure single-view velocity; dark red voxels have stronger two-view blending, peaking near alpha=0.5 at confidence-region edges/connection bands.",
                }
                with open(os.path.join(out_dir, f"{stem}_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(f"[MV_FUSION] Exported blend-strength heatmap {glb_path}")
            except Exception as e:
                ret.mask_metrics.append({"step": int(step_number), "mv_fusion_blend_heatmap_err": str(e)})
                print(f"[MV_FUSION][WARN] Failed blend-strength heatmap at step {step_number}: {e}")

        def _export_mv_fusion_voxel_diff(step_number, t_val, fused_state, base_state):
            if (not mv_fusion_enabled) or (step_number not in mv_fusion_steps):
                return
            try:
                fused_decoded = decode_voxel_fn(fused_state.detach()) > 0
                base_decoded = decode_voxel_fn(base_state.detach()) > 0
                both = fused_decoded & base_decoded
                added = fused_decoded & (~base_decoded)
                missing = base_decoded & (~fused_decoded)
                union = fused_decoded | base_decoded
                coords = torch.argwhere(union)[:, [0, 2, 3, 4]].int()
                if coords.numel() == 0:
                    return
                labels_grid = torch.zeros_like(union, dtype=torch.long)
                labels_grid[both] = 3
                labels_grid[missing] = 1
                labels_grid[added] = 2
                b, x, y, z = coords.unbind(dim=1)
                labels = labels_grid[b, 0, x, y, z].detach().cpu()
                out_dir = mv_fusion["output_dir"]
                stem = f"step_{step_number:03d}_{mv_fusion['trust_view']}_trust_vs_{mv_fusion['base_view']}_base"
                glb_path = os.path.join(out_dir, f"{stem}_voxel_diff.glb")
                _export_coords_category_glb(coords, labels, glb_path)
                stats = {
                    "step": int(step_number),
                    "t": float(t_val),
                    "trust_view": mv_fusion["trust_view"],
                    "base_view": mv_fusion["base_view"],
                    "conf_box_glb": mv_fusion.get("conf_box_glb", None),
                    "conf_box_bounds": mv_fusion.get("conf_box_bounds", None),
                    "alpha_stats": mv_fusion.get("alpha_stats", None),
                    "num_union": int(coords.shape[0]),
                    "num_shared": int(both.sum().item()),
                    "num_missing_from_fused": int(missing.sum().item()),
                    "num_added_by_fused": int(added.sum().item()),
                    "glb": glb_path,
                    "note": "Blue voxels exist in the original base-view trajectory but are missing in the fused trajectory; red voxels are newly added by fusion; gray voxels are shared.",
                }
                with open(os.path.join(out_dir, f"{stem}_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(f"[MV_FUSION] Exported {glb_path}")
            except Exception as e:
                ret.mask_metrics.append({"step": int(step_number), "mv_fusion_diff_err": str(e)})
                print(f"[MV_FUSION][WARN] Failed voxel diff at step {step_number}: {e}")

        mv_debug_enabled = (
            mv_debug is not None
            and decode_voxel_fn is not None
            and len(mv_debug.get("view_names", [])) == 2
        )
        mv_debug_steps = set(int(s) for s in mv_debug.get("steps", [1, 3, 5, 10, 20])) if mv_debug_enabled else set()
        if mv_debug_enabled:
            os.makedirs(mv_debug["output_dir"], exist_ok=True)

        def _export_mv_velocity_diff(step_number, t_val, zt_src_ref, zt_tar_ref, state_ref):
            if (not mv_debug_enabled) or (step_number not in mv_debug_steps):
                return
            try:
                source_cond = mv_debug["source_cond"]
                target_cond = mv_debug["target_cond"]
                view_names = mv_debug["view_names"]
                deltas = []
                for view_i in range(2):
                    src_cond_i = _cond_at(source_cond, view_i)
                    tar_cond_i = _cond_at(target_cond, view_i)
                    v_src_i = _predict_v_with_cond(src_cond_i, zt_src_ref, t_val, cfg_src_strength)
                    v_tar_i = _predict_v_with_cond(tar_cond_i, zt_tar_ref, t_val, cfg_tar_strength)
                    deltas.append((v_tar_i.float() - v_src_i.float()).detach())

                diff_map = reduce_vector_norm(deltas[0] - deltas[1]).detach().float()
                if combined_soft is not None:
                    diff_map = diff_map * combined_soft.detach().float()

                decoded = decode_voxel_fn(state_ref.detach()) > 0
                coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
                if coords.numel() == 0:
                    return

                diff_map_64 = F.interpolate(
                    diff_map,
                    size=decoded.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                b, x, y, z = coords.unbind(dim=1)
                voxel_values = diff_map_64[b, 0, x, y, z].detach().float().cpu()
                out_dir = mv_debug["output_dir"]
                stem = f"step_{step_number:03d}_{view_names[0]}_vs_{view_names[1]}"
                glb_path = os.path.join(out_dir, f"{stem}_delta_v_diff_heatmap.glb")
                _export_coords_scalar_heatmap_glb(coords, voxel_values, glb_path)
                stats = {
                    "step": int(step_number),
                    "t": float(t_val),
                    "views": view_names,
                    "num_voxels": int(coords.shape[0]),
                    "diff_min": float(voxel_values.min().item()),
                    "diff_mean": float(voxel_values.mean().item()),
                    "diff_max": float(voxel_values.max().item()),
                    "diff_p50": float(torch.quantile(voxel_values, 0.50).item()),
                    "diff_p90": float(torch.quantile(voxel_values, 0.90).item()),
                    "diff_p95": float(torch.quantile(voxel_values, 0.95).item()),
                    "glb": glb_path,
                    "note": "Color is normalized per exported step: pale red/white means smaller delta-velocity disagreement; dark red means larger disagreement within this GLB.",
                }
                with open(os.path.join(out_dir, f"{stem}_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(f"[MV_VEL_DIFF] Exported {glb_path}")
            except Exception as e:
                ret.mask_metrics.append({"step": int(step_number), "mv_velocity_debug_err": str(e)})
                print(f"[MV_VEL_DIFF][WARN] Failed at step {step_number}: {e}")

        iterator = enumerate(t_pairs)
        if verbose:
            try:
                iterator = tqdm(iterator, total=len(t_pairs), desc="Sampling (FlowEdit)")
            except Exception:
                pass

        base_ref_sample = sample.clone() if mv_fusion_enabled else None

        for step_i, (t, t_prev) in iterator:
            tail_refine = (steps - step_i) <= self._FE_N_MIN
            dt = t - t_prev

            if not tail_refine:
                V = torch.zeros_like(sample)
                V_base_ref = torch.zeros_like(base_ref_sample) if mv_fusion_enabled else None
                last_zt_src = last_zt_tar = None
                last_v_src  = last_v_tar  = None
                last_base_zt_src = last_base_zt_tar = None
                last_base_v_src = last_base_v_tar = None
                for _ in range(self._FE_N_AVG):
                    eps = torch.randn_like(x_src0)
                    zt_src = apply_mask_blend(forward_noise(x_src0, t, eps), x_src0, combined_soft)
                    zt_tar = sample + (zt_src - x_src0)

                    if mv_fusion_enabled:
                        source_cond = mv_fusion["source_cond"]
                        target_cond = mv_fusion["target_cond"]

                        if fusion_mode == "view_average":
                            # Ablation mode: ignore confidence regions and average
                            # velocities from every provided view equally.
                            v_src_all_list = []
                            v_tar_all_list = []
                            delta_all_list = []
                            for view_i in all_view_indices:
                                src_i = _cond_at(source_cond, view_i)
                                tar_i = _cond_at(target_cond, view_i)
                                v_src_i = _predict_v_with_cond(src_i, zt_src, t, cfg_src_strength)
                                v_tar_i = _predict_v_with_cond(tar_i, zt_tar, t, cfg_tar_strength)
                                v_src_all_list.append(v_src_i)
                                v_tar_all_list.append(v_tar_i)
                                delta_all_list.append(v_tar_i - flowedit_lambda * v_src_i)

                            _accumulate_velocity_conflict_multi(
                                v_tar_all_list,
                                view_names=list(view_names_fuse),
                            )

                            inv_n = 1.0 / float(len(delta_all_list))
                            fused_delta = torch.zeros_like(delta_all_list[0])
                            v_src_mixed = torch.zeros_like(v_src_all_list[0])
                            v_tar_mixed = torch.zeros_like(v_tar_all_list[0])
                            for delta_i, v_src_i, v_tar_i in zip(delta_all_list, v_src_all_list, v_tar_all_list):
                                fused_delta = fused_delta + delta_i * inv_n
                                v_src_mixed = v_src_mixed + v_src_i * inv_n
                                v_tar_mixed = v_tar_mixed + v_tar_i * inv_n

                            v_src_base = v_src_all_list[base_i]
                            tar_base = _cond_at(target_cond, base_i)
                        else:
                            # ---- base view (single) ----
                            src_base = _cond_at(source_cond, base_i)
                            tar_base = _cond_at(target_cond, base_i)
                            v_src_base = _predict_v_with_cond(src_base, zt_src, t, cfg_src_strength)
                            v_tar_base = _predict_v_with_cond(tar_base, zt_tar, t, cfg_tar_strength)
                            delta_base = v_tar_base - flowedit_lambda * v_src_base

                            # ---- per trust view ----
                            v_tar_trust_list = []
                            v_src_trust_list = []
                            delta_trust_list = []
                            for tv_i in trust_indices:
                                src_tv = _cond_at(source_cond, tv_i)
                                tar_tv = _cond_at(target_cond, tv_i)
                                v_src_tv = _predict_v_with_cond(src_tv, zt_src, t, cfg_src_strength)
                                v_tar_tv = _predict_v_with_cond(tar_tv, zt_tar, t, cfg_tar_strength)
                                v_src_trust_list.append(v_src_tv)
                                v_tar_trust_list.append(v_tar_tv)
                                delta_trust_list.append(v_tar_tv - flowedit_lambda * v_src_tv)

                            # Multi-view velocity conflict accumulator: mean pairwise disagreement
                            # among all trust views + base view (extends the previous 2-view
                            # |v_tar_trust - v_tar_base| definition to N views). We also pass
                            # the view names so a per-pair heatmap can be exported at the end.
                            _accumulate_velocity_conflict_multi(
                                v_tar_trust_list + [v_tar_base],
                                view_names=(list(trust_views_list) + [mv_fusion.get("base_view", "base")]),
                            )

                            # Weighted sum: sum_k alpha_k * delta_trust_k + alpha_base * delta_base
                            fused_delta = alpha_base_mask.to(dtype=delta_base.dtype) * delta_base
                            v_src_mixed = alpha_base_mask.to(dtype=v_src_base.dtype) * v_src_base
                            v_tar_mixed = alpha_base_mask.to(dtype=v_tar_base.dtype) * v_tar_base
                            for k in range(len(trust_indices)):
                                ak = alpha_masks_per_trust[k].to(dtype=delta_base.dtype)
                                fused_delta = fused_delta + ak * delta_trust_list[k]
                                v_src_mixed = v_src_mixed + ak * v_src_trust_list[k]
                                v_tar_mixed = v_tar_mixed + ak * v_tar_trust_list[k]

                        # Accumulate fused-target-velocity magnitude for the quiescent detector.
                        _accumulate_edit_magnitude(v_tar_mixed)

                        V = V + fused_delta / float(self._FE_N_AVG)
                        last_v_src = v_src_mixed
                        last_v_tar = v_tar_mixed
                        last_zt_src, last_zt_tar = zt_src, zt_tar

                        # base-only reference trajectory (unchanged from 2-view version)
                        zt_tar_base_ref = base_ref_sample + (zt_src - x_src0)
                        v_tar_base_ref = _predict_v_with_cond(tar_base, zt_tar_base_ref, t, cfg_tar_strength)
                        V_base_ref = V_base_ref + (v_tar_base_ref - flowedit_lambda * v_src_base) / float(self._FE_N_AVG)
                        last_base_zt_src, last_base_zt_tar = zt_src, zt_tar_base_ref
                        last_base_v_src, last_base_v_tar = v_src_base, v_tar_base_ref
                    else:
                        kwargs["cfg_strength"] = cfg_src_strength
                        _, _, v_src = self._get_model_prediction(model, zt_src, t, cond_src, **kwargs)
                        kwargs["cfg_strength"] = cfg_tar_strength
                        _, _, v_tar = self._get_model_prediction(model, zt_tar, t, cond_tar, **kwargs)

                        V = V + (v_tar - flowedit_lambda * v_src) / float(self._FE_N_AVG)
                        last_zt_src, last_zt_tar, last_v_src, last_v_tar = zt_src, zt_tar, v_src, v_tar

                V = _mul_mask(V)
                x = sample - dt * V
                if mv_fusion_enabled:
                    V_base_ref = _mul_mask(V_base_ref)
                    x_base_ref = base_ref_sample - dt * V_base_ref

                if step_i >= self._FE_GUID_START:
                    tv = float(t)
                    x0_tgt = last_zt_tar - tv * last_v_tar
                    x0_src = last_zt_src - tv * last_v_src
                    xi_traj = x0_tgt - x0_src
                    x = _guide(x, tv, dt, step_i, "mid",
                               v_norm_ref=V.detach().norm() + 1e-8,
                               xi_traj=xi_traj.detach())
                    if mv_fusion_enabled:
                        x0_tgt_base = last_base_zt_tar - tv * last_base_v_tar
                        x0_src_base = last_base_zt_src - tv * last_base_v_src
                        xi_traj_base = x0_tgt_base - x0_src_base
                        x_base_ref = _guide(x_base_ref, tv, dt, step_i, "mid_base_ref",
                                            v_norm_ref=V_base_ref.detach().norm() + 1e-8,
                                            xi_traj=xi_traj_base.detach())

                sample = _lock_back(x, float(t))
                if mv_fusion_enabled:
                    base_ref_sample = _lock_back(x_base_ref, float(t))
                ret.pred_x_t.append(sample)
                ret.pred_x_0.append(None)
                _export_mv_velocity_diff(step_i + 1, float(t), last_zt_src, last_zt_tar, sample)
                if mv_fusion_enabled:
                    _export_mv_fusion_voxel_diff(step_i + 1, float(t), sample, base_ref_sample)
                    _export_mv_fusion_blend_heatmap(step_i + 1, float(t), sample)
                    _export_mv_fusion_trust_region_voxels(step_i + 1, float(t), sample)

            else:
                if (steps - step_i) == self._FE_N_MIN:
                    eps = torch.randn_like(x_src0)
                    zt_src = apply_mask_blend(forward_noise(x_src0, t, eps), x_src0, combined_soft)
                    xt_tar = sample + (zt_src - x_src0)
                    if mv_fusion_enabled:
                        xt_tar_base_ref = base_ref_sample + (zt_src - x_src0)
                else:
                    zt_src = apply_mask_blend(forward_noise(x_src0, t, torch.zeros_like(x_src0)), x_src0, combined_soft)
                    xt_tar = sample
                    if mv_fusion_enabled:
                        xt_tar_base_ref = base_ref_sample

                if mv_fusion_enabled:
                    target_cond = mv_fusion["target_cond"]
                    tar_base = _cond_at(target_cond, base_i)

                    if fusion_mode == "view_average":
                        v_tar_all_list = []
                        for view_i in all_view_indices:
                            tar_i = _cond_at(target_cond, view_i)
                            v_tar_i = _predict_v_with_cond(tar_i, xt_tar, t, cfg_tar_strength)
                            v_tar_all_list.append(v_tar_i)

                        _accumulate_velocity_conflict_multi(
                            v_tar_all_list,
                            view_names=list(view_names_fuse),
                        )

                        v_tar = torch.zeros_like(v_tar_all_list[0])
                        inv_n = 1.0 / float(len(v_tar_all_list))
                        for v_tar_i in v_tar_all_list:
                            v_tar = v_tar + v_tar_i * inv_n
                    else:
                        v_tar_base = _predict_v_with_cond(tar_base, xt_tar, t, cfg_tar_strength)

                        v_tar_trust_list = []
                        for tv_i in trust_indices:
                            tar_tv = _cond_at(target_cond, tv_i)
                            v_tv = _predict_v_with_cond(tar_tv, xt_tar, t, cfg_tar_strength)
                            v_tar_trust_list.append(v_tv)

                        _accumulate_velocity_conflict_multi(
                            v_tar_trust_list + [v_tar_base],
                            view_names=(list(trust_views_list) + [mv_fusion.get("base_view", "base")]),
                        )

                        v_tar = alpha_base_mask.to(dtype=v_tar_base.dtype) * v_tar_base
                        for k in range(len(trust_indices)):
                            ak = alpha_masks_per_trust[k].to(dtype=v_tar_base.dtype)
                            v_tar = v_tar + ak * v_tar_trust_list[k]
                    # Accumulate fused-target-velocity magnitude for the quiescent detector.
                    _accumulate_edit_magnitude(v_tar)
                    pred_x0 = xt_tar - float(t) * v_tar

                    v_tar_base_ref = _predict_v_with_cond(tar_base, xt_tar_base_ref, t, cfg_tar_strength)
                    v_tar_base_ref = _mul_mask(v_tar_base_ref)
                    x_base_ref = xt_tar_base_ref - dt * v_tar_base_ref
                else:
                    kwargs["cfg_strength"] = cfg_tar_strength
                    pred_x0, _, v_tar = self._get_model_prediction(model, xt_tar, t, cond_tar, **kwargs)

                v_tar = _mul_mask(v_tar)
                x = xt_tar - dt * v_tar

                x = _guide(x, float(t), dt, step_i, "tail",
                           v_norm_ref=v_tar.detach().norm() + 1e-8,
                           xi_traj=None)
                if mv_fusion_enabled:
                    x_base_ref = _guide(x_base_ref, float(t), dt, step_i, "tail_base_ref",
                                        v_norm_ref=v_tar_base_ref.detach().norm() + 1e-8,
                                        xi_traj=None)

                sample = _lock_back(x, float(t))
                if mv_fusion_enabled:
                    base_ref_sample = _lock_back(x_base_ref, float(t))
                ret.pred_x_t.append(sample)
                ret.pred_x_0.append(pred_x0)
                _export_mv_velocity_diff(step_i + 1, float(t), zt_src, xt_tar, sample)
                if mv_fusion_enabled:
                    _export_mv_fusion_voxel_diff(step_i + 1, float(t), sample, base_ref_sample)
                    _export_mv_fusion_blend_heatmap(step_i + 1, float(t), sample)
                    _export_mv_fusion_trust_region_voxels(step_i + 1, float(t), sample)

        if mv_fusion_enabled:
            _export_mv_fusion_velocity_conflict()
            _export_mv_fusion_edit_magnitude_conflict()
            _compute_and_export_quiescent()

        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_ori(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 28,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            kwargs["t_sign"] = t
            kwargs["t_1"] = t
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_repaint(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 28,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
      
        sample = noise
        x_src = kwargs.get("x_src", None)
        mask = kwargs.get("mask", None)
        stage = kwargs.get("stage", "sparse")
        repaint_beta       = max(0.0, min(1.0, float(self._RP_BETA)))
        repaint_anchor_eps = bool(self._RP_ANCHOR_EPS)
        repaint_inv_steps  = max(0, min(int(self._RP_INV_STEPS), steps))
        cond_src           = kwargs.get("ori_cond", None)
        if repaint_inv_steps > 0 and cond_src is None:
            repaint_inv_steps = 0

        mv_fusion = kwargs.get("mv_fusion", None)
        mv_fusion_slat_enabled = (
            stage == "slat"
            and mv_fusion is not None
            and hasattr(sample, "feats")
            and hasattr(sample, "coords")
            and len(mv_fusion.get("view_names", [])) >= 2
            and "target_cond" in mv_fusion
        )
        if mv_fusion_slat_enabled:
            view_names_fuse = mv_fusion["view_names"]
            fusion_mode = str(mv_fusion.get("fusion_mode", "trust_region")).strip().lower()
            if fusion_mode not in {"trust_region", "view_average"}:
                raise ValueError(f"mv_fusion.fusion_mode must be 'trust_region' or 'view_average', got {fusion_mode!r}")
            trust_views_list = list(mv_fusion.get("trust_views", []) or [mv_fusion.get("trust_view", view_names_fuse[0])])
            trust_indices = [view_names_fuse.index(v) for v in trust_views_list]
            base_i = view_names_fuse.index(mv_fusion.get("base_view", view_names_fuse[-1]))
            trust_i = trust_indices[0]  # legacy alias

            # Per-trust-view confidence bounds (in [-0.5,0.5]^3), one entry per trust view.
            conf_box_glbs = list(mv_fusion.get("conf_box_glbs", []) or [mv_fusion.get("conf_box_glb", None)])
            conf_box_bounds_list_raw = mv_fusion.get("conf_box_bounds_list", None)
            if conf_box_bounds_list_raw is None:
                conf_box_bounds_list_raw = [mv_fusion.get("conf_box_bounds", None)]
            trust_bounds_list = []
            for gpath, provided_bounds in zip(conf_box_glbs, conf_box_bounds_list_raw):
                if provided_bounds is not None:
                    tv_bounds = np.asarray(provided_bounds, dtype=np.float32)
                elif gpath:
                    tv_bounds = _load_scene_bounds(gpath)
                else:
                    tv_bounds = None
                trust_bounds_list.append(tv_bounds)
            # Persist per-trust bounds back to mv_fusion for downstream logging.
            mv_fusion["conf_box_bounds_list"] = [
                (b.tolist() if b is not None else None) for b in trust_bounds_list
            ]
            conf_box_bounds = trust_bounds_list[0] if trust_bounds_list else None
            if conf_box_bounds is not None:
                mv_fusion["conf_box_bounds"] = conf_box_bounds.tolist()
            # Geometry-derived support: chi_v lives on the same 64^3 grid as SLAT coords.
            support_chi64_t = None
            if mv_fusion.get("support_chi64", None) is not None:
                support_chi64_t = [
                    torch.as_tensor(mv_fusion["support_chi64"][tv], dtype=torch.float32, device=sample.feats.device)
                    for tv in trust_views_list
                ]
            print(
                f"[MV_FUSION][SLAT] trust_views={trust_views_list}, base={view_names_fuse[base_i]}, "
                f"n_views={len(view_names_fuse)}, conf_box_glbs={conf_box_glbs}, "
                f"view_support={support_chi64_t is not None}, "
                f"soft_blend={bool(mv_fusion.get('soft_blend', False))}, "
                f"alpha_stats={mv_fusion.get('alpha_stats', None)}"
            )
        else:
            view_names_fuse = None
            trust_i = base_i = None
            trust_indices = None
            trust_views_list = None
            trust_bounds_list = None
            conf_box_bounds = None
            support_chi64_t = None

        def _cond_at(cond_dict, view_i):
            return {
                key: (value[view_i:view_i + 1] if torch.is_tensor(value) and value.shape[0] > view_i else value)
                for key, value in cond_dict.items()
            }

        def _clean_repaint_model_kwargs():
            drop = {
                "mode", "x_src", "mask", "stage", "ori_cond",
                "repaint_beta", "repaint_anchor_eps", "repaint_inv_steps",
                "latent", "latent_slat", "feature_path", "edit_mask",
                "cfg_src_strength", "cfg_tar_strength", "use_guidance", "mask_feather_outward",
                "enable_step_viz", "step_viz_every", "azimuth_deg",
                "debug_guidance_viz", "debug_viz_dir", "decode_voxel_fn",
                "use_mask", "ori_mask", "mv_velocity_debug", "mv_fusion",
            }
            return {k: v for k, v in kwargs.items() if k not in drop}

        def _predict_repaint_v_with_cond(cond_dict, x_t, t_val):
            clean_kwargs = _clean_repaint_model_kwargs()
            if "neg_cond" in cond_dict:
                clean_kwargs["neg_cond"] = cond_dict["neg_cond"]
            if "cfg_strength" in kwargs:
                clean_kwargs["cfg_strength"] = kwargs["cfg_strength"]
            if "cfg_interval" in kwargs:
                clean_kwargs["cfg_interval"] = kwargs["cfg_interval"]
            return self._get_model_prediction(model, x_t, t_val, cond_dict.get("cond"), **clean_kwargs)

        def _sparse_mv_alpha(sparse_sample, tv_bounds_np=None):
            """Return per-sparse-voxel alpha (weight assigned to a single trust view).

            When `tv_bounds_np` is None, falls back to the legacy primary trust view
            (`conf_box_bounds`) or the image-space bbox fallback.
            """
            coords = sparse_sample.coords.to(sparse_sample.feats.device).long()
            alpha_inside = max(0.0, min(1.0, float(mv_fusion.get("alpha_inside", 1.0))))
            alpha_outside = max(0.0, min(1.0, float(mv_fusion.get("alpha_outside", 0.0))))
            use_bounds = tv_bounds_np if tv_bounds_np is not None else conf_box_bounds
            if use_bounds is not None:
                bounds = torch.as_tensor(use_bounds, device=sparse_sample.feats.device, dtype=torch.float32)
                box_min = torch.clamp(bounds[0], -0.5, 0.5)
                box_max = torch.clamp(bounds[1], -0.5, 0.5)
                if mask is not None and torch.is_tensor(mask) and mask.ndim >= 5:
                    grid_size = torch.tensor(mask.shape[-3:], device=sparse_sample.feats.device, dtype=torch.float32)
                else:
                    grid_size = torch.full((3,), 64.0, device=sparse_sample.feats.device, dtype=torch.float32)
                xyz = (coords[:, 1:4].float() + 0.5) / grid_size - 0.5
                if bool(mv_fusion.get("soft_blend", False)):
                    alpha = _box_soft_alpha_from_xyz(
                        xyz,
                        box_min,
                        box_max,
                        edge_width=float(mv_fusion.get("soft_edge_width", 0.035)),
                        core_width=float(mv_fusion.get("soft_core_width", 0.04)),
                        outside_width=float(mv_fusion.get("soft_outside_width", 0.04)),
                        inside_plateau=float(mv_fusion.get("soft_inside_plateau", 0.9)),
                    )
                    return alpha[:, None].to(dtype=sparse_sample.feats.dtype)
                inside = ((xyz >= box_min) & (xyz <= box_max)).all(dim=1)
            else:
                x0, y0, x1, y1 = [float(v) for v in mv_fusion.get("conf_bbox", [0.36, 0.30, 0.64, 0.58])]
                grid_y = int(mask.shape[-2]) if mask is not None and torch.is_tensor(mask) and mask.ndim >= 5 else 64
                grid_z = int(mask.shape[-1]) if mask is not None and torch.is_tensor(mask) and mask.ndim >= 5 else 64
                y0i = max(0, min(grid_y, int(round(y0 * grid_y))))
                y1i = max(y0i + 1, min(grid_y, int(round(y1 * grid_y))))
                z0i = max(0, min(grid_z, int(round(x0 * grid_z))))
                z1i = max(z0i + 1, min(grid_z, int(round(x1 * grid_z))))
                inside = (coords[:, 2] >= y0i) & (coords[:, 2] < y1i) & (coords[:, 3] >= z0i) & (coords[:, 3] < z1i)
            return torch.where(
                inside[:, None],
                torch.full((coords.shape[0], 1), alpha_inside, device=sparse_sample.feats.device, dtype=sparse_sample.feats.dtype),
                torch.full((coords.shape[0], 1), alpha_outside, device=sparse_sample.feats.device, dtype=sparse_sample.feats.dtype),
            )

        def _fused_mv_repaint_prediction(x_t, t_val):
            target_cond = mv_fusion["target_cond"]

            if fusion_mode == "view_average":
                # Ablation mode: ignore confidence regions and average target
                # velocities from every provided view equally.
                v_all = []
                for view_i in range(len(view_names_fuse)):
                    _, _, v_i = _predict_repaint_v_with_cond(_cond_at(target_cond, view_i), x_t, t_val)
                    v_all.append(v_i)
                fused_feats = torch.zeros_like(v_all[0].feats)
                inv_n = 1.0 / float(len(v_all))
                for v_i in v_all:
                    fused_feats = fused_feats + v_i.feats.to(v_all[0].feats.dtype) * inv_n
                v_fused = v_all[0].replace(feats=fused_feats)
            else:
                # ---- predict every needed velocity in the SLAT stage ----
                _, _, v_base = _predict_repaint_v_with_cond(_cond_at(target_cond, base_i), x_t, t_val)
                v_trust_feats_list = []
                for tv_i in trust_indices:
                    _, _, v_tv = _predict_repaint_v_with_cond(_cond_at(target_cond, tv_i), x_t, t_val)
                    v_trust_feats_list.append(v_tv.feats.to(v_base.feats.dtype))

                # ---- compute per-trust alpha with "newest trust view wins" ----
                if support_chi64_t is not None:
                    xyz = x_t.coords[:, 1:4].long().clamp(0, support_chi64_t[0].shape[0] - 1)
                    raw_alphas = [chi[xyz[:, 0], xyz[:, 1], xyz[:, 2]][:, None] for chi in support_chi64_t]
                else:
                    raw_alphas = [
                        _sparse_mv_alpha(x_t, tv_bounds_np=b) for b in (trust_bounds_list or [None])
                    ]
                # normalize dtype for arithmetic
                raw_alphas = [a.to(v_base.feats.dtype).clamp(0.0, 1.0) for a in raw_alphas]
                remaining = torch.ones_like(raw_alphas[0])
                per_trust_alpha = [None] * len(raw_alphas)
                for k in reversed(range(len(raw_alphas))):
                    a_k = raw_alphas[k]
                    per_trust_alpha[k] = a_k * remaining
                    remaining = remaining * (1.0 - a_k)
                alpha_base = remaining

                # ---- weighted sum ----
                fused_feats = alpha_base * v_base.feats
                for k in range(len(raw_alphas)):
                    fused_feats = fused_feats + per_trust_alpha[k] * v_trust_feats_list[k]
                v_fused = v_base.replace(feats=fused_feats)
            pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t_val, v=v_fused)
            return pred_x_0, v_fused

        eps_anchor = None
        if repaint_anchor_eps and (x_src is not None):
            if hasattr(x_src, "feats") and hasattr(x_src, "coords"):
                eps_anchor = torch.randn_like(x_src.feats)
            else:
                eps_anchor = torch.randn_like(x_src)

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for step_i, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc="Sampling (Repaint)", disable=not verbose)
        ):
            kwargs["t_sign"] = t
            kwargs["t_1"] = t

            if mv_fusion_slat_enabled:
                pred_x_0, pred_v = _fused_mv_repaint_prediction(sample, t)
                z_edit = sample - (t - t_prev) * pred_v
            else:
                out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
                z_edit = out.pred_x_prev
                pred_x_0 = out.pred_x_0

            if x_src is not None and mask is not None and repaint_beta > 0.0:
                if step_i < repaint_inv_steps:
                    _drop = {
                        "x_src", "mask", "stage", "ori_cond",
                        "repaint_beta", "repaint_anchor_eps",
                        "repaint_inv_steps",
                        "mv_fusion", "mv_velocity_debug", "latent", "latent_slat",
                        "feature_path", "edit_mask", "decode_voxel_fn",
                    }
                    inner_kwargs = {k: v for k, v in kwargs.items() if k not in _drop}
                    x0_consistent = self._tweedie_self_consistent_x0(
                        model, x_src, t, cond_src,
                        eps_anchor=eps_anchor, stage=stage, **inner_kwargs,
                    )
                    z_src_t_prev = self._forward_diffuse_src(
                        x0_consistent, t_prev,
                        stage=stage, eps_override=eps_anchor,
                    )
                else:
                    z_src_t_prev = self._forward_diffuse_src(
                        x_src, t_prev,
                        stage=stage, eps_override=eps_anchor,
                    )

                z_next = self._mask_blend(
                    z_edit, z_src_t_prev, mask,
                    stage=stage, beta=repaint_beta,
                )
            else:
                z_next = z_edit

            sample = z_next
            ret.pred_x_t.append(z_next)
            ret.pred_x_0.append(pred_x_0)

        if x_src is not None and mask is not None:
            sample = self._mask_overwrite_outside(sample, x_src, mask, stage=stage)

        ret.samples = sample
        return ret

    def _tweedie_self_consistent_x0(
        self, model, x_src, t, cond_src, eps_anchor=None, stage="sparse", **kwargs
    ):
        
        z_src_naive = self._forward_diffuse_src(
            x_src, t, stage=stage, eps_override=eps_anchor,
        )
        _, _, v_src = self._get_model_prediction(model, z_src_naive, t, cond_src, **kwargs)
        tv = float(t)
        if hasattr(z_src_naive, "feats") and hasattr(z_src_naive, "coords"):
            x0_feats = z_src_naive.feats - tv * v_src.feats
            return z_src_naive.replace(feats=x0_feats)
        return z_src_naive - tv * v_src

    def _forward_diffuse_src(self, x_src, t_prev, stage="sparse", eps_override=None):
       
        t_val = float(t_prev)
        scale_src = 1.0 - t_val
        scale_eps = self.sigma_min + (1.0 - self.sigma_min) * t_val

        if hasattr(x_src, "feats") and hasattr(x_src, "coords"):
            eps = eps_override if eps_override is not None else torch.randn_like(x_src.feats)
            new_feats = scale_src * x_src.feats + scale_eps * eps
            return x_src.replace(feats=new_feats)

        eps = eps_override if eps_override is not None else torch.randn_like(x_src)
        return scale_src * x_src + scale_eps * eps

    def _mask_blend(self, z_edit, z_src_t, mask, stage="sparse", beta: float = 1.0):
        
        if not (hasattr(z_edit, "feats") and hasattr(z_edit, "coords")):
            m = mask
            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=z_edit.device, dtype=z_edit.dtype)
            else:
                m = m.to(device=z_edit.device, dtype=z_edit.dtype)
            return z_edit + (1.0 - m) * beta * (z_src_t - z_edit)

        if beta >= 1.0 - 1e-6:
            alpha = 0.0
            return blend_feats_ring1_outside_rawmask(z_edit, z_src_t, mask, alpha=alpha)

        device = z_edit.feats.device
        coords_e = z_edit.coords.to(device).long()
        coords_s = z_src_t.coords.to(device).long()
        m_grid = mask.to(device)
        if m_grid.dtype != torch.bool:
            m_grid = m_grid > 0
        b_e, x_e, y_e, z_e = coords_e.unbind(dim=1)
        is_inside = m_grid[b_e, 0, x_e, y_e, z_e]

        ori_map = {tuple(c.tolist()): i for i, c in enumerate(coords_s)}

        new_feats = z_edit.feats.clone()
        for i in range(coords_e.size(0)):
            if bool(is_inside[i]):
                continue
            j = ori_map.get(tuple(coords_e[i].tolist()))
            if j is None:
                continue
            new_feats[i] = beta * z_src_t.feats[j].to(new_feats.dtype) \
                         + (1.0 - beta) * new_feats[i]
        return z_edit.replace(feats=new_feats)

    def _mask_overwrite_outside(self, sample, x_src, mask, stage="sparse"):
       
        if not (hasattr(sample, "feats") and hasattr(sample, "coords")):
            m = mask
            if not torch.is_tensor(m):
                m = torch.as_tensor(m, device=sample.device, dtype=sample.dtype)
            else:
                m = m.to(device=sample.device, dtype=sample.dtype)
            return m * sample + (1.0 - m) * x_src

        
        device = sample.feats.device
        coords_s = sample.coords.to(device).long()
        coords_o = x_src.coords.to(device).long()
        m_grid = mask.to(device)
        if m_grid.dtype != torch.bool:
            m_grid = m_grid > 0
        b_s, x_s, y_s, z_s = coords_s.unbind(dim=1)
        is_inside = m_grid[b_s, 0, x_s, y_s, z_s]

        ori_map = {tuple(c.tolist()): i for i, c in enumerate(coords_o)}

        new_feats = sample.feats.clone()
        for i in range(coords_s.size(0)):
            if bool(is_inside[i]):
                continue
            j = ori_map.get(tuple(coords_s[i].tolist()))
            if j is None:
                continue
            new_feats[i] = x_src.feats[j].to(new_feats.dtype)
        return sample.replace(feats=new_feats)


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
