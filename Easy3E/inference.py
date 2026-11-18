import os
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import time

import torch
from tqdm import tqdm

from pipeline_edit.edit import process_single_sample
from pipeline_edit.preprocess import preprocess_single_sample
from pipeline_edit.utils import load_pipeline


if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.time()
last_t = t0


def parse_args():
    parser = argparse.ArgumentParser(description="TRELLIS 3D Reconstruction Pipeline")

    parser.add_argument("--input_dir", default="",
                      help="Input image directory path")
    parser.add_argument("--input_image", default=None,
                      help="Direct target image path; if set, input_dir is not required")
    parser.add_argument("--output_dir", default="",
                      help="Output result root directory")
    parser.add_argument("--mask_dir", default=None,
                      help="Input mask directory path (flat layout: <mask_dir>/<edit_name>.glb)")
    parser.add_argument("--mask_path", default=None,
                      help="Direct 3D mask GLB path; if set, mask_dir is not required")
    parser.add_argument("--mask_voxels", default=None,
                      help="Edit mask as a 64^3 boolean .npy in the normalized internal frame (e.g. a later "
                           "mv_fusion round's dilated conflict region). Takes precedence over --mask_path.")
    parser.add_argument("--mask_voxels_latent_threshold", type=float, default=0.5,
                      help="Coverage above which a 16^3 latent cell of --mask_voxels is editable (default 0.5 as for "
                           "GLB masks; 0 = any covered cell, recommended for small local regions).")

    parser.add_argument("--root_dir", default="", help="Processing root directory")
    parser.add_argument("--render_dir", default=None,
                      help="Direct render/cache directory containing mesh.ply, feature.npz, ss_latent.pt, slat_latent.pt and transforms.json")
    parser.add_argument("--feature_dir", default=None,
                      help="Direct directory for edit intermediate outputs; defaults to <root_dir>/feature")
    parser.add_argument("--mode", default="baseline",
                      help="Run mode: baseline | edit")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing only")

    parser.add_argument("--checkpoint", default="checkpoint/TRELLIS-image-large",
                      help="Pretrained model path")
    parser.add_argument("--obj_path", default="assets/example_mesh/T.ply",
                      help="Source 3D model directory")
    parser.add_argument("--original_image", default="",
                      help="Original reference image path")
    parser.add_argument("--encoder_name", default="ss_enc_conv3d_16l8_fp16",
                      help="SS encoder weight subdirectory name")
    parser.add_argument('--enc_pretrained', type=str, default=None,
                      help='SLAT encoder weight path')

    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")

    parser.add_argument("--cfg_src", type=float, default=5.0,
                      help="Source condition CFG strength in FlowEdit (default 5.0)")
    parser.add_argument("--cfg_tar", type=float, default=5.0,
                      help="Target condition CFG strength in FlowEdit (default 5.0)")
    parser.add_argument("--flowedit_lambda", type=float, default=0.5,
                      help="Source velocity coefficient in FlowEdit: v_target - lambda * v_src (default 0.5)")
    parser.add_argument("--mask_feather_outward", action="store_true",
                      help="Sparse-structure FlowEdit: feather the latent edit mask outward only (interior stays 1). "
                           "The default average-pool feather attenuates small or thin edit regions to ~0.3.")
    parser.add_argument("--no_guidance", action="store_true",
                      help="Disable orthographic silhouette guidance in FlowEdit")
    parser.add_argument("--source_transform_path", default=None,
                      help="Optional transforms.json whose scale/offset should be applied to the source asset during preprocessing")
    parser.add_argument("--full_edit_mask", action="store_true",
                      help="Treat the whole TRELLIS 64^3 workspace as editable and do not load a 3D mask glb")
    parser.add_argument("--enable_step_viz", action="store_true",
                      help="Export sparse-structure FlowEdit intermediate latents as GLB files")
    parser.add_argument("--step_viz_every", type=int, default=5,
                      help="Export one sparse-structure step GLB every N FlowEdit steps")
    parser.add_argument("--debug_mv_velocity_diff", action="store_true",
                      help="Export multi-view first-stage FlowEdit delta-velocity difference heatmaps without changing inference")
    parser.add_argument("--debug_mv_source_images", default="",
                      help="Comma-separated source-view images for multi-view velocity-difference debug")
    parser.add_argument("--debug_mv_target_images", default="",
                      help="Comma-separated edited target-view images for multi-view velocity-difference debug")
    parser.add_argument("--debug_mv_view_names", default="",
                      help="Comma-separated view labels matching debug_mv_source_images/debug_mv_target_images")
    parser.add_argument("--debug_mv_steps", default="1,3,5,10,20",
                      help="Comma-separated 1-based FlowEdit steps for multi-view velocity-difference heatmap export")
    parser.add_argument("--debug_mv_output_dir", default=None,
                      help="Optional output directory for multi-view velocity-difference debug files")

    parser.add_argument("--mv_fusion", action="store_true",
                      help="Enable 2-view velocity fusion in sparse FlowEdit")
    parser.add_argument("--mv_fusion_source_images", default="",
                      help="Comma-separated source-view images for velocity fusion")
    parser.add_argument("--mv_fusion_target_images", default="",
                      help="Comma-separated edited target-view images for velocity fusion")
    parser.add_argument("--mv_fusion_view_names", default="",
                      help="Comma-separated view labels matching mv_fusion_source_images/mv_fusion_target_images")
    parser.add_argument("--mv_fusion_trust_view", default="003",
                      help="[DEPRECATED, use --mv_fusion_trust_views] View label whose velocity is trusted inside the confidence box")
    parser.add_argument("--mv_fusion_trust_views", default="",
                      help="Comma-separated list of view labels whose velocity is trusted inside their per-view confidence regions. "
                           "If provided, overrides --mv_fusion_trust_view. When multiple trust views' regions overlap on the same voxel, "
                           "later views in the list take priority (i.e. right-most trust view wins).")
    parser.add_argument("--mv_fusion_base_view", default="005",
                      help="View label whose velocity is used outside any trust region")
    parser.add_argument("--mv_fusion_fusion_mode", choices=("trust_region", "view_average"), default="trust_region",
                      help="Velocity fusion mode for mv_fusion. 'trust_region' keeps the confidence-region weighted fusion; "
                           "'view_average' averages all mv_fusion views equally and ignores confidence regions.")
    parser.add_argument("--mv_fusion_conf_bbox", default="0.36,0.30,0.64,0.58",
                      help="Normalized image-space bbox x0,y0,x1,y1 for the trusted-view confidence region (single-trust-view fallback only)")
    parser.add_argument("--mv_fusion_conf_box_glb", default=None,
                      help="Optional 3D box GLB whose scene bounds define the trusted-view confidence region (single-trust-view fallback only)")
    parser.add_argument("--mv_fusion_conf_box_glbs", default="",
                      help="Comma-separated 3D region GLBs, one per entry in --mv_fusion_trust_views. Each defines the confidence region of the "
                           "corresponding trust view. Required when --mv_fusion_trust_views has more than one entry.")
    parser.add_argument("--mv_fusion_pair_crop", action="store_true",
                      help="Crop each source/target view pair with one shared box (union of both silhouettes) "
                           "instead of cropping each image to its own object. Needed when targets add parts "
                           "outside the source silhouette; targets must be aligned to the source pixel frame.")
    parser.add_argument("--mv_fusion_support_npz", default=None,
                      help="Optional view_support.npz from compute_view_support.py with geometry-derived per-view support "
                           "chi_v on the 64^3 grid. When set, it replaces the confidence-region GLBs: each trust view's chi "
                           "is its raw ownership, later trust views win on overlap, the base view takes the remainder.")
    parser.add_argument("--mv_fusion_alpha_inside", type=float, default=1.0,
                      help="Fusion alpha inside the trusted-view confidence region; 1.0 means use only the trusted-view edit velocity")
    parser.add_argument("--mv_fusion_alpha_outside", type=float, default=0.0,
                      help="Fusion alpha outside the trusted-view confidence region; 0.0 means use only the base-view edit velocity")
    parser.add_argument("--mv_fusion_soft_blend", action="store_true",
                      help="Use conservative distance-based soft blend instead of hard alpha inside/outside the confidence region")
    parser.add_argument("--mv_fusion_soft_edge_width", type=float, default=0.035,
                      help="World-space width of the high-fusion band just inside the confidence boundary")
    parser.add_argument("--mv_fusion_soft_core_width", type=float, default=0.04,
                      help="World-space width from conservative main-view plateau to pure core")
    parser.add_argument("--mv_fusion_soft_outside_width", type=float, default=0.04,
                      help="World-space width of the outside connection band around the confidence boundary")
    parser.add_argument("--mv_fusion_soft_inside_plateau", type=float, default=0.9,
                      help="Dominant-view alpha in most of the trusted core before reaching the pure center")
    parser.add_argument("--mv_fusion_trust_region_step", type=int, default=10,
                      help="1-based FlowEdit step at which to export generated voxels colored by the trusted-view confidence region; set <=0 to disable")
    parser.add_argument("--mv_fusion_steps", default="1,3,5,10,20",
                      help="Comma-separated 1-based FlowEdit steps for fused-vs-reference voxel diff export")
    parser.add_argument("--mv_fusion_output_dir", default=None,
                      help="Optional output directory for velocity-fusion voxel-diff files")
    parser.add_argument("--mv_fusion_velocity_conflict_threshold", default="auto",
                      help="Threshold for the trust-vs-base velocity conflict glb. Use 'auto' for mean+std, or a positive float for a manual threshold.")
    parser.add_argument("--mv_fusion_velocity_conflict_min_ratio", type=float, default=0.0,
                      help="If >0, at least this fraction of mask_all voxels will be marked as conflict (threshold is lowered to the corresponding quantile). Default 0.0 = pure threshold-based.")
    parser.add_argument("--mv_fusion_export_edit_magnitude_conflict", action="store_true",
                      help="Export row-3 diagnostic heatmap/mask using only large fused edit magnitude, ignoring multi-view velocity disagreement.")
    parser.add_argument("--mv_fusion_edit_magnitude_conflict_threshold", default="auto",
                      help="Threshold for the edit-magnitude-only conflict glb. Use 'auto' for mean+std, or a positive float for a manual threshold.")
    parser.add_argument("--mv_fusion_edit_magnitude_conflict_min_ratio", type=float, default=0.0,
                      help="If >0, at least this fraction of mask_all voxels will be marked in the edit-magnitude-only conflict mask.")
    # ------- Quiescent region (velocity-stable low-magnitude subset of mask_all) -------
    parser.add_argument("--mv_fusion_enable_quiescent", dest="mv_fusion_quiescent_enabled",
                      action="store_true", default=True,
                      help="Enable quiescent detection: freeze velocity-stable low-magnitude "
                           "voxels inside mask_all (reuse source mesh there). Enabled by default; "
                           "use --mv_fusion_disable_quiescent to turn off.")
    parser.add_argument("--mv_fusion_disable_quiescent", dest="mv_fusion_quiescent_enabled",
                      action="store_false",
                      help="Disable quiescent detection entirely: raw_mask stays intact for the "
                           "SLAT stage, quiescent_* files are NOT exported, and ret.quiescent_mask "
                           "is NOT attached. Overrides --mv_fusion_quiescent_ratio and both "
                           "quiescent thresholds.")
    parser.add_argument("--mv_fusion_state_mode", default="three_state",
                      choices=["three_state", "three_state_srcocc", "no_lockback", "lockback_nonconflict"],
                      help="Voxels locked back to the source before the SLAT stage: three_state = quiescent "
                           "voxels only (default); three_state_srcocc = quiescent voxels also occupied in the source; no_lockback = none (conflict / stable); "
                           "lockback_nonconflict = every non-conflict cell of mask_all (conflict / source).")
    parser.add_argument("--mv_fusion_quiescent_ratio", type=float, default=0.0,
                      help="Upper-bound fraction of mask_all voxels to mark as quiescent (0..1). "
                           "0 (default) disables the ratio cap and relies purely on the two "
                           "quiescent thresholds. When >0 both quiescent thresholds are shrunk "
                           "to the corresponding quantile so at most this fraction is chosen.")
    parser.add_argument("--mv_fusion_quiescent_disagreement_threshold", default="auto",
                      help="Upper bound on multi-view velocity disagreement for a voxel to be "
                           "quiescent. Use 'auto' (default, ~30th-percentile inside mask_all) or a positive float.")
    parser.add_argument("--mv_fusion_quiescent_edit_mag_threshold", default="auto",
                      help="Upper bound on fused edit-magnitude for a voxel to be quiescent. "
                           "Use 'auto' (default, ~30th-percentile inside mask_all) or a positive float.")

    # ------- TRELLIS sampler quality knobs -------
    parser.add_argument("--sampler_steps", type=int, default=25,
                      help="Number of sampler steps for BOTH sparse structure and slat samplers (overrides pipeline.json's default of 25). "
                           "Default 25 (official-quality, fastest); set 50 for training-time quality.")
    parser.add_argument("--sparse_structure_steps", type=int, default=None,
                      help="Override --sampler_steps for the sparse-structure sampler only. Defaults to --sampler_steps.")
    parser.add_argument("--slat_steps", type=int, default=None,
                      help="Override --sampler_steps for the slat sampler only. Defaults to --sampler_steps.")
    parser.add_argument("--texture_size", type=int, default=2048,
                      help="Baked texture resolution for the exported GLB (default 2048).")

    # ------- Method D: reuse a previous round's FINAL edited SLAT as z_src -------
    parser.add_argument("--slat_latent_override", default=None,
                      help="Optional path to a *.pt file saved by a previous flowedit run "
                           "(feature/slat_edited.pt). When set, the pipeline loads this file "
                           "as latent_slat (z_src) INSTEAD of the render/slat_latent.pt "
                           "produced by preprocess. This lets round-2 skip the lossy "
                           "re-encoder path (glb -> render 150 views -> dinov2 -> encoder) "
                           "and directly reuse round-1's final SLAT, so mask-outside SLAT is "
                           "byte-identical to what round-1's decoder saw.")
    parser.add_argument("--ss_latent_override", default=None,
                      help="Optional ss_latent_edited.pt saved by a previous flowedit run; used as the source "
                           "sparse-structure latent x_s instead of render/ss_latent.pt (round state continuation).")
    parser.add_argument("--ss_latent_from_slat_override", action="store_true",
                      help="When --slat_latent_override is set and no --ss_latent_override is given, re-encode the "
                           "source sparse-structure latent from the occupancy of the reused SLAT coords instead of "
                           "using render/ss_latent.pt (voxelized decoded mesh).")

    return parser.parse_args()


def _list_input_images(input_dir):
    return sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])


def _get_input_images(args):
    if args.input_image:
        if not os.path.isfile(args.input_image):
            raise FileNotFoundError(f"Input image not found: {args.input_image}")
        return [args.input_image]
    if not args.input_dir:
        raise ValueError("Either --input_image or --input_dir must be provided")
    return _list_input_images(args.input_dir)


def main():
    args = parse_args()
    if args.enc_pretrained is None:
        args.enc_pretrained = os.path.join(args.checkpoint, "ckpts", "slat_enc_swin8_B_64l8_fp16")

    if args.preprocess:
        preprocess_single_sample(args)
        return

    image_paths = _get_input_images(args)
    pipeline = load_pipeline(args)

    if args.mode == "edit":
        print("[INFO] Running in Edit mode...")
    else:
        print(f"[INFO] Running in {args.mode.upper()} mode...")

    for image_path in tqdm(image_paths, desc="Processing samples"):
        process_single_sample(args, pipeline, encoder=None, image_path=image_path)

    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
