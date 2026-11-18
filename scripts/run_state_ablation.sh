#!/usr/bin/env bash
# Three-state vs two-state diagnosis ablation (round 1 only, views 0,1, base view 0, auto view support).
#   MODE=three_state          quiescent voxels locked back to the source (default method)
#   MODE=three_state_srcocc   only quiescent voxels also occupied in the source (never deletes new geometry)
#   MODE=no_lockback          two states: conflict / stable, nothing locked back
#   MODE=lockback_nonconflict two states: conflict / source, every non-conflict cell of mask_all locked back
# Composition cases (compose_views.json) reuse the round-1 render cache of RENDER_CACHE and run with
# pair crop + outward feather, as in run_mv_fusion_two_rounds.sh; the other cases keep their old settings.
# Output: <item_dir>/_easy3e_out_round1_auto_support_abl_<MODE>
# Non-composition cases may pass known cameras with SUPPORT_CAMERAS=<view_cameras json> (e.g. rigged assets,
# whose bind-pose trimesh geometry breaks the silhouette camera fit).
# SEED (default 1) != 1 appends _s<SEED> to the run tag.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> MODE=<mode> [SEED=1] [RENDER_CACHE=<render dir>] [SUPPORT_CAMERAS=<json>] \
#            bash run_state_ablation.sh <item_dir>
set -euo pipefail

PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${TRELLIS_PYTHON:-python}
ITEM=$(realpath "$1")
MODE=${MODE:?set MODE}

export TRELLIS_CHECKPOINT=${TRELLIS_CHECKPOINT:-$PROJ/checkpoints/TRELLIS-image-large}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-32}

cd "$PROJ"
ARGS=(--item_dir "$ITEM" --view_support auto --no_guidance --export_edit_magnitude_conflict
      --skip_next_view_acquisition --views 0,1 --base_view 0 --state_mode "$MODE")
SEED=${SEED:-1}
TAG=abl_$MODE
if [ "$SEED" != 1 ]; then TAG+=_s$SEED; fi
ARGS+=(--seed "$SEED" --run_tag "$TAG")
if [ -f "$ITEM/compose_views.json" ]; then
    RENDER=$ITEM/_easy3e_work_round1_auto_support_$TAG/render
    mkdir -p "$RENDER"
    if [ -n "${RENDER_CACHE:-}" ] && [ ! -s "$RENDER/slat_latent.pt" ]; then
        find "$RENDER_CACHE" -maxdepth 1 -type f ! -name '*.png' -exec cp {} "$RENDER/" \;
    fi
    ARGS+=(--source_transform "$ITEM/transforms_compose.json" --support_camera_json "$ITEM/view_cameras_compose.json"
           --pair_crop --feather_outward)
elif [ -n "${SUPPORT_CAMERAS:-}" ]; then
    ARGS+=(--support_camera_json "$SUPPORT_CAMERAS")
fi
$PY -u mv_fusion/run_mv_fusion.py "${ARGS[@]}"
