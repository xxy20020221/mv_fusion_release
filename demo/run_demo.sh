#!/usr/bin/env bash
# Demo: two cases, two input modes, two rounds each (see README.md "Demo").
#   ninja : manual trust regions   (mask.glb: mask_all + mask_region_1; round 2: mask_2.glb::mask_region_new)
#   rider : geometry-derived view support (only mask.glb::mask_all is used); round 2 from the round-1 diagnosis
# No API key is needed: every target image is provided. Outputs: demo/<case>/_easy3e_out_round{1,2}*/out/*.glb
# Usage: CUDA_VISIBLE_DEVICES=0 bash demo/run_demo.sh [ninja|rider|all]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${TRELLIS_PYTHON:-python}
CASE=${1:-all}
cd "$ROOT"

if [ "$CASE" = ninja ] || [ "$CASE" = all ]; then
    # Round 1: views 0 (base) and 1 (trust view, region mask_region_1).
    $PY -u mv_fusion/run_mv_fusion.py --item_dir demo/ninja --views 0,1 \
        --no_guidance --skip_next_view_acquisition
    # Round 2: source = round-1 result, mask_all = round-1 conflict voxels minus quiescent voxels;
    # view 2 (front, az 270 / el 0) is added; its source image is rendered from the round-1 result.
    $PY -u mv_fusion/run_mv_fusion.py --item_dir demo/ninja --round 2 \
        --new_view_id 2 --new_view_angle 270,0 --no_guidance --skip_next_view_acquisition
fi

if [ "$CASE" = rider ] || [ "$CASE" = all ]; then
    # Round 1: per-view support chi_v from mask_all, fitted cameras and the source/target images.
    $PY -u mv_fusion/run_mv_fusion.py --item_dir demo/rider --views 0,1 --view_support auto \
        --no_guidance --skip_next_view_acquisition
    # Round 2: next region = Close(Dilate(conflict)) minus quiescent, all views re-rendered from the
    # round-1 result; view 2 (az 210 / el 30, target_2.png given) is added with the highest priority.
    # Use --new_view_angle auto --edit_instruction "$(cat demo/rider/prompt.txt)" without target_2.png
    # to let the planner pick the view and Nano Banana synthesize its target (needs MAGIC666_API_KEY).
    $PY -u mv_fusion/run_mv_fusion.py --item_dir demo/rider --round 2 --view_support auto \
        --new_view_id 2 --new_view_angle 210,30 --no_guidance --skip_next_view_acquisition
fi
