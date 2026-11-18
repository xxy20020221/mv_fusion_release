#!/usr/bin/env bash
# Two-round Route-Diagnose-Refine mv_fusion for a composition case (compose_views.json layout).
#   round 1: VIEWS (default 0,1; the first one is the base view), auto view support, rough mask
#            from build_rough_edit_mask.py
#   round 2: adds the remaining view (its target already exists), edit region from the round-1
#            diagnosis; skipped with ROUNDS=1
#   composition cases need PAIR_CROP=1 (source/target conditions cropped with one shared box) and
#   FEATHER_OUTWARD=1 (small edit regions are not damped by the latent-mask feather); both default to on.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [VIEWS=0,1] [ROUNDS=1|2] [TAG=<run_tag>] [PAIR_CROP=0|1] [FEATHER_OUTWARD=0|1] \
#            bash run_mv_fusion_two_rounds.sh <item_dir>
set -euo pipefail

PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${TRELLIS_PYTHON:-python}
ITEM=$(realpath "$1")

export TRELLIS_CHECKPOINT=${TRELLIS_CHECKPOINT:-$PROJ/checkpoints/TRELLIS-image-large}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-32}

cd "$PROJ"
if [ ! -f "$ITEM/mask.glb" ] || [ ! -f "$ITEM/transforms_compose.json" ]; then
    $PY -u mv_fusion/build_rough_edit_mask.py --item_dir "$ITEM"
fi

# Silhouette guidance assumes ortho 1.0 / elevation 0 / azimuth = view id * 22.5; these views differ.
COMMON=(--view_support auto --no_guidance --export_edit_magnitude_conflict --skip_next_view_acquisition)
SUFFIX=_auto_support
if [ -n "${TAG:-}" ]; then COMMON+=(--run_tag "$TAG"); SUFFIX+="_$TAG"; fi
if [ "${PAIR_CROP:-1}" = 1 ]; then COMMON+=(--pair_crop); fi
if [ "${FEATHER_OUTWARD:-1}" = 1 ]; then COMMON+=(--feather_outward); fi

VIEWS=${VIEWS:-0,1}
$PY -u mv_fusion/run_mv_fusion.py --item_dir "$ITEM" "${COMMON[@]}" --views "$VIEWS" --base_view "${VIEWS%%,*}" \
    --source_transform "$ITEM/transforms_compose.json" --support_camera_json "$ITEM/view_cameras_compose.json"
[ "${ROUNDS:-2}" = 1 ] && exit 0

# Round 2 adds the first compose view not used in round 1, at its known camera.
read -r NEW_ID NEW_ANGLE < <($PY -c "import json,sys; used=sys.argv[2].split(','); v=[v for v in json.load(open(sys.argv[1]))['views'] if v['label'] not in used][0]; print(v['label'], f\"{v['azimuth']},{v['elevation']}\")" "$ITEM/compose_views.json" "$VIEWS")
$PY -u mv_fusion/run_mv_fusion.py --item_dir "$ITEM" "${COMMON[@]}" --round 2 \
    --round1_out_dir "$ITEM/_easy3e_out_round1$SUFFIX" --new_view_id "$NEW_ID" --new_view_angle "$NEW_ANGLE"
