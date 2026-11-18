# mv_fusion: Disagreement-Aware Multi-View Editing for 3D Assets

`mv_fusion` edits an existing textured 3D asset (GLB) from **several edited views at once**. It is built on
[Easy3E](https://github.com/linchanyi/Easy3E) / [TRELLIS](https://github.com/microsoft/TRELLIS):

1. **Route** - inside a rough 3D edit mask, every edited view gets a per-voxel trust region (hand-made, or the
   geometry-derived support `chi_v` computed from visibility, the 2D edit region and the viewing angle).
2. **Fuse** - at every FlowEdit denoising step of the sparse structure (and of the SLAT repaint), the
   view-conditioned edit velocities are fused per voxel by these trust regions; later views win on overlap.
3. **Diagnose** - the multi-view velocity disagreement splits the edit mask into three states:
   *conflict* (C, unresolved), *edited-stable* (S, kept) and *quiescent* (U, locked back to the source).
4. **Refine** - a second round continues from the first round's asset/latents, edits only around the conflict
   set (minus U), and adds a new view chosen by the next-view planner (its target can be synthesized with an
   image-editing API).

```
mv_fusion_release/
├── mv_fusion/                 # pipeline
│   ├── run_mv_fusion.py       #   entry point: one item directory, round 1 or 2
│   ├── paths.py               #   repo layout, checkpoint / Blender locations (env-overridable)
│   ├── compute_view_support.py#   camera fitting + per-view support chi_v (auto trust regions)
│   ├── mv_fusion_rounds.py    #   round transition: diagnosis states, next edit region, Nano Banana synthesis
│   ├── plan_mv_fusion_next_view.py  # next-view acquisition (ours / random / farthest / coverage)
│   ├── build_rough_edit_mask.py     # rough 3D edit mask from source/target views (composition cases)
│   ├── detect_image_edit_regions.py # 2D source-vs-target edit regions + alignment
│   ├── process_easy3e_rgba_target.py# background removal of target images (rembg)
│   ├── export_easy3e_final_glb_from_slat.py, export_edit_region_on_asset.py
├── Easy3E/                    # Easy3E + TRELLIS with the mv_fusion sampler (trellis/pipelines/samplers/flow_euler.py)
├── api/                       # image-editing / VLM API clients (OpenAI-compatible gateway)
│   ├── magic666_client.py     #   text / vision / image generation (Nano Banana) with model fallback
│   └── vlm_client.py          #   JSON-schema VLM calls + image generation, with call logging
├── compose_prep/              # API pipeline that builds multi-view targets for "A + B -> composition" cases
├── blender_scripts/           # Blender 4.0 renderers used by the pipeline and the evaluation
├── evaluation/                # CLIP-T, CLIP-I, DINO-I, LPIPS, PSNR/SSIM, FID, ablations, diagnosis studies
├── scripts/                   # batch launchers (composition two-round run, state ablation)
├── demo/                      # demo inputs: rider (auto support) and ninja (manual trust regions)
└── requirements.txt
```

## 1. Installation

Tested with Linux, Python 3.10, CUDA 12.1, PyTorch 2.4.0, one GPU with >= 16 GB (a demo round peaks at ~14 GB).

### 1.1 Python environment

The environment is the TRELLIS environment plus a few packages:

```bash
conda create -n mvfusion python=3.10 -y && conda activate mvfusion
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA extensions of TRELLIS (spconv, xformers, kaolin, nvdiffrast, diff-gaussian-rasterization, utils3d)
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git third_party/TRELLIS
cd third_party/TRELLIS && . ./setup.sh --basic --xformers --spconv --mipgaussian --kaolin --nvdiffrast && cd ../..

pip install -r requirements.txt
```

`flash-attn` is not required: the pipeline runs with `ATTN_BACKEND=sdpa` and `SPARSE_ATTN_BACKEND=xformers`.

### 1.2 Model weights

| What | Where | How |
|---|---|---|
| TRELLIS-image-large (incl. the SLAT **encoder** `slat_enc_swin8_B_64l8_fp16`) | `checkpoints/TRELLIS-image-large/` (or `$TRELLIS_CHECKPOINT`) | `huggingface-cli download JeffreyXiang/TRELLIS-image-large --local-dir checkpoints/TRELLIS-image-large` |
| DINOv2 ViT-L/14-reg (image conditions, preprocessing features) | `~/.cache/torch/hub/` | downloaded automatically by `torch.hub` on the first run |
| rembg `birefnet-general` (target background removal) | `~/.u2net/` | downloaded automatically on the first run |
| Metrics: `openai/clip-vit-base-patch32`, `facebook/dinov2-base`, AlexNet (LPIPS), Inception-v3 (FID) | HF / torch caches | downloaded automatically; offline: set `CLIP_MODEL` / `DINO_MODEL` to local directories |

The `microsoft/TRELLIS-image-large` repository has no SLAT encoder; `JeffreyXiang/TRELLIS-image-large` has the same
`pipeline.json` and all checkpoints including the encoder. A separate encoder location can be set with
`TRELLIS_ENC_PRETRAINED`.

### 1.3 Blender 4.0 (rendering)

```bash
mkdir -p Easy3E/tmp && cd Easy3E/tmp
wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xf blender-4.0.0-linux-x64.tar.xz && cd ../..
# -> Easy3E/tmp/blender-4.0.0-linux-x64/blender   (or set BLENDER_PATH)
```

Blender runs headless with Cycles on the CPU. On servers without X11/OpenGL libraries, put the missing
runtime libraries under a prefix and pass it as `BLENDER_RUNTIME_PREFIX` (lib/ is added to `LD_LIBRARY_PATH`).

### 1.4 API key (optional)

Only three parts call an external API; the demo does not:

| Feature | Code | Models (defaults) |
|---|---|---|
| Target of a new round-2 view when no `target_<v>.png` is given | `mv_fusion/mv_fusion_rounds.py::synthesize_view_target` -> `api/magic666_client.py` | `gemini-3-pro-image-preview` (Nano Banana Pro) |
| Multi-view composition targets (A + B -> composite), VLM planning and QA | `compose_prep/prepare_composition_views.py` -> `api/vlm_client.py` | `gemini-3.1-pro-preview` (VLM), `gemini-3-pro-image-preview` (editing) |
| VLM residual labels in the disagreement-localization study | `evaluation/evaluate_mv_fusion_disagreement_localization.py` | `gemini-3-pro-preview` |

```bash
export MAGIC666_API_KEY=sk-...                      # never commit keys
export MAGIC666_BASE_URL=https://magic666.top/v1   # any OpenAI-compatible /chat/completions gateway
```

### 1.5 Environment variables

| Variable | Default |
|---|---|
| `TRELLIS_PYTHON` | current interpreter (used for the Easy3E subprocesses) |
| `TRELLIS_CHECKPOINT` | `checkpoints/TRELLIS-image-large` |
| `TRELLIS_ENC_PRETRAINED` | `$TRELLIS_CHECKPOINT/ckpts/slat_enc_swin8_B_64l8_fp16` |
| `BLENDER_PATH` | `Easy3E/tmp/blender-4.0.0-linux-x64/blender` |
| `TRELLIS2_REPO`, `TRELLIS2_PYTHON` | only for the 3D layout guides of `compose_prep` (optional, [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)) |

## 2. Input format

One edit = one **item directory**:

```
<item_dir>/
├── object.glb          # source asset (textured GLB, any scale/position; normalized internally)
├── mask.glb            # 3D edit mask, a GLB *scene* with named nodes (see below)
├── mask_2.glb          # manual round 2 only: node mask_region_new = trust region of the new view
├── source_<v>.png      # render of object.glb from view v (RGBA with transparent background preferred)
├── target_<v>.png      # the edited image of view v (any background; removed with rembg)
└── prompt.txt          # optional: edit text (CLIP-T, Nano Banana synthesis)
```

* **Views** `<v>` are free labels (`0`, `1`, `2`, ...). A view is used when both `source_<v>.png` and
  `target_<v>.png` exist; `--views 0,1` selects a subset. Trust views are the views with a `mask_region_<v>`
  (manual) or every view except the base view (auto, base = first view or `--base_view`); later views have
  priority where trust regions overlap.
* **Images**: square, the same camera for `source_<v>` and `target_<v>`, the object fully visible. Targets may
  have another resolution or slightly different framing; with `--pair_crop` they are aligned to the source
  (SIFT similarity, else resize) and both images are cropped with one shared box.
* **Cameras**: with `--view_support auto` the camera of every source image is fitted to `object.glb`
  (silhouette + texture, orthographic or perspective). Known cameras can be given with `--support_camera_json`
  (`{"<v>": {"azimuth", "elevation", "distance": null, "scale", "tx", "ty", "width", "height"}}`, Blender
  convention: camera at `(cos az cos el, sin az cos el, sin el)` looking at the origin, Z up). Rigged/animated
  GLBs are loaded in bind pose by trimesh; pass known cameras for them.
* **mask.glb** is expressed in the **same frame as `object.glb`** (glTF, Y up), e.g. modelled in Blender on top of
  the imported asset and exported with *Selected Objects*. Node names:

  | node | meaning | needed by |
  |---|---|---|
  | `mask_all` | closed mesh(es) covering everything that may change (rough is fine) | always |
  | `mask_region_<v>` | trust region of trust view `<v>` (where view `<v>` is reliable) | `--view_support manual` |
  | `mask_2.glb::mask_region_new` | trust region of the view added in round 2, in the normalized frame of the round-1 result (`[-0.5, 0.5]^3`) | manual round 2 |

### Two ways to provide the trust regions

| | Manual (`--view_support manual`, default) | Automatic (`--view_support auto`) |
|---|---|---|
| mask.glb | `mask_all` + `mask_region_<v>` per trust view | `mask_all` only |
| trust regions | hand-made boxes, soft-blended at the borders | `chi_v` from visibility, 2D edit support and viewing angle (Eqs. 4-7), saved in `view_support/` |
| round 2 | `mask_all` = round-1 conflict voxels minus quiescent voxels; new view from `mask_2.glb`, source rendered at `--new_view_angle` | region = Close(Dilate(C)) minus U (snapped to latent cells); every view re-rendered from the round-1 result; new view from the planner (`--new_view_angle auto`) or a fixed angle; its target is `target_<id>.png` or synthesized with Nano Banana (`--edit_instruction`) |
| outputs | `_easy3e_out_round{1,2}/` | `_easy3e_out_round{1,2}_auto_support/` |

`mask_all` itself can be drawn by hand (both demos ship hand-made masks) or built from the views:

* `mv_fusion/build_rough_edit_mask.py` carves an occlusion-aware visual hull of the 2D edit regions of all
  views and writes `mask.glb` (`mask_all`), `transforms_compose.json` and `view_cameras_compose.json`.
* For **composition** cases (two assets + a prompt), `compose_prep/prepare_composition_views.py` creates the
  views by API: VLM plan (anchor, views), Nano Banana composites per view, TRELLIS.2 layout guides, VLM QA.
  It writes `source_<v>.png`, `target_<v>.png`, `compose_views.json` (and `prompt.txt`) into the item.
  `scripts/run_mv_fusion_two_rounds.sh` then builds the mask and runs both rounds:

```bash
python compose_prep/prepare_composition_views.py --obj_a <case>/anchor.glb --obj_b <case>/part.glb \
    --item_dir <case> --prompt "the bear holds the mug with both paws"
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mv_fusion_two_rounds.sh <case>      # VIEWS=0,1 ROUNDS=2 by default
```

## 3. Demo

`demo/` contains only the inputs (rider: 10 MB, ninja: 58 MB). No API key is needed.

| case | edit | mode | views |
|---|---|---|---|
| `demo/ninja` | lower the arms, add a golden belt, hands on the buckle | manual trust regions (`mask_all`, `mask_region_1`; round 2: `mask_2.glb`) | round 1: 0 (base), 1; round 2 adds 2 (front, az 270 / el 0) |
| `demo/rider` | lower the arms, hands on the saddle | automatic support (only `mask_all` is used) | round 1: 0 (base), 1; round 2 adds 2 (az 210 / el 30) |

```bash
export TRELLIS_CHECKPOINT=$PWD/checkpoints/TRELLIS-image-large   # if not at the default location
CUDA_VISIBLE_DEVICES=0 bash demo/run_demo.sh            # both cases, two rounds each; or: ninja | rider
```

Equivalent single commands:

```bash
# ninja, manual trust regions
python mv_fusion/run_mv_fusion.py --item_dir demo/ninja --views 0,1 --no_guidance --skip_next_view_acquisition
python mv_fusion/run_mv_fusion.py --item_dir demo/ninja --round 2 --new_view_id 2 --new_view_angle 270,0 \
    --no_guidance --skip_next_view_acquisition

# rider, automatic per-view support
python mv_fusion/run_mv_fusion.py --item_dir demo/rider --views 0,1 --view_support auto --no_guidance \
    --skip_next_view_acquisition
python mv_fusion/run_mv_fusion.py --item_dir demo/rider --round 2 --view_support auto --new_view_id 2 \
    --new_view_angle 210,30 --no_guidance --skip_next_view_acquisition
# with the planner + Nano Banana instead of the given target_2.png (needs MAGIC666_API_KEY):
#   --new_view_id 3 --new_view_angle auto --edit_instruction "$(cat demo/rider/prompt.txt)"
```

Each round takes ~5-10 min on one A100/H100 (most of round 1 is the 150-view preprocessing render of the source).
Rider can also be run with manual trust regions (its `mask.glb` also contains `mask_region_1`; drop
`--view_support auto`). Ninja's source images come from another renderer whose camera the automatic fit
matches only roughly, so it is shown with manual regions.

### Outputs

```
<item_dir>/_easy3e_out_round1[_auto_support]/
├── out/
│   ├── target_<v>_rgba.glb            # FINAL edited asset (named after the primary trust view)
│   ├── quiescent_voxels.glb           # diagnosis: red = conflict C, blue = quiescent U, gray = other active voxels
│   ├── velocity_conflict_*_{heatmap,mask}.glb, *_stats.json   # multi-view disagreement D and its threshold
├── features/  slat_edited.pt, ss_latent_edited.pt (state for the next round), vox_edit.glb, mv_fusion_voxel_diff/
├── view_support/                      # auto: view_cameras.json, view_support.npz (chi_v), support_projections.png
├── round_states.glb, next_edit_region.glb, round_state.json   # auto round 2: states of round 1 and the new region
├── logs/  preprocess.log, view_support.log, edit.log
└── run_summary.json                   # resolved views, masks and the full Easy3E command
<item_dir>/_easy3e_work_round1[...]/   # render cache of the source (150 views, latents), staged images and masks
```

## 4. Main options (`mv_fusion/run_mv_fusion.py --help` for all)

| option | meaning |
|---|---|
| `--round {1,2}` | round 1 from `object.glb`, round 2 from the round-1 result |
| `--views`, `--base_view`, `--trust_views` | view subset, base view, trust views (in priority order) |
| `--view_support {manual,auto}`, `--support_camera_json` | trust regions from `mask_region_<v>` or geometry-derived `chi_v` |
| `--new_view_id`, `--new_view_angle az,el\|auto`, `--edit_instruction`, `--new_view_target` | round-2 view |
| `--next_mask_dilate/close/latent_cell` | round-2 region: Close(Dilate(C)), snapped to 4^3 latent cells |
| `--state_mode` | lock-back before the SLAT stage: `three_state` (U, default), `three_state_srcocc` (U voxels the source occupies), `no_lockback`, `lockback_nonconflict` |
| `--fusion_mode view_average` | ablation: average all views instead of routing |
| `--disable_quiescent`, `--quiescent_*_threshold` | quiescent detection (default thresholds: 30th percentiles of D and edit magnitude inside `mask_all`) |
| `--pair_crop`, `--feather_outward`, `--source_transform` | composition cases (shared crop, no feather damping of small regions, composite-centered normalization) |
| `--no_guidance` | disable Easy3E's silhouette guidance (its view assumptions do not match custom cameras; use it for all inputs above) |
| `--seed` | FlowEdit noise seed (default 1) |
| `--skip_next_view_acquisition` | skip scoring/rendering candidate next views after the round |

## 5. Evaluation (`evaluation/`)

| script | metrics |
|---|---|
| `metrics.py` | **CLIP-T** (image-text, `clip-vit-base-patch32`), **DINO-I** (`dinov2-base` mean token), **LPIPS** (AlexNet), **FID** (Inception-v3). Renders the GLBs of the given runs (`--view_azimuths/--view_elevations`) or scores an image folder (`--pred_dir`) |
| `evaluate_view_support_runs.py` | renders runs at the exact input cameras (`view_cameras.json`) and reports **PSNR, SSIM, LPIPS, DINO-I, CLIP-I, CLIP-T** per view + a comparison sheet |
| `evaluate_state_ablation.py`, `summarize_state_ablation.py` | three-state vs two-state ablation: full-image / edit-region / keep-region metrics and changed voxels per state (with `scripts/run_state_ablation.sh`) |
| `evaluate_mv_fusion_disagreement_localization.py` | does D localize the remaining errors? residuals from pixels or VLM boxes (API), lifted to 3D; AUROC, AP, Spearman |
| `experiment_unseen_view_disagreement.py`, `plot_unseen_view_rebuttal.py` | does an unseen view inflate D? (visibility-grouped disagreement and guess error) |

Examples on the demo outputs:

```bash
# rider: both rounds at the three input cameras (CLIP-T uses demo/rider/prompt.txt)
python evaluation/evaluate_view_support_runs.py --item_dir demo/rider \
    --runs _easy3e_out_round1_auto_support,_easy3e_out_round2_auto_support \
    --camera_json demo/rider/_easy3e_work_round2_auto_support/view_cameras.json \
    --target_rgba_dir demo/rider/_easy3e_work_round2_auto_support/images --shrink 0 --output_dir demo/rider/_eval

# ninja: CLIP-T / DINO-I / LPIPS / FID of both rounds
python evaluation/metrics.py --item_dir demo/ninja --easy3e_out demo/ninja/_easy3e_out_round1 \
    --easy3e_out demo/ninja/_easy3e_out_round2 --prompt "$(cat demo/ninja/prompt.txt)" --no_xvfb \
    --output_dir demo/ninja/_eval

# three-state vs two-state ablation on one item (round 1, all four lock-back modes, then the table)
for MODE in three_state three_state_srcocc no_lockback lockback_nonconflict; do
    MODE=$MODE bash scripts/run_state_ablation.sh demo/rider; done
python evaluation/evaluate_state_ablation.py --items demo/rider
python evaluation/summarize_state_ablation.py --items demo/rider --output demo/rider/state_ablation
```

## 6. Notes

* Coordinates: GLBs are glTF (Y up); internally Blender/TRELLIS Z up, `(x, y, z) -> (x, -z, y)`, normalized to
  `[-0.5, 0.5]^3` with the `scale/offset` stored in `_easy3e_work_*/render/transforms.json`.
* The SLAT texture of TRELLIS is re-baked for the whole asset, so colors outside the edit can shift slightly
  even though the latents there are unchanged; measure preservation against the decoded source (see
  `evaluate_state_ablation.py`).
* Runs are seeded (`--seed`); results on other GPUs / library versions may still differ slightly.

## Acknowledgements and licenses

Built on [Easy3E](https://github.com/linchanyi/Easy3E) (MIT, `Easy3E/LICENSE`) and
[TRELLIS](https://github.com/microsoft/TRELLIS) (MIT). Rendering uses [Blender](https://www.blender.org/) (GPL).
