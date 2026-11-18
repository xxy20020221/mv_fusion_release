#!/usr/bin/env python3
"""Prepare multi-view composition targets for Easy3E mv_fusion from two GLBs and a prompt.

Pipeline
  1. Render object A and object B separately from the candidate views with Blender
     (lighting / ortho camera / color pipeline of compose_prep/render_views_blender.py).
  2. VLM plan (one JSON call): name + describe both objects, pick the anchor (the rigid
     base that keeps its pose) and its front, the K most informative views (most important
     first) and the expected appearance of the composition in each of those views.
  3. Nano Banana composition, progressive for multi-view consistency:
       primary view : edit the anchor's render at that view, adding the other object; it is
                      reviewed alone by the VLM (and regenerated) before anything is derived
                      from it.
       3D lift      : the accepted primary is reconstructed with TRELLIS.2, aligned to the
                      anchor's frame (ICP + silhouette at the primary camera) and rendered from
                      every selected camera. These layout guides carry the 3D placement that 2D
                      editors and VLM judges get wrong (e.g. they put headphone earcups at the
                      image edges in every view). Disable with --no_lift.
       other views  : edit the anchor's render at that view, conditioned on its layout guide
                      and every composite accepted so far (nearest view first).
     Editing the anchor's own render keeps each target aligned with its camera. Every
     generation passes pixel gates first (anchor silhouette still covered -> camera and anchor
     unchanged; not a copy of a reference image; added object overlaps its layout guide),
     else it is regenerated at once.
  4. VLM QA: per-view fidelity + cross-view consistency. Flagged views are regenerated
     with the accepted views as references plus the reviewer's feedback, for up to
     --max_rounds rounds. A regenerated view replaces the current one only if it wins a
     pairwise VLM comparison against it (absolute scores are too noisy for that decision),
     so the latest set is the cumulative best.
  5. Write target_<label>.png, source_<label>.png and compose_views.json into --item_dir.
     The source is the main (anchor) object alone from the same camera, i.e. the exact canvas the
     target was edited from (RGBA, transparent background). Renders, the plan and VLM logs stay in
     --work_dir, generations / QA / preview of each run in work_dir/runs/.

Without --prompt and without an existing <item_dir>/prompt.txt, a composition instruction is
written by the VLM from the renders of both assets and saved as <item_dir>/prompt.txt.
For items whose targets already exist, --sources_only adds the sources (no API calls).

Views are expressed in the anchor GLB's own frame (Blender Z-up after glTF import, unit
normalized); the per-view camera_npy in compose_views.json lets the anchor (or a later combined
object.glb that keeps the anchor's pose) be re-rendered from exactly these cameras.

Example
  python compose_prep/prepare_composition_views.py \
      --obj_a <case>/input_1_helmet.glb --obj_b <case>/input_2_headphones.glb \
      --item_dir <case>                                  # prompt: <item_dir>/prompt.txt
  python compose_prep/prepare_composition_views.py --item_dir <case> --sources_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import lift3d  # noqa: E402
import vlm_client as vlm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = Path(__file__).resolve().parent / "render_views_blender.py"
DEFAULT_BLENDER = PROJECT_ROOT / "Easy3E" / "tmp" / "blender-4.0.0-linux-x64" / "blender"
# 8-view preset (same as blender_scripts/render_glb_mv_high_detail.py): (azimuth, elevation) in degrees.
DEFAULT_VIEWS = "30,30;120,30;210,30;300,30;60,-30;150,-30;240,-30;330,-30"
SCORE_KEYS = ("instruction_following", "anchor_preserved", "viewpoint_match", "other_identity", "image_quality")


@dataclass
class View:
    cid: str
    azimuth: float
    elevation: float


@dataclass
class Context:
    args: argparse.Namespace
    prompt: str
    views: dict[str, View]
    render_dirs: dict[str, Path]          # "a" / "b" -> render dir
    work: Path
    plan: dict[str, Any] = field(default_factory=dict)
    anchor: str = "a"
    other_sheet: Path | None = None
    run_dir: Path | None = None       # per-run generations / qa / preview
    guides: dict[str, Path] = field(default_factory=dict)   # cid -> 3D layout guide render (for metrics)
    guide_visuals: dict[str, Path] = field(default_factory=dict)   # cid -> grayscale guide shown to the editor
    lift: dict[str, Any] = field(default_factory=dict)
    glbs: dict[str, Path] = field(default_factory=dict)
    prompt_info: dict[str, Any] = field(default_factory=dict)   # where the instruction came from
    resume: Any = None   # ResumeCache of an earlier run, or None
    guide_gate: dict[str, Any] = field(default_factory=dict)   # self-calibrated guide-IoU gate

    @property
    def other(self) -> str:
        return "b" if self.anchor == "a" else "a"

    def name(self, key: str) -> str:
        return self.plan[f"object_{key}_name"]

    def desc(self, key: str) -> str:
        return self.plan[f"object_{key}_description"]

    def render(self, key: str, cid: str) -> Path:
        return self.render_dirs[key] / f"{cid}.png"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def parse_views(text: str) -> dict[str, View]:
    views = {}
    for i, item in enumerate(t for t in text.split(";") if t.strip()):
        az, el = (float(v) for v in item.split(","))
        views[f"{i:03d}"] = View(f"{i:03d}", az, el)
    return views


def renders_complete(out_dir: Path, views: dict[str, View], args: argparse.Namespace) -> bool:
    meta = out_dir / "shrink.json"
    if not meta.exists() or not all((out_dir / f"{cid}.png").exists() for cid in views):
        return False
    info = json.loads(meta.read_text())
    rendered = [(v["azimuth"], v["elevation"]) for v in info.get("views", [])]
    return (info.get("resolution") == args.resolution and info.get("ortho_scale") == args.ortho_scale
            and rendered == [(v.azimuth, v.elevation) for v in views.values()])


def render_objects(ctx: Context, glbs: dict[str, Path]) -> None:
    args = ctx.args
    procs = {}
    views_arg = ";".join(f"{v.azimuth},{v.elevation}" for v in ctx.views.values())
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    for key, glb in glbs.items():
        out_dir = ctx.render_dirs[key]
        if not args.force_render and renders_complete(out_dir, ctx.views, args):
            print(f"[render] {key}: cached in {out_dir}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(args.blender), "-b", "--factory-startup", "-P", str(RENDER_SCRIPT), "--",
               "--object", str(glb), "--output_dir", str(out_dir), "--views", views_arg,
               "--view_ids", ",".join(ctx.views), "--resolution", str(args.resolution),
               "--ortho_scale", str(args.ortho_scale), "--samples", str(args.samples)]
        if args.gpu:
            cmd.append("--gpu")
        log_path = ctx.work / "logs" / f"render_{key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[render] {key}: {glb.name} -> {out_dir} ({len(ctx.views)} views)", flush=True)
        procs[key] = (subprocess.Popen(cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT, env=env), log_path)
    for key, (proc, log_path) in procs.items():
        if proc.wait() != 0 or not renders_complete(ctx.render_dirs[key], ctx.views, args):
            raise RuntimeError(f"Blender render of object {key} failed; see {log_path}")


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def on_white(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.alpha_composite(image)
    return canvas.convert("RGB")


def pad_square(image: Image.Image) -> Image.Image:
    side = max(image.size)
    if image.size == (side, side):
        return image
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def labeled_row(images: list[Image.Image], header: str, tile: int = 512) -> Image.Image:
    bar = 44
    row = Image.new("RGB", (tile * len(images), tile + bar), (255, 255, 255))
    for i, image in enumerate(images):
        row.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (i * tile, bar))
    ImageDraw.Draw(row).text((10, 8), header, fill=(0, 0, 0), font=font(26))
    return row


def grid(images: list[Image.Image], cols: int, tile: int = 512) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))
    for i, image in enumerate(images):
        sheet.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), ((i % cols) * tile, (i // cols) * tile))
    return sheet


def view_direction(view: View) -> tuple[float, float, float]:
    az, el = math.radians(view.azimuth), math.radians(view.elevation)
    return (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))


def angular_distance(a: View, b: View) -> float:
    dot = sum(x * y for x, y in zip(view_direction(a), view_direction(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def spread_views(views: dict[str, View], count: int) -> list[str]:
    """Greedy farthest-point pick of `count` view ids (for reference sheets)."""
    ids = list(views)
    picked = [ids[0]]
    while len(picked) < min(count, len(ids)):
        picked.append(max((c for c in ids if c not in picked),
                          key=lambda c: min(angular_distance(views[c], views[p]) for p in picked)))
    return picked


def signed_azimuth_delta(target: View, ref: View) -> float:
    return (target.azimuth - ref.azimuth + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Step 2: VLM plan
# --------------------------------------------------------------------------- #
def plan_tiles(ctx: Context) -> list[Path]:
    """One image per candidate view: object A (left) and object B (right) rendered from that camera."""
    tiles = []
    for cid, view in ctx.views.items():
        tile = labeled_row([on_white(ctx.render("a", cid)), on_white(ctx.render("b", cid))],
                           f"view {cid}  azimuth={view.azimuth:.0f}  elevation={view.elevation:.0f}   "
                           f"{'(left: initial scene A, right: B alone)' if initial_scene(ctx) else '(left: A, right: B)'}")
        tile_path = ctx.work / "plan_tiles" / f"{cid}.png"
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        tile.save(tile_path)
        tiles.append(tile_path)
    return tiles


# Style references for generated instructions (prompts of existing composition cases).
PROMPT_EXAMPLES = [
    "Fit the headphones around the helmet. Widen the headband into an arch that follows the helmet crown, extend and "
    "angle the sliders so the two earcups sit just below the side rims, and compress the ear cushions where they touch "
    "the helmet. Preserve the rigid helmet shell and keep the chin strap clear.",
    "Pose the teddy bear holding the mug in front of its chest. Curl its right plush paw around the mug handle and bend "
    "its left forearm so the other paw cups the underside. Flatten the plush slightly at the contact points. Keep the "
    "mug upright and preserve the bear's face, ears, and body silhouette.",
]


def generate_prompt(ctx: Context) -> dict[str, Any]:
    """Write a composition instruction for two assets when the user did not provide one."""
    examples = "\n".join(f"- {e}" for e in PROMPT_EXAMPLES)
    prompt = f"""
Each of the {len(ctx.views)} images shows two separate 3D assets rendered from the same camera: LEFT = object A, RIGHT = object B (each normalized to unit size, so their relative scale is NOT shown). The header gives the camera azimuth and elevation.

Write ONE composition instruction that combines A and B into a single object in the way these two things most naturally go together (e.g. something worn, attached, held or placed on the other). It will drive image editing of the main object in several views, followed by 3D reconstruction, so:
- Identify the main object (the one the other is fitted onto; it keeps its pose) and the added object.
- Say where the added object goes in the main object's own terms (front/back/left/right/top), at a realistic size relative to it.
- Describe how the added object is posed, bent or adjusted to fit, the contact points, and one or two small local deformations at the contacts (compression, flattening, pinching).
- End with what must be preserved (identity, silhouette, key features of both objects).
- Only these two objects; no new parts, colors or style changes. The result must be visible and plausible from every side.
- Imperative English, 3-4 sentences, 40-70 words, in the style of these examples:
{examples}

Also give short lowercase names for A and B and the main object ("a" or "b").
""".strip()
    schema = {"type": "object", "properties": {
        "object_a_name": {"type": "string"}, "object_b_name": {"type": "string"},
        "main_object": {"type": "string", "enum": ["a", "b"]}, "prompt": {"type": "string"}},
        "required": ["object_a_name", "object_b_name", "main_object", "prompt"]}

    def validate(data: dict[str, Any]) -> None:
        data["main_object"] = str(data.get("main_object", "")).strip().lower()
        words = len(str(data.get("prompt", "")).split())
        if data["main_object"] not in ("a", "b") or not 20 <= words <= 120:
            raise ValueError(f"invalid generated prompt ({words} words, main={data['main_object']!r})")
        data["prompt"] = " ".join(str(data["prompt"]).split())

    return vlm.generate_json(prompt, plan_tiles(ctx), ctx.args.vlm_models, "prompt generation", schema=schema,
                             validate=validate, max_side=1024)


def plan_composition(ctx: Context) -> dict[str, Any]:
    plan_path = ctx.work / "plan.json"
    if plan_path.exists() and not ctx.args.force_plan:
        print(f"[plan] cached: {plan_path}", flush=True)
        return json.loads(plan_path.read_text())

    tiles = plan_tiles(ctx)
    k = ctx.args.num_views
    ids = list(ctx.views)
    scene_note = ("NOTE: object A is not a single asset but the initial scene: the main object with object B already "
                  "placed in an initial (not yet fitted) position; RIGHT shows that B part alone. The instruction edits "
                  "this scene, so the anchor is \"a\" and the views/layouts describe the scene after the edit. Name A after "
                  "its main object (without B).\n" if initial_scene(ctx) else "")
    prompt = f"""
You are planning multi-view image edits that compose two separate 3D assets into one object. The edited views will drive a multi-view 3D editing method, so together they must pin down the 3D composition.

Each of the {len(ids)} images is one candidate camera view. In every image, LEFT = object A and RIGHT = object B, rendered separately (each normalized to unit size) from the same camera. The header gives the view id, azimuth and elevation. Azimuth is the camera's orbit angle around the vertical axis; elevation > 0 looks from above, < 0 from below.

Composition instruction: "{ctx.prompt}"

Tasks:
1. Name A and B (1-3 words each) and describe each precisely (shape, parts, colors, materials) so an image editor can keep its identity.
{scene_note}2. anchor: the object that stays rigid and keeps its pose, i.e. the base the other object is fitted onto or around (follow the instruction, e.g. the object it says to preserve). The other object may be moved, re-oriented and deformed.
3. anchor_front_view: the view id that looks most directly at the anchor's front (its natural facing direction, e.g. the face opening of a helmet), or "none" if it has no clear front. Use it to reason about left/right/front/back of the anchor in every view.
4. composition_description: describe the final combined object concretely in the anchor's own terms (front/back/left/right/top): where the added object sits, contact points, required deformations, what must stay clear.
4b. anchor_changes: the changes to the ANCHOR itself that the instruction explicitly requires (e.g. re-posed limbs, a dent, bent cables), or "none" if the anchor stays completely rigid.
5. interaction_regions: the regions of the anchor where the composition happens.
6. selected_views: exactly {k} distinct view ids from {ids}, most important first. Views are defined by the ANCHOR's pose, which will not change; the other object's pose in its render does not matter because it will be re-posed.
   - The first (primary) view must make the composition least ambiguous to draw: the anchor's orientation obvious and all main contact points visible (typically the anchor's front or a slight three-quarter of it). It is generated first and every other view is derived from it.
   - Together the views must cover every interaction region (e.g. both sides for a symmetric attachment) from complementary directions; avoid near-duplicate viewpoints.
   - Avoid views where the added object would be mostly hidden behind the anchor, or where the anchor is only seen from a confusing direction (e.g. its inside or underside) unless that region is essential.
   - Image editors handle eye-level and elevated views (elevation >= 0) far more reliably than views from below: use a view from below only if an interaction region is visible from nowhere else.
7. view_reasons: one sentence per selected view id.
8. view_layouts: for each selected view id, describe how the final composition must look in THAT image, in image terms: which side of the anchor faces the camera, where each visible part of the added object appears (image left/right/top/bottom, in front of or behind the anchor) and which parts are hidden. Reason carefully in 3D: views from opposite sides swap image left and right.
""".strip()
    schema = {
        "type": "object",
        "properties": {
            "object_a_name": {"type": "string"},
            "object_a_description": {"type": "string"},
            "object_b_name": {"type": "string"},
            "object_b_description": {"type": "string"},
            "anchor": {"type": "string", "enum": ["a", "b"]},
            "anchor_front_view": {"type": "string", "enum": [*ids, "none"]},
            "composition_description": {"type": "string"},
            "anchor_changes": {"type": "string"},
            "interaction_regions": {"type": "array", "items": {"type": "string"}},
            "selected_views": {"type": "array", "items": {"type": "string", "enum": ids}, "minItems": k, "maxItems": k},
            "view_reasons": {"type": "object", "additionalProperties": {"type": "string"}},
            "view_layouts": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["object_a_name", "object_a_description", "object_b_name", "object_b_description",
                     "anchor", "anchor_front_view", "composition_description", "anchor_changes", "interaction_regions",
                     "selected_views", "view_reasons", "view_layouts"],
    }

    def validate(data: dict[str, Any]) -> None:
        missing = [key for key in schema["required"] if key not in data]
        if missing:
            raise ValueError(f"plan missing keys {missing}")
        data["anchor"] = str(data["anchor"]).strip().lower()
        if data["anchor"] not in ("a", "b"):
            raise ValueError(f"invalid anchor {data['anchor']!r}")
        selected = [normalize_id(v) for v in data["selected_views"]]
        if len(selected) != k or len(set(selected)) != k or any(v not in ctx.views for v in selected):
            raise ValueError(f"invalid selected_views {data['selected_views']!r}")
        data["selected_views"] = selected
        data["anchor_front_view"] = normalize_id(data["anchor_front_view"])
        data["view_reasons"] = {normalize_id(key): str(val) for key, val in dict(data["view_reasons"]).items()}
        data["view_layouts"] = {normalize_id(key): str(val) for key, val in dict(data["view_layouts"]).items()}
        if any(v not in data["view_layouts"] for v in selected):
            raise ValueError(f"view_layouts must cover {selected}, got {sorted(data['view_layouts'])}")

    plan = vlm.generate_json(prompt, tiles, ctx.args.vlm_models, "plan", schema=schema, validate=validate, max_side=1024)
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    return plan


def normalize_id(value: Any) -> str:
    text = str(value).strip()
    for prefix in ("view_", "view ", "view"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    return f"{int(text):03d}" if text.isdigit() else text


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #
def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def image_digest(image: Any) -> str:
    image = image if isinstance(image, Image.Image) else Image.open(image)
    rgb = image.convert("RGB")
    return hashlib.sha1(rgb.tobytes() + repr(rgb.size).encode()).hexdigest()


class ResumeCache:
    """Results of an earlier (e.g. interrupted) run of the same item. A step whose request is identical to
    one that run already made gets the stored result instead of a new API call or reconstruction:
      generation : same view/tag, same prompt text and same input images
      review     : same tag and byte-identical composites (and same prompt when it was recorded)
      3D lift    : reconstructed from a byte-identical primary composite
    Reused files are copied into the new run, so every run directory stays self-contained."""

    def __init__(self, run_dir: Path):
        self.run = run_dir
        self.hashes = {file_sha1(p) for p in run_dir.rglob("*.png")}
        print(f"[resume] reusing identical requests from {run_dir}", flush=True)

    def generation(self, label: str, tag: str, prompt: str, digests: list[str],
                   refs: list[Path]) -> list[tuple[Path, str]]:
        """Raw candidates (and the model that made them) of an identical generation request."""
        gen = self.run / "generations"
        prompt_file = gen / f"view{label}_{tag}.prompt.txt"
        if not prompt_file.is_file() or prompt_file.read_text(encoding="utf-8") != prompt:
            return []
        sidecar = gen / f"view{label}_{tag}.inputs.json"
        models: dict[str, str] = {}
        if sidecar.is_file():
            info = json.loads(sidecar.read_text())
            if info.get("inputs") != digests:
                return []
            models = info.get("models", {})
        elif not all(file_sha1(r) in self.hashes for r in refs):
            return []  # older runs did not record input hashes: require the references to come from that run
        out = []
        while (raw := gen / f"view{label}_{tag}_g{len(out)}_raw.png").is_file():
            out.append((raw, models.get(str(len(out)), "unknown (reused)")))
        return out

    def review(self, tag: str, composites: dict[str, Path], prompt: str) -> dict[str, Any] | None:
        record_path = self.run / "qa" / f"{tag}.json"
        if not record_path.is_file():
            return None
        record = json.loads(record_path.read_text())
        if record.get("prompt", prompt) != prompt or set(record["composites"]) != set(composites):
            return None
        for label, path in composites.items():
            old = Path(record["composites"][label])
            if not old.is_file() or file_sha1(old) != file_sha1(path):
                return None
        return record["qa"]

    def lift(self, primary: Path) -> Path | None:
        lift_dir = self.run / "lift"
        summary = lift_dir / "trellis2" / "summary.json"
        if not (summary.is_file() and (lift_dir / "recon_to_anchor.npy").is_file()):
            return None
        source = Path(json.loads(summary.read_text()).get("input_image", ""))
        return lift_dir if source.is_file() and file_sha1(source) == file_sha1(primary) else None


# --------------------------------------------------------------------------- #
# Step 3: composition
# --------------------------------------------------------------------------- #
def anchor_changes(ctx: Context) -> str | None:
    """Instruction-required changes to the anchor (re-posed limbs, dents...), None if it stays rigid."""
    text = str(ctx.plan.get("anchor_changes", "none")).strip()
    return None if text.lower().rstrip(".") in ("", "none", "n/a", "no") else text


def initial_scene(ctx: Context) -> bool:
    """--initial_scene: object A is a pre-composed scene that already contains object B at an initial
    placement; the canvas / source is that scene and the edit adjusts B (and A as required) in it."""
    return bool(getattr(ctx.args, "initial_scene", False))


def canvas_text(ctx: Context) -> str:
    """How the canvas (render of A from a selected camera) is described to the image editor and judges."""
    anchor, other = ctx.name(ctx.anchor), ctx.name(ctx.other)
    if initial_scene(ctx):
        return (f"the initial scene before the edit - the {anchor} ({ctx.desc(ctx.anchor)}) with the {other} already in "
                f"its initial placement")
    return f"the {anchor} ({ctx.desc(ctx.anchor)}), rendered alone"


def hard_rules(ctx: Context) -> str:
    anchor, other = ctx.name(ctx.anchor), ctx.name(ctx.other)
    changes = anchor_changes(ctx)
    anchor_rule = (f"The {anchor} stays rigid: same position, silhouette, colors and materials as in Image 1; only parts "
                   f"covered by the {other} may change." if changes is None else
                   f"The {anchor} keeps its position, size, design, colors and materials as in Image 1. Change its shape "
                   f"only as the instruction requires ({changes}); everything else stays exactly as in Image 1.")
    scene_rule = (f"\n- The {other} is already in Image 1: edit that {other} in place (keep its size unless the "
                  f"instruction says otherwise); never add a second {other}." if initial_scene(ctx) else "")
    return f"""Rules:
- Camera: keep the viewpoint, framing and scale of Image 1 exactly; do not rotate, zoom or re-center the view.
- {anchor_rule}
- The {other} keeps its identity from the reference renders (same parts, proportions, colors, materials, textures). Move, rotate, scale and deform it only as the instruction requires, at a realistic size relative to the {anchor}.{scene_rule}
- Contacts are physically plausible: touching surfaces meet with no gaps and no interpenetration.
- Same style as Image 1: clean 3D studio render, soft even lighting, pure white background, no ground shadow, no text, no extra objects; the whole composition stays fully inside the frame.
Return only the edited image."""


def compose_inputs(ctx: Context, view: View, refs: list[tuple[str, View, Path]],
                   feedback: str | None) -> tuple[list[Any], str]:
    """Images (numbered in order) and prompt for composing one view."""
    anchor, other = ctx.name(ctx.anchor), ctx.name(ctx.other)
    layout = ctx.plan["view_layouts"].get(view.cid, "")
    guide = ctx.guide_visuals.get(view.cid)
    images: list[Any] = [on_white(ctx.render(ctx.anchor, view.cid))]
    lines = [f"Image 1: {canvas_text(ctx)}, rendered from the target camera "
             f"(azimuth {view.azimuth:.0f} deg, elevation {view.elevation:.0f} deg). This is the canvas to edit."]
    if guide is not None:
        images.append(on_white(guide))
        lines.append(f"Image 2: 3D layout guide - a grayscale clay render of a rough 3D reconstruction of the approved "
                     f"composition, from exactly the same camera. It shows where the {other} must be in this view: "
                     f"position, orientation, size, visible parts and occlusion by the {anchor}. Parts of the {other} "
                     f"outside the {anchor}'s outline are tinted cyan. It carries geometry only: its gray/cyan colors and "
                     f"simplified surfaces are NOT the real appearance.")
    images.append(ctx.other_sheet)
    lines.append(f"Image {len(images)}: reference renders of the {other} ({ctx.desc(ctx.other)}) from several "
                 f"viewpoints, showing its complete 3D shape and true appearance. Its pose in these renders is arbitrary.")
    ref_ids = []
    for _, ref_view, ref_path in refs:
        images.append(ref_path)
        ref_ids.append(len(images))
        lines.append(
            f"Image {len(images)}: the approved composition seen from a DIFFERENT camera, azimuth {ref_view.azimuth:.0f} deg, "
            f"elevation {ref_view.elevation:.0f} deg (the target camera is orbited {signed_azimuth_delta(view, ref_view):+.0f} deg "
            f"around the vertical axis, counter-clockwise seen from above, and "
            f"{view.elevation - ref_view.elevation:+.0f} deg in elevation relative to it).")
    span = (f"Image {ref_ids[0]}" if len(ref_ids) == 1 else f"Images {ref_ids[0]}-{ref_ids[-1]}") if ref_ids else ""

    if not refs and guide is None:
        head = "Compose two 3D-rendered objects into one object, seen from a fixed camera."
        task = (f"Instruction: {ctx.prompt}\nIntended result: {ctx.plan['composition_description']}\n"
                f"Expected appearance from this camera: {layout}\n\n"
                f"Edit Image 1 to show the result of the instruction from exactly the camera of Image 1.")
    else:
        head = ("Compose two 3D-rendered objects into one object, seen from a fixed camera. The composition is already "
                "decided; show the SAME composed object from this camera.")
        task = (f"Instruction (already applied): {ctx.prompt}\nExpected appearance from this camera: {layout}\n\n")
        if guide is not None:
            task += (f"Edit Image 1: {'make the ' + other + ' (already in Image 1) look' if initial_scene(ctx) else 'add the ' + other} exactly where Image 2 places it - same position, orientation, size, "
                     f"visible parts and occlusion by the {anchor} - with the true design, colors and materials of the "
                     f"{other}{' from ' + span + ' and' if span else ' from'} its reference renders (never the gray "
                     f"or cyan of Image 2). If the expected appearance text disagrees with Image 2, follow Image 2.\n"
                     f"- Keep the {anchor} exactly as in Image 1; do not copy the {anchor}'s look from Image 2.")
        else:
            task += (f"Edit Image 1 to show exactly the composed object of {span} from the camera of Image 1.\n"
                     f"- The {other} must match {span}: same attachment points on the {anchor}, same orientation and "
                     f"deformation, same proportions and size relative to the {anchor}, same colors and details. Only the "
                     f"viewpoint changes; parts visible from this camera but hidden in {span} must be completed "
                     f"consistently with the reference renders.")
        if span:
            task += (f"\n- Do NOT return or copy {span}: they are seen from other cameras. The {anchor} in the output "
                     f"must have exactly the pose it has in Image 1.")
    text = f"{head}\n\n" + "\n".join(lines) + f"\n\n{task}\n{hard_rules(ctx)}"
    if feedback:
        text += f"\n\nA previous attempt at this view was rejected by a reviewer. Fix these problems: {feedback}"
    return images, text


def _mask_from_render(path: Path, size: int) -> np.ndarray:
    alpha = Image.open(path).convert("RGBA").getchannel("A").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(alpha) > 127


def _foreground(image: Image.Image, size: int) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).astype(np.int16)
    return (pixels < 235).any(axis=-1)


def _gray(image: Image.Image, size: int = 128) -> np.ndarray:
    return np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0


def gate_metrics(ctx: Context, view: View, image: Image.Image, refs: list[tuple[str, View, Path]]) -> dict[str, float]:
    """Cheap pixel checks run before any VLM review.

    anchor_coverage: share of the anchor's rendered silhouette that is still foreground in
        the composite; drops when the editor moves/rescales the anchor or changes viewpoint.
    min_ref_distance: mean gray difference to the reference composites; ~0 means the editor
        returned a reference instead of editing the canvas.
    """
    mask = _mask_from_render(ctx.render(ctx.anchor, view.cid), 256)
    coverage = float((mask & _foreground(image, 256)).sum() / max(mask.sum(), 1))
    gray = _gray(image)
    # Copy detection, relative to how different the two cameras make the anchor itself look: for a
    # rotationally symmetric anchor (a balloon) other azimuths legitimately look almost identical.
    anchor_gray = _gray(on_white(ctx.render(ctx.anchor, view.cid)))
    checks = [(float(np.abs(gray - _gray(Image.open(p))).mean()),
               min(ctx.args.min_ref_distance,
                   0.35 * float(np.abs(anchor_gray - _gray(on_white(ctx.render(ctx.anchor, v.cid)))).mean())))
              for _, v, p in refs]
    distance, distance_min = min(checks, key=lambda c: c[0] - c[1], default=(1.0, ctx.args.min_ref_distance))
    metrics = {"anchor_coverage": coverage, "min_ref_distance": distance, "ref_distance_min": distance_min,
               "anchor_coverage_min": ctx.args.min_anchor_coverage if anchor_changes(ctx) is None
               else ctx.args.min_anchor_coverage_deform}
    guide = ctx.guides.get(view.cid)
    if guide is not None and ctx.guide_gate.get("enabled"):
        # Added-object agreement with the 3D layout guide (VLM-free orientation/placement check). Only
        # where the guide shows enough of the added object outside the anchor's outline to be meaningful.
        anchor_render = ctx.render(ctx.anchor, view.cid)
        guide_region = lift3d.other_region(guide, anchor_render)
        metrics["guide_region_px"] = int(guide_region.sum())
        if guide_region.sum() >= ctx.args.min_guide_region_px:
            scratch = ctx.run_dir / "generations" / "_gate_tmp.png"
            image.save(scratch)
            metrics["guide_iou"] = lift3d.iou(lift3d.other_region(scratch, anchor_render), guide_region)
            metrics["guide_iou_min"] = ctx.guide_gate["threshold"]
    return metrics


def gates_pass(args: argparse.Namespace, metrics: dict[str, float]) -> bool:
    return (metrics["anchor_coverage"] >= metrics.get("anchor_coverage_min", args.min_anchor_coverage)
            and metrics["min_ref_distance"] >= metrics.get("ref_distance_min", args.min_ref_distance)
            and metrics.get("guide_iou", 1.0) >= metrics.get("guide_iou_min", args.min_guide_iou))


def reframe_to_anchor(image: Image.Image, anchor_render: Path, size: int = 512) -> tuple[Image.Image, dict[str, Any]]:
    """Undo editor re-framing (Nano Banana sometimes re-centers / rescales the canvas).

    Multi-scale template matching of the anchor's appearance finds the similarity that maps the anchor
    render into the generated image; the image is then warped back into the canvas frame so the target
    stays aligned with its camera. Skipped when the warp would push content out of the frame (the
    canvas is too tight for the composition: use a larger --ortho_scale).
    """
    comp_rgb = image.convert("RGB")
    comp = np.asarray(comp_rgb.convert("L").resize((size, size), Image.Resampling.LANCZOS)).astype(np.float32)
    anchor_rgba = Image.open(anchor_render).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    alpha = np.asarray(anchor_rgba.getchannel("A")) > 127
    anchor = np.asarray(on_white(anchor_render).convert("L").resize((size, size), Image.Resampling.LANCZOS)).astype(np.float32)
    ys, xs = np.nonzero(alpha)
    pad = 4
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, size)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, size)
    template = anchor[y0:y1, x0:x1]

    def match(scales: np.ndarray) -> tuple[float, float, tuple[int, int]]:
        best = (-1.0, 1.0, (x0, y0))
        for s in scales:
            t = cv2.resize(template, None, fx=float(s), fy=float(s), interpolation=cv2.INTER_AREA)
            if t.shape[0] >= size or t.shape[1] >= size or min(t.shape) < 16:
                continue
            _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(comp, t, cv2.TM_CCOEFF_NORMED))
            if score > best[0]:
                best = (float(score), float(s), loc)
        return best

    score, s, _ = match(np.arange(0.6, 1.66, 0.02))
    score, s, (lx, ly) = match(np.arange(s - 0.02, s + 0.021, 0.004))
    # anchor-render pixel a maps to generated-image pixel s * a + t (both in `size` units)
    tx, ty = lx - s * x0, ly - s * y0
    info = {"reframe_scale": s, "reframe_shift": [float(tx), float(ty)], "reframe_score": score,
            "reframed": False, "reframe_crop": 0.0}
    if score < 0.5 or (abs(s - 1.0) < 0.01 and abs(tx) <= 1.5 and abs(ty) <= 1.5):
        return comp_rgb, info

    fg = (np.asarray(comp_rgb.resize((size, size), Image.Resampling.BILINEAR)).astype(np.int16) < 235).any(axis=-1)
    qy, qx = np.nonzero(fg)
    ax, ay = (qx - tx) / s, (qy - ty) / s
    info["reframe_crop"] = float(((ax < 0) | (ax >= size) | (ay < 0) | (ay >= size)).mean()) if len(qx) else 0.0
    if info["reframe_crop"] > 0.01:
        return comp_rgb, info

    width = comp_rgb.width  # square; warp at native resolution
    k = width / size
    matrix = np.array([[s, 0.0, k * tx], [0.0, s, k * ty]], dtype=np.float64)
    warped = cv2.warpAffine(np.asarray(comp_rgb), matrix, (width, width), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    info["reframed"] = True
    return Image.fromarray(warped), info


def compose_view(ctx: Context, label: str, view: View, refs: list[tuple[str, View, Path]],
                 feedback: str | None, tag: str) -> tuple[Path, str, dict[str, float]]:
    args = ctx.args
    images, prompt = compose_inputs(ctx, view, refs, feedback)
    out_dir = ctx.run_dir / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"view{label}_{tag}.prompt.txt").write_text(prompt, encoding="utf-8")
    digests = [image_digest(im) for im in images]
    sidecar = {"inputs": digests, "models": {}}
    cached = ctx.resume.generation(label, tag, prompt, digests, [p for _, _, p in refs]) if ctx.resume else []

    best = None
    for attempt in range(args.gate_retries + 1):
        where = f"(cid {view.cid}, az={view.azimuth:.0f}, el={view.elevation:.0f}) refs={[lbl for lbl, _, _ in refs]} {tag}"
        if attempt < len(cached):
            raw_path, model = cached[attempt]
            print(f"[compose] view {label} {where} attempt {attempt}: reused {raw_path.name} from {ctx.resume.run.name}",
                  flush=True)
            image = Image.open(raw_path).convert("RGB")
        else:
            print(f"[compose] view {label} {where} attempt {attempt}", flush=True)
            image, model = vlm.generate_image(prompt, images, args.image_models, f"compose view {label} {tag}")
            image = pad_square(image)
        sidecar["models"][str(attempt)] = model
        (out_dir / f"view{label}_{tag}.inputs.json").write_text(json.dumps(sidecar, indent=1))
        image.save(out_dir / f"view{label}_{tag}_g{attempt}_raw.png")
        image, reframe = reframe_to_anchor(image, ctx.render(ctx.anchor, view.cid))
        out = out_dir / f"view{label}_{tag}_g{attempt}.png"
        image.save(out)
        metrics = {**gate_metrics(ctx, view, image, refs), **reframe}
        passed = gates_pass(args, metrics)
        guide_txt = f" guide_iou={metrics['guide_iou']:.3f}" if "guide_iou" in metrics else ""
        reframe_txt = (f" reframed(s={reframe['reframe_scale']:.3f})" if reframe["reframed"] else
                       f" not-reframed(s={reframe['reframe_scale']:.3f}, crop={reframe['reframe_crop']:.3f})"
                       if abs(reframe["reframe_scale"] - 1) >= 0.01 else "")
        print(f"[gate] view {label} {tag} g{attempt}: coverage={metrics['anchor_coverage']:.3f} "
              f"ref_distance={metrics['min_ref_distance']:.3f}{guide_txt}{reframe_txt} -> {'pass' if passed else 'reject'}",
              flush=True)
        key = (passed, metrics.get("guide_iou", 0.0), metrics["anchor_coverage"], metrics["min_ref_distance"])
        if best is None or key > best[0]:
            best = (key, out, model, metrics)
        if passed:
            break
    _, out, model, metrics = best
    return out, model, metrics


# --------------------------------------------------------------------------- #
# Step 4: QA
# --------------------------------------------------------------------------- #
def _normalize_scores(raw: Any) -> dict[str, Any]:
    """Map a reviewer's per-view entry onto SCORE_KEYS, tolerating nested 'scores' dicts and
    shortened key names (e.g. 'viewpoint' for 'viewpoint_match')."""
    if not isinstance(raw, dict):
        raise ValueError(f"per-view entry is not an object: {raw!r}")
    flat = {**raw, **raw["scores"]} if isinstance(raw.get("scores"), dict) else dict(raw)
    scores: dict[str, Any] = {"issues": str(flat.get("issues", ""))}
    for key in SCORE_KEYS:
        prefix = key.split("_")[0]
        match = key if key in flat else next((k for k in flat if str(k).lower().startswith(prefix)), None)
        if match is not None:
            scores[key] = int(flat[match])
    return scores


def qa_check(ctx: Context, labels: dict[str, View], composites: dict[str, Path], tag: str,
             placement_verified: bool = False) -> dict[str, Any]:
    anchor, other = ctx.name(ctx.anchor), ctx.name(ctx.other)
    images: list[Any] = []
    lines = []
    for label, view in labels.items():
        n = len(images)
        lines.append(f"Image {n + 1}: view {label} - {'the initial scene before the edit' if initial_scene(ctx) else 'the ' + anchor + ' alone'} "
                     f"(camera azimuth {view.azimuth:.0f} deg, elevation {view.elevation:.0f} deg).")
        images.append(on_white(ctx.render(ctx.anchor, view.cid)))
        images.append(composites[label])
        lines.append(f"Image {len(images)}: view {label} - the generated composition from the same camera. "
                     f"Expected appearance: {ctx.plan['view_layouts'].get(view.cid, '')}")
    lines.append(f"Image {len(images) + 1}: reference renders of the {other} alone.")
    images.append(ctx.other_sheet)

    single = len(labels) == 1
    changes = anchor_changes(ctx)
    verb = f"(already present in the initial scene) is re-fitted" if initial_scene(ctx) else "is added and fitted"
    anchor_note = (f"The {anchor} must stay rigid and unchanged; the {other} {verb} to it." if changes is None else
                   f"The {other} {verb} to the {anchor}; the {anchor} only changes as required: {changes}.")
    if initial_scene(ctx):
        anchor_note += f" There must be exactly one {other}."
    anchor_exception = "" if changes is None else f", apart from the required changes ({changes}), which must be present"
    verified = placement_verified
    placement_rule = (
        f"the composition matches the instruction and the intended result. The position and orientation of the {other} "
        f"in every view were already verified against a 3D reconstruction; do NOT judge or request changes to its "
        f"placement or orientation, judge only fit details (contact, compression, gaps) and completeness."
        if verified else
        f"the composition matches the instruction, the intended result and the expected appearance for that view "
        f"(check the 3D orientation of the {other} relative to the {anchor}).")
    consistency_task = (
        "Set cross_view_consistency to 10 and inconsistencies to [] (only one view is reviewed)." if single else
        "Then give cross_view_consistency from 0 to 10: do all compositions show the same 3D object (same attachment "
        "points, orientation, deformation, size ratio, colors), as if rendered from different cameras? List concrete "
        "inconsistencies.")
    prompt = f"""
You are reviewing composition images that will be used for multi-view 3D editing, so together they must depict ONE consistent 3D object.

Instruction that was applied: "{ctx.prompt}"
Intended result: {ctx.plan['composition_description']}
{anchor_note}

Images, in order:
{chr(10).join(lines)}

Score every view from 0 to 10 (10 = perfect):
- instruction_following: {placement_rule}
- anchor_preserved: the {anchor} keeps the pose, silhouette, colors and framing of its render from the same view (it must not move, rotate, rescale or change design){anchor_exception}.
- viewpoint_match: the composition is seen from the same camera as the {anchor} render of that view.
- other_identity: the {other} keeps the design, colors and materials of its reference renders.
- image_quality: clean render, white background, no artifacts, nothing cropped, no extra objects.
{consistency_task}
regenerate: labels of the views that should be regenerated (wrong, or disagreeing with the majority); empty if every view is acceptable.
feedback: for each view to regenerate, a concrete instruction of what to fix in that view.
Be strict about 3D orientation and consistency, but do not penalise differences that are only due to visibility from a different camera.
""".strip()
    view_schema = {"type": "object", "properties": {**{k: {"type": "integer", "minimum": 0, "maximum": 10} for k in SCORE_KEYS},
                                                   "issues": {"type": "string"}},
                   "required": [*SCORE_KEYS, "issues"]}
    schema = {
        "type": "object",
        "properties": {
            "per_view": {"type": "object", "properties": {label: view_schema for label in labels}, "required": list(labels)},
            "cross_view_consistency": {"type": "integer", "minimum": 0, "maximum": 10},
            "inconsistencies": {"type": "array", "items": {"type": "string"}},
            "regenerate": {"type": "array", "items": {"type": "string", "enum": list(labels)}},
            "feedback": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["per_view", "cross_view_consistency", "inconsistencies", "regenerate", "feedback"],
    }

    def validate(data: dict[str, Any]) -> None:
        per_view = {str(k).replace("view", "").strip(): v for k, v in dict(data.get("per_view", {})).items()}
        if set(per_view) != set(labels):
            raise ValueError(f"per_view labels {sorted(per_view)} != {sorted(labels)}")
        for label, raw in list(per_view.items()):
            per_view[label] = scores = _normalize_scores(raw)
            missing = [key for key in SCORE_KEYS if key not in scores]
            if missing:
                raise ValueError(f"view {label} missing scores {missing}")
        data["per_view"] = per_view
        data["cross_view_consistency"] = 10 if single else int(data["cross_view_consistency"])
        data["regenerate"] = [str(v).replace("view", "").strip() for v in data.get("regenerate", [])]
        data["regenerate"] = [v for v in data["regenerate"] if v in labels]
        data["feedback"] = {str(k).replace("view", "").strip(): str(v) for k, v in dict(data.get("feedback", {})).items()}
        data.setdefault("inconsistencies", [])

    cached = ctx.resume.review(tag, composites, prompt) if ctx.resume else None
    if cached is not None:
        print(f"[qa] {tag}: reused review of identical composites from {ctx.resume.run.name}", flush=True)
        result = cached
    else:
        result = vlm.generate_json(prompt, images, ctx.args.vlm_models, f"qa {tag}", schema=schema,
                                   validate=validate, max_side=768)
    view_scores = {label: min(s[k] for k in SCORE_KEYS) for label, s in result["per_view"].items()}
    result["view_scores"] = view_scores
    result["set_score"] = min(min(view_scores.values()), result["cross_view_consistency"])
    result["mean_score"] = sum(view_scores.values()) / len(view_scores)
    result["accepted"] = result["set_score"] >= ctx.args.accept_score and not result["regenerate"]
    out = ctx.run_dir / "qa" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"composites": {k: str(v) for k, v in composites.items()}, "prompt": prompt, "qa": result},
                              indent=2, ensure_ascii=False))
    print(f"[qa] {tag}: view_scores={view_scores} consistency={result['cross_view_consistency']} "
          f"regenerate={result['regenerate']} accepted={result['accepted']}", flush=True)
    return result


def pairwise_keep_new(ctx: Context, label: str, view: View, old: Path, new: Path,
                      context: dict[str, tuple[View, Path]], tag: str) -> tuple[bool, str]:
    """Relative judgment: is the regenerated candidate a better view `label` of the SAME object shown
    in the other views? Absolute 0-10 scores from the reviewer are too noisy to decide replacements
    (the same image scored 3..8 across rounds); pairwise choices are far more stable. Candidate order
    is shuffled per call to avoid position bias."""
    anchor, other = ctx.name(ctx.anchor), ctx.name(ctx.other)
    new_first = random.Random(f"{label}-{tag}").random() < 0.5
    candidates = [new, old] if new_first else [old, new]
    images: list[Any] = [on_white(ctx.render(ctx.anchor, view.cid))]
    lines = [f"Image 1: {'the initial scene before the edit' if initial_scene(ctx) else 'the ' + anchor + ' alone'} from the "
             f"camera of view {label} (azimuth {view.azimuth:.0f} deg, elevation {view.elevation:.0f} deg)."]
    for lbl, (ctx_view, path) in context.items():
        images.append(path)
        lines.append(f"Image {len(images)}: context - the composed object from view {lbl} (azimuth "
                     f"{ctx_view.azimuth:.0f} deg, elevation {ctx_view.elevation:.0f} deg).")
    images.append(ctx.other_sheet)
    lines.append(f"Image {len(images)}: reference renders of the {other} alone.")
    images += candidates
    lines.append(f"Image {len(images) - 1}: candidate X for view {label}.")
    lines.append(f"Image {len(images)}: candidate Y for view {label}.")
    prompt = f"""
Two candidate images were generated for view {label} of a composed object (instruction: "{ctx.prompt}").
Expected appearance in view {label}: {ctx.plan['view_layouts'].get(view.cid, '')}

Images, in order:
{chr(10).join(lines)}

Which candidate is the better view {label} of the SAME composed object shown in the context images? Judge in this order:
1. Consistency with the context views: same attachment points, deformation and size of the {other} relative to the {anchor}, same design, colors and materials.
2. Fit details: contact without gaps or interpenetration, as the instruction requires.
3. The {anchor} is unchanged{'' if anchor_changes(ctx) is None else ' apart from the required changes (' + anchor_changes(ctx) + ')'} and seen from the same camera as Image 1.
4. Clean image without artifacts or extra parts.
""".strip()
    schema = {"type": "object", "properties": {"better": {"type": "string", "enum": ["X", "Y"]},
                                               "reason": {"type": "string"}}, "required": ["better", "reason"]}

    def validate(data: dict[str, Any]) -> None:
        data["better"] = str(data.get("better", "")).strip().upper()[:1]
        if data["better"] not in ("X", "Y"):
            raise ValueError(f"invalid choice {data.get('better')!r}")

    result = vlm.generate_json(prompt, images, ctx.args.vlm_models, f"pairwise view {label} {tag}",
                               schema=schema, validate=validate, max_side=768)
    keep_new = (result["better"] == "X") == new_first
    print(f"[pairwise] view {label} {tag}: keep {'new' if keep_new else 'old'} - {result.get('reason', '')[:200]}", flush=True)
    return keep_new, str(result.get("reason", ""))


def validate_guides(ctx: Context, labels: dict[str, View], primary: str, primary_path: Path) -> tuple[bool, dict[str, Any]]:
    """Single-view reconstruction is ambiguous in depth along the primary camera: e.g. a mug held in front
    of a chest can be reconstructed floating far in front of it, which looks identical from the primary
    view but puts the mug beside or above the anchor in other views. Ask the VLM a deliberately narrow,
    2D-visible question per guide: is the added object at the same place on the anchor as in the
    approved composition? Any failure disables the guides for this case."""
    anchor, other = ctx.name(ctx.anchor), ctx.name(ctx.other)
    pview = labels[primary]
    secondary = [v for lbl, v in labels.items() if lbl != primary]
    images: list[Any] = [primary_path]
    lines = [f"Image 1: the approved composition (camera azimuth {pview.azimuth:.0f} deg, elevation {pview.elevation:.0f} deg)."]
    for view in secondary:
        images.append(on_white(ctx.guides[view.cid]))
        lines.append(f"Image {len(images)}: render {view.cid} of a rough 3D reconstruction of that composition from another "
                     f"camera (azimuth {view.azimuth:.0f} deg, elevation {view.elevation:.0f} deg).")
    prompt = f"""
Composition: {ctx.plan['composition_description']}

Images, in order:
{chr(10).join(lines)}

The reconstruction renders have approximate shapes and colors and are seen from different cameras. For each render, decide whether the {other} is at the SAME place on the {anchor} as in Image 1 (same region of the {anchor}, touching it in the same way), or clearly somewhere else: floating away from the {anchor}, next to it instead of in front of it, at a different part of it, or missing. Ignore differences that only come from the viewpoint, occlusion, or approximate shapes and colors.
""".strip()
    ids = [v.cid for v in secondary]
    schema = {"type": "object", "properties": {"per_view": {"type": "object", "properties": {
        cid: {"type": "object", "properties": {"same_place": {"type": "boolean"}, "issue": {"type": "string"}},
              "required": ["same_place", "issue"]} for cid in ids}, "required": ids}}, "required": ["per_view"]}

    def validate(data: dict[str, Any]) -> None:
        per_view = {normalize_id(k): v for k, v in dict(data.get("per_view", {})).items()}
        if set(per_view) != set(ids) or any("same_place" not in v for v in per_view.values()):
            raise ValueError(f"guide check must cover {ids}, got {per_view}")
        for v in per_view.values():
            v["same_place"] = str(v["same_place"]).strip().lower() in ("true", "1", "yes")
        data["per_view"] = per_view

    # The verdict is stored with the lift; a reused lift (identical primary and guides) reuses it too.
    record = ctx.run_dir / "lift" / "validation.json"
    stored = json.loads(record.read_text()) if record.is_file() else None
    if stored is not None and stored.get("prompt") == prompt and set(stored["result"]["per_view"]) == set(ids):
        result = stored["result"]
        print("[lift] reused guide validation of the reused reconstruction", flush=True)
    else:
        try:
            result = vlm.generate_json(prompt, images, ctx.args.vlm_models, "guide validation", schema=schema,
                                       validate=validate, max_side=768)
        except RuntimeError as error:
            print(f"[lift] guide validation call failed ({error}); not using unvalidated guides", flush=True)
            return False, {"error": str(error)}
        record.write_text(json.dumps({"prompt": prompt, "result": result}, indent=2, ensure_ascii=False))
    ok = all(v["same_place"] for v in result["per_view"].values())
    for cid, v in result["per_view"].items():
        print(f"[lift] guide {cid}: {'ok' if v['same_place'] else 'REJECTED'} - {v.get('issue', '')[:160]}", flush=True)
    return ok, result


def calibrate_guide_gate(ctx: Context, primary_view: View, primary_path: Path) -> dict[str, Any]:
    """At the primary camera the guide renders the very composite it was reconstructed from, so the
    guide-vs-primary IoU is the best agreement this case can reach. Enable the guide gate only if it is
    high enough and the added object shows enough area outside the anchor's outline (an object held in
    front of the anchor lies inside it, where the metric is meaningless), and set the threshold relative
    to it."""
    args = ctx.args
    anchor_render = ctx.render(ctx.anchor, primary_view.cid)
    guide_region = lift3d.other_region(ctx.guides[primary_view.cid], anchor_render)
    self_iou = lift3d.iou(lift3d.other_region(primary_path, anchor_render), guide_region)
    enabled = self_iou >= args.min_guide_self_iou and guide_region.sum() >= args.min_guide_region_px
    gate = {"enabled": bool(enabled), "self_iou": self_iou, "primary_region_px": int(guide_region.sum()),
            "threshold": min(args.min_guide_iou, 0.6 * self_iou)}
    print(f"[lift] guide gate {'enabled' if enabled else 'disabled'}: primary self-IoU={self_iou:.3f}, "
          f"region={gate['primary_region_px']}px, threshold={gate['threshold']:.3f}", flush=True)
    return gate


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_consistent_views(ctx: Context, labels: dict[str, View]) -> tuple[dict[str, Path], dict[str, Any], list[dict[str, Any]]]:
    args = ctx.args
    primary = next(iter(labels))
    # Secondary views go nearest-to-primary first: smaller viewpoint changes are easier to extrapolate,
    # and the last view then sees every earlier composite as a reference.
    order = [primary] + sorted((lbl for lbl in labels if lbl != primary),
                               key=lambda lbl: angular_distance(labels[lbl], labels[primary]))
    composites: dict[str, Path] = {}
    models_used: dict[str, str] = {}
    gates: dict[str, dict[str, float]] = {}

    # The primary is reviewed on its own first: every other view is derived from it, so a wrong
    # orientation here would otherwise be propagated to all views.
    feedback = None
    primary_best = None
    for attempt in range(args.primary_rounds):
        tag = f"p{attempt}"
        path, model, metrics = compose_view(ctx, primary, labels[primary], [], feedback, tag)
        qa = qa_check(ctx, {primary: labels[primary]}, {primary: path}, f"primary_{tag}")
        key = (qa["set_score"], qa["mean_score"])
        if primary_best is None or key > primary_best[0]:
            primary_best = (key, path, model, metrics)
        if qa["accepted"]:
            break
        feedback = " ".join(t for t in (qa["feedback"].get(primary, ""), qa["per_view"][primary].get("issues", "")) if t)
    _, composites[primary], models_used[primary], gates[primary] = primary_best

    # Lift the accepted primary to 3D and render it from every selected camera: these layout guides give
    # the secondary views (and their checks) the 3D placement that 2D editing alone gets wrong.
    if args.lift:
        try:
            cids = [v.cid for v in labels.values()]
            prev_lift = ctx.resume.lift(composites[primary]) if ctx.resume else None
            ctx.lift = lift3d.load_lift(prev_lift, ctx.run_dir / "lift", cids) if prev_lift else None
            if ctx.lift is not None:
                print(f"[lift] reused reconstruction of the identical primary from {ctx.resume.run.name}", flush=True)
            else:
                ctx.lift = lift3d.lift_primary(
                    composites[primary], ctx.render_dirs[ctx.anchor], ctx.glbs[ctx.anchor], labels[primary].cid,
                    {v.cid: (v.azimuth, v.elevation) for v in labels.values()}, ctx.run_dir / "lift", args.blender,
                    RENDER_SCRIPT, args.resolution, args.ortho_scale, args.gpu, args.trellis2_python, args.lift_quality)
            ctx.guides = dict(ctx.lift["guides"])
            ok, ctx.lift["validation"] = validate_guides(ctx, labels, primary, composites[primary])
            if ok:
                ctx.guide_visuals = dict(ctx.lift["visuals"])
                ctx.guide_gate = calibrate_guide_gate(ctx, labels[primary], composites[primary])
            else:
                if "error" not in ctx.lift["validation"]:
                    print("[lift] guides disagree with the primary composition; continuing without 3D layout guides",
                          flush=True)
                ctx.guides = {}
                ctx.lift["disabled"] = True
        except RuntimeError as error:
            print(f"[lift] failed ({error}); continuing without 3D layout guides", flush=True)
            ctx.lift = {"error": str(error)}

    for label in order[1:]:
        refs = [(lbl, labels[lbl], composites[lbl]) for lbl in order if lbl in composites]
        composites[label], models_used[label], gates[label] = compose_view(ctx, label, labels[label], refs, None, "r0")

    history = []
    for round_idx in range(args.max_rounds):
        try:
            # Placement counts as verified when every derived view passed the layout-guide gate.
            verified = all("guide_iou" in gates[lbl] and gates_pass(args, gates[lbl]) for lbl in order[1:])
            qa = qa_check(ctx, labels, composites, f"round{round_idx}", placement_verified=verified)
        except RuntimeError as error:
            if not history:
                raise
            # Keep the repaired set (each repair already won a pairwise check) with the last review attached.
            print(f"[qa] round{round_idx} review failed ({error}); keeping the current set unreviewed", flush=True)
            history.append({**history[-1], "round": round_idx, "unreviewed": True, "repairs": [],
                            "composites": {k: str(v) for k, v in composites.items()},
                            "models": dict(models_used), "gates": {k: dict(v) for k, v in gates.items()}})
            break
        history.append({"round": round_idx, "composites": {k: str(v) for k, v in composites.items()},
                        "models": dict(models_used), "gates": {k: dict(v) for k, v in gates.items()}, "qa": qa})
        if qa["accepted"] or round_idx == args.max_rounds - 1:
            break

        # Nothing flagged but the score is low: retry the weakest derived view (ties never pick the primary).
        fallback = min(order[1:] or order, key=lambda lbl: qa["view_scores"][lbl])
        to_regen = qa["regenerate"] or [fallback]
        to_regen = [lbl for lbl in order if lbl in to_regen]
        # The primary passed its own review and every other view derives from it; keep it fixed so the
        # set converges towards it, unless it is the only view the reviewer objects to.
        to_regen = [lbl for lbl in to_regen if lbl != primary] or to_regen
        # Only views the reviewer accepted serve as references, so a rejected view cannot drag others with it.
        refs_pool = [lbl for lbl in order if lbl not in to_regen and qa["view_scores"][lbl] >= args.accept_score]
        if not refs_pool:
            refs_pool = [lbl for lbl in order if lbl not in to_regen]
        common = "; ".join(qa.get("inconsistencies", []))
        repairs = []
        for label in to_regen:
            tag = f"r{round_idx + 1}"
            refs = [(lbl, labels[lbl], composites[lbl]) for lbl in refs_pool]
            feedback = " ".join(t for t in (qa["feedback"].get(label, ""), qa["per_view"][label].get("issues", ""),
                                            f"Cross-view inconsistencies: {common}" if common else "") if t)
            new_path, new_model, new_metrics = compose_view(ctx, label, labels[label], refs, feedback, tag)
            # Keep-if-better: a regenerated view only replaces the current one when it wins a pairwise
            # comparison against it, so repairs never make the set worse.
            new_ok, old_ok = gates_pass(args, new_metrics), gates_pass(args, gates[label])
            iou_delta = new_metrics.get("guide_iou", 0.0) - gates[label].get("guide_iou", 0.0)
            if new_ok != old_ok:
                keep_new, reason = new_ok, f"{'old' if new_ok else 'new'} candidate failed the pixel gates"
            elif "guide_iou" in new_metrics and "guide_iou" in gates[label] and abs(iou_delta) >= args.guide_iou_margin:
                keep_new = iou_delta > 0
                reason = (f"layout guide IoU {new_metrics['guide_iou']:.2f} (new) vs {gates[label]['guide_iou']:.2f} (old)")
            else:
                context = {lbl: (labels[lbl], composites[lbl]) for lbl in order if lbl != label}
                try:
                    keep_new, reason = pairwise_keep_new(ctx, label, labels[label], composites[label], new_path, context, tag)
                except RuntimeError as error:
                    keep_new, reason = False, f"pairwise comparison failed ({error}); kept current"
            repairs.append({"view": label, "candidate": str(new_path), "kept_new": keep_new, "reason": reason})
            print(f"[repair] view {label} {tag}: keep {'new' if keep_new else 'old'} ({reason[:160]})", flush=True)
            if keep_new:
                composites[label], models_used[label], gates[label] = new_path, new_model, new_metrics
            refs_pool.append(label)
        history[-1]["repairs"] = repairs

    # With keep-if-better repairs the latest reviewed set is the cumulative best; absolute QA scores are
    # too noisy across rounds to pick an earlier set by score.
    best = history[-1]
    return {k: Path(v) for k, v in best["composites"].items()}, best, history


def write_outputs(ctx: Context, labels: dict[str, View], best: dict[str, Path], best_round: dict[str, Any],
                  history: list[dict[str, Any]], glbs: dict[str, Path]) -> None:
    args = ctx.args
    item_dir = args.item_dir
    selected_dir = ctx.work / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    sources = write_sources(item_dir, ctx.render_dirs[ctx.anchor], {label: view.cid for label, view in labels.items()})
    preview_rows = []
    views_meta = []
    for i, (label, view) in enumerate(labels.items()):
        target = item_dir / f"target_{label}.png"
        shutil.copy2(best[label], target)
        anchor_png = selected_dir / f"{label}_anchor_{view.cid}.png"
        other_png = selected_dir / f"{label}_other_{view.cid}.png"
        on_white(ctx.render(ctx.anchor, view.cid)).save(anchor_png)
        on_white(ctx.render(ctx.other, view.cid)).save(other_png)
        camera_npy = selected_dir / f"{label}_camera_{view.cid}.npy"
        shutil.copy2(ctx.render_dirs[ctx.anchor] / "camera" / f"{view.cid}.npy", camera_npy)
        views_meta.append({
            "label": label,
            "candidate_id": view.cid,
            "azimuth": view.azimuth,
            "elevation": view.elevation,
            "role": "primary" if i == 0 else "secondary",
            "vlm_reason": ctx.plan["view_reasons"].get(view.cid, ""),
            "target": str(target),
            "source": str(sources[label]),
            "anchor_render": str(anchor_png),
            "other_render": str(other_png),
            "camera_npy": str(camera_npy),
            "layout_guide": str(ctx.guides[view.cid]) if view.cid in ctx.guides else None,
            "qa_scores": best_round["qa"]["per_view"][label],
            "image_model": best_round["models"][label],
            "gate_metrics": best_round["gates"][label],
        })
        row_images = [on_white(ctx.render(ctx.anchor, view.cid))]
        if view.cid in ctx.guide_visuals:
            row_images.append(on_white(ctx.guide_visuals[view.cid]))
        row_images.append(Image.open(best[label]).convert("RGB"))
        preview_rows.append(labeled_row(row_images,
                                        f"view {label} (cid {view.cid}, az={view.azimuth:.0f}, el={view.elevation:.0f})  "
                                        f"min score {best_round['qa']['view_scores'][label]}"))
    preview = Image.new("RGB", (max(r.width for r in preview_rows), sum(r.height for r in preview_rows)), (255, 255, 255))
    y = 0
    for row in preview_rows:
        preview.paste(row, (0, y))
        y += row.height
    preview.save(ctx.run_dir / "preview_targets.png")

    meta = {
        "prompt": ctx.prompt,
        "prompt_info": ctx.prompt_info,
        "source_object": {"key": ctx.anchor, "name": ctx.name(ctx.anchor), "glb": str(glbs[ctx.anchor])},
        "objects": {key: {"glb": str(glbs[key]), "name": ctx.name(key), "description": ctx.desc(key)} for key in ("a", "b")},
        "anchor": ctx.anchor,
        "composition_description": ctx.plan["composition_description"],
        "anchor_changes": anchor_changes(ctx),
        "interaction_regions": ctx.plan["interaction_regions"],
        "camera": {
            "type": "ortho",
            "distance": 2.0,
            "ortho_scale": args.ortho_scale,
            "resolution": args.resolution,
            "frame": "anchor GLB frame after glTF import into Blender (Z-up), normalized to unit longest side and centered",
            "candidate_views": {cid: [v.azimuth, v.elevation] for cid, v in ctx.views.items()},
        },
        "views": views_meta,
        "lift": ({"error": ctx.lift["error"]} if "error" in ctx.lift else
                 {"reconstruction_glb": str(ctx.lift["glb"]), "recon_to_anchor": str(ctx.lift["transform"]),
                  "used": not ctx.lift.get("disabled", False), "validation": ctx.lift.get("validation"),
                  "guide_gate": ctx.guide_gate or None,
                  **{k: v for k, v in ctx.lift["report"].items() if k != "candidates"}}) if ctx.lift else None,
        "suggested_trust_view": next(iter(labels)),
        "generation": {
            "strategy": "progressive reference-conditioned editing of the anchor render + VLM QA repair",
            "vlm_models": args.vlm_models,
            "image_models": args.image_models,
            "rounds_run": len(history),
            "selected_round": best_round["round"],
            "set_score": best_round["qa"]["set_score"],
            "cross_view_consistency": best_round["qa"]["cross_view_consistency"],
            "inconsistencies": best_round["qa"]["inconsistencies"],
            "accepted": best_round["qa"]["accepted"] and not best_round.get("unreviewed", False),
            "final_set_unreviewed": bool(best_round.get("unreviewed", False)),
            "repairs": [{"after_round": h["round"], **r} for h in history for r in h.get("repairs", [])],
        },
        "work_dir": str(ctx.work),
        "run_dir": str(ctx.run_dir),
    }
    (item_dir / "compose_views.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[done] targets: {[str(item_dir / f'target_{lbl}.png') for lbl in labels]}", flush=True)
    print(f"[done] metadata: {item_dir / 'compose_views.json'}; preview: {ctx.run_dir / 'preview_targets.png'}", flush=True)


def write_sources(item_dir: Path, render_dir: Path, label_cids: dict[str, str]) -> dict[str, Path]:
    """source_<label>.png = the main (anchor) object alone, rendered from the camera of target_<label>.png.

    It is the very canvas the target was edited from, so source and target share camera and framing
    pixel for pixel. RGBA with transparent background, like the other mv_fusion items' sources."""
    sources = {}
    for label, cid in label_cids.items():
        sources[label] = item_dir / f"source_{label}.png"
        shutil.copy2(render_dir / f"{cid}.png", sources[label])
    return sources


def sources_only(args: argparse.Namespace) -> None:
    """Add source_<label>.png to an item whose targets were generated earlier (reads compose_views.json)."""
    meta_path = args.item_dir / "compose_views.json"
    meta = json.loads(meta_path.read_text())
    anchor, cam = meta["anchor"], meta["camera"]
    work = Path(meta.get("work_dir") or args.work_dir)
    views = {cid: View(cid, float(az), float(el)) for cid, (az, el) in cam["candidate_views"].items()}
    args.resolution, args.ortho_scale = int(cam["resolution"]), float(cam["ortho_scale"])
    render_dir = work / "renders" / anchor
    if not renders_complete(render_dir, views, args):
        # The cached canvases were overwritten (e.g. a later run with another --ortho_scale): re-render
        # the anchor with the camera settings the targets were made with.
        render_dir = work / "renders_source" / anchor
        ctx = Context(args=args, prompt="", views=views, work=work, render_dirs={anchor: render_dir})
        render_objects(ctx, {anchor: Path(meta["objects"][anchor]["glb"])})
    sources = write_sources(args.item_dir, render_dir, {v["label"]: v["candidate_id"] for v in meta["views"]})
    for v in meta["views"]:
        v["source"] = str(sources[v["label"]])
        canvas = Path(v.get("anchor_render", ""))
        diff = (float(np.abs(_gray(on_white(sources[v["label"]]), 256) - _gray(Image.open(canvas), 256)).mean())
                if canvas.is_file() else float("nan"))
        print(f"[source] {sources[v['label']]} <- {render_dir / (v['candidate_id'] + '.png')} "
              f"(diff to target canvas {diff:.4f})", flush=True)
    meta["source_object"] = {"key": anchor, "name": meta["objects"][anchor]["name"], "glb": meta["objects"][anchor]["glb"]}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj_a", type=Path)
    ap.add_argument("--obj_b", type=Path)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt_file", type=Path, default=None,
                    help="default <item_dir>/prompt.txt; if it does not exist, an instruction is generated from the "
                         "two GLBs and saved there")
    ap.add_argument("--item_dir", required=True, type=Path,
                    help="mv_fusion item dir; receives target_<label>.png / source_<label>.png")
    ap.add_argument("--resume_from", default=None,
                    help="earlier run dir (or 'latest') whose results are reused for identical requests, e.g. "
                         "after an interruption")
    ap.add_argument("--sources_only", action="store_true",
                    help="only write source_<label>.png for targets generated earlier (reads compose_views.json)")
    ap.add_argument("--work_dir", type=Path, default=None, help="default: <item_dir>/_compose_prep")
    ap.add_argument("--candidate_views", default=DEFAULT_VIEWS, help="'az,el;az,el;...' (default: 8-view preset)")
    ap.add_argument("--num_views", type=int, default=3)
    ap.add_argument("--anchor", choices=("auto", "a", "b"), default="auto")
    ap.add_argument("--initial_scene", action="store_true",
                    help="obj_a is a pre-composed scene that already contains obj_b (obj_b = that part alone, as appearance "
                         "reference). Sources/canvases are renders of the scene; the edit adjusts B inside it.")
    ap.add_argument("--vlm_models", default=os.getenv("MV_COMPOSE_VLM_MODELS", "gemini-3.1-pro-preview,gemini-3-pro-preview"),
                    help="comma-separated, tried in order")
    ap.add_argument("--image_models", default=os.getenv(
        "MV_COMPOSE_IMAGE_MODELS", "gemini-3-pro-image-preview,gemini-3.1-flash-image-preview,gemini-2.5-flash-image"),
        help="comma-separated, tried in order")
    ap.add_argument("--blender", type=Path, default=Path(os.getenv("BLENDER_PATH", DEFAULT_BLENDER)))
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--ortho_scale", type=float, default=1.5,
                    help="reference script uses 1.2; a wider frame leaves room for the added object")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--gpu", action="store_true", help="Cycles on GPU (reference default is CPU)")
    ap.add_argument("--max_rounds", type=int, default=3, help="QA rounds (each failed round regenerates flagged views)")
    ap.add_argument("--accept_score", type=int, default=7, help="minimum per-view and consistency score to accept")
    ap.add_argument("--primary_rounds", type=int, default=3, help="max attempts for the primary view before propagating")
    ap.add_argument("--gate_retries", type=int, default=2, help="immediate regenerations when a pixel gate fails")
    ap.add_argument("--min_anchor_coverage_deform", type=float, default=0.85,
                    help="coverage threshold when the instruction re-poses / deforms the anchor (plan anchor_changes)")
    ap.add_argument("--min_anchor_coverage", type=float, default=0.95,
                    help="share of the anchor silhouette that must stay foreground (catches moved anchor / changed view)")
    ap.add_argument("--min_ref_distance", type=float, default=0.03,
                    help="min mean gray difference to reference composites (catches copied references)")
    ap.add_argument("--no_lift", dest="lift", action="store_false",
                    help="disable the TRELLIS.2 3D layout guides for secondary views")
    ap.add_argument("--lift_quality", default="fast", help="run_trellis2_image_to_3d.py quality preset")
    ap.add_argument("--trellis2_python", type=Path, default=lift3d.DEFAULT_TRELLIS2_PYTHON)
    ap.add_argument("--min_guide_iou", type=float, default=0.4,
                    help="min IoU of the added object's region between a composite and its 3D layout guide")
    ap.add_argument("--min_guide_self_iou", type=float, default=0.5,
                    help="enable the guide gate only if the guide reproduces the primary at least this well")
    ap.add_argument("--min_guide_region_px", type=int, default=300,
                    help="min added-object area outside the anchor outline (256px grid) for the guide gate")
    ap.add_argument("--guide_iou_margin", type=float, default=0.15,
                    help="repairs: a guide-IoU difference this large decides between candidates without a VLM call")
    ap.add_argument("--force_render", action="store_true")
    ap.add_argument("--force_plan", action="store_true")
    args = ap.parse_args()
    args.image_models = [m.strip() for m in args.image_models.split(",") if m.strip()]
    args.vlm_models = [m.strip() for m in args.vlm_models.split(",") if m.strip()]
    args.item_dir = args.item_dir.resolve()
    args.prompt_file = (args.prompt_file or args.item_dir / "prompt.txt").resolve()
    if not args.sources_only:
        if args.obj_a is None or args.obj_b is None:
            ap.error("--obj_a and --obj_b are required (unless --sources_only)")
        args.prompt_from_file = args.prompt is None and args.prompt_file.is_file()
        if args.prompt_from_file:
            args.prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    args.work_dir = (args.work_dir or args.item_dir / "_compose_prep").resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.sources_only:
        sources_only(args)
        return
    glbs = {"a": args.obj_a.resolve(), "b": args.obj_b.resolve()}
    for key, path in glbs.items():
        if not path.is_file():
            raise FileNotFoundError(f"object {key}: {path}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    vlm.set_log_path(args.work_dir / "logs" / "vlm_calls.jsonl")

    views = parse_views(args.candidate_views)
    if len(views) < args.num_views:
        raise ValueError(f"need at least {args.num_views} candidate views, got {len(views)}")
    resume = None
    if args.resume_from:
        runs = sorted(p for p in (args.work_dir / "runs").glob("2*") if p.is_dir())
        prev = runs[-1] if args.resume_from == "latest" and runs else Path(args.resume_from)
        if not prev.is_dir():
            raise FileNotFoundError(f"--resume_from: no run directory {prev}")
        resume = ResumeCache(prev.resolve())
    ctx = Context(args=args, prompt=args.prompt or "", views=views, work=args.work_dir, glbs=glbs, resume=resume,
                  render_dirs={"a": args.work_dir / "renders" / "a", "b": args.work_dir / "renders" / "b"},
                  run_dir=args.work_dir / "runs" / time.strftime("%Y%m%d_%H%M%S"))
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] work_dir={ctx.work} run_dir={ctx.run_dir}", flush=True)

    # 1. Render both objects separately.
    render_objects(ctx, glbs)

    # 1b. No instruction given: write one from the two assets and save it as the item's prompt.
    generated_record = ctx.work / "prompt_generated.json"
    if ctx.prompt:
        ctx.prompt_info = {"source": "user", "prompt_file": str(args.prompt_file) if args.prompt_from_file else None}
        if generated_record.is_file():  # prompt.txt written by an earlier run of this tool and left unchanged
            record = json.loads(generated_record.read_text())
            if record.get("prompt", "").strip() == ctx.prompt.strip():
                ctx.prompt_info = record
    else:
        generated = generate_prompt(ctx)
        ctx.prompt = generated["prompt"]
        args.prompt_file.write_text(ctx.prompt + "\n", encoding="utf-8")
        ctx.prompt_info = {"source": "generated", "prompt_file": str(args.prompt_file), **generated}
        generated_record.write_text(json.dumps(ctx.prompt_info, indent=2, ensure_ascii=False))
        print(f"[prompt] generated -> {args.prompt_file}: {ctx.prompt}", flush=True)

    # 2. Plan: object descriptions, anchor and the most informative views.
    ctx.plan = plan_composition(ctx)
    ctx.anchor = "a" if args.initial_scene else ctx.plan["anchor"] if args.anchor == "auto" else args.anchor
    print(f"[plan] anchor={ctx.anchor} ({ctx.name(ctx.anchor)}), other={ctx.name(ctx.other)}, "
          f"views={ctx.plan['selected_views']}", flush=True)
    for cid in ctx.plan["selected_views"]:
        print(f"[plan]   {cid}: {ctx.plan['view_reasons'].get(cid, '')}", flush=True)

    sheet_ids = spread_views(views, 4)
    ctx.other_sheet = ctx.work / "reference_other_sheet.png"
    grid([on_white(ctx.render(ctx.other, cid)) for cid in sheet_ids], cols=2).save(ctx.other_sheet)

    # 3-4. Progressive composition + QA repair; labels follow the VLM priority (0 = primary).
    labels = {str(i): views[cid] for i, cid in enumerate(ctx.plan["selected_views"])}
    best, best_round, history = generate_consistent_views(ctx, labels)

    # 5. Outputs.
    write_outputs(ctx, labels, best, best_round, history, glbs)


if __name__ == "__main__":
    main()
