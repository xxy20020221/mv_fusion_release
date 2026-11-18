from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

# ---------------------------------------------------------------------------
# rembg-based background removal (default path)
# ---------------------------------------------------------------------------
#
# The original HSV flood-fill logic (kept below as ``method="hsv_floodfill"``)
# assumed a light, low-saturation, connected-to-corner background. That breaks
# for renders with dark / colored / cluttered backgrounds (e.g. black-bg
# renders where the whole foreground is *not* light) and can silently produce
# an "empty foreground" RGBA. The default now delegates to ``rembg`` with a
# strong general-purpose model (BiRefNet-general) which robustly handles
# both light and dark backgrounds. The old flood-fill is retained as an
# opt-in fallback (``method="hsv_floodfill"``) for backwards compatibility.

_REMBG_SESSIONS: dict[str, object] = {}


def _get_rembg_session(model_name: str):
    """Return a cached ``rembg`` session for ``model_name`` (lazy init)."""
    if model_name not in _REMBG_SESSIONS:
        import rembg  # local import: only paid when method="rembg"

        _REMBG_SESSIONS[model_name] = rembg.new_session(model_name)
    return _REMBG_SESSIONS[model_name]


def build_alpha_with_rembg(
    image: Image.Image,
    model_name: str = "birefnet-general",
    alpha_threshold: int | None = 128,
) -> Image.Image:
    """Return a binary L-mode alpha mask (255=fg, 0=bg) produced by ``rembg``.

    Parameters
    ----------
    image:
        Input image, any mode (converted to RGB internally).
    model_name:
        ``rembg`` model identifier. ``"birefnet-general"`` is the strongest
        general-purpose model available in the installed ``rembg`` and
        handles dark / cluttered backgrounds well. Alternatives:
        ``"isnet-general-use"`` (fast, decent), ``"u2net"`` (fastest, weakest).
    alpha_threshold:
        If not None, binarize the soft alpha at this value (0-255) to match
        the behavior of the legacy HSV flood-fill (which produced a binary
        mask). Set to ``None`` to keep the soft alpha as-is.
    """
    import rembg  # noqa: F401  # ensures ImportError surfaces early

    session = _get_rembg_session(model_name)
    rgba = rembg.remove(image.convert("RGB"), session=session)
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = rgba.split()[3]  # L-mode soft alpha
    if alpha_threshold is not None:
        thr = int(alpha_threshold)
        alpha = alpha.point(lambda p, t=thr: 255 if p >= t else 0, "L")
    return alpha


def build_alpha_from_light_connected_background(
    image: Image.Image,
    saturation_max: int = 46,
    value_min: int = 194,
    seed_x: int = 0,
    seed_y: int = 0,
) -> Image.Image:
    """Build a binary alpha mask by removing the light, low-saturation background connected to a seed pixel."""
    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    _, saturation, value = hsv.split()

    low_saturation = saturation.point(lambda p: 255 if p <= saturation_max else 0, "L")
    high_value = value.point(lambda p: 255 if p >= value_min else 0, "L")
    background_candidate = ImageChops.multiply(low_saturation, high_value)

    connected_background = background_candidate.copy()
    ImageDraw.floodfill(connected_background, (seed_x, seed_y), 128, thresh=0)

    return connected_background.point(lambda p: 0 if p == 128 else 255, "L")

def write_easy3e_rgba_target(
    input_path: Path,
    output_rgba_path: Path | None = None,
    output_mask_path: Path | None = None,
    output_mask_512_path: Path | None = None,
    saturation_max: int = 46,
    value_min: int = 194,
    seed_x: int = 0,
    seed_y: int = 0,
    method: str = "rembg",
    rembg_model: str = "birefnet-general",
    alpha_threshold: int | None = 128,
) -> dict[str, object]:
    input_path = input_path.expanduser().resolve()
    stem = input_path.stem

    if output_rgba_path is None:
        output_rgba_path = input_path.with_name(f"{stem}_easy3e_rgba.png")
    if output_mask_path is None:
        output_mask_path = input_path.with_name(f"{stem}_easy3e_alpha_mask.png")
    if output_mask_512_path is None:
        output_mask_512_path = input_path.with_name(f"{stem}_easy3e_alpha_mask_512.png")

    output_rgba_path = output_rgba_path.expanduser().resolve()
    output_mask_path = output_mask_path.expanduser().resolve()
    output_mask_512_path = output_mask_512_path.expanduser().resolve()

    image = Image.open(input_path).convert("RGB")
    if method == "rembg":
        alpha = build_alpha_with_rembg(
            image,
            model_name=rembg_model,
            alpha_threshold=alpha_threshold,
        )
    elif method == "hsv_floodfill":
        alpha = build_alpha_from_light_connected_background(
            image,
            saturation_max=saturation_max,
            value_min=value_min,
            seed_x=seed_x,
            seed_y=seed_y,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Expected 'rembg' or 'hsv_floodfill'."
        )

    rgba = Image.merge("RGBA", (*image.split(), alpha))
    transparent_background = alpha.point(lambda p: 255 if p == 0 else 0, "L")
    rgba = Image.composite(Image.new("RGBA", image.size, (0, 0, 0, 0)), rgba, transparent_background)

    output_rgba_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_512_path.parent.mkdir(parents=True, exist_ok=True)

    rgba.save(output_rgba_path)
    alpha.save(output_mask_path)
    alpha.resize((512, 512), Image.Resampling.NEAREST).save(output_mask_512_path)

    histogram = alpha.histogram()
    foreground_pixels = histogram[255]
    total_pixels = image.size[0] * image.size[1]

    return {
        "input": str(input_path),
        "source_size": image.size,
        "rgba": str(output_rgba_path),
        "alpha_mask": str(output_mask_path),
        "alpha_mask_512": str(output_mask_512_path),
        "alpha_minmax": alpha.getextrema(),
        "foreground_pixels": foreground_pixels,
        "foreground_ratio": foreground_pixels / float(total_pixels),
        "foreground_bbox_xyxy": alpha.getbbox(),
        "method": method,
        "rembg_model": rembg_model if method == "rembg" else None,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an edited image into an Easy3E-compatible RGBA target and alpha masks."
    )
    parser.add_argument("input", type=Path, help="Input edited image path.")
    parser.add_argument("--output-rgba", type=Path, default=None, help="Output RGBA target path.")
    parser.add_argument("--output-mask", type=Path, default=None, help="Output original-size alpha mask path.")
    parser.add_argument("--output-mask-512", type=Path, default=None, help="Output 512x512 alpha mask path.")
    parser.add_argument("--method", choices=("rembg", "hsv_floodfill"), default="rembg",
                        help="Background removal method (default: rembg, robust to dark/cluttered backgrounds).")
    parser.add_argument("--rembg-model", default="birefnet-general",
                        help="rembg model name when --method=rembg. Common: "
                             "'birefnet-general' (best), 'isnet-general-use', 'u2net'.")
    parser.add_argument("--alpha-threshold", type=int, default=128,
                        help="Binarize rembg soft alpha at this threshold (0-255). "
                             "Pass -1 to keep the soft alpha.")
    parser.add_argument("--saturation-max", type=int, default=46, help="[hsv_floodfill only] Maximum HSV saturation for background candidates.")
    parser.add_argument("--value-min", type=int, default=194, help="[hsv_floodfill only] Minimum HSV value for background candidates.")
    parser.add_argument("--seed-x", type=int, default=0, help="[hsv_floodfill only] Flood-fill seed x coordinate for connected background.")
    parser.add_argument("--seed-y", type=int, default=0, help="[hsv_floodfill only] Flood-fill seed y coordinate for connected background.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    alpha_threshold = None if args.alpha_threshold is not None and args.alpha_threshold < 0 else args.alpha_threshold
    summary = write_easy3e_rgba_target(
        input_path=args.input,
        output_rgba_path=args.output_rgba,
        output_mask_path=args.output_mask,
        output_mask_512_path=args.output_mask_512,
        saturation_max=args.saturation_max,
        value_min=args.value_min,
        seed_x=args.seed_x,
        seed_y=args.seed_y,
        method=args.method,
        rembg_model=args.rembg_model,
        alpha_threshold=alpha_threshold,
    )

    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
