from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


Array = np.ndarray


def load_rgba(path: Path) -> tuple[Array, Array, tuple[int, int]]:
    image = Image.open(path).convert("RGBA")
    rgba = np.array(image)
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].copy()
    return rgb, alpha, image.size


def clean_binary_mask(mask: Array, open_size: int = 3, close_size: int = 5) -> Array:
    mask_u8 = (mask.astype(np.uint8) * 255)
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 127


def remove_small_components(mask: Array, min_area: int) -> Array:
    if min_area <= 1 or not np.any(mask):
        return mask.astype(bool)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    kept = np.zeros(mask.shape, dtype=bool)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            kept |= labels == label
    return kept


def connected_background_from_corners(candidate_background: Array) -> Array:
    labels_count, labels = cv2.connectedComponents(candidate_background.astype(np.uint8), 8)
    if labels_count <= 1:
        return np.zeros(candidate_background.shape, dtype=bool)

    border_labels = np.unique(
        np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]],
            axis=0,
        )
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros(candidate_background.shape, dtype=bool)
    return np.isin(labels, border_labels)


def estimate_foreground_mask(
    rgb: Array,
    alpha: Array,
    alpha_threshold: int,
    background_tolerance: float,
) -> Array:
    alpha_mask = alpha > alpha_threshold
    alpha_ratio = float(alpha_mask.mean())
    if 0.001 <= alpha_ratio <= 0.995:
        return clean_binary_mask(alpha_mask)

    h, w = rgb.shape[:2]
    patch = max(4, int(round(min(h, w) * 0.025)))
    corner_pixels = np.concatenate(
        [
            rgb[:patch, :patch].reshape(-1, 3),
            rgb[:patch, -patch:].reshape(-1, 3),
            rgb[-patch:, :patch].reshape(-1, 3),
            rgb[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    background_color = np.median(corner_pixels, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background_color[None, None, :], axis=2)
    background_candidate = distance <= float(background_tolerance)
    connected_background = connected_background_from_corners(background_candidate)
    foreground = ~connected_background

    foreground_ratio = float(foreground.mean())
    if foreground_ratio < 0.001 or foreground_ratio > 0.995:
        return np.ones((h, w), dtype=bool)
    return clean_binary_mask(foreground)


def mask_bbox(mask: Array) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def bbox_center_and_size(bbox: tuple[int, int, int, int]) -> tuple[Array, Array]:
    x0, y0, x1, y1 = bbox
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
    size = np.array([max(1, x1 - x0), max(1, y1 - y0)], dtype=np.float32)
    return center, size


def fallback_affine_from_masks(source_mask: Array, target_mask: Array) -> Array:
    sh, sw = source_mask.shape[:2]
    th, tw = target_mask.shape[:2]
    source_bbox = mask_bbox(source_mask)
    target_bbox = mask_bbox(target_mask)
    if source_bbox is None or target_bbox is None:
        scale = math.sqrt((tw * th) / float(max(1, sw * sh)))
        return np.array([[scale, 0.0, (tw - sw * scale) * 0.5], [0.0, scale, (th - sh * scale) * 0.5]], dtype=np.float32)

    source_center, source_size = bbox_center_and_size(source_bbox)
    target_center, target_size = bbox_center_and_size(target_bbox)

    canvas_scale = math.sqrt((tw * th) / float(max(1, sw * sh)))
    height_scale = float(target_size[1] / max(1.0, source_size[1]))
    area_scale = math.sqrt(float(np.count_nonzero(target_mask)) / float(max(1, np.count_nonzero(source_mask))))
    scales = [canvas_scale, height_scale, area_scale]
    scale = float(np.median([s for s in scales if np.isfinite(s) and 0.1 <= s <= 10.0]))
    translation = target_center - source_center * scale
    return np.array([[scale, 0.0, translation[0]], [0.0, scale, translation[1]]], dtype=np.float32)


def make_feature_detector(method: str) -> tuple[Any, int, float, str]:
    if method in {"auto", "sift"} and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01), cv2.NORM_L2, 0.75, "sift"
    return cv2.ORB_create(nfeatures=10000, scaleFactor=1.15, nlevels=10, edgeThreshold=8, fastThreshold=5), cv2.NORM_HAMMING, 0.82, "orb"


def estimate_affine_with_features(
    source_rgb: Array,
    target_rgb: Array,
    source_mask: Array,
    target_mask: Array,
    feature_method: str,
) -> tuple[Array | None, dict[str, Any]]:
    detector, norm_type, ratio_threshold, used_method = make_feature_detector(feature_method)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)

    source_feature_mask = cv2.erode((source_mask.astype(np.uint8) * 255), np.ones((5, 5), np.uint8), iterations=1)
    target_feature_mask = cv2.erode((target_mask.astype(np.uint8) * 255), np.ones((5, 5), np.uint8), iterations=1)
    if np.count_nonzero(source_feature_mask) < 256:
        source_feature_mask = source_mask.astype(np.uint8) * 255
    if np.count_nonzero(target_feature_mask) < 256:
        target_feature_mask = target_mask.astype(np.uint8) * 255

    source_keypoints, source_desc = detector.detectAndCompute(source_gray, source_feature_mask)
    target_keypoints, target_desc = detector.detectAndCompute(target_gray, target_feature_mask)
    info: dict[str, Any] = {
        "feature_method": used_method,
        "source_keypoints": len(source_keypoints),
        "target_keypoints": len(target_keypoints),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
    }
    if source_desc is None or target_desc is None or len(source_keypoints) < 8 or len(target_keypoints) < 8:
        return None, info

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    knn_matches = matcher.knnMatch(source_desc, target_desc, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio_threshold * second.distance:
            good_matches.append(best)
    info["good_matches"] = len(good_matches)
    if len(good_matches) < 8:
        return None, info

    source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_points = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    reproj_threshold = max(4.0, min(source_rgb.shape[0], source_rgb.shape[1], target_rgb.shape[0], target_rgb.shape[1]) * 0.008)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
        maxIters=8000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inlier_mask is None:
        return None, info

    inliers = int(inlier_mask.ravel().sum())
    inlier_ratio = inliers / float(max(1, len(good_matches)))
    scale_x = float(np.linalg.norm(matrix[0, :2]))
    scale_y = float(np.linalg.norm(matrix[1, :2]))
    info.update(
        {
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "matrix": matrix.astype(float).tolist(),
        }
    )
    if inliers < 8 or inlier_ratio < 0.12 or not (0.2 <= scale_x <= 5.0 and 0.2 <= scale_y <= 5.0):
        return None, info
    return matrix.astype(np.float32), info


def warp_mask(mask: Array, matrix: Array, target_shape: tuple[int, int]) -> Array:
    h, w = target_shape
    warped = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 127


def refine_translation_with_masks(matrix: Array, source_mask: Array, target_mask: Array, max_shift: float) -> tuple[Array, dict[str, Any]]:
    if max_shift <= 0:
        return matrix, {"enabled": False}
    th, tw = target_mask.shape[:2]
    warped_source = cv2.warpAffine(
        source_mask.astype(np.float32),
        matrix,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    target_float = target_mask.astype(np.float32)
    warped_source = cv2.GaussianBlur(warped_source, (0, 0), 3.0)
    target_float = cv2.GaussianBlur(target_float, (0, 0), 3.0)
    shift, response = cv2.phaseCorrelate(warped_source, target_float)
    dx, dy = float(shift[0]), float(shift[1])
    info = {"enabled": True, "dx": dx, "dy": dy, "response": float(response), "applied": False}
    if response >= 0.03 and math.hypot(dx, dy) <= float(max_shift):
        refined = matrix.copy()
        refined[0, 2] += dx
        refined[1, 2] += dy
        info["applied"] = True
        return refined, info
    return matrix, info


def estimate_alignment(
    source_rgb: Array,
    target_rgb: Array,
    source_mask: Array,
    target_mask: Array,
    feature_method: str,
    max_refine_shift: float,
) -> tuple[Array, dict[str, Any]]:
    feature_matrix, feature_info = estimate_affine_with_features(
        source_rgb,
        target_rgb,
        source_mask,
        target_mask,
        feature_method,
    )
    if feature_matrix is None:
        matrix = fallback_affine_from_masks(source_mask, target_mask)
        method = "foreground_bbox_fallback"
    else:
        matrix = feature_matrix
        method = f"{feature_info.get('feature_method', 'feature')}_ransac"

    matrix, refine_info = refine_translation_with_masks(matrix, source_mask, target_mask, max_refine_shift)
    return matrix, {"method": method, "feature": feature_info, "translation_refine": refine_info}


def local_min_lab_delta(
    source_rgb: Array,
    target_rgb: Array,
    source_mask: Array,
    radius: int,
) -> tuple[Array, Array]:
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    h, w = target_lab.shape[:2]
    min_delta = np.full((h, w), np.inf, dtype=np.float32)
    source_near = np.zeros((h, w), dtype=bool)

    radius = max(0, int(radius))
    for dy in range(-radius, radius + 1):
        src_y0 = max(0, dy)
        src_y1 = min(h, h + dy)
        dst_y0 = max(0, -dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        if src_y1 <= src_y0:
            continue
        for dx in range(-radius, radius + 1):
            src_x0 = max(0, dx)
            src_x1 = min(w, w + dx)
            dst_x0 = max(0, -dx)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            if src_x1 <= src_x0:
                continue

            valid_source = source_mask[src_y0:src_y1, src_x0:src_x1]
            if not np.any(valid_source):
                continue
            diff = target_lab[dst_y0:dst_y1, dst_x0:dst_x1] - source_lab[src_y0:src_y1, src_x0:src_x1]
            delta = np.sqrt(np.sum(diff.astype(np.float32) * diff.astype(np.float32), axis=2))
            delta = np.where(valid_source, delta, np.inf).astype(np.float32)
            current = min_delta[dst_y0:dst_y1, dst_x0:dst_x1]
            np.minimum(current, delta, out=current)
            source_near[dst_y0:dst_y1, dst_x0:dst_x1] |= valid_source

    return min_delta, source_near


def adaptive_delta_threshold(
    delta: Array,
    valid_mask: Array,
    min_threshold: float,
    mad_multiplier: float,
) -> tuple[float, dict[str, float]]:
    finite_valid = valid_mask & np.isfinite(delta)
    values = delta[finite_valid]
    if values.size < 64:
        return float(min_threshold), {"mode": "fallback", "min_threshold": float(min_threshold)}

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(float(min_threshold), median + float(mad_multiplier) * robust_sigma)
    threshold = min(threshold, float(np.percentile(values, 98.5)))
    threshold = max(threshold, float(min_threshold))
    return threshold, {
        "mode": "adaptive_median_mad",
        "median": median,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p98_5": float(np.percentile(values, 98.5)),
        "threshold": float(threshold),
    }


def dilate(mask: Array, radius: int) -> Array:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask.astype(bool)
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def build_edit_mask(
    target_rgb: Array,
    target_mask: Array,
    warped_source_rgb: Array,
    warped_source_mask: Array,
    local_radius: int,
    shape_tolerance: int,
    min_delta_threshold: float,
    fixed_delta_threshold: float | None,
    mad_multiplier: float,
    min_region_area: int,
    final_dilate: int,
    include_removed_source: bool,
    target_foreground_only: bool,
) -> tuple[Array, Array, dict[str, Any]]:
    delta, source_near = local_min_lab_delta(warped_source_rgb, target_rgb, warped_source_mask, local_radius)
    common = target_mask & source_near
    if fixed_delta_threshold is None:
        delta_threshold, threshold_info = adaptive_delta_threshold(delta, common, min_delta_threshold, mad_multiplier)
    else:
        delta_threshold = float(fixed_delta_threshold)
        threshold_info = {"mode": "fixed", "threshold": delta_threshold}

    finite_delta = np.where(np.isfinite(delta), delta, 0.0).astype(np.float32)
    smoothed_delta = cv2.GaussianBlur(finite_delta, (0, 0), 1.0)
    color_change = common & (smoothed_delta > delta_threshold)

    source_near_shape = dilate(warped_source_mask, shape_tolerance)
    target_near_shape = dilate(target_mask, shape_tolerance)
    new_target_shape = target_mask & ~source_near_shape
    removed_source_shape = warped_source_mask & ~target_near_shape
    shape_change = new_target_shape | (removed_source_shape if include_removed_source else np.zeros_like(new_target_shape))

    raw_mask = color_change | shape_change
    if target_foreground_only:
        raw_mask &= target_mask

    raw_mask = clean_binary_mask(raw_mask, open_size=3, close_size=9)
    default_min_area = max(64, int(round(raw_mask.size * 0.00012)))
    cleaned = remove_small_components(raw_mask, max(int(min_region_area), default_min_area))
    if final_dilate > 0 and np.any(cleaned):
        cleaned = dilate(cleaned, final_dilate)
        if target_foreground_only:
            cleaned &= dilate(target_mask, 1)

    heatmap = np.clip(smoothed_delta / max(1.0, delta_threshold) * 128.0, 0, 255).astype(np.uint8)
    info = {
        "delta_threshold": float(delta_threshold),
        "threshold_info": threshold_info,
        "raw_changed_pixels": int(np.count_nonzero(raw_mask)),
        "changed_pixels": int(np.count_nonzero(cleaned)),
        "changed_ratio_in_target": float(np.count_nonzero(cleaned) / max(1, cleaned.size)),
        "changed_bbox_xyxy": mask_bbox(cleaned),
        "color_change_pixels": int(np.count_nonzero(color_change)),
        "new_target_shape_pixels": int(np.count_nonzero(new_target_shape)),
        "removed_source_shape_pixels": int(np.count_nonzero(removed_source_shape)),
    }
    return cleaned, heatmap, info


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("Expected color as R,G,B with values in [0,255].")
    return parts[0], parts[1], parts[2]


def save_overlay(
    target_rgb: Array,
    target_alpha: Array,
    edit_mask: Array,
    output_path: Path,
    color: tuple[int, int, int],
    overlay_alpha: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_rgb = target_rgb.astype(np.float32).copy()
    color_array = np.array(color, dtype=np.float32)[None, None, :]
    mask = edit_mask.astype(bool)
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    out_rgb[mask] = out_rgb[mask] * (1.0 - alpha) + color_array * alpha
    out_alpha = target_alpha.copy()
    out_alpha[mask] = np.maximum(out_alpha[mask], np.uint8(255))
    rgba = np.dstack([np.clip(out_rgb, 0, 255).astype(np.uint8), out_alpha])
    Image.fromarray(rgba, mode="RGBA").save(output_path)


def save_mask(mask: Array, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_path)


def save_heatmap(heatmap: Array, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    Image.fromarray(colored, mode="RGB").save(output_path)


def save_aligned_source(warped_rgb: Array, warped_mask: Array, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = (warped_mask.astype(np.uint8) * 255)
    Image.fromarray(np.dstack([warped_rgb, alpha]), mode="RGBA").save(output_path)


def detect_edit_regions(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    target_stem = target_path.stem

    output_path = args.output
    if output_path is None:
        output_path = target_path.with_name(f"{target_stem}_edit_regions_overlay.png")
    mask_output_path = args.mask_output
    if mask_output_path is None:
        mask_output_path = target_path.with_name(f"{target_stem}_edit_regions_mask.png")

    source_rgb, source_alpha, source_size = load_rgba(source_path)
    target_rgb, target_alpha, target_size = load_rgba(target_path)
    source_mask = estimate_foreground_mask(source_rgb, source_alpha, args.alpha_threshold, args.background_tolerance)
    target_mask = estimate_foreground_mask(target_rgb, target_alpha, args.alpha_threshold, args.background_tolerance)

    alignment_matrix, alignment_info = estimate_alignment(
        source_rgb,
        target_rgb,
        source_mask,
        target_mask,
        args.feature_method,
        args.max_refine_shift,
    )

    th, tw = target_rgb.shape[:2]
    warped_source_rgb = cv2.warpAffine(
        source_rgb,
        alignment_matrix,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_source_mask = warp_mask(source_mask, alignment_matrix, (th, tw))

    edit_mask, heatmap, diff_info = build_edit_mask(
        target_rgb=target_rgb,
        target_mask=target_mask,
        warped_source_rgb=warped_source_rgb,
        warped_source_mask=warped_source_mask,
        local_radius=args.local_radius,
        shape_tolerance=args.shape_tolerance,
        min_delta_threshold=args.min_delta_threshold,
        fixed_delta_threshold=args.diff_threshold,
        mad_multiplier=args.mad_multiplier,
        min_region_area=args.min_region_area,
        final_dilate=args.final_dilate,
        include_removed_source=args.include_removed_source,
        target_foreground_only=not args.allow_background_marks,
    )

    save_overlay(target_rgb, target_alpha, edit_mask, output_path, args.overlay_color, args.overlay_alpha)
    save_mask(edit_mask, mask_output_path)
    if args.heatmap_output is not None:
        save_heatmap(heatmap, args.heatmap_output)
    if args.aligned_source_output is not None:
        save_aligned_source(warped_source_rgb, warped_source_mask, args.aligned_source_output)

    return {
        "source": str(source_path),
        "target": str(target_path),
        "source_size_wh": source_size,
        "target_size_wh": target_size,
        "source_foreground_bbox_xyxy": mask_bbox(source_mask),
        "target_foreground_bbox_xyxy": mask_bbox(target_mask),
        "alignment": {
            **alignment_info,
            "matrix_source_to_target": alignment_matrix.astype(float).tolist(),
        },
        "diff": diff_info,
        "outputs": {
            "overlay": str(output_path.expanduser().resolve()),
            "mask": str(mask_output_path.expanduser().resolve()),
            "heatmap": str(args.heatmap_output.expanduser().resolve()) if args.heatmap_output is not None else None,
            "aligned_source": str(args.aligned_source_output.expanduser().resolve()) if args.aligned_source_output is not None else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect likely edited regions between a source image and a generated target image. "
            "The script first aligns the source to the target with a similarity transform, "
            "then uses local-tolerant LAB differences and foreground shape changes to avoid "
            "flagging global translation/scale/resize differences as edits."
        )
    )
    parser.add_argument("source", type=Path, help="Source image path.")
    parser.add_argument("target", type=Path, help="Target image path.")
    parser.add_argument("--output", type=Path, default=None, help="Colored target overlay output path.")
    parser.add_argument("--mask-output", type=Path, default=None, help="Binary edit mask output path.")
    parser.add_argument("--heatmap-output", type=Path, default=None, help="Optional LAB-difference heatmap output path.")
    parser.add_argument("--aligned-source-output", type=Path, default=None, help="Optional aligned source RGBA debug output path.")
    parser.add_argument("--feature-method", choices=("auto", "sift", "orb"), default="auto", help="Feature matcher used for alignment.")
    parser.add_argument("--alpha-threshold", type=int, default=8, help="Alpha threshold for foreground masks.")
    parser.add_argument("--background-tolerance", type=float, default=35.0, help="RGB distance tolerance for corner-background fallback masks.")
    parser.add_argument("--max-refine-shift", type=float, default=25.0, help="Maximum phase-correlation translation refinement in pixels; use 0 to disable.")
    parser.add_argument("--local-radius", type=int, default=5, help="Local search radius that tolerates residual non-edit displacement.")
    parser.add_argument("--shape-tolerance", type=int, default=9, help="Foreground-mask dilation radius before considering shape as changed.")
    parser.add_argument("--min-delta-threshold", type=float, default=18.0, help="Minimum LAB delta threshold for color/content changes.")
    parser.add_argument("--diff-threshold", type=float, default=None, help="Fixed LAB delta threshold; by default an adaptive threshold is used.")
    parser.add_argument("--mad-multiplier", type=float, default=4.0, help="Adaptive threshold = median + multiplier * robust_sigma.")
    parser.add_argument("--min-region-area", type=int, default=0, help="Minimum connected edit-region area; 0 uses an image-size adaptive value.")
    parser.add_argument("--final-dilate", type=int, default=3, help="Final dilation radius for easier visualization.")
    parser.add_argument("--include-removed-source", action="store_true", help="Also mark places where source foreground disappeared in target.")
    parser.add_argument("--allow-background-marks", action="store_true", help="Allow marking background pixels, useful with --include-removed-source.")
    parser.add_argument("--overlay-color", type=parse_color, default=(255, 0, 255), help="Overlay color as R,G,B.")
    parser.add_argument("--overlay-alpha", type=float, default=0.58, help="Overlay opacity in [0,1].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = detect_edit_regions(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
