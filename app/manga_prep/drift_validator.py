"""Post-generation visual drift detection.

Compares generated panels against reference images and previous panels in
the same continuity group.  Produces DriftScore objects and flags panels
that should be regenerated.

Scoring uses structural similarity (SSIM) on masked regions:
  - face_similarity:       crop around estimated head position
  - outfit_similarity:     crop torso region from character box
  - background_similarity: compare non-character regions

All scores are 0-1 where 1 = identical.  A panel is flagged as "drifted"
when its overall score drops below the threshold.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import DriftScore, PanelSpec

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.35


def _load_image(path: Path):
    from PIL import Image
    return Image.open(str(path)).convert("RGB")


def _resize_to_match(img, target_size):
    if img.size != target_size:
        img = img.resize(target_size)
    return img


def _image_to_gray_array(img):
    import numpy as np
    return np.array(img.convert("L"), dtype=float)


def _ssim_simple(arr_a, arr_b) -> float:
    """Simplified SSIM between two same-shape grayscale float arrays.

    Uses the standard SSIM formula with default constants.  This avoids
    requiring scikit-image as a dependency while giving a usable 0-1
    similarity measure.
    """
    import numpy as np

    if arr_a.shape != arr_b.shape:
        return 0.0

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = arr_a.mean()
    mu_b = arr_b.mean()
    sigma_a_sq = arr_a.var()
    sigma_b_sq = arr_b.var()
    sigma_ab = ((arr_a - mu_a) * (arr_b - mu_b)).mean()

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def _crop_region(img, bbox: list[int]):
    """Crop a region from a PIL Image using [x1, y1, x2, y2] bbox."""
    x1, y1, x2, y2 = bbox
    w, h = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


def _face_region(bbox: list[int]) -> list[int]:
    """Estimate head region from a full character bbox."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    return [x1, y1, x2, y1 + max(int(h * 0.30), 30)]


def _torso_region(bbox: list[int]) -> list[int]:
    """Estimate torso/outfit region from a full character bbox."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    top = y1 + int(h * 0.25)
    bot = y1 + int(h * 0.65)
    return [x1, top, x2, bot]


def _background_mask(img, boxes: list):
    """Create a background-only crop by masking out all character boxes.

    Returns a version of the image with character regions blacked out,
    so SSIM measures only background consistency.
    """
    from PIL import ImageDraw
    bg = img.copy()
    draw = ImageDraw.Draw(bg)
    for b in boxes:
        if b.bbox and len(b.bbox) == 4:
            draw.rectangle(b.bbox, fill=(0, 0, 0))
    return bg


def score_panel_drift(
    panel: PanelSpec,
    panel_image_path: Path,
    reference_image_path: Optional[Path],
    char_refs: Dict[str, Path],
) -> DriftScore:
    """Score visual drift for a single panel against its reference.

    If no reference_image_path is provided, compares against character
    reference sheets only (face + outfit crops).
    """
    import numpy as np

    try:
        current = _load_image(panel_image_path)
    except Exception as e:
        return DriftScore(panel_id=panel.panel_id, details=f"cannot load panel: {e}")

    face_scores: list[float] = []
    outfit_scores: list[float] = []
    bg_score = 0.0
    details_parts: list[str] = []

    # Compare against previous panel in continuity group
    if reference_image_path and reference_image_path.exists():
        try:
            ref_img = _load_image(reference_image_path)
            ref_img = _resize_to_match(ref_img, current.size)

            # Background comparison
            if panel.character_boxes:
                bg_cur = _background_mask(current, panel.character_boxes)
                bg_ref = _background_mask(ref_img, panel.character_boxes)
                bg_score = _ssim_simple(
                    _image_to_gray_array(bg_cur),
                    _image_to_gray_array(bg_ref),
                )
                details_parts.append(f"bg_ssim={bg_score:.3f}")

            # Per-character face+outfit from panel vs reference panel
            for box in panel.character_boxes:
                if not box.bbox or len(box.bbox) != 4:
                    continue
                face_crop_cur = _crop_region(current, _face_region(box.bbox))
                face_crop_ref = _crop_region(ref_img, _face_region(box.bbox))
                if face_crop_cur and face_crop_ref:
                    face_crop_ref = _resize_to_match(face_crop_ref, face_crop_cur.size)
                    s = _ssim_simple(
                        _image_to_gray_array(face_crop_cur),
                        _image_to_gray_array(face_crop_ref),
                    )
                    face_scores.append(s)

                torso_cur = _crop_region(current, _torso_region(box.bbox))
                torso_ref = _crop_region(ref_img, _torso_region(box.bbox))
                if torso_cur and torso_ref:
                    torso_ref = _resize_to_match(torso_ref, torso_cur.size)
                    s = _ssim_simple(
                        _image_to_gray_array(torso_cur),
                        _image_to_gray_array(torso_ref),
                    )
                    outfit_scores.append(s)

        except Exception as e:
            details_parts.append(f"ref comparison failed: {e}")

    # Compare character face crops against canonical reference sheets
    for box in panel.character_boxes:
        char_ref = char_refs.get(box.character_id)
        if not char_ref or not char_ref.exists():
            continue
        if not box.bbox or len(box.bbox) != 4:
            continue
        try:
            ref_sheet = _load_image(char_ref)
            face_crop = _crop_region(current, _face_region(box.bbox))
            if face_crop:
                ref_face = ref_sheet.crop((0, 0, ref_sheet.width // 2, ref_sheet.height // 3))
                ref_face = _resize_to_match(ref_face, face_crop.size)
                s = _ssim_simple(
                    _image_to_gray_array(face_crop),
                    _image_to_gray_array(ref_face),
                )
                face_scores.append(s)
                details_parts.append(f"{box.character_id}_face_vs_ref={s:.3f}")
        except Exception:
            pass

    face_avg = sum(face_scores) / len(face_scores) if face_scores else 0.0
    outfit_avg = sum(outfit_scores) / len(outfit_scores) if outfit_scores else 0.0

    overall = (face_avg * 0.4 + outfit_avg * 0.3 + bg_score * 0.3)
    drifted = overall < DRIFT_THRESHOLD and (
        reference_image_path is not None or bool(char_refs)
    )

    return DriftScore(
        panel_id=panel.panel_id,
        face_similarity=round(face_avg, 4),
        outfit_similarity=round(outfit_avg, 4),
        background_similarity=round(bg_score, 4),
        overall=round(overall, 4),
        drifted=drifted,
        details="; ".join(details_parts) if details_parts else "no references available",
    )


def validate_chapter_drift(
    panels: List[PanelSpec],
    panel_paths: Dict[int, Path],
    char_refs: Dict[str, Path],
    threshold: float = DRIFT_THRESHOLD,
) -> List[DriftScore]:
    """Score all panels in a chapter. Returns the full list of DriftScores."""
    scores: List[DriftScore] = []
    group_prev_image: Dict[str, Path] = {}

    for panel in panels:
        ref_path = None
        if panel.reference_panel_id is not None:
            ref_path = panel_paths.get(panel.reference_panel_id)

        if ref_path is None and panel.continuity_group in group_prev_image:
            ref_path = group_prev_image[panel.continuity_group]

        p_path = panel_paths.get(panel.panel_id)
        if not p_path:
            scores.append(DriftScore(
                panel_id=panel.panel_id, details="panel image not found",
            ))
            continue

        ds = score_panel_drift(panel, p_path, ref_path, char_refs)
        ds.drifted = ds.overall < threshold and (ref_path is not None or bool(char_refs))
        scores.append(ds)

        if panel.continuity_group:
            group_prev_image[panel.continuity_group] = p_path

    drifted_count = sum(1 for s in scores if s.drifted)
    if drifted_count:
        logger.warning(
            "Drift check: %d/%d panels drifted (threshold %.2f)",
            drifted_count, len(scores), threshold,
        )
    else:
        logger.info("Drift check: all %d panels within threshold", len(scores))

    return scores
