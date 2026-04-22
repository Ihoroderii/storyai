"""Reference image management for character and location consistency.

Generates, stores, and loads reference images used for:
  1. Visual verification (human QA of character canon)
  2. IP-Adapter / reference-only conditioning when supported by the pipeline
  3. Post-generation face comparison

Directory layout:
  data/references/
    characters/
      kaito_yamamoto.png     (character reference sheet)
    locations/
      forest_of_algorithms.png
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_REF_ROOT = Path(__file__).parent.parent.parent / "data" / "references"
_CHAR_DIR = _REF_ROOT / "characters"
_LOC_DIR = _REF_ROOT / "locations"


def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")


def ensure_dirs():
    _CHAR_DIR.mkdir(parents=True, exist_ok=True)
    _LOC_DIR.mkdir(parents=True, exist_ok=True)


def character_ref_path(name: str) -> Path:
    return _CHAR_DIR / f"{_safe_filename(name)}.png"


def location_ref_path(location_id: str) -> Path:
    return _LOC_DIR / f"{location_id}.png"


# ---------------------------------------------------------------------------
# Reference image generation
# ---------------------------------------------------------------------------

def generate_character_reference(
    character,
    hf_client=None,
    sd_model: str = "black-forest-labs/FLUX.1-schnell",
    overwrite: bool = False,
) -> Optional[Path]:
    """Generate a reference sheet image for a character using their visual canon.

    Uses the character's reference_prompt or builds one from the VisualProfile.
    Returns the path to the saved image, or None on failure.
    """
    out = character_ref_path(character.name)
    if out.exists() and not overwrite:
        logger.info("Reference image already exists: %s", out)
        return out

    from .prompt_builder import _char_visual_prompt
    v = character.visual
    expression_notes = v.expression_style or "neutral, focused, surprised"
    face_notes = f"{v.face_shape} face shape, " if v.face_shape else ""
    prompt = (
        "character design reference pack, multi-view turnaround sheet, "
        "front view, side view, 3/4 view, full body, portrait close-up, "
        f"expression sheet showing {expression_notes}, "
        "clean white background, labeled design-sheet composition, "
        f"{face_notes}{_char_visual_prompt(character)}, "
        "same exact character in every view, consistent face, consistent outfit, "
        "Korean manhwa style, clean lineart, highly detailed, professional character sheet"
    )
    negative = (
        "blurry, deformed, extra limbs, bad anatomy, background clutter, "
        "different characters, inconsistent face, inconsistent outfit, cropped body, busy background"
    )

    ensure_dirs()

    if hf_client is not None:
        try:
            image = hf_client.text_to_image(
                prompt, model=sd_model, negative_prompt=negative,
                width=1536, height=1024,
            )
            image.save(str(out), quality=95)
            logger.info("Generated character reference: %s", out)
            return out
        except Exception as e:
            logger.warning("Failed to generate character reference for %s: %s", character.name, e)
            return None

    logger.info("No HF client — skipping reference generation for %s", character.name)
    return None


def generate_location_reference(
    location,
    hf_client=None,
    sd_model: str = "black-forest-labs/FLUX.1-schnell",
    overwrite: bool = False,
) -> Optional[Path]:
    """Generate a reference image for a location using its visual canon."""
    out = location_ref_path(location.location_id)
    if out.exists() and not overwrite:
        logger.info("Reference image already exists: %s", out)
        return out

    from .prompt_builder import _location_prompt
    prompt = (
        f"establishing shot, wide angle, environment concept art, "
        f"{_location_prompt(location)}, "
        f"Korean manhwa style, detailed, vibrant, high quality"
    )
    negative = (
        "blurry, deformed, people, characters, text, words"
    )

    ensure_dirs()

    if hf_client is not None:
        try:
            image = hf_client.text_to_image(
                prompt, model=sd_model, negative_prompt=negative,
                width=1024, height=768,
            )
            image.save(str(out), quality=95)
            logger.info("Generated location reference: %s", out)
            return out
        except Exception as e:
            logger.warning("Failed to generate location reference for %s: %s", location.name, e)
            return None

    logger.info("No HF client — skipping reference generation for %s", location.name)
    return None


def generate_all_references(bible, hf_client=None, sd_model: str = "black-forest-labs/FLUX.1-schnell"):
    """Generate reference images for all characters and locations in the bible."""
    results = {"characters": {}, "locations": {}}

    for char in bible.main_characters:
        path = generate_character_reference(char, hf_client, sd_model)
        if path:
            results["characters"][char.name] = str(path)
            char.visual.reference_image = str(path)

    for loc in bible.locations:
        path = generate_location_reference(loc, hf_client, sd_model)
        if path:
            results["locations"][loc.location_id] = str(path)
            loc.reference_image = str(path)

    logger.info(
        "Generated %d character + %d location references",
        len(results["characters"]), len(results["locations"]),
    )
    return results


# ---------------------------------------------------------------------------
# Reference loading for conditioning
# ---------------------------------------------------------------------------

def load_character_references(bible) -> Dict[str, Path]:
    """Return {character_name: Path} for all characters that have reference images on disk."""
    refs = {}
    for char in bible.main_characters:
        p = character_ref_path(char.name)
        if p.exists():
            refs[char.name] = p
    return refs


def load_location_references(bible) -> Dict[str, Path]:
    """Return {location_id: Path} for all locations that have reference images on disk."""
    refs = {}
    for loc in bible.locations:
        p = location_ref_path(loc.location_id)
        if p.exists():
            refs[loc.location_id] = p
    return refs


def build_panel_reference_image(
    panel,
    char_refs: Dict[str, Path],
    loc_refs: Dict[str, Path],
    size: tuple[int, int] = (768, 1024),
):
    """Build a simple composite reference image for API image-to-image guidance.

    The composition favors the location reference as the background and places
    character reference thumbnails along the bottom edge. This is intentionally
    lightweight and deterministic so that API mode can still benefit from the
    same reference assets used by the local IP-Adapter path.
    """
    from PIL import Image, ImageOps

    width, height = size
    location_ref = loc_refs.get(panel.location_id) if panel.location_id else None
    char_images = []

    for ch in panel.characters:
        ref_path = char_refs.get(ch.character_id)
        if ref_path and ref_path.exists():
            char_images.append(ref_path)

    if not location_ref and not char_images:
        return None

    if location_ref and location_ref.exists():
        base = Image.open(str(location_ref)).convert("RGB")
        canvas = ImageOps.fit(base, size, method=Image.Resampling.LANCZOS)
    else:
        canvas = Image.new("RGB", size, "white")

    if not char_images:
        return canvas

    thumb_count = min(3, len(char_images))
    tray_h = max(180, height // 4)
    thumb_w = width // thumb_count
    y0 = height - tray_h

    # Light bottom tray so character references remain readable over location art.
    tray = Image.new("RGBA", (width, tray_h), (255, 255, 255, 210))
    canvas = canvas.convert("RGBA")
    canvas.alpha_composite(tray, (0, y0))

    for idx, ref_path in enumerate(char_images[:thumb_count]):
        ref_img = Image.open(str(ref_path)).convert("RGB")
        thumb = ImageOps.fit(
            ref_img,
            (thumb_w - 24, tray_h - 24),
            method=Image.Resampling.LANCZOS,
        )
        x = idx * thumb_w + 12
        y = y0 + 12
        canvas.alpha_composite(thumb.convert("RGBA"), (x, y))

    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# IP-Adapter / reference conditioning hooks
# ---------------------------------------------------------------------------

def apply_reference_conditioning(
    pipeline,
    panel,
    char_refs: Dict[str, Path],
    loc_refs: Dict[str, Path],
) -> dict:
    """Attempt to apply IP-Adapter or reference conditioning to the pipeline.

    Returns extra kwargs to pass to pipeline() call.

    This is a best-effort integration:
    - If the pipeline has ip_adapter loaded, uses it
    - Otherwise returns empty dict (no conditioning)
    """
    extra_kwargs = {}

    # Check if pipeline supports IP-Adapter
    has_ip_adapter = hasattr(pipeline, "set_ip_adapter_scale")
    if not has_ip_adapter:
        return extra_kwargs

    from PIL import Image

    # Collect reference images for characters in this panel
    ref_images = []
    for ch in panel.characters:
        ref_path = char_refs.get(ch.character_id)
        if ref_path and ref_path.exists():
            ref_images.append(Image.open(str(ref_path)).convert("RGB"))

    # Add location reference if available
    loc_ref = loc_refs.get(panel.location_id) if panel.location_id else None
    if loc_ref and loc_ref.exists():
        ref_images.append(Image.open(str(loc_ref)).convert("RGB"))

    if ref_images:
        try:
            pipeline.set_ip_adapter_scale(0.6)
            extra_kwargs["ip_adapter_image"] = ref_images if len(ref_images) > 1 else ref_images[0]
            logger.info(
                "Panel %d: applying IP-Adapter with %d reference(s)",
                panel.panel_id, len(ref_images),
            )
        except Exception as e:
            logger.debug("IP-Adapter conditioning not available: %s", e)

    return extra_kwargs
