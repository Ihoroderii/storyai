"""Build stable, character-aware image generation prompts from PanelSpec + visual canon.

Prompt structure (ordered for Stable Diffusion attention):
  1. Shot framing + camera angle
  2. Character appearance (from canon VisualProfile)
  3. Character action / expression / pose
  4. Location anchors + lighting + mood
  5. Scene layout (spatial landmarks)
  6. Style suffix
"""
from __future__ import annotations
import logging
from typing import Dict, List

from ..models import WorldBible, Character, Location, VisualProfile
from .models import PanelSpec

logger = logging.getLogger(__name__)

_STYLE_SUFFIX = (
    "Korean manhwa style, detailed coloring, dramatic lighting, "
    "highly detailed faces, webtoon format, digital art, clean lineart, "
    "vibrant colors, professional coloring, high contrast, cinematic composition"
)

_NEGATIVE_BASE = (
    "blurry, deformed face, extra eyes, extra limbs, bad anatomy, ugly, "
    "distorted, chibi style, super deformed, sketch, watercolor, "
    "text, words, letters, speech bubble"
)

_NEGATIVE_FANTASY_EXTRA = (
    "modern clothing, jeans, t-shirt, sneakers, hoodie, "
    "contemporary setting, car, phone, laptop screen"
)

_GENRE_MODERN = {"modern", "slice of life", "contemporary", "romance", "thriller", "sci-fi", "cyberpunk"}

_SHOT_MODIFIERS = {
    "wide": "wide shot, full body visible, environment clearly shown",
    "medium": "medium shot, waist-up framing",
    "close-up": "close-up shot, face and shoulders, emotional detail",
    "extreme close-up": "extreme close-up, eyes and expression only",
}

_ANGLE_MODIFIERS = {
    "eye level": "eye level camera angle",
    "low angle": "low angle shot, looking up, dramatic perspective",
    "high angle": "high angle shot, looking down",
    "bird's eye": "bird's eye view, top-down perspective",
    "dutch angle": "dutch angle, tilted camera, dynamic tension",
}


def _is_modern_genre(genre: str) -> bool:
    genre_low = genre.lower()
    return any(g in genre_low for g in _GENRE_MODERN)


def _char_visual_prompt(character: Character) -> str:
    """Build a stable visual prompt fragment for a character from their canon profile."""
    v = character.visual
    if v.reference_prompt:
        extra = []
        if v.face_shape:
            extra.append(f"{v.face_shape} face shape")
        if v.expression_style:
            extra.append(f"{v.expression_style} expression style")
        if extra:
            return f"{v.reference_prompt}, " + ", ".join(extra)
        return v.reference_prompt

    parts = []
    if v.age_appearance:
        parts.append(v.age_appearance)
    if v.silhouette:
        parts.append(v.silhouette)
    elif v.body_type:
        parts.append(v.body_type)
    if v.hair_color:
        hair = v.hair_color
        if v.hair_style:
            hair += f" {v.hair_style}"
        parts.append(f"{hair} hair")
    if v.eye_color:
        parts.append(f"{v.eye_color} eyes")
    if v.face_shape:
        parts.append(f"{v.face_shape} face shape")
    if v.skin_tone:
        parts.append(f"{v.skin_tone} skin")
    if v.height:
        parts.append(v.height)
    if v.outfit_base:
        parts.append(v.outfit_base)
    if v.signature_prop:
        parts.append(v.signature_prop)
    if v.expression_style:
        parts.append(f"{v.expression_style} expression style")
    if v.distinguishing_features:
        parts.extend(v.distinguishing_features)
    if v.props:
        parts.extend(v.props)

    return ", ".join(parts) if parts else character.name


def _location_prompt(location: Location) -> str:
    """Build a visual prompt fragment for a location from canon."""
    parts = []
    if location.description:
        parts.append(location.description)
    if location.lighting:
        parts.append(location.lighting)
    if location.mood:
        parts.append(f"{location.mood} atmosphere")
    if location.anchor_objects:
        parts.append(", ".join(location.anchor_objects))
    if location.layout_notes:
        parts.append(location.layout_notes)
    # Structured layout landmarks
    ly = location.layout
    layout_bits = []
    if ly.left_landmark:
        layout_bits.append(f"{ly.left_landmark} on the left")
    if ly.right_landmark:
        layout_bits.append(f"{ly.right_landmark} on the right")
    if ly.center_landmark:
        layout_bits.append(f"{ly.center_landmark} in the center")
    if ly.depth_order:
        layout_bits.append(f"depth: {' → '.join(ly.depth_order)}")
    if layout_bits:
        parts.append(", ".join(layout_bits))
    if location.palette:
        parts.append(f"color palette: {', '.join(location.palette)}")
    return ", ".join(parts) if parts else location.name


def _build_negative_prompt(genre: str) -> str:
    """Build a negative prompt, adding fashion guard for non-modern genres."""
    if _is_modern_genre(genre):
        return _NEGATIVE_BASE
    return f"{_NEGATIVE_BASE}, {_NEGATIVE_FANTASY_EXTRA}"


def build_panel_prompts(
    panels: List[PanelSpec],
    bible: WorldBible,
) -> List[PanelSpec]:
    """Fill prompt_positive and prompt_negative on each PanelSpec using visual canon.

    Returns the same panel objects, mutated with prompts set.
    """
    char_map: Dict[str, Character] = {}
    for c in bible.main_characters:
        char_map[c.name.lower()] = c
        char_map[c.name] = c

    loc_map: Dict[str, Location] = {}
    for loc in bible.locations:
        loc_map[loc.location_id] = loc
        loc_map[loc.name.lower()] = loc

    negative = _build_negative_prompt(bible.genre)

    for panel in panels:
        prompt_parts = []

        # 1. Shot framing + camera angle
        shot_mod = _SHOT_MODIFIERS.get(panel.shot_type, "")
        if shot_mod:
            prompt_parts.append(shot_mod)

        # Resolve location first -- needed to validate camera angle
        loc = (
            loc_map.get(panel.location_id) or loc_map.get(panel.location_id.lower())
            if panel.location_id else None
        )

        # Enforce allowed_camera_angles if the location defines them
        cam = panel.camera_angle
        if loc and loc.allowed_camera_angles:
            if cam not in loc.allowed_camera_angles:
                original = cam
                cam = loc.allowed_camera_angles[0]
                logger.debug(
                    "Panel %d: camera '%s' not allowed at %s, using '%s'",
                    panel.panel_id, original, loc.location_id, cam,
                )
                panel.camera_angle = cam
        angle_mod = _ANGLE_MODIFIERS.get(cam, "")
        if angle_mod:
            prompt_parts.append(angle_mod)

        # 2–3. Characters: appearance (canon) + action/expression/pose
        for ch_on_screen in panel.characters:
            char = char_map.get(ch_on_screen.character_id) or char_map.get(ch_on_screen.character_id.lower())
            if char:
                vis = _char_visual_prompt(char)
                parts = [vis]
                if ch_on_screen.expression:
                    parts.append(f"{ch_on_screen.expression} expression")
                if ch_on_screen.action:
                    parts.append(ch_on_screen.action)
                if ch_on_screen.pose:
                    parts.append(ch_on_screen.pose)
                prompt_parts.append(", ".join(parts))
            else:
                parts = [ch_on_screen.character_id]
                if ch_on_screen.expression:
                    parts.append(ch_on_screen.expression)
                if ch_on_screen.action:
                    parts.append(ch_on_screen.action)
                prompt_parts.append(", ".join(parts))

        # 4. Location: anchors + lighting + mood
        if loc:
            prompt_parts.append(_location_prompt(loc))
        elif panel.location_description:
            prompt_parts.append(panel.location_description)

        # 5. Time of day
        if panel.time_of_day:
            prompt_parts.append(f"{panel.time_of_day} lighting")

        # 6. Style suffix
        prompt_parts.append(_STYLE_SUFFIX)

        panel.prompt_positive = ", ".join(p for p in prompt_parts if p)
        panel.prompt_negative = negative

        logger.debug("Panel %d prompt: %s", panel.panel_id, panel.prompt_positive[:200])

    return panels
