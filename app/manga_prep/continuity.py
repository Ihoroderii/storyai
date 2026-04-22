"""Visual continuity checker: validate chapter prose and PanelSpec against canon."""
from __future__ import annotations
import logging
import re
from typing import List

from ..models import WorldBible, Character, VisualProfile
from .models import PanelSpec, ContinuityIssue

logger = logging.getLogger(__name__)

# Appearance words that signal a description of a character's look
_APPEARANCE_PATTERNS = [
    (r"hair", "hair_color"),
    (r"eyes?(?:\s+color)?", "eye_color"),
    (r"outfit|wore|wearing|dressed|tunic|cloak|armor|jacket|robe", "outfit_base"),
    (r"skin|complexion", "skin_tone"),
]

# Color vocabulary for fuzzy matching
_COLOR_GROUPS = {
    "blue": {"blue", "azure", "cobalt", "sapphire", "cerulean", "cyan", "indigo", "navy", "teal"},
    "red": {"red", "crimson", "scarlet", "ruby", "vermillion", "maroon", "burgundy", "cherry"},
    "green": {"green", "emerald", "jade", "olive", "lime", "forest green", "viridian"},
    "orange": {"orange", "amber", "tangerine", "ginger", "copper"},
    "yellow": {"yellow", "gold", "golden", "blonde", "sandy"},
    "purple": {"purple", "violet", "lavender", "magenta", "plum", "lilac"},
    "pink": {"pink", "rose", "salmon", "fuchsia"},
    "black": {"black", "jet black", "ebony", "dark", "raven"},
    "white": {"white", "silver", "platinum", "pale", "snow", "ivory"},
    "brown": {"brown", "chestnut", "chocolate", "hazel", "auburn", "mahogany", "brunette"},
}


def _normalize_color(text: str) -> str:
    """Map a color description to a canonical color group."""
    text_low = text.lower().strip()
    for group, variants in _COLOR_GROUPS.items():
        for v in variants:
            if v in text_low:
                return group
    return text_low


def _colors_conflict(canon_color: str, found_color: str) -> bool:
    """Check if two color descriptions actually conflict (not just different words for same hue)."""
    if not canon_color or not found_color:
        return False
    c1 = _normalize_color(canon_color)
    c2 = _normalize_color(found_color)
    if c1 == c2:
        return False
    # "bright blue" and "blue" should not conflict
    if c1 in c2 or c2 in c1:
        return False
    return True


def _find_color_near_word(text: str, anchor: str) -> str | None:
    """Find a color word within ~5 words of an anchor word in text."""
    all_colors = set()
    for variants in _COLOR_GROUPS.values():
        all_colors.update(variants)

    pattern = rf'(\w+(?:\s+\w+)?)\s+{re.escape(anchor)}'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        candidate = match.group(1).lower()
        for color in all_colors:
            if color in candidate:
                return candidate
    pattern2 = rf'{re.escape(anchor)}\s+(?:was|were|is|are)?\s*(\w+(?:\s+\w+)?)'
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        candidate = match.group(1).lower()
        for color in all_colors:
            if color in candidate:
                return candidate
    return None


def check_character_visual(
    character: Character,
    chapter_text: str,
) -> List[ContinuityIssue]:
    """Check if chapter text contradicts a character's visual canon."""
    issues: List[ContinuityIssue] = []
    v = character.visual
    name = character.name
    text_low = chapter_text.lower()

    # Match on full name or first name (at least 3 chars)
    name_variants = [name.lower()]
    first_name = name.split()[0]
    if len(first_name) >= 3:
        name_variants.append(first_name.lower())

    if not any(variant in text_low for variant in name_variants):
        return issues

    # Find paragraphs that mention this character (any name variant)
    paragraphs = chapter_text.split("\n")
    char_paragraphs = [
        p for p in paragraphs
        if any(variant in p.lower() for variant in name_variants)
    ]
    char_text = " ".join(char_paragraphs)

    # Check hair color
    if v.hair_color:
        found = _find_color_near_word(char_text, "hair")
        if found and _colors_conflict(v.hair_color, found):
            issues.append(ContinuityIssue(
                character_id=name,
                field="hair_color",
                canon_value=v.hair_color,
                chapter_value=found,
                severity="error",
            ))

    # Check eye color
    if v.eye_color:
        found = _find_color_near_word(char_text, "eyes")
        if not found:
            found = _find_color_near_word(char_text, "eye")
        if found and _colors_conflict(v.eye_color, found):
            issues.append(ContinuityIssue(
                character_id=name,
                field="eye_color",
                canon_value=v.eye_color,
                chapter_value=found,
                severity="error",
            ))

    return issues


def check_visual_continuity(
    bible: WorldBible,
    chapter_text: str,
    chapter_num: int,
) -> List[ContinuityIssue]:
    """Check all visual canon against a chapter's text.

    Returns a list of continuity issues found. Empty list = all clear.
    """
    all_issues: List[ContinuityIssue] = []

    for character in bible.main_characters:
        issues = check_character_visual(character, chapter_text)
        all_issues.extend(issues)

    if all_issues:
        logger.warning(
            "Chapter %d: found %d visual continuity issue(s)",
            chapter_num, len(all_issues),
        )
        for issue in all_issues:
            logger.warning(
                "  %s.%s: canon='%s' chapter='%s' [%s]",
                issue.character_id, issue.field,
                issue.canon_value, issue.chapter_value,
                issue.severity,
            )
    else:
        logger.info("Chapter %d: visual continuity OK", chapter_num)

    return all_issues


# ---------------------------------------------------------------------------
# PanelSpec-level validation
# ---------------------------------------------------------------------------

def validate_panel_specs(
    panels: List[PanelSpec],
    bible: WorldBible,
) -> List[ContinuityIssue]:
    """Validate extracted PanelSpec list against the world bible.

    Checks: unknown characters, unknown locations, missing anchor objects,
    shot-type repetition, and empty panels.
    """
    char_names = {c.name.lower() for c in bible.main_characters}
    first_names = set()
    for c in bible.main_characters:
        first = c.name.split()[0].lower()
        if len(first) >= 3:
            first_names.add(first)
    known_names = char_names | first_names

    loc_map = {loc.location_id: loc for loc in bible.locations} if bible.locations else {}

    issues: List[ContinuityIssue] = []

    prev_shot = None
    consecutive_same_shot = 0

    for panel in panels:
        pid = panel.panel_id

        # -- Character lineup validation --
        if not panel.characters:
            issues.append(ContinuityIssue(
                field="characters",
                canon_value="at least 1 character expected",
                chapter_value="empty character list",
                severity="warning",
            ))
        for ch in panel.characters:
            cid_low = ch.character_id.lower()
            first_low = ch.character_id.split()[0].lower() if ch.character_id else ""
            if cid_low not in known_names and first_low not in known_names:
                issues.append(ContinuityIssue(
                    character_id=ch.character_id,
                    field="character_id",
                    canon_value=f"known: {', '.join(sorted(char_names))}",
                    chapter_value=f"panel {pid}: '{ch.character_id}' not in bible",
                    severity="warning",
                ))

        # -- Location validation --
        if panel.location_id and loc_map and panel.location_id not in loc_map:
            issues.append(ContinuityIssue(
                location_id=panel.location_id,
                field="location_id",
                canon_value=f"known: {', '.join(sorted(loc_map.keys()))}",
                chapter_value=f"panel {pid}: '{panel.location_id}' not in bible",
                severity="warning",
            ))

        # -- Anchor object check --
        # Checks prompt_positive (if built), location_description, and scene_summary
        if panel.location_id and panel.location_id in loc_map:
            loc = loc_map[panel.location_id]
            if loc.anchor_objects:
                searchable = " ".join(filter(None, [
                    (panel.prompt_positive or "").lower(),
                    (panel.location_description or "").lower(),
                    (panel.scene_summary or "").lower(),
                ]))
                missing = [
                    obj for obj in loc.anchor_objects
                    if obj.lower() not in searchable
                ]
                if missing:
                    issues.append(ContinuityIssue(
                        location_id=panel.location_id,
                        field="anchor_objects",
                        canon_value=f"required: {', '.join(loc.anchor_objects)}",
                        chapter_value=f"panel {pid} missing: {', '.join(missing)}",
                        severity="info",
                    ))

        # -- Shot progression monotony --
        if panel.shot_type == prev_shot:
            consecutive_same_shot += 1
        else:
            consecutive_same_shot = 0
        prev_shot = panel.shot_type

        if consecutive_same_shot >= 3:
            issues.append(ContinuityIssue(
                field="shot_type",
                canon_value="varied shots recommended",
                chapter_value=f"panels {pid - consecutive_same_shot}–{pid}: "
                              f"{consecutive_same_shot + 1}× '{panel.shot_type}'",
                severity="info",
            ))

    if issues:
        logger.warning("PanelSpec validation: %d issue(s) found", len(issues))
        for iss in issues:
            logger.warning(
                "  [%s] %s: canon='%s' found='%s'",
                iss.severity, iss.field, iss.canon_value, iss.chapter_value,
            )
    else:
        logger.info("PanelSpec validation: all clear")

    return issues
