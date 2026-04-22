"""manga_prep -- visual adaptation layer between storyAI narrative and manga rendering.

Responsibilities:
  - Extract structured scene data from chapter prose
  - Validate visual continuity against character/location canon
  - Build stable, character-aware image generation prompts
  - Produce PanelSpec objects that manga-ai-bot renders
  - Track scene continuity across panels within a chapter
  - Detect visual drift in generated images
"""
from .models import (
    StoryBeat, PanelSpec, CharacterOnScreen, PanelDialogue, CharacterBox,
    SceneState, DriftScore,
)
from .extractor import extract_panels, estimate_auto_panel_count, PanelExtractionError
from .continuity import check_visual_continuity, validate_panel_specs
from .prompt_builder import build_panel_prompts
from .scene_continuity import assign_continuity_groups

__all__ = [
    "PanelSpec",
    "StoryBeat",
    "CharacterOnScreen",
    "PanelDialogue",
    "CharacterBox",
    "SceneState",
    "DriftScore",
    "extract_panels",
    "estimate_auto_panel_count",
    "PanelExtractionError",
    "check_visual_continuity",
    "validate_panel_specs",
    "build_panel_prompts",
    "assign_continuity_groups",
]
