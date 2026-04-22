"""Structured panel specification models for the manga preparation layer.

This is the shared contract used by all three projects:
  storyAI (manga_prep)  -> produces PanelSpec
  manga-ai-bot          -> renders from PanelSpec
  bubble (manhwa_bubbles) -> places bubbles using PanelSpec dialogues + character_boxes
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple


class CharacterBox(BaseModel):
    """Bounding box for a character in the rendered panel image.

    Populated after image generation (YOLO detection or layout estimation).
    Used by the bubble engine to anchor speech near the correct speaker.
    """
    character_id: str
    bbox: List[int] = Field(default_factory=list, description="[x1, y1, x2, y2] pixel coords")
    head: Optional[List[int]] = Field(default=None, description="[hx, hy] estimated head center")
    confidence: float = 0.0


class CharacterOnScreen(BaseModel):
    """A character present in a panel, with visual state."""
    character_id: str
    pose: str = ""
    expression: str = ""
    action: str = ""
    must_show: List[str] = Field(default_factory=list)


class PanelDialogue(BaseModel):
    """One speech bubble within a panel."""
    character_id: str
    text: str
    emotion: str = "normal"
    bubble_type: str = "speech"
    position_hint: str = ""


class StoryBeat(BaseModel):
    """A logical narrative beat used to pace chapter-to-manga adaptation."""
    beat_id: int
    summary: str = ""
    text: str = ""
    words: int = 0
    importance: str = "medium"
    characters: List[str] = Field(default_factory=list)
    location_id: str = ""
    recommended_panels: int = 1


class PanelSpec(BaseModel):
    """Full visual specification for one manga panel.

    This is the contract between manga_prep (storyAI) and
    manga-ai-bot (rendering).  manga-ai-bot should only render
    from PanelSpec data, not invent its own story context.
    """
    panel_id: int
    location_id: str = ""
    location_description: str = ""
    time_of_day: str = ""
    shot_type: str = "medium"
    camera_angle: str = "eye level"
    characters: List[CharacterOnScreen] = Field(default_factory=list)
    dialogues: List[PanelDialogue] = Field(default_factory=list)
    character_boxes: List[CharacterBox] = Field(default_factory=list)
    prompt_positive: str = ""
    prompt_negative: str = ""
    scene_summary: str = ""

    # --- Continuity fields ---
    reference_panel_id: Optional[int] = Field(
        default=None,
        description="panel_id of the most recent panel in the same continuity group "
                    "(used as primary img2img reference during generation)",
    )
    continuity_group: str = Field(
        default="",
        description="Key grouping panels that share location + character set. "
                    "Panels in the same group reuse seeds and reference each other.",
    )
    scene_state_in: Dict[str, str] = Field(
        default_factory=dict,
        description="Visual state entering this panel (character poses, lighting, objects held, etc.)",
    )
    scene_state_out: Dict[str, str] = Field(
        default_factory=dict,
        description="Visual state leaving this panel (updated after generation).",
    )
    seed_hint: Optional[int] = Field(
        default=None,
        description="Deterministic seed. Panels in the same continuity_group share a base seed "
                    "with small per-panel offsets to keep visual coherence.",
    )
    continuity_notes: List[str] = Field(
        default_factory=list,
        description="Free-form notes about what must stay consistent with the reference panel "
                    "(e.g. 'same outfit', 'wound on left arm visible').",
    )


class SceneState(BaseModel):
    """Persisted per-chapter visual state. Updated after every panel."""
    chapter: int = 0
    last_panel_id: int = 0
    group_seeds: Dict[str, int] = Field(default_factory=dict)
    group_last_panel: Dict[str, int] = Field(default_factory=dict)
    group_last_image: Dict[str, str] = Field(default_factory=dict)
    character_states: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    location_states: Dict[str, Dict[str, str]] = Field(default_factory=dict)


class DriftScore(BaseModel):
    """Result of post-generation visual similarity check for a single panel."""
    panel_id: int
    face_similarity: float = 0.0
    outfit_similarity: float = 0.0
    background_similarity: float = 0.0
    overall: float = 0.0
    drifted: bool = False
    details: str = ""


class ContinuityIssue(BaseModel):
    """A detected visual contradiction between chapter prose and canon."""
    character_id: str = ""
    location_id: str = ""
    field: str
    canon_value: str
    chapter_value: str
    severity: str = "warning"
