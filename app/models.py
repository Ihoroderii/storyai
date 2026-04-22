from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class VisualProfile(BaseModel):
    """Stable visual identity for a character -- shared between narrative and manga."""
    hair_color: str = ""
    hair_style: str = ""
    eye_color: str = ""
    face_shape: str = ""
    skin_tone: str = ""
    age_appearance: str = ""
    body_type: str = ""
    silhouette: str = ""
    height: str = ""
    outfit_base: str = ""
    signature_prop: str = ""
    expression_style: str = ""
    distinguishing_features: List[str] = Field(default_factory=list)
    props: List[str] = Field(default_factory=list)
    color_palette: List[str] = Field(default_factory=list)
    reference_prompt: str = ""
    reference_image: str = ""


class Character(BaseModel):
    name: str
    age: Optional[int] = None
    role: str  # protagonist, antagonist, mentor...
    goals: List[str] = Field(default_factory=list)
    fears: List[str] = Field(default_factory=list)
    personality: List[str] = Field(default_factory=list)
    speech_style: List[str] = Field(default_factory=list)
    relationships: Dict[str, str] = Field(default_factory=dict)
    visual: VisualProfile = Field(default_factory=VisualProfile)


class SceneLayout(BaseModel):
    """Spatial structure for a recurring location -- enables repeatable camera geography."""
    left_landmark: str = ""
    right_landmark: str = ""
    center_landmark: str = ""
    depth_order: List[str] = Field(default_factory=list, description="front-to-back layer ordering")
    entrance_points: List[str] = Field(default_factory=list)


class Location(BaseModel):
    """Stable visual identity for a recurring location."""
    location_id: str
    name: str
    description: str = ""
    anchor_objects: List[str] = Field(default_factory=list)
    lighting: str = ""
    mood: str = ""
    time_of_day: str = ""
    palette: List[str] = Field(default_factory=list)
    layout_notes: str = ""
    layout: SceneLayout = Field(default_factory=SceneLayout)
    allowed_camera_angles: List[str] = Field(default_factory=list)
    reference_image: str = ""


class WorldBible(BaseModel):
    title: str
    genre: str
    premise: str
    setting: str
    rules: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    main_characters: List[Character] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    style_guide: List[str] = Field(default_factory=list)


class ChapterPlan(BaseModel):
    number: int
    title: str
    goal: str
    conflict: str
    twist: str
    hook: str
    pov: str  # POV character name


class Outline(BaseModel):
    volume_title: str
    chapters: List[ChapterPlan]


class ChapterDraft(BaseModel):
    number: int
    title: str
    text: str


class ChapterSummary(BaseModel):
    chapter: int
    summary_short: str
    summary_detailed: str
    new_facts: List[str] = Field(default_factory=list)
    visual_notes: List[str] = Field(default_factory=list)
    continuity_conflicts: List[str] = Field(default_factory=list)
