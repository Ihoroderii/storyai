from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Character(BaseModel):
    name: str
    age: Optional[int] = None
    role: str  # protagonist, antagonist, mentor...
    goals: List[str] = Field(default_factory=list)
    fears: List[str] = Field(default_factory=list)
    personality: List[str] = Field(default_factory=list)
    speech_style: List[str] = Field(default_factory=list)
    relationships: Dict[str, str] = Field(default_factory=dict)  # other_name -> relation


class WorldBible(BaseModel):
    title: str
    genre: str
    premise: str
    setting: str
    rules: List[str] = Field(default_factory=list)  # magic/system rules, tech limits
    themes: List[str] = Field(default_factory=list)
    main_characters: List[Character] = Field(default_factory=list)
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


class CanonFact(BaseModel):
    fact: str
    source_chapter: int


class ChapterDraft(BaseModel):
    number: int
    title: str
    text: str


class ChapterSummary(BaseModel):
    chapter: int
    summary_short: str
    summary_detailed: str
    new_facts: List[str] = Field(default_factory=list)  # store as strings first
