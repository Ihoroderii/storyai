import json
import re
from pathlib import Path
from .models import WorldBible, Outline, ChapterDraft, ChapterSummary
from .prompts import render_prompt
from .llm import call_llm
from .db import NovelDB
from .similarity import SimilarityChecker, print_similarity_report


def ensure_dirs(base: Path):
    (base / "data" / "chapters").mkdir(parents=True, exist_ok=True)


def extract_json(raw: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if not raw or not raw.strip():
        raise ValueError("Empty response from LLM")
    
    # Try to extract JSON from markdown code blocks (greedy to handle nested braces)
    json_pattern = r'```(?:json)?\s*(\{.*\})\s*```'
    match = re.search(json_pattern, raw, re.DOTALL)
    if match:
        return match.group(1)
    
    # Return as-is if no code blocks found
    return raw.strip()


def generate_bible(env, premise: str, genre: str) -> WorldBible:
    prompt = render_prompt(env, "bible.j2", premise=premise, genre=genre)
    raw = call_llm(prompt)
    
    try:
        cleaned = extract_json(raw)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\n❌ Failed to parse bible JSON")
        print(f"Error: {e}")
        print(f"\nRaw response (first 500 chars):\n{raw[:500]}")
        raise
    
    return WorldBible.model_validate(data)


def generate_outline(env, bible: WorldBible) -> Outline:
    prompt = render_prompt(env, "outline.j2", bible_json=bible.model_dump_json(indent=2))
    raw = call_llm(prompt)
    
    try:
        cleaned = extract_json(raw)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\n❌ Failed to parse outline JSON")
        print(f"Error: {e}")
        print(f"\nRaw response (first 500 chars):\n{raw[:500]}")
        raise
    
    return Outline.model_validate(data)


def _build_visual_canon(bible: WorldBible) -> list[str]:
    """Flatten bible visual profiles into concise constraint strings for the LLM."""
    lines = []
    for ch in bible.main_characters:
        v = ch.visual
        if not v.hair_color and not v.reference_prompt:
            continue
        parts = [f"{ch.name}:"]
        if v.hair_color:
            parts.append(f"hair={v.hair_color}")
            if v.hair_style:
                parts[-1] += f" ({v.hair_style})"
        if v.eye_color:
            parts.append(f"eyes={v.eye_color}")
        if v.face_shape:
            parts.append(f"face={v.face_shape}")
        if v.skin_tone:
            parts.append(f"skin={v.skin_tone}")
        if v.outfit_base:
            parts.append(f"outfit={v.outfit_base}")
        if v.expression_style:
            parts.append(f"expression_style={v.expression_style}")
        if v.distinguishing_features:
            parts.append(f"features={', '.join(v.distinguishing_features)}")
        if v.props:
            parts.append(f"props={', '.join(v.props)}")
        lines.append(" ".join(parts))
    return lines


def _build_location_canon(bible: WorldBible) -> list[str]:
    """Flatten bible locations into constraint strings for the LLM."""
    lines = []
    for loc in bible.locations:
        parts = [f"{loc.name} ({loc.location_id}):"]
        if loc.description:
            parts.append(loc.description)
        if loc.anchor_objects:
            parts.append(f"anchors={', '.join(loc.anchor_objects)}")
        if loc.lighting:
            parts.append(f"lighting={loc.lighting}")
        if loc.mood:
            parts.append(f"mood={loc.mood}")
        if loc.layout_notes:
            parts.append(f"layout={loc.layout_notes}")
        lines.append(" | ".join(parts))
    return lines


def write_chapter(env, bible: WorldBible, outline: Outline, db: NovelDB, chapter_number: int, 
                  check_similarity: bool = True, chapters_dir: Path = None) -> ChapterDraft:
    plan = next(c for c in outline.chapters if c.number == chapter_number)
    recent_summaries = db.get_recent_summaries(limit=3)
    canon_facts = db.get_canon_facts(limit=60)
    visual_canon = _build_visual_canon(bible)
    location_canon = _build_location_canon(bible)

    prompt = render_prompt(
        env,
        "chapter.j2",
        chapter_number=chapter_number,
        bible_json=bible.model_dump_json(indent=2),
        chapter_plan_json=plan.model_dump_json(indent=2),
        recent_summaries=recent_summaries,
        canon_facts=canon_facts,
        visual_canon=visual_canon,
        location_canon=location_canon,
    )
    text = call_llm(prompt).strip()
    
    # Run similarity checks if enabled
    if check_similarity and chapters_dir and chapter_number > 1:
        print("🔍 Checking for self-similarity...")
        checker = SimilarityChecker(chapters_dir)
        result = checker.check_all(text, chapter_number)
        print_similarity_report(result)
        
        # Warning if issues found, but don't block generation
        if result['has_issues']:
            print("⚠️  Similarity issues detected but continuing generation")
            print("💡 You may want to regenerate this chapter if quality is low\n")
    
    return ChapterDraft(number=chapter_number, title=plan.title, text=text)


def summarize_and_store(env, db: NovelDB, chapter: ChapterDraft,
                        bible: WorldBible | None = None) -> ChapterSummary:
    visual_canon = _build_visual_canon(bible) if bible else []

    prompt = render_prompt(
        env,
        "sum_and_extract.j2",
        chapter_number=chapter.number,
        chapter_text=chapter.text,
        visual_canon=visual_canon,
    )
    raw = call_llm(prompt)
    
    try:
        cleaned = extract_json(raw)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\n❌ Failed to parse summary JSON")
        print(f"Error: {e}")
        print(f"\nRaw response (first 500 chars):\n{raw[:500]}")
        raise
    
    summary = ChapterSummary.model_validate(data)

    if summary.continuity_conflicts:
        print(f"\n⚠️  CONTINUITY CONFLICTS in chapter {chapter.number}:")
        for conflict in summary.continuity_conflicts:
            print(f"   {conflict}")
        print()

    if summary.visual_notes:
        print(f"📝 Visual notes from chapter {chapter.number}:")
        for note in summary.visual_notes:
            print(f"   {note}")
        print()

    db.upsert_summary(summary.chapter, summary.summary_short, summary.summary_detailed)
    db.add_facts([(f, summary.chapter) for f in summary.new_facts])
    db.store_visual_audit(
        summary.chapter,
        summary.visual_notes,
        summary.continuity_conflicts,
    )
    return summary
