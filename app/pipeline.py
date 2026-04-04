import json
import re
from pathlib import Path
from pydantic import TypeAdapter
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
    
    # Try to extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
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


def write_chapter(env, bible: WorldBible, outline: Outline, db: NovelDB, chapter_number: int, 
                  check_similarity: bool = True, chapters_dir: Path = None) -> ChapterDraft:
    plan = next(c for c in outline.chapters if c.number == chapter_number)
    recent_summaries = db.get_recent_summaries(limit=3)
    canon_facts = db.get_canon_facts(limit=60)

    prompt = render_prompt(
        env,
        "chapter.j2",
        chapter_number=chapter_number,
        bible_json=bible.model_dump_json(indent=2),
        chapter_plan_json=plan.model_dump_json(indent=2),
        recent_summaries=recent_summaries,
        canon_facts=canon_facts,
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


def summarize_and_store(env, db: NovelDB, chapter: ChapterDraft) -> ChapterSummary:
    prompt = render_prompt(
        env,
        "sum_and_extract.j2",
        chapter_number=chapter.number,
        chapter_text=chapter.text,
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

    db.upsert_summary(summary.chapter, summary.summary_short, summary.summary_detailed)
    db.add_facts([(f, summary.chapter) for f in summary.new_facts])
    return summary
