#!/usr/bin/env python3
"""Regenerate bible.json with the current schema (visual profiles, locations, layout).

Useful after updating bible.j2 or models.py with new fields.
Backs up the old bible before overwriting.

Usage:
    python regenerate_bible.py               # regenerate from config.json
    python regenerate_bible.py --dry-run     # show prompt, don't call LLM
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BASE = Path(__file__).parent
DATA = BASE / "data"


def main():
    parser = argparse.ArgumentParser(description="Regenerate bible.json with visual canon")
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt without calling the LLM")
    parser.add_argument("--keep-old", action="store_true", help="Don't overwrite — save as bible_new.json")
    args = parser.parse_args()

    from app.prompts import get_env, render_prompt

    config = json.loads((BASE / "config.json").read_text())
    premise = config["premise"]
    genre = config["genre"]

    env = get_env(str(BASE / "prompts"))
    prompt = render_prompt(env, "bible.j2", premise=premise, genre=genre)

    if args.dry_run:
        print("=== BIBLE PROMPT ===")
        print(prompt)
        print(f"\n({len(prompt)} characters)")
        return

    from app.llm import call_llm

    print(f"Generating bible for: {premise[:80]}...")
    print(f"Genre: {genre}")
    raw = call_llm(prompt)

    # Parse JSON from response
    import re
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        print("ERROR: LLM did not return valid JSON")
        print(raw[:500])
        sys.exit(1)

    data = json.loads(json_match.group(0))

    # Validate against model
    from app.models import WorldBible
    bible = WorldBible.model_validate(data)

    # Check completeness
    issues = []
    for c in bible.main_characters:
        if not c.visual.hair_color:
            issues.append(f"  {c.name}: missing hair_color")
        if not c.visual.eye_color:
            issues.append(f"  {c.name}: missing eye_color")
        if not c.visual.face_shape:
            issues.append(f"  {c.name}: missing face_shape")
        if not c.visual.reference_prompt:
            issues.append(f"  {c.name}: missing reference_prompt")
        if not c.visual.silhouette:
            issues.append(f"  {c.name}: missing silhouette")
        if not c.visual.expression_style:
            issues.append(f"  {c.name}: missing expression_style")
    for loc in bible.locations:
        if not loc.anchor_objects:
            issues.append(f"  {loc.name}: missing anchor_objects")
        if not loc.allowed_camera_angles:
            issues.append(f"  {loc.name}: missing allowed_camera_angles")

    if issues:
        print(f"\nWarnings ({len(issues)}):")
        for i in issues:
            print(i)

    # Save
    out_path = DATA / ("bible_new.json" if args.keep_old else "bible.json")
    old_path = DATA / "bible.json"

    if old_path.exists() and not args.keep_old:
        backup = DATA / f"bible_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(old_path, backup)
        print(f"\nBacked up old bible -> {backup.name}")

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")
    print(f"  Title:      {bible.title}")
    print(f"  Characters: {[c.name for c in bible.main_characters]}")
    print(f"  Locations:  {[loc.name for loc in bible.locations]}")

    vis_complete = sum(1 for c in bible.main_characters if c.visual.reference_prompt)
    print(f"  Visual profiles: {vis_complete}/{len(bible.main_characters)} complete")


if __name__ == "__main__":
    main()
