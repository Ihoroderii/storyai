#!/usr/bin/env python3
"""
Continue generating chapters from where you left off.
Usage:
    python continue.py           # Generate next missing chapter
    python continue.py 5         # Generate up to chapter 5
    python continue.py 3 7       # Generate chapters 3-7
    python continue.py --all     # Generate all remaining chapters
"""

import sys
import json
from pathlib import Path
from app.prompts import get_env
from app.pipeline import write_chapter, summarize_and_store
from app.db import NovelDB
from app.models import WorldBible, Outline


BASE = Path(__file__).parent
DATA = BASE / "data"
PROMPTS = BASE / "prompts"


def get_existing_chapters() -> set[int]:
    """Get set of chapter numbers that already exist."""
    chapters_dir = DATA / "chapters"
    if not chapters_dir.exists():
        return set()
    
    existing = set()
    for file in chapters_dir.glob("*.md"):
        try:
            num = int(file.name.split('_')[0])
            existing.add(num)
        except (ValueError, IndexError):
            pass
    
    return existing


def main():
    # Check if bible and outline exist
    bible_file = DATA / "bible.json"
    outline_file = DATA / "outline.json"
    
    if not bible_file.exists() or not outline_file.exists():
        print("❌ Bible or outline not found!")
        print("Run 'python main.py' first to generate the initial setup.")
        return
    
    # Load bible and outline
    print("📖 Loading existing world bible and outline...")
    bible = WorldBible.model_validate(json.loads(bible_file.read_text(encoding="utf-8")))
    outline = Outline.model_validate(json.loads(outline_file.read_text(encoding="utf-8")))
    
    env = get_env(str(PROMPTS))
    db = NovelDB(str(DATA / "novel.db"))
    
    # Get existing chapters
    existing = get_existing_chapters()
    total_chapters = len(outline.chapters)
    
    print(f"✅ Existing chapters: {sorted(existing) if existing else 'None'}")
    print(f"📊 Total chapters in outline: {total_chapters}")
    
    # Parse arguments
    if len(sys.argv) == 1:
        # Generate next missing chapter
        if not existing:
            chapters_to_gen = [1]
        else:
            next_ch = max(existing) + 1
            if next_ch > total_chapters:
                print(f"\n🎉 All chapters already complete!")
                return
            chapters_to_gen = [next_ch]
    
    elif "--all" in sys.argv:
        # Generate all missing chapters
        chapters_to_gen = [ch for ch in range(1, total_chapters + 1) if ch not in existing]
    
    elif len(sys.argv) == 2:
        # Generate up to specified chapter
        target = int(sys.argv[1])
        chapters_to_gen = [ch for ch in range(1, target + 1) if ch not in existing]
    
    elif len(sys.argv) == 3:
        # Generate range
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        chapters_to_gen = [ch for ch in range(start, end + 1) if ch not in existing]
    
    else:
        print("Usage:")
        print("  python continue.py           # Next chapter")
        print("  python continue.py 5         # Generate up to chapter 5")
        print("  python continue.py 3 7       # Generate chapters 3-7")
        print("  python continue.py --all     # All remaining chapters")
        return
    
    if not chapters_to_gen:
        print("\n✅ Requested chapters already exist!")
        return
    
    print(f"\n✍️  Will generate: {chapters_to_gen}")
    print(f"💰 Estimated cost: ${len(chapters_to_gen) * 0.03:.2f}")
    
    # Generate chapters
    for ch in chapters_to_gen:
        print(f"\n{'='*50}")
        print(f"✍️  Writing chapter {ch}/{total_chapters}...")
        
        draft = write_chapter(env, bible, outline, db, ch,
                            check_similarity=True,
                            chapters_dir=DATA / "chapters")
        chapter_file = DATA / "chapters" / f"{ch:03d}_{draft.title.replace(' ', '_')}.md"
        chapter_file.write_text(draft.text, encoding="utf-8")
        print(f"✅ Saved: {chapter_file.name}")
        
        print(f"📝 Summarizing chapter {ch}...")
        summarize_and_store(env, db, draft)
        print(f"✅ Summary stored in database")
    
    # Final summary
    new_existing = get_existing_chapters()
    print(f"\n{'='*50}")
    print(f"✨ Done!")
    print(f"📊 Progress: {len(new_existing)}/{total_chapters} chapters complete")
    
    remaining = total_chapters - len(new_existing)
    if remaining > 0:
        print(f"📝 Remaining: {remaining} chapters")
        print(f"💡 Run 'python continue.py --all' to finish the novel")
    else:
        print(f"🎉 Novel complete!")


if __name__ == "__main__":
    main()
