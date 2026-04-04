import json
from pathlib import Path
from app.prompts import get_env
from app.pipeline import ensure_dirs, generate_bible, generate_outline, write_chapter, summarize_and_store
from app.db import NovelDB


BASE = Path(__file__).parent
DATA = BASE / "data"
PROMPTS = BASE / "prompts"


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_existing_chapters() -> set[int]:
    """Get set of chapter numbers that already exist."""
    chapters_dir = DATA / "chapters"
    if not chapters_dir.exists():
        return set()
    
    existing = set()
    for file in chapters_dir.glob("*.md"):
        try:
            # Extract chapter number from filename like "001_Title.md"
            num = int(file.name.split('_')[0])
            existing.add(num)
        except (ValueError, IndexError):
            pass
    
    return existing


def load_or_generate_bible(env, premise: str, genre: str):
    """Load existing bible or generate new one."""
    bible_file = DATA / "bible.json"
    
    if bible_file.exists():
        print("📖 Loading existing world bible...")
        data = json.loads(bible_file.read_text(encoding="utf-8"))
        from app.models import WorldBible
        return WorldBible.model_validate(data)
    
    print("🎭 Generating world bible...")
    bible = generate_bible(env, premise, genre)
    save_json(bible_file, bible.model_dump())
    print(f"✅ Created: {bible.title}")
    return bible


def load_or_generate_outline(env, bible):
    """Load existing outline or generate new one."""
    outline_file = DATA / "outline.json"
    
    if outline_file.exists():
        print("📋 Loading existing outline...")
        data = json.loads(outline_file.read_text(encoding="utf-8"))
        from app.models import Outline
        return Outline.model_validate(data)
    
    print("📋 Generating outline...")
    outline = generate_outline(env, bible)
    save_json(outline_file, outline.model_dump())
    print(f"✅ Outline: {len(outline.chapters)} chapters")
    return outline


def main():
    ensure_dirs(BASE)
    env = get_env(str(PROMPTS))
    db = NovelDB(str(DATA / "novel.db"))

    premise = "A tired IT engineer is reborn in a world where spells compile like code, and bugs become monsters."
    genre = "Isekai fantasy / system"

    # Load or generate bible and outline
    bible = load_or_generate_bible(env, premise, genre)
    outline = load_or_generate_outline(env, bible)

    # Check which chapters already exist
    existing_chapters = get_existing_chapters()
    total_chapters = len(outline.chapters)
    
    if existing_chapters:
        print(f"\n✅ Found existing chapters: {sorted(existing_chapters)}")
    
    # Determine which chapters to generate
    chapters_to_generate = [ch for ch in range(1, total_chapters + 1) if ch not in existing_chapters]
    
    if not chapters_to_generate:
        print(f"\n🎉 All {total_chapters} chapters already generated!")
        print(f"📖 Chapters: {DATA / 'chapters'}/")
        return
    
    print(f"\n✍️  Will generate chapters: {chapters_to_generate}")
    
    # Generate missing chapters
    for ch in chapters_to_generate:
        print(f"\n✍️  Writing chapter {ch}...")
        draft = write_chapter(env, bible, outline, db, ch, 
                            check_similarity=True, 
                            chapters_dir=DATA / "chapters")
        chapter_file = DATA / "chapters" / f"{ch:03d}_{draft.title.replace(' ', '_')}.md"
        chapter_file.write_text(draft.text, encoding="utf-8")
        print(f"✅ Saved: {chapter_file.name}")
        
        print(f"📝 Summarizing chapter {ch}...")
        summarize_and_store(env, db, draft)

    print("\n✨ Done! Check data/ for outputs.")
    print(f"📚 Bible: {DATA / 'bible.json'}")
    print(f"📋 Outline: {DATA / 'outline.json'}")
    print(f"📖 Chapters: {DATA / 'chapters'}/")
    print(f"📊 Generated: {len(chapters_to_generate)} chapters")
    print(f"📊 Total: {len(existing_chapters) + len(chapters_to_generate)}/{total_chapters} chapters")


if __name__ == "__main__":
    main()
