#!/usr/bin/env python3
"""
Check similarity between existing chapters.
Usage: python check_similarity.py
"""

from pathlib import Path
from app.similarity import SimilarityChecker, print_similarity_report


BASE = Path(__file__).parent
DATA = BASE / "data"
CHAPTERS_DIR = DATA / "chapters"


def main():
    if not CHAPTERS_DIR.exists():
        print("❌ No chapters directory found")
        return
    
    # Get all chapter files
    chapter_files = sorted(CHAPTERS_DIR.glob("*.md"))
    
    if len(chapter_files) < 2:
        print("Need at least 2 chapters to compare")
        return
    
    print(f"📊 Analyzing {len(chapter_files)} chapters for self-similarity...\n")
    
    checker = SimilarityChecker(CHAPTERS_DIR)
    
    all_warnings = {
        'paragraph': [],
        'chapter': []
    }
    
    # Check each chapter against previous ones
    for i, file in enumerate(chapter_files, 1):
        if i == 1:
            continue  # Skip first chapter
        
        print(f"Checking chapter {i}...")
        text = file.read_text(encoding="utf-8")
        
        result = checker.check_all(text, i)
        
        if result['has_issues']:
            all_warnings['paragraph'].extend(result['paragraph_warnings'])
            all_warnings['chapter'].extend(result['chapter_warnings'])
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SIMILARITY ANALYSIS COMPLETE")
    print("="*60)
    
    if not all_warnings['paragraph'] and not all_warnings['chapter']:
        print("\n✅ No significant similarity issues found!")
        print("All chapters appear to be unique and varied.\n")
        return
    
    # Print detailed report
    if all_warnings['paragraph']:
        print(f"\n🔴 Found {len(all_warnings['paragraph'])} paragraph-level duplicates:")
        for w in all_warnings['paragraph']:
            print(f"\n  • Chapter {w['current_chapter']}, para {w['current_paragraph']} ≈ "
                  f"Chapter {w['previous_chapter']}, para {w['previous_paragraph']} "
                  f"({w['similarity']:.1%} similar)")
            print(f"    Preview: \"{w['preview']}\"")
    
    if all_warnings['chapter']:
        print(f"\n🟡 Found {len(all_warnings['chapter'])} chapter-level similarities:")
        for w in all_warnings['chapter']:
            print(f"\n  • Chapter {w['current_chapter']} ≈ Chapter {w['previous_chapter']} "
                  f"({w['similarity']:.1%} similar)")
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    print("="*60)
    
    if all_warnings['paragraph']:
        print("• Paragraph duplicates suggest copy-paste errors")
        print("  → Consider regenerating affected chapters")
    
    if all_warnings['chapter']:
        print("• Chapter similarities suggest repetitive plot structure")
        print("  → Add more variety to chapter prompts")
        print("  → Adjust outline to have more distinct goals/conflicts")
    
    print()


if __name__ == "__main__":
    main()
