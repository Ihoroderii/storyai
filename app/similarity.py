"""
Self-similarity detection to prevent repetitive content.
Uses both paragraph-level and chapter-level checks.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
from difflib import SequenceMatcher


# Thresholds
PARAGRAPH_SIMILARITY_THRESHOLD = 0.90  # Strict - catches copy-paste
CHAPTER_SIMILARITY_THRESHOLD = 0.70    # Medium - catches plot repetition
MIN_PARAGRAPH_LENGTH = 100             # Ignore very short paragraphs


class SimilarityChecker:
    def __init__(self, chapters_dir: Path):
        self.chapters_dir = chapters_dir
        self.chapter_cache = {}  # {chapter_num: {'text': str, 'paragraphs': list}}
    
    def _load_chapter(self, chapter_num: int) -> Optional[dict]:
        """Load a chapter from disk and parse it."""
        if chapter_num in self.chapter_cache:
            return self.chapter_cache[chapter_num]
        
        # Find chapter file
        for file in self.chapters_dir.glob(f"{chapter_num:03d}_*.md"):
            text = file.read_text(encoding="utf-8")
            
            # Split into paragraphs (ignore copyright headers)
            lines = text.split('\n')
            paragraphs = []
            current_para = []
            
            in_copyright = False
            for line in lines:
                # Skip copyright sections
                if line.startswith('---'):
                    in_copyright = not in_copyright
                    continue
                if in_copyright or line.startswith('*©') or line.startswith('©'):
                    continue
                
                # Build paragraphs
                line = line.strip()
                if line:
                    current_para.append(line)
                elif current_para:
                    para_text = ' '.join(current_para)
                    if len(para_text) >= MIN_PARAGRAPH_LENGTH:
                        paragraphs.append(para_text)
                    current_para = []
            
            # Add last paragraph
            if current_para:
                para_text = ' '.join(current_para)
                if len(para_text) >= MIN_PARAGRAPH_LENGTH:
                    paragraphs.append(para_text)
            
            self.chapter_cache[chapter_num] = {
                'text': text,
                'paragraphs': paragraphs
            }
            return self.chapter_cache[chapter_num]
        
        return None
    
    def _similarity_ratio(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts (0.0 to 1.0)."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def check_paragraph_similarity(self, new_chapter_text: str, current_chapter_num: int) -> List[dict]:
        """
        Check if any paragraphs in new chapter match previous chapters.
        Returns list of warnings.
        """
        warnings = []
        
        # Parse new chapter into paragraphs
        lines = new_chapter_text.split('\n')
        new_paragraphs = []
        current_para = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                current_para.append(line)
            elif current_para:
                para_text = ' '.join(current_para)
                if len(para_text) >= MIN_PARAGRAPH_LENGTH:
                    new_paragraphs.append(para_text)
                current_para = []
        
        if current_para:
            para_text = ' '.join(current_para)
            if len(para_text) >= MIN_PARAGRAPH_LENGTH:
                new_paragraphs.append(para_text)
        
        # Compare against previous chapters
        for ch_num in range(1, current_chapter_num):
            old_chapter = self._load_chapter(ch_num)
            if not old_chapter:
                continue
            
            for new_idx, new_para in enumerate(new_paragraphs):
                for old_idx, old_para in enumerate(old_chapter['paragraphs']):
                    similarity = self._similarity_ratio(new_para, old_para)
                    
                    if similarity >= PARAGRAPH_SIMILARITY_THRESHOLD:
                        warnings.append({
                            'type': 'paragraph',
                            'severity': 'high',
                            'similarity': similarity,
                            'current_chapter': current_chapter_num,
                            'current_paragraph': new_idx + 1,
                            'previous_chapter': ch_num,
                            'previous_paragraph': old_idx + 1,
                            'preview': new_para[:100] + '...' if len(new_para) > 100 else new_para
                        })
        
        return warnings
    
    def check_chapter_similarity(self, new_chapter_text: str, current_chapter_num: int) -> List[dict]:
        """
        Check if the entire new chapter is too similar to previous chapters.
        Returns list of warnings.
        """
        warnings = []
        
        # Compare against previous chapters (skip immediate previous - some overlap is normal)
        for ch_num in range(1, current_chapter_num - 1):
            old_chapter = self._load_chapter(ch_num)
            if not old_chapter:
                continue
            
            similarity = self._similarity_ratio(new_chapter_text, old_chapter['text'])
            
            if similarity >= CHAPTER_SIMILARITY_THRESHOLD:
                warnings.append({
                    'type': 'chapter',
                    'severity': 'medium',
                    'similarity': similarity,
                    'current_chapter': current_chapter_num,
                    'previous_chapter': ch_num
                })
        
        return warnings
    
    def check_all(self, new_chapter_text: str, current_chapter_num: int) -> dict:
        """
        Run all similarity checks.
        Returns: {
            'has_issues': bool,
            'paragraph_warnings': list,
            'chapter_warnings': list
        }
        """
        paragraph_warnings = self.check_paragraph_similarity(new_chapter_text, current_chapter_num)
        chapter_warnings = self.check_chapter_similarity(new_chapter_text, current_chapter_num)
        
        return {
            'has_issues': len(paragraph_warnings) > 0 or len(chapter_warnings) > 0,
            'paragraph_warnings': paragraph_warnings,
            'chapter_warnings': chapter_warnings
        }


def print_similarity_report(result: dict):
    """Print a formatted similarity report."""
    if not result['has_issues']:
        print("✅ No similarity issues detected")
        return
    
    print("\n" + "="*60)
    print("⚠️  SIMILARITY WARNINGS DETECTED")
    print("="*60)
    
    # Paragraph warnings (high severity)
    if result['paragraph_warnings']:
        print("\n🔴 PARAGRAPH-LEVEL DUPLICATES (High Priority):")
        for w in result['paragraph_warnings']:
            print(f"\n  Chapter {w['current_chapter']}, para {w['current_paragraph']} is {w['similarity']:.1%} similar to")
            print(f"  Chapter {w['previous_chapter']}, para {w['previous_paragraph']}")
            print(f"  Preview: \"{w['preview']}\"")
    
    # Chapter warnings (medium severity)
    if result['chapter_warnings']:
        print("\n🟡 CHAPTER-LEVEL SIMILARITY (Medium Priority):")
        for w in result['chapter_warnings']:
            print(f"\n  Chapter {w['current_chapter']} is {w['similarity']:.1%} similar to Chapter {w['previous_chapter']}")
            print(f"  This may indicate repetitive plot or scene structure")
    
    print("\n" + "="*60)
    print("💡 Consider regenerating with more variety in the prompts")
    print("="*60 + "\n")
