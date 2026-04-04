#!/usr/bin/env python3
"""
Add copyright headers to all chapter files.
Run: python add_copyright.py
"""

from pathlib import Path
from datetime import datetime

COPYRIGHT_HEADER = """---
copyright: "© {year} All Rights Reserved"
license: "This work is protected by copyright. Unauthorized use is prohibited."
generated: "{date}"
---

*© {year} All Rights Reserved. This chapter may not be reproduced, distributed, 
or transmitted in any form without written permission.*

---

"""

def add_copyright_to_chapters():
    chapters_dir = Path("data/chapters")
    
    if not chapters_dir.exists():
        print("❌ No chapters directory found.")
        return
    
    year = datetime.now().year
    date = datetime.now().strftime("%Y-%m-%d")
    
    for chapter_file in sorted(chapters_dir.glob("*.md")):
        content = chapter_file.read_text(encoding="utf-8")
        
        # Skip if already has copyright
        if "© " in content and "All Rights Reserved" in content:
            print(f"⏭️  {chapter_file.name} - Already protected")
            continue
        
        # Add header
        header = COPYRIGHT_HEADER.format(year=year, date=date)
        new_content = header + content
        
        chapter_file.write_text(new_content, encoding="utf-8")
        print(f"✅ {chapter_file.name} - Copyright added")
    
    print("\n✨ Done! All chapters are now copyright protected.")

if __name__ == "__main__":
    add_copyright_to_chapters()
