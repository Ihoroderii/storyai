# StoryAI - Light Novel Generator

Automatically generate light novels with consistent world-building, character development, and plot progression using LLMs.

## Features

- 🌍 **World Bible Generation** - Creates consistent world rules, characters, and themes
- 📖 **Outline Planning** - Generates structured chapter plans with hooks and conflicts
- ✍️ **Chapter Writing** - Produces full chapters with proper pacing and style
- 🧠 **Canon Memory** - Tracks facts and summaries to maintain consistency
- 💾 **SQLite Storage** - Persistent storage for chapter summaries and world facts

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your LLM provider in `.env`:

**Option A: Anthropic Claude (Recommended)**
- Get API key: https://console.anthropic.com/
- Edit `.env` and uncomment: `ANTHROPIC_API_KEY=your-key-here`

**Option B: OpenAI**
- Get API key: https://platform.openai.com/api-keys
- Edit `.env` and uncomment: `OPENAI_API_KEY=your-key-here`

**Option C: Mock Mode (Testing)**
- Already set in `.env` as `LLM_PROVIDER=mock`
- Uses dummy responses to test the pipeline

## Usage

### Generate Content

**First Time - Generate Initial Setup:**
```bash
python main.py
```

This will:
1. Generate a world bible based on the premise
2. Create a 10-chapter outline
3. Detect existing chapters and generate missing ones
4. Store summaries and canon facts in the database

**Continue Generating Chapters:**
```bash
# Generate next chapter only
python continue.py

# Generate up to chapter 5
python continue.py 5

# Generate chapters 3-7
python continue.py 3 7

# Generate ALL remaining chapters (3-10)
python continue.py --all
```

**Smart Features:**
- ✅ Skips chapters that already exist
- ✅ Uses existing bible and outline
- ✅ Continues from database (maintains story consistency)
- ✅ Shows cost estimate before generating
- ✅ **Auto-detects self-similarity** (prevents repetitive content)

**Check Existing Chapters for Similarity:**
```bash
python check_similarity.py
```
- Analyzes all chapters for duplicates
- Reports paragraph-level and chapter-level similarities
- Helps identify repetitive content

### View in Browser

After generating content, view it in a beautiful web interface:
```bash
python serve.py
```

This will:
- Start a local web server
- Open your browser to `http://localhost:8000/viewer.html`
- Display your novel with tabs for Bible, Outline, and Chapters
- Provide easy chapter navigation

## Output Structure

```
data/
├── bible.json          # World building document
├── outline.json        # Chapter outline
├── novel.db           # SQLite database with summaries & facts
└── chapters/
    ├── 001_chapter_title.md
    └── 002_chapter_title.md
```

## Customization

Edit `main.py` to change:
- **Premise**: The story concept
- **Genre**: Story genre/style
- **Number of chapters**: Modify the loop in `main()`

Edit templates in `prompts/` to customize:
- `bible.j2` - World building prompt
- `outline.j2` - Outline generation
- `chapter.j2` - Chapter writing style
- `sum_and_extract.j2` - Summary extraction

## Cost Estimation

### Anthropic Claude Haiku (Recommended)
- Bible: ~$0.01
- Outline: ~$0.01  
- Chapter (2000 words): ~$0.02
- Summary: ~$0.01
- **Per chapter: ~$0.03**
- **Full 10-chapter novel: ~$0.30**

### OpenAI GPT-4o-mini
- Bible: ~$0.01
- Outline: ~$0.01
- Chapter (2000 words): ~$0.02-0.03
- Summary: ~$0.01
- **Per chapter: ~$0.03-0.04**
- **Full 10-chapter novel: ~$0.35**

### Mock Mode
- **Free** - For testing the pipeline

## Architecture

```
StoryAI/
├── app/
│   ├── models.py      # Pydantic data models
│   ├── db.py          # SQLite database interface
│   ├── llm.py         # LLM API calls
│   ├── prompts.py     # Jinja2 template rendering
│   └── pipeline.py    # Generation pipeline
├── prompts/           # Jinja2 prompt templates
├── main.py           # Entry point
└── requirements.txt   # Python dependencies
```

## Copyright & Licensing

### Generated Content
All generated novel content (text, characters, plot, world-building) is:
- **© 2026 [Your Name]. All Rights Reserved.**
- Protected by international copyright law
- See `LICENSE.txt` for full terms

### Code/Software
The generator code itself (Python scripts, HTML viewer) is:
- **MIT Licensed** - Free to use, modify, distribute
- See code comments for details

### Important
- The **stories you generate** are YOUR copyright
- The **tool to generate them** is MIT licensed
- You own your creative output!
