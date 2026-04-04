# Self-Similarity Detection System

## 🎯 What It Does

Prevents your AI from accidentally repeating itself by detecting:
- **Paragraph duplicates** (≥90% similarity) - Copy-paste errors
- **Chapter repetition** (≥70% similarity) - Repetitive plot structure

## 🔍 How It Works

### Level 1: Paragraph-Level Checks (Strict)
```
Threshold: ≥90% similarity
Checks: Every paragraph vs all previous chapters
```

**Catches:**
- Exact duplicates
- Near-identical descriptions
- Repeated dialogue

**Example:**
```
⚠️ Chapter 5, para 12 is 91% similar to Chapter 3, para 8
   "Kaito raised his hand and began to compile the spell..."
```

### Level 2: Chapter-Level Checks (Medium)
```
Threshold: ≥70% similarity
Checks: Full chapter text vs all previous chapters (except immediate previous)
```

**Catches:**
- Repeated plot beats
- Similar scene structure
- Repetitive character interactions

**Example:**
```
⚠️ Chapter 5 is 78% similar to Chapter 3
   High overlap in: character dialogue, battle scene structure
```

## 🚀 How to Use

### Automatic Detection (Built-in)
When generating chapters, similarity checks run automatically:

```bash
python continue.py --all
```

Output:
```
✍️ Writing chapter 3...
🔍 Checking for self-similarity...
✅ No similarity issues detected
```

### Manual Check (Analyze Existing)
Check all existing chapters:

```bash
python check_similarity.py
```

Output:
```
📊 Analyzing 10 chapters for self-similarity...

✅ No significant similarity issues found!
All chapters appear to be unique and varied.
```

## ⚙️ Configuration

Edit thresholds in `app/similarity.py`:

```python
PARAGRAPH_SIMILARITY_THRESHOLD = 0.90  # 90% - Strict
CHAPTER_SIMILARITY_THRESHOLD = 0.70    # 70% - Medium
MIN_PARAGRAPH_LENGTH = 100             # Ignore short paragraphs
```

## 📊 What the Thresholds Mean

### Paragraph Similarity (0.90)
- **0.90-1.00** = Near-identical or copy-paste ⚠️
- **0.70-0.89** = Very similar (acceptable for recurring phrases)
- **< 0.70** = Different enough

### Chapter Similarity (0.70)
- **0.80-1.00** = Major structural repetition ⚠️
- **0.70-0.79** = Moderate similarity (warning) 🟡
- **< 0.70** = Acceptably different

## 🎨 Features

### Smart Filtering
- ✅ Ignores copyright headers
- ✅ Skips very short paragraphs (< 100 chars)
- ✅ Case-insensitive comparison
- ✅ Doesn't check adjacent chapters (some overlap is natural)

### Performance
- ✅ Fast: ~2 seconds per chapter
- ✅ Cached: Doesn't re-read files
- ✅ Minimal memory: Only loads needed chapters

### Non-Blocking
- ⚠️ Shows warnings but doesn't stop generation
- 💡 You decide whether to regenerate
- 📊 Reports at end for batch operations

## 🔧 Troubleshooting

### False Positives

**Problem:** Normal recurring phrases flagged
```
⚠️ "Kaito focused his mana" appears in multiple chapters
```

**Solution:** Lower the threshold or add to ignore list

### Missing Duplicates

**Problem:** Obvious repetition not caught
```
Two chapters have same plot but different wording
```

**Solution:** Lower CHAPTER_SIMILARITY_THRESHOLD to 0.60

### Too Slow

**Problem:** Checking takes too long

**Solution:** Increase MIN_PARAGRAPH_LENGTH to 200

## 📈 Interpreting Results

### Healthy Novel
```
✅ No similarity issues detected
```
- Good variety
- Unique chapters
- No repetition

### Minor Issues
```
🟡 Chapter 5 is 72% similar to Chapter 3
```
- Acceptable level
- Monitor for patterns
- Consider more variety in future

### Major Issues
```
🔴 Chapter 5, para 12 is 95% similar to Chapter 3, para 8
```
- Likely copy-paste
- Regenerate affected chapter
- Check prompts for variety

## 💡 Best Practices

### 1. Check After Generation
```bash
# Generate chapters
python continue.py 3 5

# Verify quality
python check_similarity.py
```

### 2. Regenerate Problem Chapters
```bash
# If chapter 5 is too similar, delete and regenerate
rm data/chapters/005_*.md
python continue.py 5
```

### 3. Improve Prompts
If seeing repetition, enhance `prompts/chapter.j2`:
- Add more variety instructions
- Reference specific differences from previous chapters
- Emphasize unique elements

### 4. Monitor Patterns
Track which chapters are similar:
- Same POV characters?
- Similar scene types (all battles)?
- Repetitive dialogue patterns?

Adjust outline to add variety.

## 🔬 Technical Details

### Algorithm: SequenceMatcher
Uses Python's `difflib.SequenceMatcher`:
- Fast (O(n²) worst case, but optimized)
- Accurate (Ratcliff-Obershelp algorithm)
- Built-in (no extra dependencies)

### Comparison Method
```python
similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
```

Returns 0.0-1.0:
- 1.0 = Identical
- 0.9 = Near-duplicate
- 0.7 = Similar structure
- 0.5 = Some overlap
- 0.0 = Completely different

### Memory Usage
- Only loads chapters being compared
- Caches in memory during batch operations
- ~1-2 MB per chapter cached

### Speed
- Paragraph check: ~0.1s per paragraph pair
- Chapter check: ~0.5s per chapter pair
- Total for 10 chapters: ~5-10 seconds

## 🚫 What It Doesn't Catch

### Semantic Similarity
```
"Kaito cast a spell" vs "He invoked magic"
```
Same meaning, different words = **Not caught**

Need: Semantic embeddings (TF-IDF, word2vec) - more complex

### Plot Similarity
```
Chapter 3: Hero fights dragon
Chapter 7: Hero fights different dragon
```
Same plot beat, different execution = **Not caught**

Need: Plot structure analysis - requires NLP

### Character Voice
```
All characters sound the same
```
Need: Style analysis - specialized tools

## 🎓 When to Worry

### Red Flags 🚨
- Multiple paragraph duplicates (≥90%)
- Chapter similarity ≥80%
- Same paragraphs in 3+ chapters
- Increasing similarity over time

### Yellow Flags 🟡
- Chapter similarity 70-79%
- Recurring scene structures
- Similar dialogue patterns
- 1-2 paragraph duplicates

### Green Flags ✅
- No warnings
- Chapter similarity <60%
- Varied scene types
- Unique descriptions

## 📚 Further Reading

- [Plagiarism Detection Algorithms](https://en.wikipedia.org/wiki/Plagiarism_detection)
- [Ratcliff-Obershelp Algorithm](https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching)
- [Text Similarity Metrics](https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50)

---

**Questions?** The similarity checker is now integrated into all generation scripts and runs automatically!
