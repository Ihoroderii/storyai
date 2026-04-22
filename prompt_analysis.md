# Analysis: Why the prompts do not relate to the chapter

Yes — the prompts are clearly **not derived from the chapter you showed**.

Your chapter is about:

- **Kaito Yamamoto**
- **Mira the Compiler**
- **Codebound**
- fantasy/digital world
- coding magic
- a bug-monster

But the generated scenario contains:

- **Hana**
- **Raven**
- **space station maintenance bay**
- **neon / rain / alley**
- cyberpunk vibe

So this is not just “slightly inaccurate.” It looks like your pipeline is pulling the **wrong source context** before image generation.

## What the log tells us

The failure starts here:

```text
=== Scenario (4 panels) ===
Panel 1: context=space station maintenance bay
...
speaker=Hana
...
speaker=Raven
```

That means the problem is **already present in Step 1**:

```text
Step 1/4: Converting chapter text to panel scenario ...
```

So Step 2 is not the main issue.  
Image generation is simply following the bad scenario it received.

## Most likely causes

### 1. Wrong text is being sent to the LLM
This is the most likely reason.

Your log says:

```text
Processing chapter 1: 001_Rebooting_Reality.md
Chapter 1: 1182 words
```

But the scenario returned has zero connection to that chapter.

That usually means one of these happened:

- you loaded the correct file name, but passed a **different variable** to the scenario function
- you accidentally passed a **previous chapter text**
- you passed a **default sample prompt** instead of the real chapter
- your code read the chapter correctly, but the API payload used another field

Typical bug pattern:

```python
chapter_text = load_chapter(path)
...
scenario = generate_scenario(previous_text)   # wrong variable
```

or

```python
scenario_input = example_text if debug else chapter_text
```

and `debug` stayed on.

### 2. Conversation history contamination
If you use chat history and keep previous messages in the same conversation, the model may reuse old story context.

The names **Hana** and **Raven** look like persistent character memory from an earlier generation session, not random hallucination.

This can happen if your request to the chat model looks like this:

```python
messages = [
    *history,
    {"role": "user", "content": f"Convert this chapter into 4 panels:\n{chapter_text}"}
]
```

If `history` contains earlier manga-generation runs with Hana/Raven, the model may continue that storyline.

This is especially likely when:

- you reuse one global `messages` array
- you do not reset chat state per chapter
- you send previous scenario outputs back into the same conversation

For scenario extraction, you usually want a **fresh stateless prompt** for each chapter.

### 3. Your system prompt or few-shot example is dominating the task
The returned scenario looks like a generic example template:

- cyberpunk setting
- short dramatic dialogue
- two-character tension setup

That often happens when the system prompt includes an example like:

> Example:  
> Panel 1: Hana enters a neon alley...  
> Panel 2: Raven watches...

and the model copies the example instead of using the user chapter.

If your prompt has a few-shot example, the model may be overfitting to it.

### 4. The chapter text may not actually be included in the request
Another common bug: the function says “convert chapter text,” but only sends metadata like title or filename.

For example:

```python
prompt = f"Create a 4-panel scenario for chapter: {chapter_name}"
```

instead of:

```python
prompt = f"Create a 4-panel scenario for this chapter:\n\n{chapter_text}"
```

Or the chapter text is truncated to almost nothing before the actual important content.

### 5. Cached or reused scenario result
If you have caching, you may be reusing an old scenario.

Possible signs:

- same 4-panel shape every run
- same characters across unrelated chapters
- scenario file overwritten incorrectly
- cache key based only on `"chapter_001"` or `"4_panels"` instead of content hash

For example:

```python
cache_key = f"{chapter_number}_{panel_count}"
```

That is risky.

Better:

```python
cache_key = sha256(chapter_text.encode()).hexdigest()
```

### 6. Post-processing bug maps wrong fields into the scenario
Less likely, but possible.

Maybe the model returned something correct, and then your parser filled fields from stale data:

- `speaker` taken from previous scenario object
- `context` defaulting to old value
- `scene` tags hardcoded or reused

But because **all panels** are wrong in a consistent way, this feels more like the LLM input was wrong, not only parsing.

## Why Step 2 also looks weak

Even after Step 1, your image prompt is too generic:

```text
space station maintenance bay. dramatic, neon, rain, alley, tension, Korean manhwa style...
```

Problems:

- no character names in visual prompt
- no character appearance
- no action details
- no chapter-specific objects
- no spell/code/fantasy elements
- same prompt for all four panels except bubbles

So even with a correct scenario, images would still feel repetitive.

## Best diagnosis order

### First check the exact payload sent to HF router
Before the API call, print the full `messages` content.

You need to log:

- system prompt
- user prompt
- any prior assistant messages
- whether chapter text is fully included
- first 500 and last 500 chars of chapter text

For example:

```python
logger.info("SYSTEM PROMPT:\n%s", system_prompt)
logger.info("USER PROMPT:\n%s", user_prompt[:3000])
logger.info("CHAPTER TEXT PREVIEW START:\n%s", chapter_text[:800])
logger.info("CHAPTER TEXT PREVIEW END:\n%s", chapter_text[-800:])
logger.info("MESSAGE COUNT: %d", len(messages))
```

If you do this, you will probably spot the problem quickly.

### Second, force a stateless request
For scenario generation, send only:

- one system prompt
- one user prompt
- no previous history

If the output suddenly becomes correct, the problem was history contamination.

### Third, save raw LLM response before parsing
Log the raw response exactly as returned.

Example:

```python
raw = response.choices[0].message.content
logger.info("RAW SCENARIO RESPONSE:\n%s", raw)
```

This tells you whether:

- the model itself returned Hana/Raven
- or your parser transformed a correct answer into wrong fields

## What your prompt should probably do instead

Your scenario prompt should explicitly force grounding in the chapter.

Example structure:

```text
You are converting a chapter into manga panels.

Rules:
1. Use ONLY characters, locations, and events explicitly present in the chapter.
2. Do NOT invent new names unless the chapter contains unnamed extras.
3. Keep the setting faithful to the chapter.
4. Output exactly 4 panels in JSON.
5. Each panel must include:
   - context
   - characters
   - action
   - emotion
   - camera
   - speech
6. If information is missing, stay generic instead of inventing unrelated sci-fi/cyberpunk details.

Chapter text:
---
{chapter_text}
---
```

Very important line:

> **Use ONLY characters, locations, and events explicitly present in the chapter.**

That one sentence helps a lot.

## What the correct scenario should look more like for this chapter

Something closer to:

- **Panel 1:** Kaito wakes in the surreal Codebound landscape
- **Panel 2:** Mira introduces herself and explains coding magic
- **Panel 3:** Kaito writes and compiles the LightSpell
- **Panel 4:** the bug-monster appears from the forest

Contexts should be things like:

- digital fantasy forest
- floating code windows
- glowing data river
- corrupted forest edge

Not:

- space station maintenance bay
- neon alley
- rain

## Concrete fixes

### Fix 1: Reset chat messages per chapter
Do not reuse previous messages.

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": build_user_prompt(chapter_text)}
]
```

### Fix 2: Add anti-hallucination constraints
In the prompt:

```text
Do not reuse characters or settings from previous tasks.
Do not invent cyberpunk, sci-fi, or noir settings unless explicitly present.
```

### Fix 3: Log the exact request body
This is essential.

Without logging the outgoing prompt, you are debugging blind.

### Fix 4: Validate scenario against chapter before image generation
Add a sanity-check layer.

For example, extract allowed entities from the chapter:

```python
allowed_names = {"Kaito", "Mira", "Codebound"}
```

Then reject scenario if it includes:

- Hana
- Raven
- unrelated locations

Pseudo-check:

```python
def validate_scenario(scenario_text, chapter_text):
    forbidden_signals = ["Hana", "Raven", "space station", "neon alley"]
    for token in forbidden_signals:
        if token.lower() in scenario_text.lower():
            return False
    return True
```

Better: compare scenario entities against named entities from chapter.

If validation fails, regenerate with a stricter prompt.

### Fix 5: Generate panel prompts from structured panel data
Instead of one repeated environment prompt, make each panel prompt specific.

Example:

```text
Panel 1:
Kaito Yamamoto standing in a surreal digital forest, glowing code floating in the air, rivers of data nearby, confused expression, fantasy-tech world, Korean manhwa style

Panel 2:
Mira the Compiler with bright blue hair showing holographic code windows to Kaito, cheerful and energetic, magical coding tutorial scene, Korean manhwa style
```

That will dramatically improve relevance.

## Conclusion

The root problem is almost certainly in **Step 1**, not image generation.

Most likely causes, in order:

1. **wrong chapter text passed to the scenario generator**
2. **old chat history leaking into the request**
3. **few-shot/example prompt dominating the output**
4. **cached previous scenario being reused**
5. **parser/post-processing reusing stale values**

The fastest way to confirm it is:

- log the exact `messages` payload
- make scenario generation stateless
- log raw LLM output before parsing
