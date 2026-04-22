"""Extract structured PanelSpec data from chapter text using LLM or fallback."""
from __future__ import annotations
import json
import logging
import re
from typing import List, Optional, Tuple

from ..models import WorldBible
from .models import StoryBeat, PanelSpec, CharacterOnScreen, PanelDialogue

logger = logging.getLogger(__name__)

# After this many consecutive empty/failed LLM extractions per chapter, skip LLM for remaining beats.
_MAX_CONSECUTIVE_LLM_MISSES = 3


class PanelExtractionError(Exception):
    """Raised when panel extraction fails and no panels can be produced."""


# ---------------------------------------------------------------------------
# Text cleaning -- strip markdown artifacts before scene extraction
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`]+`")
_MD_HEADER = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_MD_BOLD_ITALIC = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_MD_UNDERLINE = re.compile(r"_{1,3}([^_]+)_{1,3}")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_HORIZONTAL_RULE = re.compile(r"^[\-\*_]{3,}\s*$", re.MULTILINE)


def _clean_chapter_text(text: str) -> str:
    """Remove markdown formatting that would contaminate scene extraction and prompts."""
    text = _CODE_FENCE.sub("", text)
    text = _MD_IMAGE.sub("", text)
    text = _MD_HEADER.sub("", text)
    text = _HORIZONTAL_RULE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(lambda m: m.group(0).strip("`"), text)
    text = _MD_BOLD_ITALIC.sub(r"\1", text)
    text = _MD_UNDERLINE.sub(r"\1", text)
    # Collapse excessive blank lines left after stripping
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_llm_json_array(raw: str) -> Optional[list]:
    """Parse a JSON array from an LLM response: handles fences, trailing commas, nested brackets."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fence = re.match(r"```(?:json)?\s*([\s\S]*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    else:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = re.sub(r",\s*([\]}])", r"\1", text)
    start = text.find("[")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    # Last resort: greedy match (often wrong if multiple arrays; raw_decode is preferred)
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            fixed = re.sub(r",\s*([\]}])", r"\1", json_match.group(0))
            obj = json.loads(fixed)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Content-driven cinematography
# ---------------------------------------------------------------------------

_ACTION_WORDS = re.compile(
    r'\b(fight|fought|slash|slashed|strike|struck|charge|charged|run|ran|'
    r'dodge|dodged|block|blocked|clash|clashed|explode|exploded|shatter|shattered|'
    r'crash|crashed|attack|attacked|leap|leaped|leapt|burst|sprint|sprinted|'
    r'battle|punch|punched|kick|kicked|throw|threw|swing|swung|'
    r'summon|summoned|cast|blaze|blazed|surge|surged)\b',
    re.IGNORECASE,
)
_EMOTION_WORDS = re.compile(
    r'\b(tear|tears|cry|cried|smile|smiled|laugh|laughed|sob|sobbed|'
    r'tremble|trembled|whisper|whispered|gasp|gasped|sigh|sighed|'
    r'blush|blushed|shudder|shuddered|grief|joy|rage|fear|'
    r'scream|screamed|yell|yelled|shout|shouted|weep|wept)\b',
    re.IGNORECASE,
)
_ARRIVAL_WORDS = re.compile(
    r'\b(enter|arrive|step into|walk into|emerge|approach|appear|came to|'
    r'reached|stood before|open the door|opened the door)\b',
    re.IGNORECASE,
)
_REVEAL_WORDS = re.compile(
    r'\b(reveal|behold|discover|realize|truth|secret|uncover|unmask|shock)\b',
    re.IGNORECASE,
)

_TIME_PATTERNS = [
    (re.compile(r'\b(dawn|sunrise|early morning|first light)\b', re.I), "dawn"),
    (re.compile(r'\b(morning|mid-morning|forenoon)\b', re.I), "morning"),
    (re.compile(r'\b(noon|midday|high sun)\b', re.I), "noon"),
    (re.compile(r'\b(afternoon|late afternoon)\b', re.I), "afternoon"),
    (re.compile(r'\b(dusk|sunset|twilight|evening)\b', re.I), "dusk"),
    (re.compile(r'\b(night|midnight|moonlight|starlight|dark)\b', re.I), "night"),
]

_VISUAL_OBJECT_WORDS = re.compile(
    r'\b(sword|staff|wand|shield|scroll|book|crystal|orb|gem|ring|amulet|'
    r'potion|arrow|bow|blade|axe|hammer|dagger|cloak|crown|mask|'
    r'flame|fire|lightning|ice|spell|rune|glyph|portal|barrier|'
    r'dragon|monster|beast|creature|demon)\b',
    re.IGNORECASE,
)

_BEAT_BREAK_WORDS = re.compile(
    r"\b(suddenly|but before|in that moment|moments later|a moment later|"
    r"then|after that|before .* could|without warning|at once|meanwhile|"
    r"just then|the moment|as soon as)\b",
    re.IGNORECASE,
)


def _infer_time_of_day(text: str) -> str:
    """Extract time of day from text if explicitly mentioned."""
    for pattern, label in _TIME_PATTERNS:
        if pattern.search(text):
            return label
    return ""


def _extract_must_show(text: str, characters: list) -> List[str]:
    """Extract visually important objects that should appear in the panel."""
    items = []
    for match in _VISUAL_OBJECT_WORDS.finditer(text):
        word = match.group(0).lower()
        if word not in items:
            items.append(word)
    # Also include character signature props via must_show
    for ch in characters:
        if ch.must_show:
            for item in ch.must_show:
                if item.lower() not in items:
                    items.append(item.lower())
    return items[:5]


def _infer_shot_and_angle(
    chunk: str,
    panel_idx: int,
    total_panels: int,
    n_characters: int,
    has_dialogue: bool,
) -> Tuple[str, str]:
    """Derive shot_type and camera_angle from the content of a text chunk."""
    text = chunk.lower()

    action_hits = len(_ACTION_WORDS.findall(text))
    emotion_hits = len(_EMOTION_WORDS.findall(text))
    arrival_hits = len(_ARRIVAL_WORDS.findall(text))
    reveal_hits = len(_REVEAL_WORDS.findall(text))

    # First panel defaults to a wide establishing shot
    if panel_idx == 0 and arrival_hits == 0 and action_hits < 2:
        return "wide", "eye level"

    # Last panel: pull back for dramatic closure
    if panel_idx == total_panels - 1 and action_hits < 2:
        if emotion_hits > 0:
            return "close-up", "eye level"
        return "wide", "high angle"

    # Heavy action -> wide + dynamic angles
    if action_hits >= 2:
        return "wide", "low angle"

    # Emotional beat -> close-up + eye level for intimacy
    if emotion_hits >= 2:
        return "close-up", "eye level"

    # Single-character reveal or realization
    if reveal_hits > 0 and n_characters <= 1:
        return "extreme close-up", "eye level"

    # Arrival / establishing a new location
    if arrival_hits > 0:
        return "wide", "eye level"

    # Dialogue-heavy with 2+ characters -> medium two-shot
    if has_dialogue and n_characters >= 2:
        return "medium", "eye level"

    # Dialogue-heavy, single speaker -> medium close
    if has_dialogue:
        return "medium", "eye level"

    # Crowd or many characters
    if n_characters >= 3:
        return "wide", "high angle"

    # Default: medium shot, alternate angle to avoid monotony
    angles = ["eye level", "low angle", "eye level"]
    return "medium", angles[panel_idx % len(angles)]


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _first_sentence(text: str, limit: int = 100) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return sentence[:limit].strip()


def _importance_score(text: str) -> float:
    text_low = text.lower()
    score = 1.0
    score += len(_ACTION_WORDS.findall(text_low)) * 1.6
    score += len(_EMOTION_WORDS.findall(text_low)) * 1.1
    score += len(_ARRIVAL_WORDS.findall(text_low)) * 1.0
    score += len(_REVEAL_WORDS.findall(text_low)) * 1.3
    if '"' in text or "\u201c" in text:
        score += 0.8
    if "!" in text:
        score += 0.4
    if _BEAT_BREAK_WORDS.search(text_low):
        score += 0.8
    return score


def _importance_label(score: float) -> str:
    if score >= 4.5:
        return "high"
    if score >= 2.3:
        return "medium"
    return "low"


def _character_alias_map(bible: WorldBible) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for c in bible.main_characters:
        alias_map[c.name.lower()] = c.name
        first = c.name.split()[0].lower()
        if len(first) >= 3 and first not in alias_map:
            alias_map[first] = c.name
    return alias_map


def _detect_characters(text: str, bible: WorldBible) -> List[str]:
    text_low = text.lower()
    aliases = _character_alias_map(bible)
    found: List[str] = []
    for alias, canonical in aliases.items():
        if alias in text_low and canonical not in found:
            found.append(canonical)
    return found


def _detect_location_id(text: str, bible: WorldBible) -> str:
    text_low = text.lower()
    if not bible.locations:
        return ""
    for loc in bible.locations:
        words = [w for w in loc.name.lower().split() if len(w) > 3]
        if any(w in text_low for w in words):
            return loc.location_id
        if loc.location_id and loc.location_id.replace("_", " ") in text_low:
            return loc.location_id
    return ""


def estimate_auto_panel_count(
    chapter_text: str,
    min_panels: int = 12,
    max_panels: int = 24,
) -> int:
    """Estimate a good manga panel count from chapter size and beat density."""
    cleaned = _clean_chapter_text(chapter_text)
    if not cleaned.strip():
        return min_panels

    paragraphs = _split_paragraphs(cleaned)
    words = len(cleaned.split())
    base_by_words = words / 85.0
    base_by_paragraphs = len(paragraphs) * 0.85
    event_hits = (
        len(_ACTION_WORDS.findall(cleaned))
        + len(_REVEAL_WORDS.findall(cleaned))
        + len(_ARRIVAL_WORDS.findall(cleaned))
    )
    event_bonus = min(3, round(event_hits / 8))
    estimated = round((base_by_words * 0.65) + (base_by_paragraphs * 0.35) + event_bonus)
    return max(min_panels, min(max_panels, estimated))


def _allocate_panels_to_beats(beats: List[StoryBeat], target_panels: int) -> List[StoryBeat]:
    """Assign 1-3 panels per beat until the requested total is reached."""
    if not beats:
        return beats

    target_panels = max(len(beats), target_panels)
    counts = [1 for _ in beats]
    remaining = target_panels - len(beats)

    importance_bonus = {"low": 0.0, "medium": 0.8, "high": 1.6}
    weights = [
        max(1.0, (beat.words / 120.0) + importance_bonus.get(beat.importance, 0.5))
        for beat in beats
    ]

    while remaining > 0:
        idx = max(
            range(len(beats)),
            key=lambda i: (weights[i] - (counts[i] - 1) * 0.95) if counts[i] < 3 else -10_000,
        )
        if counts[idx] >= 3:
            break
        counts[idx] += 1
        remaining -= 1

    for beat, count in zip(beats, counts):
        beat.recommended_panels = count
    return beats


def _build_beat_prompt(
    chapter_text: str,
    chapter_num: int,
    bible: WorldBible,
    target_panels: int,
) -> str:
    char_names = [c.name for c in bible.main_characters]
    loc_ids = [loc.location_id for loc in bible.locations] if bible.locations else []
    desired_beats = max(5, min(10, round(target_panels / 2.2)))
    return "\n".join(
        p for p in [
            f"Chapter {chapter_num} text:\n---\n{chapter_text}\n---\n",
            f"Known characters: {', '.join(char_names)}" if char_names else "",
            f"Known location IDs: {', '.join(loc_ids)}" if loc_ids else "",
            (
                f"Split the chapter into {desired_beats} logical story beats for manga adaptation. "
                f"Keep each beat contiguous and coherent. We are targeting about {target_panels} manga panels total."
            ),
            (
                "Return ONLY a JSON list. Each beat object must include: "
                "beat_id, summary, text, importance, characters, location_id. "
                "Use exact chapter names. text should be a compact contiguous excerpt or compressed excerpt "
                "from that beat, around 60-220 words. location_id must be a string or empty string, never a number."
            ),
        ] if p
    )


def _normalize_summary(value: object, text: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        joined = ", ".join(str(v).strip() for v in value if str(v).strip())
        if joined:
            return joined
    if value is not None and not isinstance(value, (dict, list)):
        raw = str(value).strip()
        if raw:
            return raw
    return _first_sentence(text)


def _normalize_importance(value: object, score: float) -> str:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        mapping = {
            "low": "low",
            "minor": "low",
            "small": "low",
            "medium": "medium",
            "med": "medium",
            "mid": "medium",
            "normal": "medium",
            "high": "high",
            "major": "high",
            "important": "high",
        }
        if cleaned in mapping:
            return mapping[cleaned]
    return _importance_label(score)


def _normalize_characters(value: object, text: str, bible: WorldBible) -> List[str]:
    aliases = _character_alias_map(bible)
    canonical_names = set(aliases.values())
    characters: List[str] = []

    def _add_name(raw_name: object):
        name = str(raw_name).strip()
        if not name:
            return
        canonical = aliases.get(name.lower(), name)
        if canonical in canonical_names and canonical not in characters:
            characters.append(canonical)

    if isinstance(value, list):
        for item in value:
            _add_name(item)
    elif isinstance(value, str):
        for part in re.split(r",|/|;|\band\b|\|", value):
            _add_name(part)
    elif value is not None and not isinstance(value, dict):
        _add_name(value)

    return characters or _detect_characters(text, bible)


def _normalize_location_id(value: object, text: str, bible: WorldBible) -> str:
    detected = _detect_location_id(text, bible)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return detected
        if cleaned.isdigit():
            return detected
        return cleaned
    return detected


def extract_story_beats_llm(
    chapter_text: str,
    chapter_num: int,
    bible: WorldBible,
    target_panels: int,
    llm_call,
) -> List[StoryBeat]:
    """Use LLM to split a chapter into logical adaptation beats."""
    cleaned = _clean_chapter_text(chapter_text)
    prompt = (
        "You are a manga adaptation planner. Split a prose chapter into logical visual beats. "
        "Do not invent scenes, names, or locations. Return only JSON.\n\n"
        + _build_beat_prompt(cleaned, chapter_num, bible, target_panels)
    )
    try:
        raw = llm_call(prompt)
    except Exception as e:
        logger.warning("Story beats LLM call failed: %s", e)
        return []

    data = _parse_llm_json_array(raw)
    if data is None:
        logger.warning("Failed to parse story beats JSON from LLM (first %d chars): %s", min(200, len(raw)), raw[:200])
        return []

    beats: List[StoryBeat] = []
    if not isinstance(data, list):
        return beats

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        raw_text = item.get("text")
        if isinstance(raw_text, str):
            text = raw_text.strip()
        elif raw_text is None:
            text = ""
        else:
            text = str(raw_text).strip()
        if not text:
            continue
        score = _importance_score(text)
        try:
            beat_id = int(item.get("beat_id") or idx)
        except (TypeError, ValueError):
            beat_id = idx

        try:
            beats.append(
                StoryBeat(
                    beat_id=beat_id,
                    summary=_normalize_summary(item.get("summary"), text),
                    text=text,
                    words=len(text.split()),
                    importance=_normalize_importance(item.get("importance"), score),
                    characters=_normalize_characters(item.get("characters"), text, bible),
                    location_id=_normalize_location_id(item.get("location_id"), text, bible),
                )
            )
        except Exception as e:
            logger.warning("Skipping invalid story beat %d: %s", idx, e)
            continue

    return _allocate_panels_to_beats(beats, target_panels)


def extract_story_beats_fallback(
    chapter_text: str,
    chapter_num: int,
    bible: WorldBible,
    target_panels: int,
) -> List[StoryBeat]:
    """Split a chapter into logical beats using paragraph grouping and word targets."""
    cleaned = _clean_chapter_text(chapter_text)
    paragraphs = _split_paragraphs(cleaned)
    if not paragraphs:
        return []

    desired_beats = min(max(5, min(10, round(target_panels / 2.2))), target_panels)
    target_words = max(80, len(cleaned.split()) / max(1, desired_beats))

    beats: List[StoryBeat] = []
    current: List[str] = []
    current_words = 0
    current_score = 0.0

    for idx, para in enumerate(paragraphs):
        para_words = len(para.split())
        para_score = _importance_score(para)
        current.append(para)
        current_words += para_words
        current_score += para_score

        next_para = paragraphs[idx + 1] if idx + 1 < len(paragraphs) else ""
        avg_score = current_score / max(1, len(current))
        have_room = len(beats) + 1 < desired_beats
        is_last = idx == len(paragraphs) - 1

        boundary = False
        if have_room:
            if current_words >= target_words * 1.35:
                boundary = True
            elif current_words >= target_words * 0.85 and (
                para_score >= 4.0 or _BEAT_BREAK_WORDS.search(para) or _BEAT_BREAK_WORDS.search(next_para)
            ):
                boundary = True
            elif current_words >= target_words and avg_score >= 2.8:
                boundary = True

        if boundary or is_last:
            beat_text = "\n\n".join(current).strip()
            beat_score = current_score / max(1, len(current))
            beats.append(
                StoryBeat(
                    beat_id=len(beats) + 1,
                    summary=_first_sentence(beat_text),
                    text=beat_text,
                    words=len(beat_text.split()),
                    importance=_importance_label(beat_score),
                    characters=_detect_characters(beat_text, bible),
                    location_id=_detect_location_id(beat_text, bible),
                )
            )
            current = []
            current_words = 0
            current_score = 0.0

    return _allocate_panels_to_beats(beats, target_panels)


def extract_story_beats(
    chapter_text: str,
    chapter_num: int,
    bible: WorldBible,
    target_panels: int,
    llm_call=None,
) -> List[StoryBeat]:
    """Split the chapter into logical beats before generating panel specs."""
    if llm_call is not None:
        beats = extract_story_beats_llm(chapter_text, chapter_num, bible, target_panels, llm_call)
        if beats:
            logger.info(
                "LLM extracted %d story beats for chapter %d (target panels=%d)",
                len(beats), chapter_num, target_panels,
            )
            return beats
        logger.warning("LLM beat extraction failed, using fallback for chapter %d", chapter_num)

    beats = extract_story_beats_fallback(chapter_text, chapter_num, bible, target_panels)
    logger.info(
        "Fallback extracted %d story beats for chapter %d (target panels=%d)",
        len(beats), chapter_num, target_panels,
    )
    return beats

SYSTEM_PROMPT = """\
You are a manga scene extractor. Given a chapter of prose, extract the most \
visually dramatic moments and convert them into structured panel specifications.

CRITICAL RULES:
1. Use ONLY characters, locations, and events from the chapter.
2. Do NOT invent characters, settings, or dialogue.
3. character_id must match a name from the bible characters list.
4. location_id should match a bible location_id if the scene matches, \
   or use a descriptive snake_case id for new locations.
5. Dialogues must be real quotes from the chapter (may be shortened).

Return ONLY a JSON list of panels. Each panel:
{
  "panel_id": 1,
  "location_id": "forest_of_algorithms",
  "location_description": "Dense code-forest with glowing branches",
  "time_of_day": "twilight",
  "shot_type": "wide/medium/close-up/extreme close-up",
  "camera_angle": "eye level/low angle/high angle/bird's eye/dutch angle",
  "characters": [
    {
      "character_id": "Kaito Yamamoto",
      "pose": "standing, hand raised",
      "expression": "determined",
      "action": "casting a light spell",
      "must_show": ["glowing code on palm"]
    }
  ],
  "dialogues": [
    {
      "character_id": "Kaito Yamamoto",
      "text": "I did it!",
      "emotion": "excited",
      "bubble_type": "speech"
    }
  ],
  "scene_summary": "Kaito successfully casts his first light spell in the code-forest"
}"""


def _build_user_prompt(
    chapter_text: str,
    chapter_num: int,
    n_panels: int,
    bible: WorldBible,
) -> str:
    char_names = [c.name for c in bible.main_characters]
    loc_ids = [loc.location_id for loc in bible.locations] if bible.locations else []

    parts = [
        f"Chapter {chapter_num} text:\n---\n{chapter_text}\n---\n",
        f"Known characters: {', '.join(char_names)}" if char_names else "",
        f"Known location IDs: {', '.join(loc_ids)}" if loc_ids else "",
        f"\nExtract exactly {n_panels} panels from the most dramatic moments.",
        "Pick moments with strong visual impact: arrivals, confrontations, spells, reveals.",
        "Vary shot_type and camera_angle across panels for visual interest.",
    ]
    return "\n".join(p for p in parts if p)


def extract_panels_llm(
    chapter_text: str,
    chapter_num: int,
    n_panels: int,
    bible: WorldBible,
    llm_call,
) -> List[PanelSpec]:
    """Use an LLM to extract PanelSpec list from chapter text."""
    chapter_text = _clean_chapter_text(chapter_text)
    user_prompt = _build_user_prompt(chapter_text, chapter_num, n_panels, bible)
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    try:
        raw = llm_call(full_prompt)
    except Exception as e:
        logger.warning("Panel extraction LLM call failed: %s", e)
        return []

    logger.info("Raw panel extraction response (%d chars)", len(raw or ""))

    data = _parse_llm_json_array(raw or "")
    if data is None:
        logger.warning(
            "Failed to parse panel JSON from LLM (showing start of response): %s",
            (raw or "")[:400],
        )
        return []

    if not isinstance(data, list):
        return []

    panels = []
    for item in data:
        try:
            panels.append(PanelSpec.model_validate(item))
        except Exception as e:
            logger.warning("Skipping invalid panel: %s", e)
    return panels


def extract_panels_fallback(
    chapter_text: str,
    chapter_num: int,
    n_panels: int,
    bible: WorldBible,
) -> List[PanelSpec]:
    """Heuristic fallback: split chapter into scenes by paragraph clusters."""
    chapter_text = _clean_chapter_text(chapter_text)
    paragraphs = [p.strip() for p in chapter_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [chapter_text]

    chunk_size = max(1, len(paragraphs) // n_panels)
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunks.append("\n".join(paragraphs[i:i + chunk_size]))
    chunks = chunks[:n_panels]
    while len(chunks) < n_panels:
        chunks.append(chunks[-1] if chunks else "")

    char_names: dict[str, str] = {}
    for c in bible.main_characters:
        char_names[c.name.lower()] = c.name
        first = c.name.split()[0].lower()
        if first not in char_names and len(first) >= 3:
            char_names[first] = c.name
    loc_map = {loc.location_id: loc for loc in bible.locations} if bible.locations else {}

    panels = []
    for idx, chunk in enumerate(chunks):
        chunk_low = chunk.lower()

        # Find characters (deduplicate by canonical name)
        seen_chars: set[str] = set()
        characters = []
        for name_low, name in char_names.items():
            if name_low in chunk_low and name not in seen_chars:
                seen_chars.add(name)
                characters.append(CharacterOnScreen(
                    character_id=name,
                    expression="neutral",
                    action="",
                ))

        # Find location
        location_id = ""
        location_desc = ""
        for loc_id, loc in loc_map.items():
            loc_words = loc.name.lower().split()
            if any(w in chunk_low for w in loc_words if len(w) > 3):
                location_id = loc_id
                location_desc = loc.description
                break

        # Extract dialogues from quotes in this chunk
        dialogues = []
        all_quotes = re.findall(r'\u201c([^\u201d]{3,})\u201d|"([^"]{3,})"', chunk)
        for q_match in all_quotes:
            quote = q_match[0] or q_match[1]
            if not quote.strip():
                continue
            speaker = characters[0].character_id if characters else "Narrator"
            for ch in characters:
                first_name = ch.character_id.split()[0]
                if first_name.lower() in chunk_low:
                    quote_pos = chunk.find(quote[:20])
                    if quote_pos >= 0:
                        nearby = chunk[max(0, quote_pos - 60):quote_pos + len(quote) + 60].lower()
                        if first_name.lower() in nearby:
                            speaker = ch.character_id
                            ch.expression = "speaking"
                            break

            emotion = "normal"
            if quote.endswith("!") or quote.isupper():
                emotion = "excited"
            elif "..." in quote:
                emotion = "hesitant"

            dialogues.append(PanelDialogue(
                character_id=speaker,
                text=quote[:120],
                emotion=emotion,
                bubble_type="speech",
            ))

        # Infer character expressions/actions from content
        for ch in characters:
            if ch.expression == "neutral":
                if _EMOTION_WORDS.search(chunk_low):
                    ch.expression = "emotional"
                elif _ACTION_WORDS.search(chunk_low):
                    ch.expression = "intense"
                    ch.action = "in motion"
            if not ch.action and _ACTION_WORDS.search(chunk_low):
                match = _ACTION_WORDS.search(chunk_low)
                if match:
                    ch.action = match.group(0) + "ing" if not match.group(0).endswith("e") else match.group(0)[:-1] + "ing"

        first_sentence = chunk.split(".")[0].strip()[:80] if chunk else ""

        # Content-driven shot and angle selection
        shot_type, camera_angle = _infer_shot_and_angle(
            chunk,
            panel_idx=idx,
            total_panels=n_panels,
            n_characters=len(characters),
            has_dialogue=bool(dialogues),
        )

        # Extract time of day and visually important objects
        time_of_day = _infer_time_of_day(chunk)
        must_show_items = _extract_must_show(chunk, characters)
        for ch in characters:
            if must_show_items and not ch.must_show:
                ch.must_show = must_show_items[:2]

        panels.append(PanelSpec(
            panel_id=idx + 1,
            location_id=location_id,
            location_description=location_desc or first_sentence,
            time_of_day=time_of_day,
            shot_type=shot_type,
            camera_angle=camera_angle,
            characters=characters,
            dialogues=dialogues,
            scene_summary=first_sentence,
        ))

    return panels


def extract_panels_from_beats(
    beats: List[StoryBeat],
    chapter_num: int,
    bible: WorldBible,
    llm_call=None,
) -> List[PanelSpec]:
    """Convert logical beats into 1-3 panels each, then renumber globally."""
    all_panels: List[PanelSpec] = []
    next_panel_id = 1
    consecutive_llm_misses = 0

    for beat in beats:
        beat_text = _clean_chapter_text(beat.text)
        if not beat_text.strip():
            continue

        beat_panels: List[PanelSpec] = []
        use_llm = (
            llm_call is not None
            and consecutive_llm_misses < _MAX_CONSECUTIVE_LLM_MISSES
        )
        if use_llm:
            beat_panels = extract_panels_llm(
                beat_text,
                chapter_num,
                beat.recommended_panels,
                bible,
                llm_call,
            )
            if beat_panels:
                consecutive_llm_misses = 0
            else:
                consecutive_llm_misses += 1
                if consecutive_llm_misses >= _MAX_CONSECUTIVE_LLM_MISSES:
                    logger.warning(
                        "LLM panel extraction failed %d times in a row; "
                        "using deterministic fallback for remaining beats",
                        _MAX_CONSECUTIVE_LLM_MISSES,
                    )

        if not beat_panels:
            beat_panels = extract_panels_fallback(
                beat_text,
                chapter_num,
                beat.recommended_panels,
                bible,
            )

        for panel in beat_panels:
            panel.panel_id = next_panel_id
            if not panel.scene_summary:
                panel.scene_summary = beat.summary
            if not panel.location_id and beat.location_id:
                panel.location_id = beat.location_id
            if not panel.location_description and beat.summary:
                panel.location_description = beat.summary
            if not panel.characters and beat.characters:
                panel.characters = [
                    CharacterOnScreen(character_id=name, expression="neutral")
                    for name in beat.characters
                ]
            all_panels.append(panel)
            next_panel_id += 1

    return all_panels


def extract_panels(
    chapter_text: str,
    chapter_num: int,
    n_panels: int | str,
    bible: WorldBible,
    llm_call=None,
) -> List[PanelSpec]:
    """Main entry point: extract PanelSpec list from chapter text.

    Uses LLM when available, falls back to deterministic heuristic extraction.
    Raises PanelExtractionError if neither path produces any panels.
    """
    if isinstance(n_panels, str) and n_panels.lower() == "auto":
        target_panels = estimate_auto_panel_count(chapter_text)
        beats = extract_story_beats(
            chapter_text,
            chapter_num,
            bible,
            target_panels,
            llm_call=llm_call,
        )
        if not beats:
            raise PanelExtractionError(
                f"Chapter {chapter_num}: auto beat extraction produced 0 beats."
            )

        panels = extract_panels_from_beats(
            beats,
            chapter_num,
            bible,
            llm_call=llm_call,
        )
        if not panels:
            raise PanelExtractionError(
                f"Chapter {chapter_num}: beat-based extraction produced 0 panels."
            )
        logger.info(
            "Auto panel mode selected %d panels across %d beats for chapter %d",
            len(panels), len(beats), chapter_num,
        )
        return panels

    n_panels = int(n_panels)

    if llm_call is not None:
        panels = extract_panels_llm(chapter_text, chapter_num, n_panels, bible, llm_call)
        if panels:
            logger.info("LLM extracted %d panels for chapter %d", len(panels), chapter_num)
            return panels
        logger.warning("LLM extraction failed, using fallback for chapter %d", chapter_num)

    panels = extract_panels_fallback(chapter_text, chapter_num, n_panels, bible)
    if not panels:
        raise PanelExtractionError(
            f"Chapter {chapter_num}: both LLM and deterministic extraction produced "
            f"0 panels. Check that the chapter file contains actual prose."
        )
    logger.info("Fallback extracted %d panels for chapter %d", len(panels), chapter_num)
    return panels
