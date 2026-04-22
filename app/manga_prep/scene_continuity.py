"""Scene continuity engine for cross-panel visual coherence.

Responsibilities:
  1. Assign continuity_group to panels sharing location + character set
  2. Link reference_panel_id chains within each group
  3. Manage deterministic seeds (base per group + small offset per panel)
  4. Populate scene_state_in from the running state and scene_state_out after generation
  5. Persist scene_state.json per chapter
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import PanelSpec, SceneState

logger = logging.getLogger(__name__)


def _group_key(panel: PanelSpec) -> str:
    """Build a stable group key from location + sorted character ids."""
    loc = panel.location_id or "unknown"
    chars = sorted(c.character_id.lower() for c in panel.characters)
    return f"{loc}:{'|'.join(chars)}" if chars else loc


def _base_seed_for_group(group_key: str) -> int:
    """Deterministic base seed derived from the group key."""
    digest = hashlib.sha256(group_key.encode()).hexdigest()
    return int(digest[:8], 16)


def assign_continuity_groups(panels: List[PanelSpec]) -> List[PanelSpec]:
    """Assign continuity_group, reference_panel_id, and seed_hint to panels.

    Panels that share the same location + character set belong to the same
    continuity group.  Within a group, each panel's reference_panel_id points
    to the most recent earlier panel in that group so the generator can use
    the prior image as an img2img reference.

    seed_hint is base_seed + panel_index_within_group so nearby shots have
    similar but not identical noise.
    """
    group_latest: Dict[str, int] = {}
    group_count: Dict[str, int] = {}

    for panel in panels:
        gk = _group_key(panel)
        panel.continuity_group = gk

        if gk in group_latest:
            panel.reference_panel_id = group_latest[gk]
        else:
            panel.reference_panel_id = None

        group_latest[gk] = panel.panel_id

        idx_in_group = group_count.get(gk, 0)
        group_count[gk] = idx_in_group + 1

        base = _base_seed_for_group(gk)
        panel.seed_hint = (base + idx_in_group * 7) % (2**32)

        notes = []
        if panel.reference_panel_id is not None:
            notes.append(f"continues from panel {panel.reference_panel_id}")
            char_ids = [c.character_id for c in panel.characters]
            if char_ids:
                notes.append(f"same characters: {', '.join(char_ids)}")
            if panel.location_id:
                notes.append(f"same location: {panel.location_id}")
        panel.continuity_notes = notes

    groups_used = len(set(p.continuity_group for p in panels))
    logger.info(
        "Continuity: %d panels -> %d groups, %d cross-references",
        len(panels), groups_used,
        sum(1 for p in panels if p.reference_panel_id is not None),
    )
    return panels


def build_scene_state_in(
    panel: PanelSpec,
    running_state: SceneState,
) -> Dict[str, str]:
    """Build scene_state_in for a panel from the running chapter state."""
    state: Dict[str, str] = {}

    if panel.location_id:
        loc_state = running_state.location_states.get(panel.location_id, {})
        for k, v in loc_state.items():
            state[f"location.{k}"] = v

    for ch in panel.characters:
        ch_state = running_state.character_states.get(ch.character_id, {})
        for k, v in ch_state.items():
            state[f"{ch.character_id}.{k}"] = v

    if panel.continuity_group in running_state.group_last_image:
        state["prev_image"] = running_state.group_last_image[panel.continuity_group]

    return state


def update_scene_state_out(
    panel: PanelSpec,
    panel_image_path: str,
    running_state: SceneState,
    seed_used: int | None = None,
) -> SceneState:
    """Update the running state after a panel has been generated."""
    running_state.last_panel_id = panel.panel_id
    gk = panel.continuity_group

    running_state.group_last_panel[gk] = panel.panel_id
    running_state.group_last_image[gk] = panel_image_path

    if seed_used is not None:
        running_state.group_seeds[gk] = seed_used

    panel.scene_state_out = {
        "panel_image": panel_image_path,
        "seed": str(seed_used) if seed_used else "",
        "shot_type": panel.shot_type,
        "camera_angle": panel.camera_angle,
    }
    for ch in panel.characters:
        key = ch.character_id
        char_state = running_state.character_states.get(key, {})
        if ch.expression:
            char_state["expression"] = ch.expression
        if ch.action:
            char_state["action"] = ch.action
        if ch.pose:
            char_state["pose"] = ch.pose
        running_state.character_states[key] = char_state
        panel.scene_state_out[f"{key}.expression"] = ch.expression
        panel.scene_state_out[f"{key}.action"] = ch.action

    if panel.location_id:
        loc_state = running_state.location_states.get(panel.location_id, {})
        if panel.time_of_day:
            loc_state["time_of_day"] = panel.time_of_day
        loc_state["last_angle"] = panel.camera_angle
        running_state.location_states[panel.location_id] = loc_state

    return running_state


def save_scene_state(state: SceneState, ch_dir: Path) -> Path:
    """Persist scene_state.json to the chapter output directory."""
    out = ch_dir / "scene_state.json"
    out.write_text(
        json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Scene state saved -> %s", out.name)
    return out


def load_scene_state(ch_dir: Path) -> Optional[SceneState]:
    """Load scene_state.json if it exists (for resuming)."""
    p = ch_dir / "scene_state.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return SceneState.model_validate(data)
    return None


def collect_reference_images(
    panel: PanelSpec,
    panel_paths: Dict[int, Path],
    char_refs: Dict[str, Path],
    loc_refs: Dict[str, Path],
) -> List[Path]:
    """Build an ordered list of reference images for generation.

    Priority order:
      1. Previous panel in same continuity group (strongest style anchor)
      2. Character reference images
      3. Location reference image
    """
    refs: List[Path] = []

    if panel.reference_panel_id is not None:
        prev_path = panel_paths.get(panel.reference_panel_id)
        if prev_path and prev_path.exists():
            refs.append(prev_path)

    for ch in panel.characters:
        char_ref = char_refs.get(ch.character_id)
        if char_ref and char_ref.exists() and char_ref not in refs:
            refs.append(char_ref)

    if panel.location_id:
        loc_ref = loc_refs.get(panel.location_id)
        if loc_ref and loc_ref.exists() and loc_ref not in refs:
            refs.append(loc_ref)

    return refs
