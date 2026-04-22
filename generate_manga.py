#!/usr/bin/env python3
"""
Generate manga from storyAI chapters.

Architecture:
  storyAI      -> narrative canon (bible, chapters, canon facts)
  manga_prep   -> visual adaptation layer (scene extraction, continuity, prompt building)
  manga-ai-bot -> rendering (image generation, assembly)
  bubble       -> text overlay and placement

Pipeline per chapter:
  1. Load bible + chapter text
  2. manga_prep: extract PanelSpec from chapter (LLM or fallback)
  3. manga_prep: check visual continuity against character canon
  4. manga_prep: build stable SD prompts from PanelSpec + visual profiles
  5. Generate panel images (HF API or local GPU)
  6. Apply speech bubbles
  7. Assemble into chapter strip

Usage:
    python generate_manga.py                       # API mode, all chapters
    python generate_manga.py --chapter 3           # Single chapter
    python generate_manga.py --panels 6            # 6 panels per chapter
    python generate_manga.py --panels auto         # auto-estimate by chapter beats
    python generate_manga.py --no-llm-panels     # heuristic extraction only (no LLM credits)
    python generate_manga.py --no-bubbles          # Skip bubble overlay
    python generate_manga.py --provider local      # Use local GPU
    python generate_manga.py --verbose             # Show all prompts

Prerequisites:
    pip install -e /path/to/manga-ai-bot
    pip install -e /path/to/bubble
    Set HF_TOKEN in .env
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
DATA = BASE / "data"
MANGA_DIR = DATA / "manga"


# ---------------------------------------------------------------------------
# Imports / setup
# ---------------------------------------------------------------------------

def _check_imports():
    """Verify that manga-ai-bot and bubble are importable."""
    missing = []
    try:
        import manga_ai  # noqa: F401
    except ImportError:
        missing.append("manga-ai  (pip install -e /path/to/manga-ai-bot)")
    try:
        import manhwa_bubbles  # noqa: F401
    except ImportError:
        missing.append("manhwa-bubbles  (pip install -e /path/to/bubble)")
    if missing:
        print("Missing packages:\n  " + "\n  ".join(missing))
        print("\nInstall them first. See README for instructions.")
        sys.exit(1)


def get_chapter_files() -> dict[int, Path]:
    """Return {chapter_number: path} for all existing chapter .md files."""
    chapters_dir = DATA / "chapters"
    if not chapters_dir.exists():
        return {}
    result = {}
    for f in sorted(chapters_dir.glob("*.md")):
        try:
            num = int(f.name.split("_")[0])
            result[num] = f
        except (ValueError, IndexError):
            pass
    return result


def load_bible() -> "WorldBible":
    """Load the world bible from data/bible.json."""
    from app.models import WorldBible

    bible_path = DATA / "bible.json"
    if not bible_path.exists():
        print("No bible.json found. Run 'python main.py' first.")
        sys.exit(1)
    data = json.loads(bible_path.read_text(encoding="utf-8"))
    return WorldBible.model_validate(data)


def build_manga_config(panels: int | str = "auto") -> "manga_ai.config.Config":
    """Create a manga-ai-bot Config pre-populated from env and storyAI settings."""
    from manga_ai.config import Config

    cfg = Config.from_env()
    cfg.scenario.panels = panels if isinstance(panels, int) else 6
    cfg.output.save_individual_panels = True
    return cfg


# ---------------------------------------------------------------------------
# HF Inference client
# ---------------------------------------------------------------------------

def _get_hf_inference_client():
    """Create a HuggingFace InferenceClient using HF_TOKEN from env."""
    from huggingface_hub import InferenceClient

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_KEY")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        print("ERROR: HF_TOKEN not set. Add it to your .env file.")
        print("Get a free token at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    return InferenceClient(token=token)


def _build_panel_llm_call(config):
    """LLM for panel/beat extraction: try HF Inference router first, then app.llm.

    Hugging Face monthly inference credits can run out (HTTP 402). In that case
    we automatically fall back to OPENAI_API_KEY / ANTHROPIC_API_KEY via ``app.llm.call_llm``.
    """
    from app.llm import call_llm as story_llm

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_KEY")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    llm_model = config.model.llm_model
    hf_client = None
    if hf_token:
        from openai import OpenAI

        hf_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )

    def llm_call(prompt: str) -> str:
        if hf_client:
            try:
                resp = hf_client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500,
                )
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                logger.warning(
                    "HF Inference router LLM failed (%s). "
                    "Falling back to OPENAI_API_KEY / ANTHROPIC_API_KEY (app.llm) ...",
                    e,
                )
        return story_llm(prompt)

    return llm_call


# ---------------------------------------------------------------------------
# Image generation (sequential, continuity-aware)
# ---------------------------------------------------------------------------

def _generate_single_panel_api(
    panel,
    output_dir: Path,
    reference_images: list[Path],
    sd_model: str,
    hf_client,
    verbose: bool = False,
) -> tuple[Path, dict, int | None]:
    """Generate one panel via HF API with continuity-aware reference ordering."""
    from app.manga_prep.references import build_panel_reference_image

    prompt = panel.prompt_positive
    negative = panel.prompt_negative
    i = panel.panel_id
    seed_used = panel.seed_hint

    log_entry = {
        "panel": i,
        "model": sd_model,
        "prompt": prompt,
        "negative_prompt": negative,
        "shot_type": panel.shot_type,
        "camera_angle": panel.camera_angle,
        "characters": [c.character_id for c in panel.characters],
        "location": panel.location_id,
        "scene_summary": panel.scene_summary,
        "continuity_group": panel.continuity_group,
        "reference_panel_id": panel.reference_panel_id,
        "seed_hint": panel.seed_hint,
    }

    if verbose:
        print(f"\n  --- Panel {i} (API) ---")
        print(f"  Shot:       {panel.shot_type} / {panel.camera_angle}")
        print(f"  Group:      {panel.continuity_group}")
        print(f"  Ref panel:  {panel.reference_panel_id}")
        print(f"  Seed hint:  {panel.seed_hint}")
        print(f"  Refs:       {[p.name for p in reference_images]}")
        print(f"  Prompt:     {prompt[:200]}...")

    logger.info("  Panel %d (model=%s, group=%s) ...", i, sd_model, panel.continuity_group[:30])

    # Build composite reference from the ordered reference_images list
    reference_image = None
    if reference_images:
        from PIL import Image as PILImage, ImageOps
        imgs = []
        for rp in reference_images[:3]:
            try:
                imgs.append(PILImage.open(str(rp)).convert("RGB"))
            except Exception:
                pass
        if imgs:
            if len(imgs) == 1:
                reference_image = ImageOps.fit(imgs[0], (768, 1024))
            else:
                canvas = PILImage.new("RGB", (768, 1024), "white")
                slot_h = 1024 // len(imgs)
                for idx, im in enumerate(imgs):
                    thumb = ImageOps.fit(im, (768, slot_h))
                    canvas.paste(thumb, (0, idx * slot_h))
                reference_image = canvas

    try:
        if reference_image is not None:
            image = hf_client.image_to_image(
                reference_image,
                prompt=prompt,
                negative_prompt=negative,
                model=sd_model,
                num_inference_steps=30,
                guidance_scale=7.5,
            )
            log_entry["status"] = "ok_img2img"
            log_entry["reference_conditioning"] = True
        else:
            image = hf_client.text_to_image(
                prompt, model=sd_model, negative_prompt=negative,
                width=768, height=1024,
            )
            log_entry["status"] = "ok"
            log_entry["reference_conditioning"] = False
    except Exception as e:
        logger.warning("  Panel %d API call failed: %s", i, e)
        try:
            image = hf_client.text_to_image(
                prompt, model=sd_model, negative_prompt=negative,
                width=768, height=1024,
            )
            log_entry["status"] = "ok_text_fallback"
            log_entry["reference_conditioning"] = False
        except Exception as e2:
            logger.error("  Panel %d failed completely: %s", i, e2)
            from PIL import Image as PILImage2, ImageDraw
            image = PILImage2.new("RGB", (768, 1024), color=(32, 32, 32))
            draw = ImageDraw.Draw(image)
            draw.text((30, 40), f"Panel {i} failed: {e2}", fill=(240, 240, 240))
            log_entry["status"] = f"failed: {e2}"

    panel_path = output_dir / f"panel_{i:02d}.png"
    image.save(str(panel_path), quality=95)
    log_entry["seed"] = seed_used
    logger.info("  Saved %s", panel_path.name)
    return panel_path, log_entry, seed_used


def _generate_single_panel_local(
    panel,
    output_dir: Path,
    reference_images: list[Path],
    config,
    pipeline_cache: dict,
    verbose: bool = False,
) -> tuple[Path, dict, int]:
    """Generate one panel locally with continuity-aware seed and references."""
    import torch
    from manga_ai.pipelines.diffusion import get_cached_pipeline

    if "pipeline" not in pipeline_cache:
        pipeline_cache["pipeline"], (pipeline_cache["device"], pipeline_cache["dtype"]) = (
            get_cached_pipeline(
                config.model.stable_diffusion_model,
                config.model.device,
                (config.model.REMOVED_TOKENtoken or None),
            )
        )
    pipeline = pipeline_cache["pipeline"]
    device = pipeline_cache["device"]

    prompt = panel.prompt_positive
    negative = panel.prompt_negative
    i = panel.panel_id

    seed = panel.seed_hint if panel.seed_hint is not None else __import__("random").randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    log_entry = {
        "panel": i,
        "model": config.model.stable_diffusion_model,
        "prompt": prompt,
        "negative_prompt": negative,
        "characters": [c.character_id for c in panel.characters],
        "location": panel.location_id,
        "shot_type": panel.shot_type,
        "continuity_group": panel.continuity_group,
        "reference_panel_id": panel.reference_panel_id,
        "seed_hint": panel.seed_hint,
    }

    if verbose:
        print(f"\n  --- Panel {i} (local, seed={seed}) ---")
        print(f"  Group:    {panel.continuity_group}")
        print(f"  Ref panel: {panel.reference_panel_id}")
        print(f"  Refs:     {[p.name for p in reference_images]}")
        print(f"  Prompt:   {prompt[:200]}...")

    # IP-Adapter with ordered references: prev panel first, then char, then loc
    ref_kwargs = {}
    if reference_images and hasattr(pipeline, "set_ip_adapter_scale"):
        from PIL import Image as PILImage
        try:
            pipeline.set_ip_adapter_scale(0.6)
            ref_imgs = [PILImage.open(str(rp)).convert("RGB") for rp in reference_images[:4]]
            ref_kwargs["ip_adapter_image"] = ref_imgs if len(ref_imgs) > 1 else ref_imgs[0]
            if verbose:
                print(f"  IP-Adapter: {len(ref_imgs)} reference(s)")
        except Exception as e:
            logger.debug("IP-Adapter not available: %s", e)

    logger.info("  Generating panel %d (local, seed=%d, group=%s) ...", i, seed, panel.continuity_group[:30])
    try:
        result_img = pipeline(
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=getattr(config.generation, "guidance_scale", 7.5),
            num_inference_steps=getattr(config.generation, "num_inference_steps", 50),
            generator=generator,
            width=768, height=1024,
            **ref_kwargs,
        ).images[0]
        log_entry["seed"] = seed
        log_entry["status"] = "ok"
        log_entry["ip_adapter"] = bool(ref_kwargs)
    except Exception as e:
        logger.error("  Panel %d failed: %s", i, e)
        from PIL import Image as PILImage2, ImageDraw
        result_img = PILImage2.new("RGB", (768, 1024), color=(32, 32, 32))
        draw = ImageDraw.Draw(result_img)
        draw.text((30, 40), f"Panel {i} failed: {e}", fill=(240, 240, 240))
        log_entry["status"] = f"failed: {e}"

    panel_path = output_dir / f"panel_{i:02d}.png"
    result_img.save(str(panel_path), quality=95)
    logger.info("  Saved %s (seed=%d)", panel_path.name, seed)
    return panel_path, log_entry, seed


# ---------------------------------------------------------------------------
# Bubble overlay
# ---------------------------------------------------------------------------

def _estimate_character_positions(
    panel,
    img_w: int = 768,
    img_h: int = 1024,
) -> list[dict]:
    """Estimate character bounding boxes from PanelSpec for bubble placement.

    Distributes characters left-to-right across the panel, with vertical
    placement adjusted for shot type. Returns dicts compatible with the
    bubble engine's CharacterPosition schema.
    """
    chars = panel.characters
    if not chars:
        return []

    n = len(chars)
    slot_w = img_w // max(n, 1)

    # Vertical region depends on shot type
    if panel.shot_type == "close-up" or panel.shot_type == "extreme close-up":
        y1_frac, y2_frac = 0.10, 0.80
    elif panel.shot_type == "wide":
        y1_frac, y2_frac = 0.20, 0.90
    else:
        y1_frac, y2_frac = 0.15, 0.85

    positions = []
    for i, ch in enumerate(chars):
        x1 = i * slot_w + slot_w // 6
        x2 = (i + 1) * slot_w - slot_w // 6
        y1 = int(img_h * y1_frac)
        y2 = int(img_h * y2_frac)
        positions.append({
            "name": ch.character_id,
            "bbox": [x1, y1, x2, y2],
        })
    return positions


def _dialogue_position_hint(char_idx: int, n_chars: int) -> str:
    """Assign a position hint based on character slot in the panel."""
    if n_chars <= 1:
        return "center"
    if char_idx == 0:
        return "left"
    if char_idx == n_chars - 1:
        return "right"
    return "center"


def apply_bubbles(
    panel_paths: list[Path],
    panels: list,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[list[Path], list[dict]]:
    """Overlay speech bubbles from PanelSpec dialogues.

    Uses panel.character_boxes as the single source of truth for speaker
    positions (populated earlier in process_chapter). Passes ALL dialogues
    with position hints to the bubble engine.
    """
    from manhwa_bubbles.pipeline import process_manga_page

    bubble_paths: list[Path] = []
    bubble_logs: list[dict] = []
    for panel_path, panel in zip(panel_paths, panels):
        idx = panel.panel_id

        active_dialogues = [d for d in panel.dialogues if d.text.strip()]
        if not active_dialogues:
            bubble_paths.append(panel_path)
            bubble_logs.append({"panel": idx, "skipped": True})
            continue

        # Use character_boxes as single source of truth for positions
        box_map = {b.character_id: i for i, b in enumerate(panel.character_boxes)}
        n_boxes = len(panel.character_boxes)

        scenario_dialogues = []
        for dlg in active_dialogues:
            ci = box_map.get(dlg.character_id, 0)
            hint = _dialogue_position_hint(ci, n_boxes)
            scenario_dialogues.append({
                "character": dlg.character_id,
                "text": dlg.text,
                "emotion": dlg.emotion or "normal",
                "position_hint": hint,
            })

        char_positions = [
            {"name": b.character_id, "bbox": b.bbox}
            for b in panel.character_boxes
        ]

        log_entry = {
            "panel": idx,
            "dialogues": [
                {"speaker": d.character_id, "text": d.text, "emotion": d.emotion}
                for d in active_dialogues
            ],
            "character_positions": char_positions,
        }

        if verbose:
            print(f"\n  --- Bubble panel {idx} ({len(active_dialogues)} dialogue(s)) ---")
            for d in active_dialogues:
                ci = box_map.get(d.character_id, 0)
                hint = _dialogue_position_hint(ci, n_boxes)
                print(f"    {d.character_id} [{hint}]: \"{d.text[:60]}\" ({d.emotion})")
            if char_positions:
                print(f"    Positions: {[(p['name'], p['bbox']) for p in char_positions]}")

        scenario = {
            "panels": [{
                "panel_id": idx,
                "image": panel_path.name,
                "dialogues": scenario_dialogues,
                "characters": char_positions,
            }]
        }

        out_path = output_dir / f"panel_{idx:02d}_bubbles.png"

        try:
            process_manga_page(
                str(panel_path),
                scenario,
                output_path=str(out_path),
                use_yolo=True,
            )
            bubble_paths.append(out_path)
            log_entry["status"] = "ok"
            log_entry["yolo"] = True
            logger.info("  Bubbles added (YOLO) -> %s", out_path.name)
        except Exception:
            try:
                process_manga_page(
                    str(panel_path),
                    scenario,
                    output_path=str(out_path),
                    use_yolo=False,
                )
                bubble_paths.append(out_path)
                log_entry["status"] = "ok"
                log_entry["yolo"] = False
                logger.info("  Bubbles added (no YOLO) -> %s", out_path.name)
            except Exception as e:
                logger.warning("  Bubble overlay failed for panel %d: %s", idx, e)
                bubble_paths.append(panel_path)
                log_entry["status"] = f"failed: {e}"

        bubble_logs.append(log_entry)
    return bubble_paths, bubble_logs


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_chapter(panel_paths: list[Path], output_path: Path) -> Path:
    """Stack panels vertically into a single manhwa chapter image."""
    from manga_ai.pipelines.assemble import ManhwaAssembler
    from PIL import Image

    panels = [Image.open(str(p)) for p in panel_paths]
    assembler = ManhwaAssembler()
    assembler.assemble_panels(panels, output_path=str(output_path))
    logger.info("  Assembled -> %s", output_path.name)
    return output_path


# ---------------------------------------------------------------------------
# Main per-chapter pipeline
# ---------------------------------------------------------------------------

def process_chapter(
    chapter_num: int,
    chapter_path: Path,
    bible,
    config,
    llm_call,
    n_panels: int | str = "auto",
    use_bubbles: bool = True,
    provider: str = "api",
    verbose: bool = False,
    max_regen: int = 1,
) -> Path:
    """Full pipeline for one chapter using manga_prep layer.

    Generation is sequential: each panel can reference the previous panel
    in the same continuity group, reuse deterministic seeds, and track
    scene state.  After all panels are generated, a drift validator scores
    visual consistency and regenerates the worst offenders.
    """
    from app.manga_prep import (
        extract_panels, check_visual_continuity, validate_panel_specs,
        build_panel_prompts, assign_continuity_groups,
    )
    from app.manga_prep.models import CharacterBox, SceneState
    from app.manga_prep.scene_continuity import (
        build_scene_state_in, update_scene_state_out,
        save_scene_state, collect_reference_images,
    )
    from app.manga_prep.drift_validator import validate_chapter_drift
    from app.manga_prep.references import load_character_references, load_location_references

    ch_dir = MANGA_DIR / f"chapter_{chapter_num:03d}"
    ch_dir.mkdir(parents=True, exist_ok=True)

    chapter_text = chapter_path.read_text(encoding="utf-8")
    logger.info("Chapter %d: %d words", chapter_num, len(chapter_text.split()))

    # Step 1: Extract structured PanelSpec from chapter text
    logger.info("Step 1/6: Extracting panel scenes from chapter ...")
    panels = extract_panels(
        chapter_text, chapter_num, n_panels, bible, llm_call=llm_call,
    )
    logger.info(
        "Chapter %d adaptation target=%s -> extracted %d panels",
        chapter_num,
        n_panels,
        len(panels),
    )
    if verbose:
        print(f"\n  === Extracted {len(panels)} panels ===")
        for p in panels:
            chars = [c.character_id for c in p.characters]
            dlg = p.dialogues[0].text[:60] if p.dialogues else "(no dialogue)"
            print(f"  Panel {p.panel_id}: {p.shot_type} | chars={chars} | loc={p.location_id}")
            print(f"    summary: {p.scene_summary}")
            print(f"    dialogue: {dlg}")

    # Step 2: Visual continuity check (prose-level)
    logger.info("Step 2/6: Checking visual continuity ...")
    issues = check_visual_continuity(bible, chapter_text, chapter_num)
    if issues:
        print(f"\n  VISUAL CONTINUITY ISSUES in chapter {chapter_num}:")
        for issue in issues:
            print(f"    [{issue.severity}] {issue.character_id}.{issue.field}: "
                  f"canon='{issue.canon_value}' chapter='{issue.chapter_value}'")
        print()

    # Step 3: Build prompts + assign continuity groups + validate
    logger.info("Step 3/6: Building prompts & continuity groups ...")
    panels = build_panel_prompts(panels, bible)
    panels = assign_continuity_groups(panels)

    panel_issues = validate_panel_specs(panels, bible)
    if panel_issues:
        print(f"\n  PANEL SPEC ISSUES ({len(panel_issues)}):")
        for pi in panel_issues:
            label = pi.character_id or pi.location_id or "panel"
            print(f"    [{pi.severity}] {label}.{pi.field}: {pi.chapter_value}")
        print()

    # Populate character_boxes
    for panel in panels:
        positions = _estimate_character_positions(panel)
        panel.character_boxes = [
            CharacterBox(
                character_id=pos["name"],
                bbox=pos["bbox"],
                confidence=0.3,
            )
            for pos in positions
        ]

    if verbose:
        groups = {}
        for p in panels:
            groups.setdefault(p.continuity_group, []).append(p.panel_id)
        print(f"\n  Continuity groups: {len(groups)}")
        for gk, pids in groups.items():
            print(f"    {gk[:50]} -> panels {pids}")

    # Load references
    char_refs = load_character_references(bible)
    loc_refs = load_location_references(bible)
    if char_refs or loc_refs:
        logger.info("  Loaded %d character + %d location reference images",
                     len(char_refs), len(loc_refs))

    # Step 4: Sequential generation with continuity tracking
    logger.info("Step 4/6: Generating panel images sequentially ...")
    scene_state = SceneState(chapter=chapter_num)
    panel_path_map: dict[int, Path] = {}
    panel_paths: list[Path] = []
    image_logs: list[dict] = []

    hf_client = _get_hf_inference_client() if provider == "api" else None
    sd_model = os.getenv("SD_MODEL", "black-forest-labs/FLUX.1-schnell")
    pipeline_cache: dict = {}

    for panel in panels:
        panel.scene_state_in = build_scene_state_in(panel, scene_state)

        ref_images = collect_reference_images(panel, panel_path_map, char_refs, loc_refs)

        if provider == "api":
            p_path, log_entry, seed = _generate_single_panel_api(
                panel, ch_dir, ref_images, sd_model, hf_client, verbose=verbose,
            )
        else:
            p_path, log_entry, seed = _generate_single_panel_local(
                panel, ch_dir, ref_images, config, pipeline_cache, verbose=verbose,
            )

        panel_path_map[panel.panel_id] = p_path
        panel_paths.append(p_path)
        image_logs.append(log_entry)

        scene_state = update_scene_state_out(panel, str(p_path), scene_state, seed_used=seed)

    save_scene_state(scene_state, ch_dir)

    # Save panel specs (fully populated with continuity data)
    panel_dicts = [p.model_dump() for p in panels]
    (ch_dir / "panel_specs.json").write_text(
        json.dumps(panel_dicts, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Step 5: Post-generation drift validation + selective regeneration
    logger.info("Step 5/6: Checking visual drift ...")
    drift_scores = validate_chapter_drift(panels, panel_path_map, char_refs)
    drift_log = [d.model_dump() for d in drift_scores]

    drifted = [d for d in drift_scores if d.drifted]
    if drifted and max_regen > 0:
        drifted_sorted = sorted(drifted, key=lambda d: d.overall)
        to_regen = drifted_sorted[:max_regen]
        logger.info("  Regenerating %d drifted panel(s) ...", len(to_regen))
        for ds in to_regen:
            panel = next((p for p in panels if p.panel_id == ds.panel_id), None)
            if not panel:
                continue
            ref_images = collect_reference_images(panel, panel_path_map, char_refs, loc_refs)
            if provider == "api":
                p_path, log_entry, seed = _generate_single_panel_api(
                    panel, ch_dir, ref_images, sd_model, hf_client, verbose=verbose,
                )
            else:
                p_path, log_entry, seed = _generate_single_panel_local(
                    panel, ch_dir, ref_images, config, pipeline_cache, verbose=verbose,
                )
            log_entry["regenerated"] = True
            log_entry["drift_score_before"] = ds.overall
            image_logs.append(log_entry)
            idx = next(i for i, pp in enumerate(panel_paths) if pp.name == p_path.name)
            panel_paths[idx] = p_path
            panel_path_map[panel.panel_id] = p_path
            scene_state = update_scene_state_out(panel, str(p_path), scene_state, seed_used=seed)
        save_scene_state(scene_state, ch_dir)
    elif drifted:
        logger.warning("  %d panel(s) drifted but max_regen=0, skipping", len(drifted))

    if verbose and drift_scores:
        print(f"\n  === Drift scores ===")
        for ds in drift_scores:
            flag = " DRIFTED" if ds.drifted else ""
            print(f"  Panel {ds.panel_id}: overall={ds.overall:.3f} "
                  f"face={ds.face_similarity:.3f} outfit={ds.outfit_similarity:.3f} "
                  f"bg={ds.background_similarity:.3f}{flag}")

    # Step 6: Bubbles + assembly
    bubble_logs = []
    if use_bubbles:
        logger.info("Step 6/6: Adding speech bubbles + assembling ...")
        final_panels, bubble_logs = apply_bubbles(panel_paths, panels, ch_dir, verbose=verbose)
    else:
        logger.info("Step 6/6: Skipping bubbles, assembling ...")
        final_panels = panel_paths

    chapter_img = assemble_chapter(final_panels, ch_dir / "chapter.png")

    # Save full log
    full_log = {
        "chapter": chapter_num,
        "provider": provider,
        "panel_specs": panel_dicts,
        "continuity_issues": [i.model_dump() for i in issues] if issues else [],
        "panel_validation_issues": [i.model_dump() for i in panel_issues] if panel_issues else [],
        "image_prompts": image_logs,
        "drift_scores": drift_log,
        "bubbles": bubble_logs,
    }
    (ch_dir / "prompts.json").write_text(
        json.dumps(full_log, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("  Full log saved -> %s", ch_dir / "prompts.json")

    return chapter_img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    def _parse_panels_arg(value: str) -> int | str:
        value = value.strip().lower()
        if value == "auto":
            return "auto"
        try:
            parsed = int(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError("panels must be an integer or 'auto'") from e
        if parsed <= 0:
            raise argparse.ArgumentTypeError("panels must be positive")
        return parsed

    parser = argparse.ArgumentParser(description="Generate manga from storyAI chapters")
    parser.add_argument("--chapter", type=int, help="Process a single chapter number")
    parser.add_argument(
        "--panels",
        type=_parse_panels_arg,
        default="auto",
        help="Panels per chapter (integer) or 'auto' for beat-based estimation. Default: auto",
    )
    parser.add_argument("--no-bubbles", action="store_true", help="Skip bubble overlay step")
    parser.add_argument(
        "--provider",
        choices=["api", "local"],
        default=os.getenv("IMAGE_PROVIDER", "api"),
        help="Image generation backend: 'api' (HF Inference, no GPU) or 'local' (needs GPU). Default: api",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the SD model (e.g. black-forest-labs/FLUX.1-schnell)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all prompts (image, scenario, bubble text) to the console",
    )
    parser.add_argument(
        "--generate-refs",
        action="store_true",
        help="Generate reference images for all characters and locations before processing chapters",
    )
    parser.add_argument(
        "--max-regen",
        type=int,
        default=1,
        help="Max panels to regenerate per chapter if drift is detected (default 1, 0 to disable)",
    )
    parser.add_argument(
        "--no-llm-panels",
        action="store_true",
        help="Skip LLM for panel/beat extraction; use deterministic heuristic only (no HF/OpenAI credits)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _check_imports()

    MANGA_DIR.mkdir(parents=True, exist_ok=True)

    # Load narrative canon
    bible = load_bible()
    print(f"Loaded bible: {bible.title}")
    print(f"Characters:   {[c.name for c in bible.main_characters]}")
    if bible.locations:
        print(f"Locations:    {[loc.name for loc in bible.locations]}")

    # Optionally generate reference images for consistency
    if args.generate_refs:
        from app.manga_prep.references import generate_all_references
        print("\nGenerating reference images for characters and locations ...")
        hf = _get_hf_inference_client()
        sd = args.model or os.getenv("SD_MODEL", "black-forest-labs/FLUX.1-schnell")
        refs = generate_all_references(bible, hf_client=hf, sd_model=sd)
        print(f"  Characters: {list(refs['characters'].keys())}")
        print(f"  Locations:  {list(refs['locations'].keys())}")
        print()

    chapter_files = get_chapter_files()
    if not chapter_files:
        print("No chapters found. Run 'python main.py' first to generate story chapters.")
        return

    if args.chapter:
        if args.chapter not in chapter_files:
            print(f"Chapter {args.chapter} not found. Available: {sorted(chapter_files.keys())}")
            return
        chapter_files = {args.chapter: chapter_files[args.chapter]}

    config = build_manga_config(panels=args.panels)

    if args.model:
        config.model.stable_diffusion_model = args.model
        os.environ["SD_MODEL"] = args.model

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_KEY")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

    # LLM for scene/beat extraction (HF router with fallback to app.llm)
    llm_call = None
    if not args.no_llm_panels:
        llm_call = _build_panel_llm_call(config)
    else:
        logger.info("Panel extraction: heuristic only (--no-llm-panels)")

    provider = args.provider
    sd_model = args.model or os.getenv("SD_MODEL", "black-forest-labs/FLUX.1-schnell")

    print(f"\nProvider:  {provider}" + (f"  (model: {sd_model})" if provider == "api" else " (local GPU)"))
    print(f"Chapters:  {sorted(chapter_files.keys())}")
    if args.panels == "auto":
        print("Panels:    auto (estimated from chapter beats/word count)")
    else:
        print(f"Panels:    {args.panels} per chapter")
    print(f"Bubbles:   {'yes' if not args.no_bubbles else 'no'}")
    print(f"Panel LLM: {'heuristic only' if args.no_llm_panels else 'HF router -> app.llm fallback'}")
    print(f"Output:    {MANGA_DIR}/\n")

    if provider == "api" and not hf_token:
        print("ERROR: HF_TOKEN is required for API mode.")
        print("Add HF_TOKEN=hf_... to your .env file")
        print("Get a free token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    for ch_num, ch_path in sorted(chapter_files.items()):
        print(f"{'=' * 50}")
        print(f"Processing chapter {ch_num}: {ch_path.name}")
        print(f"{'=' * 50}")
        try:
            result = process_chapter(
                ch_num, ch_path, bible, config, llm_call,
                n_panels=args.panels,
                use_bubbles=not args.no_bubbles,
                provider=provider,
                verbose=args.verbose,
                max_regen=args.max_regen,
            )
            print(f"Done -> {result}\n")
        except Exception as e:
            logger.error("Chapter %d failed: %s", ch_num, e, exc_info=True)
            print(f"FAILED: {e}\n")

    print(f"\nAll done! Manga output: {MANGA_DIR}/")
    print("Run 'python serve.py' to view in browser.")


if __name__ == "__main__":
    main()
