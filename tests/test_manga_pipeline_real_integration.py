"""Slow integration tests for the real story -> image -> bubble pipeline.

These are opt-in on purpose. They call a real image provider and the real
bubble engine, so they may be slow, require credentials, or consume credits.

Examples:
  STORYAI_RUN_REAL_IMAGE_TESTS=1 STORYAI_REAL_PROVIDER=api \
    ./menv/bin/python -m unittest tests.test_manga_pipeline_real_integration -v

  STORYAI_RUN_REAL_IMAGE_TESTS=1 STORYAI_REAL_PROVIDER=local \
  STORYAI_REAL_SD_MODEL=stabilityai/stable-diffusion-3.5-large \
    ./menv/bin/python -m unittest tests.test_manga_pipeline_real_integration -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import generate_manga
from tests.test_manga_pipeline import (
    CHAPTER_TEXT,
    build_test_bible,
    changed_pixel_ratio,
    count_dark_pixels,
    grayscale_entropy,
    grayscale_stddev,
)


def _real_tests_enabled() -> bool:
    return os.getenv("STORYAI_RUN_REAL_IMAGE_TESTS") == "1"


def _has_hf_credentials() -> bool:
    return any(
        os.getenv(key)
        for key in ("HF_TOKEN", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN")
    )


def _auto_local_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@unittest.skipUnless(
    _real_tests_enabled(),
    "Set STORYAI_RUN_REAL_IMAGE_TESTS=1 to run real provider integration tests.",
)
class RealProviderMangaPipelineTests(unittest.TestCase):
    def _build_real_config(self, provider: str):
        config = generate_manga.build_manga_config(panels=1)
        config.output.save_individual_panels = True
        config.generation.num_inference_steps = int(os.getenv("STORYAI_REAL_NUM_STEPS", "8"))

        chosen_model = os.getenv("STORYAI_REAL_SD_MODEL") or os.getenv("SD_MODEL")
        if chosen_model:
            config.model.stable_diffusion_model = chosen_model

        if provider == "local":
            config.model.device = os.getenv("STORYAI_REAL_DEVICE", _auto_local_device())

        return config

    def test_process_chapter_with_real_provider_produces_images_bubbles_and_quality_signal(self):
        provider = os.getenv("STORYAI_REAL_PROVIDER", "api").strip().lower()
        if provider not in {"api", "local"}:
            self.skipTest("STORYAI_REAL_PROVIDER must be 'api' or 'local'.")

        if provider == "api" and not _has_hf_credentials():
            self.skipTest("HF credentials are required for STORYAI_REAL_PROVIDER=api.")

        if provider == "local" and not (os.getenv("STORYAI_REAL_SD_MODEL") or os.getenv("SD_MODEL")):
            self.skipTest("Set STORYAI_REAL_SD_MODEL or SD_MODEL for STORYAI_REAL_PROVIDER=local.")

        bible = build_test_bible()
        config = self._build_real_config(provider)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            chapter_path = temp_root / "001_real_integration.md"
            chapter_path.write_text(CHAPTER_TEXT, encoding="utf-8")

            output_root = temp_root / "manga_output"

            env_updates = {}
            chosen_model = os.getenv("STORYAI_REAL_SD_MODEL")
            if chosen_model:
                env_updates["SD_MODEL"] = chosen_model

            with (
                patch.object(generate_manga, "MANGA_DIR", output_root),
                patch.dict(os.environ, env_updates, clear=False),
            ):
                result = generate_manga.process_chapter(
                    chapter_num=1,
                    chapter_path=chapter_path,
                    bible=bible,
                    config=config,
                    llm_call=None,
                    n_panels=1,
                    use_bubbles=True,
                    provider=provider,
                    verbose=False,
                    max_regen=0,
                )

            chapter_dir = output_root / "chapter_001"
            panel_path = chapter_dir / "panel_01.png"
            bubble_path = chapter_dir / "panel_01_bubbles.png"
            prompts_path = chapter_dir / "prompts.json"
            scene_state_path = chapter_dir / "scene_state.json"

            self.assertEqual(result, chapter_dir / "chapter.png")
            self.assertTrue(panel_path.exists())
            self.assertTrue(bubble_path.exists())
            self.assertTrue(result.exists())
            self.assertTrue(prompts_path.exists())
            self.assertTrue(scene_state_path.exists())

            panel_img = Image.open(panel_path).convert("RGB")
            bubble_img = Image.open(bubble_path).convert("RGB")
            chapter_img = Image.open(result).convert("RGB")

            self.assertEqual(panel_img.size, (768, 1024))
            self.assertEqual(bubble_img.size, panel_img.size)
            self.assertEqual(chapter_img.width, panel_img.width)
            self.assertGreaterEqual(chapter_img.height, panel_img.height)

            self.assertGreater(grayscale_stddev(panel_img), 8.0)
            self.assertGreater(grayscale_entropy(panel_img), 2.0)
            self.assertGreater(changed_pixel_ratio(panel_img, bubble_img), 0.003)
            self.assertGreater(count_dark_pixels(bubble_img), count_dark_pixels(panel_img))
            self.assertGreater(grayscale_stddev(chapter_img), 8.0)
            self.assertGreater(grayscale_entropy(chapter_img), 2.0)

            prompt_log = json.loads(prompts_path.read_text(encoding="utf-8"))
            self.assertEqual(prompt_log["provider"], provider)
            self.assertEqual(len(prompt_log["image_prompts"]), 1)
            self.assertEqual(len(prompt_log["bubbles"]), 1)
            self.assertFalse(str(prompt_log["image_prompts"][0]["status"]).startswith("failed"))
            self.assertEqual(prompt_log["bubbles"][0]["status"], "ok")

            scene_state = json.loads(scene_state_path.read_text(encoding="utf-8"))
            self.assertEqual(scene_state["chapter"], 1)
            self.assertEqual(scene_state["last_panel_id"], 1)


if __name__ == "__main__":
    unittest.main()
