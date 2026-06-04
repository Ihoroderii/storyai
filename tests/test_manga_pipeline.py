import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_manga
from PIL import Image, ImageChops, ImageDraw, ImageStat
from app.manga_prep.extractor import extract_panels
from app.manga_prep.models import CharacterBox, PanelDialogue, PanelSpec
from app.models import Character, Location, VisualProfile, WorldBible


CHAPTER_TEXT = """
At night, Aiko Tanaka pushed open the hatch to the Clocktower Rooftop. Wind snapped at her braid as Ren Mori followed with the brass compass glowing in his hand.

"We're late," Aiko whispered. Moonlight painted the old gears blue while Ren scanned the broken skyline for the signal fire.

A siren rune burst above the rooftop and Ren lunged toward the railing. Aiko grabbed his sleeve and shouted, "Don't jump!"

For one breath the world went silent. Ren looked at Aiko, tears bright in his eyes, and said, "Then trust me."

He raised the brass compass, and a portal of white fire opened over the tower as both of them stepped forward.
""".strip()


def build_test_bible() -> WorldBible:
    return WorldBible(
        title="Clockwork Oath",
        genre="fantasy",
        premise="Two scouts chase a city-sized mystery through living machinery.",
        setting="A vertical city of towers, gears, and rune engines.",
        main_characters=[
            Character(
                name="Aiko Tanaka",
                role="protagonist",
                visual=VisualProfile(
                    hair_color="silver",
                    eye_color="green",
                    outfit_base="storm scout coat",
                    signature_prop="signal knife",
                ),
            ),
            Character(
                name="Ren Mori",
                role="deuteragonist",
                visual=VisualProfile(
                    hair_color="black",
                    eye_color="amber",
                    outfit_base="dark mechanic uniform",
                    signature_prop="brass compass",
                ),
            ),
        ],
        locations=[
            Location(
                location_id="clocktower_rooftop",
                name="Clocktower Rooftop",
                description="An exposed rooftop crowded with rusted gears and signal chains.",
                anchor_objects=["gears", "railing", "signal chains"],
                lighting="moonlit",
                mood="tense",
            )
        ],
    )


def grayscale_stddev(image: Image.Image) -> float:
    return ImageStat.Stat(image.convert("L")).stddev[0]


def grayscale_entropy(image: Image.Image) -> float:
    return image.convert("L").entropy()


def changed_pixel_ratio(before: Image.Image, after: Image.Image, tolerance: int = 3) -> float:
    diff = ImageChops.difference(before.convert("RGB"), after.convert("RGB")).convert("L")
    hist = diff.histogram()
    unchanged = sum(hist[:tolerance + 1])
    total = before.width * before.height
    return 1.0 - (unchanged / total)


def count_dark_pixels(image: Image.Image, threshold: int = 40) -> int:
    hist = image.convert("L").histogram()
    return sum(hist[:threshold + 1])


class MangaPipelineTests(unittest.TestCase):
    def test_extract_panels_auto_fallback_builds_story_panels(self):
        bible = build_test_bible()

        panels = extract_panels(
            chapter_text=CHAPTER_TEXT,
            chapter_num=3,
            n_panels="auto",
            bible=bible,
            llm_call=None,
        )

        self.assertGreaterEqual(len(panels), 5)
        self.assertEqual([panel.panel_id for panel in panels], list(range(1, len(panels) + 1)))
        self.assertTrue(all(panel.scene_summary for panel in panels))
        self.assertTrue(any(panel.location_id == "clocktower_rooftop" for panel in panels))
        self.assertTrue(any(panel.time_of_day == "night" for panel in panels))

        all_character_ids = {
            character.character_id
            for panel in panels
            for character in panel.characters
        }
        self.assertIn("Aiko Tanaka", all_character_ids)
        self.assertIn("Ren Mori", all_character_ids)

        all_dialogue_text = [
            dialogue.text
            for panel in panels
            for dialogue in panel.dialogues
        ]
        self.assertTrue(any("We're late" in text for text in all_dialogue_text))
        self.assertTrue(any("Don't jump!" in text for text in all_dialogue_text))

    def test_apply_bubbles_real_engine_changes_image_and_preserves_quality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            source_path = temp_root / "panel_01.png"

            source = Image.new("RGB", (768, 1024), "#f2f2f2")
            draw = ImageDraw.Draw(source)
            draw.rounded_rectangle((70, 110, 350, 940), radius=28, fill="#cfcfcf", outline="#8a8a8a", width=4)
            draw.rounded_rectangle((420, 110, 700, 940), radius=28, fill="#d8d8d8", outline="#8a8a8a", width=4)
            source.save(source_path)

            panel = PanelSpec(
                panel_id=1,
                dialogues=[
                    PanelDialogue(character_id="Aiko Tanaka", text="We're late!", emotion="excited"),
                    PanelDialogue(character_id="Ren Mori", text="Then trust me.", emotion="normal"),
                ],
                character_boxes=[
                    CharacterBox(character_id="Aiko Tanaka", bbox=[70, 110, 350, 940], confidence=0.95),
                    CharacterBox(character_id="Ren Mori", bbox=[420, 110, 700, 940], confidence=0.95),
                ],
            )

            output_paths, logs = generate_manga.apply_bubbles([source_path], [panel], temp_root, verbose=False)
            bubble_path = output_paths[0]

            self.assertTrue(bubble_path.exists())
            self.assertEqual(logs[0]["status"], "ok")

            before = Image.open(source_path).convert("RGB")
            after = Image.open(bubble_path).convert("RGB")

            self.assertEqual(before.size, after.size)
            self.assertGreater(changed_pixel_ratio(before, after), 0.02)
            self.assertGreater(count_dark_pixels(after), count_dark_pixels(before) + 500)
            self.assertGreater(grayscale_stddev(after), 10.0)
            self.assertGreater(grayscale_entropy(after), 1.5)

    def test_process_chapter_writes_pipeline_artifacts_with_mocked_generation(self):
        bible = build_test_bible()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            chapter_path = temp_root / "007_test.md"
            chapter_path.write_text(CHAPTER_TEXT, encoding="utf-8")

            output_root = temp_root / "manga_output"

            def fake_generate_single_panel_local(
                panel,
                output_dir: Path,
                reference_images,
                config,
                pipeline_cache,
                verbose: bool = False,
            ):
                panel_path = output_dir / f"panel_{panel.panel_id:02d}.png"
                panel_path.write_bytes(b"fake-panel")
                log_entry = {
                    "panel": panel.panel_id,
                    "status": "ok",
                    "seed": panel.seed_hint,
                    "reference_count": len(reference_images),
                }
                return panel_path, log_entry, panel.seed_hint

            def fake_assemble_chapter(panel_paths, output_path: Path):
                output_path.write_bytes(b"fake-chapter")
                return output_path

            with (
                patch.object(generate_manga, "MANGA_DIR", output_root),
                patch.object(generate_manga, "_generate_single_panel_local", side_effect=fake_generate_single_panel_local),
                patch.object(generate_manga, "assemble_chapter", side_effect=fake_assemble_chapter),
                patch("app.manga_prep.references.load_character_references", return_value={}),
                patch("app.manga_prep.references.load_location_references", return_value={}),
                patch("app.manga_prep.drift_validator.validate_chapter_drift", return_value=[]),
            ):
                result = generate_manga.process_chapter(
                    chapter_num=7,
                    chapter_path=chapter_path,
                    bible=bible,
                    config=object(),
                    llm_call=None,
                    n_panels=4,
                    use_bubbles=False,
                    provider="local",
                    verbose=False,
                    max_regen=0,
                )

            chapter_dir = output_root / "chapter_007"
            panel_specs_path = chapter_dir / "panel_specs.json"
            prompts_path = chapter_dir / "prompts.json"
            scene_state_path = chapter_dir / "scene_state.json"

            self.assertEqual(result, chapter_dir / "chapter.png")
            self.assertTrue(result.exists())
            self.assertTrue(panel_specs_path.exists())
            self.assertTrue(prompts_path.exists())
            self.assertTrue(scene_state_path.exists())

            panel_specs = json.loads(panel_specs_path.read_text(encoding="utf-8"))
            self.assertEqual(len(panel_specs), 4)
            self.assertTrue(all(spec["prompt_positive"] for spec in panel_specs))
            self.assertTrue(all(spec["prompt_negative"] for spec in panel_specs))
            self.assertTrue(all(spec["continuity_group"] for spec in panel_specs))
            self.assertTrue(all(isinstance(spec["seed_hint"], int) for spec in panel_specs))
            self.assertTrue(any(spec["character_boxes"] for spec in panel_specs))

            prompt_log = json.loads(prompts_path.read_text(encoding="utf-8"))
            self.assertEqual(prompt_log["chapter"], 7)
            self.assertEqual(prompt_log["provider"], "local")
            self.assertEqual(len(prompt_log["panel_specs"]), 4)
            self.assertEqual(len(prompt_log["image_prompts"]), 4)
            self.assertEqual(prompt_log["drift_scores"], [])

            scene_state = json.loads(scene_state_path.read_text(encoding="utf-8"))
            self.assertEqual(scene_state["chapter"], 7)
            self.assertEqual(scene_state["last_panel_id"], 4)


if __name__ == "__main__":
    unittest.main()
