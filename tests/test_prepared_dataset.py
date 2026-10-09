"""Small synthetic checks for cache reuse, split safety, and interrupted builds."""

import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image, ImageOps


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepared_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepared_dataset", MODULE_PATH)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


def png_bytes(size=(12, 24), value=40):
    image = Image.new("L", size, 255)
    image.putpixel((size[0] // 2, size[1] // 2), value)
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


class PreparedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "source.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("train.png", png_bytes())
            archive.writestr("val.png", png_bytes(value=90))
            archive.writestr("reserved_test.png", b"MUST NOT READ THIS")
        self.printed = self.root / "printed.png"
        self.printed.write_bytes(png_bytes((224, 224), 10))
        self.frame = pd.DataFrame([
            dict(archive_member="train.png", image_path="", split="train", character="ا", target_idx=0,
                 source_group="form-train", source_type="handwritten", is_prepared=False),
            dict(archive_member="val.png", image_path="", split="val", character="ا", target_idx=0,
                 source_group="form-val", source_type="handwritten", is_prepared=False),
            dict(archive_member="", image_path=str(self.printed), split="train", character="آ", target_idx=32,
                 source_group="font-train", source_type="printed", is_prepared=True),
        ])
        self.prepare_calls = 0
        self.archive_calls = 0
        self.raw_reads = []
        self.args = dict(frame=self.frame, cache_root=self.root / "drive-cache",
                         runtime_root=self.root / "runtime", preprocessing={"output_size": 224, "content_size": 192},
                         preprocessing_source="def prepare_image(image):\n    pass\n", labels_sha256="labels-v1",
                         source_manifest_sha256="manifest-v1", archive_sha256="archive-v1",
                         font_splits={"font-train": "train"}, source_config={"manifest_sha256": "manifest-v1"},
                         prepare_fn=self.prepare, ensure_archive=self.ensure_archive)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, image):
        self.prepare_calls += 1
        gray = image.convert("L")
        return ImageOps.pad(gray, (224, 224), color=255)

    def ensure_archive(self):
        self.archive_calls += 1
        return self.archive

    def call(self, **overrides):
        return cache.ensure_prepared_cache(**(self.args | overrides))

    def test_cache_hit_reads_no_originals_and_reuses_exact_prepared_pixels(self):
        original_read = zipfile.ZipFile.read

        def read(archive, name, *args, **kwargs):
            if Path(archive.filename) == self.archive:
                self.raw_reads.append(name)
            return original_read(archive, name, *args, **kwargs)

        with mock.patch.object(zipfile.ZipFile, "read", read):
            first, built = self.call()
        self.assertFalse(built["cache_hit"])
        self.assertEqual(self.prepare_calls, 2)  # Printed input is already prepared.
        self.assertEqual(self.archive_calls, 1)
        self.assertEqual(self.raw_reads, ["train.png", "val.png"])
        pixels = [Path(path).read_bytes() for path in first["prepared_image_path"]]
        for row in first.itertuples():
            with Image.open(row.prepared_image_path) as prepared:
                self.assertEqual(prepared.mode, "L")
                self.assertEqual(prepared.size, (224, 224))
                if row.archive_member:
                    with zipfile.ZipFile(self.archive) as archive:
                        with Image.open(io.BytesIO(archive.read(row.archive_member))) as raw:
                            expected = ImageOps.pad(raw.convert("L"), (224, 224), color=255)
                            self.assertEqual(prepared.tobytes(), expected.tobytes())
        self.archive.unlink()
        self.printed.unlink()
        shutil.rmtree(self.root / "runtime")
        second, restored = self.call(prepare_fn=mock.Mock(side_effect=AssertionError("Preparation on a hit")),
                                     ensure_archive=mock.Mock(side_effect=AssertionError("Archive on a hit")))
        self.assertTrue(restored["cache_hit"])
        self.assertEqual(pixels, [Path(path).read_bytes() for path in second["prepared_image_path"]])
        self.assertTrue(second["is_prepared"].all())
        self.assertEqual(list(second["archive_member"]), list(self.frame["archive_member"]))
        self.assertEqual(second.loc[second["split"] == "train", "raw_preview_image_path"].tolist(), ["", ""])
        preview_path = second.loc[second["split"] == "val", "raw_preview_image_path"].iloc[0]
        with Image.open(preview_path) as thumbnail:
            self.assertEqual(thumbnail.mode, "RGB")
            self.assertLessEqual(max(thumbnail.size), 112)
            expected = ImageOps.contain(Image.open(io.BytesIO(png_bytes(value=90))).convert("RGB"), (112, 112))
            self.assertEqual(thumbnail.tobytes(), expected.tobytes())
        with zipfile.ZipFile(Path(restored["cache_dir"]) / "prepared_images.zip") as archive:
            self.assertEqual(len(archive.infolist()), len(second) + 1)
            self.assertTrue(all(entry.compress_type == zipfile.ZIP_STORED for entry in archive.infolist()))

    def test_settings_manifest_and_source_identity_changes_make_new_versions(self):
        _, first = self.call()
        for changes in [
            {"preprocessing": {"output_size": 224, "content_size": 180}},
            {"preprocessing_source": "def prepare_image(image):\n    return image\n"},
            {"source_manifest_sha256": "manifest-v2"},
            {"labels_sha256": "labels-v2"},
            {"font_splits": {"font-train": "val"}},
            {"source_config": {"manifest_sha256": "manifest-v1", "font_hash": "new-font"}},
        ]:
            with self.subTest(changes=changes):
                _, changed = self.call(**changes)
                self.assertNotEqual(changed["cache_id"], first["cache_id"])
                self.assertFalse(changed["cache_hit"])

    def test_run_location_changes_reuse_the_same_preparation_snapshot(self):
        _, first = self.call(source_config={"manifest_sha256": "manifest-v1", "run_id": "first",
                                           "runtime_root": "/content/first"})
        _, second = self.call(source_config={"manifest_sha256": "manifest-v1", "run_id": "second",
                                            "runtime_root": "/content/second"})
        self.assertEqual(first["cache_id"], second["cache_id"])
        self.assertTrue(second["cache_hit"])

    def test_incomplete_cache_is_rebuilt(self):
        _, first = self.call()
        folder = Path(first["cache_dir"])
        (folder / "cache_config.json").unlink()
        (folder / "prepared_images.zip").write_bytes(b"partial interrupted upload")
        _, second = self.call()
        self.assertFalse(second["cache_hit"])
        self.assertEqual(second["cache_id"], first["cache_id"])
        self.assertEqual(self.prepare_calls, 4)
        self.assertEqual(json.loads((folder / "cache_config.json").read_text())["status"], "complete")

    def test_completed_corrupt_cache_fails_without_preparation(self):
        _, first = self.call()
        folder = Path(first["cache_dir"])
        with (folder / "prepared_images.zip").open("ab") as handle:
            handle.write(b"corruption")
        before = self.prepare_calls
        with self.assertRaisesRegex(cache.PreparedCacheError, "SHA-256"):
            self.call()
        self.assertEqual(self.prepare_calls, before)

    def test_explicit_incomplete_status_is_rebuilt(self):
        _, first = self.call()
        marker = Path(first["cache_dir"]) / "cache_config.json"
        marker.write_text(json.dumps({"status": "building"}))
        _, second = self.call()
        self.assertFalse(second["cache_hit"])
        self.assertEqual(json.loads(marker.read_text())["status"], "complete")

    def test_corrupt_runtime_copy_restores_without_reading_sources(self):
        first, info = self.call()
        prepared_path = Path(first.iloc[0]["prepared_image_path"])
        expected = prepared_path.read_bytes()
        prepared_path.write_bytes(b"corrupt local runtime copy")
        _, restored = self.call(prepare_fn=mock.Mock(side_effect=AssertionError("Preparation on a hit")),
                                 ensure_archive=mock.Mock(side_effect=AssertionError("Archive on a hit")))
        self.assertTrue(restored["cache_hit"])
        self.assertEqual(prepared_path.read_bytes(), expected)

    def test_corrupt_validation_thumbnail_restores_without_reading_sources(self):
        first, _ = self.call()
        thumbnail_path = Path(first.loc[first["split"] == "val", "raw_preview_image_path"].iloc[0])
        expected = thumbnail_path.read_bytes()
        thumbnail_path.write_bytes(b"corrupt thumbnail")
        self.archive.unlink()
        self.printed.unlink()
        _, restored = self.call(prepare_fn=mock.Mock(side_effect=AssertionError("Preparation on a hit")),
                                 ensure_archive=mock.Mock(side_effect=AssertionError("Archive on a hit")))
        self.assertTrue(restored["cache_hit"])
        self.assertEqual(thumbnail_path.read_bytes(), expected)

    def test_test_rows_and_split_overlap_fail_before_raw_reads(self):
        test_row = self.frame.iloc[0].copy()
        test_row["split"] = "test"
        test_row["archive_member"] = "reserved_test.png"
        test_row["source_group"] = "form-test"
        with self.assertRaisesRegex(ValueError, "disallowed split"):
            self.call(frame=pd.concat([self.frame, test_row.to_frame().T], ignore_index=True))
        overlap = self.frame.copy()
        overlap.loc[1, "source_group"] = "form-train"
        with self.assertRaisesRegex(ValueError, "Source groups overlap"):
            self.call(frame=overlap)
        overlap = self.frame.copy()
        overlap.loc[1, "archive_member"] = "train.png"
        with self.assertRaisesRegex(ValueError, "Source images overlap"):
            self.call(frame=overlap)
        self.assertEqual(self.archive_calls, 0)
        self.assertEqual(self.prepare_calls, 0)

    def test_unsafe_extraction_paths_are_rejected_even_with_matching_package_hashes(self):
        _, first = self.call()
        folder = Path(first["cache_dir"])
        manifest_path = folder / "prepared_manifest.csv"
        frame = pd.read_csv(manifest_path, keep_default_na=False)
        frame.loc[0, "prepared_relative_path"] = "../escape.png"
        frame.to_csv(manifest_path, index=False)
        marker = folder / "cache_config.json"
        config = json.loads(marker.read_text())
        config["files_sha256"]["prepared_manifest.csv"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        marker.write_text(json.dumps(config))
        with self.assertRaisesRegex(cache.PreparedCacheError, "Unsafe prepared-image path"):
            self.call()
        self.assertFalse((self.root / "escape.png").exists())

    def test_interrupted_preparation_never_publishes_a_complete_cache(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.call(prepare_fn=mock.Mock(side_effect=RuntimeError("interrupted")))
        self.assertEqual(list((self.root / "drive-cache").glob("*/cache_config.json")), [])
        _, rebuilt = self.call()
        self.assertFalse(rebuilt["cache_hit"])

    def test_prototype_selection_is_deterministic_and_excludes_unreviewed_manual_and_test(self):
        rows = []
        for split, count in [("train", 120), ("val", 40), ("test", 10)]:
            for index in range(count):
                rows.append(dict(archive_member=f"{split}/{index}.png", image_path="", split=split,
                                 character="ا", target_idx=0, source_type="handwritten",
                                 source_group=f"{split}-form-{index}", is_prepared=False))
            rows.append(dict(archive_member="", image_path=f"{split}-font.png", split=split,
                             character="آ", target_idx=32, source_type="printed",
                             source_group=f"{split}-font", is_prepared=True))
            rows.append(dict(archive_member="", image_path=f"{split}-manual.png", split=split,
                             character="آ", target_idx=32, source_type="handwritten",
                             source_group=f"aa-writer:{split}", is_prepared=False))
        frame = pd.DataFrame(rows)
        selected = cache.select_prototype_manifest(frame)
        repeated = cache.select_prototype_manifest(frame)
        pd.testing.assert_frame_equal(selected, repeated)
        self.assertEqual(selected.groupby("split").size().to_dict(), {"train": 101, "val": 26})
        self.assertFalse(((selected["character"] == "آ") & (selected["source_type"] == "handwritten")).any())
        reviewed = cache.select_prototype_manifest(frame, label_review_approved=True)
        self.assertEqual(len(reviewed), len(selected) + 2)
        self.assertEqual(set(reviewed["split"]), {"train", "val"})


if __name__ == "__main__":
    unittest.main()
