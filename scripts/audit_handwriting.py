"""Audit the source ZIP and create reproducible, source-grouped letter manifests.

Reads the original archive without extracting or changing it. No model training.
"""

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont

LETTERS = list("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
NAMES = ["Alif", "Be", "Pe", "Te", "Se", "Jim", "Che", "He (ح)", "Khe",
         "Dal", "Zal", "Re", "Ze", "Zhe", "Sin", "Shin", "Sad", "Zad",
         "Ta", "Za", "Ayn", "Ghayn", "Fe", "Qaf", "Kaf", "Gaf", "Lam",
         "Mim", "Nun", "Vav", "He (ه)", "Ye"]
SPLITS = ("train", "val", "test")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/phase1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import zipfile

    counts = Counter()
    sizes, modes, backgrounds = Counter(), Counter(), Counter()
    prefixes = defaultdict(set)
    source_forms = defaultdict(set)
    records, failures, unusual = [], [], []
    sample_paths = defaultdict(list)
    digest_records = defaultdict(list)
    with zipfile.ZipFile(args.archive) as archive:
        infos = archive.infolist()
        repeated_names = [name for name, count in Counter(x.filename for x in infos).items() if count > 1]
        for entry in infos:
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if not (len(parts) == 4 and parts[0] == "04_final" and parts[1] in SPLITS
                    and parts[2].isdigit() and parts[3].lower().endswith((".jpg", ".jpeg", ".png"))):
                unusual.append(entry.filename)
                continue
            old_split, folder = parts[1], int(parts[2])
            counts[(old_split, folder)] += 1
            # Source code appends a column number to the original form basename.
            stem, column = Path(parts[3]).stem.rsplit("_", 1)
            form_type = "a" if folder in (0, 1, 2, 3) or 10 <= folder <= 26 else "b"
            prefixes[stem].add(old_split)
            source_forms[f"{form_type}:{stem}"].add(old_split)
            try:
                data = archive.read(entry)  # zipfile also checks the CRC while reading.
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    modes[image.mode] += 1
                    gray = image.convert("L")
                    width, height = gray.size
                    pixels = gray.tobytes()
                    lo, hi = gray.getextrema()
                    sizes[f"{width}x{height}"] += 1
                    border = pixels[:width] + pixels[-width:] + pixels[::width] + pixels[width - 1::width]
                    border_mean = sum(border) / len(border)
                    backgrounds["light" if border_mean >= 127.5 else "dark"] += 1
                    digest = hashlib.sha256(f"{width},{height}:L:".encode() + pixels).hexdigest()
            except Exception as error:
                failures.append({"archive_member": entry.filename, "error": str(error)})
                continue
            record = {"archive_member": entry.filename, "original_split": old_split,
                      "source_folder": folder, "character": LETTERS[folder - 10] if 10 <= folder <= 41 else "",
                      "source_form": f"{form_type}:{stem}", "source_group": stem,
                      "writer_id": "", "column": column, "width": width, "height": height,
                      "pixel_sha256": digest, "uniform": lo == hi, "border_mean": round(border_mean, 2)}
            records.append(record)
            digest_records[digest].append(record)
            if old_split == "train" and folder >= 10:
                sample_paths[folder].append(entry.filename)

        # Training samples only: a visual spot check is distinct from native-speaker review.
        canvas = Image.new("RGB", (1440, 8 * 160), "#f1f1f1")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        except OSError:
            font = ImageFont.load_default()
        for index, folder in enumerate(range(10, 42)):
            x, y = (index % 4) * 360, (index // 4) * 160
            draw.text((x + 8, y + 5), f"Folder {folder}: {NAMES[index]}", fill="black", font=font)
            paths = sorted(sample_paths[folder])
            chosen = random.Random(args.seed + folder).sample(paths, min(4, len(paths)))
            for n, path in enumerate(chosen):
                with Image.open(io.BytesIO(archive.read(path))) as image:
                    # Upscale tiny source images solely for the audit preview.
                    preview = image.convert("RGB").resize((82, 82), Image.Resampling.NEAREST)
                    canvas.paste(preview, (x + 8 + n * 88, y + 40))
        canvas.save(args.out / "handwriting_samples.png")

    removed, retained = [], []
    for group in digest_records.values():
        letters = [r for r in group if 10 <= r["source_folder"] <= 41]
        if not letters:
            continue
        labels = {r["source_folder"] for r in group}
        if letters[0]["uniform"]:
            removed.extend({**r, "reason": "uniform_image"} for r in letters)
        elif len(labels) > 1:
            removed.extend({**r, "reason": "identical_pixels_conflicting_labels"} for r in letters)
        else:
            ordered = sorted(letters, key=lambda r: r["archive_member"])
            retained.append(ordered[0])
            removed.extend({**r, "reason": "identical_pixels_same_label"} for r in ordered[1:])

    # Conservative grouping: identical form basenames across a/b stay together.
    # It may over-group unrelated forms; it does not establish writer identity.
    group_ids = sorted({r["source_group"] for r in retained})
    random.Random(args.seed).shuffle(group_ids)
    n_train, n_val = round(len(group_ids) * .70), round(len(group_ids) * .15)
    assigned = {g: "train" if i < n_train else "val" if i < n_train + n_val else "test"
                for i, g in enumerate(group_ids)}
    new_counts = Counter()
    for record in retained:
        record["split"] = assigned[record["source_group"]]
        new_counts[(record["split"], record["source_folder"])] += 1
    fields = ["archive_member", "original_split", "source_folder", "character", "source_form",
              "source_group", "writer_id", "column", "width", "height", "pixel_sha256", "uniform", "border_mean"]
    write_csv(args.out / "letter_manifest.csv", ["split"] + fields, sorted(retained, key=lambda r: r["archive_member"]))
    write_csv(args.out / "excluded_letter_images.csv", fields + ["reason"], sorted(removed, key=lambda r: r["archive_member"]))
    write_csv(args.out / "class_counts.csv", ["source_folder", "character", "train", "val", "test", "total"],
              [{"source_folder": c, "character": LETTERS[c - 10] if c >= 10 else "digit",
                **{s: counts[(s, c)] for s in SPLITS}, "total": sum(counts[(s, c)] for s in SPLITS)} for c in range(42)])
    write_csv(args.out / "grouped_class_counts.csv", ["source_folder", "character", "train", "val", "test", "total"],
              [{"source_folder": c, "character": LETTERS[c - 10],
                **{s: new_counts[(s, c)] for s in SPLITS}, "total": sum(new_counts[(s, c)] for s in SPLITS)} for c in range(10, 42)])

    duplicate_groups = [g for g in digest_records.values() if len(g) > 1]
    cross_split_duplicates = [g for g in duplicate_groups if len({r["original_split"] for r in g}) > 1]
    letter_duplicate_groups = [[r for r in g if r["source_folder"] >= 10] for g in duplicate_groups]
    letter_duplicate_groups = [g for g in letter_duplicate_groups if len(g) > 1]
    cell_records = defaultdict(list)
    for record in records:
        cell_records[(record["source_folder"], PurePosixPath(record["archive_member"]).name)].append(record)
    new_group_splits = defaultdict(set)
    new_digest_splits = defaultdict(set)
    for record in retained:
        new_group_splits[record["source_group"]].add(record["split"])
        new_digest_splits[record["pixel_sha256"]].add(record["split"])
    with args.archive.open("rb") as handle:
        archive_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"archive": str(args.archive.resolve()), "archive_sha256": archive_digest,
              "image_count": sum(counts.values()), "decoded_count": len(records), "decode_failures": failures,
              "repeated_archive_member_names": repeated_names, "non_image_files": unusual,
              "original_counts": {s: sum(v for (split, cls), v in counts.items() if split == s) for s in SPLITS},
              "digit_count": sum(v for (split, cls), v in counts.items() if cls < 10),
              "letter_count": sum(v for (split, cls), v in counts.items() if cls >= 10),
              "original_letter_counts": {s: sum(v for (split, cls), v in counts.items() if split == s and cls >= 10) for s in SPLITS},
              "source_folders": sorted({cls for split, cls in counts}), "image_sizes": dict(sizes), "image_modes": dict(modes),
              "background_polarity_by_border": dict(backgrounds),
              "form_basename_count": len(prefixes), "source_form_count_including_type": len(source_forms),
              "form_basenames_in_all_original_splits": sum(len(v) == 3 for v in prefixes.values()),
              "source_forms_in_all_original_splits": sum(len(v) == 3 for v in source_forms.values()),
              "source_form_counts_by_type": dict(Counter(k.split(":", 1)[0] for k in source_forms)),
              "source_cell_identifiers_repeated_across_original_splits": sum(len({r["original_split"] for r in g}) > 1 for g in cell_records.values()),
              "duplicate_pixel_groups": len(duplicate_groups), "duplicate_pixel_extra_images": sum(len(g)-1 for g in duplicate_groups),
              "cross_original_split_duplicate_groups": len(cross_split_duplicates),
              "letter_duplicate_pixel_groups": len(letter_duplicate_groups),
              "letter_cross_original_split_duplicate_groups": sum(len({r["original_split"] for r in g}) > 1 for g in letter_duplicate_groups),
              "letter_exclusion_reasons": dict(Counter(r["reason"] for r in removed)),
              "retained_letters": len(retained), "new_split_seed": args.seed,
              "new_split_group_key": "source filename basename; matching basenames would stay together across form types a/b; writer IDs unknown",
              "new_split_group_counts": dict(Counter(assigned.values())),
              "new_split_image_counts": {s: sum(v for (split, cls), v in new_counts.items() if split == s) for s in SPLITS},
              "new_split_letter_class_counts": {s: sum(new_counts[(s,c)] > 0 for c in range(10,42)) for s in SPLITS},
              "new_split_source_group_overlap": sum(len(v) > 1 for v in new_group_splits.values()),
              "new_split_identical_pixel_overlap": sum(len(v) > 1 for v in new_digest_splits.values()),
              "separate_aa_class": "not present in the source label scheme; visual presence within Alif remains unverified",
              "mapping_status": "verified from repository prediction list and numeric training-folder order; independent Persian-reader review pending"}
    assert sum(report["new_split_image_counts"].values()) == len(retained)
    assert all(n == 32 for n in report["new_split_letter_class_counts"].values())
    assert len({r["pixel_sha256"] for r in retained}) == len(retained)
    assert report["new_split_source_group_overlap"] == 0
    assert report["new_split_identical_pixel_overlap"] == 0
    (args.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("image_count", "digit_count", "letter_count", "image_sizes", "decode_failures",
          "source_form_count_including_type", "source_forms_in_all_original_splits", "duplicate_pixel_groups",
          "cross_original_split_duplicate_groups", "letter_exclusion_reasons", "retained_letters", "new_split_image_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
