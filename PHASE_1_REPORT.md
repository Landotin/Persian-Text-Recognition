# Phase 1 — Dataset audit

Completed on 7 October 2026. The technical audit is complete; independent Persian annotation review and separately labeled handwritten آ remain requirements for the final complete-handwriting study. No model has been trained.

**Planning update, 7 October 2026:** PROJECT_PLAN.md now permits a provisional 33-label prototype using the archive's 32 handwritten classes and controlled printed data for all 33 labels. Manual handwriting and independent review may follow that engineering experiment. Handwritten آ performance must remain explicitly unvalidated. The audit findings below are unchanged; missing handwriting as a training prerequisite applies to the final complete-handwriting study, not this newly authorized prototype route.

## Dataset and scope

The received `04_final.zip` contains 122,494 readable grayscale JPEG images, all 56×56 pixels. Every image was decoded and its ZIP CRC checked; no decode failures or repeated full archive-member names were found. There are 23,829 digit images and 98,665 letter images.

The source supports 32 base-letter classes in folders 10–41. The repository's numeric training-folder order and prediction character list establish this mapping:

| Folder | Letter | Folder | Letter | Folder | Letter | Folder | Letter |
|---:|:---:|---:|:---:|---:|:---:|---:|:---:|
| 10 | ا | 18 | خ | 26 | ص | 34 | ک |
| 11 | ب | 19 | د | 27 | ض | 35 | گ |
| 12 | پ | 20 | ذ | 28 | ط | 36 | ل |
| 13 | ت | 21 | ر | 29 | ظ | 37 | م |
| 14 | ث | 22 | ز | 30 | ع | 38 | ن |
| 15 | ج | 23 | ژ | 31 | غ | 39 | و |
| 16 | چ | 24 | س | 32 | ف | 40 | ه |
| 17 | ح | 25 | ش | 33 | ق | 41 | ی |

The README's 10–42 range is inconsistent with the actual archive and config. No separately declared آ class exists. This does not prove that no آ-shaped image occurs within the Alif folder. The project keeps its proposed 33 recognition labels: the 32 source letters retain target indices 0–31, and آ is appended at index 32. Separate correctly labeled handwritten آ samples are required; ambiguous Alif samples must be reviewed rather than automatically relabeled. [Training order](https://raw.githubusercontent.com/sadafnazari/Persian_handwriting_recognition/master/src/train_models.py), [prediction labels](https://raw.githubusercontent.com/sadafnazari/Persian_handwriting_recognition/master/src/prediction.py), [configuration](https://raw.githubusercontent.com/sadafnazari/Persian_handwriting_recognition/master/config/config.yaml).

## Duplicate images and split overlap

All 348 source forms (173 type A, 175 type B) occur in every original split. Filenames preserve source-form basenames and column numbers, but the archive supplies no verified writer identity table. Source code splits individual filenames within each class, rather than grouping forms. Its current configured ratios also must not be used to describe the received archive; the measured counts below are the evidence. [Extraction and splitting code](https://raw.githubusercontent.com/sadafnazari/Persian_handwriting_recognition/master/src/data_preprocessing.py).

The audit found 23,429 repeated copies among the letter images, identified by identical decoded grayscale pixels and dimensions. There are 21,956 letter duplicate groups; 20,575 span multiple original splits. Retaining one deterministic representative per image leaves 75,236 unique letter images. No uniform-image or conflicting-label pixel groups needed exclusion. Digits are excluded from the recognition manifest.

A draft partition now keeps every source-form basename in one split, using seed 42 and approximately 70/15/15 of source groups. It contains all 32 available letters in every split, with zero shared source groups or identical pixel images between splits. Per-class totals range from 2,338 to 2,366 unique images.

| Split | Original images, including digits | Original letter images | New unique letter images | New source groups |
|---|---:|---:|---:|---:|
| Training | 94,565 | 76,172 | 52,633 | 244 |
| Validation | 18,449 | 14,863 | 11,209 | 52 |
| Test | 9,480 | 7,630 | 11,394 | 52 |

The new manifest replaces the original partitions and is frozen before any model fitting. Source-form separation is established; separation of writers and all near-duplicate variants is not established. A test from verified new writers remains necessary for a new-writer claim. If annotation review removes or relabels images, or new data is added, regenerate and version the partition before training.

## Image quality and printed data

A visual spot check used four original training images from each of the 32 letter folders, not final test predictions. Most archive images have light strokes on dark backgrounds (122,225 have dark average borders). Examples show table-border remnants and clipped strokes or dots. The source images are already small and processed; resizing to 224×224 will not restore missing details.

Notebook preprocessing must handle both background polarities, preserve detached dots and the maddah, and show previews. Review the training preview with a Persian reader before approving labels and sample quality. A spot check cannot certify every annotation.

The proposed secondary dataset is a promising optional printed-data candidate: its demo visibly shows typeset glyphs, and `list_fonts.txt` lists 379 font/style names. Its 91 letter-form labels collapse to 33 distinct base characters after removing the joining extension (tatweel), retaining آ separately from ا. However, the full ZIP has not been inspected, and the files do not establish which font family produced each image. It can support neither a font-held-out evaluation claim nor separate handwritten آ coverage without additional evidence. [Candidate repository](https://github.com/msbsoft2/Persian-written-Characters-and-Digits-Dataset), [labels](https://github.com/msbsoft2/Persian-written-Characters-and-Digits-Dataset/blob/main/labels.json), [font list](https://github.com/msbsoft2/Persian-written-Characters-and-Digits-Dataset/blob/main/list_fonts.txt), [printed demo](https://github.com/msbsoft2/Persian-written-Characters-and-Digits-Dataset/blob/main/Demo.jpg).

Phase 2 will use the controlled printed set with recorded font family, size, style, and letter as the primary printed source; font families will be held apart across splits. Keep the proposal's 12-family target subject to actual available compatible fonts. If the candidate is selected as an additional source, download only its `dataset.zip` (77,656,537 bytes) and audit per-image provenance before mixing it into the font partition. No additional download is needed to close this phase or start notebook development.

## Roman-letter outputs

`config/characters.json` saves the 33-class order, source-folder mapping, names, simplified display values, and context notes. Matching Roman outputs do not merge Persian classes. The display uses `aa`, `ch`, `kh`, `zh`, `sh`, and `gh`; ا, و, ی, ه, and ع carry context notes. It is a simplified isolated-letter display referenced to the [Library of Congress Persian romanization table](https://www.loc.gov/catdir/cpso/romanization/persian.pdf), rather than a complete word romanization system.

## Saved checkpoint

- `config/characters.json`: target labels, source mapping, and Roman display metadata.
- `artifacts/phase1/audit.json`: archive fingerprint, integrity checks, counts, and duplicate/split findings.
- `artifacts/phase1/class_counts.csv`: original counts for every folder.
- `artifacts/phase1/grouped_class_counts.csv`: counts for the new unique-letter partition.
- `artifacts/phase1/letter_manifest.csv`: ZIP member paths, source groups, pixel hashes, and assigned splits; extraction is unnecessary for this manifest.
- `artifacts/phase1/excluded_letter_images.csv`: repeated copies and exclusion reasons.
- `artifacts/phase1/handwriting_samples.png`: training-only review sheet.
- `scripts/audit_handwriting.py`: reproducible audit and partition generator.

Next under the revised plan: adapt the prepared Colab notebook for a short prototype using archive handwriting plus printed coverage of all 33 outputs, then show real predictions. Add reviewed manual handwritten آ in a later data version before claiming complete handwriting coverage. The notebook must not present an output lacking handwritten evidence as handwritten-validated.
