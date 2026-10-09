# Persian Character Recognition Project Plan

Updated 9 October 2026. Phases 1 and 2 are complete. The seed-42 DenseNet121 prototype has completed, its bundle is verified, and a one-time post-hoc diagnostic was run on the frozen test split. The test results were not used to tune or retrain the checkpoint.

## Decision: prototype first, final study later

Keep DenseNet121 and the existing 33-label order. Train the prototype from the archive's 32 handwritten classes plus controlled printed examples of all 33 labels, including آ. Additional manual handwriting does not block this experiment. Handwritten آ performance remains unvalidated, and independent Persian-reader review remains pending.

Preserving 33 outputs lets later data improve the same classifier without changing indices or the Roman-letter lookup. A printed-only source for آ supplies training examples; it does not demonstrate handwriting recognition for that class.

Use the Python notebook in Colab for preparation/training/export, and a separate Streamlit app for saved-model inference. Save durable artifacts to Drive; train from local runtime copies where practical. Colab resources and runtime availability vary. ([Colab FAQ](https://research.google.com/colaboratory/faq.html))

## Available inputs

| Input | Current evidence | Use now |
|---|---|---|
| Handwriting archive | 75,236 unique images, 32 labeled letters | Prototype training/validation |
| Source-form manifest | 52,633 train / 11,209 validation / 11,394 test; no exact-pixel or source-group overlap | Preserve assignments; reserve final tests |
| Uploaded fonts | 12 regular font files across ten conservative design groups; all 33 labels rendered in each | Prototype training/validation, with reader review still pending |
| Manual handwriting | Not yet available | Add in a later data version |
| Independent annotation review | Pending | Record prototype labels as provisional |

Font files/folders are not automatically independent designs. Conservatively keep Noto Sans Arabic with its UI variant, and Noto Naskh Arabic with its UI variant, in the same partition. This gives ten current design groups. Use those for the prototype; twelve independent families and the proposal's 8/2/2 split remain a final-study target. Record actual assignments and inspect all 33 glyphs in every selected font.

The archive has no separate declared آ class. Do not automatically relabel ا images. Source-form grouping does not establish distinct writers. See PHASE_1_REPORT.md for audit evidence.

## Phases

| Phase | Task | Completion evidence |
|---|---|---|
| 1 — Audit | Archive, labels, duplicates, source-form split | Technical audit complete; human review pending |
| 2 — Prototype preparation | Add prototype modes; fix preprocessing; group fonts; generate and inspect samples | Complete: 2,376 printed samples; 33 printed labels and 32 archive handwriting labels in train/validation; versioned manifests and font checks |
| 3 — First Colab training | Train and export the seed-42 prototype | Verified checkpoint, bundle, and source-specific validation outputs; one-time post-hoc test diagnostic completed |
| 4 — Streamlit prototype | Verify upload/result flow around saved bundle | Local app loads the verified bundle and returns the expected class for a saved validation glyph |
| 5 — Manual data and final study | Review new handwriting; version data; complete full-data experiments | Fixed method, three seed runs, source-specific final evaluation |
| 6 — Deployment and report | Measure CPU inference; deploy if needed; document findings | Working demonstration and accurate coverage/limitations |

The first prototype run has completed and produced validation predictions. Manual collection, three final seed runs, test-time augmentation, and deployment remain later work. The frozen test split has since been used once for a post-hoc prototype diagnostic; it must not be used for tuning. If those results inform redesign, obtain a fresh independent holdout for future final claims.

## Phase 2 — Prototype readiness

Reuse the existing notebook, mapping, and audited manifest. The notebook uses the Drive path `MyDrive/Persian Character Recognition`, preserves archive/source-form assignments, and freezes related-font grouping before generation. Data preparation is a persistent snapshot, independent of timestamped model runs.

Separate preparation, prototype-training, and final-experiment modes:

| Check | Prototype | Final complete-handwriting study |
|---|---|---|
| Readable files, coherent label order, split integrity, usable preprocessing | Required | Required |
| All 33 labels represented in train/validation from an available source | Required | Required in relevant evaluation partitions too |
| Reviewed handwritten آ | Optional; explicitly unvalidated | Required across writer-separated partitions with assessed sample diversity |
| Independent Persian-reader approval | May remain pending; record provisional status | Required |
| Three seeds / full training data | Deferred | Required by the proposal |
| Twelve independent printed families | Use current usable grouped fonts | Verify target or document a justified scope amendment |

The notebook has explicit `prepare`, `prototype`, and `final` modes. Prototype readiness does not depend on reader approval or handwritten آ. Human-review flags remain false until review occurs.

Apply deterministic image preparation once: orientation/transparency, background polarity, complete glyph/detached marks, and aspect-preserving resize/padding to grayscale 224×224. Save these pixels; repeat RGB tensor conversion, random training augmentation, and ImageNet normalization per batch. Printed glyphs are framed from their shaped glyph bounds with a controlled white margin before that shared pass; archive images are not foreground-cropped. Initially accept isolated character crops on plain backgrounds. Whole handwritten sheets require segmentation; broader phone-photo support requires foreground cropping/crop adjustment. All 12 generated font sheets were visually inspected; machine checks found no blank glyphs and confirmed ا differs from آ in each font. Independent Persian-reader review remains pending.

### Persistent prepared dataset

Default `DATA_ACTION='reuse'` loads an immutable, hash-verified preparation snapshot without repeating the archive audit or printed generation. Its first use builds only the selected prototype train/validation images; subsequent runs restore a complete cache. `prepare_new` explicitly creates a new preparation version and uses the frozen source-form manifest rather than repartitioning with the model seed.

Save `artifacts/prepared/<cache_id>/prepared_images.zip`, a relative-path prepared manifest, validation input thumbnails, exact preprocessing source/settings, font splits, source configuration, and a completion marker with hashes. Build locally and publish metadata last. Restore the ZIP into runtime storage so training avoids individual Drive reads. Cache identity binds source manifests, ordered labels, fixed partitions, preprocessing, and selected samples. New source images or edited inputs require a new preparation snapshot; incomplete caches rebuild and corrupt completed caches fail clearly.

The training dataset reads local cached PNGs while retaining original sample IDs for provenance. Cached thumbnails support notebook input/prepared previews without reopening the raw archive. Only the explicitly enabled reviewed final-study branch can cache or load final-test images. Keep each run's metadata consistent and require complete bundle files at export.

## Phase 3 — Small training run and real output

- Start with seed 42 and deterministic subsets selected within the frozen partitions. Initial budget: up to 100 archive training samples and 25 archive validation samples per class, plus usable printed train/validation samples. Save chosen sample IDs.
- Start with two classifier-only epochs and up to three epochs fine-tuning the final dense block, batch size 16 if memory permits. This is a prototype budget, not an accuracy/runtime guarantee; expand only when validation errors justify it.
- Represent printed and handwritten sources in training. Inspect per-class/source counts and record sampling weights. Later small manual classes must not dominate solely through excessive repetition.
- Select the checkpoint on validation. Track printed macro F1 over its 33 represented classes and archive-handwritten macro F1 over its 32 represented classes; use an equal mean as the planned selection score. Incorrect predictions into آ still count as errors. Its absent handwritten support is unavailable, not an artificial zero.
- Keep final test partitions unused during tuning and demonstration. The user's one-time post-hoc diagnostic was run after the seed-42 checkpoint was fixed; do not tune that checkpoint from those results. Show a fixed validation demonstration set, including ش → sh and printed آ → aa; identify these as development examples.
- Export a versioned bundle: checkpoint, ordered labels, Roman lookup, exact preprocessing implementation/settings, data/manifest hashes, font groups, seed, training configuration, and source/class limitations.

The preview shows input, prepared image, predicted Persian character, Roman-letter display, and top-1 score. The score is a model output rather than a measured probability of correctness. Prototype diagnostic metrics do not replace the Phase 5 multi-seed, writer-disjoint final study.

## Phase 4 — Streamlit prototype

Keep four responsibilities: upload/results, shared preparation, model prediction, and Roman-letter lookup. Load the bundle once with st.cache_resource; the app performs inference only. ([Streamlit caching documentation](https://docs.streamlit.io/develop/concepts/architecture/caching))

Use real predictions, show the prototype coverage note, verify notebook/app agreement, and handle unreadable/empty images. Defer accounts, databases, full-page OCR, extra architectures, elaborate styling, and test-time augmentation.

## Phase 5 — Add handwriting and complete experiments

1. Assign anonymous writer IDs and partitions before using samples. Every character and collection batch from one person stays in one partition. Preserve existing archive/font assignments; any necessary repartitioning creates a new version.
2. Separate collection goals: handwritten آ fills missing source coverage; all 33 labels from genuinely held-out writers support whole-alphabet new-writer evaluation. Writers used for training cannot also provide the independent test set.
3. Three writers populate train/validation/test mechanically, but one writer per partition with 10–20 samples is a small pilot. Assess usable counts/diversity instead of treating that minimum as proof of adequacy.
4. Have a Persian reader review mappings, representative archive quality, all font glyphs, manual labels, and ambiguous ا/آ examples. Version corrections and keep duplicates/variants together.
5. Integrate reviewed samples into a new dataset/run version while retaining the 33-label order. Fine-tune the prototype for an early update if useful. Final three-seed runs should use the same declared initialization, preferably the proposal's ImageNet weights, rather than inherit different prototype histories.
6. Fix preprocessing, sampler, validation-selection rules, and any test-time augmentation before testing. Run full-data seeds 7, 21, and 42; report mean and spread. Choose models/settings by validation, never final test results.
7. Report printed, archive handwriting, and independent manual handwriting separately: class support, accuracy, macro F1, confusion matrix, and inference time. A manual test containing only آ cannot establish whole-alphabet new-writer accuracy. If test results drive later redesign, retain that record and obtain a fresh independent holdout for new final claims.

## Phase 6 — Deploy and report

Measure model loading/CPU inference and check the intended host's current limits before deployment. Keep datasets/writer identifiers in private Drive; deploy the inference app separately when sharing is needed. A working local demonstration is the first delivery.

Distinguish prototype validation demonstrations from final evidence, and report actual fonts, samples, annotation review, and writer coverage. Limited new-writer data restricts that claim without preventing a useful prototype.

## Pipeline

```text
Audited handwriting (32 classes) + controlled printed data (33 classes)
  → versioned manifest with fixed source-form/font-group partitions
  → shared preparation once + training-only augmentation
  → mixed-source DenseNet121 training (33 outputs)
  → validation-selected checkpoint + versioned model bundle
  → notebook prediction preview → Streamlit upload/results

Later: reviewed manual crops + whole-writer assignments
  → new data version → same training/export pipeline → final evaluation
```

## Phase 2 implementation completed

The updated notebook separates preparation, prototype, and final modes; retains the stricter final-study gates; uses one shared preparation pass; and saves each run in a new timestamped directory. Related font variants share partitions. Every supplied font/style gets a 33-label preview and render check.

The locally executed Phase 2 bundle is `artifacts/phase2/prototype_prepare_20261007T153407357883Z/`. Its Drive archive is [phase2-prototype-preparation-20261007.zip](https://drive.google.com/file/d/13JVaCeol8VONn-CwqOcPQYft0GlgD_PX/view?usp=drivesdk). It contains audit evidence, manifests, generated samples, font sheets/checks, and preprocessing/run metadata. The run confirmed 33 printed labels plus 32 archive handwriting labels in train/validation. It did not train a model or access final test images for evaluation.

## Phase 3 checkpoint and current status

The 9 October seed-42 DenseNet121 run used the persistent prepared-image cache and completed on CPU. Its fixed bundle contains the checkpoint, exact preprocessing and run metadata, font groups, validation predictions, and a complete hash manifest. The final validation results were archive handwriting (800 samples, 32 classes: macro F1 80.78%, accuracy 81.13%) and printed (792 samples, 33 classes: macro F1 96.82%, accuracy 96.84%). Handwritten `آ` remains unsupported and reader review remains pending.

At the user's request, the fixed checkpoint received one post-hoc diagnostic on 9 October using the frozen test assignments. The raw archive, full manifest, printed test images, and bundle hashes were verified before inference. Results were archive handwriting (11,394 samples, 32 classes: macro F1 82.24%, accuracy 82.38%) and printed (198 samples, 33 classes: macro F1 84.86%, accuracy 85.35%). This is a prototype diagnostic, not the Phase 5 final study. The printed result is based on one held-out Noto Kufi Arabic design group; writer identities are unavailable for the archive handwriting split. The test split is consumed for this diagnostic and must not guide tuning of this checkpoint. If results inform redesign, reserve a fresh independent holdout.

The detailed aggregate-only protocol and limitations are in `reports/test_evaluation.md`. The Streamlit app was launched locally in an isolated CPU environment with Streamlit 1.65.0, PyTorch 2.11.0+cpu, and TorchVision 0.26.0+cpu. Its bundle checks passed and its inference path predicted a saved validation printed glyph (expected ا) as ا at 83.35%, matching the app output. This is a one-example smoke check, not a performance estimate. Next, publish the source, README, aggregate report, and exact model bundle to the user's private GitHub repository. Keep raw images and per-image test predictions out of that repository.

For future turns, read PROGRESS.md first and only the relevant plan section. Update that checkpoint after each milestone; keep historical audit evidence in PHASE_1_REPORT.md. Avoid rereading the full conversation or large manifests when the checkpoint is sufficient.

For future turns, read PROGRESS.md first and only the relevant plan section. Update that checkpoint after each milestone; keep historical audit evidence in PHASE_1_REPORT.md. Avoid rereading the entire conversation or large manifests when the checkpoint is sufficient.
