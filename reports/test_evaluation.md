# Seed-42 prototype: one-time held-out test diagnostic

- Date: 9 October 2026
- Model: torchvision DenseNet121, seed 42, 33 outputs
- Checkpoint: `prototype_seed_42_20261009T054030882857Z`
- Device: Tesla T4, PyTorch 2.11.0+cu130, Torchvision 0.26.0+cu130

## Protocol

The checkpoint had already been selected using validation data. At the user's request, it was then evaluated once on all rows assigned to the frozen test split. The model was not trained or tuned during this evaluation. The bundle's 15 member hashes, 33-label order, architecture, and exact preprocessing implementation were verified before inference. Handwriting was prepared with the bundled preprocessing code; the already-prepared printed PNGs were loaded directly.

- Test rows: 11,592 total — 11,394 archive-handwriting and 198 printed.
- Archive SHA-256: `1faaff1234e743897652ab41e4bf3c3a2c2d87fdf1eda7886516f9bf031786e9`.
- Full training-manifest SHA-256: `cd8f0298575fc4e88eee224247c28e1f09bd87f375a05380a762d1b06792aaff`.
- Model bundle SHA-256: `857d8fcd4349a31d2bb2ba3190cc74f58b0fcc72da66310455ae82cf7b5985a0`.
- Inference batch size: 64. No test images or per-image predictions are included in this report.

## Results

| Source | Split | Samples | Supported classes | Macro F1 | Accuracy | Test minus validation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Archive handwriting | Validation | 800 | 32 | 80.78% | 81.13% | — |
| Archive handwriting | Test | 11,394 | 32 | 82.24% | 82.38% | +1.47 / +1.25 pp |
| Printed glyphs | Validation | 792 | 33 | 96.82% | 96.84% | — |
| Printed glyphs | Test | 198 | 33 | 84.86% | 85.35% | −11.96 / −11.49 pp |

Handwriting macro F1 averages over its 32 labels with ground-truth support. Test support per handwriting class ranges from 280 to 424 (median 420). The archive split is source-form-disjoint, but writer identities are unknown, so it does not establish generalization to new writers. There are no handwritten **آ** test examples. The model predicted **آ** for two handwriting images; both are errors in accuracy.

The printed test contains six images per label, all from the held-out Noto Kufi Arabic design group. The validation set contains 792 printed examples from its validation font groups. The lower test score is a substantial gap on this one unseen font design; 198 examples from one group are too narrow to represent all printed styles.

## Most frequent confusions

| Source | True character | Predicted character | Count |
| --- | --- | --- | ---: |
| Handwriting | چ | ج | 135 |
| Handwriting | ح | ع | 127 |
| Handwriting | ک | گ | 85 |
| Handwriting | س | ص | 79 |
| Handwriting | ث | ت | 65 |
| Printed | ر | ا | 5 |
| Printed | ز | ا | 3 |
| Printed | ک | ط | 3 |
| Printed | ی | ک | 3 |
| Printed | آ | ا | 2 |

The weakest handwriting classes by F1 include **ع** (0.680), **ح** (0.709), **ک** (0.710), and **چ** (0.716). Printed class support is only six each; its weakest F1 values were **ر** (0.286), **ک** (0.364), and **ا** (0.545).

## Interpretation and limits

- The handwriting test result is slightly above validation. Together with the training log (training loss fell and validation metrics rose through the five-epoch run), this provides no clear overfitting signal for this source-form split. It still cannot rule out writer leakage or establish new-writer performance.
- The printed test result is about 12 percentage points below validation. This indicates sensitivity to the held-out font group or a broader font-domain shift. It does not, by itself, show whether the remedy is more data, different augmentation, or a different training method.
- The results do not support a single overall “overfit” or “underfit” label: handwriting and printed behavior differ. The checkpoint remains unchanged as V1, with a clear printed-font limitation.
- This is a one-time prototype diagnostic, not the planned multi-seed final study or an independent-writer evaluation. Do not tune this checkpoint using these scores. If these findings guide a redesign, reserve a fresh independent holdout for later final claims.
- Handwritten **آ** remains unsupported; Persian-reader review is pending. Printed **آ** results do not validate handwriting recognition for that character.
