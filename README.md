# Persian Character Recognition — V1

An inference demo and research prototype for recognizing isolated Persian characters. The model is a seed-42 DenseNet121 with the project's fixed 33-label order. It uses archive handwriting for 32 labels and controlled printed glyphs for all 33.

This is a prototype. Handwritten **آ** has no reviewed training or test examples, and independent Persian-reader review is pending. Roman spellings are simplified displays for isolated characters, not full-word transliteration.

## Model and evaluation

The verified model bundle is stored in the project's private Drive folder and is excluded from Git. For remote use, attach it as a private GitHub Release asset named `prototype_model_bundle.zip`. The bundle includes the trained weights, ordered labels, preprocessing implementation and settings, run configuration, validation outputs, and SHA-256 manifest. The raw archive, source images, fonts, and individual test predictions are not included in this repository.

The checkpoint was selected using validation data. After it was fixed, the user requested one post-hoc evaluation on the frozen test split. The model was not retrained or tuned from those results. `final_test_used=false` in the run metadata means test data was not used for training or checkpoint selection; it does not mean the later diagnostic was never run.

| Source | Split | Samples | Supported classes | Macro F1 | Accuracy | Test minus validation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Archive handwriting | Validation | 800 | 32 | 80.78% | 81.13% | — |
| Archive handwriting | Test | 11,394 | 32 | 82.24% | 82.38% | +1.47 / +1.25 pp |
| Printed glyphs | Validation | 792 | 33 | 96.82% | 96.84% | — |
| Printed glyphs | Test | 198 | 33 | 84.86% | 85.35% | −11.96 / −11.49 pp |

The handwriting test score is close to validation, but its split is by archive source form and writer identities are unknown. The test has no handwritten **آ** labels; two handwriting images were predicted as **آ** and count as errors. The printed test has six images per class from one held-out Noto Kufi Arabic design group. Its lower score shows a font-group generalization gap and should not be generalized to every printed font. Training loss fell and validation scores rose during the short five-epoch run, with no clear overfitting signal in those logs. These results do not establish independent-writer performance or settle underfitting across all sources.

See [the detailed evaluation report](reports/test_evaluation.md) for the protocol, source coverage, and common confusions. This test is now a one-time prototype diagnostic. Do not use it to tune the current model; if it informs a redesign, use a fresh independent holdout for future final claims.

## Run the Streamlit demo

Use Python 3.10 or newer, install the dependencies, download the private release asset, and point the app to the bundle:

```bash
python -m pip install -r requirements-streamlit.txt
PERSIAN_MODEL_BUNDLE=/path/to/prototype_model_bundle.zip streamlit run streamlit_app.py
```

The app checks the bundle hashes, 33-label ordering, DenseNet121 shape, and preprocessing source before loading the weights. It predicts one uploaded isolated character and displays the top three outputs.

## Repository contents

- `streamlit_app.py`, `app_preprocessing.py` — inference-only demo and the matching image preparation code.
- `notebooks/01_data_preparation_training.ipynb` — Colab preparation and training workflow. Its saved settings may enable prototype training; do not use Run All to evaluate a saved model.
- `config/characters.json` — fixed output-label order and display metadata.
- `scripts/` and `tests/` — data-audit and persistent prepared-image cache helpers.
- `requirements-streamlit.txt` — demo dependencies.
- `reports/` — aggregate evaluation report; no per-image predictions or raw test files.

The source dataset, fonts, handwritten collection, and generated training artifacts are excluded by `.gitignore`. Keep those materials in the project’s private Drive folder.

## Further study

Before making a complete handwriting claim, collect and review labeled examples of handwritten **آ**, obtain independent Persian-reader review, and use verified writer-disjoint partitions. The planned full study uses three seeds and reports the handwriting and printed sources separately. If this prototype test split informs a redesign, reserve a fresh independent holdout for the final evaluation.

## Project status

See [PROGRESS.md](PROGRESS.md) for the current checkpoint and [PROJECT_PLAN.md](PROJECT_PLAN.md) for the planned study. Historical audit evidence is in [PHASE_1_REPORT.md](PHASE_1_REPORT.md).
