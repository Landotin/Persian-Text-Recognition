# Local Streamlit prototype

This app runs inference with the saved seed-42 DenseNet121 bundle. It does not train a model or load test images. The fixed checkpoint has a separate one-time test diagnostic; see [reports/test_evaluation.md](reports/test_evaluation.md).

1. Download `prototype_model_bundle.zip` from the Phase 3 run folder in Drive, or upload the ZIP in the app.
2. Install the app dependencies with `python -m pip install -r requirements-streamlit.txt`.
3. Start the app with `streamlit run streamlit_app.py`.

The default bundle path is `artifacts/phase3/prototype_seed_42_20261009T054030882857Z/prototype_model_bundle.zip`. Set `PERSIAN_MODEL_BUNDLE` to a different local ZIP path if needed. The app checks the bundle hashes, label order, architecture, and preprocessing source before loading weights.

The app is an inference demo. Handwritten `آ` remains unvalidated and Persian-reader review is pending. Metrics shown in the app are validation results; the separate one-time test diagnostic is limited by unknown handwriting writers and one held-out printed font group. Roman displays apply to isolated letters, not complete-word transliteration.
