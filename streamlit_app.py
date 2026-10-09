"""Local inference-only demo for the Persian character prototype."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models

from app_preprocessing import prepare_image


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "artifacts/phase3/prototype_seed_42_20261009T054030882857Z/prototype_model_bundle.zip"
REQUIRED_MEMBERS = {
    "bundle_manifest.json",
    "labels.json",
    "model_spec.json",
    "model_state_dict.pth",
    "preprocessing.json",
    "preprocessing_source.py",
    "prototype_validation.json",
    "source_validation_metrics.json",
}
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class BundleError(ValueError):
    """Raised when the selected model bundle is incomplete or incompatible."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_json(data: bytes, filename: str) -> dict:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"{filename} is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{filename} must contain a JSON object.")
    return value


def _read_bundle(bundle_bytes: bytes) -> dict:
    try:
        archive = zipfile.ZipFile(io.BytesIO(bundle_bytes))
    except (zipfile.BadZipFile, OSError) as exc:
        raise BundleError("The selected file is not a readable ZIP bundle.") from exc

    with archive:
        members = {Path(info.filename).name: info for info in archive.infolist() if not info.is_dir()}
        missing = sorted(REQUIRED_MEMBERS - set(members))
        if missing:
            raise BundleError(f"The bundle is missing required files: {', '.join(missing)}")

        contents = {name: archive.read(info) for name, info in members.items()}
        manifest = _parse_json(contents["bundle_manifest.json"], "bundle_manifest.json")
        expected_hashes = manifest.get("files_sha256")
        if not isinstance(expected_hashes, dict):
            raise BundleError("bundle_manifest.json has no files_sha256 object.")
        for filename, expected in expected_hashes.items():
            if filename not in contents or _sha256(contents[filename]) != expected:
                raise BundleError(f"Bundle integrity check failed for {filename}.")

        labels = _parse_json(contents["labels.json"], "labels.json")
        model_spec = _parse_json(contents["model_spec.json"], "model_spec.json")
        preprocessing = _parse_json(contents["preprocessing.json"], "preprocessing.json")
        metrics = _parse_json(contents["source_validation_metrics.json"], "source_validation_metrics.json")
        validation = _parse_json(contents["prototype_validation.json"], "prototype_validation.json")

        class_order = labels.get("class_order")
        if not isinstance(class_order, list) or len(class_order) != 33:
            raise BundleError("This demo expects the prototype's fixed 33-label order.")
        classes = labels.get("classes")
        if not isinstance(classes, list) or len(classes) != len(class_order):
            raise BundleError("labels.json does not contain one class record per output.")
        if any(item.get("index") != index or item.get("character") != class_order[index]
               for index, item in enumerate(classes)):
            raise BundleError("Class records in labels.json do not match the saved class order.")
        if model_spec.get("architecture") != "torchvision DenseNet121":
            raise BundleError("The bundle architecture is not the supported DenseNet121 prototype.")
        if model_spec.get("classifier_outputs") != len(class_order):
            raise BundleError("The model output size does not match labels.json.")
        if model_spec.get("input_shape") != [3, 224, 224]:
            raise BundleError("The model input shape does not match the 224×224 app preprocessing.")
        expected_mean = [0.485, 0.456, 0.406]
        expected_std = [0.229, 0.224, 0.225]
        if (model_spec.get("normalization_mean") != expected_mean
                or model_spec.get("normalization_std") != expected_std
                or preprocessing.get("output_size") != 224
                or preprocessing.get("content_size") != 192
                or preprocessing.get("mean") != expected_mean
                or preprocessing.get("std") != expected_std):
            raise BundleError("Model normalization settings do not match the app preprocessing.")
        if model_spec.get("final_test_used") is not False:
            raise BundleError("The selected bundle must show that test data was not used for training or checkpoint selection.")

        source = contents["preprocessing_source.py"]
        source_hash = _sha256(source)
        if preprocessing.get("source_sha256") != source_hash:
            raise BundleError("The preprocessing source does not match preprocessing.json.")
        app_source = Path(__import__("app_preprocessing").__file__).read_text(encoding="utf-8")
        start = app_source.index("def prepare_image(")
        app_function_source = app_source[start:]
        if source.decode("utf-8").rstrip() != app_function_source.rstrip():
            raise BundleError("The app's preprocessing function differs from the model bundle.")

        project_labels_path = ROOT / "config/characters.json"
        if project_labels_path.exists():
            project_labels = _parse_json(project_labels_path.read_bytes(), "config/characters.json")
            if project_labels.get("class_order") != class_order:
                raise BundleError("Bundle label order differs from config/characters.json.")

        state = torch.load(io.BytesIO(contents["model_state_dict.pth"]), map_location="cpu", weights_only=True)
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_order))
        model.load_state_dict(state, strict=True)
        model.eval()

        return {
            "model": model,
            "labels": labels,
            "model_spec": model_spec,
            "preprocessing": preprocessing,
            "metrics": metrics,
            "validation": validation,
        }


@st.cache_resource(show_spinner=False)
def _load_bundle_cached(bundle_key: str, _bundle_bytes: bytes) -> dict:
    del bundle_key  # The digest is the cache key; bytes are excluded from hashing.
    return _read_bundle(_bundle_bytes)


def load_bundle(bundle_bytes: bytes) -> dict:
    return _load_bundle_cached(_sha256(bundle_bytes), bundle_bytes)


def _tensor_for_model(image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
    prepared = prepare_image(image)
    pixels = np.asarray(prepared, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).unsqueeze(0).repeat(3, 1, 1)
    return (tensor - MEAN) / STD, prepared


def _predict(model: torch.nn.Module, image: Image.Image, labels: dict) -> tuple[list[dict], Image.Image]:
    tensor, prepared = _tensor_for_model(image)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
    values, indices = torch.topk(probabilities, k=3)
    classes = labels["classes"]
    results = []
    for probability, index in zip(values.tolist(), indices.tolist()):
        info = classes[index]
        results.append({
            "character": info["character"],
            "display": info.get("display", ""),
            "roman_values": info.get("roman_values", []),
            "context_note": info.get("context_note", ""),
            "context_required": bool(info.get("context_required", False)),
            "score": float(probability),
        })
    return results, prepared


def _display_validation(metrics: dict) -> None:
    st.subheader("Saved validation results")
    st.caption("These scores are validation metrics. A separate one-time test diagnostic is documented in the project report.")
    cols = st.columns(2)
    for col, source, heading in (
        (cols[0], "handwritten", "Archive handwriting"),
        (cols[1], "printed", "Printed glyphs"),
    ):
        values = metrics.get(source)
        with col:
            st.markdown(f"**{heading}**")
            if not isinstance(values, dict):
                st.write("No saved metric for this source.")
                continue
            st.metric("Macro F1", f"{values.get('macro_f1', float('nan')):.3f}")
            st.caption(
                f"Accuracy {values.get('accuracy', float('nan')):.3f} · "
                f"{values.get('samples', '—')} examples · "
                f"{values.get('supported_classes', '—')} supported classes"
            )


def main() -> None:
    st.set_page_config(page_title="Persian Character Prototype", page_icon="✍️", layout="centered")
    st.title("Persian character recognition")
    st.write("Upload one isolated Persian character to see the prototype prediction and its simplified Roman display.")
    st.warning(
        "Prototype only: handwritten آ is unvalidated and independent Persian-reader review is pending. "
        "The fixed checkpoint has a one-time test diagnostic; see the project report. Roman values are for isolated letters, not full words."
    )

    default_path = os.environ.get("PERSIAN_MODEL_BUNDLE", str(DEFAULT_BUNDLE))
    path_text = st.text_input("Model bundle ZIP path", value=default_path)
    uploaded_bundle = st.file_uploader("Or choose prototype_model_bundle.zip", type=["zip"], key="bundle")
    bundle_bytes = None
    bundle_name = ""
    if uploaded_bundle is not None:
        bundle_bytes = uploaded_bundle.getvalue()
        bundle_name = uploaded_bundle.name
    elif path_text.strip():
        path = Path(path_text).expanduser()
        if path.is_file():
            try:
                bundle_bytes = path.read_bytes()
                bundle_name = path.name
            except OSError as exc:
                st.error(f"Could not read the model bundle: {exc}")
        else:
            st.info("Select the model bundle ZIP above, or set the path after downloading it from Drive.")

    if bundle_bytes is None:
        st.stop()

    try:
        with st.spinner("Loading the verified model bundle…"):
            bundle = load_bundle(bundle_bytes)
    except Exception as exc:
        st.error(f"Could not load {bundle_name}: {exc}")
        st.stop()

    st.caption(f"Loaded seed-{bundle['model_spec'].get('seed', '—')} DenseNet121 prototype from {bundle_name}.")
    image_file = st.file_uploader(
        "Upload an image containing one isolated character",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        key="image",
    )
    if image_file is not None:
        image_bytes = image_file.getvalue()
        if not image_bytes:
            st.error("The uploaded image is empty.")
        else:
            try:
                with Image.open(io.BytesIO(image_bytes)) as opened:
                    opened.load()
                    image = opened.copy()
                if image.width < 1 or image.height < 1:
                    raise ValueError("The image has no pixels.")
                predictions, prepared = _predict(bundle["model"], image, bundle["labels"])
                st.image(image, caption="Uploaded image", width=240)
                st.image(prepared, caption="Prepared model input", width=240)
                best = predictions[0]
                st.subheader(f"Prediction: {best['character']} — {best['display']}")
                st.caption("Scores are softmax outputs and are not calibrated probabilities.")
                st.dataframe(
                    [
                        {
                            "Rank": rank,
                            "Character": item["character"],
                            "Roman display": item["display"],
                            "Score": f"{item['score']:.1%}",
                        }
                        for rank, item in enumerate(predictions, start=1)
                    ],
                    hide_index=True,
                    width="stretch",
                )
                if best["context_required"] and best["context_note"]:
                    st.info(best["context_note"])
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                st.error(f"This image could not be read: {exc}")

    with st.expander("Validation metrics and coverage"):
        _display_validation(bundle["metrics"])
        st.caption(
            "These validation outputs are development metrics. The separate test diagnostic is in reports/test_evaluation.md; "
            "neither result establishes independent-writer generalization."
        )


if __name__ == "__main__":
    main()
