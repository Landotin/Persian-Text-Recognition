"""Persistent, deterministic prepared-image snapshots for the training notebook.

Callers must supply an immutable preparation manifest and its verified hash. Editing
source images or adding samples requires a new preparation snapshot. Cache hits read
only the cache package and metadata, never the original archive, fonts, or images.
Random augmentation and model normalization belong in the training loader.
"""

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image, ImageOps


CACHE_SCHEMA = 2
METADATA_FILES = (
    "prepared_manifest.csv", "prepared_images.zip", "preprocessing.json",
    "preprocessing_source.py", "font_splits.json", "source_run_config.json",
)


class PreparedCacheError(RuntimeError):
    """A completed cache is corrupt or incompatible with its recorded identity."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
        return ""
    if hasattr(value, "item"):
        return value.item()
    return value


def _stable_config(value):
    """Exclude execution locations/times; retain source provenance and review state."""
    if isinstance(value, dict):
        return {str(key): _stable_config(item) for key, item in value.items()
                if str(key).lower() not in {"run_id", "run_dir", "runtime", "runtime_dir",
                                            "runtime_root", "created_at", "generated_at"}
                and not str(key).lower().startswith("runtime_")}
    if isinstance(value, (list, tuple)):
        return [_stable_config(item) for item in value]
    return _plain(value)


def _as_bool(value):
    return value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}


def select_prototype_manifest(frame, seed=42, label_review_approved=False):
    """Keep the existing 100/25 archive sampling rule within frozen train/val splits."""
    chosen = []
    for split, cap in (("train", 100), ("val", 25)):
        part = frame[frame["split"] == split]
        archive = part[(part["source_type"] == "handwritten") & (part["character"] != "آ")]
        for _, group in archive.groupby("character", sort=True):
            chosen.append(group.sample(n=min(cap, len(group)), random_state=seed))
        printed = part[part["source_type"] == "printed"]
        if len(printed):
            chosen.append(printed)
        if label_review_approved:
            manual = part[(part["source_type"] == "handwritten") & (part["character"] == "آ")]
            if len(manual):
                chosen.append(manual.sample(n=min(cap, len(manual)), random_state=seed))
    if not chosen:
        return frame.iloc[:0].copy().reset_index(drop=True)
    return pd.concat(chosen, ignore_index=True).drop_duplicates(
        subset=["split", "source_group", "archive_member", "image_path"]
    ).reset_index(drop=True)


def _validate_frame(frame, allow_test):
    required = {"archive_member", "image_path", "source_group", "source_type",
                "split", "character", "target_idx"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared-image manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot cache an empty prepared-image manifest.")
    allowed = {"train", "val", "test"} if allow_test else {"train", "val"}
    invalid = set(frame["split"]) - allowed
    if invalid:
        raise ValueError(f"Refusing image reads from disallowed split(s): {sorted(invalid)}")
    if (frame.groupby(["source_type", "source_group"])["split"].nunique() > 1).any():
        raise ValueError("Source groups overlap between splits.")
    source_ids = frame.apply(lambda row: "archive:" + row["archive_member"] if row["archive_member"]
                            else "image:" + row["image_path"], axis=1)
    if (pd.DataFrame({"source": source_ids, "split": frame["split"]})
            .groupby("source")["split"].nunique() > 1).any():
        raise ValueError("Source images overlap between splits.")
    targets = frame["target_idx"].astype(int)
    if not targets.between(0, 32).all():
        raise ValueError("Prepared data must use the stable 33-label target range.")


def _safe_relative(value, prefix=None):
    path = PurePosixPath(str(value))
    if (not str(value) or "\\" in str(value) or path.is_absolute()
            or any(part in {".", "..", ""} for part in path.parts)
            or path.parts[0] not in {"images", "previews"} or len(path.parts) != 2
            or (prefix is not None and path.parts[0] != prefix)):
        raise PreparedCacheError(f"Unsafe prepared-image path: {value!r}")
    return path


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreparedCacheError(f"Cannot read cache metadata: {path}") from exc


def _verify_completed(cache_dir, identity, cache_id):
    config = _read_json(cache_dir / "cache_config.json")
    if (config.get("status") != "complete" or config.get("cache_schema") != CACHE_SCHEMA
            or config.get("cache_id") != cache_id or config.get("identity") != identity):
        raise PreparedCacheError("Completed cache has an incompatible identity or status.")
    hashes = config.get("files_sha256", {})
    if set(hashes) != set(METADATA_FILES):
        raise PreparedCacheError("Completed cache is missing required metadata hashes.")
    for name in METADATA_FILES:
        path = cache_dir / name
        if not path.is_file() or _sha256(path) != hashes[name]:
            raise PreparedCacheError(f"Completed cache failed its SHA-256 check: {name}")
    try:
        prepared = pd.read_csv(cache_dir / "prepared_manifest.csv", keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise PreparedCacheError("Cannot read prepared-image manifest.") from exc
    if len(prepared) != config.get("image_count"):
        raise PreparedCacheError("Prepared-image count differs from the completed metadata.")
    for name in ["prepared_relative_path", "prepared_sha256", "source_sha256"]:
        if name not in prepared or (prepared[name].astype(str).str.len() == 0).any():
            raise PreparedCacheError(f"Prepared manifest lacks {name} values.")
    if not prepared["prepared_relative_path"].is_unique:
        raise PreparedCacheError("Prepared-image manifest contains duplicate output paths.")
    for relative in prepared["prepared_relative_path"]:
        _safe_relative(relative, "images")
    for name in ["raw_preview_relative_path", "raw_preview_sha256"]:
        if name not in prepared:
            raise PreparedCacheError(f"Prepared manifest lacks {name}.")
    preview_mask = prepared["raw_preview_relative_path"].astype(str).ne("")
    if not preview_mask.equals(prepared["split"].eq("val")):
        raise PreparedCacheError("Raw preview thumbnails must cover validation rows only.")
    if (prepared.loc[preview_mask, "raw_preview_sha256"].astype(str).str.len() != 64).any():
        raise PreparedCacheError("Validation thumbnails lack their SHA-256 values.")
    if not prepared.loc[preview_mask, "raw_preview_relative_path"].is_unique:
        raise PreparedCacheError("Validation thumbnail paths are duplicated.")
    for relative in prepared.loc[preview_mask, "raw_preview_relative_path"]:
        _safe_relative(relative, "previews")
    return prepared, config


def _has_completed_marker(cache_dir):
    marker = cache_dir / "cache_config.json"
    if not marker.is_file():
        return False
    status = _read_json(marker).get("status")
    if status == "complete":
        return True
    if status in {"building", "incomplete"}:
        return False
    raise PreparedCacheError("Cache metadata has an unrecognized completion status.")


def _runtime_matches(folder, prepared, output_size):
    if not folder.is_dir():
        return False
    for row in prepared.itertuples(index=False):
        path = folder / row.prepared_relative_path
        if not path.is_file() or _sha256(path) != row.prepared_sha256:
            return False
        try:
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (output_size, output_size):
                    return False
        except OSError:
            return False
        if row.raw_preview_relative_path:
            preview_path = folder / row.raw_preview_relative_path
            if not preview_path.is_file() or _sha256(preview_path) != row.raw_preview_sha256:
                return False
            try:
                with Image.open(preview_path) as preview:
                    if (preview.mode != "RGB" or min(preview.size) <= 0
                            or max(preview.size) > 112):
                        return False
            except OSError:
                return False
    return True


def _restore(cache_dir, runtime_root, cache_id, prepared, output_size):
    runtime_root.mkdir(parents=True, exist_ok=True)
    destination = runtime_root / cache_id
    if not _runtime_matches(destination, prepared, output_size):
        staging = Path(tempfile.mkdtemp(prefix=f".{cache_id}.restore-", dir=runtime_root))
        try:
            expected = set(prepared["prepared_relative_path"])
            expected.update(relative for relative in prepared["raw_preview_relative_path"] if relative)
            # A Drive-mounted ZIP is slow to seek for each member. Copy it in one
            # sequential stream, then perform member reads on runtime storage.
            local_package = staging / "prepared_images.zip"
            with (cache_dir / "prepared_images.zip").open("rb") as source, local_package.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)
            with zipfile.ZipFile(local_package) as archive:
                infos = archive.infolist()
                if len(infos) != len(expected) or {entry.filename for entry in infos} != expected:
                    raise PreparedCacheError("Prepared ZIP members differ from its manifest.")
                for entry in infos:
                    _safe_relative(entry.filename)
                    if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
                        raise PreparedCacheError("Prepared ZIP must contain ordinary PNG files only.")
                    target = staging / entry.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
            local_package.unlink()
            if not _runtime_matches(staging, prepared, output_size):
                raise PreparedCacheError("Extracted prepared images failed integrity or shape checks.")
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(staging, destination)
        except (OSError, zipfile.BadZipFile) as exc:
            raise PreparedCacheError("Cannot restore the prepared-image package.") from exc
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    result = prepared.copy()
    result["prepared_image_path"] = [str(destination / relative)
                                     for relative in result["prepared_relative_path"]]
    result["raw_preview_image_path"] = [str(destination / relative) if relative else ""
                                        for relative in result["raw_preview_relative_path"]]
    result["is_prepared"] = True
    return result


def ensure_prepared_cache(frame, cache_root, runtime_root, preprocessing, preprocessing_source,
                          labels_sha256, source_manifest_sha256, archive_sha256, font_splits,
                          source_config, prepare_fn, ensure_archive, scope="prototype",
                          allow_test=False):
    """Build once or restore a verified package of deterministic prepared PNGs.

    ``ensure_archive()`` is called lazily on a miss only if selected archive rows
    exist, and returns the verified original ZIP path. The caller verifies the
    immutable input manifest against ``source_manifest_sha256`` before this call.
    A completed corrupt cache raises PreparedCacheError; incomplete builds may be
    rebuilt. Returned paths point to runtime storage, not individual Drive files.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", scope):
        raise ValueError("Cache scope must be a simple name, such as prototype or final.")
    frame = frame.copy().astype(object).where(frame.notna(), "").reset_index(drop=True)
    _validate_frame(frame, allow_test)
    output_size = int(preprocessing.get("output_size", 224))
    if output_size != 224:
        raise ValueError("DenseNet121 prepared inputs must remain 224 by 224.")
    source_sha = hashlib.sha256(preprocessing_source.encode("utf-8")).hexdigest()
    if preprocessing.get("source_sha256", source_sha) != source_sha:
        raise ValueError("Preparation source differs from its declared SHA-256.")
    identity_frame = frame.drop(columns=["prepared_image_path", "prepared_relative_path", "prepared_sha256",
                                         "raw_preview_image_path", "raw_preview_relative_path",
                                         "raw_preview_sha256"], errors="ignore")
    identity = _plain({
        "cache_schema": CACHE_SCHEMA, "scope": scope, "allow_test": bool(allow_test),
        "archive_sha256": archive_sha256, "labels_sha256": labels_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "preprocessing": preprocessing, "preprocessing_source_sha256": source_sha,
        "font_splits": font_splits, "source_config": _stable_config(source_config),
        "selected_records": identity_frame.to_dict(orient="records"),
    })
    cache_id = f"{scope}_" + hashlib.sha256(_json_bytes(identity)).hexdigest()[:24]
    cache_root, runtime_root = Path(cache_root), Path(runtime_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / cache_id
    marker = cache_dir / "cache_config.json"
    cache_hit = _has_completed_marker(cache_dir)
    if cache_hit:
        prepared, config = _verify_completed(cache_dir, identity, cache_id)
    else:
        staging = Path(tempfile.mkdtemp(prefix=f".{cache_id}.build-", dir=runtime_root))
        durable_staging = None
        raw_archive = None
        try:
            rows = []
            for index, row in frame.iterrows():
                if row["archive_member"]:
                    if raw_archive is None:
                        raw_archive = zipfile.ZipFile(ensure_archive())
                    source_bytes = raw_archive.read(row["archive_member"])
                else:
                    source_bytes = Path(row["image_path"]).read_bytes()
                row_source_sha = hashlib.sha256(source_bytes).hexdigest()
                if row.get("source_sha256") and row["source_sha256"] != row_source_sha:
                    raise ValueError("Source image differs from the saved preparation snapshot.")
                with Image.open(io.BytesIO(source_bytes)) as image:
                    preview_relative = preview_sha = ""
                    if row["split"] == "val":
                        thumbnail = ImageOps.contain(image.convert("RGB"), (112, 112))
                        preview_relative = f"previews/{index:07d}_{row_source_sha[:16]}.png"
                        preview_path = staging / preview_relative
                        preview_path.parent.mkdir(parents=True, exist_ok=True)
                        thumbnail.save(preview_path, format="PNG")
                        preview_sha = _sha256(preview_path)
                    prepared_image = (image.convert("L") if _as_bool(row.get("is_prepared", False))
                                      else prepare_fn(image).convert("L"))
                    if prepared_image.size != (output_size, output_size):
                        raise ValueError(f"Prepared image has unexpected shape: {prepared_image.size}")
                    relative = f"images/{index:07d}_{row_source_sha[:16]}.png"
                    output = staging / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    prepared_image.save(output, format="PNG")
                record = row.to_dict()
                record.update(prepared_relative_path=relative, prepared_sha256=_sha256(output),
                              source_sha256=row_source_sha, is_prepared=True,
                              raw_preview_relative_path=preview_relative, raw_preview_sha256=preview_sha)
                record.pop("prepared_image_path", None)
                record.pop("raw_preview_image_path", None)
                rows.append(record)
            prepared = pd.DataFrame(rows)
            prepared.to_csv(staging / "prepared_manifest.csv", index=False)
            # PNG already compresses its pixels; storing avoids a second pass.
            with zipfile.ZipFile(staging / "prepared_images.zip", "w", compression=zipfile.ZIP_STORED) as archive:
                for relative in prepared["prepared_relative_path"]:
                    archive.write(staging / relative, relative)
                for relative in prepared["raw_preview_relative_path"]:
                    if relative:
                        archive.write(staging / relative, relative)
            payloads = {"preprocessing.json": preprocessing, "font_splits.json": font_splits,
                        "source_run_config.json": source_config}
            for name, payload in payloads.items():
                (staging / name).write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
            (staging / "preprocessing_source.py").write_text(preprocessing_source, encoding="utf-8")
            config = {"cache_schema": CACHE_SCHEMA, "cache_id": cache_id, "status": "complete",
                      "identity": identity, "image_count": len(prepared),
                      "split_counts": prepared["split"].value_counts().to_dict(),
                      "files_sha256": {name: _sha256(staging / name) for name in METADATA_FILES}}
            # Publish complete metadata last. Partially copied packages never count as hits.
            durable_staging = Path(tempfile.mkdtemp(prefix=f".{cache_id}.publish-", dir=cache_root))
            for name in METADATA_FILES:
                shutil.copy2(staging / name, durable_staging / name)
            marker_temp = durable_staging / "cache_config.json.tmp"
            marker_temp.write_text(json.dumps(_plain(config), ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            os.replace(marker_temp, durable_staging / "cache_config.json")
            if cache_dir.exists():
                if _has_completed_marker(cache_dir):
                    prepared, config = _verify_completed(cache_dir, identity, cache_id)
                else:
                    shutil.rmtree(cache_dir)
                    os.replace(durable_staging, cache_dir)
            else:
                os.replace(durable_staging, cache_dir)
            prepared, config = _verify_completed(cache_dir, identity, cache_id)
        finally:
            if raw_archive is not None:
                raw_archive.close()
            if staging.exists():
                shutil.rmtree(staging)
            if durable_staging is not None and durable_staging.exists():
                shutil.rmtree(durable_staging)
    prepared = _restore(cache_dir, runtime_root, cache_id, prepared, output_size)
    return prepared, {"cache_id": cache_id, "cache_dir": str(cache_dir), "cache_hit": cache_hit,
                      "metadata": config}
