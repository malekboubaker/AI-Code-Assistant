from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_SOURCE = ROOT / "ai-code-assistant-phase2.zip"
ZIP_TARGET = ROOT / "ai-code-assistant-phase2-fixed.zip"
FILES = {
    "train": DATA_DIR / "train.jsonl",
    "val": DATA_DIR / "val.jsonl",
    "test": DATA_DIR / "test.jsonl",
}


def read_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_metadata_keys(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    keys = set()
    for rows in rows_by_split.values():
        for row in rows:
            meta = row.get("metadata")
            if isinstance(meta, dict):
                keys.update(meta.keys())
    return sorted(keys)


def infer_metadata_defaults(rows_by_split: Dict[str, List[Dict[str, Any]]], metadata_keys: List[str]) -> Dict[str, Any]:
    inferred_types: Dict[str, type] = {}
    for rows in rows_by_split.values():
        for row in rows:
            meta = row.get("metadata")
            if not isinstance(meta, dict):
                continue
            for key in metadata_keys:
                value = meta.get(key)
                if value is None:
                    continue
                value_type = bool if isinstance(value, bool) else type(value)
                if key not in inferred_types:
                    inferred_types[key] = value_type
                elif inferred_types[key] is int and value_type is float:
                    inferred_types[key] = float

    defaults: Dict[str, Any] = {}
    for key in metadata_keys:
        inferred = inferred_types.get(key, str)
        if inferred is bool:
            defaults[key] = False
        elif inferred is int:
            defaults[key] = 0
        elif inferred is float:
            defaults[key] = 0.0
        else:
            defaults[key] = ""
    return defaults


def normalize_row(row: Dict[str, Any], metadata_keys: List[str], metadata_defaults: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    meta = normalized.get("metadata")
    if isinstance(meta, dict):
        normalized["metadata"] = {key: meta.get(key, metadata_defaults[key]) for key in metadata_keys}
    elif meta is None:
        normalized["metadata"] = {key: metadata_defaults[key] for key in metadata_keys}
    else:
        normalized["metadata"] = {key: metadata_defaults[key] for key in metadata_keys}
    return normalized


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def validate_rows(original_rows: List[Dict[str, Any]], normalized_rows: List[Dict[str, Any]], metadata_keys: List[str]) -> None:
    if len(original_rows) != len(normalized_rows):
        raise AssertionError(f"Row count changed: {len(original_rows)} -> {len(normalized_rows)}")
    for index, row in enumerate(normalized_rows):
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            raise AssertionError(f"Row {index} missing metadata dict")
        if sorted(meta.keys()) != metadata_keys:
            raise AssertionError(f"Row {index} metadata keys mismatch: {sorted(meta.keys())}")


def inspect_schema(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    for split, rows in rows_by_split.items():
        meta_counts = Counter()
        for row in rows:
            meta = row.get("metadata")
            if isinstance(meta, dict):
                meta_counts.update(meta.keys())
        print(f"{split}: rows={len(rows)} metadata_keys={sorted(meta_counts.keys())}")


def rebuild_zip(fixed_paths: Dict[str, Path], zip_source: Path, zip_target: Path) -> None:
    if not zip_source.exists():
        print(f"Skipping zip rebuild because source archive is missing: {zip_source}")
        return

    with zipfile.ZipFile(zip_source, "r") as source_zip, zipfile.ZipFile(zip_target, "w", compression=zipfile.ZIP_DEFLATED) as target_zip:
        replaced_targets = {
            "data/train.jsonl": fixed_paths["train"],
            "data/val.jsonl": fixed_paths["val"],
        }
        if fixed_paths.get("test") is not None:
            replaced_targets["data/test.jsonl"] = fixed_paths["test"]

        for item in source_zip.infolist():
            archive_name = item.filename
            archive_tail = archive_name.split("ai-code-assistant-phase2/", 1)[-1]
            if archive_tail in replaced_targets:
                with replaced_targets[archive_tail].open("rb") as handle:
                    target_zip.writestr(archive_name, handle.read())
            else:
                target_zip.writestr(item, source_zip.read(item.filename))

    print(f"Rebuilt archive: {zip_target}")


def main() -> int:
    missing = [name for name, path in FILES.items() if not path.exists()]
    if missing:
        print(f"Missing input files: {missing}")
        return 1

    rows_by_split = {split: read_rows(path) for split, path in FILES.items()}
    inspect_schema(rows_by_split)

    metadata_keys = collect_metadata_keys(rows_by_split)
    print(f"Unified metadata keys: {metadata_keys}")
    metadata_defaults = infer_metadata_defaults(rows_by_split, metadata_keys)
    print(f"Metadata defaults: {metadata_defaults}")

    fixed_paths: Dict[str, Path] = {}
    for split, original_path in FILES.items():
        rows = rows_by_split[split]
        normalized_rows = [normalize_row(row, metadata_keys, metadata_defaults) for row in rows]
        fixed_path = original_path.with_name(f"{original_path.stem}_fixed{original_path.suffix}")
        write_jsonl(fixed_path, normalized_rows)
        validate_rows(rows, normalized_rows, metadata_keys)
        fixed_paths[split] = fixed_path
        print(f"Wrote {fixed_path.name}: rows={len(normalized_rows)}")

    smoke_paths = {key: str(path) for key, path in fixed_paths.items() if key in {"train", "val", "test"}}
    if "train" in smoke_paths and "val" in smoke_paths:
        loaded = load_dataset("json", data_files={"train": smoke_paths["train"], "validation": smoke_paths["val"]})
        print(
            "datasets.load_dataset smoke test passed: "
            f"train={len(loaded['train'])}, validation={len(loaded['validation'])}"
        )
        print(f"Loaded train columns: {loaded['train'].column_names}")
        print(f"Loaded validation columns: {loaded['validation'].column_names}")

    rebuild_zip(fixed_paths, ZIP_SOURCE, ZIP_TARGET)
    print("Validation complete.")
    for split, fixed_path in fixed_paths.items():
        print(f"{split}_fixed: {fixed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())