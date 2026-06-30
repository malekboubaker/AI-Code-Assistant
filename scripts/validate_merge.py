from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPECTED_COUNTS = {
    "train": 165_033,
    "val": 13_646,
    "test": 15_624,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
EXPECTED_OBJECTIVES = {
    "auto_complete",
    "code_gen",
    "bug_detection",
    "bug_fix",
    "perf_opt",
    "test_gen",
    "refactoring",
}


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def has_null_or_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, dict):
        return any(has_null_or_nan(item) for item in value.values())
    if isinstance(value, list):
        return any(has_null_or_nan(item) for item in value)
    return False


def validate_file(path: Path, expected_count: int, split_name: str, id_to_split: Dict[str, str], objective_splits: Dict[str, Set[str]]) -> Tuple[bool, int, int]:
    ok = True
    actual_count = 0
    cross_split_duplicates = 0

    if not path.exists():
        print(f"❌ {path.name} missing")
        return False, 0

    try:
        line_count = count_lines(path)
        if line_count == expected_count:
            print(f"✅ {path.name} has {line_count:,} lines")
        else:
            print(f"❌ {path.name} has {line_count:,} lines, expected {expected_count:,}")
            ok = False

        for line_no, row in iter_jsonl(path):
            actual_count += 1
            if not isinstance(row, dict):
                print(f"❌ {path.name} line {line_no} is not a JSON object")
                ok = False
                continue

            if has_null_or_nan(row):
                print(f"❌ {path.name} line {line_no} contains null or NaN")
                ok = False

            input_text = str(row.get("input", ""))
            if not input_text.startswith("[TASK: "):
                print(f"❌ {path.name} line {line_no} missing [TASK: prefix in input")
                ok = False

            if str(row.get("split", "")).strip().lower() != split_name:
                print(f"❌ {path.name} line {line_no} has split={row.get('split')!r}, expected {split_name!r}")
                ok = False

            sample_id = row.get("id")
            if not sample_id:
                print(f"❌ {path.name} line {line_no} missing id")
                ok = False
            else:
                sample_id = str(sample_id)
                previous_split = id_to_split.get(sample_id)
                if previous_split is None:
                    id_to_split[sample_id] = split_name
                elif previous_split != split_name:
                    print(
                        f"❌ Duplicate id across files: {sample_id} found in {path.name} line {line_no} "
                        f"(previously seen in {previous_split})"
                    )
                    ok = False
                    cross_split_duplicates += 1

            objective = str(row.get("objective", "")).strip()
            if objective:
                objective_splits[objective].add(split_name)

    except json.JSONDecodeError as exc:
        print(f"❌ {path.name} has invalid JSON at line {exc.lineno}: {exc.msg}")
        ok = False
    except Exception as exc:
        print(f"❌ Unexpected error while validating {path.name}: {exc}")
        ok = False

    return ok, actual_count, cross_split_duplicates


def main() -> int:
    files = {
        "train": DATA_DIR / "train.jsonl",
        "val": DATA_DIR / "val.jsonl",
        "test": DATA_DIR / "test.jsonl",
    }

    overall_ok = True
    id_to_split: Dict[str, str] = {}
    objective_splits: Dict[str, Set[str]] = defaultdict(set)
    total_rows = 0
    leakage_count = 0

    print("=== VALIDATE MERGE ===")

    for split_name, path in files.items():
        ok, count, duplicates = validate_file(path, EXPECTED_COUNTS[split_name], split_name, id_to_split, objective_splits)
        total_rows += count
        leakage_count += duplicates
        overall_ok = overall_ok and ok

    if total_rows == EXPECTED_TOTAL:
        print(f"✅ Total across all files = {total_rows:,}")
    else:
        print(f"❌ Total across all files = {total_rows:,}, expected {EXPECTED_TOTAL:,}")
        overall_ok = False

    objectives_present = set(objective_splits.keys())
    if EXPECTED_OBJECTIVES.issubset(objectives_present):
        print("✅ All 7 objectives represented across the merged dataset")
    else:
        missing = sorted(EXPECTED_OBJECTIVES - objectives_present)
        print(f"❌ Missing objectives: {missing}")
        overall_ok = False

    split_coverage_ok = True
    for objective in sorted(EXPECTED_OBJECTIVES):
        splits = objective_splits.get(objective, set())
        if splits == {"train", "val", "test"}:
            print(f"✅ {objective} appears in train/val/test")
        else:
            print(f"❌ {objective} split coverage incomplete: {sorted(splits)}")
            split_coverage_ok = False

    if split_coverage_ok:
        print("✅ All 7 objectives represented in all 3 splits")
    else:
        overall_ok = False

    if leakage_count == 0:
        print("✅ No duplicate ids across files")
    else:
        print(f"❌ Duplicate ids detected across files: {leakage_count}")
        overall_ok = False

    if overall_ok:
        print("✅ PASS")
        return 0

    print("❌ FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())