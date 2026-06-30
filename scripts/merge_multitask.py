from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_ROOT = ROOT / "data" / "finetune"
OUTPUT_DIR = ROOT / "data"
OUTPUT_FILES = {
    "train": OUTPUT_DIR / "train.jsonl",
    "val": OUTPUT_DIR / "val.jsonl",
    "test": OUTPUT_DIR / "test.jsonl",
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def iter_source_files(root: Path) -> Iterable[Tuple[str, Path]]:
    for objective_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for jsonl_file in sorted(objective_dir.glob("*.jsonl")):
            yield objective_dir.name, jsonl_file


def safe_json_loads(line: str, path: Path, line_no: int) -> Dict[str, Any] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        logging.warning("Skipping corrupted JSON in %s at line %d: %s", path, line_no, exc)
        return None
    if not isinstance(payload, dict):
        logging.warning("Skipping non-object JSON in %s at line %d", path, line_no)
        return None
    return payload


def add_task_prefix(sample: Dict[str, Any], objective: str) -> Dict[str, Any]:
    merged = dict(sample)
    input_text = str(merged.get("input", ""))
    prefix = f"[TASK: {objective}]"
    if not input_text.startswith(prefix):
        merged["input"] = f"{prefix}\n{input_text}" if input_text else prefix
    else:
        merged["input"] = input_text
    return merged


def main() -> int:
    configure_logging()

    if not FINETUNE_ROOT.exists():
        logging.error("Missing finetune directory: %s", FINETUNE_ROOT)
        return 1

    objective_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    corrupted_lines = 0
    missing_split = 0

    source_files = list(iter_source_files(FINETUNE_ROOT))
    if not source_files:
        logging.error("No source JSONL files found under %s", FINETUNE_ROOT)
        return 1

    for path in OUTPUT_FILES.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with (
        OUTPUT_FILES["train"].open("w", encoding="utf-8", newline="\n") as train_handle,
        OUTPUT_FILES["val"].open("w", encoding="utf-8", newline="\n") as val_handle,
        OUTPUT_FILES["test"].open("w", encoding="utf-8", newline="\n") as test_handle,
    ):
        writers = {"train": train_handle, "val": val_handle, "test": test_handle}

        for objective, jsonl_file in source_files:
            logging.info("Processing %s...", objective)
            file_count = 0
            with jsonl_file.open("r", encoding="utf-8") as handle:
                for line_no, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    sample = safe_json_loads(line, jsonl_file, line_no)
                    if sample is None:
                        corrupted_lines += 1
                        continue

                    split = str(sample.get("split", "")).strip().lower()
                    if split not in writers:
                        missing_split += 1
                        logging.warning(
                            "Skipping %s line %d because split is missing/invalid: %r",
                            jsonl_file,
                            line_no,
                            sample.get("split"),
                        )
                        continue

                    merged = add_task_prefix(sample, objective)
                    writers[split].write(json.dumps(merged, ensure_ascii=False) + "\n")
                    objective_counts[objective] += 1
                    split_counts[split] += 1
                    file_count += 1

            logging.info("Processing %s... %s samples", objective, f"{file_count:,}")

    total = sum(split_counts.values())
    print("\n=== MERGE SUMMARY ===")
    print(f"Total train samples: {split_counts['train']:,}")
    print(f"Total val samples: {split_counts['val']:,}")
    print(f"Total test samples: {split_counts['test']:,}")
    print(f"Combined total: {total:,}")
    print("\nObjective counts:")
    for objective in sorted(objective_counts):
        print(f"  {objective}: {objective_counts[objective]:,}")

    if corrupted_lines:
        print(f"\nSkipped corrupted JSON lines: {corrupted_lines}")
    if missing_split:
        print(f"Skipped rows with missing/invalid split: {missing_split}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())