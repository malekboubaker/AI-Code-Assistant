from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FILES = {
    "train": DATA_DIR / "train.jsonl",
    "val": DATA_DIR / "val.jsonl",
    "test": DATA_DIR / "test.jsonl",
}


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def read_line_at(path: Path, target_index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            if index == target_index:
                return json.loads(raw_line)
    raise IndexError(f"Line index {target_index} out of range for {path}")


def truncate_text(value: Any, limit: int = 100) -> str:
    text = str(value if value is not None else "")
    return text[:limit] + ("..." if len(text) > limit else "")


def preview_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": sample.get("id"),
        "objective": sample.get("objective"),
        "language": sample.get("language"),
        "input": truncate_text(sample.get("input"), 100),
        "output": truncate_text(sample.get("output"), 100),
        "split": sample.get("split"),
    }


def sample_rows(path: Path, k: int = 3, seed: int = 42) -> List[Dict[str, Any]]:
    total = count_lines(path)
    if total == 0:
        return []
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(total), k=min(k, total)))
    return [read_line_at(path, index) for index in picks]


def main() -> int:
    missing = [name for name, path in FILES.items() if not path.exists()]
    if missing:
        print(f"❌ Missing merged files: {missing}")
        return 1

    objective_counts = Counter()
    for split_name, path in FILES.items():
        print(f"\nSample from {path.name}:")
        samples = sample_rows(path, k=3)
        for sample in samples:
            print(json.dumps(preview_sample(sample), indent=2, ensure_ascii=False))

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                objective_counts[str(sample.get("objective", "unknown"))] += 1

    print("\nSample counts per objective across all splits:")
    for objective, count in sorted(objective_counts.items()):
        print(f"  {objective}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())