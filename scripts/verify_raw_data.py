import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import yaml
from datasets import DatasetDict, IterableDatasetDict, load_from_disk

CONFIG_PATH = os.path.join("config", "datasets.yaml")
RAW_ROOT = os.path.join("data", "raw")
REPORT_PATH = os.path.join("data", "metadata", "verify_report.json")

REQUIRED_OBJECTIVES = [
    "auto_complete",
    "code_gen",
    "bug_detection",
    "bug_fix",
    "perf_opt",
    "test_gen",
    "refactoring",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hf_rows(ds: Any) -> Dict[str, int]:
    rows = {}
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        for split, split_ds in ds.items():
            try:
                rows[split] = int(len(split_ds))
            except Exception:
                rows[split] = -1
    else:
        try:
            rows["train"] = int(len(ds))
        except Exception:
            rows["train"] = -1
    return rows


def verify_dataset(name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    source = str(meta.get("source", "")).lower().strip()
    folder = os.path.join(RAW_ROOT, name)
    out: Dict[str, Any] = {
        "dataset": name,
        "objective": meta.get("objective", "unknown"),
        "source": source,
        "folder": folder,
        "status": "EMPTY",
        "usable": False,
    }

    if not os.path.isdir(folder):
        out["status"] = "EMPTY"
        out["reason"] = "Folder missing"
        return out

    items = os.listdir(folder)
    out["item_count"] = len(items)
    if len(items) == 0:
        out["status"] = "EMPTY"
        out["reason"] = "Folder exists but no files"
        return out

    if source == "github":
        out["status"] = "MANUAL_SOURCE_PRESENT"
        out["usable"] = True
        return out

    if source == "huggingface":
        try:
            ds = load_from_disk(folder)
            out["status"] = "VALID_HF_DATASET"
            out["usable"] = True
            out["splits"] = hf_rows(ds)
            return out
        except Exception as e:
            out["status"] = "INVALID_PARTIAL"
            out["reason"] = str(e)
            return out

    out["status"] = "INVALID_PARTIAL"
    out["reason"] = f"Unsupported source type: {source}"
    return out


def main() -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    cfg = load_config()

    datasets_cfg = cfg.get("datasets", {})
    results: Dict[str, Any] = {}
    objective_to_usable: Dict[str, List[str]] = {obj: [] for obj in REQUIRED_OBJECTIVES}

    for name, meta in datasets_cfg.items():
        res = verify_dataset(name, meta)
        results[name] = res
        obj = str(meta.get("objective", "unknown"))
        if obj in objective_to_usable and res.get("usable"):
            objective_to_usable[obj].append(name)

    covered = sorted([k for k, v in objective_to_usable.items() if v])
    uncovered = sorted([k for k, v in objective_to_usable.items() if not v])

    summary = {
        "generated_at": utc_now(),
        "required_objectives": REQUIRED_OBJECTIVES,
        "covered_objectives": covered,
        "uncovered_objectives": uncovered,
        "objective_to_usable_datasets": objective_to_usable,
        "dataset_status_counts": {
            "EMPTY": sum(1 for x in results.values() if x["status"] == "EMPTY"),
            "VALID_HF_DATASET": sum(1 for x in results.values() if x["status"] == "VALID_HF_DATASET"),
            "INVALID_PARTIAL": sum(1 for x in results.values() if x["status"] == "INVALID_PARTIAL"),
            "MANUAL_SOURCE_PRESENT": sum(1 for x in results.values() if x["status"] == "MANUAL_SOURCE_PRESENT"),
        },
        "datasets": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if uncovered:
        sys.exit(2)


if __name__ == "__main__":
    main()
