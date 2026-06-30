import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import yaml
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_from_disk

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from converters import converter_pie, get_converter

OUT_PATH = "data/processed/unified_raw.jsonl"
REPORT_PATH = "data/metadata/convert_report.json"
PROGRESS_PATH = "data/metadata/progress_phase_1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def flatten_rows(ds: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        for _, split in ds.items():
            for row in split:
                yield row
        return
    if isinstance(ds, Dataset):
        for row in ds:
            yield row


def update_progress(convert_done: bool) -> None:
    if not os.path.exists(PROGRESS_PATH):
        return
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    state.setdefault("phase_status", {})
    state["phase_status"]["phase_1_3_convert"] = "completed" if convert_done else "blocked"
    state["updated_at"] = utc_now()

    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw datasets to unified JSONL schema")
    parser.add_argument("--max-per-dataset", type=int, default=0, help="Cap converted samples per dataset (0 = no cap)")
    args = parser.parse_args()

    with open("config/datasets.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    allowed_languages: List[str] = cfg.get("allowed_languages", ["python", "java", "javascript", "cpp", "csharp"])
    datasets_cfg = cfg.get("datasets", {})

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)

    dataset_summary: Dict[str, Any] = {}
    global_langs: Counter = Counter()
    global_obj: Counter = Counter()
    total = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for dataset_name, meta in datasets_cfg.items():
            raw_dir = os.path.join("data", "raw", dataset_name)
            converter_name = meta.get("converter", "")
            objective = meta.get("objective", "unknown")
            source = str(meta.get("source", "")).lower().strip()

            if not os.path.isdir(raw_dir):
                dataset_summary[dataset_name] = {"status": "missing_raw", "records": 0}
                continue

            try:
                converter = get_converter(converter_name)
            except Exception as e:
                dataset_summary[dataset_name] = {"status": "converter_missing", "records": 0, "error": str(e)}
                continue

            records_count = 0
            lang_counter: Counter = Counter()

            if source == "github" and dataset_name == "pie":
                record_iter = converter_pie.convert_repo(raw_dir, dataset_name, objective, allowed_languages)
            else:
                try:
                    ds = load_from_disk(raw_dir)
                except Exception as e:
                    dataset_summary[dataset_name] = {"status": "load_failed", "records": 0, "error": str(e)}
                    continue

                rows = flatten_rows(ds)
                record_iter = converter(dataset_name, objective, rows, allowed_languages)

            for rec in record_iter:
                if args.max_per_dataset > 0 and records_count >= args.max_per_dataset:
                    break
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_count += 1
                total += 1
                lang = rec.get("language", "unknown")
                obj = rec.get("objective", "unknown")
                lang_counter[lang] += 1
                global_langs[lang] += 1
                global_obj[obj] += 1

            dataset_summary[dataset_name] = {
                "status": "success",
                "records": records_count,
                "objective": objective,
                "languages": dict(lang_counter),
            }

    report = {
        "generated_at": utc_now(),
        "output_path": OUT_PATH,
        "total_records": total,
        "language_distribution": dict(global_langs),
        "objective_distribution": dict(global_obj),
        "datasets": dataset_summary,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    update_progress(convert_done=total > 0)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
