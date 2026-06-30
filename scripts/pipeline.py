import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_from_disk

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from converters import converter_pie, get_converter
from converters.base import init_conversion_stats

REPORT_PATH = os.path.join("data", "metadata", "phase13_pipeline_report.json")
CONVERSION_REPORT_PATH = os.path.join("data", "metadata", "conversion_report.json")

OBJECTIVE_OUTPUT_FILES = {
    "auto_complete": ("auto_complete", "finetune_autocomplete.jsonl"),
    "code_gen": ("code_gen", "finetune_codegen.jsonl"),
    "bug_detection": ("bug_detection", "finetune_bugdetect.jsonl"),
    "bug_fix": ("bug_fix", "finetune_bugfix.jsonl"),
    "perf_opt": ("perf_opt", "finetune_perfopt.jsonl"),
    "test_gen": ("test_gen", "finetune_testgen.jsonl"),
    "refactoring": ("refactoring", "finetune_refactor.jsonl"),
}


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


def iter_jsonl_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_raw_dir(meta: Dict[str, Any], dataset_name: str, datasets_cfg: Dict[str, Any]) -> str:
    reuse_from = str(meta.get("reuse_raw_from", "") or "").strip()
    if reuse_from:
        parent = datasets_cfg.get(reuse_from, {})
        parent_raw = str(parent.get("raw_dir_name", reuse_from) or reuse_from)
        return os.path.join("data", "raw", parent_raw)
    raw_name = str(meta.get("raw_dir_name", dataset_name) or dataset_name)
    return os.path.join("data", "raw", raw_name)


def ensure_dirs() -> None:
    os.makedirs(os.path.join("data", "finetune"), exist_ok=True)
    os.makedirs(os.path.join("data", "metadata"), exist_ok=True)
    for folder, _ in OBJECTIVE_OUTPUT_FILES.values():
        os.makedirs(os.path.join("data", "finetune", folder), exist_ok=True)


def output_path_for_objective(objective: str) -> str:
    if objective not in OBJECTIVE_OUTPUT_FILES:
        raise KeyError(f"Unknown objective mapping: {objective}")
    folder, filename = OBJECTIVE_OUTPUT_FILES[objective]
    return os.path.join("data", "finetune", folder, filename)


def validate_record_schema(rec: Dict[str, Any]) -> Tuple[bool, str]:
    required_keys = ["id", "feature", "language", "input", "output", "context", "metadata"]
    for key in required_keys:
        if key not in rec:
            return False, f"missing_{key}"
    if not isinstance(rec.get("metadata"), dict) or len(rec.get("metadata", {})) == 0:
        return False, "invalid_metadata"
    if not str(rec.get("input", "")).strip() or not str(rec.get("output", "")).strip():
        return False, "empty_input_or_output"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.3 converter orchestrator")
    parser.add_argument("--max-per-dataset", type=int, default=0, help="Cap converted records per dataset (0 means no cap)")
    args = parser.parse_args()

    ensure_dirs()

    with open(os.path.join("config", "datasets.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    allowed_languages: List[str] = cfg.get("allowed_languages", ["python", "java", "javascript", "cpp", "csharp"])
    datasets_cfg: Dict[str, Any] = cfg.get("datasets", {})

    # One output file per objective, as required by the blueprint.
    objective_file_handles: Dict[str, Any] = {}
    for objective in OBJECTIVE_OUTPUT_FILES:
        objective_file_handles[objective] = open(output_path_for_objective(objective), "w", encoding="utf-8")

    report: Dict[str, Any] = {
        "generated_at": utc_now(),
        "max_per_dataset": args.max_per_dataset,
        "datasets": {},
        "objective_distribution": {},
        "language_distribution": {},
        "outputs": {obj: output_path_for_objective(obj) for obj in OBJECTIVE_OUTPUT_FILES},
    }
    conversion_report: Dict[str, Any] = {
        "generated_at": report["generated_at"],
        "max_per_dataset": args.max_per_dataset,
        "datasets": {},
        "totals": {
            "raw_count": 0,
            "kept_count": 0,
            "dropped_count": 0,
            "retention_rate": 0.0,
            "drop_reasons": {},
        },
    }

    global_obj = Counter()
    global_lang = Counter()
    global_drop_reasons = Counter()

    try:
        for dataset_name, meta in datasets_cfg.items():
            objective = str(meta.get("objective", "")).strip()
            converter_name = str(meta.get("converter", "")).strip()
            source = str(meta.get("source", "")).lower().strip()
            raw_dir = resolve_raw_dir(meta, dataset_name, datasets_cfg)

            if objective not in OBJECTIVE_OUTPUT_FILES:
                report["datasets"][dataset_name] = {
                    "status": "skipped",
                    "reason": f"objective '{objective}' is not in output mapping",
                    "records": 0,
                }
                conversion_report["datasets"][dataset_name] = {
                    "status": "skipped",
                    "reason": f"objective '{objective}' is not in output mapping",
                    "raw_count": 0,
                    "kept_count": 0,
                    "dropped_count": 0,
                    "retention_rate": 0.0,
                    "drop_reasons": {},
                    "languages": {},
                }
                continue

            if not os.path.isdir(raw_dir):
                report["datasets"][dataset_name] = {
                    "status": "missing_raw",
                    "records": 0,
                    "raw_dir": raw_dir,
                }
                conversion_report["datasets"][dataset_name] = {
                    "status": "missing_raw",
                    "raw_dir": raw_dir,
                    "raw_count": 0,
                    "kept_count": 0,
                    "dropped_count": 0,
                    "retention_rate": 0.0,
                    "drop_reasons": {},
                    "languages": {},
                }
                continue

            try:
                converter = get_converter(converter_name)
            except Exception as ex:
                report["datasets"][dataset_name] = {
                    "status": "converter_error",
                    "records": 0,
                    "error": str(ex),
                }
                conversion_report["datasets"][dataset_name] = {
                    "status": "converter_error",
                    "error": str(ex),
                    "raw_count": 0,
                    "kept_count": 0,
                    "dropped_count": 0,
                    "retention_rate": 0.0,
                    "drop_reasons": {},
                    "languages": {},
                }
                continue

            if source == "github" and dataset_name == "pie":
                conversion_stats = init_conversion_stats()
                rec_iter = converter_pie.convert_repo(raw_dir, dataset_name, objective, allowed_languages, stats=conversion_stats)
            else:
                raw_jsonl = os.path.join(raw_dir, "raw.jsonl")
                conversion_stats = init_conversion_stats()
                if os.path.exists(raw_jsonl):
                    rec_iter = converter(dataset_name, objective, iter_jsonl_rows(raw_jsonl), allowed_languages, stats=conversion_stats)
                else:
                    try:
                        ds = load_from_disk(raw_dir)
                    except Exception as ex:
                        report["datasets"][dataset_name] = {
                            "status": "load_failed",
                            "records": 0,
                            "error": str(ex),
                        }
                        conversion_report["datasets"][dataset_name] = {
                            "status": "load_failed",
                            "error": str(ex),
                            "raw_count": 0,
                            "kept_count": 0,
                            "dropped_count": 0,
                            "retention_rate": 0.0,
                            "drop_reasons": {},
                            "languages": {},
                        }
                        continue
                    rec_iter = converter(dataset_name, objective, flatten_rows(ds), allowed_languages, stats=conversion_stats)

            written = 0
            local_lang = Counter()
            local_drop_reasons = Counter()
            handle = objective_file_handles[objective]

            dataset_cap = args.max_per_dataset if args.max_per_dataset > 0 else int(meta.get("max_samples", 0) or 0)

            for rec in rec_iter:
                if dataset_cap > 0 and written >= dataset_cap:
                    break

                valid, reason = validate_record_schema(rec)
                if not valid:
                    conversion_stats["dropped_count"] += 1
                    conversion_stats["drop_reasons"][f"pipeline_{reason}"] = conversion_stats["drop_reasons"].get(f"pipeline_{reason}", 0) + 1
                    local_drop_reasons[f"pipeline_{reason}"] += 1
                    continue

                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                local_lang[str(rec.get("language", "unknown"))] += 1
                global_lang[str(rec.get("language", "unknown"))] += 1
                global_obj[str(rec.get("feature", rec.get("objective", "unknown")))] += 1

                if written % 10000 == 0:
                    handle.flush()

            raw_count = int(conversion_stats.get("raw_count", 0))
            kept_count = int(conversion_stats.get("kept_count", 0))
            dropped_count = int(conversion_stats.get("dropped_count", 0))
            if dataset_cap > 0 and kept_count > written:
                capped_drop = kept_count - written
                dropped_count += capped_drop
                conversion_stats["drop_reasons"]["max_per_dataset_cap"] = int(
                    conversion_stats["drop_reasons"].get("max_per_dataset_cap", 0)
                ) + capped_drop
                kept_count = written

            for reason, count in conversion_stats.get("drop_reasons", {}).items():
                local_drop_reasons[reason] += int(count)
                global_drop_reasons[reason] += int(count)

            report["datasets"][dataset_name] = {
                "status": "success",
                "records": written,
                "objective": objective,
                "languages": dict(local_lang),
            }
            retention = (float(kept_count) / float(raw_count)) if raw_count else 0.0
            conversion_report["datasets"][dataset_name] = {
                "status": "success",
                "feature": objective,
                "raw_count": raw_count,
                "kept_count": kept_count,
                "dropped_count": dropped_count,
                "retention_rate": round(retention, 6),
                "drop_reasons": dict(local_drop_reasons),
                "languages": dict(local_lang),
            }

            conversion_report["totals"]["raw_count"] += raw_count
            conversion_report["totals"]["kept_count"] += kept_count
            conversion_report["totals"]["dropped_count"] += dropped_count

    finally:
        for handle in objective_file_handles.values():
            handle.close()

    report["objective_distribution"] = dict(global_obj)
    report["language_distribution"] = dict(global_lang)
    report["total_records"] = int(sum(global_obj.values()))
    totals_raw = int(conversion_report["totals"]["raw_count"])
    totals_kept = int(conversion_report["totals"]["kept_count"])
    conversion_report["totals"]["retention_rate"] = round((totals_kept / totals_raw), 6) if totals_raw else 0.0
    conversion_report["totals"]["drop_reasons"] = dict(global_drop_reasons)
    conversion_report["objective_distribution"] = dict(global_obj)
    conversion_report["language_distribution"] = dict(global_lang)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(CONVERSION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(conversion_report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
