from __future__ import annotations

import ast
import csv
import json
import logging
import math
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FINETUNE_DIR = DATA_DIR / "finetune"
METADATA_DIR = DATA_DIR / "metadata"
CONVERTERS_DIR = ROOT / "converters"
SCRIPTS_DIR = ROOT / "scripts"
AZURE_SRC_DIR = ROOT / "azure_src"
OUTPUTS_DIR = ROOT / "outputs"
AZURE_DIR = ROOT / ".azure"
JOB_YAML = ROOT / "job.yaml"
PACKAGE_ZIP = ROOT / "ai-code-assistant-phase2.zip"
REPORT_PATH = METADATA_DIR / "pre_azure_verification_report.json"

RAW_DATASETS = {
    "codesearchnet": 5.0,
    "magicoder": 0.1,
    "d2a": 1.0,
    "megadiff": 6.0,
    "pie": 0.5,
    "methods2test": 2.0,
    "codereviewer": 1.0,
    "commitpackft": 1.0,
    "codealpaca": 0.1,
    "stack_v2_subset": 0.1,
}

CONVERTER_FILES = [
    "converter_codesearchnet.py",
    "converter_magicoder.py",
    "converter_d2a.py",
    "converter_megadiff.py",
    "converter_pie.py",
    "converter_methods2test.py",
    "converter_codereviewer.py",
    "converter_commitpackft.py",
    "converter_codealpaca.py",
    "converter_stack_v2_subset.py",
]

OBJECTIVES = [
    "auto_complete",
    "code_gen",
    "bug_detection",
    "bug_fix",
    "perf_opt",
    "test_gen",
    "refactoring",
]

OBJECTIVE_MINIMUMS = {
    "auto_complete": 1000,
    "code_gen": 1000,
    "bug_detection": 500,
    "bug_fix": 500,
    "perf_opt": 500,
    "test_gen": 1000,
    "refactoring": 500,
}

EXPECTED_MERGED_COUNTS = {
    "train": 165_033,
    "val": 13_646,
    "test": 15_624,
}

EXPECTED_ZIP_CONTENTS = {
    "ai-code-assistant-phase2/data/train.jsonl",
    "ai-code-assistant-phase2/data/val.jsonl",
    "ai-code-assistant-phase2/data/test.jsonl",
    "ai-code-assistant-phase2/scripts/train.py",
    "ai-code-assistant-phase2/scripts/evaluate.py",
    "ai-code-assistant-phase2/config.yaml",
    "ai-code-assistant-phase2/requirements.txt",
    "ai-code-assistant-phase2/README.md",
}

ALLOWED_OBJECTIVES = set(OBJECTIVES)
ALLOWED_LANGUAGES = {"python", "java", "javascript", "cpp", "csharp", "c", "php", "go", "ruby"}
ALLOWED_SPLITS = {"train", "val", "test"}


@dataclass
class CheckResult:
    ok: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionResult:
    name: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.ok for check in self.checks) if self.checks else False


def setup_logging() -> logging.Logger:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("pre_azure_verifier")


def human_gb(size_bytes: int) -> float:
    return round(size_bytes / (1024**3), 3)


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, dict):
        return any(has_null_or_nan(item) for item in value.values())
    if isinstance(value, list):
        return any(has_null_or_nan(item) for item in value)
    return False


def run_python_syntax_check(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        return True
    except Exception:
        return False


def safe_import(module_path: Path) -> bool:
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return True
    except Exception:
        return False


def read_json(path: Path) -> Tuple[bool, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return True, json.load(handle)
    except Exception:
        return False, None


def check_raw_datasets(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Phase 1 - Raw datasets")
    details: Dict[str, Any] = {}
    for dataset_name, min_size_gb in RAW_DATASETS.items():
        path = RAW_DIR / dataset_name
        exists = path.exists()
        size_gb = human_gb(dir_size_bytes(path)) if exists else 0.0
        data_files = 0
        can_load = False
        if exists:
            data_files = sum(1 for p in path.rglob("*") if p.is_file())
            can_load = any((path / name).exists() for name in ["raw.jsonl", "dataset_info.json", "state.json", "dataset_dict.json", "README.md", "README", "data"])
        ok = exists and size_gb >= min_size_gb and data_files > 0
        section.checks.append(
            CheckResult(
                ok=ok,
                message=f"{dataset_name}: exists={exists}, size_gb={size_gb}, files={data_files}, can_load={can_load}",
                details={"exists": exists, "size_gb": size_gb, "data_files": data_files, "can_load": can_load},
            )
        )
        details[dataset_name] = section.checks[-1].details
        logger.info("RAW %s -> %s", dataset_name, section.checks[-1].message)
    return section, details


def check_converters(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Phase 1 - Converters")
    details: Dict[str, Any] = {}
    for filename in CONVERTER_FILES:
        path = CONVERTERS_DIR / filename
        exists = path.exists()
        syntax_ok = run_python_syntax_check(path) if exists else False
        import_ok = safe_import(path) if exists and syntax_ok else False
        has_convert = False
        returns_list_of_dicts = False
        if exists and syntax_ok:
            try:
                source = path.read_text(encoding="utf-8")
                has_convert = "def convert_dataset" in source
                returns_list_of_dicts = "return" in source and ("list[dict" in source.lower() or "list of dict" in source.lower())
            except Exception:
                pass
        ok = exists and syntax_ok and has_convert
        section.checks.append(
            CheckResult(
                ok=ok,
                message=f"{filename}: exists={exists}, syntax={syntax_ok}, import={import_ok}, convert_dataset={has_convert}",
                details={
                    "exists": exists,
                    "syntax_ok": syntax_ok,
                    "import_ok": import_ok,
                    "has_convert_dataset": has_convert,
                    "returns_list_of_dicts": returns_list_of_dicts,
                },
            )
        )
        details[filename] = section.checks[-1].details
        logger.info("CONVERTER %s -> %s", filename, section.checks[-1].message)
    return section, details


def check_objective_files(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Phase 1 - Converted data")
    details: Dict[str, Any] = {}
    objective_paths = {objective: FINETUNE_DIR / objective for objective in OBJECTIVES}
    for objective, path in objective_paths.items():
        exists = path.exists()
        jsonl_files = sorted(path.glob("*.jsonl")) if exists else []
        sample_count = 0
        required_fields_ok = False
        file_size_mb = round(sum(p.stat().st_size for p in jsonl_files) / (1024**2), 2) if jsonl_files else 0.0
        if jsonl_files:
            sample_path = jsonl_files[0]
            try:
                sample_count = count_lines(sample_path)
                required_fields_ok = True
                for _, row in iter_jsonl(sample_path):
                    if not isinstance(row, dict) or not all(field in row for field in ("id", "objective", "input", "output")):
                        required_fields_ok = False
                        break
            except Exception:
                required_fields_ok = False
        ok = exists and bool(jsonl_files) and required_fields_ok
        section.checks.append(
            CheckResult(
                ok=ok,
                message=f"{objective}: exists={exists}, jsonl={bool(jsonl_files)}, size_mb={file_size_mb}, samples={sample_count}, fields={required_fields_ok}",
                details={
                    "exists": exists,
                    "has_jsonl": bool(jsonl_files),
                    "file_size_mb": file_size_mb,
                    "sample_count": sample_count,
                    "required_fields_ok": required_fields_ok,
                    "folder": str(path),
                },
            )
        )
        details[objective] = section.checks[-1].details
        logger.info("OBJECTIVE %s -> %s", objective, section.checks[-1].message)
    return section, details


def check_metadata_reports(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Phase 1 - Metadata reports")
    details: Dict[str, Any] = {}
    files = {
        "verify_report": METADATA_DIR / "verify_report.json",
        "filter_split_report": METADATA_DIR / "filter_split_report.json",
        "conversion_report": METADATA_DIR / "conversion_report.json",
    }
    for name, path in files.items():
        exists = path.exists()
        parsed_ok, payload = read_json(path) if exists else (False, None)
        ok = exists and parsed_ok
        if ok and isinstance(payload, dict):
            if name == "verify_report":
                ok = bool(
                    payload.get("total_samples", 0) >= 150000
                    and payload.get("ready_for_phase_2") is True
                    and payload.get("validation_ok") is True
                    and payload.get("all_objectives_covered") is True
                )
            elif name == "filter_split_report":
                split_sizes = payload.get("split_sizes", {})
                per_objective = payload.get("per_objective", {})
                ok = bool(
                    payload.get("filter_metrics", {}).get("after_rebalance", 0) == 194303
                    and all(split_sizes.get(split, 0) > 0 for split in ("train", "val", "test"))
                    and all(obj in per_objective for obj in OBJECTIVES)
                )
            elif name == "conversion_report":
                totals = payload.get("totals", {})
                ok = bool(totals.get("retention_rate", 0) >= 0.95)
        section.checks.append(CheckResult(ok=ok, message=f"{name}: exists={exists}, json={parsed_ok}", details={"exists": exists, "json_ok": parsed_ok, "payload": payload}))
        details[name] = payload if parsed_ok else {}
        logger.info("REPORT %s -> %s", name, section.checks[-1].message)
    return section, details


def check_merged_files(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Phase 2 - Merged JSONL")
    details: Dict[str, Any] = {}
    for split, expected in EXPECTED_MERGED_COUNTS.items():
        path = DATA_DIR / f"{split}.jsonl"
        exists = path.exists()
        line_count = 0
        valid_json = True
        has_task_prefix = True
        has_split_field = True
        no_null_nan = True
        objective_counts = Counter()
        if exists:
            try:
                line_count = count_lines(path)
                for _, row in iter_jsonl(path):
                    if not isinstance(row, dict):
                        valid_json = False
                        continue
                    if has_null_or_nan(row):
                        no_null_nan = False
                    if not str(row.get("input", "")).startswith("[TASK:"):
                        has_task_prefix = False
                    if "split" not in row:
                        has_split_field = False
                    objective_counts[str(row.get("objective", "unknown"))] += 1
            except Exception:
                valid_json = False
        tolerance = int(expected * 0.05)
        count_ok = abs(line_count - expected) <= tolerance
        ok = exists and valid_json and has_task_prefix and has_split_field and no_null_nan and count_ok
        section.checks.append(
            CheckResult(
                ok=ok,
                message=(
                    f"{split}: exists={exists}, lines={line_count}, valid_json={valid_json}, task_prefix={has_task_prefix}, "
                    f"split_field={has_split_field}, no_null_nan={no_null_nan}"
                ),
                details={
                    "exists": exists,
                    "line_count": line_count,
                    "expected": expected,
                    "valid_json": valid_json,
                    "has_task_prefix": has_task_prefix,
                    "has_split_field": has_split_field,
                    "no_null_nan": no_null_nan,
                    "objective_counts": dict(objective_counts),
                },
            )
        )
        details[split] = section.checks[-1].details
        logger.info("MERGED %s -> %s", split, section.checks[-1].message)
    return section, details


def check_zip_file(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Phase 2 - Package ZIP")
    exists = PACKAGE_ZIP.exists()
    size_mb = round(PACKAGE_ZIP.stat().st_size / (1024**2), 2) if exists else 0.0
    corrupted = False
    contents: List[str] = []
    if exists:
        try:
            with zipfile.ZipFile(PACKAGE_ZIP, "r") as zf:
                corrupted = zf.testzip() is not None
                contents = zf.namelist()
        except Exception:
            corrupted = True
    content_set = set(contents)
    required_ok = EXPECTED_ZIP_CONTENTS.issubset(content_set)
    size_ok = 75.0 <= size_mb <= 92.0
    ok = exists and not corrupted and required_ok and size_ok
    section.checks.append(
        CheckResult(
            ok=ok,
            message=f"zip: exists={exists}, size_mb={size_mb}, corrupted={corrupted}, required={required_ok}",
            details={"exists": exists, "size_mb": size_mb, "corrupted": corrupted, "contents": contents},
        )
    )
    logger.info("ZIP -> %s", section.checks[-1].message)
    return section, section.checks[-1].details


def check_split_integrity(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Phase 2 - Split integrity")
    details: Dict[str, Any] = {}
    files = {split: DATA_DIR / f"{split}.jsonl" for split in ALLOWED_SPLITS}
    ids_by_split: Dict[str, set] = {split: set() for split in ALLOWED_SPLITS}
    objectives_by_split: Dict[str, Counter] = {split: Counter() for split in ALLOWED_SPLITS}
    languages_by_split: Dict[str, Counter] = {split: Counter() for split in ALLOWED_SPLITS}

    for split, path in files.items():
        if not path.exists():
            section.checks.append(CheckResult(False, f"{split}: missing"))
            continue
        try:
            for _, row in iter_jsonl(path):
                sample_id = str(row.get("id", ""))
                if sample_id:
                    ids_by_split[split].add(sample_id)
                objectives_by_split[split][str(row.get("objective", "unknown"))] += 1
                languages_by_split[split][str(row.get("language", "unknown"))] += 1
        except Exception:
            section.checks.append(CheckResult(False, f"{split}: invalid JSON while reading"))
            continue
        section.checks.append(CheckResult(True, f"{split}: ids/objectives/languages collected"))

    leakage = (ids_by_split["train"] & ids_by_split["val"]) | (ids_by_split["train"] & ids_by_split["test"]) | (ids_by_split["val"] & ids_by_split["test"])
    no_leakage = len(leakage) == 0
    all_objectives_present = all(all(obj in objectives_by_split[split] for obj in OBJECTIVES) for split in ALLOWED_SPLITS if files[split].exists())
    section.checks.append(CheckResult(no_leakage, f"no_data_leakage={no_leakage}", details={"leakage_count": len(leakage)}))
    section.checks.append(CheckResult(all_objectives_present, f"all_objectives_present={all_objectives_present}", details={"objectives_by_split": {k: dict(v) for k, v in objectives_by_split.items()}}))
    details.update(
        {
            "leakage_count": len(leakage),
            "ids_by_split": {k: len(v) for k, v in ids_by_split.items()},
            "objectives_by_split": {k: dict(v) for k, v in objectives_by_split.items()},
            "languages_by_split": {k: dict(v) for k, v in languages_by_split.items()},
        }
    )
    logger.info("SPLIT -> leakage=%s objectives=%s", len(leakage), all_objectives_present)
    return section, details


def check_azure_files(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Phase 3 - Azure prep")
    files = {
        "job_yaml": JOB_YAML,
        "azure_src_dir": AZURE_SRC_DIR,
        "azure_train": AZURE_SRC_DIR / "train.py",
        "azure_evaluate": AZURE_SRC_DIR / "evaluate.py",
        "azure_requirements": AZURE_SRC_DIR / "requirements.txt",
        "azure_config": AZURE_SRC_DIR / "config.yaml",
    }
    details: Dict[str, Any] = {}
    azure_src_exists = AZURE_SRC_DIR.exists()
    outputs_exists = OUTPUTS_DIR.exists()
    job_yaml_exists = JOB_YAML.exists()
    checks = [
        ("job.yaml exists", job_yaml_exists),
        ("azure_src exists", azure_src_exists),
        ("train.py exists", (AZURE_SRC_DIR / "train.py").exists()),
        ("evaluate.py exists", (AZURE_SRC_DIR / "evaluate.py").exists()),
        ("requirements.txt exists", (AZURE_SRC_DIR / "requirements.txt").exists()),
        ("config.yaml exists", (AZURE_SRC_DIR / "config.yaml").exists()),
    ]
    for msg, ok in checks:
        section.checks.append(CheckResult(ok, msg))

    syntax_ok = (AZURE_SRC_DIR / "train.py").exists() and run_python_syntax_check(AZURE_SRC_DIR / "train.py")
    eval_syntax_ok = (AZURE_SRC_DIR / "evaluate.py").exists() and run_python_syntax_check(AZURE_SRC_DIR / "evaluate.py")
    import_ok = syntax_ok and eval_syntax_ok
    config_ok = False
    if (AZURE_SRC_DIR / "config.yaml").exists():
        try:
            if yaml is None:
                raise RuntimeError("PyYAML unavailable")
            with (AZURE_SRC_DIR / "config.yaml").open("r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
            config_ok = all(key in cfg for key in ("model", "epochs", "batch_size", "learning_rate", "lora_rank", "max_seq_length"))
            details["config"] = cfg
        except Exception:
            config_ok = False
    section.checks.append(CheckResult(syntax_ok, "train.py syntax valid"))
    section.checks.append(CheckResult(eval_syntax_ok, "evaluate.py syntax valid"))
    section.checks.append(CheckResult(import_ok, "imports resolved (syntax-based smoke)"))
    section.checks.append(CheckResult(config_ok, "config.yaml has required keys"))
    details.update({"job_yaml_exists": job_yaml_exists, "azure_src_exists": azure_src_exists, "outputs_exists": outputs_exists})
    logger.info("AZURE -> job=%s azure_src=%s syntax=%s config=%s", job_yaml_exists, azure_src_exists, import_ok, config_ok)
    return section, details


def check_tokenizer_smoke(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Smoke - Tokenization")
    details: Dict[str, Any] = {}
    path = DATA_DIR / "train.jsonl"
    if not path.exists():
        section.checks.append(CheckResult(False, "train.jsonl missing for tokenization smoke"))
        return section, details

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-Next", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        rows = list(iter_jsonl(path))
        rng = random.Random(17)
        picks = rng.sample(range(len(rows)), k=min(10, len(rows)))
        lengths: List[int] = []
        for idx in picks:
            _, row = rows[idx]
            prompt = str(row.get("input", ""))
            encoded = tokenizer(prompt, truncation=True, max_length=8192, padding=False)
            lengths.append(len(encoded.get("input_ids", [])))
            if not encoded.get("input_ids"):
                raise ValueError("empty tokenization result")
        ok = all(length <= 8192 for length in lengths)
        section.checks.append(CheckResult(ok, f"tokenized {len(lengths)} samples", details={"lengths": lengths}))
        details.update({"ok": ok, "lengths": lengths})
        logger.info("TOKENIZER smoke -> %s", ok)
    except Exception as exc:
        section.checks.append(CheckResult(False, f"tokenization smoke failed: {exc}"))
        details.update({"ok": False, "error": str(exc)})
        logger.info("TOKENIZER smoke failed: %s", exc)
    return section, details


def check_model_smoke(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Smoke - Model accessibility")
    details: Dict[str, Any] = {}
    try:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-Coder-Next", trust_remote_code=True)
        ok = bool(config)
        section.checks.append(CheckResult(ok, "model config loaded", details={"model_type": getattr(config, "model_type", None), "architectures": getattr(config, "architectures", None)}))
        details.update({"ok": ok, "model_type": getattr(config, "model_type", None), "architectures": getattr(config, "architectures", None)})
        logger.info("MODEL smoke -> %s", ok)
    except Exception as exc:
        section.checks.append(CheckResult(False, f"model config load failed: {exc}"))
        details.update({"ok": False, "error": str(exc)})
        logger.info("MODEL smoke failed: %s", exc)
    return section, details


def sample_spot_checks(logger: logging.Logger) -> Tuple[SectionResult, Dict[str, Any]]:
    section = SectionResult("Spot checks")
    rng = random.Random(42)
    details: Dict[str, Any] = {}
    sample_specs = [("train", 5), ("val", 3), ("test", 3)]
    sampled_rows: Dict[str, List[Dict[str, Any]]] = {}
    all_ok = True
    for split, sample_count in sample_specs:
        path = DATA_DIR / f"{split}.jsonl"
        rows = list(iter_jsonl(path)) if path.exists() else []
        if not rows:
            section.checks.append(CheckResult(False, f"{split}: unavailable for spot check"))
            all_ok = False
            continue
        picks = sorted(rng.sample(range(len(rows)), k=min(sample_count, len(rows))))
        split_samples: List[Dict[str, Any]] = []
        for idx in picks:
            _, row = rows[idx]
            split_samples.append(row)
            required_ok = all(field in row for field in ("id", "objective", "language", "input", "output", "split", "metadata"))
            input_ok = isinstance(row.get("input"), str) and bool(str(row["input"]).strip())
            output_ok = isinstance(row.get("output"), str) and bool(str(row["output"]).strip())
            objective_ok = str(row.get("objective")) in ALLOWED_OBJECTIVES
            language_ok = str(row.get("language")) in ALLOWED_LANGUAGES
            split_ok = str(row.get("split")) in ALLOWED_SPLITS
            task_prefix_ok = str(row.get("input", "")).startswith("[TASK:")
            quality = row.get("metadata", {}).get("quality_score") if isinstance(row.get("metadata"), dict) else None
            quality_ok = isinstance(quality, (int, float)) and 0.0 <= float(quality) <= 1.0
            row_ok = all([required_ok, input_ok, output_ok, objective_ok, language_ok, split_ok, task_prefix_ok, quality_ok])
            if not row_ok:
                all_ok = False
            section.checks.append(CheckResult(row_ok, f"{split}[{idx}] spot check", details={"objective": row.get("objective"), "language": row.get("language")}))
        sampled_rows[split] = split_samples
    details["samples"] = sampled_rows
    section.checks.append(CheckResult(all_ok, f"spot_checks_pass={all_ok}"))
    logger.info("SPOT CHECKS -> %s", all_ok)
    return section, details


def critical_checks(section_data: Dict[str, SectionResult], report_data: Dict[str, Any]) -> Dict[str, bool]:
    raw_ok = section_data["raw"].passed
    converters_ok = section_data["converters"].passed
    finetune_ok = section_data["finetune"].passed
    metadata_ok = section_data["metadata"].passed
    merged_ok = section_data["merged"].passed
    zip_ok = section_data["zip"].passed
    split_ok = section_data["split"].passed
    azure_ok = section_data["azure"].passed
    spot_ok = section_data["spot"].passed

    no_leakage = report_data.get("split", {}).get("leakage_count", 1) == 0
    quality_thresholds_met = all(
        report_data.get("finetune", {}).get(obj, {}).get("sample_count", 0) >= minimum
        for obj, minimum in OBJECTIVE_MINIMUMS.items()
    )
    tokenization_works = bool(report_data.get("tokenization_smoke", {}).get("ok", False))
    model_loads = bool(report_data.get("model_load_smoke", {}).get("ok", False))

    return {
        "phase_1_status": raw_ok and converters_ok and finetune_ok and metadata_ok,
        "phase_2_status": merged_ok and zip_ok and split_ok,
        "phase_3_prep_status": azure_ok,
        "no_data_leakage": no_leakage,
        "quality_thresholds_met": quality_thresholds_met,
        "tokenization_works": tokenization_works,
        "model_loads": model_loads,
        "final_ready": raw_ok and converters_ok and finetune_ok and metadata_ok and merged_ok and zip_ok and split_ok and azure_ok and spot_ok and no_leakage and quality_thresholds_met and tokenization_works and model_loads,
    }


def generate_report(section_results: Dict[str, SectionResult], section_details: Dict[str, Any]) -> Dict[str, Any]:
    critical = critical_checks(section_results, section_details)
    by_objective = section_details.get("finetune", {})
    sample_counts = {
        "train_total": section_details.get("merged", {}).get("train", {}).get("line_count", 0),
        "val_total": section_details.get("merged", {}).get("val", {}).get("line_count", 0),
        "test_total": section_details.get("merged", {}).get("test", {}).get("line_count", 0),
        "by_objective": {
            objective: by_objective.get(objective, {}).get("sample_count", 0) for objective in OBJECTIVES
        },
    }

    structure = {
        "disk_usage_gb": round((dir_size_bytes(DATA_DIR) + dir_size_bytes(ROOT / "scripts") + dir_size_bytes(CONVERTERS_DIR)) / (1024**3), 3),
        "critical_files_present": all([
            DATA_DIR.exists(),
            (DATA_DIR / "train.jsonl").exists(),
            (DATA_DIR / "val.jsonl").exists(),
            (DATA_DIR / "test.jsonl").exists(),
            PACKAGE_ZIP.exists(),
        ]),
    }

    blockers: List[str] = []
    for section_name, section in section_results.items():
        for check in section.checks:
            if not check.ok:
                message = check.message
                if message not in blockers:
                    blockers.append(message)
    if not critical["phase_3_prep_status"] and "Azure prep files are missing because azure_src/ and job.yaml were removed." not in blockers:
        blockers.append("Azure prep files are missing because azure_src/ and job.yaml were removed.")

    final_verdict = "READY_FOR_AZURE" if critical["final_ready"] else "BLOCKED_FIX_ISSUES"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase_1_status": "PASS" if critical["phase_1_status"] else "FAIL",
        "phase_2_status": "PASS" if critical["phase_2_status"] else "FAIL",
        "phase_3_prep_status": "PASS" if critical["phase_3_prep_status"] else "FAIL",
        "critical_checks": {
            "no_data_leakage": critical["no_data_leakage"],
            "quality_thresholds_met": critical["quality_thresholds_met"],
            "tokenization_works": critical["tokenization_works"],
            "model_loads": critical["model_loads"],
        },
        "sample_counts": sample_counts,
        "project_structure": structure,
        "final_verdict": final_verdict,
        "blockers": blockers,
        "details": section_details,
    }


def print_console_report(report: Dict[str, Any]) -> None:
    print("=" * 40)
    print("PRE-AZURE VERIFICATION COMPLETE")
    print("=" * 40)
    print()
    print(f"Phase 1 (Data Collection): {report['phase_1_status']}")
    print(f"Phase 2 (Data Merging): {report['phase_2_status']}")
    print(f"Phase 3 Prep (Azure Setup): {report['phase_3_prep_status']}")
    print()
    print("Critical Checks:")
    for label, value in report["critical_checks"].items():
        print(f"  ✓ {label.replace('_', ' ')}: {'PASS' if value else 'FAIL'}")
    print()
    counts = report["sample_counts"]
    print("Sample Counts:")
    print(f"  Train: {counts['train_total']:,}")
    print(f"  Val: {counts['val_total']:,}")
    print(f"  Test: {counts['test_total']:,}")
    print(f"  Total: {(counts['train_total'] + counts['val_total'] + counts['test_total']):,}")
    print()
    if report["final_verdict"] == "READY_FOR_AZURE":
        print("Status: READY_FOR_AZURE")
    else:
        print("Status: BLOCKED - FIX:")
        for idx, blocker in enumerate(report["blockers"], start=1):
            print(f"  {idx}. {blocker}")


def main() -> int:
    logger = setup_logging()
    section_results: Dict[str, SectionResult] = {}
    section_details: Dict[str, Any] = {}

    logger.info("Starting pre-Azure verification")

    raw_section, raw_details = check_raw_datasets(logger)
    section_results["raw"] = raw_section
    section_details["raw"] = raw_details

    converter_section, converter_details = check_converters(logger)
    section_results["converters"] = converter_section
    section_details["converters"] = converter_details

    finetune_section, finetune_details = check_objective_files(logger)
    section_results["finetune"] = finetune_section
    section_details["finetune"] = finetune_details

    metadata_section, metadata_details = check_metadata_reports(logger)
    section_results["metadata"] = metadata_section
    section_details["metadata"] = metadata_details

    merged_section, merged_details = check_merged_files(logger)
    section_results["merged"] = merged_section
    section_details["merged"] = merged_details

    zip_section, zip_details = check_zip_file(logger)
    section_results["zip"] = zip_section
    section_details["zip"] = zip_details

    split_section, split_details = check_split_integrity(logger)
    section_results["split"] = split_section
    section_details["split"] = split_details

    azure_section, azure_details = check_azure_files(logger)
    section_results["azure"] = azure_section
    section_details["azure"] = azure_details

    spot_section, spot_details = sample_spot_checks(logger)
    section_results["spot"] = spot_section
    section_details["spot"] = spot_details

    token_section, token_details = check_tokenizer_smoke(logger)
    section_results["tokenization"] = token_section
    section_details["tokenization_smoke"] = token_details

    model_section, model_details = check_model_smoke(logger)
    section_results["model"] = model_section
    section_details["model_load_smoke"] = model_details

    report = generate_report(section_results, section_details)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print_console_report(report)
    logger.info("Wrote report to %s", REPORT_PATH)

    return 0 if report["final_verdict"] == "READY_FOR_AZURE" else 1


if __name__ == "__main__":
    raise SystemExit(main())