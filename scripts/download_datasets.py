import argparse
import glob
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from datasets import Dataset, DatasetDict, IterableDatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from converters.base import clean_text, guess_language_from_text, normalize_lang

MAX_RETRIES = 3
REPORT_PATH = "data/metadata/download_report.json"
PROGRESS_PATH = "data/metadata/progress_phase_1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> Dict[str, Any]:
    with open("config/datasets.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)


def get_candidates(meta: Dict[str, Any]) -> List[str]:
    ids = []
    if "hf_ids" in meta and isinstance(meta["hf_ids"], list):
        ids.extend([x for x in meta["hf_ids"] if x])
    if "hf_id" in meta and meta["hf_id"]:
        ids.append(meta["hf_id"])
    # stable de-dup
    return list(dict.fromkeys(ids))


def resolve_hf_id(candidates: List[str]) -> Tuple[str, str]:
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    api = HfApi()
    errors: List[str] = []
    for hf_id in candidates:
        try:
            api.dataset_info(hf_id, token=token)
            return hf_id, ""
        except Exception as e:
            errors.append(f"{hf_id}: {e}")
    return "", " | ".join(errors)


def split_stats(dataset_obj: Any) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    if isinstance(dataset_obj, (DatasetDict, IterableDatasetDict)):
        for split_name, split_ds in dataset_obj.items():
            try:
                stats[split_name] = int(len(split_ds))
            except Exception:
                stats[split_name] = -1
    else:
        try:
            stats["train"] = int(len(dataset_obj))
        except Exception:
            stats["train"] = -1
    return stats


def total_rows(stats: Dict[str, int]) -> int:
    return sum(v for v in stats.values() if v > 0)


def try_download(hf_id: str, output_dir: str, config_name: str = "") -> Tuple[bool, Dict[str, int], str]:
    last_error = ""
    rows: Dict[str, int] = {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if config_name:
                dataset = load_dataset(hf_id, name=config_name)
            else:
                dataset = load_dataset(hf_id)

            rows = split_stats(dataset)
            if total_rows(rows) <= 0:
                raise ValueError("dataset resolved but appears empty (0 rows)")

            dataset.save_to_disk(output_dir)
            return True, rows, ""
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = str(e)
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")
            if attempt < MAX_RETRIES:
                delay = attempt * 5
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)

    return False, rows, last_error


def _first_present(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key])
    return default


def _detect_stream_language(row: Dict[str, Any]) -> str:
    lang = normalize_lang(
        _first_present(
            row,
            ["language", "lang", "programming_language", "ext", "file_language", "repo_language"],
            "",
        )
    )
    if lang:
        return lang

    probe = "\n".join(
        [
            _first_present(row, ["content", "code", "text", "input", "output", "before", "after"]),
            _first_present(row, ["path", "file_path", "filename"]),
        ]
    )
    guessed = normalize_lang(guess_language_from_text(probe))
    return guessed or ""


def _looks_like_snippet(text: str) -> bool:
    t = text.lower()
    markers = ["function ", "class ", "=>", "def ", "public ", "private ", "protected ", "#include"]
    return any(m in t for m in markers)


def _normalize_stream_row(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    before = clean_text(
        _first_present(
            row,
            [
                "buggy_code",
                "before",
                "old_code",
                "source",
                "changed_hunks_before",
                "diff_before",
            ],
        )
    )
    after = clean_text(
        _first_present(
            row,
            [
                "fixed_code",
                "after",
                "new_code",
                "target",
                "changed_hunks_after",
                "diff_after",
            ],
        )
    )
    code = clean_text(_first_present(row, ["content", "code", "text", "whole_func_string", "function"]))

    normalized = dict(row)
    normalized["language"] = _detect_stream_language(row)

    if before and after:
        normalized.setdefault("before", before)
        normalized.setdefault("after", after)
    if code:
        normalized.setdefault("code", code)

    normalized.setdefault("repo", _first_present(row, ["repo", "repository", "repo_name", "project"], "unknown"))
    normalized.setdefault("path", _first_present(row, ["path", "file_path", "filename"], ""))
    if not normalized.get("path") and row.get("__source_file"):
        normalized["path"] = str(row.get("__source_file"))
    normalized.setdefault("source", dataset_name)
    # Stack-style rows often expose only generic text content.
    if not normalized.get("code"):
        fallback_code = clean_text(_first_present(normalized, ["content", "text", "body", "source"], ""))
        if fallback_code:
            normalized["code"] = fallback_code
    return normalized


def _validate_required_semantic_columns(rows: List[Dict[str, Any]]) -> Tuple[bool, str, List[str]]:
    if not rows:
        return False, "no_rows", []
    keys = sorted(list(rows[0].keys()))
    has_code = any(k in rows[0] for k in ["whole_func_string", "func_code_string", "code", "function", "content", "text"])
    has_lang = any(k in rows[0] for k in ["language", "lang", "programming_language"])
    if not has_code:
        return False, "missing_code_like_column", keys
    if not has_lang:
        return False, "missing_language_like_column", keys
    return True, "ok", keys


def _save_dataset_dir(train_rows: List[Dict[str, Any]], output_dir: str) -> Tuple[bool, str]:
    if not train_rows:
        return False, "No rows to save"

    tmp_dir = output_dir + ".tmp_hf_dataset"
    backup_dir = output_dir + ".backup"
    try:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        ds = Dataset.from_list(train_rows)
        ds.save_to_disk(tmp_dir)

        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
        if os.path.exists(output_dir):
            os.replace(output_dir, backup_dir)

        os.replace(tmp_dir, output_dir)
        shutil.rmtree(backup_dir, ignore_errors=True)
        return True, ""
    except Exception as e:
        return False, str(e)


def _stack_collect_rows(
    ds_iter: Iterable[Dict[str, Any]],
    dataset_name: str,
    max_samples: int,
    max_chars: int,
    source_path_hint: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    drops: Dict[str, int] = {}
    langs: Dict[str, int] = {}

    def bump(reason: str) -> None:
        drops[reason] = int(drops.get(reason, 0)) + 1

    for item in ds_iter:
        row = dict(item)
        if source_path_hint and "__source_file" not in row:
            row["__source_file"] = source_path_hint

        normalized = _normalize_stream_row(row, dataset_name)
        content_probe = clean_text(_first_present(normalized, ["code", "content", "text", "source"], ""))
        if not content_probe:
            bump("missing_code_content")
            continue
        if max_chars > 0 and len(content_probe) > max_chars:
            bump("max_chars_exceeded")
            continue
        if not _looks_like_snippet(content_probe):
            bump("not_snippet_like")
            continue

        lang = normalize_lang(str(normalized.get("language", "")))
        if not lang:
            source_path = (str(normalized.get("path", "")) + " " + str(normalized.get("__source_file", ""))).lower()
            if "typescript" in source_path:
                lang = "javascript"
            elif "javascript" in source_path:
                lang = "javascript"
        if not lang:
            bump("missing_language")
            continue

        normalized["language"] = lang
        rows.append(normalized)
        langs[lang] = int(langs.get(lang, 0)) + 1

        if max_samples > 0 and len(rows) >= max_samples:
            break

    return rows, drops, langs


def prepare_stack_v2_subset(
    dataset_name: str,
    hf_id: str,
    output_dir: str,
    meta: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any], str]:
    token = (os.getenv("HF_TOKEN") or "").strip()
    max_samples = int(meta.get("max_samples", 0) or 0)
    max_chars = int(meta.get("max_chars", 0) or 0)

    local_parquets = sorted(
        glob.glob(os.path.join(output_dir, "*.parquet"))
        + glob.glob(os.path.join(output_dir, "**", "*.parquet"), recursive=True)
    )

    # Fallback A: try direct gated dataset access with HF token.
    if token:
        config_candidates = ["JavaScript", "TypeScript", "javascript", "typescript"]
        assembled: List[Dataset] = []
        all_rows = 0
        lang_counts: Dict[str, int] = {}
        drops_total: Dict[str, int] = {}
        try:
            for cfg_name in config_candidates:
                if max_samples > 0 and all_rows >= max_samples:
                    break
                ds_stream = load_dataset(hf_id, name=cfg_name, split="train", streaming=True, token=token)
                remaining = (max_samples - all_rows) if max_samples > 0 else 0
                rows, drops, langs = _stack_collect_rows(
                    ds_iter=ds_stream,
                    dataset_name=dataset_name,
                    max_samples=remaining,
                    max_chars=max_chars,
                    source_path_hint=cfg_name,
                )
                if rows:
                    assembled.append(Dataset.from_list(rows))
                    all_rows += len(rows)
                for k, v in drops.items():
                    drops_total[k] = int(drops_total.get(k, 0)) + int(v)
                for k, v in langs.items():
                    lang_counts[k] = int(lang_counts.get(k, 0)) + int(v)

            if assembled:
                merged = assembled[0] if len(assembled) == 1 else concatenate_datasets(assembled)
                ok_cols, col_reason, columns = _validate_required_semantic_columns(merged[:1])
                if not ok_cols:
                    return False, {
                        "mode": "stack_v2_hf_token",
                        "rows_written": len(merged),
                        "columns": columns,
                        "drop_reasons": drops_total,
                        "languages": lang_counts,
                    }, f"Column validation failed: {col_reason}"

                ok_save, save_err = _save_dataset_dir(merged.to_list(), output_dir)
                if not ok_save:
                    return False, {
                        "mode": "stack_v2_hf_token",
                        "rows_written": len(merged),
                        "columns": merged.column_names,
                        "drop_reasons": drops_total,
                        "languages": lang_counts,
                    }, f"save_to_disk failed: {save_err}"

                return True, {
                    "mode": "stack_v2_hf_token",
                    "rows_written": len(merged),
                    "columns": merged.column_names,
                    "drop_reasons": drops_total,
                    "languages": lang_counts,
                    "dataset_dir": output_dir,
                }, ""
        except Exception as e:
            # Continue to local parquet fallback.
            hf_err = str(e)
    else:
        hf_err = "HF_TOKEN is not set; skipping gated Stack v2 remote load"

    # Fallback B/C: local parquet files -> normalized HF dataset directory.
    if not local_parquets:
        return False, {
            "mode": "stack_v2_local_parquet",
            "rows_written": 0,
            "local_parquet_files": 0,
        }, f"{hf_err}. No local parquet files found under {output_dir}"

    try:
        rows: List[Dict[str, Any]] = []
        drops: Dict[str, int] = {}
        langs: Dict[str, int] = {}
        files_ok = 0
        files_failed = 0

        for parquet_path in local_parquets:
            if max_samples > 0 and len(rows) >= max_samples:
                break
            try:
                ds_stream = load_dataset("parquet", data_files=[parquet_path], split="train", streaming=True)
                feature_keys = set(list(getattr(ds_stream, "features", {}).keys()))
                code_like = {"whole_func_string", "func_code_string", "code", "function", "content", "text", "source"}
                lang_like = {"language", "lang", "programming_language", "path", "file_path", "filename"}
                if not (feature_keys & code_like):
                    files_failed += 1
                    drops["parquet_missing_code_columns"] = int(drops.get("parquet_missing_code_columns", 0)) + 1
                    continue
                if not (feature_keys & lang_like):
                    files_failed += 1
                    drops["parquet_missing_language_columns"] = int(drops.get("parquet_missing_language_columns", 0)) + 1
                    continue

                remaining = (max_samples - len(rows)) if max_samples > 0 else 0
                part_rows, part_drops, part_langs = _stack_collect_rows(
                    ds_iter=ds_stream,
                    dataset_name=dataset_name,
                    max_samples=remaining,
                    max_chars=max_chars,
                    source_path_hint=parquet_path,
                )
                rows.extend(part_rows)
                files_ok += 1
                for k, v in part_drops.items():
                    drops[k] = int(drops.get(k, 0)) + int(v)
                for k, v in part_langs.items():
                    langs[k] = int(langs.get(k, 0)) + int(v)
            except Exception:
                files_failed += 1
                drops["parquet_read_failed"] = int(drops.get("parquet_read_failed", 0)) + 1

        if not rows:
            return False, {
                "mode": "stack_v2_local_parquet",
                "rows_written": 0,
                "local_parquet_files": len(local_parquets),
                "local_parquet_files_ok": files_ok,
                "local_parquet_files_failed": files_failed,
                "drop_reasons": drops,
                "languages": langs,
            }, "Local parquet fallback produced zero rows"

        ok_cols, col_reason, columns = _validate_required_semantic_columns(rows)
        if not ok_cols:
            return False, {
                "mode": "stack_v2_local_parquet",
                "rows_written": len(rows),
                "local_parquet_files": len(local_parquets),
                "columns": columns,
                "drop_reasons": drops,
                "languages": langs,
            }, f"Column validation failed: {col_reason}"

        ok_save, save_err = _save_dataset_dir(rows, output_dir)
        if not ok_save:
            return False, {
                "mode": "stack_v2_local_parquet",
                "rows_written": len(rows),
                "local_parquet_files": len(local_parquets),
                "columns": columns,
                "drop_reasons": drops,
                "languages": langs,
            }, f"save_to_disk failed: {save_err}"

        return True, {
            "mode": "stack_v2_local_parquet",
            "rows_written": len(rows),
            "local_parquet_files": len(local_parquets),
            "local_parquet_files_ok": files_ok,
            "local_parquet_files_failed": files_failed,
            "columns": columns,
            "drop_reasons": drops,
            "languages": langs,
            "dataset_dir": output_dir,
            "remote_error": hf_err,
        }, ""
    except Exception as e:
        return False, {
            "mode": "stack_v2_local_parquet",
            "rows_written": 0,
            "local_parquet_files": len(local_parquets),
        }, f"{hf_err}. Local parquet fallback failed: {e}"


def _alias_lang(value: str) -> str:
    v = (value or "").strip().lower()
    table = {
        "js": "javascript",
        "javascript": "javascript",
        "ts": "typescript",
        "typescript": "typescript",
        "c++": "cpp",
        "cpp": "cpp",
        "c#": "csharp",
        "cs": "csharp",
        "python": "python",
        "java": "java",
        "csharp": "csharp",
    }
    return table.get(v, v)


def resolve_stream_data_files(hf_id: str, meta: Dict[str, Any]) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(hf_id, repo_type="dataset")
    include_langs = {_alias_lang(x) for x in (meta.get("stream_include_languages", []) or [])}
    file_format = str(meta.get("stream_file_format", "json")).lower()

    selected: List[str] = []

    if hf_id == "bigcode/commitpackft":
        # commitpackft layout: data/<language>/data.jsonl
        for p in files:
            if not p.startswith("data/") or not p.endswith("/data.jsonl"):
                continue
            lang_dir = p.split("/")[1]
            lang_norm = _alias_lang(lang_dir)
            if include_langs and lang_norm not in include_langs:
                continue
            selected.append(p)
        return selected

    if hf_id == "bigcode/the-stack-v2":
        # stack-v2 layout: data/<Language>/train-*.parquet
        allowed_dirs = {"javascript", "typescript"}
        for p in files:
            if not p.startswith("data/"):
                continue
            if file_format == "parquet" and not p.endswith(".parquet"):
                continue
            parts = p.split("/")
            if len(parts) < 3:
                continue
            lang_dir = _alias_lang(parts[1])
            if lang_dir in allowed_dirs:
                selected.append(p)
        return selected

    # Generic fallback by extension + optional language folder hint
    ext = ".parquet" if file_format == "parquet" else ".jsonl"
    for p in files:
        if not p.endswith(ext):
            continue
        if include_langs:
            low = p.lower()
            if not any(re.search(rf"(^|/)({re.escape(lang)})(/|$)", low) for lang in include_langs):
                continue
        selected.append(p)
    return selected


def stream_download_subset(
    dataset_name: str,
    hf_id: str,
    output_dir: str,
    meta: Dict[str, Any],
    allowed_languages: List[str],
) -> Tuple[bool, Dict[str, Any], str]:
    os.makedirs(output_dir, exist_ok=True)
    raw_jsonl_path = os.path.join(output_dir, "raw.jsonl")

    max_samples = int(meta.get("max_samples", 0) or 0)
    max_chars = int(meta.get("max_chars", 0) or 0)
    split = str(meta.get("hf_split", "train"))
    config_name = str(meta.get("hf_config_name", "") or "")
    only_langs = {normalize_lang(x) for x in (meta.get("stream_include_languages", []) or [])}
    only_langs.discard(None)
    require_snippet = bool(meta.get("require_snippet", False))
    require_language_match = bool(meta.get("stream_require_language_match", True))
    use_data_files = bool(meta.get("stream_data_files", False))
    file_format = str(meta.get("stream_file_format", "json")).lower()
    allowed = set(allowed_languages)

    seen = 0
    written = 0
    drops: Dict[str, int] = {}
    lang_counter: Dict[str, int] = {}

    def bump(reason: str) -> None:
        drops[reason] = int(drops.get(reason, 0)) + 1

    token = (os.getenv("HF_TOKEN") or "").strip() or None
    try:
        if use_data_files:
            repo_files = resolve_stream_data_files(hf_id, meta)
            if not repo_files:
                return False, {"rows_seen_stream": 0, "rows_written": 0, "drop_reasons": {}}, "No matching stream data files resolved"
            hf_files = [f"hf://datasets/{hf_id}/{p}" for p in repo_files]
            reader = "parquet" if file_format == "parquet" else "json"
            ds = load_dataset(reader, data_files=hf_files, split="train", streaming=True, token=token)
        elif config_name:
            ds = load_dataset(hf_id, name=config_name, split=split, streaming=True, token=token)
        else:
            ds = load_dataset(hf_id, split=split, streaming=True, token=token)

        with open(raw_jsonl_path, "w", encoding="utf-8") as out:
            for row in ds:
                seen += 1
                source_file = ""
                if isinstance(row, dict):
                    source_file = str(row.get("__source_file", "") or "")
                normalized = _normalize_stream_row(dict(row), dataset_name)
                if source_file and not normalized.get("path"):
                    normalized["path"] = source_file
                language = normalize_lang(str(normalized.get("language", "")))

                if require_language_match:
                    if not language or language not in allowed:
                        bump("unsupported_language")
                        continue
                    if only_langs and language not in only_langs:
                        bump("not_in_stream_include_languages")
                        continue

                content_probe = clean_text(
                    _first_present(
                        normalized,
                        ["code", "content", "before", "after", "input", "output", "text"],
                    )
                )
                if max_chars > 0 and len(content_probe) > max_chars:
                    bump("max_chars_exceeded")
                    continue
                if require_snippet and not _looks_like_snippet(content_probe):
                    bump("not_snippet_like")
                    continue

                if language:
                    normalized["language"] = language
                out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                written += 1
                lang_key = language or "unknown"
                lang_counter[lang_key] = int(lang_counter.get(lang_key, 0)) + 1

                if max_samples > 0 and written >= max_samples:
                    break

                if written > 0 and written % 10000 == 0:
                    out.flush()

        stats = {
            "rows_seen_stream": seen,
            "rows_written": written,
            "drop_reasons": drops,
            "languages": lang_counter,
            "raw_jsonl": raw_jsonl_path,
            "mode": "streaming_subset",
        }
        if written <= 0:
            return False, stats, "Streaming dataset produced zero rows after filtering"
        return True, stats, ""
    except Exception as e:
        msg = str(e)
        if ("gated" in msg.lower() or "401" in msg.lower()) and not token:
            msg = "HF_TOKEN is missing and dataset appears gated; set HF_TOKEN and retry. Original error: " + msg
        return False, {"rows_seen_stream": seen, "rows_written": written, "drop_reasons": drops}, msg


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_progress(download_report: Dict[str, Any]) -> Dict[str, Any]:
    datasets = download_report.get("datasets", {})
    counts = {
        "success": 0,
        "failed": 0,
        "manual": 0,
    }
    for info in datasets.values():
        status = info.get("status")
        if status in counts:
            counts[status] += 1

    return {
        "generated_at": utc_now(),
        "phase_status": {
            "phase_1_2_download": "completed" if counts["failed"] == 0 else "blocked",
            "phase_1_3_convert": "pending",
            "phase_1_4_filter_dedup_split": "pending",
            "phase_1_5_validate": "pending",
        },
        "download_summary": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets with fallback HF IDs and progress reporting")
    parser.add_argument("--only", nargs="*", default=[], help="Optional dataset names to process")
    args = parser.parse_args()

    ensure_dirs()
    config = load_config()
    allowed_languages = config.get("allowed_languages", ["python", "java", "javascript", "cpp", "csharp"])
    targets = set(args.only)

    report: Dict[str, Any] = {
        "generated_at": utc_now(),
        "datasets": {},
    }

    datasets_cfg = config.get("datasets", {})
    for dataset_name, meta in datasets_cfg.items():
        if targets and dataset_name not in targets:
            continue

        obj = meta.get("objective", "unknown")
        raw_dir_name = str(meta.get("raw_dir_name", dataset_name) or dataset_name)
        output_dir = f"data/raw/{raw_dir_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n>>> Downloading {dataset_name} for {obj}...")

        if dataset_name == "stack_v2_subset":
            resolved_hf_id, resolve_error = resolve_hf_id(get_candidates(meta))
            if not resolved_hf_id:
                report["datasets"][dataset_name] = {
                    "status": "failed",
                    "objective": obj,
                    "source": "huggingface",
                    "error": resolve_error,
                    "updated_at": utc_now(),
                }
                save_json(REPORT_PATH, report)
                continue

            print(f"    HF ID: {resolved_hf_id}")
            ok, rows, err = prepare_stack_v2_subset(
                dataset_name=dataset_name,
                hf_id=resolved_hf_id,
                output_dir=output_dir,
                meta=meta,
            )
            if ok:
                print(f"    Saved to {output_dir}")
                report["datasets"][dataset_name] = {
                    "status": "success",
                    "objective": obj,
                    "source": "huggingface",
                    "resolved_hf_id": resolved_hf_id,
                    "splits": rows,
                    "mode": rows.get("mode", "stack_v2_fallback"),
                    "updated_at": utc_now(),
                }
            else:
                print(f"    Failed after retries: {err}")
                report["datasets"][dataset_name] = {
                    "status": "failed",
                    "objective": obj,
                    "source": "huggingface",
                    "resolved_hf_id": resolved_hf_id,
                    "splits": rows,
                    "error": err,
                    "updated_at": utc_now(),
                }
            save_json(REPORT_PATH, report)
            continue

        reuse_from = str(meta.get("reuse_raw_from", "") or "").strip()
        if reuse_from:
            source_raw_dir_name = str(datasets_cfg.get(reuse_from, {}).get("raw_dir_name", reuse_from) or reuse_from)
            source_raw_dir = os.path.join("data", "raw", source_raw_dir_name)
            source_jsonl = os.path.join(source_raw_dir, "raw.jsonl")
            if os.path.exists(source_jsonl) or os.path.isdir(source_raw_dir):
                print(f"    Reusing raw data from {reuse_from} -> {source_raw_dir}")
                report["datasets"][dataset_name] = {
                    "status": "success",
                    "objective": obj,
                    "source": meta.get("source", "huggingface"),
                    "reused_from": reuse_from,
                    "raw_dir": source_raw_dir,
                    "updated_at": utc_now(),
                }
                continue
            report["datasets"][dataset_name] = {
                "status": "failed",
                "objective": obj,
                "source": meta.get("source", "huggingface"),
                "error": f"reuse_raw_from target missing: {source_jsonl}",
                "updated_at": utc_now(),
            }
            continue

        if meta.get("source") == "github":
            url = meta.get("github_url", "")
            print(f"    GitHub source: manual fetch required -> {url}")
            report["datasets"][dataset_name] = {
                "status": "manual",
                "objective": obj,
                "source": "github",
                "url": url,
                "updated_at": utc_now(),
            }
            continue

        candidates = get_candidates(meta)
        if not candidates:
            report["datasets"][dataset_name] = {
                "status": "failed",
                "objective": obj,
                "source": meta.get("source"),
                "error": "No hf_id/hf_ids configured",
                "updated_at": utc_now(),
            }
            continue

        resolved_hf_id, resolve_error = resolve_hf_id(candidates)
        if not resolved_hf_id:
            print("    Could not resolve a valid HF dataset ID")
            report["datasets"][dataset_name] = {
                "status": "failed",
                "objective": obj,
                "source": "huggingface",
                "candidates": candidates,
                "error": resolve_error,
                "updated_at": utc_now(),
            }
            continue

        print(f"    HF ID: {resolved_hf_id}")
        use_stream = bool(meta.get("stream", False))
        if use_stream:
            ok, rows, err = stream_download_subset(
                dataset_name=dataset_name,
                hf_id=resolved_hf_id,
                output_dir=output_dir,
                meta=meta,
                allowed_languages=allowed_languages,
            )
        else:
            ok, rows, err = try_download(
                hf_id=resolved_hf_id,
                output_dir=output_dir,
                config_name=meta.get("hf_config_name", ""),
            )

        if ok:
            print(f"    Saved to {output_dir}")
            report["datasets"][dataset_name] = {
                "status": "success",
                "objective": obj,
                "source": "huggingface",
                "resolved_hf_id": resolved_hf_id,
                "splits": rows,
                "mode": "streaming_subset" if use_stream else "disk_snapshot",
                "updated_at": utc_now(),
            }
        else:
            print(f"    Failed after retries: {err}")
            report["datasets"][dataset_name] = {
                "status": "failed",
                "objective": obj,
                "source": "huggingface",
                "resolved_hf_id": resolved_hf_id,
                "splits": rows,
                "error": err,
                "updated_at": utc_now(),
            }

        save_json(REPORT_PATH, report)

    save_json(REPORT_PATH, report)
    save_json(PROGRESS_PATH, build_progress(report))
    print(f"\n>>> Download report: {REPORT_PATH}")
    print(f">>> Progress report: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()