import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datasets import DatasetDict, IterableDatasetDict, get_dataset_config_names, load_dataset, load_from_disk
from huggingface_hub import HfApi

CONFIG_PATH = os.path.join("config", "datasets.yaml")
REPORT_PATH = os.path.join("data", "metadata", "download_report.json")
PROGRESS_PATH = os.path.join("data", "metadata", "progress_phase_1.json")
RAW_ROOT = os.path.join("data", "raw")
MAX_RETRIES = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    os.makedirs(RAW_ROOT, exist_ok=True)


def update_progress(report: Dict[str, Any]) -> None:
    phase_state = {
        "phase_1_2_download": "completed",
        "phase_1_3_convert": "pending",
        "phase_1_4_filter_dedup_split": "pending",
        "phase_1_5_validate": "pending",
    }

    failed = [name for name, payload in report.get("datasets", {}).items() if payload.get("status") == "FAILED"]
    if failed:
        phase_state["phase_1_2_download"] = "blocked"

    payload = {
        "generated_at": utc_now(),
        "phase_status": phase_state,
        "download_summary": {
            "success": sum(1 for p in report.get("datasets", {}).values() if p.get("status") == "SUCCESS"),
            "skipped": sum(1 for p in report.get("datasets", {}).values() if p.get("status") == "SKIPPED"),
            "manual_required": sum(1 for p in report.get("datasets", {}).values() if p.get("status") == "MANUAL_REQUIRED"),
            "failed": len(failed),
        },
    }

    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_valid_hf_disk_dataset(path: str) -> bool:
    try:
        _ = load_from_disk(path)
        return True
    except Exception:
        return False


def hf_dataset_rows(ds: Any) -> Dict[str, int]:
    rows: Dict[str, int] = {}
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


def validate_hf_id(api: HfApi, hf_id: str) -> Tuple[bool, str]:
    try:
        api.dataset_info(hf_id)
        return True, ""
    except Exception as e:
        return False, str(e)


def try_hf_download(hf_id: str, output_dir: str, config_name: str = "") -> Tuple[bool, str, Dict[str, int], str]:
    last_err = ""
    rows: Dict[str, int] = {}
    used_config = config_name

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if used_config:
                ds = load_dataset(hf_id, name=used_config)
            else:
                ds = load_dataset(hf_id)
            rows = hf_dataset_rows(ds)
            ds.save_to_disk(output_dir)
            return True, "", rows, used_config
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = str(e)

            # Auto-fallback if dataset requires a config and none was specified.
            if (not used_config) and ("Config name is missing" in last_err):
                try:
                    names = get_dataset_config_names(hf_id)
                    if names:
                        used_config = names[0]
                        continue
                except Exception:
                    pass

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 3)

    return False, last_err, rows, used_config


def build_manual_command(resolved_id: str, out_dir: str, config_name: str) -> str:
    if config_name:
        return (
            f"py -c \"from datasets import load_dataset; "
            f"ds=load_dataset('{resolved_id}', name='{config_name}'); "
            f"ds.save_to_disk(r'{out_dir}')\""
        )
    return (
        f"py -c \"from datasets import load_dataset; "
        f"ds=load_dataset('{resolved_id}'); ds.save_to_disk(r'{out_dir}')\""
    )


def git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
        return True
    except Exception:
        return False


def handle_github(dataset_name: str, meta: Dict[str, Any], out_dir: str, force: bool, dry_run: bool) -> Dict[str, Any]:
    url = meta.get("github_url", "")
    if not url:
        return {
            "status": "MANUAL_REQUIRED",
            "reason": "Missing github_url in config",
            "manual_command": f"git clone <REPO_URL> \"{out_dir}\"",
        }

    if os.path.isdir(out_dir) and os.listdir(out_dir) and not force:
        return {
            "status": "SKIPPED",
            "reason": "Folder already non-empty",
        }

    if dry_run:
        return {
            "status": "MANUAL_REQUIRED",
            "reason": "Dry-run mode for GitHub source",
            "manual_command": f"git clone --depth 1 {url} \"{out_dir}\"",
            "url": url,
        }

    if not git_available():
        return {
            "status": "MANUAL_REQUIRED",
            "reason": "git executable not found on PATH",
            "manual_command": f"git clone --depth 1 {url} \"{out_dir}\"",
            "url": url,
        }

    try:
        if os.path.isdir(out_dir) and force:
            shutil.rmtree(out_dir)
        subprocess.run(["git", "clone", "--depth", "1", url, out_dir], check=True)
        return {
            "status": "SUCCESS",
            "url": url,
            "note": "GitHub repository cloned",
        }
    except Exception as e:
        return {
            "status": "MANUAL_REQUIRED",
            "reason": f"Automated clone failed: {e}",
            "manual_command": f"git clone --depth 1 {url} \"{out_dir}\"",
            "url": url,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust dataset downloader v3")
    parser.add_argument("--only", nargs="*", default=[], help="Optional dataset names")
    parser.add_argument("--force", action="store_true", help="Force re-download/overwrite")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate only; do not write data")
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config()
    api = HfApi()
    targets = set(args.only)

    report: Dict[str, Any] = {
        "generated_at": utc_now(),
        "dry_run": args.dry_run,
        "force": args.force,
        "datasets": {},
    }

    for name, meta in cfg.get("datasets", {}).items():
        if targets and name not in targets:
            continue

        out_dir = os.path.join(RAW_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)

        source = str(meta.get("source", "")).lower().strip()
        objective = meta.get("objective", "unknown")

        if source == "github":
            result = handle_github(name, meta, out_dir, args.force, args.dry_run)
            result["objective"] = objective
            result["source"] = source
            result["updated_at"] = utc_now()
            report["datasets"][name] = result
            continue

        if source != "huggingface":
            report["datasets"][name] = {
                "status": "FAILED",
                "objective": objective,
                "source": source,
                "reason": "Unsupported source type",
                "updated_at": utc_now(),
            }
            continue

        if not args.force and is_valid_hf_disk_dataset(out_dir):
            report["datasets"][name] = {
                "status": "SKIPPED",
                "objective": objective,
                "source": source,
                "reason": "Existing valid load_from_disk dataset",
                "updated_at": utc_now(),
            }
            continue

        ids = meta.get("hf_ids") or []
        if isinstance(ids, str):
            ids = [ids]

        if not ids:
            report["datasets"][name] = {
                "status": "FAILED",
                "objective": objective,
                "source": source,
                "reason": "No hf_ids configured",
                "updated_at": utc_now(),
            }
            continue

        hf_config_name = str(meta.get("hf_config_name", "")).strip()
        attempts: List[Dict[str, Any]] = []
        success_payload: Optional[Dict[str, Any]] = None
        first_valid_id: Optional[str] = None
        first_used_config = hf_config_name

        for hf_id in ids:
            ok_id, id_err = validate_hf_id(api, hf_id)
            if not ok_id:
                attempts.append({"hf_id": hf_id, "stage": "resolve", "ok": False, "error": id_err})
                continue

            if first_valid_id is None:
                first_valid_id = hf_id

            if args.dry_run:
                success_payload = {
                    "status": "SUCCESS",
                    "objective": objective,
                    "source": source,
                    "resolved_hf_id": hf_id,
                    "hf_config_name": hf_config_name or None,
                    "note": "Dry-run validation only",
                    "attempts": attempts,
                    "updated_at": utc_now(),
                }
                break

            dl_ok, dl_err, rows, used_config = try_hf_download(hf_id, out_dir, hf_config_name)
            first_used_config = used_config or first_used_config
            if dl_ok:
                success_payload = {
                    "status": "SUCCESS",
                    "objective": objective,
                    "source": source,
                    "resolved_hf_id": hf_id,
                    "hf_config_name": used_config or None,
                    "splits": rows,
                    "attempts": attempts,
                    "updated_at": utc_now(),
                }
                break

            attempts.append({"hf_id": hf_id, "stage": "download", "ok": False, "error": dl_err, "hf_config_name": used_config or None})

        if success_payload is not None:
            report["datasets"][name] = success_payload
        else:
            fallback_id = first_valid_id or ids[0]
            report["datasets"][name] = {
                "status": "FAILED",
                "objective": objective,
                "source": source,
                "resolved_hf_id": fallback_id,
                "hf_config_name": first_used_config or None,
                "reason": "All hf_ids failed to download/resolve",
                "attempts": attempts,
                "manual_command": build_manual_command(fallback_id, out_dir, first_used_config or ""),
                "updated_at": utc_now(),
            }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    update_progress(report)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
