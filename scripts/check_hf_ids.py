import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datasets import get_dataset_config_names
from huggingface_hub import HfApi

CONFIG_PATH = os.path.join("config", "datasets.yaml")
REPORT_PATH = os.path.join("data", "metadata", "hf_id_check_report.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_hf_id(api: HfApi, hf_id: str) -> Tuple[bool, str]:
    try:
        api.dataset_info(hf_id)
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_config_name(hf_id: str, config_name: str) -> Tuple[bool, str, List[str]]:
    try:
        names = get_dataset_config_names(hf_id)
        ok = config_name in names
        if ok:
            return True, "", names
        return False, f"Config '{config_name}' not in available configs", names
    except Exception as e:
        return False, str(e), []


def pick_first_working_id(api: HfApi, ids: List[str]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    attempts = []
    for hf_id in ids:
        ok, err = validate_hf_id(api, hf_id)
        attempts.append({"hf_id": hf_id, "ok": ok, "error": err})
        if ok:
            return hf_id, attempts
    return None, attempts


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    api = HfApi()

    datasets_cfg = cfg.get("datasets", {})
    report: Dict[str, Any] = {
        "generated_at": utc_now(),
        "config_path": CONFIG_PATH,
        "results": {},
        "summary": {
            "total": 0,
            "resolved": 0,
            "failed": 0,
            "manual_sources": 0,
        },
    }

    for dataset_name, meta in datasets_cfg.items():
        report["summary"]["total"] += 1
        source = str(meta.get("source", "")).strip().lower()

        if source != "huggingface":
            report["summary"]["manual_sources"] += 1
            report["results"][dataset_name] = {
                "status": "MANUAL_SOURCE",
                "source": source,
                "reason": "Non-HuggingFace source",
                "github_url": meta.get("github_url", ""),
            }
            continue

        ids = meta.get("hf_ids") or []
        if isinstance(ids, str):
            ids = [ids]

        hf_config_name = str(meta.get("hf_config_name", "")).strip()
        first_id, attempts = pick_first_working_id(api, ids)

        result: Dict[str, Any] = {
            "status": "FAILED",
            "attempts": attempts,
            "resolved_hf_id": None,
            "hf_config_name": hf_config_name or None,
            "config_check": None,
        }

        if first_id:
            result["resolved_hf_id"] = first_id
            if hf_config_name:
                ok, err, available = validate_config_name(first_id, hf_config_name)
                result["config_check"] = {
                    "ok": ok,
                    "error": err,
                    "available_configs": available,
                }
                if ok:
                    result["status"] = "RESOLVED"
                    report["summary"]["resolved"] += 1
                else:
                    result["status"] = "FAILED"
                    report["summary"]["failed"] += 1
            else:
                result["status"] = "RESOLVED"
                report["summary"]["resolved"] += 1
        else:
            report["summary"]["failed"] += 1

        report["results"][dataset_name] = result

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
