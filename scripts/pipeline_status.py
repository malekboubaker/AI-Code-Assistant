import json
import os
from datetime import datetime, timezone

PROGRESS_PATH = "data/metadata/progress_phase_1.json"
DOWNLOAD_REPORT = "data/metadata/download_report.json"
CONVERT_REPORT = "data/metadata/convert_report.json"
FILTER_REPORT = "data/metadata/filter_split_report.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    progress = read_json(PROGRESS_PATH)
    download = read_json(DOWNLOAD_REPORT)
    convert = read_json(CONVERT_REPORT)
    filt = read_json(FILTER_REPORT)

    status = {
        "generated_at": utc_now(),
        "phase_status": progress.get("phase_status", {}),
        "downloads": download.get("datasets", {}),
        "convert_total_records": convert.get("total_records", 0),
        "split_sizes": filt.get("split_sizes", {}),
    }

    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
