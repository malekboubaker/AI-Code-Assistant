import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict

FINETUNE_ROOT = os.path.join("data", "finetune")
REPORT_PATH = os.path.join("data", "metadata", "feature_counts_report.json")

OBJECTIVE_FILE_MAP = {
    "auto_complete": os.path.join(FINETUNE_ROOT, "auto_complete", "finetune_autocomplete.jsonl"),
    "code_gen": os.path.join(FINETUNE_ROOT, "code_gen", "finetune_codegen.jsonl"),
    "bug_detection": os.path.join(FINETUNE_ROOT, "bug_detection", "finetune_bugdetect.jsonl"),
    "bug_fix": os.path.join(FINETUNE_ROOT, "bug_fix", "finetune_bugfix.jsonl"),
    "perf_opt": os.path.join(FINETUNE_ROOT, "perf_opt", "finetune_perfopt.jsonl"),
    "test_gen": os.path.join(FINETUNE_ROOT, "test_gen", "finetune_testgen.jsonl"),
    "refactoring": os.path.join(FINETUNE_ROOT, "refactoring", "finetune_refactor.jsonl"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    objective_totals: Dict[str, int] = {}
    objective_split_counts: Dict[str, Dict[str, int]] = {}
    objective_language_counts: Dict[str, Dict[str, int]] = {}
    global_language_counts: Counter = Counter()

    for objective, path in OBJECTIVE_FILE_MAP.items():
        split_counter: Counter = Counter()
        lang_counter: Counter = Counter()
        total = 0

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    total += 1
                    split_counter[str(row.get("split", "unknown"))] += 1
                    lang = str(row.get("language", "unknown"))
                    lang_counter[lang] += 1
                    global_language_counts[lang] += 1

        objective_totals[objective] = total
        objective_split_counts[objective] = dict(split_counter)
        objective_language_counts[objective] = dict(lang_counter)

    total_rows = int(sum(objective_totals.values()))
    global_language_distribution = {
        lang: (count / total_rows if total_rows > 0 else 0.0)
        for lang, count in dict(global_language_counts).items()
    }

    report: Dict[str, Any] = {
        "generated_at": utc_now(),
        "objective_totals": objective_totals,
        "objective_split_counts": objective_split_counts,
        "objective_language_counts": objective_language_counts,
        "global_language_counts": dict(global_language_counts),
        "global_language_distribution": global_language_distribution,
        "total_rows": total_rows,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
