import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

try:
    from tree_sitter_languages import get_parser  # type: ignore
except Exception:
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
    except Exception:
        get_parser = None

FINETUNE_ROOT = os.path.join("data", "finetune")
HOLDOUT_ROOT = os.path.join("data", "holdout")
REPORT_PATH = os.path.join("data", "metadata", "filter_split_report.json")
TEMP_FILTERED_PATH = os.path.join("data", "metadata", "_phase14_filtered_dedup_tmp.jsonl")

TARGET_LANGS = ["python", "java", "javascript", "cpp", "csharp"]
SPLIT_RATIO = (0.85, 0.07, 0.08)
ENCODING_NON_ASCII_THRESHOLD = 0.20

OBJECTIVE_FILE_MAP = {
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


def ensure_dirs() -> None:
    os.makedirs(FINETUNE_ROOT, exist_ok=True)
    os.makedirs(HOLDOUT_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


def input_path_for_objective(objective: str) -> str:
    folder, filename = OBJECTIVE_FILE_MAP[objective]
    return os.path.join(FINETUNE_ROOT, folder, filename)


def iter_input_rows() -> Iterable[Dict[str, Any]]:
    for objective in OBJECTIVE_FILE_MAP:
        path = input_path_for_objective(objective)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["objective"] = objective
                yield row


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / max(1, total)


def encoding_ok(record: Dict[str, Any], threshold: float = ENCODING_NON_ASCII_THRESHOLD) -> bool:
    blob = "\n".join(
        [
            str(record.get("input", "")),
            str(record.get("output", "")),
            str(record.get("context", "")),
        ]
    )
    return non_ascii_ratio(blob) <= threshold


def parser_for_language(lang: str):
    if get_parser is None:
        return None
    key = {
        "python": "python",
        "java": "java",
        "javascript": "javascript",
        "cpp": "cpp",
        "csharp": "csharp",
    }.get(lang)
    if not key:
        return None
    try:
        return get_parser(key)
    except Exception:
        return None


def extract_code_for_language(text: str, lang: str) -> str:
    if not text:
        return ""

    blocks = re.findall(r"```([A-Za-z0-9_+#-]*)\n([\s\S]*?)```", text)
    if blocks:
        aliases = {
            "python": {"python", "py"},
            "java": {"java"},
            "javascript": {"javascript", "js", "typescript", "ts", "jsx", "tsx"},
            "cpp": {"cpp", "c++", "cc", "cxx"},
            "csharp": {"csharp", "cs", "c#"},
        }.get(lang, {lang})

        for tag, body in blocks:
            if tag.lower().strip() in aliases:
                return body
        return blocks[0][1]

    starts = {
        "python": ["def ", "class ", "import ", "from "],
        "java": ["package ", "import ", "public class ", "class "],
        "javascript": ["function ", "const ", "let ", "var ", "class "],
        "cpp": ["#include", "using namespace", "int main", "class "],
        "csharp": ["using ", "namespace ", "public class ", "class "],
    }.get(lang, ["class "])

    idx = -1
    for marker in starts:
        i = text.find(marker)
        if i != -1 and (idx == -1 or i < idx):
            idx = i
    if idx != -1:
        return text[idx:]

    return text


def ast_valid(record: Dict[str, Any]) -> bool:
    lang = str(record.get("language", ""))
    parser = parser_for_language(lang)
    if parser is None:
        return False

    objective = str(record.get("objective", ""))
    candidates = [extract_code_for_language(str(record.get("input", "")), lang)]

    if objective in {"bug_detection", "bug_fix", "refactoring", "code_gen", "perf_opt", "test_gen", "auto_complete"}:
        candidates.append(extract_code_for_language(str(record.get("context", "")), lang))
        candidates.append(extract_code_for_language(str(record.get("output", "")), lang))

    for code in candidates:
        if not code:
            continue
        try:
            tree = parser.parse(code.encode("utf-8", errors="ignore"))
            if bool(tree and tree.root_node and not tree.root_node.has_error):
                return True
        except Exception:
            continue
    return False


def exact_fingerprint(record: Dict[str, Any]) -> str:
    key = "|".join(
        [
            str(record.get("objective", "")),
            str(record.get("language", "")),
            str(record.get("input", "")),
            str(record.get("output", "")),
            str(record.get("context", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()


def minhash_signature(text: str, num_perm: int = 16) -> Tuple[int, ...]:
    tokens = text.lower().split()
    if len(tokens) < 5:
        tokens = tokens + ["_"] * (5 - len(tokens))
    shingles = set(" ".join(tokens[i : i + 5]) for i in range(max(1, len(tokens) - 4)))

    sig: List[int] = []
    for i in range(num_perm):
        salt = str(i)
        min_val = None
        for sh in shingles:
            hv = int(hashlib.md5((salt + sh).encode("utf-8", errors="ignore")).hexdigest(), 16)
            if min_val is None or hv < min_val:
                min_val = hv
        sig.append(min_val if min_val is not None else 0)
    return tuple(sig)


def signature_similarity(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / max(1, len(a))


def signature_band_keys(sig: Tuple[int, ...], bands: int = 4) -> List[str]:
    chunk = max(1, len(sig) // bands)
    keys: List[str] = []
    for i in range(bands):
        part = sig[i * chunk : (i + 1) * chunk]
        keys.append(f"b{i}:{hash(part)}")
    return keys


def repo_key(record: Dict[str, Any]) -> str:
    meta = record.get("metadata", {}) or {}
    repo = str(meta.get("repo") or meta.get("repository") or "").strip()
    if repo and repo.lower() != "unknown":
        return repo
    fallback = "|".join(
        [
            str(meta.get("dataset", "")),
            str(meta.get("path", "")),
            str(meta.get("problem_id", "")),
            str(record.get("id", "")),
        ]
    )
    return "pseudo-" + hashlib.sha1(fallback.encode("utf-8", errors="ignore")).hexdigest()[:12]


def split_repo_map(repo_values: List[str], seed: int = 42) -> Dict[str, str]:
    repos = sorted(set(repo_values))
    rng = random.Random(seed)
    rng.shuffle(repos)

    n = len(repos)
    i_train = int(n * SPLIT_RATIO[0])
    i_val = int(n * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))

    train_repos = set(repos[:i_train])
    val_repos = set(repos[i_train:i_val])

    out: Dict[str, str] = {}
    for repo in repos:
        if repo in train_repos:
            out[repo] = "train"
        elif repo in val_repos:
            out[repo] = "val"
        else:
            out[repo] = "test"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.4 filter/dedup/split")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for processed input records (0 means no cap)")
    args = parser.parse_args()

    ensure_dirs()

    if get_parser is None:
        raise RuntimeError("tree_sitter parser backend is unavailable. Install tree_sitter_languages or tree_sitter_language_pack.")

    metrics: Dict[str, int] = {
        "input_records": 0,
        "drop_lang": 0,
        "drop_encoding": 0,
        "drop_ast": 0,
        "drop_exact": 0,
        "drop_near": 0,
    }

    objective_before = Counter()
    objective_after = Counter()
    language_after = Counter()

    seen_fp = set()
    band_index: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
    repos_kept: List[str] = []

    with open(TEMP_FILTERED_PATH, "w", encoding="utf-8") as tmp_out:
        for row in iter_input_rows():
            metrics["input_records"] += 1
            if args.max_records > 0 and metrics["input_records"] > args.max_records:
                break

            objective = str(row.get("objective", "unknown"))
            objective_before[objective] += 1

            lang = str(row.get("language", ""))
            if lang not in TARGET_LANGS:
                metrics["drop_lang"] += 1
                continue

            if not encoding_ok(row):
                metrics["drop_encoding"] += 1
                continue

            if not ast_valid(row):
                metrics["drop_ast"] += 1
                continue

            fp = exact_fingerprint(row)
            if fp in seen_fp:
                metrics["drop_exact"] += 1
                continue
            seen_fp.add(fp)

            txt = "\n".join([str(row.get("input", "")), str(row.get("output", "")), str(row.get("context", ""))])
            sig = minhash_signature(txt)
            candidate_sigs: List[Tuple[int, ...]] = []
            for bkey in signature_band_keys(sig):
                candidate_sigs.extend(band_index.get(bkey, []))

            is_near = any(signature_similarity(sig, other) >= 0.95 for other in candidate_sigs)
            if is_near:
                metrics["drop_near"] += 1
                continue

            for bkey in signature_band_keys(sig):
                bucket = band_index[bkey]
                if len(bucket) < 64:
                    bucket.append(sig)

            rk = repo_key(row)
            row["metadata"] = row.get("metadata", {}) or {}
            row["metadata"]["repo"] = rk
            repos_kept.append(rk)

            objective_after[objective] += 1
            language_after[lang] += 1
            tmp_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    repo_to_split = split_repo_map(repos_kept)

    train_path = os.path.join(HOLDOUT_ROOT, "train.jsonl")
    val_path = os.path.join(HOLDOUT_ROOT, "val.jsonl")
    test_path = os.path.join(HOLDOUT_ROOT, "test.jsonl")

    objective_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    split_counts = Counter()
    objective_split_counts: Dict[str, Counter] = defaultdict(Counter)

    with open(TEMP_FILTERED_PATH, "r", encoding="utf-8") as tmp_in, \
        open(train_path, "w", encoding="utf-8") as train_f, \
        open(val_path, "w", encoding="utf-8") as val_f, \
        open(test_path, "w", encoding="utf-8") as test_f:

        for line in tmp_in:
            row = json.loads(line)
            obj = str(row.get("objective", "unknown"))
            repo = str((row.get("metadata", {}) or {}).get("repo", ""))
            split = repo_to_split.get(repo, "train")
            row["split"] = split

            if split == "train":
                train_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif split == "val":
                val_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                test_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            split_counts[split] += 1
            objective_split_counts[obj][split] += 1
            objective_rows[obj].append(row)

    for obj, (folder, filename) in OBJECTIVE_FILE_MAP.items():
        out_path = os.path.join(FINETUNE_ROOT, folder, filename)
        write_jsonl(out_path, objective_rows.get(obj, []))

    per_objective_report: Dict[str, Any] = {}
    for obj in OBJECTIVE_FILE_MAP:
        per_objective_report[obj] = {
            "total_before_filter": int(objective_before.get(obj, 0)),
            "total_after_filter": int(objective_after.get(obj, 0)),
            "train": int(objective_split_counts[obj].get("train", 0)),
            "val": int(objective_split_counts[obj].get("val", 0)),
            "test": int(objective_split_counts[obj].get("test", 0)),
        }

    report = {
        "generated_at": utc_now(),
        "input_source": "data/finetune/*",
        "max_records": args.max_records,
        "filter_metrics": {
            **metrics,
            "after_filter_and_dedup": int(sum(objective_after.values())),
        },
        "counts_per_language": dict(language_after),
        "split_sizes": {
            "train": int(split_counts.get("train", 0)),
            "val": int(split_counts.get("val", 0)),
            "test": int(split_counts.get("test", 0)),
        },
        "per_objective": per_objective_report,
        "outputs": {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        },
        "tree_sitter_enabled": True,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
