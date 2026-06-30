import argparse
import hashlib
import json
import math
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

TARGET_LANGS = ["python", "java", "javascript", "cpp", "csharp"]
SPLIT_RATIO = (0.85, 0.07, 0.08)
ENCODING_NON_ASCII_THRESHOLD = 0.20
# Realistic language distribution targets based on available source data
# (Not all languages have equal representation in source datasets)
LANGUAGE_TARGET_RATIOS = {
    "python": 0.25,       # Downsample from ~45%
    "java": 0.25,        # Downsample from ~40%
    "cpp": 0.20,         # Boost from ~10%
    "javascript": 0.15,  # Keep all available (~3%)
    "csharp": 0.15,      # Keep all available (~3%)
}

OBJECTIVE_FILE_MAP = {
    "auto_complete": ("auto_complete", "finetune_autocomplete.jsonl"),
    "code_gen": ("code_gen", "finetune_codegen.jsonl"),
    "bug_detection": ("bug_detection", "finetune_bugdetect.jsonl"),
    "bug_fix": ("bug_fix", "finetune_bugfix.jsonl"),
    "perf_opt": ("perf_opt", "finetune_perfopt.jsonl"),
    "test_gen": ("test_gen", "finetune_testgen.jsonl"),
    "refactoring": ("refactoring", "finetune_refactor.jsonl"),
}

AST_STRICTNESS = {
    "auto_complete": "strict",
    "code_gen": "strict",
    "refactoring": "strict",
    "test_gen": "strict",
    "perf_opt": "strict",
    "bug_detection": "lenient",
    "bug_fix": "lenient",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    os.makedirs(FINETUNE_ROOT, exist_ok=True)
    os.makedirs(HOLDOUT_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    for folder, _ in OBJECTIVE_FILE_MAP.values():
        os.makedirs(os.path.join(FINETUNE_ROOT, folder), exist_ok=True)


def objective_path(objective: str) -> str:
    folder, filename = OBJECTIVE_FILE_MAP[objective]
    return os.path.join(FINETUNE_ROOT, folder, filename)


def iter_rows() -> Iterable[Dict[str, Any]]:
    for objective in OBJECTIVE_FILE_MAP:
        path = objective_path(objective)
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


def parse_stats(parser: Any, code: str) -> Tuple[bool, float]:
    """Return (strict_valid, error_ratio)."""
    try:
        tree = parser.parse(code.encode("utf-8", errors="ignore"))
    except Exception:
        return False, 1.0

    if not tree or not tree.root_node:
        return False, 1.0

    root = tree.root_node
    total = 0
    errors = 0
    stack = [root]
    while stack:
        node = stack.pop()
        total += 1
        if getattr(node, "type", "") == "ERROR" or bool(getattr(node, "is_missing", False)):
            errors += 1
        stack.extend(list(node.children))

    ratio = errors / max(1, total)
    strict_valid = not root.has_error and errors == 0
    return strict_valid, ratio


def ast_keep(record: Dict[str, Any]) -> bool:
    lang = str(record.get("language", ""))
    objective = str(record.get("objective", ""))
    mode = AST_STRICTNESS.get(objective, "strict")

    # Skip AST check entirely for bug_fix: diffs are often code fragments
    # Only validate encoding, which is already done in main filter.
    if objective == "bug_fix":
        return True

    parser = parser_for_language(lang)
    if parser is None:
        return False

    candidates = [
        extract_code_for_language(str(record.get("input", "")), lang),
        extract_code_for_language(str(record.get("context", "")), lang),
        extract_code_for_language(str(record.get("output", "")), lang),
    ]

    best_ratio = 1.0
    saw_code = False
    for code in candidates:
        if not code:
            continue
        saw_code = True
        strict_valid, ratio = parse_stats(parser, code)
        if strict_valid:
            return True
        if ratio < best_ratio:
            best_ratio = ratio

    if not saw_code:
        return False

    # Lenient mode for fragment-heavy tasks.
    if mode == "lenient":
        return best_ratio <= 0.5
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


def minhash_signature(text: str, num_perm: int = 32) -> Tuple[int, ...]:
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


def rebalance_languages(rows: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    """Rebalance languages by downsampling to target ratios.
    This function simply applies the target percentages to all languages.
    """
    if not rows:
        return rows, 0, {}

    by_lang: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lang = str(row.get("language", "unknown"))
        by_lang[lang].append(row)

    total_rows = len(rows)
    rng = random.Random(seed)
    kept: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    
    # For each language, downsample to its target ratio
    for lang in TARGET_LANGS:
        lang_rows = by_lang.get(lang, [])
        available = len(lang_rows)
        target = int(math.floor(total_rows * LANGUAGE_TARGET_RATIOS.get(lang, 0.0)))
        
        if available == 0:
            counts[lang] = 0
            continue
        
        # Keep up to target amount (or all if we don't have enough)
        cap = min(available, target)
        shuffled = lang_rows[:]
        rng.shuffle(shuffled)
        kept.extend(shuffled[:cap])
        counts[lang] = cap
    
    rng.shuffle(kept)
    removed = len(rows) - len(kept)
    
    # Log results
    total_kept = len(kept)
    print("\n=== Language Rebalancing Results ===")
    for lang in TARGET_LANGS:
        count = counts.get(lang, 0)
        actual_pct = (count / total_kept * 100) if total_kept else 0
        target_pct = LANGUAGE_TARGET_RATIOS.get(lang, 0.0) * 100
        available = len(by_lang.get(lang, []))
        print(f"{lang:12} -> {count:6d}/{available:6d} samples ({actual_pct:5.1f}% actual, {target_pct:5.1f}% target)")
    print(f"{'Total':12} -> {total_kept:6d} samples (removed {removed})\n")
    
    return kept, removed, counts


def _split_rows_fallback(rows: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Row-level fallback when objective has too few unique repos."""
    local = rows[:]
    random.Random(seed).shuffle(local)
    n = len(local)
    if n <= 2:
        # Keep this explicit; caller decides whether to fail fast.
        return local, [], []

    n_train = max(1, int(n * SPLIT_RATIO[0]))
    n_val = max(1, int(n * SPLIT_RATIO[1]))
    if n_train + n_val >= n:
        n_val = 1
        n_train = max(1, n - 2)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    train = local[:n_train]
    val = local[n_train : n_train + n_val]
    test = local[n_train + n_val :]
    return train, val, test


def _split_objective_by_repo(rows: List[Dict[str, Any]], objective: str, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[repo_key(row)].append(row)

    repos = list(grouped.keys())
    random.Random(seed).shuffle(repos)
    repo_count = len(repos)

    if repo_count < 3:
        train, val, test = _split_rows_fallback(rows, seed=seed)
        for row in train:
            row["split"] = "train"
        for row in val:
            row["split"] = "val"
        for row in test:
            row["split"] = "test"
        return train, val, test, {"repo_count": repo_count}

    n_train_repos = max(1, int(repo_count * SPLIT_RATIO[0]))
    n_val_repos = max(1, int(repo_count * SPLIT_RATIO[1]))
    # Ensure at least one repo left for test.
    if n_train_repos + n_val_repos >= repo_count:
        n_val_repos = 1
        n_train_repos = max(1, repo_count - 2)

    train_repos = set(repos[:n_train_repos])
    val_repos = set(repos[n_train_repos : n_train_repos + n_val_repos])

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for row in rows:
        rk = repo_key(row)
        if rk in train_repos:
            row["split"] = "train"
            train.append(row)
        elif rk in val_repos:
            row["split"] = "val"
            val.append(row)
        else:
            row["split"] = "test"
            test.append(row)

    # Safety fallback if skewed repo sizing causes an empty split.
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        train, val, test = _split_rows_fallback(rows, seed=seed)
        for row in train:
            row["split"] = "train"
        for row in val:
            row["split"] = "val"
        for row in test:
            row["split"] = "test"

    return train, val, test, {"repo_count": repo_count}


def split_by_repo(rows: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Split per objective to avoid one objective collapsing into a single split."""
    by_objective: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_objective[str(row.get("objective", "unknown"))].append(row)

    train_all: List[Dict[str, Any]] = []
    val_all: List[Dict[str, Any]] = []
    test_all: List[Dict[str, Any]] = []
    split_debug: Dict[str, Dict[str, int]] = {}

    for objective, obj_rows in by_objective.items():
        train, val, test, extra = _split_objective_by_repo(obj_rows, objective, seed=seed)
        train_all.extend(train)
        val_all.extend(val)
        test_all.extend(test)

        split_debug[objective] = {
            "total": len(obj_rows),
            "repo_count": extra.get("repo_count", 0),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

        print(
            f"Split {objective} ({len(obj_rows)} total): repo_count={extra.get('repo_count', 0)} "
            f"train={len(train)} val={len(val)} test={len(test)}"
        )

    return train_all, val_all, test_all, split_debug


def count_lang(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter([str(r.get("language", "unknown")) for r in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.4 filter/dedup/split with AST tuning and language rebalance")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap over input records (0 means no cap)")
    args = parser.parse_args()

    ensure_dirs()

    if get_parser is None:
        raise RuntimeError("tree_sitter parser backend is unavailable.")

    metrics: Dict[str, int] = {
        "input_records": 0,
        "drop_lang": 0,
        "drop_encoding": 0,
        "drop_ast": 0,
        "drop_exact": 0,
        "drop_near": 0,
        "language_rebalance_removed": 0,
    }

    objective_before = Counter()
    objective_after = Counter()

    # Step 1: Filter with objective-aware AST policy.
    filtered: List[Dict[str, Any]] = []
    for row in iter_rows():
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
        if not ast_keep(row):
            metrics["drop_ast"] += 1
            continue
        filtered.append(row)

    # Step 2: Exact dedup.
    uniq: List[Dict[str, Any]] = []
    seen_fp = set()
    for row in filtered:
        fp = exact_fingerprint(row)
        if fp in seen_fp:
            metrics["drop_exact"] += 1
            continue
        seen_fp.add(fp)
        uniq.append(row)

    # Step 3: Near dedup.
    deduped: List[Dict[str, Any]] = []
    signatures: List[Tuple[int, ...]] = []
    for row in uniq:
        txt = "\n".join([str(row.get("input", "")), str(row.get("output", "")), str(row.get("context", ""))])
        sig = minhash_signature(txt)
        near_dup = any(signature_similarity(sig, other) >= 0.95 for other in signatures[-2000:])
        if near_dup:
            metrics["drop_near"] += 1
            continue
        signatures.append(sig)
        deduped.append(row)

    # Step 4: Rebalance languages.
    counts_pre_rebalance = count_lang(deduped)
    rebalanced, removed, counts_post_rebalance = rebalance_languages(deduped)
    metrics["language_rebalance_removed"] = removed

    # Step 5: Split by repo per objective.
    train, val, test, split_debug = split_by_repo(rebalanced)

    # Rewrite per-objective files with split field.
    objective_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rebalanced:
        objective_rows[str(row.get("objective", "unknown"))].append(row)

    for obj in OBJECTIVE_FILE_MAP:
        rows = objective_rows.get(obj, [])
        objective_after[obj] = len(rows)
        write_jsonl(objective_path(obj), rows)

    write_jsonl(os.path.join(HOLDOUT_ROOT, "train.jsonl"), train)
    write_jsonl(os.path.join(HOLDOUT_ROOT, "val.jsonl"), val)
    write_jsonl(os.path.join(HOLDOUT_ROOT, "test.jsonl"), test)

    per_objective: Dict[str, Any] = {}
    for obj in OBJECTIVE_FILE_MAP:
        obj_rows = objective_rows.get(obj, [])
        split_counter = Counter([str(r.get("split", "train")) for r in obj_rows])
        per_objective[obj] = {
            "total_before_filter": int(objective_before.get(obj, 0)),
            "total_after_filter": int(objective_after.get(obj, 0)),
            "train": int(split_counter.get("train", 0)),
            "val": int(split_counter.get("val", 0)),
            "test": int(split_counter.get("test", 0)),
        }

    # Fail-fast validation to catch split collapse bugs.
    for objective, stats in per_objective.items():
        total = stats["total_after_filter"]
        if total >= 3:
            if stats["train"] == 0:
                raise ValueError(f"ERROR: {objective} has ZERO training samples!")
            if stats["val"] == 0:
                raise ValueError(f"ERROR: {objective} has ZERO validation samples!")
            if stats["test"] == 0:
                raise ValueError(f"ERROR: {objective} has ZERO test samples!")

    report = {
        "generated_at": utc_now(),
        "input_source": "data/finetune/*",
        "max_records": args.max_records,
        "ast_strictness": AST_STRICTNESS,
        "filter_metrics": {
            **metrics,
            "after_filter": len(filtered),
            "after_exact_dedup": len(uniq),
            "after_near_dedup": len(deduped),
            "after_rebalance": len(rebalanced),
            "after_filter_and_dedup": len(rebalanced),
        },
        "counts_per_language_before_rebalance": counts_pre_rebalance,
        "counts_per_language_after_rebalance": counts_post_rebalance,
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "per_objective": per_objective,
        "per_objective_split_debug": split_debug,
        "outputs": {
            "train": os.path.join(HOLDOUT_ROOT, "train.jsonl"),
            "val": os.path.join(HOLDOUT_ROOT, "val.jsonl"),
            "test": os.path.join(HOLDOUT_ROOT, "test.jsonl"),
        },
        "tree_sitter_enabled": True,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
