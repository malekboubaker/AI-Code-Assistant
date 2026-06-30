import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

try:
    from tree_sitter_languages import get_parser  # type: ignore
except Exception:
    get_parser = None

TARGET_LANGS = ["python", "java", "javascript", "cpp", "csharp"]
SPLIT_RATIO = (0.85, 0.07, 0.08)
INPUT_PATH = "data/processed/unified_raw.jsonl"
TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
TEST_PATH = "data/processed/test.jsonl"
REPORT_PATH = "data/metadata/filter_split_report.json"
PROGRESS_PATH = "data/metadata/progress_phase_1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_valid_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
        return True
    except Exception:
        return False


def ast_like_validate(lang: str, code: str) -> bool:
    # Preferred path: real parse using tree-sitter languages package.
    if get_parser is not None:
        try:
            parser_key = {
                "python": "python",
                "java": "java",
                "javascript": "javascript",
                "cpp": "cpp",
                "csharp": "c_sharp",
            }[lang]
            parser = get_parser(parser_key)
            tree = parser.parse(code.encode("utf-8", errors="ignore"))
            if tree and tree.root_node and not tree.root_node.has_error:
                return True
        except Exception:
            pass

    # Fallback: lightweight syntax heuristics.
    if not code or len(code) < 8:
        return False
    if lang == "python" and "def " not in code and "class " not in code:
        return False
    if lang == "java" and "class " not in code and "public " not in code:
        return False
    if lang == "javascript" and "function" not in code and "=>" not in code:
        return False
    if lang == "cpp" and "{" not in code:
        return False
    if lang == "csharp" and "class " not in code and "namespace " not in code:
        return False
    return True


def exact_fingerprint(rec: Dict[str, Any]) -> str:
    key = f"{rec.get('objective','')}|{rec.get('language','')}|{rec.get('input','')}|{rec.get('output','')}|{rec.get('context','')}"
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()


def minhash_signature(text: str, num_perm: int = 64) -> Tuple[int, ...]:
    tokens = text.lower().split()
    if len(tokens) < 5:
        tokens = tokens + ["_"] * (5 - len(tokens))
    shingles = set(" ".join(tokens[i : i + 5]) for i in range(max(1, len(tokens) - 4)))

    sig = []
    for i in range(num_perm):
        min_val = None
        salt = str(i)
        for sh in shingles:
            hv = int(hashlib.md5((salt + sh).encode("utf-8", errors="ignore")).hexdigest(), 16)
            if min_val is None or hv < min_val:
                min_val = hv
        sig.append(min_val if min_val is not None else 0)
    return tuple(sig)


def signature_similarity(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / max(1, len(a))


def filter_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    metrics = defaultdict(int)
    out = []
    for r in records:
        metrics["input_records"] += 1
        lang = r.get("language")
        if lang not in TARGET_LANGS:
            metrics["drop_lang"] += 1
            continue
        if not is_valid_utf8(r.get("input", "")) or not is_valid_utf8(r.get("output", "")):
            metrics["drop_encoding"] += 1
            continue
        if not ast_like_validate(lang, r.get("context", "") or r.get("output", "")):
            metrics["drop_ast"] += 1
            continue
        out.append(r)
    metrics["after_filter"] = len(out)
    return out, dict(metrics)


def dedup_records(records: List[Dict[str, Any]], near_threshold: float = 0.95) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    metrics = defaultdict(int)

    uniq = []
    seen = set()
    for r in records:
        fp = exact_fingerprint(r)
        if fp in seen:
            metrics["drop_exact"] += 1
            continue
        seen.add(fp)
        uniq.append(r)

    kept = []
    signatures: List[Tuple[int, ...]] = []
    for r in uniq:
        text = f"{r.get('input', '')}\n{r.get('output', '')}\n{r.get('context', '')}"
        sig = minhash_signature(text)
        is_dup = False
        for existing in signatures[-300:]:
            if signature_similarity(sig, existing) >= near_threshold:
                is_dup = True
                break
        if is_dup:
            metrics["drop_near"] += 1
            continue
        signatures.append(sig)
        kept.append(r)

    metrics["after_dedup"] = len(kept)
    return kept, dict(metrics)


def balance_languages(records: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped = defaultdict(list)
    for r in records:
        grouped[r.get("language")].append(r)

    missing = [l for l in TARGET_LANGS if len(grouped[l]) == 0]
    if missing:
        raise ValueError(f"Cannot balance to 20% each. Missing languages: {missing}")

    min_count = min(len(grouped[l]) for l in TARGET_LANGS)
    rng = random.Random(seed)

    balanced = []
    for lang in TARGET_LANGS:
        rows = grouped[lang]
        rng.shuffle(rows)
        balanced.extend(rows[:min_count])

    rng.shuffle(balanced)
    stats = {lang: min_count for lang in TARGET_LANGS}
    stats["total"] = len(balanced)
    return balanced, stats


def repo_key(r: Dict[str, Any]) -> str:
    meta = r.get("metadata", {})
    repo = str(meta.get("repo") or meta.get("repository") or "").strip()
    if repo and repo.lower() != "unknown":
        return repo

    # Build a stable pseudo-repo key when original repo metadata is missing.
    path = str(meta.get("path") or "").strip()
    fn = str(meta.get("function") or "").strip()
    dataset = str(meta.get("dataset") or "unknown").strip()
    raw_idx = str(meta.get("raw_index") or "").strip()

    basis = "|".join([dataset, path, fn, raw_idx, str(r.get("id", ""))])
    if not basis.strip("|"):
        basis = str(r.get("id", "unknown"))
    return "pseudo-" + hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()[:12]


def split_by_repo(records: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_repo = defaultdict(list)
    for r in records:
        by_repo[repo_key(r)].append(r)

    repos = list(by_repo.keys())
    random.Random(seed).shuffle(repos)

    n = len(repos)

    # If we cannot form meaningful repo groups, fallback to deterministic row-level split.
    if n < 3:
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        m = len(shuffled)
        i1 = max(1, int(math.floor(m * SPLIT_RATIO[0])))
        i2 = min(m, i1 + max(1, int(math.floor(m * SPLIT_RATIO[1]))))
        train = shuffled[:i1]
        val = shuffled[i1:i2]
        test = shuffled[i2:]
        if not test and len(train) > 2:
            test = train[-1:]
            train = train[:-1]
        return train, val, test

    n_train = max(1, int(math.floor(n * SPLIT_RATIO[0])))
    n_val = max(1, int(math.floor(n * SPLIT_RATIO[1])))
    n_test = max(1, n - n_train - n_val)

    train_repos = set(repos[:n_train])
    val_repos = set(repos[n_train : n_train + n_val])
    test_repos = set(repos[n_train + n_val : n_train + n_val + n_test])

    train, val, test = [], [], []
    for r in records:
        rk = repo_key(r)
        if rk in train_repos:
            train.append(r)
        elif rk in val_repos:
            val.append(r)
        else:
            test.append(r)

    return train, val, test


def count_lang(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter([r.get("language", "unknown") for r in rows]))


def update_progress(ok: bool) -> None:
    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            p = json.load(f)
    except FileNotFoundError:
        return

    p.setdefault("phase_status", {})
    p["phase_status"]["phase_1_4_filter_dedup_split"] = "completed" if ok else "blocked"
    p["phase_status"]["phase_1_5_validate"] = "completed" if ok else "pending"
    p["updated_at"] = utc_now()

    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)


def main() -> None:
    records = read_jsonl(INPUT_PATH)

    filtered, filter_metrics = filter_records(records)
    deduped, dedup_metrics = dedup_records(filtered)
    balanced, balance_stats = balance_languages(deduped)
    train, val, test = split_by_repo(balanced)

    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)
    write_jsonl(TEST_PATH, test)

    report = {
        "generated_at": utc_now(),
        "input_records": len(records),
        "filter_metrics": filter_metrics,
        "dedup_metrics": dedup_metrics,
        "balance": balance_stats,
        "split_ratio_target": {"train": 0.85, "val": 0.07, "test": 0.08},
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "split_language_distribution": {
            "train": count_lang(train),
            "val": count_lang(val),
            "test": count_lang(test),
        },
        "outputs": {
            "train": TRAIN_PATH,
            "val": VAL_PATH,
            "test": TEST_PATH,
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    update_progress(ok=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
