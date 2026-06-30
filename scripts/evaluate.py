import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_DIR = os.environ.get("MODEL_DIR", "/outputs/qwen3-finetuned")
DATA_DIR = os.environ.get("DATA_DIR") or os.environ.get("AZUREML_DATA_DIR") or "/tmp/data"
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
REPORT_PATH = os.environ.get("REPORT_PATH", "/outputs/evaluation_report.json")
LOG_DIR = os.environ.get("LOG_DIR", "/outputs/logs")
MAX_SAMPLES = int(os.environ.get("EVAL_SAMPLES", "100"))
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))


def setup_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(REPORT_PATH) or ".", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(LOG_DIR, "evaluate.log"), mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger("qwen3_evaluate")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[index : index + n]) for index in range(max(len(tokens) - n + 1, 0)))


def brevity_penalty(candidate_len: int, reference_len: int) -> float:
    if candidate_len == 0:
        return 0.0
    if candidate_len > reference_len:
        return 1.0
    return pow(2.718281828459045, 1.0 - (reference_len / max(candidate_len, 1)))


def sentence_bleu(candidate: str, reference: str, max_n: int = 4, smoothing: float = 1.0) -> float:
    candidate_tokens = tokenize_text(candidate)
    reference_tokens = tokenize_text(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        candidate_ngrams = ngrams(candidate_tokens, n)
        reference_ngrams = ngrams(reference_tokens, n)
        candidate_total = max(sum(candidate_ngrams.values()), 0)
        if candidate_total == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, reference_ngrams[ngram]) for ngram, count in candidate_ngrams.items())
        precisions.append((overlap + smoothing) / (candidate_total + smoothing))

    if any(precision <= 0 for precision in precisions):
        return 0.0

    log_precision = sum(torch.log(torch.tensor(precision)).item() for precision in precisions) / max_n
    bp = brevity_penalty(len(candidate_tokens), len(reference_tokens))
    return float(bp * torch.exp(torch.tensor(log_precision)).item())


def rouge_l(candidate: str, reference: str) -> float:
    candidate_tokens = tokenize_text(candidate)
    reference_tokens = tokenize_text(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    rows = len(candidate_tokens) + 1
    cols = len(reference_tokens) + 1
    table = [[0] * cols for _ in range(rows)]
    for i, candidate_token in enumerate(candidate_tokens, start=1):
        for j, reference_token in enumerate(reference_tokens, start=1):
            if candidate_token == reference_token:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    lcs = table[-1][-1]
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_test_dataset(logger: logging.Logger):
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")
    try:
        dataset = load_dataset("json", data_files={"test": TEST_FILE})["test"]
    except Exception as exc:
        raise RuntimeError(f"Failed to load test dataset from {TEST_FILE}") from exc

    required_fields = {"input", "output", "objective"}
    missing = required_fields.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Test split is missing required fields: {sorted(missing)}")

    logger.info("Loaded test split from %s with %s rows", TEST_FILE, f"{len(dataset):,}")
    logger.info("Test schema: %s", dataset.features)
    logger.info("First test sample: %s", dataset[0])
    return dataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(logger: logging.Logger):
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Missing fine-tuned model directory: {MODEL_DIR}")

    peft_config = PeftConfig.from_pretrained(MODEL_DIR)
    base_model_name = peft_config.base_model_name_or_path
    logger.info("Base model for adapter: %s", base_model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()

    device = get_device()
    first_param_device = next(model.parameters()).device
    logger.info("Loaded fine-tuned model with LoRA adapter")
    logger.info("Model parameter device: %s", first_param_device)
    if torch.cuda.is_available():
        logger.info("✅ GPU available: True")
        logger.info("✅ GPU: %s", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        logger.info("✅ GPU memory: %.2f GB", props.total_memory / (1024**3))
        logger.info("GPU memory allocated: %.2f GB", torch.cuda.memory_allocated(0) / (1024**3))
    else:
        logger.warning("CUDA is not available; evaluation will run on CPU.")

    return model, tokenizer, device


def build_prompt(sample: Dict[str, str]) -> str:
    objective = str(sample.get("objective", ""))
    user_input = str(sample.get("input", ""))
    return (
        f"### Objective: {objective}\n"
        f"### Input:\n{user_input}\n"
        f"### Response:\n"
    )


def generate_prediction(model, tokenizer, sample: Dict[str, str], device: torch.device) -> str:
    prompt = build_prompt(sample)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=False,
    )
    encoded = encoded.to(device)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
    return normalize_text(generated_text)


@dataclass
class ObjectiveStats:
    count: int = 0
    exact_match: int = 0
    bleu_scores: List[float] = None  # type: ignore[assignment]
    rouge_scores: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.bleu_scores is None:
            self.bleu_scores = []
        if self.rouge_scores is None:
            self.rouge_scores = []


def evaluate_samples(logger: logging.Logger, model, tokenizer, dataset, device: torch.device):
    limit = min(MAX_SAMPLES, len(dataset))
    logger.info("Evaluating first %s samples from the test split", f"{limit:,}")

    overall_exact = 0
    bleu_scores: List[float] = []
    rouge_scores: List[float] = []
    objective_stats: Dict[str, ObjectiveStats] = defaultdict(ObjectiveStats)

    for index in range(limit):
        sample = dataset[index]
        try:
            predicted = generate_prediction(model, tokenizer, sample, device)
        except Exception as exc:
            raise RuntimeError(
                f"Inference failed for sample index {index} with objective={sample.get('objective')!r}"
            ) from exc

        expected = normalize_text(str(sample.get("output", "")))
        objective = str(sample.get("objective", "unknown"))
        exact_match = int(predicted == expected)
        bleu = sentence_bleu(predicted, expected)
        rouge = rouge_l(predicted, expected)

        overall_exact += exact_match
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)

        stats = objective_stats[objective]
        stats.count += 1
        stats.exact_match += exact_match
        stats.bleu_scores.append(bleu)
        stats.rouge_scores.append(rouge)

        logger.info(
            "[%03d/%03d] objective=%s exact_match=%s bleu=%.4f rouge=%.4f",
            index + 1,
            limit,
            objective,
            bool(exact_match),
            bleu,
            rouge,
        )

    report = {
        "total_samples": limit,
        "exact_match": overall_exact,
        "avg_token_overlap": round(mean(bleu_scores) if bleu_scores else 0.0, 6),
        "avg_rouge": round(mean(rouge_scores) if rouge_scores else 0.0, 6),
        "results_per_objective": {},
    }

    for objective, stats in sorted(objective_stats.items()):
        report["results_per_objective"][objective] = {
            "exact_match": stats.exact_match,
            "count": stats.count,
            "avg_token_overlap": round(mean(stats.bleu_scores) if stats.bleu_scores else 0.0, 6),
            "avg_rouge": round(mean(stats.rouge_scores) if stats.rouge_scores else 0.0, 6),
        }

    return report


def save_report(report: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(REPORT_PATH) or ".", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def print_summary(logger: logging.Logger, report: Dict[str, object]) -> None:
    total = report["total_samples"]
    exact = report["exact_match"]
    bleu = report["avg_token_overlap"]
    rouge = report["avg_rouge"]
    logger.info("=== EVALUATION SUMMARY ===")
    logger.info("Total samples: %s", f"{total:,}")
    logger.info("Exact match: %s", f"{exact:,}")
    logger.info("Avg token overlap (BLEU): %.4f", bleu)
    logger.info("Avg ROUGE-L: %.4f", rouge)
    logger.info("Report saved to %s", REPORT_PATH)
    logger.info("Results per objective:")
    for objective, stats in sorted(report["results_per_objective"].items()):
        logger.info(
            "  %s: exact_match=%s/%s, bleu=%.4f, rouge=%.4f",
            objective,
            stats["exact_match"],
            stats["count"],
            stats["avg_token_overlap"],
            stats["avg_rouge"],
        )


def main() -> int:
    logger = setup_logging()
    try:
        dataset = load_test_dataset(logger)
        model, tokenizer, device = load_model_and_tokenizer(logger)
        report = evaluate_samples(logger, model, tokenizer, dataset, device)
        save_report(report)
        print_summary(logger, report)
        return 0
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())