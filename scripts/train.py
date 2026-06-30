import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import argparse


try:
    import bitsandbytes as bnb  # noqa: F401

    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-Next")
DATA_DIR = os.environ.get("DATA_DIR") or os.environ.get("AZUREML_DATA_DIR") or "/tmp/data"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.environ.get("AZUREML_OUTPUT_DIR") or "/outputs"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "qwen3-finetuned")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "8192"))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "100"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "50"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "500"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "500"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "3"))

parser = argparse.ArgumentParser()

parser.add_argument("--training_data", type=str)
parser.add_argument("--model_output", type=str)

args = parser.parse_args()

print("Training data path:", args.training_data)
print("Model output path:", args.model_output)


def setup_logging() -> logging.Logger:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")],
    )
    return logging.getLogger("qwen3_finetune")


def gb(value: int) -> float:
    return value / (1024**3)


def log_gpu_info(logger: logging.Logger) -> None:
    cuda_available = torch.cuda.is_available()
    logger.info("✅ GPU available: %s", cuda_available)
    if not cuda_available:
        logger.warning("CUDA is not available. Training will be extremely slow or fail.")
        return

    device_index = 0
    logger.info("✅ GPU: %s", torch.cuda.get_device_name(device_index))
    props = torch.cuda.get_device_properties(device_index)
    logger.info("✅ GPU memory: %.2f GB", gb(props.total_memory))
    logger.info(
        "GPU capability: %s.%s | multi_processor_count=%s",
        props.major,
        props.minor,
        props.multi_processor_count,
    )
    logger.info("CUDA memory allocated: %.2f GB", gb(torch.cuda.memory_allocated(device_index)))
    logger.info("CUDA memory reserved: %.2f GB", gb(torch.cuda.memory_reserved(device_index)))


def get_hf_token(logger: logging.Logger) -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN is not set. Hugging Face access will rely on public model access.")
    return token


def verify_data_files(logger: logging.Logger) -> None:
    for path in (TRAIN_FILE, VAL_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        logger.info("Found dataset file: %s", path)


def load_and_validate_datasets(logger: logging.Logger):
    verify_data_files(logger)
    try:
        datasets = load_dataset(
            "json",
            data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load datasets from {TRAIN_FILE} and {VAL_FILE}") from exc

    train_count = len(datasets["train"])
    val_count = len(datasets["validation"])
    logger.info("✅ Loaded train.jsonl (%s samples)", f"{train_count:,}")
    logger.info("✅ Loaded val.jsonl (%s samples)", f"{val_count:,}")

    required_fields = {"input", "output", "objective"}
    for split_name in ("train", "validation"):
        split = datasets[split_name]
        missing = required_fields.difference(split.column_names)
        if missing:
            raise ValueError(f"{split_name} split is missing required fields: {sorted(missing)}")
        logger.info("%s schema: %s", split_name, split.features)
        logger.info("%s first sample: %s", split_name, split[0])

    return datasets


def load_tokenizer(logger: logging.Logger, token: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_LENGTH
    logger.info("✅ Loaded tokenizer")
    return tokenizer


def build_prompt(example: dict) -> str:
    objective = str(example.get("objective", ""))
    user_input = str(example.get("input", ""))
    return (
        f"### Objective: {objective}\n"
        f"### Input:\n{user_input}\n"
        f"### Response:\n"
    )


def preprocess_batch(batch, tokenizer, logger: logging.Logger):
    input_ids = []
    attention_masks = []
    labels = []

    for index in range(len(batch["input"])):
        sample = {key: batch[key][index] for key in batch}
        try:
            prompt = build_prompt(sample)
            completion = str(sample.get("output", ""))
            full_text = prompt + completion + tokenizer.eos_token

            encoded = tokenizer(
                full_text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
            )
            prompt_tokens = tokenizer(
                prompt,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
            )

            sample_input_ids = encoded["input_ids"]
            sample_attention_mask = encoded["attention_mask"]
            sample_labels = list(sample_input_ids)

            prompt_length = min(sum(prompt_tokens["attention_mask"]), len(sample_labels))
            for position in range(prompt_length):
                sample_labels[position] = -100
            for position, mask_value in enumerate(sample_attention_mask):
                if mask_value == 0:
                    sample_labels[position] = -100

            input_ids.append(sample_input_ids)
            attention_masks.append(sample_attention_mask)
            labels.append(sample_labels)
        except Exception as exc:
            raise RuntimeError(
                f"Tokenization failed for sample with objective={sample.get('objective')!r} "
                f"and input={sample.get('input', '')[:500]!r}"
            ) from exc

    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


def tokenize_datasets(logger: logging.Logger, datasets, tokenizer):
    columns_to_remove = datasets["train"].column_names

    def map_fn(batch):
        return preprocess_batch(batch, tokenizer, logger)

    tokenized = datasets.map(
        map_fn,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing datasets",
    )
    logger.info("✅ Tokenized datasets")
    logger.info("Train dataset size: %s", f"{len(tokenized['train']):,}")
    logger.info("Val dataset size: %s", f"{len(tokenized['validation']):,}")
    return tokenized


def load_quantized_model(logger: logging.Logger, token: Optional[str]):
    use_8bit = torch.cuda.is_available() and BNB_AVAILABLE
    if use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            token=token,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        logger.info("Using 8-bit loading with bitsandbytes.")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            token=token,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Falling back to FP32 model loading on CPU.")
        elif not BNB_AVAILABLE:
            logger.warning("bitsandbytes is unavailable. Falling back to FP32 model loading.")

    if hasattr(model, "config"):
        model.config.use_cache = False

    logger.info("✅ Loaded Qwen3-Coder-Next (8-bit quantized)")
    if torch.cuda.is_available():
        logger.info("GPU memory allocated: %.2f GB", gb(torch.cuda.memory_allocated(0)))
        logger.info("GPU memory reserved: %.2f GB", gb(torch.cuda.memory_reserved(0)))
    return model


def apply_lora(logger: logging.Logger, model):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    percentage = (trainable_params / total_params * 100.0) if total_params else 0.0
    logger.info("✅ LoRA applied")
    logger.info("Trainable params: %s (%.2f%% of total)", f"{trainable_params:,}", percentage)
    logger.info("All params: %s", f"{total_params:,}")
    return model, lora_config


def create_training_args() -> TrainingArguments:
    use_cuda = torch.cuda.is_available()
    use_8bit_optim = use_cuda and BNB_AVAILABLE
    return TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        learning_rate=LEARNING_RATE,
        fp16=use_cuda,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        report_to=[],
        remove_unused_columns=False,
        save_safetensors=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_8bit_optim else "adamw_torch",
        logging_first_step=True,
    )


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_artifacts(logger: logging.Logger, trainer: Trainer, tokenizer, lora_config, training_args: TrainingArguments) -> None:
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    save_json(os.path.join(OUTPUT_DIR, "training_args.json"), training_args.to_dict())
    save_json(os.path.join(OUTPUT_DIR, "lora_config.json"), lora_config.to_dict())
    logger.info("✅ Model saved to /outputs/")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = []
    for candidate in output_path.glob("checkpoint-*"):
        if candidate.is_dir():
            try:
                step = int(candidate.name.split("-")[-1])
            except ValueError:
                continue
            checkpoints.append((step, str(candidate)))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def main() -> None:
    logger = setup_logging()
    start_time = time.monotonic()
    tokenizer = None
    model = None
    trainer = None

    try:
        log_gpu_info(logger)
        hf_token = get_hf_token(logger)
        datasets = load_and_validate_datasets(logger)
        tokenizer = load_tokenizer(logger, hf_token)
        tokenized_datasets = tokenize_datasets(logger, datasets, tokenizer)
        model = load_quantized_model(logger, hf_token)
        model, lora_config = apply_lora(logger, model)
        training_args = create_training_args()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=default_data_collator,
        )
        logger.info("✅ Trainer initialized")

        resume_checkpoint = find_latest_checkpoint(MODEL_OUTPUT_DIR)
        if resume_checkpoint:
            logger.info("Resuming from checkpoint: %s", resume_checkpoint)

        logger.info("Starting training...")
        try:
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        except torch.cuda.OutOfMemoryError:
            logger.exception(
                "CUDA out of memory. Reduce TRAIN_BATCH_SIZE, MAX_LENGTH, or enable gradient checkpointing with smaller batches."
            )
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.exception(
                    "CUDA out of memory. Reduce TRAIN_BATCH_SIZE, MAX_LENGTH, or enable gradient checkpointing with smaller batches."
                )
            raise

        logger.info("Training metrics: %s", train_result.metrics)
        save_json(os.path.join(OUTPUT_DIR, "train_metrics.json"), train_result.metrics)

        eval_metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", eval_metrics)
        save_json(os.path.join(OUTPUT_DIR, "eval_metrics.json"), eval_metrics)

        save_artifacts(logger, trainer, tokenizer, lora_config, training_args)

        elapsed = time.monotonic() - start_time
        logger.info("=== TRAINING COMPLETE ===")
        logger.info("Total time: %.2f hours", elapsed / 3600.0)
        if "loss" in train_result.metrics:
            logger.info("Final loss: %s", train_result.metrics["loss"])
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        if trainer is not None:
            try:
                fallback_checkpoint = os.path.join(MODEL_OUTPUT_DIR, "last_failed_checkpoint")
                os.makedirs(fallback_checkpoint, exist_ok=True)
                trainer.save_model(fallback_checkpoint)
                if tokenizer is not None:
                    tokenizer.save_pretrained(fallback_checkpoint)
                logger.info("Saved recovery checkpoint to %s", fallback_checkpoint)
            except Exception as save_exc:
                logger.exception("Failed to save recovery checkpoint: %s", save_exc)
        sys.exit(1)


if __name__ == "__main__":
    main()