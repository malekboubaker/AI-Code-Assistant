#!/usr/bin/env python3
"""
Simple fine-tuning script for Azure ML.
Loads data from mounted input path.
Saves model to output path.
"""

import os
import json
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType


parser = argparse.ArgumentParser()

parser.add_argument("--training_data", type=str)
parser.add_argument("--model_output", type=str)

args = parser.parse_args()

print("Training data path:", args.training_data)
print("Model output path:", args.model_output)


def main():
    # Get paths from Azure
    # data_dir comes from mounted input: /mnt/batch/tasks/shared/LS_root/mounts/cluster_code/...
    # output_dir comes from mounted output
    
    input_path = args.training_data
    output_path = args.model_output
    
    # Fallback for local testing
    if not os.path.exists(input_path):
        input_path = "."
        output_path = "./outputs"
    
    os.makedirs(output_path, exist_ok=True)
    
    print("="*80)
    print("FINE-TUNING: Qwen2.5-Coder-Next-0.5B")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    
    # Find JSONL files (they might be in a subfolder)
    import glob
    train_files = glob.glob(f"{input_path}/**/train.jsonl", recursive=True)
    val_files = glob.glob(f"{input_path}/**/val.jsonl", recursive=True)
    
    if not train_files:
        print(f"ERROR: train.jsonl not found in {input_path}")
        print(f"Files present: {os.listdir(input_path)}")
        return
    
    train_file = train_files[0]
    val_file = val_files[0] if val_files else None
    
    print(f"Using training file: {train_file}")
    print(f"Using validation file: {val_file}")
    
    datasets = load_dataset(
        'json',
        data_files={'train': train_file, 'validation': val_file} if val_file else {'train': train_file}
    )
    
    print(f"✅ Train: {len(datasets['train'])} samples")
    if 'validation' in datasets:
        print(f"✅ Val: {len(datasets['validation'])} samples")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess
    def preprocess(batch):
        inputs = tokenizer(
            batch['input'],
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        outputs = tokenizer(
            batch['output'],
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        inputs['labels'] = outputs['input_ids']
        return inputs
    
    print("Tokenizing...")
    remove_cols = [col for col in [
        'input',
        'output',
        'objective',
        'language',
        'context',
        'metadata',
        'split',
        'id'
    ] if col in datasets['train'].column_names]

    datasets = datasets.map(
        preprocess,
        batched=True,
        remove_columns=remove_cols
    )
    
    # Load model with QLoRA
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model.config.use_cache = False

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Train
    print("\nSetting up training...")
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        learning_rate=2e-4,
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets.get('validation', None),
        tokenizer=tokenizer,
    )
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    trainer.train()
    
    # Save
    print("\nSaving model...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ Training complete! Saved to {output_path}")
    
    # Metadata
    meta = {
        "model": model_name,
        "epochs": 1,
        "train_samples": len(datasets['train']),
        "status": "complete"
    }
    with open(f"{output_path}/metadata.json", 'w') as f:
        json.dump(meta, f)

if __name__ == "__main__":
    main()