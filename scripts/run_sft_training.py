"""SFT training on encoded CoT data using transformers Trainer + peft LoRA.

Trains a model to reproduce encoded reasoning traces while producing correct
\\boxed{} answers, following the cipher approach from Sam Marks et al.

Usage:
    uv run python scripts/run_sft_training.py train
    uv run python scripts/run_sft_training.py train --epochs=5 --learning_rate=1e-5
"""

import os
from datetime import datetime

import fire
import torch

from src import RESULTS_PATH, utils


def tokenize_chat(example, tokenizer, max_length: int = 8192):
    """Tokenize a chat example with label masking on the prompt portion.

    Uses apply_chat_template to format messages, then masks prompt tokens
    (system + user) so the model only trains on the assistant response.
    """
    messages = example["messages"]

    # Full conversation: system + user + assistant
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Prompt only: system + user (with generation prompt to get the right boundary)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    full_tokens = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )
    prompt_tokens = tokenizer(
        prompt_text, truncation=True, max_length=max_length, add_special_tokens=False
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    prompt_len = len(prompt_tokens["input_ids"])

    # Mask prompt tokens in labels (-100 = ignore in cross-entropy loss)
    labels = list(input_ids)
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train(
    model_id: str = "vg10101/qwen3-4b-prime-mult-k5-step50",
    dataset_path: str = f"{RESULTS_PATH}/data/encoded_cot_k5_sft.jsonl",
    run_id: str | None = None,
    # LoRA config
    lora_rank: int = 32,
    lora_alpha: int = 32,
    # Training config
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    warmup_steps: int = 20,
    max_length: int = 8192,
    seed: int = 42,
):
    """Run SFT training with LoRA on encoded CoT dataset."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"

    # Run ID and output dir
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_cipher_sft"
    model_short = model_id.split("/")[-1].lower()
    output_dir = f"{RESULTS_PATH}/runs/{model_short}/{run_id}"
    adapter_dir = f"{output_dir}/adapter"

    print(f"Run ID: {run_id}")
    print(f"Output dir: {output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    raw_data = utils.read_jsonl_all(dataset_path)
    print(f"Loaded {len(raw_data)} SFT examples from {dataset_path}")

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda ex: tokenize_chat(ex, tokenizer, max_length=max_length),
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    print(f"Tokenized dataset: {len(dataset)} examples")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA config (matching project conventions from src/train/config.py)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        seed=seed,
        report_to="wandb",
        run_name=run_id,
        remove_unused_columns=False,
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save LoRA adapter
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nAdapter saved to {adapter_dir}")

    return adapter_dir


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({"train": train})
