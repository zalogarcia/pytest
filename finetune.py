#!/usr/bin/env python3
"""
Fine-tune Qwen3.5-122B-A10B with LoRA using Unsloth on RunPod.

Hardware: 4x A100 80GB SXM (320GB VRAM)
Framework: Unsloth + TRL SFTTrainer
Method: LoRA on bf16 base weights

Usage:
  # Standard training run
  python3 finetune.py

  # Custom settings
  python3 finetune.py --epochs 5 --lr 2e-5 --output ./my_model

  # Resume from checkpoint
  python3 finetune.py --resume-from ./output/checkpoint-100

  # Dry run (validate data only, no training)
  python3 finetune.py --dry-run
"""

import argparse
import json
import os
import sys

# ─── CONFIG DEFAULTS ────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3.5-122B-A10B",
    "dataset_path": "maya_training_data.jsonl",
    "output_dir": "./output",
    "max_seq_length": 2048,

    # LoRA hyperparameters
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,

    # Training hyperparameters
    "num_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,  # effective batch size = 1 * 4 GPUs * 8 = 32
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "fp16": False,
    "bf16": True,
    "seed": 42,

    # Logging
    "logging_steps": 5,
    "save_steps": 50,
    "save_total_limit": 3,
}


# ─── DATA LOADING ──────────────────────────────────────────────

def load_dataset_from_jsonl(path):
    """Load JSONL training data and validate format."""
    conversations = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON on line {line_num}: {e}")
                sys.exit(1)

            messages = entry.get("messages", [])
            if not messages:
                print(f"ERROR: Line {line_num} has no messages")
                sys.exit(1)

            if messages[0]["role"] != "system":
                print(f"ERROR: Line {line_num} missing system prompt")
                sys.exit(1)

            conversations.append(entry)

    return conversations


def format_for_training(conversations, tokenizer):
    """Convert conversations to the format Unsloth/TRL expects.

    Uses the model's chat template to apply proper formatting.
    Returns a HuggingFace Dataset.
    """
    from datasets import Dataset

    formatted = []
    for conv in conversations:
        # Apply chat template -- this handles Qwen3.5's specific format
        text = tokenizer.apply_chat_template(
            conv["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


# ─── TRAINING ──────────────────────────────────────────────────

def run_training(config, resume_from=None):
    """Run the full training pipeline."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    print(f"\n{'='*60}")
    print(f"LOADING MODEL: {config['model_name']}")
    print(f"{'='*60}\n")

    # Load model with Unsloth -- handles multi-GPU automatically
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,         # auto-detect (bf16 on A100)
        load_in_4bit=False, # use full bf16 -- NOT 4bit for MoE
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=config["seed"],
    )

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load and format dataset
    print(f"\nLoading dataset from {config['dataset_path']}...")
    conversations = load_dataset_from_jsonl(config["dataset_path"])
    print(f"Loaded {len(conversations)} conversations")

    dataset = format_for_training(conversations, tokenizer)
    print(f"Formatted {len(dataset)} training examples")

    # Token length stats
    lengths = [len(tokenizer.encode(ex["text"])) for ex in dataset]
    print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

    if max(lengths) > config["max_seq_length"]:
        over = sum(1 for l in lengths if l > config["max_seq_length"])
        print(f"WARNING: {over} examples exceed max_seq_length ({config['max_seq_length']}). They will be truncated.")

    # Training config
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        seed=config["seed"],
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        packing=True,  # pack short examples together for efficiency
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Train
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['per_device_train_batch_size']} x {config['gradient_accumulation_steps']} accum")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  LoRA rank: {config['lora_r']}, alpha: {config['lora_alpha']}")
    print(f"  Output: {config['output_dir']}")
    print()

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save final LoRA adapters
    lora_path = os.path.join(config["output_dir"], "lora_adapters")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"\nLoRA adapters saved to: {lora_path}")

    return model, tokenizer, lora_path


# ─── MERGE & EXPORT ───────────────────────────────────────────

def merge_and_export(model, tokenizer, config):
    """Merge LoRA adapters into base model and save for deployment."""
    from unsloth import FastLanguageModel

    merged_path = os.path.join(config["output_dir"], "merged_bf16")
    print(f"\nMerging LoRA adapters into base model...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to: {merged_path}")

    return merged_path


def export_gguf(model, tokenizer, config):
    """Export to GGUF format for llama.cpp / Ollama deployment."""
    gguf_path = os.path.join(config["output_dir"], "gguf")
    print(f"\nExporting to GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
    print(f"GGUF model saved to: {gguf_path}")
    return gguf_path


# ─── MAIN ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5-122B-A10B with LoRA via Unsloth"
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_CONFIG["dataset_path"],
                        help="Path to training JSONL")
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Output directory for model and checkpoints")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_CONFIG["lora_r"],
                        help="LoRA rank")
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_CONFIG["per_device_train_batch_size"],
                        help="Per-device batch size")
    parser.add_argument("--max-seq-len", type=int,
                        default=DEFAULT_CONFIG["max_seq_length"],
                        help="Max sequence length")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA into base model after training")
    parser.add_argument("--gguf", action="store_true",
                        help="Export to GGUF after training (implies --merge)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset and config only, no training")
    args = parser.parse_args()

    # Build config from defaults + CLI args
    config = DEFAULT_CONFIG.copy()
    config["dataset_path"] = args.dataset
    config["output_dir"] = args.output
    config["num_epochs"] = args.epochs
    config["learning_rate"] = args.lr
    config["lora_r"] = args.lora_r
    config["lora_alpha"] = args.lora_r * 2  # keep alpha = 2x rank
    config["per_device_train_batch_size"] = args.batch_size
    config["max_seq_length"] = args.max_seq_len

    if args.gguf:
        args.merge = True

    # Dry run -- just validate data
    if args.dry_run:
        print("DRY RUN -- validating dataset only\n")
        conversations = load_dataset_from_jsonl(config["dataset_path"])
        print(f"Dataset: {len(conversations)} conversations")
        print(f"Config: {json.dumps(config, indent=2)}")

        # Check total tokens (rough estimate: 4 chars per token)
        total_chars = sum(
            sum(len(m["content"]) for m in conv["messages"])
            for conv in conversations
        )
        est_tokens = total_chars // 4
        print(f"\nEstimated total tokens: ~{est_tokens:,}")
        print(f"Estimated tokens per epoch: ~{est_tokens:,}")
        print(f"Estimated total training tokens ({config['num_epochs']} epochs): ~{est_tokens * config['num_epochs']:,}")
        print("\nDry run complete. Everything looks good.")
        return

    # Full training
    model, tokenizer, lora_path = run_training(config, resume_from=args.resume_from)

    if args.merge:
        merged_path = merge_and_export(model, tokenizer, config)

    if args.gguf:
        export_gguf(model, tokenizer, config)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  LoRA adapters: {lora_path}")
    if args.merge:
        print(f"  Merged model:  {merged_path}")
    if args.gguf:
        print(f"  GGUF export:   {os.path.join(config['output_dir'], 'gguf')}")
    print(f"\nNext steps:")
    print(f"  1. Test the model: python3 test_model.py --model {lora_path}")
    print(f"  2. Deploy via vLLM: vllm serve {lora_path if not args.merge else merged_path}")


if __name__ == "__main__":
    main()
