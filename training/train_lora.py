"""Fine-tune Qwen2.5-3B-Instruct with reviewed action/reason examples."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
VALID_ACTIONS = {"KEEP", "DELETE", "ARCHIVE", "REVIEW", "UNSUBSCRIBE"}


def load_records(root: Path, split: str) -> list[dict]:
    paths = sorted(root.glob(f"*/splits/{split}.jsonl"))
    rows = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Split files are prepared exclusively from a domain's curated/approved file.
    rows = [r for r in rows if r.get("reviewed", False) and r.get("actual_action") in VALID_ACTIONS and r.get("item_text")]
    if not rows:
        raise SystemExit(f"No approved {split} records found. Add reviewed records to datasets/<domain>/curated/approved.csv or .jsonl, then run scripts/prepare_splits.py.")
    return rows


def chat_text(tokenizer, row: dict) -> str:
    action = row["actual_action"]
    reason = str(row.get("reason", "Reviewed label."))
    messages = [
        {"role": "system", "content": "You are a cautious digital organization classifier. Return one valid JSON object with action (one of KEEP, DELETE, ARCHIVE, REVIEW, UNSUBSCRIBE), confidence (0 to 1), and a short reason. Choose REVIEW when evidence is insufficient. Never claim an item was deleted."},
        {"role": "user", "content": f"Digital asset domain: {row.get('domain', 'unknown')}\nItem details: {row['item_text']}"},
        {"role": "assistant", "content": json.dumps({"action": action, "confidence": 0.9, "reason": reason}, ensure_ascii=False)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets", type=Path)
    parser.add_argument("--output-dir", default="artifacts/qwen2.5-3b-digital-declutter-lora", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    train_rows = load_records(args.data_root, "train")
    eval_rows = load_records(args.data_root, "validation")
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training needs a CUDA GPU. Run this script in a CUDA-enabled environment such as a GPU notebook runtime.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds = Dataset.from_list([{"text": chat_text(tokenizer, row)} for row in train_rows])
    eval_ds = Dataset.from_list([{"text": chat_text(tokenizer, row)} for row in eval_rows])
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=quant, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=2e-4,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, report_to="none", save_total_limit=2,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=eval_ds,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "training_manifest.json").write_text(json.dumps({"base_model": args.model_id, "method": "QLoRA", "train_records": len(train_rows), "validation_records": len(eval_rows)}, indent=2), encoding="utf-8")
    print(f"Adapter saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
