"""Upload reviewed chat JSONL and submit an optional LoRA fine-tune to Nebius.

This script can incur Nebius charges. It only submits when explicitly run by the user.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
LABELS = {"KEEP", "DELETE", "ARCHIVE", "REVIEW", "UNSUBSCRIBE"}
SYSTEM = "You are a cautious digital organization classifier. Return one JSON object with action (KEEP, DELETE, ARCHIVE, REVIEW, or UNSUBSCRIBE), confidence from 0 to 1, and a short reason. Choose REVIEW when evidence is insufficient. Never claim an item was deleted."


def load_split(data_root: Path, split: str, include_synthetic: bool = False) -> list[dict]:
    rows = []
    for path in sorted(data_root.glob(f"*/splits/{split}.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows = [r for r in rows if (r.get("reviewed", False) or (include_synthetic and r.get("synthetic", False))) and r.get("actual_action") in LABELS and r.get("item_text")]
    if not rows:
        raise SystemExit(f"No eligible {split} records. Add reviewed curated data, or explicitly use --include-synthetic with synthetic splits.")
    return rows


def format_record(row: dict) -> dict:
    target = {"action": row["actual_action"], "confidence": 0.9, "reason": str(row.get("reason", "Reviewed label."))}
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Digital asset domain: {row.get('domain', 'unknown')}\nItem details: {row['item_text']}"},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--model", default=os.getenv("NEBIUS_FINE_TUNE_BASE_MODEL", os.getenv("DECLUTTER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--include-synthetic", action="store_true", help="Train on generated synthetic splits. The resulting benchmark is synthetic, not a real-user performance claim.")
    args = parser.parse_args()
    if not os.getenv("NEBIUS_API_KEY"):
        raise SystemExit("Set NEBIUS_API_KEY in your shell environment first. Do not put it in source files.")
    train = load_split(args.data_root, "train", args.include_synthetic)
    validation = load_split(args.data_root, "validation", args.include_synthetic)
    train_path, val_path = args.output_dir / "nebius_train.jsonl", args.output_dir / "nebius_validation.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path.write_text("".join(json.dumps(format_record(r), ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
    val_path.write_text("".join(json.dumps(format_record(r), ensure_ascii=False) + "\n" for r in validation), encoding="utf-8")
    client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"), api_key=os.environ["NEBIUS_API_KEY"])
    # Explicit execution uploads the reviewed training and validation files and starts a billed job.
    training_file = client.files.create(file=train_path, purpose="fine-tune")
    validation_file = client.files.create(file=val_path, purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        validation_file=validation_file.id,
        model=args.model,
        hyperparameters={"n_epochs": args.epochs, "batch_size": args.batch_size, "lora": True},
        seed=42,
    )
    manifest = {"job_id": job.id, "model": args.model, "dataset_provenance": "synthetic" if args.include_synthetic else "human_reviewed",
                "training_file_id": training_file.id,
                "validation_file_id": validation_file.id, "training_records": len(train), "validation_records": len(validation)}
    (args.output_dir / "nebius_job.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Nebius fine-tuning job submitted: {job.id}")
    print("Job metadata saved to artifacts/nebius_job.json. Track the job in Token Factory; deploy a successful checkpoint and set DECLUTTER_FINE_TUNED_MODEL.")


if __name__ == "__main__":
    main()
