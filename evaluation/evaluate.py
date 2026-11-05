"""Run base and fine-tuned models on the same held-out examples and save metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from backend.model_service import service

LABELS = ["KEEP", "DELETE", "ARCHIVE", "REVIEW", "UNSUBSCRIBE"]


def score(actual, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(actual, predicted), "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(), "labels": LABELS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--include-synthetic", action="store_true", help="Evaluate against generated synthetic test splits and label the report accordingly.")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.data_root.glob("*/splits/test.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows = [r for r in rows if (r.get("reviewed", False) or (args.include_synthetic and r.get("synthetic", False))) and r.get("actual_action") in LABELS]
    if not rows:
        raise SystemExit("No approved held-out records found in per-domain splits/test.jsonl files.")
    base = service.predict_batch(rows, fine_tuned=False)
    service.unload()
    tuned = service.predict_batch(rows, fine_tuned=True)
    service.unload()
    actual = [r["actual_action"] for r in rows]
    report = {"evaluation_records": len(rows), "dataset_provenance": "synthetic" if args.include_synthetic else "human_reviewed", "labels": LABELS, "base_model": score(actual, [p["action"] for p in base]),
              "fine_tuned_model": score(actual, [p["action"] for p in tuned]),
              "predictions": [{"id": r.get("id"), "domain": r.get("domain"), "actual_action": r["actual_action"], "base": b, "fine_tuned": t} for r, b, t in zip(rows, base, tuned)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {len(rows)} held-out approved records; wrote {args.output}")


if __name__ == "__main__":
    main()
