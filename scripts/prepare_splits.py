"""Prepare deterministic stratified train/validation/test JSONL files per domain."""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-synthetic", action="store_true", help="Use generated raw synthetic records when no reviewed curated file exists.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets", help="Dataset root containing per-domain folders.")
    args = parser.parse_args()
    for domain_dir in sorted(args.data_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        approved_jsonl = domain_dir / "curated" / "approved.jsonl"
        approved_csv = domain_dir / "curated" / "approved.csv"
        source = approved_jsonl if approved_jsonl.exists() else approved_csv
        synthetic_source = False
        if not source.exists():
            generated = domain_dir / "raw" / "synthetic_records.jsonl"
            if args.include_synthetic and generated.exists():
                source = generated
                synthetic_source = True
            else:
                print(f"{domain_dir.name}: skipped (add curated/approved.csv or rerun with --include-synthetic)")
                continue
        if source.suffix == ".jsonl":
            records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            import pandas as pd
            records = pd.read_csv(source).to_dict(orient="records")
        records = [r for r in records if r.get("actual_action") in {"KEEP", "DELETE", "ARCHIVE", "REVIEW", "UNSUBSCRIBE"} and r.get("item_text")]
        if len(records) < 10:
            print(f"{domain_dir.name}: skipped (need at least 10 approved, labeled records)")
            continue
        # Curated files mark human-reviewed examples. Generated candidates stay
        # explicitly unreviewed, even when the caller opts into synthetic splits.
        for record in records:
            record["reviewed"] = not synthetic_source
        groups: dict[str, list[dict]] = {}
        for record in records:
            groups.setdefault(record["actual_action"], []).append(record)
        splits = {"train": [], "validation": [], "test": []}
        rng = random.Random(17)
        for group in groups.values():
            rng.shuffle(group)
            n = len(group)
            n_train, n_validation = int(n * .8), int(n * .1)
            splits["train"].extend(group[:n_train])
            splits["validation"].extend(group[n_train:n_train + n_validation])
            splits["test"].extend(group[n_train + n_validation:])
        out = domain_dir / "splits"
        out.mkdir(exist_ok=True)
        for name, rows in splits.items():
            (out / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(f"{domain_dir.name}: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
