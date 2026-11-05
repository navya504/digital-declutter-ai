# Downloads dataset

This domain has its own `raw/`, `curated/`, and `splits/` directories. The generator writes synthetic examples to `raw/`. Human-review and approve examples before placing them in `curated/` for fine-tuning.

Fields: `id`, `domain`, `item_text`, `actual_action`, `reason`, `synthetic`. Actions: KEEP, DELETE, ARCHIVE, REVIEW, UNSUBSCRIBE.
