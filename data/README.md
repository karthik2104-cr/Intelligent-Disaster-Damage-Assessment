# Data layout

Keep raw imagery and derived files separate.

```
data/
├── raw/          # Untouched download. Never modified by the pipeline.
│   └── MEDIC/    # Created by scripts/download_dataset.py
├── interim/      # Scanned manifests, exclusion lists, resized cache
├── processed/    # train.csv / val.csv / test.csv consumed by Part 2
└── metadata/     # acquisition.json, label_mapping.json, class_weights.json, task.json
```

None of the image files in `raw/` or `interim/`, and none of the processed
CSV manifests, are committed to git. Recreate them with
`scripts/download_dataset.py`, `scripts/verify_dataset.py`, and
`scripts/prepare_dataset.py`. Metadata JSON under `data/metadata/` is tracked.

