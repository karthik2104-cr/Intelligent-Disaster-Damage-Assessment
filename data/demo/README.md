# MEDIC 100-image demo subset

This folder is a **local development/demo copy** of 100 real MEDIC images.

It is **not** the training dataset, **not** a replacement for MEDIC v1.0, and
**must not** be used to retrain or re-evaluate the published models.

| | Count |
|---|---:|
| Total images | 100 |
| `little_or_none` | 34 |
| `mild` | 33 |
| `severe` | 33 |

Images were copied (not moved) from the existing downloaded MEDIC files using
the cleaned train/validation manifests (`data/processed/train.csv`,
`data/processed/val.csv`). The official test split was not sampled. Original
filenames and damage-severity labels are preserved. Content hashes are unique
(no duplicate files in this subset).

Layout:

```
data/demo/
├── README.md
├── demo_manifest.csv
├── little_or_none/
├── mild/
└── severe/
```

`demo_manifest.csv` maps each demo file back to its MEDIC `image_path`, event,
disaster type, and humanitarian label.

Do not commit the image files to GitHub (they remain MEDIC-licensed
CC BY-NC-SA 4.0). The full dataset remains under `data/raw/MEDIC/`.
