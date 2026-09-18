# MEDIC dataset

This project uses **MEDIC** as the primary dataset for Part 1 and Part 2.

xBD / xView2 is documented at the end of this file as an **optional** later
dataset for localisation or segmentation. It is not required to train the
classification models.

## Selection

Candidates compared before choosing MEDIC:

| Dataset | Images (published) | Classes | Classification | Localisation | Segmentation | Access | Storage | Suitability |
|---|---|---|---|---|---|---|---|---|
| MEDIC (QCRI / CrisisNLP) | 71,198 | damage severity: 3; plus 3 auxiliary tasks | Yes (image-level) | No | No | Public download, no account | ~11 GB | Primary: severity classification on real disaster photos |
| xBD / xView2 | 22,068 images; 850,736 building polygons | Joint Damage Scale: 4 (+ unclassified) | Building-level | Yes (polygons) | Polygons can be rasterised | Registration on xview2.org | ~10–11 GB challenge release; more with tier-3 | Better for localisation; blocked on login; overkill for a first classifier |
| Cao & Choe post-hurricane (IEEE DataPort) | 23,000 chips (128×128 JPEG) | damaged / undamaged | Binary only | No | No | IEEE account even for open-access files | 63 MB | Too coarse (binary) and login-gated |
| AIDER / AIDERv2 (Zenodo) | AIDER ~6.5k; AIDERv2 16,723 | disaster *type* (fire/flood/collapse/normal), not severity | Yes | No | No | CC BY 4.0, public | AIDERv2 ~1.5 GB | Wrong label granularity: event type, not damage severity |
| RescueNet | 4,494 UAV images (3000×4000) | 11 semantic classes including 4 building-damage levels | Scene + pixel | Pixel masks | Yes | Figshare / Dropbox, CC BY-NC-ND | Not verified here | Strong for segmentation; small; ND license blocks derivatives |

**Why MEDIC**

- Directly supports **damage-severity classification** (little/none, mild, severe).
- 71,198 labelled images is enough for transfer-learning experiments.
- Official train / dev / test files exist.
- Public download: no account, API key, or terms-bypass required.
- CC BY-NC-SA 4.0 is compatible with a non-commercial portfolio project.
- Fits local disk (~11 GB plus working space).

xBD is a better dataset for *building-level* assessment, but the official host
requires registration and the task is localisation + classification, which is
Part 2+/4 work. MEDIC is the correct primary dataset for this project's
classification goal.

## Official source

| Field | Value |
|---|---|
| Name | MEDIC: A Multi-Task Learning Dataset for Disaster Image Classification |
| Version | v1.0 |
| Host | Qatar Computing Research Institute (CrisisNLP) |
| Page | https://crisisnlp.qcri.org/medic/ |
| Canonical archive | https://crisisnlp.qcri.org/data/medic/MEDIC.tar.gz (~10.87 GB, Content-Length 10871170591) |
| Author mirror | https://huggingface.co/datasets/QCRI/MEDIC (same organisation, same splits) |
| Code | https://github.com/firojalam/medic/ |
| Paper | Alam et al., *Neural Computing and Applications*, 35(3):2609–2632, 2023. https://arxiv.org/abs/2108.12828 |
| License | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) |
| Authentication | None |

The Hugging Face repo `QCRI/MEDIC` is published by the dataset authors. This
project downloads it by default because it is substantially faster than the
QCRI host in practice. `python scripts/download_dataset.py --source qcri`
fetches the canonical `MEDIC.tar.gz` instead. Both reconstruct the same on-disk
layout: images at their official `image_path` plus `MEDIC_{train,dev,test}.tsv`.

## Access procedure

1. Accept CC BY-NC-SA 4.0 and the MEDIC terms of use (shipped in the QCRI
   archive as `LICENSE_CC_BY_NC_SA_4.0.txt` and `terms-of-use.txt`).
2. From the repo root, with the project virtualenv active:

   ```
   python scripts/download_dataset.py
   python scripts/verify_dataset.py
   python scripts/prepare_dataset.py
   ```

No API keys or passwords are used. Do not commit `data/raw`.

## Published size (not a measurement of our download)

The official page states **71,198 images**. Hugging Face split counts for
`QCRI/MEDIC` are train 49,353 / dev 6,157 / test 15,688 (sum 71,198). After
download, **use `reports/dataset_summary.txt`** for the numbers actually
counted on disk. Do not treat the published figures as a substitute for
verification.

## Tasks and classes

MEDIC is a **multi-task image classification** dataset. Every image has
image-level labels for four tasks. There are **no bounding boxes and no
segmentation masks**.

Primary task used by this project: **damage severity**.

| Task | Classes (encoded names used in the TSV / HF export) |
|---|---|
| `damage_severity` | `little_or_none`, `mild`, `severe` |
| `informative` | `informative`, `not_informative` |
| `humanitarian` | `affected_injured_or_dead_people`, `infrastructure_and_utility_damage`, `not_humanitarian`, `rescue_volunteering_or_donation_effort` |
| `disaster_types` | `earthquake`, `fire`, `flood`, `hurricane`, `landslide`, `not_disaster`, `other_disaster` |

Official page wording for damage severity: *Little or no damage*, *Mild
damage*, *Severe damage*. The files use underscore identifiers.

Damage severity is **ordinal** (none < mild < severe). Encoding keeps that
order so Part 3 can compute ordinal-aware errors if wanted.

## On-disk structure after download

```
data/raw/MEDIC/
├── MEDIC_train.tsv
├── MEDIC_dev.tsv
├── MEDIC_test.tsv
└── data/
    ├── aidr_disaster_types/
    ├── aidr_info/
    ├── ASONAM17_Damage_Image_Dataset/
    ├── crisismmd/
    └── multimodal-deep-learning-disaster-response-mouzannar/
```

TSV columns: `image_id`, `event_name`, `image_path`, `damage_severity`,
`informative`, `humanitarian`, `disaster_types`.

`image_path` is the unique sample key. `image_id` is a **source** identifier
(often a tweet id or dataset name) and repeats across many images.

## Official splits

MEDIC ships train / dev / test TSVs. This project **preserves them**:

- `train` → train
- `dev` → val
- `test` → test

That keeps the held-out test set comparable with published MEDIC baselines.
`configs/data_config.yaml` `split.strategy: regenerate` exists only for
ablations; it assigns **events** wholly to one split so photographs of the
same incident cannot appear on both sides of the cut.

## Machine-learning problem (Part 2 will train this)

- **Input:** one RGB photograph (social-media / ground-level disaster image),
  resized to 224×224 and ImageNet-normalised.
- **Target:** `damage_severity`.
- **Task:** 3-class image classification.
- **Labels:** class name in the TSV; integer index via
  `data/metadata/label_mapping.json`.
- **Not in scope:** localisation, segmentation, disaster-type classification
  as the main objective (the other three label columns are kept as optional
  auxiliary targets).

## Limitations (verified from the dataset description)

- Images are **social media photographs**, not satellite or nadir UAV imagery.
  Quality, resolution, compression and viewpoint vary widely.
- Labels are **image-level**, not building-level. An image tagged `severe` may
  contain both destroyed and intact structures.
- Sources include Twitter, so near-duplicates and memes/off-topic photos occur
  (the informativeness task exists for that reason).
- Class imbalance is expected; the exact ratio is measured after verification,
  not assumed here.
- License is **non-commercial share-alike**.
- Hugging Face integer labels are decoded with the published ClassLabel order,
  which is checked against the live schema before download. Do not mix a HF
  reconstruction with a QCRI tar.gz extract in the same run.

## Verified download (this repository)

Counted by `scripts/verify_dataset.py` from the files on disk after
`scripts/download_dataset.py` (source `QCRI/MEDIC` @ `main`). These are
**not** copied from the paper.

| | Count |
|---|---|
| Labelled rows / images reconstructed | 71,198 |
| Official train / dev / test TSV rows | 49,353 / 6,157 / 15,688 |
| Readable, labelled, kept after cleaning | 68,245 |
| Excluded (with recorded reason) | 2,953 |
| Missing files | 0 |
| Missing labels | 0 |
| Invalid labels | 0 |
| Unreadable (truncated JPEG) | 1 |
| Too small (min side < 32 px) | 57 |
| Exact duplicate extra copies excluded | 2,055 |
| Label-conflicting duplicate groups excluded | 171 groups / 840 images |

Damage severity on the **68,245 kept** samples:

| Class | Count | Share |
|---|---|---|
| `little_or_none` | 40,986 | 60.06% |
| `severe` | 21,447 | 31.43% |
| `mild` | 5,812 | 8.52% |

Formats among kept samples: JPEG 66,723; PNG 1,498; MPO 24.
Median size 749×562 px (range 37–8856 width, 35–6240 height).

The one unreadable file is
`data/multimodal-deep-learning-disaster-response-mouzannar/multimodal/damaged_infrastructure/images/wreckedcar_2017-02-23_21-15-25.jpg`
(truncated JPEG). It remains in `data/raw` and is listed in
`data/interim/excluded_samples.csv`.

## xBD (optional, later)

Official page: https://xview2.org/dataset  
Signup: https://xview2.org/signup  
License: CC BY-NC-SA 4.0  
Paper: https://arxiv.org/abs/1911.09296

Joint Damage Scale: `no-damage`, `minor-damage`, `major-damage`, `destroyed`
(plus unclassified polygons in the paper). Pre/post 1024×1024 RGB PNG,
building polygons in JSON.

**Registration is required.** Run `python scripts/download_xbd.py` for the
legitimate procedure. This script will not scrape or bypass xview2.org.
Third-party mirrors exist; this project does not use them to skip the official
terms.
