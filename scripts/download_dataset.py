"""Download the MEDIC dataset from an official public source.

Two equivalent sources are supported. Both are published by the dataset authors,
both are public, and neither needs an account, API key or token - so this script
reads no credentials from the environment and stores none.

  hf    QCRI/MEDIC on Hugging Face, maintained by the authors' own organisation.
        Ships the images plus all four task labels and the official
        train/dev/test assignment as Parquet shards. Roughly ten times faster
        than the QCRI host in practice, which is why it is the default.

  qcri  The canonical ``MEDIC.tar.gz`` from crisisnlp.qcri.org, exactly as cited
        in the paper. Use this if you want the original archive bit-for-bit,
        including ``terms-of-use.txt`` and the licence file.

Both paths produce the same on-disk layout under ``data/raw/MEDIC``: images at
their official relative ``image_path`` plus ``MEDIC_{train,dev,test}.tsv``.

    python scripts/download_dataset.py                    # default (hf)
    python scripts/download_dataset.py --source qcri
    python scripts/download_dataset.py --source qcri --archive path/to/MEDIC.tar.gz

Both downloads are resumable: an interrupted run leaves ``.part`` files behind
and a later run continues them with HTTP Range requests.

By using the dataset you accept the CC BY-NC-SA 4.0 licence and the MEDIC terms
of use. Please cite Alam et al. (2023) as documented in docs/dataset.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import DataConfig, load_config  # noqa: E402

CHUNK_BYTES = 1 << 20  # 1 MiB


def _human(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PiB"


def remote_size(url: str, timeout: int = 60) -> int | None:
    """Content-Length advertised by the host, or None if not advertised."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! HEAD request failed ({exc}); continuing without a size check")
        return None
    length = response.headers.get("Content-Length")
    return int(length) if length is not None else None


def download(url: str, destination: Path, *, timeout: int = 60) -> Path:
    """Stream ``url`` to ``destination``, resuming a partial download if present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    total = remote_size(url, timeout=timeout)

    if destination.is_file() and total is not None and destination.stat().st_size == total:
        print(f"  archive already complete: {destination} ({_human(total)})")
        return destination

    already = partial.stat().st_size if partial.is_file() else 0
    headers = {}
    mode = "wb"
    if already:
        headers["Range"] = f"bytes={already}-"
        mode = "ab"
        print(f"  resuming from {_human(already)}")

    with requests.get(url, stream=True, headers=headers, timeout=timeout) as response:
        if already and response.status_code == 200:
            # Server ignored the Range header and restarted the body.
            already, mode = 0, "wb"
        response.raise_for_status()

        received = already
        started = time.monotonic()
        last_report = 0.0
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                handle.write(chunk)
                received += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5.0:
                    last_report = now
                    rate = (received - already) / max(now - started, 1e-6)
                    pct = f" ({100 * received / total:.1f}%)" if total else ""
                    print(
                        f"  {_human(received)}{pct} at {_human(rate)}/s",
                        flush=True,
                    )

    if total is not None and partial.stat().st_size != total:
        raise RuntimeError(
            f"download incomplete: got {partial.stat().st_size} bytes, expected {total}. "
            "Re-run the script to resume."
        )

    partial.replace(destination)
    print(f"  saved {destination} ({_human(destination.stat().st_size)})")
    return destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract(archive: Path, destination: Path) -> Path:
    """Extract ``archive`` into ``destination``, refusing paths that escape it."""
    destination.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {destination}")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise RuntimeError(f"refusing link member in archive: {member.name}")
            if not _is_within(destination, destination / member.name):
                raise RuntimeError(f"refusing member outside destination: {member.name}")
        for index, member in enumerate(members, start=1):
            tar.extract(member, destination, set_attrs=False)
            if index % 5000 == 0:
                print(f"    {index}/{len(members)} members", flush=True)
    print(f"    {len(members)}/{len(members)} members")
    return destination


# --------------------------------------------------------------------------
# Hugging Face (Parquet) source
# --------------------------------------------------------------------------

HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

# MEDIC's own split names. "dev" is the official validation split; the pipeline
# exposes it as "val" downstream but the raw TSVs keep the published name.
HF_SPLIT_FILES = {"train": "MEDIC_train.tsv", "dev": "MEDIC_dev.tsv", "test": "MEDIC_test.tsv"}
TSV_COLUMNS = [
    "image_id",
    "event_name",
    "image_path",
    "damage_severity",
    "informative",
    "humanitarian",
    "disaster_types",
]


def hf_list_parquet(repo: str, revision: str, *, timeout: int = 60) -> list[str]:
    """Relative paths of the Parquet shards in the dataset repo."""
    url = f"{HF_API}/{repo}/tree/{revision}/data"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    files = [entry["path"] for entry in response.json() if entry["path"].endswith(".parquet")]
    if not files:
        raise RuntimeError(f"no Parquet shards found under {repo}@{revision}/data")
    return sorted(files)


def _split_of(shard: str) -> str:
    """Infer the split from a shard filename such as ``data/train-00000-of-...``."""
    stem = Path(shard).name
    for split in HF_SPLIT_FILES:
        if stem.startswith(f"{split}-"):
            return split
    raise RuntimeError(f"cannot infer split from shard name {shard!r}")


def download_from_hf(config: DataConfig, *, keep_shards: bool) -> Path:
    """Rebuild the official MEDIC layout from the authors' Hugging Face repo.

    Shards are processed one at a time and deleted straight after being unpacked
    (unless ``keep_shards``), so peak disk use stays near the size of the
    extracted images rather than images plus a full second copy.
    """
    import pyarrow.parquet as pq

    repo, revision = config.dataset.hf_repo, config.dataset.hf_revision
    dataset_root = config.dataset_root
    images_written = 0
    rows_by_split: dict[str, list[tuple[str, ...]]] = {split: [] for split in HF_SPLIT_FILES}

    shards = hf_list_parquet(repo, revision)
    print(f"  {len(shards)} Parquet shards in {repo}@{revision}")
    shard_dir = config.paths.interim / "hf_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    for position, shard in enumerate(shards, start=1):
        split = _split_of(shard)
        local = shard_dir / Path(shard).name
        print(f"[{position}/{len(shards)}] {Path(shard).name} (split={split})")
        download(HF_RESOLVE.format(repo=repo, revision=revision, path=shard), local)

        table = pq.read_table(local)
        columns = {name: table.column(name).to_pylist() for name in TSV_COLUMNS if name in table.column_names}
        images = table.column("image").to_pylist()
        del table

        for index, image in enumerate(images):
            relative = columns["image_path"][index]
            target = dataset_root / relative
            if not _is_within(dataset_root, target):
                raise RuntimeError(f"refusing image_path outside dataset root: {relative!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image["bytes"])
            images_written += 1
            rows_by_split[split].append(
                tuple(str(columns[name][index]) for name in TSV_COLUMNS)
            )
        del images, columns

        if not keep_shards:
            local.unlink()
        print(f"  images written so far: {images_written}", flush=True)

    _write_label_tsvs(dataset_root, rows_by_split)
    if not keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"  reconstructed {images_written} images under {dataset_root}")
    return dataset_root


# Integer label codes in the Parquet shards map to these class names. The order
# is the ClassLabel order published in the QCRI/MEDIC dataset features and is
# asserted against the repo metadata before any decoding happens.
HF_CLASS_NAMES = {
    "damage_severity": ["little_or_none", "mild", "severe"],
    "informative": ["informative", "not_informative"],
    "humanitarian": [
        "affected_injured_or_dead_people",
        "infrastructure_and_utility_damage",
        "not_humanitarian",
        "rescue_volunteering_or_donation_effort",
    ],
    "disaster_types": [
        "earthquake",
        "flood",
        "hurricane",
        "fire",
        "landslide",
        "not_disaster",
        "other_disaster",
    ],
}


def verify_hf_class_names(repo: str, *, timeout: int = 60) -> None:
    """Fail loudly if the published ClassLabel order differs from HF_CLASS_NAMES.

    The Parquet shards store labels as integers, so decoding them with a stale
    class order would silently mislabel the whole dataset.
    """
    url = f"https://datasets-server.huggingface.co/info?dataset={repo.replace('/', '%2F')}"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        features = list(response.json()["dataset_info"].values())[0]["features"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"could not read the published feature schema for {repo} ({exc}). "
            "Refusing to decode integer labels without confirming the class order."
        ) from exc

    for task, expected in HF_CLASS_NAMES.items():
        published = features.get(task, {}).get("names")
        if published != expected:
            raise RuntimeError(
                f"class order for {task!r} changed upstream: published={published} "
                f"expected={expected}. Update HF_CLASS_NAMES before continuing."
            )
    print("  verified upstream class-label order matches HF_CLASS_NAMES")


def _write_label_tsvs(dataset_root: Path, rows_by_split: dict[str, list[tuple[str, ...]]]) -> None:
    """Write MEDIC_{train,dev,test}.tsv with integer labels decoded to names."""
    label_indices = {
        name: TSV_COLUMNS.index(name) for name in HF_CLASS_NAMES if name in TSV_COLUMNS
    }
    for split, filename in HF_SPLIT_FILES.items():
        rows = rows_by_split[split]
        path = dataset_root / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("\t".join(TSV_COLUMNS) + "\n")
            for row in rows:
                values = list(row)
                for task, index in label_indices.items():
                    values[index] = HF_CLASS_NAMES[task][int(values[index])]
                handle.write("\t".join(values) + "\n")
        print(f"  {path.name}: {len(rows)} rows")


def record_provenance(
    config: DataConfig,
    *,
    source: str,
    archive: Path | None = None,
    checksum: str | None = None,
) -> Path:
    """Write the acquisition record used to reproduce this exact download."""
    config.paths.metadata.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "dataset_name": config.dataset.name,
        "dataset_version": config.dataset.version,
        "official_page": config.dataset.official_url,
        "license": config.dataset.license,
        "source": source,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authentication_required": False,
    }
    if source == "hf":
        record["hf_repo"] = config.dataset.hf_repo
        record["hf_revision"] = config.dataset.hf_revision
        record["class_label_order"] = HF_CLASS_NAMES
    else:
        record["download_url"] = config.dataset.download_url
        record["archive_name"] = archive.name if archive else None
        record["archive_bytes"] = archive.stat().st_size if archive and archive.is_file() else None
        record["archive_sha256"] = checksum
    path = config.paths.metadata / "acquisition.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"  provenance -> {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to data config YAML")
    parser.add_argument(
        "--source",
        choices=("hf", "qcri"),
        default=None,
        help="acquisition source; defaults to dataset.source in the config",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="(qcri) use an already-downloaded archive instead of fetching it",
    )
    parser.add_argument("--no-extract", action="store_true", help="(qcri) download only")
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="(hf) keep the Parquet shards instead of deleting them after unpacking",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="compute the SHA-256 of the archive (slow on a 10 GiB file)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        default=True,
        help="keep the archive after extraction (default)",
    )
    parser.add_argument(
        "--delete-archive",
        dest="keep_archive",
        action="store_false",
        help="delete the archive after a successful extraction to save disk",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    source = args.source or config.dataset.source
    print(f"Dataset : {config.dataset.name} {config.dataset.version}")
    print(f"Licence : {config.dataset.license}")
    print(f"Source  : {source}")
    print(f"Free disk: {_human(shutil.disk_usage(config.paths.root).free)}")

    if source == "hf":
        if not config.dataset.hf_repo:
            print("error: dataset.hf_repo is not set in the config", file=sys.stderr)
            return 1
        verify_hf_class_names(config.dataset.hf_repo)
        download_from_hf(config, keep_shards=args.keep_shards)
        record_provenance(config, source=source)
    else:
        if args.archive:
            archive = Path(args.archive).expanduser().resolve()
            if not archive.is_file():
                print(f"error: archive not found: {archive}", file=sys.stderr)
                return 1
            print(f"  using local archive {archive}")
        else:
            print(f"  url: {config.dataset.download_url}")
            archive = download(config.dataset.download_url, config.archive_path)

        checksum = None
        if args.checksum:
            print("  computing sha256 ...")
            checksum = sha256(archive)
            print(f"  sha256 = {checksum}")

        if not args.no_extract:
            extract(archive, config.paths.raw)
            if not config.dataset_root.is_dir():
                print(
                    f"warning: expected {config.dataset_root} after extraction but it is missing; "
                    "check dataset.root_dirname in the config",
                    file=sys.stderr,
                )

        record_provenance(config, source=source, archive=archive, checksum=checksum)

        if not args.keep_archive and not args.no_extract and archive == config.archive_path:
            archive.unlink()
            print("  archive deleted")

    print("Done. Next: python scripts/verify_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
