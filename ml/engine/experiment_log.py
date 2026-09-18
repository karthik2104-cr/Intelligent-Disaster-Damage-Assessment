"""Append-only experiment log. Every number written here comes from a real run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def load_experiment_record(directory: Path, model: str) -> dict[str, Any]:
    """Load the latest on-disk experiment JSON for ``model`` (never invent metrics)."""
    matches = sorted(directory.glob(f"{model}_20*.json"))
    if not matches:
        raise FileNotFoundError(f"no experiment JSON for {model!r} under {directory}")
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def log_experiment(directory: Path, record: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **dict(record),
    }
    path = directory / "experiment_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    run_id = payload.get("experiment_id", "run")
    (directory / f"{run_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
