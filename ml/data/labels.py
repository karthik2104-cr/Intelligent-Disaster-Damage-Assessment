"""Task definitions and label encoding.

Class lists are derived from the labels actually present in the dataset, not
hard-coded. ``EXPECTED_CLASSES`` exists only as a tripwire: it holds the class
sets published in the MEDIC feature schema, and :func:`build_label_mapping`
refuses to continue if the data disagrees with it. That way a silently changed
or truncated download surfaces as an error instead of as a mislabelled model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# The four MEDIC classification tasks. `damage_severity` is this project's
# primary target; the rest are kept for optional multi-task work in Part 2.
TASKS: tuple[str, ...] = ("damage_severity", "informative", "humanitarian", "disaster_types")

# Published class sets (QCRI/MEDIC feature schema). Order here is NOT the
# encoding order - the encoder always sorts observed classes so that the mapping
# is a deterministic function of the data alone.
EXPECTED_CLASSES: Mapping[str, frozenset[str]] = {
    "damage_severity": frozenset({"little_or_none", "mild", "severe"}),
    "informative": frozenset({"informative", "not_informative"}),
    "humanitarian": frozenset(
        {
            "affected_injured_or_dead_people",
            "infrastructure_and_utility_damage",
            "not_humanitarian",
            "rescue_volunteering_or_donation_effort",
        }
    ),
    "disaster_types": frozenset(
        {
            "earthquake",
            "flood",
            "hurricane",
            "fire",
            "landslide",
            "not_disaster",
            "other_disaster",
        }
    ),
}

# Damage severity is ordinal. Part 3 can use this for ordinal-aware metrics such
# as a weighted confusion matrix or mean absolute severity error.
DAMAGE_SEVERITY_ORDER: tuple[str, ...] = ("little_or_none", "mild", "severe")


class LabelError(ValueError):
    """Raised for unknown, missing or inconsistent labels."""


@dataclass(frozen=True)
class LabelMapping:
    """Bidirectional mapping between class names and contiguous integer indices."""

    task: str
    classes: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.classes)}

    @property
    def index_to_class(self) -> dict[int, str]:
        return dict(enumerate(self.classes))

    def encode(self, label: str) -> int:
        try:
            return self.class_to_index[label]
        except KeyError as exc:
            raise LabelError(
                f"unknown {self.task} label {label!r}; known classes: {list(self.classes)}"
            ) from exc

    def encode_all(self, labels: Iterable[str]) -> list[int]:
        return [self.encode(label) for label in labels]

    def decode(self, index: int) -> str:
        try:
            return self.classes[int(index)]
        except (IndexError, ValueError) as exc:
            raise LabelError(
                f"{self.task} index {index!r} out of range for {self.num_classes} classes"
            ) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "classes": list(self.classes),
            "num_classes": self.num_classes,
            "class_to_index": self.class_to_index,
        }


def build_label_mapping(
    task: str,
    labels: Sequence[str],
    *,
    strict: bool = True,
) -> LabelMapping:
    """Build a mapping from the labels observed in the dataset.

    Args:
        task: Task name, used for error messages and the tripwire lookup.
        labels: Every label value observed for this task.
        strict: When True and the task appears in ``EXPECTED_CLASSES``, require
            the observed classes to be a subset of the published set, and
            require the published set to be fully covered.
    """
    observed = {label for label in labels if label is not None and str(label).strip() != ""}
    if not observed:
        raise LabelError(f"no usable labels found for task {task!r}")

    if strict and task in EXPECTED_CLASSES:
        expected = EXPECTED_CLASSES[task]
        unexpected = observed - expected
        if unexpected:
            raise LabelError(
                f"task {task!r} contains labels not in the published class set: "
                f"{sorted(unexpected)}"
            )
        missing = expected - observed
        if missing:
            raise LabelError(
                f"task {task!r} is missing published classes {sorted(missing)}; "
                "the download is probably incomplete"
            )

    # Damage severity keeps its natural ordinal order so that index distance is
    # meaningful. Everything else is sorted alphabetically for determinism.
    if task == "damage_severity" and observed == set(DAMAGE_SEVERITY_ORDER):
        classes = DAMAGE_SEVERITY_ORDER
    else:
        classes = tuple(sorted(observed))
    return LabelMapping(task=task, classes=classes)


def save_label_mappings(mappings: Mapping[str, LabelMapping], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_task": "damage_severity",
        "ordinal_tasks": {"damage_severity": list(DAMAGE_SEVERITY_ORDER)},
        "tasks": {task: mapping.to_dict() for task, mapping in mappings.items()},
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_label_mappings(path: str | Path) -> dict[str, LabelMapping]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        task: LabelMapping(task=task, classes=tuple(spec["classes"]))
        for task, spec in payload["tasks"].items()
    }
