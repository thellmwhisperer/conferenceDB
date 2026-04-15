#!/usr/bin/env python3
"""Classify transcript turns as speaker, audience, or uncertain."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import List, Optional, Sequence


QUESTION_STARTERS = (
    "what ",
    "when ",
    "where ",
    "who ",
    "why ",
    "how ",
    "can ",
    "could ",
    "would ",
    "should ",
    "do ",
    "did ",
    "does ",
    "is ",
    "are ",
    "will ",
)


@dataclass(frozen=True)
class SegmentMeasurement:
    index: int
    start_time: float
    end_time: float
    text: str
    lufs: float

    @property
    def duration_secs(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass(frozen=True)
class ClassifiedTurn:
    index: int
    start_time: float
    end_time: float
    duration_secs: float
    text: str
    lufs: float
    baseline_lufs: float
    local_reference_lufs: float
    delta_from_baseline: float
    delta_from_local: float
    word_count: int
    question_like: bool
    short_question_like: bool
    surrounded_by_long_turns: bool
    label: str
    score: float
    reasons: List[str]


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def infer_source_id(path: Path) -> str:
    name = path.stem
    for suffix in ("_lufs", "_roles"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_measurements(path: Path) -> List[SegmentMeasurement]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("segments") or payload.get("items") or payload.get("measurements") or []
    else:
        items = payload

    measurements: List[SegmentMeasurement] = []
    for index, item in enumerate(items, start=1):
        measurements.append(
            SegmentMeasurement(
                index=int(item.get("index") or index),
                start_time=float(item["start"]),
                end_time=float(item["end"]),
                text=normalize_whitespace(str(item.get("text", ""))),
                lufs=float(item["lufs"]),
            )
        )
    return measurements


def compute_baseline_lufs(
    measurements: Sequence[SegmentMeasurement],
    baseline_window_secs: float = 600.0,
) -> float:
    if not measurements:
        raise ValueError("Cannot compute baseline without measurements.")

    early_values = [segment.lufs for segment in measurements if segment.start_time < baseline_window_secs]
    values = early_values or [segment.lufs for segment in measurements]
    return float(median(values))


def compute_local_reference_lufs(
    measurements: Sequence[SegmentMeasurement],
    current_index: int,
    neighbor_radius: int,
    fallback: float,
) -> float:
    start = max(0, current_index - neighbor_radius)
    end = min(len(measurements), current_index + neighbor_radius + 1)
    neighbor_values = [
        measurements[index].lufs
        for index in range(start, end)
        if index != current_index
    ]
    if not neighbor_values:
        return fallback
    return float(median(neighbor_values))


def is_question_like(text: str) -> bool:
    normalized = normalize_whitespace(text).lower()
    if not normalized:
        return False
    if "?" in normalized:
        return True
    return normalized.startswith(QUESTION_STARTERS)


def has_intro_phrase(text: str) -> bool:
    normalized = normalize_whitespace(text).lower()
    if not normalized:
        return False
    return (
        normalized.startswith("i have a question")
        or normalized.startswith("can you")
        or normalized.startswith("how do you")
        or normalized.startswith("what about")
        or normalized.startswith("is there")
        or normalized.startswith("do you think")
    )


def is_short_question_like(text: str) -> bool:
    normalized = normalize_whitespace(text)
    word_count = count_words(normalized)
    if word_count == 0:
        return False
    if word_count <= 15 and is_question_like(normalized):
        return True
    return word_count <= 20 and has_intro_phrase(normalized)


def is_surrounded_by_long_turns(
    measurements: Sequence[SegmentMeasurement],
    current_index: int,
    long_turn_word_count: int = 15,
) -> bool:
    if current_index <= 0 or current_index >= len(measurements) - 1:
        return False
    previous_words = count_words(measurements[current_index - 1].text)
    next_words = count_words(measurements[current_index + 1].text)
    return previous_words >= long_turn_word_count and next_words >= long_turn_word_count


def classify_segment(
    measurement: SegmentMeasurement,
    baseline_lufs: float,
    local_reference_lufs: float,
    word_count: int,
    short_question_like: bool,
    surrounded_by_long_turns: bool,
) -> ClassifiedTurn:
    delta_from_baseline = baseline_lufs - measurement.lufs
    delta_from_local = local_reference_lufs - measurement.lufs
    question_like = is_question_like(measurement.text)
    score = 0.0
    reasons: List[str] = []
    acoustic_audience = False
    acoustic_borderline = False
    textual_audience = False
    textual_borderline = False

    if delta_from_baseline >= 6.0:
        score += 2.5
        reasons.append("large_drop_from_baseline")
        acoustic_audience = True
    elif delta_from_baseline >= 3.0:
        score += 1.0
        reasons.append("moderate_drop_from_baseline")
        acoustic_borderline = True

    if delta_from_local >= 4.0:
        score += 1.5
        reasons.append("large_drop_from_neighbors")
        acoustic_audience = True
    elif delta_from_local >= 2.0:
        score += 0.5
        reasons.append("moderate_drop_from_neighbors")
        acoustic_borderline = True

    if measurement.duration_secs <= 12.0:
        score += 0.5
        reasons.append("short_turn")

    if question_like:
        reasons.append("question_like_text")

    if short_question_like:
        score += 0.5
        reasons.append("short_question_like")
        textual_borderline = True

    if surrounded_by_long_turns:
        reasons.append("surrounded_by_long_turns")

    if short_question_like and surrounded_by_long_turns:
        textual_audience = True

    if delta_from_baseline <= 1.5:
        score -= 1.0
        reasons.append("near_session_baseline")

    if acoustic_audience and (textual_audience or short_question_like or measurement.duration_secs <= 12.0):
        label = "audience"
    elif acoustic_audience and delta_from_baseline >= 4.0:
        label = "audience"
    elif textual_audience and acoustic_borderline:
        label = "audience"
    elif textual_audience:
        label = "uncertain"
        score += 0.75
    elif score >= 1.5 or delta_from_baseline >= 3.0 or delta_from_local >= 2.5:
        label = "uncertain"
    else:
        label = "speaker"

    return ClassifiedTurn(
        index=measurement.index,
        start_time=measurement.start_time,
        end_time=measurement.end_time,
        duration_secs=measurement.duration_secs,
        text=measurement.text,
        lufs=measurement.lufs,
        baseline_lufs=baseline_lufs,
        local_reference_lufs=local_reference_lufs,
        delta_from_baseline=delta_from_baseline,
        delta_from_local=delta_from_local,
        word_count=word_count,
        question_like=question_like,
        short_question_like=short_question_like,
        surrounded_by_long_turns=surrounded_by_long_turns,
        label=label,
        score=score,
        reasons=reasons,
    )


def classify_segments(
    measurements: Sequence[SegmentMeasurement],
    baseline_window_secs: float = 600.0,
    neighbor_radius: int = 2,
) -> List[ClassifiedTurn]:
    baseline_lufs = compute_baseline_lufs(measurements, baseline_window_secs=baseline_window_secs)
    classified: List[ClassifiedTurn] = []
    for index, measurement in enumerate(measurements):
        local_reference_lufs = compute_local_reference_lufs(
            measurements,
            current_index=index,
            neighbor_radius=neighbor_radius,
            fallback=baseline_lufs,
        )
        classified.append(
            classify_segment(
                measurement,
                baseline_lufs,
                local_reference_lufs,
                word_count=count_words(measurement.text),
                short_question_like=is_short_question_like(measurement.text),
                surrounded_by_long_turns=is_surrounded_by_long_turns(measurements, index),
            )
        )
    return classified


def write_jsonl(path: Path, source_id: str, classified_turns: Sequence[ClassifiedTurn]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for turn in classified_turns:
            row = asdict(turn)
            row["source_id"] = source_id
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, source_id: str, classified_turns: Sequence[ClassifiedTurn]) -> None:
    fieldnames = [
        "source_id",
        "index",
        "start_time",
        "end_time",
        "duration_secs",
        "text",
        "lufs",
        "baseline_lufs",
        "local_reference_lufs",
        "delta_from_baseline",
        "delta_from_local",
        "question_like",
        "label",
        "score",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for turn in classified_turns:
            row = asdict(turn)
            row["source_id"] = source_id
            row["reasons"] = "|".join(turn.reasons)
            writer.writerow(row)


def summarize_labels(classified_turns: Sequence[ClassifiedTurn]) -> str:
    counts = {"speaker": 0, "audience": 0, "uncertain": 0}
    for turn in classified_turns:
        counts[turn.label] = counts.get(turn.label, 0) + 1
    return ", ".join(f"{label}={counts[label]}" for label in ("speaker", "audience", "uncertain"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lufs_json", type=Path, help="Path to the LUFS sidecar JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the sidecar. Defaults next to the input as *_roles.jsonl or .csv.",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Output format for the classified sidecar.",
    )
    parser.add_argument(
        "--source-id",
        help="Optional override for the source id stored in the sidecar.",
    )
    parser.add_argument(
        "--baseline-window-secs",
        type=float,
        default=600.0,
        help="How much early-session audio to use for the speaker baseline.",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=2,
        help="How many neighboring segments to consider for the local loudness reference.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source_id = args.source_id or infer_source_id(args.lufs_json)
    if args.output:
        output_path = args.output
    else:
        suffix = ".csv" if args.format == "csv" else ".jsonl"
        output_path = args.lufs_json.with_name(f"{source_id}_roles{suffix}")

    measurements = load_measurements(args.lufs_json)
    classified_turns = classify_segments(
        measurements,
        baseline_window_secs=args.baseline_window_secs,
        neighbor_radius=args.neighbor_radius,
    )

    if args.format == "csv":
        write_csv(output_path, source_id, classified_turns)
    else:
        write_jsonl(output_path, source_id, classified_turns)

    print(f"Wrote {output_path}")
    print(f"Baseline LUFS: {compute_baseline_lufs(measurements, args.baseline_window_secs):.1f}")
    print(f"Labels: {summarize_labels(classified_turns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
