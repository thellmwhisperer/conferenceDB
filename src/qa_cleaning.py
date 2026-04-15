#!/usr/bin/env python3
"""Summarize cleaner changes between raw and cleaned SRT files for human QA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ingest import DEFAULT_OUTPUT_DIR, TranscriptEntry, format_hms, parse_srt_file


@dataclass(frozen=True)
class RemovedRange:
    source_id: str
    start_time: float
    end_time: float
    count: int
    samples: Tuple[str, ...]


def build_entry_key(entry: TranscriptEntry) -> Tuple[float, float, str]:
    return (entry.start_time, entry.end_time, entry.text)


def find_removed_entries(raw_entries: Sequence[TranscriptEntry], clean_entries: Sequence[TranscriptEntry]) -> List[TranscriptEntry]:
    clean_keys = {build_entry_key(entry) for entry in clean_entries}
    return [entry for entry in raw_entries if build_entry_key(entry) not in clean_keys]


def group_removed_ranges(source_id: str, removed_entries: Sequence[TranscriptEntry], gap_threshold_secs: float = 5.0) -> List[RemovedRange]:
    if not removed_entries:
        return []

    groups: List[List[TranscriptEntry]] = [[removed_entries[0]]]
    for entry in removed_entries[1:]:
        current_group = groups[-1]
        previous = current_group[-1]
        if entry.start_time - previous.end_time <= gap_threshold_secs:
            current_group.append(entry)
        else:
            groups.append([entry])

    ranges = []
    for group in groups:
        sample_texts = tuple(entry.text for entry in group[:3])
        ranges.append(
            RemovedRange(
                source_id=source_id,
                start_time=group[0].start_time,
                end_time=group[-1].end_time,
                count=len(group),
                samples=sample_texts,
            )
        )
    return ranges


def discover_source_ids(output_dir: Path) -> List[str]:
    raw_paths = sorted(output_dir.glob("*_raw.srt"))
    return [path.name.removesuffix("_raw.srt") for path in raw_paths]


def summarize_source(output_dir: Path, source_id: str, top_ranges: int) -> str:
    raw_path = output_dir / f"{source_id}_raw.srt"
    clean_path = output_dir / f"{source_id}.srt"

    if not raw_path.exists():
        return f"{source_id}: raw SRT not found at {raw_path}"
    if not clean_path.exists():
        return f"{source_id}: cleaned SRT not found at {clean_path}"

    raw_entries = parse_srt_file(raw_path)
    clean_entries = parse_srt_file(clean_path)
    removed_entries = find_removed_entries(raw_entries, clean_entries)
    removed_ranges = group_removed_ranges(source_id, removed_entries)

    lines = [
        f"{source_id}: raw={len(raw_entries)} clean={len(clean_entries)} removed={len(removed_entries)} ranges={len(removed_ranges)}"
    ]

    for removed_range in removed_ranges[:top_ranges]:
        lines.append(
            f"  - {format_hms(removed_range.start_time)} -> {format_hms(removed_range.end_time)} "
            f"({removed_range.count} blocks)"
        )
        for sample in removed_range.samples:
            lines.append(f'      "{sample}"')

    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing raw and cleaned SRT files.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Specific source id to inspect, e.g. day1_01. Can be passed multiple times.",
    )
    parser.add_argument(
        "--top-ranges",
        type=int,
        default=5,
        help="How many removed ranges to show per source.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source_ids = args.source_ids or discover_source_ids(args.output_dir)

    if not source_ids:
        print("No *_raw.srt files found.")
        return 0

    for index, source_id in enumerate(source_ids):
        if index:
            print()
        print(summarize_source(args.output_dir, source_id, args.top_ranges))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
