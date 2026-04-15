#!/usr/bin/env python3
"""Ingest conference transcripts into a local SQLite database."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "umprompted.db"
DEFAULT_VIDEO_DIR = Path("/Volumes/CrucialX9/tmp/mellador")
DEFAULT_OUTPUT_DIR = DEFAULT_VIDEO_DIR / "output"
DEFAULT_FILEMAP_PATH = DEFAULT_VIDEO_DIR / "filemap.txt"
DEFAULT_SCHEMA_PATH = SRC_DIR / "schema.sql"
DEFAULT_MANIFEST_DIR = DATA_DIR / "manifests"

DEFAULT_MIN_SEGMENT_SECS = 20.0
DEFAULT_MAX_SEGMENT_SECS = 40.0
DEFAULT_PAUSE_THRESHOLD_SECS = 1.0

FALLBACK_TITLES = {
    "day1_01": "Stage 1 Late Morning — Agents in Production & Vulnerability Discovery",
    "day1_02": "Stage 1 Morning — Opening & Offensive AI",
    "day1_03": "Stage 2 Early Afternoon — Confidential AI, Synthetic Vulns",
    "day1_04": "Stage 2 Late Afternoon — Interpretability, Risk Frameworks & Physical AI",
    "day1_05": "Stage 2 Late Morning — Military AI, DFIR & GenAI Detection",
    "day1_06": "Stage 2 Morning — AI Governance",
    "day2_07": "Birds-of-a-Feather: Agentic Identity Architecture, Chris Cochran",
    "day2_08": "Stage 1 Early Afternoon — Securing GenAI at Scale",
    "day2_09": "Stage 1 Late Morning — Bug Hunting, Red Teaming & Secure Agents",
    "day2_10": "Stage 1 Morning — Opening, Offensive AI & macOS Research",
    "day2_11": "Stage 2 Early Afternoon — SOC Automation, AI Guarantees & Threat Hunting",
    "day2_12": "Stage 2 Late Morning — Secure Coding, Prompt Injection & Policy Enforcement",
    "day2_13": "Stage 2 Morning — Agent Recon, IDE Attacks & Capability Auth",
    "day3_14": "Stage 1 Late Afternoon — Threat Intel, Deception & Promptware",
    "day3_15": "Stage 2 Late Afternoon — Agentic SOC, AI Incidents & Vuln Discovery",
}


@dataclass(frozen=True)
class TranscriptEntry:
    index: int
    start_time: float
    end_time: float
    text: str


@dataclass(frozen=True)
class SpeakerAssignment:
    display_name: str
    role: str = "speaker"
    affiliation: Optional[str] = None


@dataclass(frozen=True)
class Segment:
    session_id: str
    start_time: float
    end_time: float
    text: str
    word_count: int


@dataclass(frozen=True)
class ConferenceSpec:
    conference_id: str
    name: str
    slug: str
    year: Optional[int]
    start_date: Optional[str]
    end_date: Optional[str]
    location: Optional[str]
    description: Optional[str]
    source_type: str
    video_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class SessionSpec:
    source_id: str
    conference_id: str
    day: Optional[int]
    stage: Optional[int]
    time_slot: Optional[str]
    title: str
    session_type: str
    speakers: List[SpeakerAssignment]
    artifact_stem: Optional[str] = None
    video_filename: Optional[str] = None


@dataclass(frozen=True)
class SessionBundle:
    session_id: str
    conference_id: str
    source_id: str
    day: Optional[int]
    stage: Optional[int]
    time_slot: Optional[str]
    title: str
    session_type: str
    speakers: List[SpeakerAssignment]
    duration_secs: Optional[float]
    video_path: str
    srt_path: str
    json_path: str
    txt_path: str
    segments: List[Segment]
    source_kind: Optional[str]


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "item"


def normalize_person_name(value: str) -> str:
    return normalize_whitespace(value)


def build_session_key(conference_id: str, source_id: str) -> str:
    return f"{conference_id}:{source_id}"


def parse_timecode(value: str) -> float:
    text = value.strip().replace(".", ",")
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", text)
    if not match:
        raise ValueError(f"Unsupported timecode: {value!r}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def format_hms(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


def parse_srt_content(content: str) -> List[TranscriptEntry]:
    blocks = re.split(r"\n\s*\n+", content.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip())
    entries: List[TranscriptEntry] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        if "-->" in lines[0]:
            time_line = lines[0]
            text_lines = lines[1:]
            raw_index = len(entries) + 1
        elif len(lines) >= 2 and "-->" in lines[1]:
            time_line = lines[1]
            text_lines = lines[2:]
            raw_index = int(lines[0]) if lines[0].isdigit() else len(entries) + 1
        else:
            continue

        if not text_lines:
            continue

        start_raw, end_raw = [part.strip() for part in time_line.split("-->", 1)]
        entries.append(
            TranscriptEntry(
                index=raw_index,
                start_time=parse_timecode(start_raw),
                end_time=parse_timecode(end_raw),
                text=normalize_whitespace(" ".join(text_lines)),
            )
        )

    return entries


def parse_srt_file(path: Path) -> List[TranscriptEntry]:
    return parse_srt_content(path.read_text(encoding="utf-8"))


def parse_json_file(path: Path) -> List[TranscriptEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks = payload.get("transcription") or payload.get("segments") or []
    entries: List[TranscriptEntry] = []

    for index, block in enumerate(blocks, start=1):
        text = normalize_whitespace(block.get("text", ""))
        if not text:
            continue

        timestamps = block.get("timestamps") or {}
        offsets = block.get("offsets") or {}
        start_time = None
        end_time = None

        if timestamps.get("from") and timestamps.get("to"):
            start_time = parse_timecode(timestamps["from"])
            end_time = parse_timecode(timestamps["to"])
        elif "from" in offsets and "to" in offsets:
            start_time = float(offsets["from"]) / 1000.0
            end_time = float(offsets["to"]) / 1000.0

        if start_time is None or end_time is None:
            continue

        entries.append(
            TranscriptEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        )

    return entries


def parse_whisper_log_file(path: Path) -> List[TranscriptEntry]:
    entries: List[TranscriptEntry] = []
    pattern = re.compile(
        r"^\[(\d{2}:\d{2}:\d{2}[.,]\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}[.,]\d{3})\]\s+(.*)$"
    )

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        start_raw, end_raw, text = match.groups()
        clean_text = normalize_whitespace(text)
        if not clean_text:
            continue
        entries.append(
            TranscriptEntry(
                index=len(entries) + 1,
                start_time=parse_timecode(start_raw),
                end_time=parse_timecode(end_raw),
                text=clean_text,
            )
        )

    return entries


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]*$", text.strip()))


def make_segment(session_id: str, entries: Sequence[TranscriptEntry]) -> Segment:
    text = normalize_whitespace(" ".join(entry.text for entry in entries))
    return Segment(
        session_id=session_id,
        start_time=entries[0].start_time,
        end_time=entries[-1].end_time,
        text=text,
        word_count=word_count(text),
    )


def build_segments(
    session_id: str,
    entries: Sequence[TranscriptEntry],
    min_segment_secs: float = DEFAULT_MIN_SEGMENT_SECS,
    max_segment_secs: float = DEFAULT_MAX_SEGMENT_SECS,
    pause_threshold_secs: float = DEFAULT_PAUSE_THRESHOLD_SECS,
) -> List[Segment]:
    if not entries:
        return []

    bucket: List[TranscriptEntry] = [entries[0]]
    segments: List[Segment] = []

    for entry in entries[1:]:
        current_duration = bucket[-1].end_time - bucket[0].start_time
        next_duration = entry.end_time - bucket[0].start_time
        gap = entry.start_time - bucket[-1].end_time
        should_split = False

        if current_duration >= min_segment_secs and (gap > pause_threshold_secs or ends_sentence(bucket[-1].text)):
            should_split = True
        elif gap > pause_threshold_secs and current_duration >= (min_segment_secs * 0.6) and ends_sentence(bucket[-1].text):
            should_split = True
        elif next_duration > max_segment_secs:
            should_split = True

        if should_split:
            segments.append(make_segment(session_id, bucket))
            bucket = [entry]
        else:
            bucket.append(entry)

    if bucket:
        segments.append(make_segment(session_id, bucket))

    return segments


def session_sort_key(source_id: str) -> Tuple[int, int]:
    match = re.fullmatch(r"day(\d+)_(\d+)", source_id)
    if not match:
        return (sys.maxsize, sys.maxsize)
    return (int(match.group(1)), int(match.group(2)))


def extract_day(source_id: str) -> Optional[int]:
    match = re.fullmatch(r"day(\d+)_(\d+)", source_id)
    return int(match.group(1)) if match else None


def parse_stage_and_slot(title: str) -> Tuple[Optional[int], Optional[str]]:
    match = re.match(r"Stage\s+(\d+)\s+(.+?)\s+[—-]\s+.+", title)
    if not match:
        return None, None
    stage = int(match.group(1))
    slot = re.sub(r"[^a-z0-9]+", "_", match.group(2).strip().lower()).strip("_")
    return stage, slot


def infer_session_type(title: str) -> str:
    normalized = title.lower()
    if "birds-of-a-feather" in normalized:
        return "birds_of_a_feather"
    return "talk"


def load_filemap_titles(filemap_path: Path) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    if not filemap_path.exists():
        return titles

    for raw_line in filemap_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue

        left, right = [part.strip() for part in line.split("|", 1)]
        source_id = Path(left).stem
        clean_title = Path(right).stem.strip()
        clean_title = re.sub(r"\s+", " ", clean_title).strip(" .")
        titles[source_id] = clean_title

    return titles


def choose_title(source_id: str, filemap_titles: Dict[str, str]) -> str:
    candidate = filemap_titles.get(source_id)
    fallback = FALLBACK_TITLES.get(source_id)

    if candidate and not candidate.endswith("&"):
        return candidate
    if fallback:
        return fallback
    if candidate:
        return candidate
    return source_id


def resolve_path(raw_value: Optional[str], manifest_path: Path, default: Path) -> Path:
    if raw_value is None:
        return default.resolve()
    path = Path(raw_value)
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    return path


def parse_speaker_assignments(raw_speakers: Sequence[object]) -> List[SpeakerAssignment]:
    assignments: List[SpeakerAssignment] = []
    for raw in raw_speakers:
        if isinstance(raw, str):
            name = normalize_person_name(raw)
            if name:
                assignments.append(SpeakerAssignment(display_name=name))
            continue

        if isinstance(raw, dict):
            name = normalize_person_name(str(raw.get("display_name") or raw.get("name") or ""))
            if not name:
                continue
            assignments.append(
                SpeakerAssignment(
                    display_name=name,
                    role=normalize_whitespace(str(raw.get("role") or "speaker")),
                    affiliation=normalize_whitespace(str(raw.get("affiliation"))) if raw.get("affiliation") else None,
                )
            )

    return assignments


def load_manifest(
    path: Path,
    default_video_dir: Path,
    default_output_dir: Path,
) -> Tuple[ConferenceSpec, List[SessionSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    conference_payload = payload.get("conference") or {}
    conference_id = str(conference_payload.get("id") or path.stem)
    conference_name = str(conference_payload.get("name") or conference_id)
    conference = ConferenceSpec(
        conference_id=conference_id,
        name=conference_name,
        slug=str(conference_payload.get("slug") or slugify(conference_name)),
        year=conference_payload.get("year"),
        start_date=conference_payload.get("start_date"),
        end_date=conference_payload.get("end_date"),
        location=conference_payload.get("location"),
        description=conference_payload.get("description"),
        source_type=str(conference_payload.get("source_type") or "local_files"),
        video_dir=resolve_path(conference_payload.get("video_dir"), path, default_video_dir),
        output_dir=resolve_path(conference_payload.get("output_dir"), path, default_output_dir),
    )

    sessions: List[SessionSpec] = []
    for session_payload in payload.get("sessions") or []:
        source_id = str(session_payload["id"])
        title = normalize_whitespace(str(session_payload.get("title") or source_id))
        stage = session_payload.get("stage")
        time_slot = session_payload.get("time_slot")
        if stage is None or time_slot is None:
            inferred_stage, inferred_slot = parse_stage_and_slot(title)
            if stage is None:
                stage = inferred_stage
            if time_slot is None:
                time_slot = inferred_slot

        sessions.append(
            SessionSpec(
                source_id=source_id,
                conference_id=conference_id,
                day=session_payload.get("day", extract_day(source_id)),
                stage=stage,
                time_slot=time_slot,
                title=title,
                session_type=str(session_payload.get("session_type") or infer_session_type(title)),
                speakers=parse_speaker_assignments(session_payload.get("speakers") or []),
                artifact_stem=session_payload.get("artifact_stem"),
                video_filename=session_payload.get("video_filename"),
            )
        )

    return conference, sessions


def discover_source_ids(output_dir: Path) -> List[str]:
    source_ids = set()
    if not output_dir.exists():
        return []

    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".srt", ".json", ".txt", ".log", ".wav"}:
            continue
        source_ids.add(path.stem)

    return sorted(source_ids, key=session_sort_key)


def build_default_specs(
    video_dir: Path,
    output_dir: Path,
    filemap_path: Path,
) -> List[Tuple[ConferenceSpec, List[SessionSpec]]]:
    filemap_titles = load_filemap_titles(filemap_path)
    source_ids = sorted(
        set(FALLBACK_TITLES) | set(filemap_titles) | set(discover_source_ids(output_dir)),
        key=session_sort_key,
    )
    conference = ConferenceSpec(
        conference_id="umprompted",
        name="Umprompted",
        slug="umprompted",
        year=None,
        start_date=None,
        end_date=None,
        location=None,
        description="Default conference manifest inferred from local file mappings.",
        source_type="local_files",
        video_dir=video_dir.resolve(),
        output_dir=output_dir.resolve(),
    )
    sessions = []
    for source_id in source_ids:
        title = choose_title(source_id, filemap_titles)
        stage, time_slot = parse_stage_and_slot(title)
        sessions.append(
            SessionSpec(
                source_id=source_id,
                conference_id=conference.conference_id,
                day=extract_day(source_id),
                stage=stage,
                time_slot=time_slot,
                title=title,
                session_type=infer_session_type(title),
                speakers=[],
            )
        )
    return [(conference, sessions)]


def load_conference_specs(
    manifest_dir: Path,
    video_dir: Path,
    output_dir: Path,
    filemap_path: Path,
) -> List[Tuple[ConferenceSpec, List[SessionSpec]]]:
    manifest_paths = sorted(manifest_dir.glob("*.json")) if manifest_dir.exists() else []
    if not manifest_paths:
        return build_default_specs(video_dir, output_dir, filemap_path)

    specs = []
    for manifest_path in manifest_paths:
        specs.append(load_manifest(manifest_path, video_dir, output_dir))
    return specs


def load_transcript_entries(output_dir: Path, artifact_stem: str) -> Tuple[List[TranscriptEntry], Optional[str]]:
    srt_path = output_dir / f"{artifact_stem}.srt"
    json_path = output_dir / f"{artifact_stem}.json"
    log_path = output_dir / f"{artifact_stem}.log"

    if srt_path.exists():
        return parse_srt_file(srt_path), "srt"
    if json_path.exists():
        return parse_json_file(json_path), "json"
    if log_path.exists():
        return parse_whisper_log_file(log_path), "log"
    return [], None


def build_session_bundle(conference: ConferenceSpec, session: SessionSpec) -> SessionBundle:
    artifact_stem = session.artifact_stem or session.source_id
    video_filename = session.video_filename or f"{artifact_stem}.mp4"
    internal_session_id = build_session_key(conference.conference_id, session.source_id)

    entries, source_kind = load_transcript_entries(conference.output_dir, artifact_stem)
    segments = build_segments(internal_session_id, entries)
    duration_secs = entries[-1].end_time if entries else None

    return SessionBundle(
        session_id=internal_session_id,
        conference_id=conference.conference_id,
        source_id=session.source_id,
        day=session.day,
        stage=session.stage,
        time_slot=session.time_slot,
        title=session.title,
        session_type=session.session_type,
        speakers=session.speakers,
        duration_secs=duration_secs,
        video_path=str((conference.video_dir / video_filename).resolve()),
        srt_path=str((conference.output_dir / f"{artifact_stem}.srt").resolve()),
        json_path=str((conference.output_dir / f"{artifact_stem}.json").resolve()),
        txt_path=str((conference.output_dir / f"{artifact_stem}.txt").resolve()),
        segments=segments,
        source_kind=source_kind,
    )


def build_dataset(
    manifest_dir: Path,
    video_dir: Path,
    output_dir: Path,
    filemap_path: Path,
) -> Tuple[List[ConferenceSpec], List[SessionBundle]]:
    conference_specs = load_conference_specs(manifest_dir, video_dir, output_dir, filemap_path)
    conferences: List[ConferenceSpec] = []
    bundles: List[SessionBundle] = []

    for conference, sessions in conference_specs:
        conferences.append(conference)
        for session in sessions:
            bundles.append(build_session_bundle(conference, session))

    return conferences, bundles


def recreate_database(db_path: Path, schema_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(schema_path.read_text(encoding="utf-8"))
    return connection


def insert_data(
    connection: sqlite3.Connection,
    conferences: Sequence[ConferenceSpec],
    bundles: Sequence[SessionBundle],
) -> None:
    conference_rows = [
        (
            conference.conference_id,
            conference.name,
            conference.slug,
            conference.year,
            conference.start_date,
            conference.end_date,
            conference.location,
            conference.description,
            conference.source_type,
            str(conference.video_dir),
        )
        for conference in conferences
    ]

    session_rows = [
        (
            bundle.session_id,
            bundle.conference_id,
            bundle.source_id,
            bundle.day,
            bundle.stage,
            bundle.time_slot,
            bundle.title,
            bundle.session_type,
            bundle.duration_secs,
            bundle.video_path,
            bundle.srt_path,
            bundle.json_path,
            bundle.txt_path,
        )
        for bundle in bundles
    ]

    speaker_map: Dict[str, Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]] = {}
    session_speaker_rows = []
    for bundle in bundles:
        for speaker_order, assignment in enumerate(bundle.speakers, start=1):
            normalized_name = normalize_person_name(assignment.display_name).lower()
            speaker_id = slugify(normalized_name)
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = (
                    speaker_id,
                    assignment.display_name,
                    normalized_name,
                    assignment.affiliation,
                    None,
                    None,
                )
            session_speaker_rows.append(
                (
                    bundle.session_id,
                    speaker_id,
                    speaker_order,
                    assignment.role,
                )
            )

    segment_rows = [
        (
            segment.session_id,
            segment.start_time,
            segment.end_time,
            segment.text,
            segment.word_count,
        )
        for bundle in bundles
        for segment in bundle.segments
    ]

    connection.executemany(
        """
        INSERT INTO conferences (
            id, name, slug, year, start_date, end_date,
            location, description, source_type, base_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        conference_rows,
    )
    connection.executemany(
        """
        INSERT INTO sessions (
            id, conference_id, source_id, day, stage, time_slot,
            title, session_type, duration_secs, video_path,
            srt_path, json_path, txt_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        session_rows,
    )
    if speaker_map:
        connection.executemany(
            """
            INSERT INTO speakers (
                id, display_name, normalized_name, affiliation, role, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            list(speaker_map.values()),
        )
    if session_speaker_rows:
        connection.executemany(
            """
            INSERT INTO session_speakers (
                session_id, speaker_id, speaker_order, role
            ) VALUES (?, ?, ?, ?)
            """,
            session_speaker_rows,
        )
    if segment_rows:
        connection.executemany(
            """
            INSERT INTO segments (
                session_id, start_time, end_time, text, word_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            segment_rows,
        )
    connection.commit()


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64


def generate_embeddings(
    db_path: Path,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> int:
    """Generate and store embeddings for all segments missing them."""
    try:
        import sqlite_vec
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"Skipping embeddings: {exc}")
        return 0

    connection = sqlite3.connect(db_path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)

    connection.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS segments_vec
        USING vec0(segment_id INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIM}])
        """
    )

    rows = connection.execute(
        """
        SELECT s.id, s.text
        FROM segments s
        WHERE s.id NOT IN (SELECT segment_id FROM embeddings WHERE model = ?)
        ORDER BY s.id
        """,
        (model_name,),
    ).fetchall()

    if not rows:
        print(f"All segments already have {model_name} embeddings.")
        connection.close()
        return 0

    print(f"Generating {model_name} embeddings for {len(rows)} segments...")
    model = SentenceTransformer(model_name)
    count = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        ids = [row[0] for row in batch]
        texts = [row[1] for row in batch]
        vectors = model.encode(texts, normalize_embeddings=True)

        connection.executemany(
            "INSERT INTO embeddings (segment_id, model, vector) VALUES (?, ?, ?)",
            [(sid, model_name, vec.tobytes()) for sid, vec in zip(ids, vectors)],
        )
        connection.executemany(
            "INSERT OR REPLACE INTO segments_vec (segment_id, embedding) VALUES (?, ?)",
            [(sid, vec.tobytes()) for sid, vec in zip(ids, vectors)],
        )
        count += len(batch)
        print(f"  {count}/{len(rows)}", end="\r")

    connection.commit()
    connection.close()
    print(f"\nEmbedded {count} segments with {model_name}.")
    return count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Default transcript directory when a manifest does not specify one.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Default video directory when a manifest does not specify one.")
    parser.add_argument("--filemap-path", type=Path, default=DEFAULT_FILEMAP_PATH, help="Fallback metadata mapping file.")
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH, help="Schema SQL file.")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR, help="Directory containing conference manifest JSON files.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip vector embedding generation.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    conferences, bundles = build_dataset(
        args.manifest_dir,
        args.video_dir,
        args.output_dir,
        args.filemap_path,
    )

    connection = recreate_database(args.db_path, args.schema_path)
    try:
        insert_data(connection, conferences, bundles)
    finally:
        connection.close()

    session_count = len(bundles)
    conference_count = len(conferences)
    speaker_count = len(
        {
            slugify(normalize_person_name(assignment.display_name).lower())
            for bundle in bundles
            for assignment in bundle.speakers
        }
    )
    segment_count = sum(len(bundle.segments) for bundle in bundles)
    sources: Dict[str, int] = {}
    for bundle in bundles:
        if bundle.source_kind:
            sources[bundle.source_kind] = sources.get(bundle.source_kind, 0) + 1

    print(
        f"Built {args.db_path} with {conference_count} conferences, "
        f"{session_count} sessions, {speaker_count} speakers, and {segment_count} segments."
    )
    if sources:
        details = ", ".join(f"{kind}={count}" for kind, count in sorted(sources.items()))
        print(f"Transcript sources used: {details}")
    else:
        print("No transcript files found yet; sessions were created from manifest metadata only.")

    if not args.skip_embeddings:
        generate_embeddings(args.db_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
