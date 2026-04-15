# AGENTS.md — ConferenceDB Operational Contract

This file is the source of truth for any AI agent operating on this project.
Read this file completely before taking any action.

## What this project does

ConferenceDB turns conference video recordings into a searchable local knowledge base.

Input: video files (mp4) + optional metadata (conference website, speaker lists).
Output: a SQLite database with timestamped transcript segments, full-text search (FTS5), and semantic vector search (sqlite-vec).

The operator (human) provides videos and answers questions about the conference.
The agent processes them, builds the database, and then answers natural-language queries using evidence from the transcripts.

## Project structure

```
conferenceDB/
├── AGENTS.md                  ← you are here
├── CLAUDE.md                  ← points here
├── README.md                  ← for humans
├── docs/
│   └── semantic_layer.md        ← DB semantic layer (read before querying)
├── src/
│   ├── ingest.py              ← SRT → SQLite + FTS5 + embeddings
│   ├── query.py               ← search CLI (fts/vec/hybrid)
│   ├── classify_turns.py      ← speaker/audience classification (WIP)
│   ├── qa_cleaning.py         ← diff raw vs cleaned SRTs
│   └── schema.sql             ← SQLite schema
├── tests/
│   ├── conftest.py            ← adds src/ to Python path
│   └── test_*.py
├── data/                      ← gitignored
│   ├── manifests/
│   │   └── <conference>.json  ← conference metadata
│   └── <conference>.db        ← generated database
└── .venv/                     ← Python virtual environment
```

## Dependencies

Before processing any video, verify these tools are available.
Ask the operator for permission before installing anything.

### Required

| Tool | Purpose | Verify |
|------|---------|--------|
| ffmpeg | Audio extraction, format conversion | `ffmpeg -version` |
| whisper-cli | Speech-to-text transcription | `whisper-cli --help` |
| python3 | Runtime (3.10+) | `python3 --version` |

### Required Python packages (install in .venv)

| Package | Purpose | Verify |
|---------|---------|--------|
| sqlite-vec | Vector search in SQLite | `python3 -c "import sqlite_vec"` |
| sentence-transformers | Embedding generation | `python3 -c "from sentence_transformers import SentenceTransformer"` |
| pytest | Test suite | `python3 -m pytest --version` |

### Optional

| Tool | Purpose | When needed |
|------|---------|-------------|
| demucs | Vocal separation from background noise | Audio has music, crowd noise, or low SNR |

### Setting up the venv

```bash
cd <project_root>
python3 -m venv .venv
.venv/bin/pip install sqlite-vec sentence-transformers pytest
```

If demucs is needed:
```bash
.venv/bin/pip install demucs
```

Always run project scripts with `.venv/bin/python`.

## Pipeline

Process a conference end-to-end following these steps in order.
Narrate each step to the operator before executing it.

### Step 1: Gather metadata

Ask the operator for:
- Conference name, dates, location
- Video files (paths or directory)
- Conference website URL (for speaker names, session titles)
- Any speaker lists or event pages

Create a manifest at `data/manifests/<slug>.json`:
```json
{
  "conference": {
    "id": "<slug>",
    "name": "Conference Name",
    "slug": "<slug>",
    "year": 2026,
    "start_date": "2026-03-03",
    "end_date": "2026-03-04",
    "location": "City, Country",
    "description": "One-line description",
    "source_type": "local_files",
    "video_dir": "/path/to/videos",
    "output_dir": "/path/to/videos/output"
  },
  "sessions": []
}
```

Populate `sessions` from the event page or by asking the operator.
Each session needs: `id` (e.g. "day1_01"), `title`, `speakers` array, `day`, `stage`.

### Step 2: Extract audio

For each video file, extract mono 16kHz WAV for whisper:

```bash
ffmpeg -i <video>.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 <output>/<session_id>.wav
```

### Step 3: Assess audio quality (optional)

Measure integrated loudness to decide if demucs is needed:

```bash
ffmpeg -i <file>.wav -af ebur128 -f null - 2>&1 | grep "I:"
```

- Above -30 LUFS: clean enough, skip demucs
- Below -30 LUFS: consider demucs vocal separation

If demucs is needed:
```bash
.venv/bin/python -m demucs --two-stems=vocals -o <output_dir> <file>.wav
```
Use the `htdemucs/<filename>/vocals.wav` output for whisper.

### Step 4: Transcribe with whisper

```bash
whisper-cli -m <model_path> -f <audio>.wav --output-srt -of <output>/<session_id> --max-context 0 --entropy-thold 2.2
```

Recommended model: `ggml-large-v3.bin` (or whatever the operator has).
This produces `<session_id>.srt`.

### Step 5: Clean transcriptions

**Loop/hallucination removal**: Compare raw output against cleaned versions.
Save the raw whisper output as `<session_id>_raw.srt` before any edits.

```bash
cp <output>/<session_id>.srt <output>/<session_id>_raw.srt
```

**Proper noun correction**: Whisper systematically distorts unfamiliar names.
If you have a speaker list (from the manifest or conference website), search each SRT for distorted variants and correct them.

Common whisper distortion patterns:
- Phonetic substitution: "Pauly" → "Polley", "Sacks" → "Saxe"
- Name hallucination: whisper invents plausible but wrong full names
- Company names: "trailer bits" → "Trail of Bits", "Atropic" → "Anthropic"
- Tech terms: "spiffy" → "SPIFFE"

Strategy: grep each SRT for speaker introductions ("my name is", "please welcome", "I'm"), compare against the manifest, and build a correction mapping.

### Step 6: Ingest into database

```bash
.venv/bin/python src/ingest.py
```

This rebuilds the database from scratch:
1. Reads manifests from `data/manifests/`
2. Parses SRT files into segments (20-40 second chunks)
3. Populates SQLite tables + FTS5 index
4. Generates embeddings with all-MiniLM-L6-v2 (384d vectors)
5. Populates the vec0 virtual table for kNN search

To skip embeddings (faster, for iteration):
```bash
.venv/bin/python src/ingest.py --skip-embeddings
```

### Step 7: Verify

Run the test suite:
```bash
.venv/bin/python -m pytest tests/ -v
```

Test a query:
```bash
.venv/bin/python src/query.py "test query about a conference topic" --mode hybrid
```

Report to the operator: number of sessions, speakers, segments, and embedding count.

## Answering questions

Once the database is populated, the operator will ask natural-language questions about the conference content.

Read `docs/semantic_layer.md` for the full semantic layer: table schemas, query patterns, what can and cannot be inferred.

Key principles:
- **Always cite evidence**: include session ID, timestamp, and speaker when answering
- **Use the right search mode**: FTS5 for specific terms/names, vec for semantic/conceptual, SQL for structured queries (counts, lists)
- **Don't hallucinate**: if the transcripts don't contain the answer, say so
- **Timestamps are video timestamps**: the operator can jump to that point in the video to verify

## Metadata enrichment

Better metadata = better results. When processing a new conference, actively seek:

1. **Conference website**: scrape speaker names, titles, affiliations, session descriptions
2. **Speaker social profiles**: LinkedIn, Twitter handles for disambiguation
3. **Session descriptions**: match against SRT content to verify correct session-speaker mapping
4. **Related conferences**: if the operator has processed others, note cross-references

Ask the operator for these sources naturally during processing.
Don't block on them — process what you have and enrich later.

## What NOT to do

- Don't install system packages without explicit operator permission
- Don't modify video files
- Don't upload any data anywhere
- Don't delete raw SRT files — always keep `_raw.srt` originals
- Don't guess speaker identity from voice alone — use text evidence
- Don't skip the verification step
- Don't process more than one conference at a time without operator confirmation
