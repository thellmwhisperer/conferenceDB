# ConferenceDB

Turn conference video recordings into a searchable local knowledge base.

```
22.5 hours of video → 2,726 searchable segments → answers in seconds
```

ConferenceDB takes raw conference recordings and builds a SQLite database
with lexical search (FTS5), semantic search (sqlite-vec), and timestamped
citations back to the original video. Every answer points to a moment you
can verify.

**Agent-native by design.** The project ships an operational contract
(`AGENTS.md`) that teaches any AI agent to run the full pipeline — from raw
video to queryable knowledge base — with zero hand-holding.

## What it looks like

```bash
# Semantic — find concepts across 22 hours of talks
$ .venv/bin/python src/query.py "how agents discover vulnerabilities" --mode vec

# Lexical — exact names, quoted phrases
$ .venv/bin/python src/query.py '"Trail of Bits"' --mode fts --limit 10

# Hybrid — FTS first, vector fallback
$ .venv/bin/python src/query.py "prompt injection defenses" --mode hybrid
```

Every result includes session, speaker, and timestamp — jump straight to
that point in the video to hear it yourself.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install sqlite-vec sentence-transformers pytest
```

External tools (for processing new video):

| Tool | Purpose |
|------|---------|
| ffmpeg | Audio extraction from video |
| whisper-cli | Speech-to-text (whisper.cpp) |

## Pipeline

```
video files ──→ ffmpeg ──→ whisper-cli ──→ src/ingest.py ──→ SQLite
   .mp4          .wav         .srt           FTS5 + vec0      src/query.py
```

1. **Gather metadata** — conference, sessions, speakers → `data/manifests/<slug>.json`
2. **Extract audio** — ffmpeg to mono 16kHz WAV
3. **Transcribe** — whisper-cli to SRT, fix proper nouns
4. **Ingest** — `src/ingest.py` builds segments, FTS5 index, and 384d embeddings
5. **Query** — `src/query.py` or direct SQL/Python

## Search modes

| Mode | Flag | Best for |
|------|------|----------|
| Lexical | `--mode fts` | Exact terms, speaker names, quoted phrases |
| Semantic | `--mode vec` | Conceptual queries, paraphrases |
| Hybrid | `--mode hybrid` | General use — FTS first, vec fallback |

## Data model

```
conferences → sessions → segments (20-40s timestamped chunks)
                ↓
             speakers (via session_speakers)
```

- **segments_fts** — FTS5 full-text index (BM25 ranking)
- **segments_vec** — sqlite-vec kNN index (all-MiniLM-L6-v2, 384 dimensions)

Full schema and query patterns: [`docs/semantic_layer.md`](docs/semantic_layer.md)

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## For AI agents

This project is designed to be operated by AI agents.
Read [`AGENTS.md`](AGENTS.md) for the full operational contract.
