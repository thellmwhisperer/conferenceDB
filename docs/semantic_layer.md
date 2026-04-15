# ConferenceDB — Semantic Layer

This document describes the database schema, query patterns, and reasoning guidelines for answering questions from conference transcript data.

## Schema

### conferences

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | Unique identifier (e.g. "umprompted") |
| name | TEXT | Display name |
| slug | TEXT | URL-safe identifier |
| year | INTEGER | Year of the event |
| start_date | TEXT | ISO date |
| end_date | TEXT | ISO date |
| location | TEXT | City, venue |
| description | TEXT | One-line summary |
| source_type | TEXT | Always "local_files" for now |
| base_path | TEXT | Root path to video files |

### sessions

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | Namespaced: `conference_id:source_id` (e.g. "umprompted:day1_01") |
| conference_id | TEXT FK | → conferences.id |
| source_id | TEXT | Human-readable ID (e.g. "day1_01") |
| day | INTEGER | Conference day number |
| stage | INTEGER | Stage/room number (NULL for BoF, workshops) |
| time_slot | TEXT | e.g. "morning", "late_afternoon" |
| title | TEXT | Session title |
| session_type | TEXT | "talk" or "birds_of_a_feather" |
| duration_secs | REAL | Total duration from last transcript entry |
| video_path | TEXT | Absolute path to source video |
| srt_path | TEXT | Absolute path to SRT file |

### speakers

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | Slugified normalized name |
| display_name | TEXT | Original casing (e.g. "Josiah Peedikayil") |
| normalized_name | TEXT | Lowercase for dedup |
| affiliation | TEXT | Company/org (may be NULL) |

### session_speakers

Many-to-many join. `speaker_order` reflects the order speakers appear in the manifest.

### segments

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_id | TEXT FK | → sessions.id |
| start_time | REAL | Seconds from video start |
| end_time | REAL | Seconds from video start |
| text | TEXT | Transcript text for this segment |
| word_count | INTEGER | Word count |

Segments are 20-40 second chunks built from SRT entries, split at sentence boundaries and natural pauses.

### segments_fts (FTS5 virtual table)

Full-text search index over `segments.text`. Linked by rowid = segments.id.

### embeddings

| Column | Type | Description |
|--------|------|-------------|
| segment_id | INTEGER FK | → segments.id |
| model | TEXT | Embedding model name (e.g. "all-MiniLM-L6-v2") |
| vector | BLOB | Raw float32 bytes (384 dimensions = 1536 bytes) |

### segments_vec (vec0 virtual table)

sqlite-vec kNN index. Same vectors as `embeddings`, optimized for nearest-neighbor search.

## Query patterns

### When to use each mode

| Question type | Mode | Example |
|---------------|------|---------|
| Specific term or name | FTS5 | "What did Andrew Bullen say about prompt injection?" |
| Conceptual/semantic | vec | "How do speakers think about agent authorization?" |
| Structured/counting | SQL | "How many speakers per session?" "Which sessions have most segments?" |
| Fallback | hybrid | Any question — tries FTS5 first, falls back to vec |

### FTS5 queries (lexical)

```sql
-- Exact phrase
SELECT s.*, bm25(segments_fts) AS score
FROM segments_fts
JOIN segments s ON s.id = segments_fts.rowid
WHERE segments_fts MATCH '"prompt injection"'
ORDER BY score;

-- Multiple terms (AND)
SELECT s.*, bm25(segments_fts) AS score
FROM segments_fts
JOIN segments s ON s.id = segments_fts.rowid
WHERE segments_fts MATCH 'capability AND authorization'
ORDER BY score;

-- With session context
SELECT s.*, sess.title, sess.source_id, bm25(segments_fts) AS score
FROM segments_fts
JOIN segments s ON s.id = segments_fts.rowid
JOIN sessions sess ON sess.id = s.session_id
WHERE segments_fts MATCH '"Trail of Bits"'
ORDER BY score;
```

### Vector queries (semantic)

```sql
-- kNN search (requires sqlite-vec loaded)
SELECT knn.segment_id, knn.distance, s.text, sess.source_id, s.start_time
FROM (
    SELECT segment_id, distance
    FROM segments_vec
    WHERE embedding MATCH ? AND k = 10
) knn
JOIN segments s ON s.id = knn.segment_id
JOIN sessions sess ON sess.id = s.session_id
ORDER BY knn.distance;
```

The `?` parameter is a 1536-byte BLOB (384 float32 values). Use `query.py --mode vec` or encode with SentenceTransformer('all-MiniLM-L6-v2').

### SQL queries (structured)

```sql
-- Speakers per session
SELECT sess.source_id, sess.title, COUNT(ss.speaker_id) AS num_speakers
FROM sessions sess
JOIN session_speakers ss ON ss.session_id = sess.id
GROUP BY sess.id ORDER BY num_speakers DESC;

-- All segments by a speaker's session
SELECT s.start_time, s.end_time, s.text
FROM segments s
JOIN session_speakers ss ON ss.session_id = s.session_id
JOIN speakers sp ON sp.id = ss.speaker_id
WHERE sp.display_name = 'Dan Guido'
ORDER BY s.start_time;

-- Session duration ranking
SELECT source_id, title, ROUND(duration_secs / 60.0, 1) AS minutes
FROM sessions ORDER BY duration_secs DESC;

-- Segment count per session
SELECT sess.source_id, COUNT(*) AS segments
FROM segments s JOIN sessions sess ON sess.id = s.session_id
GROUP BY sess.id ORDER BY segments DESC;
```

## Formatting answers

When answering the operator's questions:

1. **Cite sources**: Always include session, timestamp, and speaker when available
   ```
   According to Andrew Bullen (Stripe) in day2_09 @ 0:42:52:
   "We basically de-risk prompt injection by assuming it will happen..."
   ```

2. **Timestamps are video-ready**: The operator can seek to `start_time` in the video file at `video_path` to hear the original

3. **Multiple sources**: When a topic spans sessions, show evidence from each with session context

4. **Confidence signals**:
   - Direct quote from transcript = high confidence
   - Paraphrased from multiple segments = medium confidence
   - Inferred from context = say so explicitly

## What can and cannot be inferred

### Can be answered from the data
- What was said about topic X
- Who spoke in which session
- What companies/orgs were represented
- Chronological flow of a session
- Cross-session topic coverage
- Direct quotes with timestamps

### Cannot be reliably answered
- Speaker identity within a session (segments are not diarized — we don't know which speaker said which segment)
- Audience questions vs speaker statements (classification is WIP, not in the DB yet)
- Slides or visual content (only audio was transcribed)
- Off-mic conversations, hallway track
- Speaker opinions vs statements of fact (the LLM should not editorialize)

### Edge cases
- Some speakers were never mentioned by full name in the audio — they're in the manifest but not searchable by name in FTS5
- Whisper proper noun corrections were applied but some distortions may remain
- Segment boundaries don't align with speaker changes — a segment may contain words from multiple speakers
