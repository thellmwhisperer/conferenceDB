PRAGMA foreign_keys = ON;

CREATE TABLE conferences (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    year INTEGER,
    start_date TEXT,
    end_date TEXT,
    location TEXT,
    description TEXT,
    source_type TEXT,
    base_path TEXT
);

CREATE TABLE speakers (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    affiliation TEXT,
    role TEXT,
    notes TEXT
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    conference_id TEXT NOT NULL REFERENCES conferences(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    day INTEGER,
    stage INTEGER,
    time_slot TEXT,
    title TEXT NOT NULL,
    session_type TEXT,
    duration_secs REAL,
    video_path TEXT,
    srt_path TEXT,
    json_path TEXT,
    txt_path TEXT,
    UNIQUE (conference_id, source_id)
);

CREATE TABLE session_speakers (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    speaker_order INTEGER NOT NULL DEFAULT 1,
    role TEXT,
    PRIMARY KEY (session_id, speaker_id)
);

CREATE TABLE segments (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT NOT NULL,
    word_count INTEGER NOT NULL
);

CREATE INDEX idx_sessions_conference_source
    ON sessions(conference_id, source_id);

CREATE INDEX idx_segments_session_time
    ON segments(session_id, start_time);

CREATE INDEX idx_session_speakers_order
    ON session_speakers(session_id, speaker_order);

CREATE VIRTUAL TABLE segments_fts
USING fts5(
    text,
    content = 'segments',
    content_rowid = 'id'
);

CREATE TRIGGER segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text)
    VALUES (new.id, new.text);
END;

CREATE TRIGGER segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO segments_fts(rowid, text)
    VALUES (new.id, new.text);
END;

CREATE TABLE embeddings (
    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (segment_id, model)
);
