import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import sqlite_vec

from ingest import EMBEDDING_DIM, EMBEDDING_MODEL, generate_embeddings, recreate_database

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "src" / "schema.sql"


class DeterministicModel:
    """Embedding model that maps specific topics to distinct vector regions."""

    TOPIC_VECTORS = {
        "prompt injection": np.array([1.0, 0.0, 0.0] + [0.0] * (EMBEDDING_DIM - 3), dtype="float32"),
        "confidential computing": np.array([0.0, 1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 3), dtype="float32"),
        "incident response": np.array([0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3), dtype="float32"),
    }

    def encode(self, texts, normalize_embeddings=False):
        vecs = []
        for t in texts if isinstance(texts, list) else [texts]:
            matched = False
            for topic, vec in self.TOPIC_VECTORS.items():
                if topic in t.lower():
                    vecs.append(vec.copy())
                    matched = True
                    break
            if not matched:
                vecs.append(np.random.default_rng(hash(t) % 2**32).random(EMBEDDING_DIM).astype("float32"))
        result = np.array(vecs, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-10)
        return result


@pytest.fixture
def db_with_embeddings(tmp_path):
    db_path = tmp_path / "test.db"
    conn = recreate_database(db_path, SCHEMA_PATH)
    conn.execute(
        "INSERT INTO conferences (id, name, slug, source_type) VALUES (?, ?, ?, ?)",
        ("test", "Test Conf", "test", "local_files"),
    )
    conn.execute(
        "INSERT INTO sessions (id, conference_id, source_id, title) VALUES (?, ?, ?, ?)",
        ("test:s1", "test", "s1", "Test Session"),
    )
    segments = [
        (1, "Prompt injection is the central unsolved problem of AI security."),
        (2, "Confidential computing uses hardware enclaves for data protection."),
        (3, "Incident response teams need AI-augmented forensics tools."),
        (4, "Prompt injection payloads can be hidden in calendar invitations."),
        (5, "TEE attestation proves confidential computing is genuine."),
    ]
    for sid, text in segments:
        conn.execute(
            "INSERT INTO segments (id, session_id, start_time, end_time, text, word_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, "test:s1", float(sid * 30), float(sid * 30 + 29), text, len(text.split())),
        )
    conn.commit()
    conn.close()

    with patch("sentence_transformers.SentenceTransformer", return_value=DeterministicModel()):
        generate_embeddings(db_path)

    return db_path


def test_semantic_search_finds_related_segments(db_with_embeddings):
    from query import run_vector_search

    conn = sqlite3.connect(db_with_embeddings)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row

    model = DeterministicModel()

    results = run_vector_search(conn, "prompt injection attacks", limit=3, encode_fn=model.encode)

    assert len(results) > 0
    texts = [r["text"] for r in results]
    assert any("prompt injection" in t.lower() for t in texts)


def test_semantic_search_returns_distances(db_with_embeddings):
    from query import run_vector_search

    conn = sqlite3.connect(db_with_embeddings)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row

    model = DeterministicModel()
    results = run_vector_search(conn, "confidential computing TEE", limit=5, encode_fn=model.encode)

    assert len(results) > 0
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_semantic_search_empty_db(tmp_path):
    from query import run_vector_search

    db_path = tmp_path / "empty.db"
    conn = recreate_database(db_path, SCHEMA_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS segments_vec "
        f"USING vec0(segment_id INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIM}])"
    )
    conn.row_factory = sqlite3.Row

    model = DeterministicModel()
    results = run_vector_search(conn, "anything", limit=5, encode_fn=model.encode)
    assert results == []
