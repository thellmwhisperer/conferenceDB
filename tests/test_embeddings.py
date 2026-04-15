import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import sqlite_vec

from ingest import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    generate_embeddings,
    recreate_database,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "src" / "schema.sql"


@pytest.fixture
def db_with_segments(tmp_path):
    """Create a temporary DB with schema + 3 test segments."""
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
    for i in range(1, 4):
        conn.execute(
            "INSERT INTO segments (id, session_id, start_time, end_time, text, word_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, "test:s1", float(i * 10), float(i * 10 + 9), f"Segment {i} about AI security.", 5),
        )
    conn.commit()
    conn.close()
    return db_path


class FakeModel:
    def encode(self, texts, normalize_embeddings=False):
        vecs = np.array(
            [[float(hash(t) % 1000) / 1000.0] * EMBEDDING_DIM for t in texts],
            dtype="float32",
        )
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-10)
        return vecs


def test_generate_embeddings_skips_when_deps_missing(db_with_segments):
    with patch.dict("sys.modules", {"sqlite_vec": None, "sentence_transformers": None}):
        # Re-import to get the patched version
        import importlib
        import ingest
        importlib.reload(ingest)
        count = ingest.generate_embeddings(db_with_segments)
        # Restore
        importlib.reload(ingest)

    assert count == 0


@patch("sentence_transformers.SentenceTransformer", return_value=FakeModel())
def test_generate_embeddings_inserts_all_segments(mock_st, db_with_segments):
    count = generate_embeddings(db_with_segments)

    assert count == 3

    conn = sqlite3.connect(db_with_segments)
    rows = conn.execute(
        "SELECT segment_id, model, vector FROM embeddings ORDER BY segment_id"
    ).fetchall()
    conn.close()

    assert len(rows) == 3
    for row in rows:
        assert row[1] == EMBEDDING_MODEL
        assert len(row[2]) == EMBEDDING_DIM * 4  # float32 = 4 bytes each


@patch("sentence_transformers.SentenceTransformer", return_value=FakeModel())
def test_generate_embeddings_is_idempotent(mock_st, db_with_segments):
    count1 = generate_embeddings(db_with_segments)
    assert count1 == 3

    count2 = generate_embeddings(db_with_segments)
    assert count2 == 0

    conn = sqlite3.connect(db_with_segments)
    total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    assert total == 3


@patch("sentence_transformers.SentenceTransformer", return_value=FakeModel())
def test_vec_table_supports_knn_query(mock_st, db_with_segments):
    generate_embeddings(db_with_segments)

    conn = sqlite3.connect(db_with_segments)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    query_vec = np.ones(EMBEDDING_DIM, dtype="float32")
    query_vec = query_vec / np.linalg.norm(query_vec)

    results = conn.execute(
        "SELECT segment_id, distance FROM segments_vec "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT 3",
        (query_vec.tobytes(),),
    ).fetchall()
    conn.close()

    assert len(results) == 3
    assert all(isinstance(r[1], float) for r in results)
    assert results[0][1] <= results[1][1] <= results[2][1]
