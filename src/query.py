#!/usr/bin/env python3
"""Query the local conference index using lexical (FTS5) and semantic (sqlite-vec) search."""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ingest import DEFAULT_DB_PATH, EMBEDDING_MODEL, format_hms

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "for",
    "from",
    "get",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "said",
    "say",
    "says",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def truncate(text: str, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def extract_terms(query: str) -> List[str]:
    phrases = [phrase.strip() for phrase in re.findall(r'"([^"]+)"', query) if phrase.strip()]
    stripped = re.sub(r'"[^"]+"', " ", query.lower())
    words = re.findall(r"[a-z0-9][a-z0-9_-]*", stripped)
    filtered = [word for word in words if word not in STOPWORDS]

    terms = [f'"{phrase.replace(chr(34), " ").strip()}"' for phrase in phrases]
    terms.extend(filtered)

    if terms:
        return terms

    fallback_words = re.findall(r"[a-z0-9][a-z0-9_-]*", query.lower())
    return fallback_words


def build_match_queries(query: str) -> Tuple[str, Optional[str], List[str]]:
    terms = extract_terms(query)
    if not terms:
        raise ValueError("Query did not contain any searchable terms.")

    primary = " AND ".join(terms)
    fallback = " OR ".join(terms) if len(terms) > 1 else None
    return primary, fallback, terms


def run_search(
    connection: sqlite3.Connection,
    match_query: str,
    limit: int,
    conference_filter: Optional[str] = None,
) -> List[sqlite3.Row]:
    where_clauses = ["segments_fts MATCH ?"]
    params: List[object] = [match_query]

    if conference_filter:
        where_clauses.append("(conf.id = ? OR conf.slug = ?)")
        params.extend([conference_filter, conference_filter])

    params.append(limit)
    sql = f"""
        SELECT
            sess.source_id,
            sess.title,
            sess.session_type,
            s.start_time,
            s.end_time,
            s.text,
            sess.video_path,
            conf.name AS conference_name,
            conf.slug AS conference_slug,
            bm25(segments_fts) AS score
        FROM segments_fts
        JOIN segments AS s
            ON s.id = segments_fts.rowid
        JOIN sessions AS sess
            ON sess.id = s.session_id
        JOIN conferences AS conf
            ON conf.id = sess.conference_id
        WHERE {' AND '.join(where_clauses)}
        ORDER BY score, s.start_time
        LIMIT ?
    """
    return connection.execute(sql, params).fetchall()


def run_vector_search(
    connection: sqlite3.Connection,
    query: str,
    limit: int = 5,
    encode_fn=None,
    conference_filter: Optional[str] = None,
) -> List[sqlite3.Row]:
    """Semantic kNN search over segment embeddings via sqlite-vec.

    Encodes *query* with the given *encode_fn* (or loads the default
    SentenceTransformer model), then performs a kNN MATCH against the
    ``segments_vec`` virtual table.  Results are JOINed back to segments,
    sessions and conferences so callers get the same column set as
    ``run_search``, plus a ``distance`` column.

    When *conference_filter* is supplied the result set is post-filtered
    to rows whose conference id or slug matches.
    """
    if encode_fn is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        encode_fn = model.encode

    import numpy as np
    query_vec = encode_fn([query], normalize_embeddings=True)
    if isinstance(query_vec, np.ndarray) and query_vec.ndim == 2:
        query_vec = query_vec[0]
    query_bytes = query_vec.astype("float32").tobytes()

    # vec0 requires k=? inside the virtual table WHERE, not an outer LIMIT.
    # Use a subquery for kNN, then JOIN metadata and optionally filter by conference.
    conf_filter_clause = ""
    params: list = [query_bytes, limit]

    if conference_filter:
        conf_filter_clause = "AND (conf.id = ? OR conf.slug = ?)"
        params.extend([conference_filter, conference_filter])

    sql = f"""
        SELECT
            knn.segment_id,
            knn.distance,
            sess.source_id,
            sess.title,
            sess.session_type,
            s.start_time,
            s.end_time,
            s.text,
            sess.video_path,
            conf.name AS conference_name,
            conf.slug AS conference_slug
        FROM (
            SELECT segment_id, distance
            FROM segments_vec
            WHERE embedding MATCH ? AND k = ?
        ) knn
        JOIN segments s ON s.id = knn.segment_id
        JOIN sessions sess ON sess.id = s.session_id
        JOIN conferences conf ON conf.id = sess.conference_id
        {conf_filter_clause}
        ORDER BY knn.distance
    """
    return connection.execute(sql, params).fetchall()


def _load_vec(connection: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        return True
    except (ImportError, Exception):
        return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language query to search for.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path.")
    parser.add_argument("--conference", help="Optional conference id or slug filter.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to show.")
    parser.add_argument("--mode", choices=["fts", "vec", "hybrid"], default="hybrid",
                        help="Search mode: fts (lexical), vec (semantic), hybrid (fts with vec fallback).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    connection = sqlite3.connect(args.db_path)
    connection.row_factory = sqlite3.Row
    rows: List[sqlite3.Row] = []

    try:
        if args.mode in ("fts", "hybrid"):
            primary_query, fallback_query, terms = build_match_queries(args.query)
            rows = run_search(connection, primary_query, args.limit, args.conference)
            if not rows and fallback_query:
                rows = run_search(connection, fallback_query, args.limit, args.conference)

        if args.mode == "vec" or (args.mode == "hybrid" and not rows):
            if _load_vec(connection):
                rows = run_vector_search(connection, args.query, args.limit,
                                         conference_filter=args.conference)
    finally:
        connection.close()

    if not rows:
        print("No matches found.")
        return 0

    for index, row in enumerate(rows, start=1):
        source = row["source_id"]
        start = format_hms(row["start_time"])
        title = row["title"]
        text = truncate(row["text"])
        dist = f" (dist={row['distance']:.4f})" if "distance" in row.keys() else ""

        print(f"[{index}] {source} @ {start} — {title}{dist}")
        print(f"    Conference: {row['conference_name']}")
        print(f'    "{text}"')
        print(f"    Video: {row['video_path']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
