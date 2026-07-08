"""SQLite search index: catalog, chunks, keyword FTS, reference edges, embeddings.

One file on disk (data/index.db), built offline by `uv run index`, read at chat
time by the search tools. This module is the storage seam: ingest imports the
builder, tools import the connection — swapping SQLite for real infrastructure
(pgvector, Elasticsearch) at scale means replacing this module only.

Design notes:
- documents is keyed (doc_id, revision): one row per revision, title stored
  per revision (titles change across revisions in this corpus). "Latest
  non-obsolete" is a query-time decision, never a stored column.
- revision '' means unversioned (treated as current).
- chunks_fts uses the porter stemmer so "verification" matches "verify".
- embeddings is populated by the (optional) embedding pass; absence of rows
  simply disables semantic mode.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB = Path("data/index.db")

_SCHEMA = """
CREATE TABLE documents (
    doc_id       TEXT NOT NULL,
    revision     TEXT NOT NULL DEFAULT '',  -- '' = unversioned
    title        TEXT NOT NULL,
    type_prefix  TEXT NOT NULL,
    is_signed    INTEGER NOT NULL,
    is_obsolete  INTEGER NOT NULL,
    is_empty     INTEGER NOT NULL,
    sw_version   TEXT,
    filename     TEXT NOT NULL,
    content_hash TEXT,
    char_count   INTEGER NOT NULL,
    PRIMARY KEY (doc_id, revision)
);
CREATE INDEX idx_documents_type ON documents(type_prefix);

CREATE TABLE chunks (
    id           INTEGER PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    revision     TEXT NOT NULL,
    section_path TEXT NOT NULL,   -- 'METHODS > Procedure'; '' = no headings
    position     INTEGER NOT NULL, -- order within the document
    kind         TEXT NOT NULL,    -- 'text' | 'table' | 'mixed'
    text         TEXT NOT NULL
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id, revision);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TABLE refs (
    from_doc   TEXT NOT NULL,
    from_rev   TEXT NOT NULL,
    to_doc     TEXT NOT NULL,
    n_mentions INTEGER NOT NULL,
    context    TEXT NOT NULL,     -- excerpt around the first mention
    PRIMARY KEY (from_doc, from_rev, to_doc)
);
CREATE INDEX idx_refs_to ON refs(to_doc);

CREATE TABLE embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
    vector   BLOB NOT NULL
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open the index with row access by column name.

    check_same_thread=False: the index is read-only after build, and LangGraph
    executes tools in worker threads — a shared connection is safe under
    SQLite's default serialized threading mode.
    """
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def create(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    """Create a fresh index file, replacing any existing one."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = connect(db_path)
    con.executescript(_SCHEMA)
    return con


def rebuild_fts(con: sqlite3.Connection) -> None:
    """Populate the FTS mirror from chunks (call once, after bulk insert)."""
    con.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")
