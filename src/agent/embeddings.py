"""Chunk embeddings: built at ingest, queried by semantic search.

Each chunk is embedded WITH a context header (doc ID, revision, title,
section) so table rows and short sections carry their provenance into the
vector — "Result: Pass" embeds as noise, but not with its header attached.

Vectors are L2-normalized float32 BLOBs in the `embeddings` table; query-time
similarity is one dot product over a matrix loaded once per process. Brute
force is exact and takes milliseconds at this corpus size — an ANN index
earns its complexity around 100k+ vectors (see plan.md scaling ladder).

Backfill an existing index without re-extracting docx:

    uv run python -m agent.embeddings
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np

from agent import store

EMBED_MODEL = "text-embedding-3-small"
_BATCH = 128


def _embedder():
    from langchain_openai import OpenAIEmbeddings  # lazy: keyword-only use needs no key

    return OpenAIEmbeddings(model=EMBED_MODEL)


def chunk_context_text(
    doc_id: str, revision: str, title: str, section_path: str, text: str
) -> str:
    """The string that gets embedded: context header + chunk text."""
    header = f"{doc_id} Rev {revision or '—'} — {title}"
    if section_path:
        header += f"\n§ {section_path}"
    return f"{header}\n---\n{text}"


def build_embeddings(con: sqlite3.Connection, progress=print) -> int:
    """Embed every chunk that doesn't have a vector yet. Idempotent.

    Note: `uv run index` recreates the DB, so a full rebuild re-embeds the
    whole corpus (~$0.12 / ~2 min here). Reusing vectors by content hash is
    deliberate future work, not worth the complexity at this scale.
    """
    # Refuse to mix models: resumed backfills after a model change would
    # silently produce meaningless similarities.
    stored = con.execute(
        "SELECT value FROM meta WHERE key = 'embed_model'"
    ).fetchone()
    if stored and stored["value"] != EMBED_MODEL:
        raise ValueError(
            f"index has {stored['value']} embeddings; refusing to append "
            f"{EMBED_MODEL} — rebuild the index or delete the embeddings table"
        )
    if not stored:
        con.execute(
            "INSERT OR REPLACE INTO meta VALUES ('embed_model', ?)", (EMBED_MODEL,)
        )

    rows = con.execute(
        """
        SELECT c.id, c.doc_id, c.revision, c.section_path, c.text, d.title
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id AND d.revision = c.revision
        WHERE c.id NOT IN (SELECT chunk_id FROM embeddings)
        ORDER BY c.id
        """
    ).fetchall()
    if not rows:
        return 0

    embedder = _embedder()
    for i in range(0, len(rows), _BATCH):
        batch = rows[i : i + _BATCH]
        texts = [
            chunk_context_text(
                r["doc_id"], r["revision"], r["title"], r["section_path"], r["text"]
            )
            for r in batch
        ]
        vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        con.executemany(
            "INSERT INTO embeddings VALUES (?, ?)",
            [(r["id"], v.tobytes()) for r, v in zip(batch, vectors)],
        )
        con.commit()
        progress(f"  embedded {min(i + _BATCH, len(rows))}/{len(rows)} chunks")
    return len(rows)


class SemanticIndex:
    """In-memory similarity search over the stored vectors."""

    def __init__(self, con: sqlite3.Connection):
        rows = con.execute("SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id").fetchall()
        self.chunk_ids = np.array([r["chunk_id"] for r in rows])
        self.matrix = (
            np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32)
            .reshape(len(rows), -1)
            if rows
            else np.empty((0, 0), dtype=np.float32)
        )
        self._embedder = None

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """[(chunk_id, similarity)] in descending similarity; all chunks when
        top_k is None — scores are already computed for the whole corpus, so
        callers can filter as deeply as they need with zero recall loss."""
        if len(self.chunk_ids) == 0:
            return []
        if self._embedder is None:
            self._embedder = _embedder()
        q = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = self.matrix @ q
        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        return [(int(self.chunk_ids[i]), float(scores[i])) for i in order]


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    con = store.connect()
    n = build_embeddings(con)
    total = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"embedded {n} new chunks ({total} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
