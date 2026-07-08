"""Build the search index from a corpus directory: `uv run index`.

One pass over every .docx: parse the filename, extract sections, pack them into
chunks, scan chunk text for cross-document references, and write everything to
SQLite. Ends with a summary report — the sanity check that the corpus was
understood, not just swallowed.

Policy reminders (see plan.md):
- Facts are stored per revision; "latest non-obsolete" is decided at query time.
- Empty docs get catalog rows (is_empty=1) and no chunks: "exists but empty"
  must stay distinguishable from "not found".
- Duplicate files ("(1)") are skipped only when their content hash matches the
  original; a differing duplicate is kept and reported.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from agent import store
from agent.ingest.docx_extract import Section, extract_sections
from agent.ingest.filename import DocMeta, parse_corpus

# Chunk packing: blocks are appended until the chunk would exceed this. Blocks
# are never split (tables arrive pre-bounded from docx_extract).
MAX_CHUNK_CHARS = 2_000

# Heading-styled body sentences exist in the corpus; keep citation labels sane.
MAX_HEADING_PART_CHARS = 80


@dataclass
class Chunk:
    section_path: str
    position: int
    kind: str  # 'text' | 'table' | 'mixed'
    text: str


def _heading_label(parts: tuple[str, ...]) -> str:
    trimmed = [
        p if len(p) <= MAX_HEADING_PART_CHARS else p[: MAX_HEADING_PART_CHARS - 1] + "…"
        for p in parts
    ]
    return " > ".join(trimmed)


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """Pack section blocks into chunks of ~MAX_CHUNK_CHARS, tables kept whole."""
    chunks: list[Chunk] = []

    for section in sections:
        # No-heading docs (22 in corpus) get "body" so citations read
        # [3P-P01-33 Rev A, §body] rather than an empty section label.
        label = _heading_label(section.heading_path) or "body"
        buf: list = []

        def flush() -> None:
            if not buf:
                return
            kinds = {b.kind for b in buf}
            kind = kinds.pop() if len(kinds) == 1 else "mixed"
            chunks.append(
                Chunk(
                    section_path=label,
                    position=len(chunks),
                    kind=kind,
                    text="\n".join(b.text for b in buf),
                )
            )
            buf.clear()

        for block in section.blocks:
            if buf and sum(len(b.text) for b in buf) + len(block.text) > MAX_CHUNK_CHARS:
                flush()
            buf.append(block)
        flush()

    return chunks


def _reference_regex(type_prefixes: set[str]) -> re.Pattern:
    """Doc-ID mentions in text, restricted to known corpus type prefixes.

    The restriction keeps standards ("IEC 60601-1"), fixtures ("T-174") and
    workstation IDs ("WS-002") out of the graph. P00 predecessor references
    match too — they become dangling edges, which is intentional.

    Deliberately different from filename._DOC_ID_RE (anchored, accepts any
    prefix — filenames are trusted) and tools._ID_QUERY_RE (loosest — user
    queries get validated against the catalog). Same ID concept, three
    different trust levels; merging them would couple their contracts.
    """
    alternatives = "|".join(sorted(type_prefixes - {"IFU"}, key=len, reverse=True))
    return re.compile(
        rf"\b(IFU-MX1|(?:{alternatives})(?:-[A-Z0-9]{{2,3}})?-\d{{2,4}})\b"
    )


def extract_refs(
    chunks: list[Chunk], own_doc_id: str, pattern: re.Pattern
) -> dict[str, tuple[int, str]]:
    """target doc_id -> (mention count, excerpt around first mention)."""
    found: dict[str, tuple[int, str]] = {}
    for chunk in chunks:
        for m in pattern.finditer(chunk.text):
            target = m.group(1)
            if target == own_doc_id:
                continue
            if target in found:
                n, ctx = found[target]
                found[target] = (n + 1, ctx)
            else:
                lo, hi = max(0, m.start() - 60), m.end() + 60
                found[target] = (1, chunk.text[lo:hi].strip())
    return found


def build_index(corpus_dir: Path, db_path: Path) -> dict:
    """Ingest the corpus into a fresh index; returns summary stats."""
    t0 = time.time()
    metas, failures = parse_corpus(corpus_dir)
    if failures:
        for name, err in failures:
            print(f"PARSE FAILURE {name}: {err}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} filenames failed to parse; aborting")

    ref_re = _reference_regex({m.type_prefix for m in metas})

    # Originals before "(n)" copies: on a (doc_id, rev) collision the original
    # must win, and sorted() puts "_B(1)" before "_B" ('(' < '.').
    metas.sort(key=lambda m: m.is_duplicate)

    con = store.create(db_path)
    seen: dict[tuple[str, str], str] = {}  # (doc_id, rev) -> content_hash
    stats = defaultdict(int)
    stats["files"] = len(metas)

    for meta in metas:
        sections = extract_sections(corpus_dir / meta.filename)
        chunks = chunk_sections(sections)
        text_all = "\n".join(c.text for c in chunks)
        content_hash = hashlib.sha256(text_all.encode()).hexdigest()
        rev = meta.revision or ""

        key = (meta.doc_id, rev)
        if key in seen:
            if seen[key] == content_hash:
                stats["duplicates_skipped"] += 1
            else:
                stats["duplicates_conflicting"] += 1
                print(
                    f"WARNING: {meta.filename} duplicates {key} with DIFFERENT "
                    "content; keeping the first, skipping this one",
                    file=sys.stderr,
                )
            continue
        seen[key] = content_hash

        is_empty = not chunks
        if is_empty:
            stats["empty_docs"] += 1

        con.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                meta.doc_id, rev, meta.title, meta.type_prefix,
                meta.is_signed, meta.is_obsolete, is_empty,
                meta.sw_version, meta.filename, content_hash, len(text_all),
            ),
        )
        con.executemany(
            "INSERT INTO chunks (doc_id, revision, section_path, position, kind, text)"
            " VALUES (?,?,?,?,?,?)",
            [
                (meta.doc_id, rev, c.section_path, c.position, c.kind, c.text)
                for c in chunks
            ],
        )
        stats["chunks"] += len(chunks)

        for target, (n, context) in extract_refs(chunks, meta.doc_id, ref_re).items():
            con.execute(
                "INSERT OR IGNORE INTO refs VALUES (?,?,?,?,?)",
                (meta.doc_id, rev, target, n, context),
            )
            stats["ref_edges"] += 1

    store.rebuild_fts(con)
    con.execute(
        "INSERT INTO meta VALUES ('corpus_dir', ?), ('chunk_max_chars', ?)",
        (str(corpus_dir), str(MAX_CHUNK_CHARS)),
    )
    con.commit()

    stats["documents"] = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    stats["doc_ids"] = con.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM documents"
    ).fetchone()[0]
    stats["rev_chains"] = con.execute(
        "SELECT COUNT(*) FROM (SELECT doc_id FROM documents GROUP BY doc_id"
        " HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    stats["dangling_refs"] = con.execute(
        "SELECT COUNT(DISTINCT to_doc) FROM refs"
        " WHERE to_doc NOT IN (SELECT doc_id FROM documents)"
    ).fetchone()[0]
    stats["seconds"] = round(time.time() - t0, 1)
    con.close()
    return dict(stats)


def _report(stats: dict, db_path: Path) -> None:
    g = stats.get  # counters are absent (not zero) when nothing incremented them
    print(f"\nindex built: {db_path} ({g('seconds')}s)")
    print(f"  files scanned:        {g('files', 0)}")
    print(f"  documents indexed:    {g('documents', 0)} "
          f"({g('doc_ids', 0)} doc IDs, {g('rev_chains', 0)} revision chains)")
    print(f"  empty (catalog-only): {g('empty_docs', 0)}")
    print(f"  duplicates skipped:   {g('duplicates_skipped', 0)}"
          + (f"  CONFLICTING: {g('duplicates_conflicting')}"
             if g("duplicates_conflicting") else ""))
    print(f"  chunks:               {g('chunks', 0)}")
    print(f"  reference edges:      {g('ref_edges', 0)} "
          f"({g('dangling_refs', 0)} dangling targets)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the search index")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="skip semantic embeddings (keyword/catalog/graph still work)",
    )
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"corpus directory not found: {args.corpus}", file=sys.stderr)
        return 2

    stats = build_index(args.corpus, args.db)
    _report(stats, args.db)

    if not args.no_embeddings:
        from dotenv import load_dotenv

        from agent.embeddings import build_embeddings

        load_dotenv()
        try:
            n = build_embeddings(store.connect(args.db))
            print(f"  embeddings:           {n} chunks")
        except Exception as e:  # index stays fully usable in keyword mode
            print(
                f"  embeddings SKIPPED ({type(e).__name__}: {e}) — semantic "
                "mode disabled; backfill later with `uv run python -m agent.embeddings`",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
