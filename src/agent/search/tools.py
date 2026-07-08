"""The four search tools, as plain functions over the SQLite index.

No LLM anywhere in this module. Policy that must be exact lives here — revision
resolution, obsolete filtering, counting — so the model orchestrates but never
computes facts. Every function returns JSON-serializable dicts whose fields
carry citation provenance (doc_id, revision, section_path).

Honesty-over-silent-defaults: tools return structured signals instead of
guessing — `no_current_revision`, `is_empty`, `closest_matches` on a miss —
and the agent surfaces them to the user.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from rapidfuzz import fuzz

from agent import store

# Results below this fuzzy score are not considered matches (0-100). High on
# purpose: honest not-found beats generous wrong matches. Typo tolerance comes
# from the deterministic ID matcher and the type-name map, not a low bar.
FUZZY_THRESHOLD = 75

# Common names for document types, so "bill of materials" finds BOM-* docs
# whose titles never contain those words.
TYPE_NAMES = {
    "BOM": "bill of materials",
    "VVPR": "verification protocol report",
    "VVAM": "verification validation acceptance matrix trace",
    "ECR": "engineering change request",
    "ESF": "engineering specification",
    "RSK": "risk analysis assessment",
    "PLN": "plan",
    "DHF": "design history file",
    "DMR": "device master record",
    "DR": "design review",
    "IFU": "instructions for use",
    "TRA": "training",
    "QSR": "quality system record",
    "MEMO": "memo",
    "3P": "third party report",
}

# An ID-shaped fragment: prefix letters, optional project code, and a number —
# "vvpr 151", "VVPR-P01-151", "ecr 577" — anywhere in the query. Looser than
# ingest's regexes on purpose (see filename._DOC_ID_RE for the strict shape);
# a fragment only counts if it names a real (type, number) pair in the catalog.
_ID_QUERY_RE = re.compile(
    r"\b([A-Za-z0-9]{2,4})[\s._-]*(?:P0\d[\s._-]*)?(\d{2,4})\b", re.IGNORECASE
)

# read_document returns full text up to this; larger docs return an outline
# (the trace matrix alone is ~2.9M chars — never dump that into a context).
MAX_FULL_DOC_CHARS = 30_000

_db_path: Path = store.DEFAULT_DB
_con: sqlite3.Connection | None = None
_chains_cache: list[dict] | None = None
_semantic_cache: object | None = None


def set_db_path(path: Path) -> None:
    """Point the tools at a different index (used by tests)."""
    global _db_path, _con, _chains_cache, _semantic_cache
    _db_path = path
    _con = None
    _chains_cache = None
    _semantic_cache = None


def _semantic():
    """The in-memory semantic index, loaded once (None if never embedded)."""
    global _semantic_cache
    if _semantic_cache is None:
        from agent.embeddings import SemanticIndex  # lazy: needs numpy/openai

        _semantic_cache = SemanticIndex(_db())
    return _semantic_cache


def _db() -> sqlite3.Connection:
    global _con
    if _con is None:
        if not Path(_db_path).exists():
            raise FileNotFoundError(
                f"search index not found at {_db_path} — run `uv run index` first"
            )
        _con = store.connect(_db_path)
    return _con


# ---------------------------------------------------------------- catalog --


def _norm_id(s: str) -> str:
    """Normalize for doc-ID comparison: 'vvpr 151' ~ 'VVPR-P01-151'."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _rev_key(revision: str) -> tuple[int, str]:
    """Chronological revision order: A < B < ... < Z < AA (not lexicographic)."""
    return (len(revision), revision)


def _chains() -> list[dict]:
    """Catalog grouped into revision chains, one entry per doc_id.

    Cached: the index only changes via offline rebuild. Callers must not
    mutate the returned dicts (search_catalog copies before annotating).
    """
    global _chains_cache
    if _chains_cache is not None:
        return _chains_cache
    rows = _db().execute(
        "SELECT * FROM documents ORDER BY doc_id, revision"
    ).fetchall()
    chains: dict[str, dict] = {}
    for r in rows:
        c = chains.setdefault(
            r["doc_id"],
            {"doc_id": r["doc_id"], "type": r["type_prefix"], "revisions": []},
        )
        c["revisions"].append(
            {
                "revision": r["revision"],
                "title": r["title"],
                "is_signed": bool(r["is_signed"]),
                "is_obsolete": bool(r["is_obsolete"]),
                "is_empty": bool(r["is_empty"]),
                "sw_version": r["sw_version"],
            }
        )
    for c in chains.values():
        c["revisions"].sort(key=lambda r: _rev_key(r["revision"]))
        current = [r for r in c["revisions"] if not r["is_obsolete"]]
        latest = max(current, key=lambda r: _rev_key(r["revision"])) if current else None
        c["latest_current_revision"] = latest["revision"] if latest else None
        c["title"] = (latest or c["revisions"][-1])["title"]
        if latest is None:
            c["note"] = "all revisions are obsolete — no current version"
    _chains_cache = list(chains.values())
    return _chains_cache


def _id_score(chain: dict, query: str) -> float:
    """Deterministic doc-ID match: 'vvpr 151', 'VVPR-P01-151', or an ID
    embedded in a sentence ('find VVPR-P01-151 for me') scores 100."""
    if _norm_id(query) == _norm_id(chain["doc_id"]):
        return 100
    chain_number = chain["doc_id"].rsplit("-", 1)[-1]
    for m in _ID_QUERY_RE.finditer(query):
        prefix, number = m.group(1).upper(), m.group(2)
        # Only a real (type, number) pair counts — junk like "20 23" from
        # dates finds no chain and scores nothing.
        if chain["type"] == prefix and chain_number.lstrip("0") == number.lstrip("0"):
            return 100
    return 0


# Words too generic in THIS corpus to count as evidence (every other title
# says "MX1" or "system"), plus ordinary stopwords.
_STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "to", "in", "with",
              "mx1", "system"}


def _token_score(query: str, text: str) -> float:
    """Length-weighted token coverage: every meaningful query token must find
    a (fuzzy) home in the text. Partial hits ("summary" alone out of "510(k)
    summary") score low — unlike WRatio, which rewards them."""
    q_tokens = [
        t for t in re.findall(r"[a-z0-9]{2,}", query.lower()) if t not in _STOPWORDS
    ]
    if not q_tokens:
        return 0
    t_tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
    if not t_tokens:
        return 0
    total = weighted = 0.0
    for qt in q_tokens:
        best = max(fuzz.ratio(qt, tt) for tt in t_tokens)
        if best < 55:  # no real home for this token — not partial evidence
            best = 0
        weighted += len(qt) * best
        total += len(qt)
    return weighted / total


def _score_chain(chain: dict, query: str) -> float:
    """Best of: deterministic ID match, title coverage, type-name coverage."""
    title_score = max(
        _token_score(query, r["title"]) for r in chain["revisions"]
    )
    # Type-name matches ("bill of materials" -> BOM) are weaker evidence than
    # a title hit: cap them so they rank below real title matches.
    type_score = min(_token_score(query, TYPE_NAMES.get(chain["type"], "")), 88.0)
    return max(_id_score(chain, query), title_score, type_score)


def search_catalog(
    query: str | None = None,
    doc_type: str | None = None,
    status: str | None = None,
    sw_version: str | None = None,
) -> dict:
    """Search document metadata. Exhaustive: returns ALL matches plus the count.

    query: fuzzy-matched against doc IDs and titles (typo-tolerant).
    doc_type: filter by type prefix, e.g. "VVPR", "ECR".
    status: "signed" (some revision is signed)
            | "obsolete" (NO current version — every revision is obsolete)
            | "current" (has a non-obsolete revision)
            | "empty" (no content in any revision).
    sw_version: software version from document titles, e.g. "v3.0.0" or
                "3.0.0" — answers "what did we test for v3.0.0?".
    """
    chains = _chains()

    if doc_type:
        chains = [c for c in chains if c["type"] == doc_type.upper()]
    if sw_version:
        v = sw_version.lower()
        v = v if v.startswith("v") else f"v{v}"
        chains = [
            c for c in chains
            if any((r["sw_version"] or "").lower() == v for r in c["revisions"])
        ]
    if status:
        s = status.lower()
        if s == "signed":
            chains = [c for c in chains if any(r["is_signed"] for r in c["revisions"])]
        elif s == "obsolete":
            chains = [c for c in chains if c["latest_current_revision"] is None]
        elif s == "current":
            chains = [c for c in chains if c["latest_current_revision"] is not None]
        elif s == "empty":
            chains = [c for c in chains if all(r["is_empty"] for r in c["revisions"])]
        else:
            return {"error": f"unknown status {status!r}"}

    if query:
        scored = [(c, _score_chain(c, query)) for c in chains]
        scored = [(c, s) for c, s in scored if s >= FUZZY_THRESHOLD]
        scored.sort(key=lambda cs: -cs[1])
        chains = [dict(c, match_score=round(s)) for c, s in scored]

    result = {"count": len(chains), "documents": chains}
    if query and not chains:
        # A miss still helps: closest few below threshold, clearly labeled.
        all_chains = _chains()
        near = sorted(all_chains, key=lambda c: -_score_chain(c, query))[:3]
        result["closest_matches"] = [
            {"doc_id": c["doc_id"], "title": c["title"]} for c in near
        ]
        result["note"] = "no documents matched; closest_matches are below threshold"
    return result


# ---------------------------------------------------------------- content --


def _default_revisions() -> dict[tuple[str, str], bool]:
    """Each doc's latest revision -> whether that revision is current.

    Content search defaults to the latest revision of EVERY document — for the
    11 doc IDs whose every revision is obsolete (incl. the risk assessment),
    excluding them would make their content silently unsearchable. Obsolete-
    latest docs are included and flagged is_current=False so the agent caveats.
    """
    out: dict[tuple[str, str], bool] = {}
    for c in _chains():
        latest_any = max(c["revisions"], key=lambda r: _rev_key(r["revision"]))
        out[(c["doc_id"], latest_any["revision"])] = (
            c["latest_current_revision"] == latest_any["revision"]
        )
    return out


def search_content(
    query: str,
    mode: str = "keyword",
    top_k: int = 10,
    include_superseded: bool = False,
) -> dict:
    """Search inside document content; returns chunks with citation labels.

    mode: "keyword" (exact words, BM25) — use for jargon, IDs, standards.
          "semantic" (meaning) — use when the user's words may differ from the
          document's words (e.g. "electrical safety" vs "dielectric/leakage").
    include_superseded: also search obsolete/older revisions (history questions).
    """
    defaults = _default_revisions()

    def assemble(rows) -> list[dict]:
        """Filter + format rows (any iterable, relevance-ordered) into results."""
        results: list[dict] = []
        per_doc: dict[str, int] = {}
        for r in rows:
            key = (r["doc_id"], r["revision"])
            is_current = defaults.get(key, False)
            if not include_superseded and key not in defaults:
                continue
            # Result diversity: giant docs (trace matrix, risk files) would
            # otherwise flood the top-K with near-duplicate chunks.
            if per_doc.get(r["doc_id"], 0) >= 2:
                continue
            per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
            results.append(
                {
                    "doc_id": r["doc_id"],
                    "revision": r["revision"],
                    "title": r["title"],
                    "section": r["section_path"],
                    "kind": r["kind"],
                    "is_current": is_current,
                    "excerpt": r["text"][:700],
                    "chunk_id": r["id"],
                }
            )
            if len(results) >= top_k:
                break
        return results

    if mode == "semantic":
        index = _semantic()
        if index is None or len(index) == 0:
            return {
                "error": "semantic mode unavailable: no embeddings in index "
                "(run `uv run python -m agent.embeddings`) — use keyword mode"
            }

        def semantic_rows():
            # Similarity is computed corpus-wide anyway: walk the full order
            # in batches so filtering can never exhaust the candidates.
            hits = index.search(query)
            for i in range(0, len(hits), 400):
                batch = dict(hits[i : i + 400])
                placeholders = ",".join("?" * len(batch))
                rows = _db().execute(
                    f"""
                    SELECT c.id, c.doc_id, c.revision, c.section_path, c.kind,
                           c.text, d.title, d.is_obsolete
                    FROM chunks c
                    JOIN documents d ON d.doc_id = c.doc_id AND d.revision = c.revision
                    WHERE c.id IN ({placeholders})
                    """,
                    list(batch),
                ).fetchall()
                yield from sorted(rows, key=lambda r: -batch[r["id"]])

        results = assemble(semantic_rows())
    else:
        terms = re.findall(r"[A-Za-z0-9]+", query)
        if not terms:
            return {"error": "empty query"}

        def fts_rows(match_expr: str, limit: int) -> list:
            return _db().execute(
                """
                SELECT c.id, c.doc_id, c.revision, c.section_path, c.kind, c.text,
                       d.title, d.is_obsolete, bm25(chunks_fts) AS score
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                JOIN documents d ON d.doc_id = c.doc_id AND d.revision = c.revision
                WHERE chunks_fts MATCH ?
                ORDER BY score LIMIT ?
                """,
                (match_expr, limit),
            ).fetchall()

        over_fetch = top_k * 4
        quoted = [f'"{t}"' for t in terms]
        rows = fts_rows(" ".join(quoted), over_fetch)  # AND first (precision)...
        if not rows:
            rows = fts_rows(" OR ".join(quoted), over_fetch)  # ...OR fallback
        results = assemble(rows)
        if len(results) < top_k and len(rows) == over_fetch:
            # Filtering starved the window — refetch deep once, then accept.
            results = assemble(fts_rows(" ".join(quoted), 2000) or
                               fts_rows(" OR ".join(quoted), 2000))

    return {"count": len(results), "mode": mode, "results": results}


# ------------------------------------------------------------------- read --


def read_document(
    doc_id: str, revision: str | None = None, section: str | None = None
) -> dict:
    """Read a document. Defaults to the latest non-obsolete revision.

    Large documents return a section outline instead of full text; pass
    `section` to fetch one section. Structured signals instead of guesses:
    not found -> closest_matches; all revisions obsolete -> no_current_revision;
    empty file -> is_empty.
    """
    chains = {c["doc_id"]: c for c in _chains()}
    # Same loose ID forms search_catalog accepts: "vvpr-p01-151", "vvpr 151".
    hits = [c for c in chains.values() if _id_score(c, doc_id) == 100]
    chain = hits[0] if len(hits) == 1 else None
    if chain is None:
        near = search_catalog(query=doc_id)
        return {
            "error": f"no document with ID {doc_id!r}",
            "closest_matches": [
                {"doc_id": c["doc_id"], "title": c["title"]}
                for c in near["documents"][:3]
            ]
            or near.get("closest_matches", []),
        }

    revisions_summary = chain["revisions"]
    if revision is None:
        if chain["latest_current_revision"] is None:
            return {
                "doc_id": chain["doc_id"],
                "no_current_revision": True,
                "note": "every revision is marked obsolete — confirm with the "
                "user before falling back to an obsolete revision",
                "revisions": revisions_summary,
            }
        revision = chain["latest_current_revision"]
    else:
        revision = revision.upper()
        if revision not in {r["revision"] for r in revisions_summary}:
            return {
                "doc_id": chain["doc_id"],
                "error": f"no revision {revision!r}",
                "revisions": revisions_summary,
            }

    rev_info = next(r for r in revisions_summary if r["revision"] == revision)
    base = {
        "doc_id": chain["doc_id"],
        "revision": revision,
        "title": rev_info["title"],
        "is_obsolete": rev_info["is_obsolete"],
        "revisions": revisions_summary,
    }
    if rev_info["is_empty"]:
        return {**base, "is_empty": True,
                "note": "this document exists in the corpus but its file is empty"}

    rows = _db().execute(
        "SELECT section_path, kind, text FROM chunks"
        " WHERE doc_id=? AND revision=? ORDER BY position",
        (chain["doc_id"], revision),
    ).fetchall()

    if section is not None:
        s = section.lower()
        rows = [r for r in rows if s in r["section_path"].lower()]
        if not rows:
            return {**base, "error": f"no section matching {section!r}",
                    "sections": _outline(chain["doc_id"], revision)}

    total = sum(len(r["text"]) for r in rows)
    if total > MAX_FULL_DOC_CHARS and section is None:
        return {
            **base,
            "too_large": True,
            "char_count": total,
            "note": "document too large to return whole — pass `section` to "
            "read one section",
            "sections": _outline(chain["doc_id"], revision),
        }

    body = []
    last_path = object()
    included = 0
    truncated = False
    for r in rows:
        # Even a single section can be huge (trace-matrix sections average
        # 64k chars) — cap what one call returns, and say so.
        if included + len(r["text"]) > 2 * MAX_FULL_DOC_CHARS:
            truncated = True
            break
        if r["section_path"] != last_path:
            body.append(f"\n§ {r['section_path'] or '(no heading)'}")
            last_path = r["section_path"]
        body.append(r["text"])
        included += len(r["text"])

    result = {**base, "char_count": total, "text": "\n".join(body).strip()}
    if truncated:
        result["truncated"] = True
        result["note"] = (
            f"returned first {included:,} of {total:,} chars — narrow with a "
            "more specific `section`, or use search_content to find passages"
        )
    return result


def _outline(doc_id: str, revision: str) -> list[dict]:
    rows = _db().execute(
        "SELECT section_path, SUM(LENGTH(text)) chars, COUNT(*) chunks FROM chunks"
        " WHERE doc_id=? AND revision=? GROUP BY section_path ORDER BY MIN(position)",
        (doc_id, revision),
    ).fetchall()
    return [
        {"section": r["section_path"] or "(no heading)", "chars": r["chars"]}
        for r in rows
    ]


# ------------------------------------------------------------------ graph --


def get_references(doc_id: str, direction: str = "both") -> dict:
    """Cross-document references: which documents cite / are cited by this one.

    Edges come from doc-ID mentions in text (incl. revision-history tables).
    `resolved: false` targets are cited but not in this corpus (e.g. P00
    predecessors) — real references, just not readable here.
    """
    aliases = {"in": "inbound", "out": "outbound", "inbound": "inbound",
               "outbound": "outbound", "both": "both"}
    direction = aliases.get(direction.lower())
    if direction is None:
        return {"error": "direction must be 'inbound', 'outbound', or 'both'"}

    con = _db()
    wanted = _norm_id(doc_id)
    known = {
        _norm_id(r["doc_id"]): r["doc_id"]
        for r in con.execute("SELECT DISTINCT doc_id FROM documents")
    }
    canonical = known.get(wanted)
    if canonical is None:
        return {"error": f"no document with ID {doc_id!r}"}

    in_corpus = set(known.values())
    out: dict = {"doc_id": canonical}
    if direction in ("outbound", "both"):
        rows = con.execute(
            "SELECT to_doc, MAX(n_mentions) n, MAX(context) context FROM refs"
            " WHERE from_doc=? GROUP BY to_doc ORDER BY n DESC",
            (canonical,),
        ).fetchall()
        out["references"] = [
            {"doc_id": r["to_doc"], "mentions": r["n"],
             "resolved": r["to_doc"] in in_corpus, "context": r["context"][:160]}
            for r in rows
        ]
    if direction in ("inbound", "both"):
        rows = con.execute(
            "SELECT from_doc, from_rev, n_mentions, context FROM refs"
            " WHERE to_doc=? ORDER BY n_mentions DESC",
            (canonical,),
        ).fetchall()
        out["cited_by"] = [
            {"doc_id": r["from_doc"], "revision": r["from_rev"],
             "mentions": r["n_mentions"], "context": r["context"][:160]}
            for r in rows
        ]
    return out
