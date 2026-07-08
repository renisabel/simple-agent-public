# Code Study Guide

How to read this codebase and own every decision in it. Reading time: ~45 min.

## The one-diagram version

```
OFFLINE (once, `uv run index`)              ONLINE (every chat turn)
────────────────────────────────            ─────────────────────────────
data/corpus/*.docx                          user question
  │  ingest/filename.py                       │  cli.py (prints tool trace)
  │    filename → DocMeta                     ▼
  │  ingest/docx_extract.py                 core.py: deep agent (LLM loop)
  │    docx → Sections (text+tables)          │  picks tools, iterates,
  │  ingest/pipeline.py                       │  then answers with citations
  │    Sections → Chunks                      │  behavior defined in prompts.py
  │    chunk text → reference edges           ▼
  │  embeddings.py                          search/tools.py  (the 4 tools)
  │    chunk+header → vector                  │ search_catalog   ─ metadata, exhaustive
  ▼                                           │ search_content   ─ FTS5 / embeddings
store.py → data/index.db (SQLite)      ◄──────┤ read_document    ─ full text, guarded
  documents / chunks / chunks_fts             │ get_references   ─ graph walk
  refs / embeddings / meta                    ▼
                                            evals/ (layer 1: tools direct, no LLM;
                                                    layer 2: whole agent + citations)
```

Two rules explain the whole architecture:

1. **The chunk is the unit of everything.** One section of one revision of one
   document = one chunk (tables kept whole). We search chunks, read chunks,
   cite chunks. Every chunk carries (doc_id, revision, section_path), so
   citations exist by construction — the model can't see text without its label.
2. **Policy in tools, judgment in the model, evals police the boundary.**
   Anything that must be exact — revision resolution, obsolete filtering,
   counting — happens in Python. The LLM only decides *which* tool to call
   and how to synthesize results.

## Reading order

### 1. `src/agent/ingest/filename.py` (~10 min)

Start here; it's small and introduces the domain. `parse_filename()` turns
`"VVPR-P01-151 - Half Value Layer..._B-signed.docx"` into a `DocMeta`.

Know cold:
- The filename convention IS the metadata layer — type, ID, title, revision,
  signed/obsolete — no file needs opening to build the catalog.
- **Title is per-revision** (titles change between revs — BOM-079, ESF-P01-003).
- Unknown status suffix (`_B-Draft`) → parses the rev, sets `warning`, never
  silently corrupts. Missing rev → `revision=None` = "unversioned, current".
- Run its verify harness: `uv run python -m agent.ingest.filename data/corpus`

### 2. `src/agent/ingest/docx_extract.py` (~7 min)

`extract_sections()` walks the Word body in order, splitting at Heading 1/2.
Tables become markdown, split by rows past 4k chars with the header repeated.

Know cold:
- Some doc types (ECR, BOM) are **100% tables** — no table handling means
  those types are invisible to search.
- List numbers live in Word metadata (`numPr`), not text → we prefix `- `.
- The docstring records *verified* skip decisions (headers/footers duplicate
  filename info; content controls are just ToCs). Say "verified against the
  corpus", not "assumed".
- Empty docs → `[]`. No headings (22 real files) → one section, empty path.

### 3. `src/agent/ingest/pipeline.py` + `src/agent/store.py` (~8 min)

`build_index()`: parse → extract → chunk (~2k chars, blocks never split) →
hash → dedupe → write SQLite → scan for reference edges → summary report.

Know cold:
- **Facts stored per revision; "latest/current" computed at query time** —
  never baked in at ingest. This one principle handles every revision edge case.
- Empty docs get catalog rows (`is_empty=1`), no chunks — so "IFU exists but
  is empty" is answerable ("not found" would be a lie).
- Duplicate `(1)` files: skipped only if content hash matches; originals win
  (ASCII sorting trap: `(` < `.`).
- Reference regex is **restricted to known type prefixes** — that's what keeps
  `IEC 60601-1` and `WS-002` fixture IDs out of the graph. Dangling edges
  (P00 docs, out-of-corpus ECRs) are kept deliberately: the corpus is a sample.
- store.py is the swap seam: replace this one module for pgvector/Elasticsearch
  at scale. `check_same_thread=False` because LangGraph runs tools in worker
  threads and the index is read-only after build.

### 4. `src/agent/search/tools.py` (~12 min — the heart)

Four functions, all returning JSON-able dicts with citation fields.

`search_catalog` — exhaustive (ALL matches + exact count; counting queries
can never be wrong-by-truncation). Matching is three-tier, best score wins:
1. `_id_score`: deterministic — "vvpr 151" or an ID inside a sentence → 100,
   validated against the catalog (junk like "20 23" can't match).
2. `_token_score` vs titles: length-weighted token coverage; a token with no
   fuzzy home scores 0 (that's what makes "510(k) summary" fail honestly
   while "bill of materails" still works); "mx1"/"system" are corpus stopwords.
3. Same scorer vs `TYPE_NAMES` ("bill of materials" → BOM), capped at 88 so
   type matches rank below real title hits.

`search_content` — keyword (FTS5, porter-stemmed, AND then OR fallback) or
semantic (embeddings). Both share: default = **latest revision of every doc**
(obsolete-only docs stay searchable, flagged `is_current: false` — hiding
them was a real bug we fixed), per-doc cap of 2 (the 2.9M-char trace matrix
was flooding top-K), 4× over-fetch before filtering.

`read_document` — latest non-obsolete by default. Structured signals, never
guesses: `no_current_revision` (+ ask before falling back), `is_empty`,
`closest_matches`, `too_large` → outline + `section=` param, and even a
single section is capped (`truncated: true`).

`get_references` — in/outbound edges with context excerpts; `resolved: false`
= cited but not in corpus. Direction aliases validated (a bad enum used to
silently no-op — worth retelling).

### 5. `src/agent/embeddings.py` (~5 min)

Each chunk is embedded WITH a context header (`DOC-ID Rev X — Title / §Section`)
so a bare table row carries its provenance into the vector. Normalized
float32 BLOBs in SQLite; query = one numpy dot product — **brute force is
exact**; ANN indexes only earn their complexity past ~100k vectors.
`meta.embed_model` guard refuses to mix vectors from two models (silent
similarity corruption). Backfill is idempotent (`python -m agent.embeddings`).

### 6. `src/agent/prompts.py` + `core.py` + `cli.py` (~8 min)

prompts.py is the domain playbook — read it fully, it's the "product":
vocabulary (incl. **MEMO is a junk drawer** — the SRS/SDS/design reviews are
MEMOs; type filters lie), the five rules (ground everything / cite with
verbatim quotes / counts from catalog only / revision transparency / honesty
over silent defaults), recipes per query pattern, the **search budget** rule
(~3 fruitless rounds → answer with best sources + name the gap; added after
haiku reformulated itself into the recursion limit), and plain-text-only
citations (the model invented `javascript:` links until told).

core.py is 20 lines: `init_chat_model` (provider-agnostic) + the 4 tools +
playbook. Tool docstrings become the tool descriptions the model reads —
they are prompts.

cli.py streams the loop live (`→ search_catalog(...) · 3 results`) — the
demo IS this trace. Errors keep the session alive.

### 7. `evals/` + `tests/` (~5 min)

Two layers: `test_retrieval.py` calls tools directly — deterministic, free,
seconds (the inner loop). `test_agent.py` runs the whole agent — key-gated.
Cases are YAML per query pattern in `evals/cases/`; `tests/` is write-path
unit tests (the parser's corpus traps as named regressions).

The eval-driven war stories (tell these — they're the best evidence the
methodology works):
- Ground truth said 4 all-obsolete docs; the tool said 11. **The eval was
  wrong** — hand-tally missed obsolete singletons. Noted in the YAML.
- "510(k) summary" fuzzy-matched 4 docs on "summary" alone → replaced WRatio
  with token-coverage scoring.
- Replacing the starter's "2+2" test with a real corpus question immediately
  caught the cross-thread SQLite bug.

## Trace two queries end-to-end (do this actively)

1. **"How many ECRs are in the system?"** → agent reads rule 3 → calls
   `search_catalog(doc_type="ECR")` → tool returns `{count: 3, documents:[...]}`
   → agent reports the number it was handed. The model never counts.
2. **"Which protocols trace back to the risk analysis?"** →
   `search_catalog(query="risk analysis")` (finds RSK docs, one flagged
   no-current-revision) → `get_references(doc_id, "inbound")` (edges from
   REFERENCES sections + revision-history tables) → reads to confirm → table
   with per-hop citations. Spot-check any edge yourself:
   `sqlite3 data/index.db "SELECT * FROM refs WHERE to_doc='RSK-P01-010' LIMIT 5"`

## Questions you should be able to answer without notes

- Why four tools and not one search bar? (query patterns break different
  primitives; enumeration can't come from top-N)
- Why is counting in the tool, not the model? Where else does that principle
  show up? (revision resolution, obsolete filtering)
- What happens when the only revision is obsolete? Empty? Not found? Too big?
  (structured signals; agent surfaces, asks, never silently substitutes)
- Why SQLite? What changes at 100× corpus size, and what *doesn't*? (the tool
  interface — store.py is the seam)
- Why did keyword search alone fail the electrical-safety query? (vocabulary
  mismatch — "electrical safety" lives in a "Dielectric and Leakage" report;
  semantic bridges it; the budget rule stops the thrash)
- Why brute-force vectors? (exact, milliseconds at 8k chunks; ANN trades
  recall for speed you don't need below ~100k)
- How do you KNOW citations aren't hallucinated? (by construction from chunk
  labels + layer-2 eval mechanically verifies cited docs/revisions exist)
