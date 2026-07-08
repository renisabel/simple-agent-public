# MX1 Internal Search — MVP Plan

Make the chat agent useful for searching a 189-doc QMS corpus (MedAI MX1 portable X-ray).
Approach in one line: **an agent with a small toolbox of search primitives — catalog, keyword,
semantic, reference graph — where deterministic things live in tools, judgment lives in the
model, and evals police the boundary.**

## Why a toolbox: each query pattern breaks a different search method

| Query pattern (from the brief) | What answers it |
|---|---|
| Known-item ("find the BOM") | catalog: exact/fuzzy ID + title match, rev resolution |
| Exploratory ("all risk docs") | catalog by type + keyword/semantic for coverage |
| Compliance cross-reference | reference graph + semantic + agent reasoning over gaps |
| Extraction & synthesis | keyword/semantic → read full doc → cite sections |
| Revision & change tracking | catalog rev chains → read both revs → diff |
| Enumeration & counting | catalog, exhaustive — top-N retrieval can never count |

No single method covers the board (generic RAG = keyword + semantic only, and it fails
known-item, enumeration, and traceability). The domain-specific wins are the **catalog**
and the **reference graph** — that's the answer to "why is ChatGPT-style search superficial
here": it can't walk your document graph or count your documents.

## Assumptions

- **Correctness and completeness over latency.** Regulated industry; a wrong or unsourced
  answer is worse than a slow one. Justifies multi-step agentic search.
- **Corpus scale ≈ this one** (hundreds–low-thousands of docs). In-process SQLite is
  right-sized; the tool interface is the seam where real infra (pgvector, Elasticsearch)
  would slot in at scale.
- **Corpus is static during a session** — batch re-index, no live sync.
- **Filenames are a trustworthy metadata source** (`TYPE-PROJ-NUM - Title_Rev[-status].docx`),
  but rev letters are treated as optional so the parser generalizes to messier corpora.
- **"Latest non-obsolete revision" is the default user intent**; historical revs on request.
- **No access control** (single-tenant). Production needs per-user ACLs from the source system.
- **Never answer from model memory** — if retrieval finds nothing, say "not found."

## Architecture

```
query → agent (ReAct loop, LangChain deep agent)
          ├─ search_catalog (query?, type?, status?) → ALL matches + exact count (known-item, enumeration)
          ├─ search_content (query, mode)            → chunks w/ citations attached (keyword | semantic)
          ├─ read_document (doc_id, rev?)            → full text, defaults latest non-obsolete
          └─ get_references (doc_id, direction)      → reference graph (traceability)
        ← answer with [DOC-ID Rev X, §Section] citations + verbatim quotes

offline: data/corpus/*.docx → ingest → SQLite (catalog + FTS5 + embeddings + ref graph)
```

## 1. Pre-processing / ingest (offline, one command)

- **Parse filenames** → doc_id, type, project, number, title, revision, status
  (signed/obsolete). Normalize suffix inconsistencies (`-Obsolete` vs `_Obsolete`, etc.).
- **Extract content** (python-docx): paragraphs + heading styles (docs use real
  Heading1/Heading2 — OBJECTIVE, METHODS, ACCEPTANCE CRITERIA...) → section paths.
- **Tables → markdown**, kept whole inside a chunk, inheriting their section heading.
  Mandatory: some docs (ECRs, BOMs — flattened spreadsheets) are 100% tables.
  Oversized tables split by rows with the header row repeated.
- **Chunking:** by section. The chunk is the unit of everything — we search chunks,
  read chunks, cite chunks. Every chunk carries (doc_id, revision, section_path, index),
  so anything the agent sees arrives pre-labeled: citations by construction.
- **Chunk headers:** prepend "DOC-ID Rev X — Title > §Section" to chunk text before
  embedding, so table rows / short sections embed with their context.
- **Reference graph:** regex-scan text for doc IDs → edges table (who cites whom).
  Protocols have literal REFERENCES sections; this is cheap.
- **Hygiene:** skip empty docs, dedupe by content hash (`..._B(1).docx`), revision
  chains grouped by doc_id. Extract facts **per revision**; resolve "current" at
  query time, never at ingest.
- **Store:** single SQLite file — catalog table, FTS5 (BM25), embeddings, edges.
- **Ingest prints a summary report:** N files → N docs (N revisions, N chains),
  N empty skipped, N duplicates, N chunks, N graph edges. Sanity check for us,
  credibility beat in the demo.

## 2. Search tools

| Tool | Primitive | Notes |
|---|---|---|
| `search_catalog(query?, type?, status?)` | catalog: fuzzy + exhaustive | returns ALL matches (ranked if query given) + exact count; typo-tolerant IDs ("vvpr 151") |
| `search_content(query, mode)` | BM25 or embeddings | separate modes; agent reconciles (no RRF for now) |
| `read_document(doc_id, rev?)` | direct fetch | latest non-obsolete by default |
| `get_references(doc_id, direction)` | graph walk | traceability, compliance cross-ref |

Four tools, not more — fewer tools means fewer wrong-tool choices. Catalog search and
exhaustive listing are one tool because at this scale a catalog query can always return
everything (no top-N cutoff to get counting wrong).

Policy lives **in the tools**: revision resolution, obsolete filtering, counting.
The model reports numbers it was handed; it never counts search results.

## 3. Agent loop

- LangChain `create_deep_agent` ReAct loop: model picks tools, sees results, iterates
  (1 hop for lookups, ~6 for traceability), then answers.
- **Domain system prompt**, not generic: QMS vocabulary (VVPR/ECR/RSK... prefix table),
  citation format `[DOC-ID Rev X, §Section]` + short verbatim quote, per-pattern recipes
  (known-item → catalog first; counts → catalog ALWAYS, report its count; traceability →
  get_references and walk; "what changed" → read both revs and diff).
- **Honesty over silent defaults** — a behavior principle, applied consistently:
  - nothing found → say "not found," show closest matches (never answer from memory)
  - only rev is obsolete → say there's no current version, offer the latest earlier
    non-obsolete rev if one exists
  - ambiguous query ("the risk analysis" matches several RSK docs) → list the candidates
    and ask, rather than guessing one
  Tools detect these situations and return structured signals; the prompt tells the
  agent to surface them. The tool never silently substitutes.
- **Ask vs. assume:** if a safe default exists, state it and proceed ("showing latest
  Rev G; E–F also exist"). Ask a follow-up only when candidates genuinely diverge —
  over-asking is a failure mode too.
- **Visible search trace:** the CLI streams each tool call as it happens
  (`→ search_catalog(type="ECR") · 3 results`). Makes the loop debuggable for us
  and makes the architecture visible in the demo.

## 4. Evals

YAML case files per query pattern; **two shared runners** parametrized over all cases
(the cases differ per pattern; the runners don't — no per-folder test scaffolding):

```
evals/
  cases/
    known_item.yaml  exploratory.yaml  compliance.yaml  extraction.yaml
    revisions.yaml   cross_document.yaml  enumeration.yaml
    # each case: query + expected doc_ids / count / must_cite / rubric
  test_retrieval.py  # layer 1: search returns right docs — fast, deterministic, no LLM
  test_agent.py      # layer 2: end-to-end answer; citation validity checked on EVERY case
```

- Layer 1 is the inner dev loop (seconds, free). Layer 2 runs before checkpoints.
- Citation eval is mechanical: the quoted text must actually appear in the cited doc.
- Enumeration eval checks exact counts programmatically.
- **Headline metrics:** recall@5 on known-item, exact-count rate on enumeration,
  citation-validity rate (quotes verified in source), not-found honesty rate.
  In production the same ideas become telemetry: citation click-through, thumbs,
  rate of answers with zero citations (alarm), not-found rate by customer.
- Eval suite is provider-agnostic (any `provider:model` string), judge model configurable.
- Personas drive eval weighting (Reg Affairs → compliance/enumeration; Quality Eng →
  known-item; Writers → synthesis) — **not** persona-conditioned ranking (no behavioral
  data; same question should give same cited answer).

## Edge cases (all present in this corpus)

- **Empty docs: 24 files (13%), including EVERY revision of some docs** (IFU-MX1, both
  revs). Stay in catalog flagged `is_empty`, no chunks → "exists but empty," never
  "not found."
- **Giant docs:** trace matrix is 2.9M chars → `read_document` returns outline +
  requested section, never the whole doc (context-window guard).
- **22 substantive docs have no heading styles** (incl. both 3P reports, most PLNs) →
  fallback paragraph-window chunking, `section_path="body"`, citation = doc + quote.
- **MEMO is a junk drawer:** SRS, SDS, architecture, and the Phase Design Reviews are
  all MEMOs; only one doc has the DR prefix. Vocabulary table must say so — design
  reviews ≠ `type=DR`.
- **File dates unusable** (synthetic/missing core properties) → date queries ("ECRs
  filed last year") need dates parsed from content, not file metadata.
- Titles change across revisions (ESF-P01-003, BOM-079) → title is per-revision;
  chain search matches any revision's title.
- Latest rev is Obsolete with no successor (ESF-P01-003 G) → the "honesty over silent
  defaults" behavior above
- Rev-letter order ≠ chronology when titles carry product versions (BOM-079 P=v3.4, M=v4.0)
- Gaps in rev chains (IFU-MX1: D → L); duplicate file `(1).docx`; suffix
  inconsistencies; filename typos ("Diagnositc")
- Nonexistent docs (a 510(k) summary may not exist) → honest not-found + closest matches
- Dangling references to P00 predecessor docs not in corpus

Extra query types the corpus supports (eval stretch cases): status inventory ("which
docs are unsigned / obsolete without replacement" — pure catalog), software-version
scoped ("what did we test for v3.3.0" — version strings extracted from titles at
ingest), test-outcome synthesis ("which verification tests failed"), group-by
aggregates ("latest rev of every VVPR").

## Alternatives considered (and why not)

- **One-shot RAG** (retrieve once, stuff prompt, answer) — can't follow reference chains
  or enumerate; those need iteration. Cost of agentic: latency — accepted per assumption #1.
- **Hardcoded query router** (classify query → fixed pipeline per pattern) — predictable,
  but brittle on queries that straddle patterns ("how many protocols trace to the risk
  file" = enumeration + traceability), and can't recover mid-search.
- **Agent skills / progressive prompt loading** — our whole playbook is ~1–2k tokens;
  skills would save nothing and add a triggering failure mode (skill not loaded → silently
  worse search). Policy belongs in the always-loaded prompt *because* it can't fail to load.
- **Vector DB service (Pinecone/Weaviate/etc.)** — infrastructure with no payoff at 189
  docs; SQLite is zero-setup and demo-reliable. The tool interface is the swap seam.

## Punted (with the path back)

- **RRF score fusion** — keyword/semantic stay separate tools; add fusion if a single
  `search()` tool proves better.
- **Reranker, multi-query/HyDE, doc-summary index** — scale features, not needed at 189 docs.
- **Persona-conditioned ranking** — needs usage data; personas used for eval coverage
  instead. Also: two users asking the same question should get the same cited answer.

## If the corpus were 100× bigger

| Scale | Store | What else changes |
|---|---|---|
| ~200 docs (now) | SQLite FTS5 + in-process embeddings | — |
| 10k–100k | Postgres + pgvector (or Qdrant) | incremental sync from connectors; ACL filtering at query time |
| 100k+ multi-tenant | Elasticsearch/OpenSearch hybrid | + cross-encoder reranker; per-tenant indexes; usage-data ranking |

The agent and its tool interface don't change — only the backend behind the tools.

## Build order

1. Ingest: filename parser → docx extraction → catalog + FTS5 (known-item + enumeration work)
2. Tools wired into agent + domain system prompt → first end-to-end demo query
3. Embeddings + semantic mode; reference graph
4. Eval folders + cases across patterns; iterate against them
5. DESIGN.md; README quickstart (`uv run index` → `uv run chat` → `uv run pytest`);
   demo script (below); web frontend if time

## Demo script (3:00) — one query per pattern, ending on honesty

1. Known-item: "Find the Bill of Materials for the MX1" → latest rev, notes chain E/F/G
2. Enumeration: "How many ECRs are in the system?" → exact count + list
3. Traceability: "Which verification protocols trace back to the risk analysis?" → graph walk
4. Extraction: "Acceptance criteria for the electrical safety test?" → cited quote from §
5. Honesty: "Where is the 510(k) summary?" → not-found + closest matches (feature, not failure)

## If core lands early — where extra time goes (ranked)

1. **Eval results table** — run the suite, report the headline metrics with real numbers.
   "Recall@5 = 0.9, citation validity = 100%" beats any additional feature.
2. **Clickable citations in the web frontend** — citation opens the source chunk.
   Makes the citations story visceral for the demo.
3. **Failure analysis** — 3–4 queries that still fail and *why*, in DESIGN.md.
   Knowing your system's edges is the strongest maturity signal.
4. **Compliance gap analysis polish** — the 21 CFR 820.30 query end-to-end (hardest
   pattern). Architecture already supports it; the gap is knowledge, not machinery:
   add a curated checklist at `data/reference/cfr-820-30.md` (citable, vs. model
   memory), a compliance recipe in the prompt, and one eval case asserting a known
   gap is reported. Corpus even has its own `DHF-008` checklist to cross-walk.
5. **Cost/latency numbers per query pattern** — tokens and seconds, ties back to
   the correctness-over-speed assumption with data.
