"""Layer-1 evals: the search tools return the right documents.

Deterministic and LLM-free — this is the inner dev loop. Run with:

    uv run pytest evals/test_retrieval.py -q
"""

import os

import pytest

from evals.helpers import case_params

TOP_K = 5  # a known-item hit must appear in the first TOP_K catalog results


@pytest.mark.parametrize("case", case_params("known_item"))
def test_known_item(tools, case):
    expect = case["expect"]
    result = tools.search_catalog(query=case["query"])
    top = [d["doc_id"] for d in result["documents"][:TOP_K]]

    if expect.get("none"):
        from evals.helpers import record

        record("known_item_honest_notfound", 1.0 if result["count"] == 0 else 0.0)
        assert result["count"] == 0, f"expected no matches, got {top}"
        return

    from evals.helpers import record

    hit = all(doc_id in top for doc_id in expect["doc_ids"])
    record("known_item_recall_at_5", 1.0 if hit else 0.0)
    for doc_id in expect["doc_ids"]:
        assert doc_id in top, f"{doc_id} not in top {TOP_K}: {top}"

    hit = next(d for d in result["documents"] if d["doc_id"] == expect["doc_ids"][0])
    if "latest_current_revision" in expect:
        assert hit["latest_current_revision"] == expect["latest_current_revision"]
    if expect.get("no_current_revision"):
        assert hit["latest_current_revision"] is None
    if expect.get("all_revisions_empty"):
        assert all(r["is_empty"] for r in hit["revisions"])


def test_semantic_vocabulary_mismatch(tools):
    """The reason semantic search exists: 'electrical safety' never appears in
    the dielectric/leakage report's text, but must find it anyway."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set (semantic mode needs embeddings)")
    result = tools.search_content(
        "electrical safety test results for the device", mode="semantic", top_k=5
    )
    if "error" in result:
        pytest.skip(result["error"])
    docs = [r["doc_id"] for r in result["results"]]
    assert "3P-P01-33" in docs, f"dielectric report not in top 5: {docs}"


def test_obsolete_only_docs_are_searchable(tools):
    """Docs whose every revision is obsolete (e.g. the risk assessment) must
    still be findable by default content search — flagged, not hidden."""
    result = tools.search_content("hazard severity occurrence", top_k=10)
    hits = {r["doc_id"]: r["is_current"] for r in result["results"]}
    assert "RSK-P01-010" in hits, f"risk assessment invisible: {list(hits)}"
    assert hits["RSK-P01-010"] is False  # flagged as not current


@pytest.mark.parametrize("case", case_params("enumeration"))
def test_enumeration(tools, case):
    expect = case["expect"]
    result = tools.search_catalog(**case["filter"])
    ids = [d["doc_id"] for d in result["documents"]]

    if "count" in expect:
        from evals.helpers import record

        record("enumeration_count_exact", 1.0 if result["count"] == expect["count"] else 0.0)
        assert result["count"] == expect["count"], f"got {result['count']}: {ids}"
    if "doc_ids" in expect:
        assert sorted(ids) == sorted(expect["doc_ids"])
    if "doc_ids_include" in expect:
        for doc_id in expect["doc_ids_include"]:
            assert doc_id in ids, f"{doc_id} missing from {ids}"
