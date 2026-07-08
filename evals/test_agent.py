"""Layer-2 evals: the full agent answers correctly, with valid citations.

Every case with an `expect_answer` block in evals/cases/*.yaml runs here.
Checks, in order of mechanical certainty:

1. answer_contains — exact substrings (counts, doc IDs) present
2. must_cite / require_citations — citation-formatted references present
3. citation validity — EVERY parsed citation names a real (doc, revision)
   in the index; a fabricated citation fails the case
4. rubric — LLM judge, only where correctness is qualitative

Requires ANTHROPIC_API_KEY (agent + judge run on a cheap model; override via
EVAL_AGENT_MODEL / EVAL_JUDGE_MODEL). Run just this layer:

    uv run pytest evals/test_agent.py -q
"""

import pytest

from evals.conftest import JUDGE_MODEL
from evals.helpers import (
    agent_case_params,
    citation_is_valid,
    judge,
    parse_citations,
    record,
)


@pytest.mark.parametrize("case", agent_case_params())
def test_agent_case(run_agent, index_con, case):
    expect = case["expect_answer"]
    answer = run_agent(case["query"])
    problems = []

    for needle in expect.get("answer_contains", []):
        if needle.lower() not in answer.lower():
            problems.append(f"missing expected content {needle!r}")

    citations = parse_citations(answer)
    cited_docs = {doc for doc, _ in citations}

    if expect.get("require_citations") and not citations:
        problems.append("no citations found in answer")
    for doc_id in expect.get("must_cite", []):
        if doc_id not in cited_docs:
            problems.append(f"expected citation of {doc_id}, cited: {sorted(cited_docs)}")

    invalid = [
        f"{d} Rev {r}" for d, r in citations if not citation_is_valid(d, r, index_con)
    ]
    record("citation_validity", 1.0 if not invalid else 0.0)
    if invalid:
        problems.append(f"citations of nonexistent doc/revision: {invalid}")

    if "rubric" in expect:
        passed, verdict = judge(answer, expect["rubric"], JUDGE_MODEL)
        record("rubric_pass", 1.0 if passed else 0.0)
        if not passed:
            problems.append(f"judge: {verdict}")

    record("agent_case_pass", 0.0 if problems else 1.0)
    assert not problems, "\n".join(problems) + f"\n\nANSWER:\n{answer[:1500]}"
