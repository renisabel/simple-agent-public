"""Shared eval utilities: case loading, citation checks, metrics collection."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

CASES_DIR = Path(__file__).parent / "cases"

# [BOM-055 Rev G] / [VVPR-P01-151 Rev B, §ACCEPTANCE CRITERIA]
CITATION_RE = re.compile(r"\[([A-Z0-9][A-Za-z0-9-]*)\s+Rev\.?\s+([A-Z0-9.]+)")

# metric name -> list of 0/1 observations; printed as a table at session end
METRICS: dict[str, list[float]] = {}


def record(metric: str, value: float) -> None:
    METRICS.setdefault(metric, []).append(value)


def load_cases(pattern: str) -> list[dict]:
    """Load one pattern's case file, e.g. load_cases("known_item")."""
    path = CASES_DIR / f"{pattern}.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def case_params(pattern: str):
    """Layer-1 (case, id) params: only cases with an `expect` block —
    agent-only cases (expect_answer) run in test_agent.py instead."""
    import pytest

    return [pytest.param(c, id=c["id"]) for c in load_cases(pattern) if "expect" in c]


def agent_case_params():
    """All cases across every pattern file that define expect_answer."""
    import pytest

    params = []
    for path in sorted(CASES_DIR.glob("*.yaml")):
        for case in yaml.safe_load(path.read_text()) or []:
            if "expect_answer" in case:
                params.append(
                    pytest.param(case, id=f"{path.stem}:{case['id']}")
                )
    return params


def parse_citations(answer: str) -> list[tuple[str, str]]:
    """[(doc_id, revision), ...] found in citation-formatted brackets."""
    return [(m.group(1), m.group(2)) for m in CITATION_RE.finditer(answer)]


def citation_is_valid(doc_id: str, revision: str, con) -> bool:
    """A citation is valid iff that exact (doc, revision) exists in the index."""
    return (
        con.execute(
            "SELECT 1 FROM documents WHERE doc_id = ? AND revision = ?",
            (doc_id, revision),
        ).fetchone()
        is not None
    )


def judge(answer: str, rubric: str, model_str: str) -> tuple[bool, str]:
    """LLM judge: does the answer satisfy the rubric? Returns (passed, raw)."""
    from langchain.chat_models import init_chat_model

    model = init_chat_model(model_str)
    verdict = model.invoke(
        "You are grading a search assistant's answer against a rubric.\n\n"
        f"RUBRIC: {rubric}\n\nANSWER:\n{answer}\n\n"
        "Does the answer satisfy the rubric? Reply with exactly PASS or FAIL "
        "on the first line, then one sentence of reasoning."
    )
    text = str(verdict.content).strip()
    return text.upper().startswith("PASS"), text
