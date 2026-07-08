import os

import pytest
from dotenv import load_dotenv

from evals.helpers import METRICS

load_dotenv()

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "anthropic:claude-haiku-4-5-20251001")
AGENT_MODEL = os.environ.get("EVAL_AGENT_MODEL", "anthropic:claude-haiku-4-5-20251001")


@pytest.fixture(scope="session")
def tools():
    """The search tools module, pointed at the built index (skip if absent)."""
    from agent import store
    from agent.search import tools as t

    if not store.DEFAULT_DB.exists():
        pytest.skip("search index not built — run `uv run index`")
    return t


@pytest.fixture(scope="session")
def index_con(tools):
    from agent import store

    return store.connect(store.DEFAULT_DB)


@pytest.fixture(scope="session")
def run_agent(tools):
    """Session-cached agent runner: query -> final answer text."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — layer-2 evals need the agent")

    from agent.cli import _content_text
    from agent.core import make_agent

    # No temperature override: Claude 5-family models reject the parameter,
    # and measured run-to-run variance was no better at temp 0 anyway.
    agent = make_agent(AGENT_MODEL)
    cache: dict[str, str] = {}

    def run(query: str) -> str:
        if query not in cache:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": 60},
            )
            cache[query] = _content_text(result["messages"][-1])
        return cache[query]

    return run


def pytest_terminal_summary(terminalreporter):
    """The eval scorecard — headline metrics across the whole run."""
    if not METRICS:
        return
    tr = terminalreporter
    tr.section("eval metrics")
    for name in sorted(METRICS):
        values = METRICS[name]
        rate = sum(values) / len(values)
        tr.write_line(f"  {name:32} {rate:6.1%}  ({int(sum(values))}/{len(values)})")
