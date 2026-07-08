from langchain.chat_models import init_chat_model

from deepagents import create_deep_agent

from agent.prompts import SYSTEM_PROMPT
from agent.search.tools import (
    get_references,
    read_document,
    search_catalog,
    search_content,
)

SEARCH_TOOLS = [search_catalog, search_content, read_document, get_references]


def make_agent(
    model_str: str = "anthropic:claude-haiku-4-5-20251001",
    system_prompt: str | None = None,
    temperature: float | None = None,
):
    """Create the QMS search agent.

    Args:
        model_str: Provider and model in "provider:model" format.
                   Examples: "openai:gpt-4o", "anthropic:claude-haiku-4-5-20251001",
                   "google_genai:gemini-2.5-flash"
        system_prompt: Optional system prompt override (defaults to the QMS
                       search playbook in agent.prompts).
        temperature: Optional sampling temperature. Leave None for provider
                     defaults — some models (Claude 5 family) reject it.

    Returns:
        A compiled LangGraph agent supporting .invoke(), .stream(), .astream().
    """
    kwargs = {} if temperature is None else {"temperature": temperature}
    model = init_chat_model(model_str, **kwargs)
    return create_deep_agent(
        model=model,
        tools=SEARCH_TOOLS,
        system_prompt=system_prompt or SYSTEM_PROMPT,
    )
