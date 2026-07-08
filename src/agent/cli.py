import argparse
import json
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from agent.core import make_agent

DIM = "\033[2m"
RESET = "\033[0m"

# Traceability walks legitimately take many hops; don't let LangGraph's
# default (25) kill a deep search.
RECURSION_LIMIT = 60
_CONFIG = {"recursion_limit": RECURSION_LIMIT}


def _content_text(msg) -> str:
    """Message content as plain text — Anthropic models sometimes return a
    list of content blocks instead of a string."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def _compact_args(args: dict, limit: int = 90) -> str:
    s = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _result_summary(msg: ToolMessage) -> str:
    """One-line gist of a tool result: counts beat character dumps."""
    try:
        data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
    except (json.JSONDecodeError, TypeError):
        return f"{len(str(msg.content)):,} chars"
    if not isinstance(data, dict):
        return f"{len(str(data)):,} chars"
    for key in ("error", "note"):
        if key in data:
            return str(data[key])[:80]
    if "count" in data:
        return f"{data['count']} results"
    if "cited_by" in data or "references" in data:
        parts = [f"{k}: {len(data[k])}" for k in ("references", "cited_by") if k in data]
        return ", ".join(parts)
    if "text" in data:
        return f"{data.get('doc_id')} Rev {data.get('revision')}, {data.get('char_count', 0):,} chars"
    return f"{len(str(data)):,} chars"


def _stream_turn(agent, messages: list) -> list:
    """Run one agent turn, printing tool calls as they happen.

    Returns the full updated message list from the final state.
    """
    state = {"messages": messages}
    for step in agent.stream(state, stream_mode="values", config=_CONFIG):
        new = step["messages"][len(messages):]
        for msg in new:
            if isinstance(msg, AIMessage):
                for call in msg.tool_calls:
                    print(f"{DIM}  → {call['name']}({_compact_args(call['args'])}){RESET}")
            elif isinstance(msg, ToolMessage):
                print(f"{DIM}    · {_result_summary(msg)}{RESET}")
        messages = step["messages"]
    return messages


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MedAI QMS search assistant")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument("--system", default=None, help="Custom system prompt")
    parser.add_argument("--quiet", action="store_true", help="Hide the tool-call trace")
    args = parser.parse_args()

    agent = make_agent(args.model, args.system)
    messages = []

    print("MedAI QMS search. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})
        try:
            if args.quiet:
                messages = agent.invoke({"messages": messages}, config=_CONFIG)[
                    "messages"
                ]
            else:
                messages = _stream_turn(agent, messages)
        except Exception as e:  # keep the session alive on API hiccups
            print(f"\n[error] {e}\n", file=sys.stderr)
            messages.pop()
            continue

        print(f"\nAssistant: {_content_text(messages[-1])}\n")


if __name__ == "__main__":
    main()
