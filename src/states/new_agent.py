from typing import Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class AgentState(TypedDict):
    messages: list[HumanMessage | SystemMessage | ToolMessage | AIMessage]
    current_agent: str
    next_agent: str
    revision: int
    max_revision: int
    tool_calls: list[Callable]
