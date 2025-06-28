from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class AgentState(TypedDict):
    task: str
    messages: list[HumanMessage | SystemMessage | ToolMessage | AIMessage]
    current_agent: str
    next_agent: str
    allow_tools: bool
    revision: int
    max_revision: int
    scenario: str | None
