from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class AgentState(TypedDict):
    task: str
    messages: list[HumanMessage | SystemMessage | ToolMessage | AIMessage]
    scenario: str | None
    current_agent: str
    passed: bool
    can_publish: bool
    next_agent: str | None
    revision: int
    max_revision: int
