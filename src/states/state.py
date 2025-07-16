from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class AgentState(TypedDict):
    task: str
    messages: list[HumanMessage | SystemMessage | ToolMessage | AIMessage]
    current_agent: str
    director_can_publish: bool
    inspector_passed: bool
    writer_scenario: str | None
    next_agent: str | None
    revision: int
    max_revision: int
