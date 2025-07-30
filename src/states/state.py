from typing import Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.messages.custom_messages import (
    DirectorMessage,
    InspectorMessage,
    WriterMessage,
)


class AgentState(TypedDict):
    task: str
    messages: list[
        HumanMessage | SystemMessage | ToolMessage | AIMessage | DirectorMessage | InspectorMessage | WriterMessage
    ]
    current_agent: str
    director_can_publish: bool
    inspector_passed: bool
    writer_scenario: str
    next_agent: str
    revision: int
    max_revision: int
    tool_calls: list[Callable]
