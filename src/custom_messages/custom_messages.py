from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END
from pydantic import Field


class CustomBaseMessage(AIMessage):
    """Customized base message class"""

    next_agent: str = Field(description="The next agent to call.")

    def __init__(self, content, **kwargs):
        """Initialize the custom message with additional fields."""
        super().__init__(content=content, **kwargs)


class DirectorMessage(CustomBaseMessage):
    """Custom response format for the director agent."""

    type: Literal["director"] = "director"  # type: ignore
    next_agent: str = Field(
        default=END, description="The next agent to call after the director.", examples=["writer", "inspector", END]
    )
    passed: bool = Field(default=False, description="Indicates whether the scenario can publish.")
    feedbacks: dict[str, Any] = Field(
        description="Feedbacks from the director. Feedback should be provided when `passed` is False."
    )

    def __init__(self, content, **kwargs):
        return super().__init__(content=content, **kwargs)


class WriterMessage(CustomBaseMessage):
    """Response format for the scenario writer agent."""

    next_agent: Literal["inspector"] = "inspector"  # type: ignore
    type: Literal["writer"] = "writer"  # type: ignore
    scenario: str = Field(description="The generated scenario text.")

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)


class InspectorMessage(CustomBaseMessage):
    """Response format for the inspector agent."""

    next_agent: str = Field(description="The next agent to call after the inspector.", examples=["director", "writer"])
    type: Literal["inspector"] = "inspector"  # type: ignore
    passed: bool = Field(default=False, description="Indicates whether the scenario passed inspection.")
    feedbacks: dict[str, Any] = Field(
        description="Feedbacks from the inspector. Feedback should be provided when `passed` is False."
    )

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)
