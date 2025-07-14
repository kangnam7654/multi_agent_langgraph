from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage
from pydantic import Field


class CustomBaseMessage(AIMessage):
    """Customized base message class"""

    def __init__(self, content, **kwargs):
        """Initialize the custom message with additional fields."""
        super().__init__(content=content, **kwargs)


class DirectorMessage(CustomBaseMessage):
    """Custom response format for the director agent."""

    type: Literal["director"] = "director"  # type: ignore
    can_publish: bool = Field(default=False, description="Indicates whether the scenario can publish.")

    def __init__(self, content, **kwargs):
        return super().__init__(content=content, **kwargs)


class WriterMessage(CustomBaseMessage):
    """Response format for the scenario writer agent."""

    type: Literal["writer"] = "writer"  # type: ignore
    scenario: str = Field(description="The generated scenario text.")

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)


class InspectorMessage(CustomBaseMessage):
    """Response format for the inspector agent."""

    type: Literal["inspector"] = "inspector"  # type: ignore
    passed: bool = Field(default=False, description="Indicates whether the scenario passed inspection.")

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)
