from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END
from pydantic import Field


class CustomBaseMessage(AIMessage):
    """Customized base message class"""

    def __init__(self, content, **kwargs):
        """Initialize the custom message with additional fields."""
        super().__init__(content=content, **kwargs)


class DirectorMessage(CustomBaseMessage):
    """Custom response format for the director agent."""

    director_can_publish: bool = Field(default=False, description="Indicates whether the scenario can publish.")

    def __init__(self, content, director_can_publish, next_agent=END, **kwargs):
        super().__init__(content=content, **kwargs)
        self.director_can_publish = director_can_publish
        self._next_agent = next_agent

    @property
    def next_agent(self):
        """Return the next agent to handle the scenario."""
        return self._next_agent

    @next_agent.setter
    def next_agent(self, value):
        """Set the next agent to handle the scenario."""
        if value not in ["writer", END]:
            raise ValueError("Next agent must be 'writer' or END.")
        self._next_agent = value


class WriterMessage(CustomBaseMessage):
    """Response format for the scenario writer agent."""

    writer_scenario: str = Field(default="", description="The generated scenario text.")

    def __init__(self, content, writer_scenario="", next_agent="inspector", **kwargs):
        super().__init__(content=content, **kwargs)
        self.writer_scenario = writer_scenario
        self._next_agent = next_agent

    @property
    def next_agent(self):
        """Return the next agent to handle the scenario."""
        return self._next_agent

    @next_agent.setter
    def next_agent(self, value):
        """Set the next agent to handle the scenario."""
        if value not in ["inspector", "self"]:
            raise ValueError("Next agent must be 'inspector' or END.")
        self._next_agent = value


class InspectorMessage(CustomBaseMessage):
    """Response format for the inspector agent."""

    inspector_passed: bool = Field(default=False, description="Indicates whether the scenario passed inspection.")

    def __init__(self, content, inspector_passed: bool, next_agent="director", **kwargs):
        super().__init__(content=content, **kwargs)
        self.inspector_passed = inspector_passed
        self._next_agent = next_agent

    @property
    def next_agent(self):
        """Return the next agent to handle the scenario."""
        return self._next_agent

    @next_agent.setter
    def next_agent(self, value):
        """Set the next agent to handle the scenario."""
        if value not in ["director", "writer", "self"]:
            raise ValueError("Next agent must be 'director' or END.")
        self._next_agent = value
