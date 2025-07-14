from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langgraph.types import Command


class CustomBaseAgent(ABC):
    def __init__(
        self,
        name: str,
        system_prompt: str | None = None,
        *,
        model: BaseChatModel | None = None,
        tools: list[Callable] | None = None,
    ):
        """Initialize the CustomBaseAgent"""
        self.name: str = name
        self.model: BaseChatModel | ChatGroq | None = model or self.load_default_model()
        self.tools: list[Callable] | None = tools
        self.system_prompt: str | None = system_prompt

    @abstractmethod
    def __call__(self, state: Any) -> Command[Any]:
        """Define the call method for the agent."""
        pass

    def load_default_model(self) -> ChatGroq:
        model = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
        return model

    def bind_tools(self, tools: list[Callable]) -> None:
        """Bind tools to the model."""
        self.model.bind_tools(tools)  # type: ignore

    def _default_system_prompt(self) -> str:
        """Return a default system prompt if none is provided."""
        return f"You are a helpful assistant named {self.name}."

    def has_reached_max_revisions(self, state: dict[str, Any]) -> bool:
        """Return the maximum revision number for the agent."""
        return state.get("revision", 0) >= state.get("max_revision", 10)
