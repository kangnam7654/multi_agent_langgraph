from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langgraph.graph import END
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
        model = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
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


# if __name__ == "__main__":
#     model = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
#     model = model.with_structured_output(BaseResponseFormat)

#     response = model.invoke(
#         [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "What is the capital of France?"},
#         ]
#     )
#     pass
