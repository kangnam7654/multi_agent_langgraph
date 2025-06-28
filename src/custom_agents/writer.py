from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from src.custom_agents.base_agent import CustomBaseAgent
from src.custom_messages.custom_messages import WriterMessage


class AgentWriter(CustomBaseAgent):
    def __init__(
        self,
        name: str | None = None,
        system_prompt: str | None = None,
        *,
        tools: list[Callable] | None = None,
        model: BaseChatModel | None = None,
    ):
        """Initialize the AgentScenarioWriter"""
        if name is None:
            name = "Scenario Writer"
        if system_prompt is None:
            system_prompt = "You are a helpful assistant named Scenario Writer tasked with writing the story based on the provided outline."

        super().__init__(name=name, system_prompt=system_prompt, model=model, tools=tools)
        self.model = self.model.with_structured_output(WriterMessage)  # type: ignore

    def __call__(self, state: dict[str, Any]) -> Command[Any]:  # type: ignore
        revision: int = state["revision"]
        if self.has_reached_max_revisions(state):
            return Command(goto="inspector", update={"messages": state["messages"], "revision": revision})
        messages: list = state["messages"]

        messages_with_system_message = [SystemMessage(content=self.system_prompt)] + messages
        # response: WriterMessage = self.model.invoke(messages_with_system_message)  # type: ignore

        try:
            print(f"Attempting to invoke model with {len(messages_with_system_message)} messages")
            print(f"System prompt length: {len(self.system_prompt)}")
            response: WriterMessage = self.model.invoke(messages_with_system_message)  # type: ignore
        except Exception as e:
            print(f"Error invoking model: {e}")
            print(
                f"Messages: {[msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100] for msg in messages_with_system_message]}"
            )
            raise

        messages.append(response)  # type: ignore
        revision += 1
        to_update = {
            "messages": messages,
            "current_agent": response.type.lower(),
            "next_agent": response.next_agent,
            "revision": revision,
            "scenario": response.scenario,
        }
        return Command(goto=response.next_agent, update=to_update)
