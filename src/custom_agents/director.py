from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from src.custom_agents.base_agent import CustomBaseAgent
from src.custom_messages.custom_messages import DirectorMessage


class AgentDirector(CustomBaseAgent):
    def __init__(
        self,
        name: str | None = None,
        system_prompt: str | None = None,
        *,
        tools: list[Callable] | None = None,
        model: BaseChatModel | None = None,
    ):
        """Initialize the AgentDirector"""
        if name is None:
            name = "Director"
        if system_prompt is None:
            system_prompt = "You are a helpful assistant named Director tasked with directing to inspector and story writer to create a immersive story."

        super().__init__(name=name, system_prompt=system_prompt, model=model, tools=tools)
        self.model = self.model.with_structured_output(DirectorMessage)  # type: ignore

    def __call__(self, state: dict[str, Any]) -> Command[Any]:
        """Invoke the director agent with the current state."""

        # | Get Revision |
        revision = state.get("revision", 0)

        # | Check Max Revisions |
        if self.has_reached_max_revisions(state):
            return Command(goto=END, update={"messages": state["messages"], "revision": revision})

        # | Prepare Messages |
        if revision == 0:
            messages = [HumanMessage(content=state["task"])]
        else:
            messages = state["messages"]

        messages_with_system_message = [SystemMessage(content=self.system_prompt)] + messages
        # response: DirectorMessage = self.model.invoke(messages_with_system_message)  # type: ignore
        try:
            print(f"Attempting to invoke model with {len(messages_with_system_message)} messages")
            print(f"System prompt length: {len(self.system_prompt)}")
            response: DirectorMessage = self.model.invoke(messages_with_system_message)  # type: ignore
        except Exception as e:
            print(f"Error invoking model: {e}")
            print(
                f"Messages: {[msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100] for msg in messages_with_system_message]}"
            )
            raise
        converted = response
        messages.append(converted)  # type: ignore

        revision += 1
        to_update = {
            "messages": messages,
            "passed": response.passed,
            "feedbacks": response.feedbacks if not response.passed else {},
            "current_agent": response.type.lower(),
            "next_agent": response.next_agent,
            "revision": revision,
        }
        if response.passed:
            return Command(goto=END, update=to_update)
        return Command(goto=response.next_agent, update=to_update)
