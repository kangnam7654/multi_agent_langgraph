import json
import re
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.types import Command

from src.agents.base_agent import CustomBaseAgent
from src.messages.custom_messages import WriterMessage


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
        # self.model = self.model.with_structured_output(WriterMessage)  # type: ignore

    def __call__(self, state: dict[str, Any]) -> Command[Any]:
        revision: int = state["revision"]
        if self.has_reached_max_revisions(state):
            return Command(goto="inspector", update={"messages": state["messages"], "revision": revision})
        messages: list = state["messages"]

        messages_with_system_message = [SystemMessage(content=self.system_prompt)] + messages

        try:
            print(f"Attempting to invoke model with {len(messages_with_system_message)} messages")
            print(f"System prompt length: {len(self.system_prompt)}")

            # Get raw response
            raw_response = self.model.invoke(messages_with_system_message + [f'last scenario{state.get("scenario")}'])

            # Parse JSON from response
            content = raw_response.content
            try:
                # Try to extract JSON from the response
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_data = json.loads(json_str)
                else:
                    # Fallback data
                    parsed_data = {"type": "writer", "scenario": content, "content": "Scenario generated"}

                # Create WriterMessage object
                response = WriterMessage(
                    type=parsed_data.get("type", "writer"),
                    scenario=parsed_data.get("scenario", content),
                    content=parsed_data.get("content", "Scenario generated"),
                )

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Failed to parse JSON, using fallback: {e}")
                response = WriterMessage(type="writer", scenario=content, content="Scenario generated (fallback)")

        except Exception as e:
            print(f"Error invoking model: {e}")
            print(
                f"Messages: {[msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100] for msg in messages_with_system_message]}"
            )
            # Create fallback response
            response = WriterMessage(
                type="writer", scenario="Error generating scenario. Please try again.", content="Error occurred"
            )

        messages.append(response)
        revision += 1
        to_update = {
            "messages": messages,
            "scenario": response.scenario,
            "current_agent": response.type.lower(),
            "next_agent": "inspector",
            "revision": revision,
        }
        return Command(update=to_update)
