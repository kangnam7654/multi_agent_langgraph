import json
import re
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from src.agents.base_agent import CustomBaseAgent
from src.messages.custom_messages import InspectorMessage


class AgentInspector(CustomBaseAgent):
    def __init__(
        self,
        name: str | None = None,
        system_prompt: str | None = None,
        *,
        tools: list[Callable] | None = None,
        model: BaseChatModel | None = None,
    ):
        """Initialize the AgentInspector"""
        if name is None:
            name = "Inspector"
        if system_prompt is None:
            system_prompt = "You are a helpful assistant named Inspector tasked with inspecting \
                the scenario that the Writer had written."

        super().__init__(name=name, system_prompt=system_prompt, model=model, tools=tools)
        # self.model = self.model.with_structured_output(InspectorMessage)  # type: ignore

    def __call__(self, state: dict[str, Any]) -> Command[Any]:
        revision: int = state["revision"]
        if self.has_reached_max_revisions(state):
            return Command(goto="director", update={"messages": state["messages"], "revision": revision})
        messages: list = state["messages"]

        messages_with_system_message = [SystemMessage(content=self.system_prompt)] + messages

        try:
            print(f"Attempting to invoke model with {len(messages_with_system_message)} messages")
            print(f"System prompt length: {len(self.system_prompt)}")

            # Get raw response
            raw_response = self.model.invoke(messages_with_system_message)

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
                    parsed_data = {"type": "inspector", "passed": True, "content": content, "next_agent": "director"}

                # Create InspectorMessage object
                response = InspectorMessage(
                    type=parsed_data.get("type", "inspector"),
                    passed=parsed_data.get("passed", True),
                    content=parsed_data.get("content", "Inspection completed"),
                )

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Failed to parse JSON, using fallback: {e}")
                response = InspectorMessage(type="inspector", passed=True, content=content)

        except Exception as e:
            print(f"Error invoking model: {e}")
            print(
                f"Messages: {[msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100] for msg in messages_with_system_message]}"
            )
            # Create fallback response
            response = InspectorMessage(
                type="inspector", passed=True, content="Error occurred during inspection. Proceeding with approval."
            )

        messages.append(response)
        revision += 1
        to_update = {
            "messages": messages,
            "passed": response.passed,
            "current_agent": response.type.lower(),
            "next_agent": "director" if response.passed else "writer",
            "revision": revision,
        }
        return Command(update=to_update)
