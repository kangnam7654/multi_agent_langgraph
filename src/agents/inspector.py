import json
import logging
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from src.agents.base_agent import CustomBaseAgent
from src.messages.custom_messages import InspectorMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AgentInspector(CustomBaseAgent):
    def __init__(
        self,
        name: str | None = None,
        system_prompt: str | None = None,
        model: BaseChatModel | None = None,
        last_n_messages: int = 3,
        max_tries: int = 5,
    ):
        """Initialize the AgentInspector"""
        if name is None:
            name = "inspector"
        if system_prompt is None:
            system_prompt = "You are a helpful assistant named Inspector tasked with inspecting \
                the scenario that the Writer had written."

        super().__init__(name=name, system_prompt=system_prompt, model=model, last_n_messages=last_n_messages)
        # self.model = self.model.with_structured_output(InspectorMessage)  # type: ignore
        self.max_tries = max_tries

    def __call__(self, state: dict[str, Any]) -> Command[Any]:
        revision: int = state["revision"] + 1

        # Check Max Revisions
        if self.has_reached_max_revisions(state):
            to_update = {
                "task": state.get("task", ""),
                "meesages": state["messages"].append(
                    InspectorMessage(content="MAX REVISION REACHED.", inspector_passed=False)
                ),
                "director_can_publish": state.get("director_can_publish", False),
                "inspector_passed": False,
                "writer_scenario": state.get("writer_scenario", ""),
                "current_agent": "inspector",
                "next_agent": "director",
                "revision": revision,
                "max_revision": self.max_revision,
            }
            return Command(update=to_update)

        # Prepare Message
        messages: list = state["messages"]
        task = messages[0]
        if len(messages) > self.last_n_messages:
            messages = messages[-self.last_n_messages :]
        messages_with_system_message = [SystemMessage(content=self.system_prompt), task] + messages
        current_try = 0
        parsed_data = {}

        while current_try < self.max_tries:
            raw_response = self.model.invoke(messages_with_system_message)
            content = raw_response.content

            # Parse JSON from response
            try:
                # Try to extract JSON from the response
                parsed_data = self.extract_json_from_content(content)
                break
            except (json.JSONDecodeError, AttributeError) as e:
                current_try += 1

        if not parsed_data:
            if not state.get("writer_scenario"):
                parsed_data = {
                    "content": "WARNING: No scenario provided by the writer. Write again.",
                    "inspector_passed": False,
                    "next_agent": "writer",
                }
        response = InspectorMessage(
            content=parsed_data.get("content", "No content provided"),
            inspector_passed=parsed_data.get("inspector_passed", False),
            tool_calls=parsed_data.get("tool_calls", []),
            next_agent=parsed_data.get("next_agent", "director"),
        )

        messages.append(response)
        to_update = {
            "task": state.get("task", ""),
            "messages": messages,
            "director_can_publish": state.get("director_can_publish", False),
            "inspector_passed": response.inspector_passed,
            "writer_scenario": state.get("writer_scenario", ""),
            "current_agent": "inspector",
            "next_agent": response.next_agent,
            "revision": revision,
            "max_revision": self.max_revision,
        }
        return Command(update=to_update)
