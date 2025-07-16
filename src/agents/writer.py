import json
import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from src.agents.base_agent import CustomBaseAgent
from src.messages.custom_messages import WriterMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AgentWriter(CustomBaseAgent):
    def __init__(
        self,
        name: str | None = None,
        system_prompt: str | None = None,
        model: BaseChatModel | None = None,
        last_n_messages: int = 3,
        max_tries: int = 5,
    ):
        """Initialize the AgentScenarioWriter"""
        if name is None:
            name = "writer"
        if system_prompt is None:
            system_prompt = "You are a helpful assistant named Scenario Writer tasked with writing the story based on the provided outline."

        super().__init__(name=name, system_prompt=system_prompt, model=model, last_n_messages=last_n_messages)
        # self.model = self.model.with_structured_output(WriterMessage)  # type: ignore
        self.max_tries = max_tries

    def __call__(self, state: dict[str, Any]) -> Command[Any]:
        revision: int = state["revision"] + 1

        if self.has_reached_max_revisions(state):
            to_update = {
                "task": state.get("task", ""),
                "messages": state["messages"].append(
                    WriterMessage(
                        content="MAX REVISION REACHED.",
                        writer_scenario=state.get("writer_scenario", ""),
                        next_agent="inspector",
                    )
                ),
                "director_can_publish": state.get("director_can_publish", False),
                "inspector_passed": state.get("inspector_passed", False),
                "writer_scenario": state.get("writer_scenario", ""),
                "current_agent": "writer",
                "next_agent": "inspector",
                "revision": revision,
                "max_revision": self.max_revision,
            }
            return Command(update=to_update)
        messages: list = state["messages"]
        task = messages[0]
        if len(messages) > self.last_n_messages:
            messages = messages[-self.last_n_messages :]
        messages_with_system_message = [SystemMessage(content=self.system_prompt), task] + messages

        current_try = 0
        parsed_data = {}
        while current_try < self.max_tries:
            # Get raw response
            raw_response = self.model.invoke(messages_with_system_message)

            # Parse JSON from response
            content = raw_response.content
            try:
                # Try to extract JSON from the response
                parsed_data = self.extract_json_from_content(content)
                break
            except (json.JSONDecodeError, AttributeError) as e:
                current_try += 1

        if not parsed_data:
            parsed_data = {
                "content": "WARNING: Failed to parse JSON from response. Using fallback content.",
                "writer_scenario": "ERROR",
                "tool_calls": [],
            }
        response = WriterMessage(
            content=parsed_data.get("content"),
            writer_scenario=parsed_data.get("writer_scenario", ""),
            tool_calls=parsed_data.get("tool_calls", []),
        )

        if response.tool_calls:
            next_agent = "self"
        else:
            next_agent = "inspector"

        response.next_agent = next_agent
        messages.append(response)
        to_update = {
            "task": state.get("task", ""),
            "messages": messages,
            "director_can_publish": state.get("director_can_publish", False),
            "inspector_passed": state.get("inspector_passed", True),
            "writer_scenario": response.writer_scenario,
            "current_agent": "writer",
            "next_agent": next_agent,
            "revision": revision,
            "max_revision": self.max_revision,
        }
        return Command(update=to_update)
