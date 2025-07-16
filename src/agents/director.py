import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from src.agents.base_agent import CustomBaseAgent
from src.messages.custom_messages import DirectorMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AgentDirector(CustomBaseAgent):
    def __init__(
        self,
        name: str | None = None,
        system_prompt: str = "",
        model: BaseChatModel | None = None,
        last_n_messages: int = 3,
        max_tries: int = 5,
    ):
        """Initialize the AgentDirector"""
        if name is None:
            name = "director"
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant named Director tasked with directing "
                "to inspector and story writer to create a immersive story."
            )

        super().__init__(name=name, system_prompt=system_prompt, model=model, last_n_messages=last_n_messages)
        # self.model = self.model.with_structured_output(DirectorMessage)  # type: ignore
        self.max_tries = max_tries

    def __call__(self, state: dict[str, Any]) -> Command[Any]:
        """Invoke the director agent with the current state."""

        # ================
        # | Get Revision |
        # ================
        revision = state.get("revision", 1)

        # =======================
        # | Check Max Revisions |
        # =======================
        if self.has_reached_max_revisions(state):
            to_update = {
                "task": state.get("task", ""),
                "current_agent": "director",
                "next_agent": END,
                "writer_scenario": state.get("writer_scenario", ""),
                "inspector_passed": state.get("inspector_passed", False),
                "director_can_publish": False,
                "messages": state["messages"].append(
                    DirectorMessage(content="MAX REVISION REACHED.", director_can_publish=False, next_agent=END)
                ),
                "revision": revision,
                "max_revision": self.max_revision,
            }
            return Command(update=to_update)

        # ====================
        # | Prepare Messages |
        # ====================
        if revision == 1:
            messages = [HumanMessage(content=state["task"])]
            messages_with_system_message = [SystemMessage(content=self.system_prompt)] + messages
        else:
            messages = state["messages"]
            if len(messages) > self.last_n_messages:
                messages = messages[-self.last_n_messages :]
            messages_with_system_message = [SystemMessage(content=self.system_prompt)] + messages

        logger.debug(f"System prompt length: {len(self.system_prompt)}")
        current_try = 0
        parsed_data = {}  # Initialize parsed_data before try block

        while current_try < self.max_tries:
            raw_response = self.model.invoke(messages_with_system_message)

            # Parse JSON from response
            content = raw_response.content
            try:
                # Try to extract JSON from the response
                parsed_data = self.extract_json_from_content(content)
                break  # Exit loop if successful
            except (json.JSONDecodeError, AttributeError) as e:
                current_try += 1

        if not parsed_data:
            # Fallback data - check if inspector passed or failed
            inspector_passed = state.get("inspector_passed", True)
            if inspector_passed and revision:
                # Inspector approved, consider publishing
                parsed_data = {
                    "content": "WARNING: No JSON found, but inspector passed. Story is ready for publishing.",
                    "director_can_publish": True,
                }
            else:
                # Inspector rejected, send back to writer
                parsed_data = {
                    "content": "WARNING: No JSON found, but inspector failed. Please revise the scenario.",
                    "director_can_publish": False,
                }
                # Create DirectorMessage object
        if revision == 1:
            parsed_data["director_can_publish"] = False
        response = DirectorMessage(
            content=parsed_data.get("content"),
            director_can_publish=parsed_data.get("director_can_publish", False),
        )

        # Determine next agent based on can_publish and inspector feedback
        if response.director_can_publish:
            next_agent = END  # End the workflow
        else:
            next_agent = "writer"  # Send back to writer for revision
        response.next_agent = next_agent

        messages.append(response)
        to_update = {
            "tesk": state.get("task", ""),
            "messages": messages,
            "current_agent": "director",
            "next_agent": response.next_agent,
            "director_can_publish": response.director_can_publish,
            "inspector_passed": state.get("inspector_passed", False),
            "writer_scenario": state.get("writer_scenario", ""),
            "revision": revision,
            "max_revision": 10,
        }
        return Command(update=to_update)
