import json
import logging
import re
from typing import Any, Callable

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
        # self.model = self.model.with_structured_output(DirectorMessage)  # type: ignore

    def __call__(self, state: dict[str, Any]) -> Command[Any]:
        """Invoke the director agent with the current state."""

        # ================
        # | Get Revision |
        # ================
        revision = state.get("revision", 0)

        # =======================
        # | Check Max Revisions |
        # =======================
        if self.has_reached_max_revisions(state):
            return Command(goto=END, update={"messages": state["messages"], "revision": revision})

        # ====================
        # | Prepare Messages |
        # ====================
        if revision == 0:
            messages = [HumanMessage(content=state["task"])]
            messages_with_system_message = [SystemMessage(content=self.system_prompt)] + messages
        else:
            messages = state["messages"]
            messages_with_system_message = (
                [SystemMessage(content=self.system_prompt)] + messages + [f'inspector pass{state.get("passed", True)}']
            )

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
                    # Fallback data - check if inspector passed or failed
                    inspector_passed = state.get("passed", True)
                    if inspector_passed:
                        # Inspector approved, consider publishing
                        parsed_data = {
                            "type": "director",
                            "content": "The scenario has been approved by the inspector. Story is ready for publishing.",
                            "can_publish": True,
                            "next_agent": None,
                        }
                    else:
                        # Inspector rejected, send back to writer
                        parsed_data = {
                            "type": "director",
                            "content": f"The inspector found issues. Please revise the scenario. Inspector feedback: {content}",
                            "can_publish": False,
                            "next_agent": "writer",
                        }

                # Create DirectorMessage object
                response = DirectorMessage(
                    type=parsed_data.get("type", "director"),
                    content=parsed_data.get("content", "Please proceed with the task"),
                    can_publish=parsed_data.get("can_publish", False),
                )

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Failed to parse JSON, using fallback: {e}")
                # Use inspector feedback to decide
                inspector_passed = state.get("passed", True)
                if inspector_passed:
                    response = DirectorMessage(
                        type="director",
                        content="The scenario has been approved. Story is ready for publishing.",
                        can_publish=True,
                    )
                else:
                    response = DirectorMessage(
                        type="director",
                        content=f"Please revise the scenario based on inspector feedback: {content}",
                        can_publish=False,
                    )

        except Exception as e:
            print(f"Error invoking model: {e}")
            print(
                f"Messages: {[msg.content[:100] if hasattr(msg, 'content') else str(msg)[:100] for msg in messages_with_system_message]}"
            )
            # Create fallback response
            response = DirectorMessage(
                type="director",
                content="Error occurred. Please proceed with writing the scenario.",
                can_publish=False,
            )

        messages.append(response)
        revision += 1

        # Determine next agent based on can_publish and inspector feedback
        if response.can_publish:
            next_agent = None  # End the workflow
        else:
            next_agent = "writer"  # Send back to writer for revision

        to_update = {
            "messages": messages,
            "current_agent": response.type.lower(),
            "next_agent": next_agent,
            "can_publish": response.can_publish,
            "revision": revision,
            "max_revision": 10,
        }
        return Command(update=to_update)
