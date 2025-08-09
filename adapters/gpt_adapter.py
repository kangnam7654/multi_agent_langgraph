import json
import logging
from typing import Any

import requests
import torch
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OllamaAdapter(BaseChatModel):
    model: Any = Field(default=None, description="The underlying model to use for generation.")

    def __init__(
        self,
        model: Any = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model

    @property
    def _llm_type(self) -> str:
        return self.model.__class__.__name__

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ):

        converted_messages = self.convert_to_dict_messages(messages)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=converted_messages,
            stream=True,
        )
        stream_string = []

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                content = data.get("message", {}).get("content")
                if content:
                    stream_string.append(content)
                if data.get("done"):
                    break

        content = "".join(stream_string)
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    def convert_to_dict_messages(self, prompts: list[BaseMessage]) -> list[dict[str, str]]:
        result = []
        for prompt in prompts:
            if isinstance(prompt, SystemMessage):
                role = "system"
            elif isinstance(prompt, HumanMessage):
                role = "user"
            elif isinstance(prompt, AIMessage):
                role = "assistant"
            else:
                role = getattr(prompt, "__name__", "unknown")
            content = prompt.content
            result.append({"role": role, "content": content})
        return result

    def auto_define_device(self) -> str:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device

    def auto_define_dtype(self) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        elif torch.backends.mps.is_available() or (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] <= 7
        ):
            return torch.float16
        else:
            return torch.float32

    def _format_tools_for_prompt(self, tools) -> str:
        """Convert tools to prompt format"""
        if not tools:
            return ""

        tool_descriptions = []
        for tool in tools:
            schema = tool.args_schema.model_json_schema() if hasattr(tool, "args_schema") and tool.args_schema else {}
            tool_desc = f"""
    Function: {tool.name}
    Description: {tool.description}
    Parameters: {json.dumps(schema.get('properties', {}), indent=2)}
    """
            tool_descriptions.append(tool_desc)

        joined_tool_descriptions = "\n".join(tool_descriptions)
        return f"""Available Functions:
    {joined_tool_descriptions}

    When you need to call a function, use this format:
    <function_call>
    {{"name": "function_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
    </function_call>
    """

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        """Bind tools to the model."""
        if not tools:
            return self

        tool_descriptions = self._format_tools_for_prompt(tools)
        if tool_choice:
            tool_descriptions += f"\nTool choice: {tool_choice}"

        logger.info("Tools bound to the model with descriptions:\n%s", tool_descriptions)
        return self
