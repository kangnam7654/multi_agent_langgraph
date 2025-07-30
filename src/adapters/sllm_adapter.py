import logging
from typing import Any

import torch
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
import json
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TransformersLangChainAdapter(BaseChatModel):
    model: Any = Field(default=None, description="The underlying model to use for generation.")
    tokenizer: Any = Field(default=None, description="The tokenizer to use for the model.")
    model_kwargs: dict = {}
    generate_kwargs: dict = {}
    device: str = ""
    # is_chatmodel: bool = False

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_kwargs: dict = {},
        generate_kwargs: dict = {},
        # is_chatmodel: bool = False,
        device: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.model_kwargs = model_kwargs
        self.generate_kwargs = generate_kwargs
        # self.is_chatmodel = is_chatmodel
        self.device = device or self.auto_define_device()

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
        chat_input = self.tokenize(messages, **kwargs)
        if "max_new_tokens" not in self.generate_kwargs:
            self.generate_kwargs["max_new_tokens"] = 512
        if "pad_token_id" not in self.generate_kwargs:
            self.generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model.generate(**chat_input, **self.generate_kwargs)
        output_ids = outputs[0][len(chat_input.input_ids[0]) :].tolist()
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation], llm_output={"thinking_content": thinking_content})

    def tokenize(self, messages: list[BaseMessage] | list[dict[str, Any]], **kwargs):
        if not isinstance(messages[0], dict):
            converted_messages = self.convert_to_dict_messages(messages)
        else:
            converted_messages = messages
        if kwargs.get("enable_thinking", False):
            try:
                template_prompt = self.tokenizer.apply_chat_template(
                    converted_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True  # type: ignore[reportArgumentTyepe]
                )
            except Exception as e:
                logger.error("Error applying chat template with enable_thinking=True: %s", e)
                template_prompt = self.tokenizer.apply_chat_template(
                    converted_messages, tokenize=False, add_generation_prompt=True  # type: ignore[reportArgumentTyepe]
                )
        else:
            template_prompt = self.tokenizer.apply_chat_template(
                converted_messages, tokenize=False, add_generation_prompt=True  # type: ignore[reportArgumentTyepe]
            )
        chat_input = self.tokenizer(template_prompt, return_tensors="pt").to(self.device)
        return chat_input

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
            schema = (
                tool.args_schema.model_json_schema()
                if hasattr(tool, "args_schema") and tool.args_schema
                else {}
            )
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

        self.generate_kwargs["tools"] = tool_descriptions
        logger.info("Tools bound to the model with descriptions:\n%s", tool_descriptions)
        return self