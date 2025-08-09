import logging
import re
from typing import Any

import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Phi4Adapter:
    def __init__(self, model, processor, generation_config, generate_kwargs: dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.processor = processor
        self.generation_config = generation_config
        self.generate_kwargs = generate_kwargs

    def __call__(self, messages, images=[], audios=[]) -> Any:
        return self.invoke(messages, images, audios)

    def invoke(self, messages, images=[], audios=[]):
        inputs = self.process(messages, images, audios)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generate_kwargs, generation_config=self.generation_config)

        generate_ids = outputs[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response

    def process(self, messages, images, audios):
        if not isinstance(messages[0], dict):
            converted_messages = self.convert_to_dict_messages(messages)
        else:
            converted_messages = messages
        template_prompt = self.apply_chat_template(converted_messages)
        inputs = self.processor(template_prompt, images=images, audios=audios, return_tensors="pt")
        return inputs

    def recognize_image_index(self, prompt) -> int:
        """
        Recognizes the index of the image in the prompt.
        The image index is expected to be in the format <|image_{index}|>.
        """
        matches = re.findall(r"<\|image_(\d+)\|>", prompt)
        if matches:
            return int(matches[-1])
        return 1

    def recognize_audio_index(self, prompt) -> int:
        """
        Recognizes the index of the audio in the prompt.
        The audio index is expected to be in the format <|audio_{index}|>.
        """
        matches = re.findall(r"<\|audio_(\d+)\|>", prompt)
        if matches:
            return int(matches[-1])
        return 1

    def image_token(self, index: int) -> str:
        """
        Returns the image token for the given index.
        """
        return f"<|image_{index}|>"

    def audio_token(self, index: int) -> str:
        """
        Returns the audio token for the given index.
        """
        return f"<|audio_{index}|>"

    def user_token(self) -> str:
        """
        Returns the user token.
        """
        return "<|user|>"

    def assistant_token(self) -> str:
        """
        Returns the assistant token.
        """
        return "<|assistant|>"

    def end_token(self) -> str:
        """
        Returns the end token.
        """
        return "<|end|>"

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

    def apply_chat_template(self, messages: list[dict[str, Any]]):
        result = ""
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            if role == "system":
                result += f"<|system|>{content}<|end|>"
            elif role == "user":
                result += f"<|user|>{content}<|end|>"
            elif role == "assistant":
                result += f"<|assistant|>{content}<|end|>"
            else:
                result += f"<|{role}|>{content}<|end|>"

        result += "<|assistant|>"
        return result
