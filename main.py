from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapters.sllm_adapter import TransformersLangChainAdapter
from src.models.resnet import ResNet18Model, ResNetPreprocessor
from src.nodes.vision.sllm_node import LLMNode
from src.nodes.vision.vision_node import VisionNode


class AgentVisionState(TypedDict):
    messages: list[Any]
    images: list[Any]
    revision: int
    current: str
    next: str


def exist_image_input(state: AgentVisionState) -> bool:
    """Check if there are any images in the state."""
    return bool(state["images"])


def main():
    with open("src/prompts/vision/vision_llm_v1.md", "r") as f:
        system_prompt = f.read()
    resnet_model = ResNet18Model()
    resnet_preprocessor = ResNetPreprocessor()
    node_vision = VisionNode("resnet18", resnet_model, resnet_preprocessor)

    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto", device_map="auto")
    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    node_llm = LLMNode(
        "llm", TransformersLangChainAdapter(model=hf_model, tokenizer=hf_tokenizer), system_prompt=system_prompt
    )

    builder = StateGraph(AgentVisionState)
    builder.add_node("llm", node_llm)
    builder.add_node("vision", node_vision)

    builder.set_entry_point("llm")
    builder.add_edge("vision", "llm")
    builder.add_conditional_edges("llm", exist_image_input, {True: "vision", False: END})
    graph = builder.compile()

    for chunk in graph.stream(
        AgentVisionState(
            messages=[HumanMessage(content="Process these images")],
            images=["cat.jpg"],  # Example image paths
            revision=0,
            current="llm",
            next="vision",
        )
    ):
        for _, event in chunk.items():
            print(event.get("messages", [])[-1].content)


if __name__ == "__main__":
    main()
