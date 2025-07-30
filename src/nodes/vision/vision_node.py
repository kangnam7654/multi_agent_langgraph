import torch
from langchain_core.messages import AIMessage
from PIL import Image


class VisionNode:
    def __init__(self, name, model, preprocessor):
        self.name = name
        self.model = model
        self.preprocessor = preprocessor

    def __call__(self, state):
        if state["images"]:
            processed_images = [
                self.preprocessor(Image.open(image).convert("RGB")).unsqueeze(0) for image in state["images"]
            ]
            batch = torch.concat(processed_images, dim=0)
            labels = self.model.predict_with_label(batch)
            message = f"Processed {len(state['images'])} images with labels: {', '.join(labels)}"
            state["messages"].append(AIMessage(content=message))
            state["images"] = []  # Clear images after processing
        return state
