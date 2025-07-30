import torch
from langchain_core.tools import tool
from PIL import Image

from src.models.resnet import ResNet18Model, ResNetPreprocessor


def _init():
    global resnet_model, resnet_preprocessor
    resnet_model = ResNet18Model()
    resnet_model.eval()
    resnet_preprocessor = ResNetPreprocessor()


@tool
def image_classify(images: list[str]) -> str:
    """
    Classify images using a pre-trained ResNet18 model.

    Args:
        images: List of image's path.

    Returns:
        str: Classification results as a string.
    """
    global resnet_model, resnet_preprocessor
    if not resnet_model or not resnet_preprocessor:
        _init()
    processed_images = [resnet_preprocessor(image).unsqueeze(0) for image in images]
    batch = torch.concat(processed_images, dim=0)
    labels = resnet_model.predict_with_label(batch)
    return f"Processed {len(images)} images with labels: {', '.join(labels)}"


