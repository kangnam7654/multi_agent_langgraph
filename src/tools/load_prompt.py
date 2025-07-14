import yaml


def load_prompt(path: str) -> str:
    """
    Load a prompt from the specified path.

    Args:
        path (str): The path to the prompt file.

    Returns:
        str: The content of the prompt file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["system"]
