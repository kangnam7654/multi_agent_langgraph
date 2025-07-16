import json
import re


def find_json_in_content(content: str) -> dict:
    """
    Extract JSON data from a string content.

    Args:
        content (str): The string content to search for JSON data.

    Returns:
        dict: Parsed JSON data if found, otherwise an empty dictionary.
    """
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}
    return {}
