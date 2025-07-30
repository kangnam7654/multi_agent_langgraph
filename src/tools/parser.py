import json
import re


def find_json_in_content(content: str, required_keys: list = None) -> dict:
    """
    Extract JSON data from a string content.

    Args:
        content (str): The string content to search for JSON data.
        required_keys (list, optional): List of required keys to validate in JSON.

    Returns:
        dict: Parsed JSON data if found, otherwise an empty dictionary.
    """
    open_brace_exist = content.count("{") > 0
    close_brace_exist = content.count("}") > 0

    json_str = None

    if open_brace_exist and close_brace_exist:  # Check if both braces exist
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group()

    elif open_brace_exist and not close_brace_exist:  # Only open brace exists
        # Find from first { to end of content and try to add }
        start_idx = content.find("{")
        incomplete_json = content[start_idx:]
        json_str = incomplete_json + "}"

    elif not open_brace_exist and close_brace_exist:  # Only close brace exists
        # Find from start to last } and try to add {
        end_idx = content.rfind("}") + 1
        incomplete_json = content[:end_idx]
        json_str = "{" + incomplete_json

    if json_str:
        try:
            parsed_json = json.loads(json_str)

            # Validate required keys if provided
            if required_keys:
                missing_keys = [key for key in required_keys if key not in parsed_json]
                if missing_keys:
                    print(f"Warning: Missing required keys: {missing_keys}")

            return parsed_json
        except json.JSONDecodeError:
            return {}

    return {}
