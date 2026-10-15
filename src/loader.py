"""Loading and Validating the input JSON files."""

import json

from src.models import PromptItem


def load_prompts(file_path: str) -> list[PromptItem]:
    """Load and validate the prompts file.

    Args:
        file_path: path to the prompts JSON file.

    Returns:
        The validated prompts.
    """
    with open(file_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list at top level")
    prompts = []
    for item in data:
        prompts.append(PromptItem.model_validate(item))
    return prompts
