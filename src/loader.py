"""Loading and Validating the input JSON files."""

import json
from typing import Any

from pydantic import ValidationError

from src.models import PromptItem, FunctionDef


def read_json_list(file_path: str) -> list[Any]:
    """Read a JSON file and check its top level is a list.

    Args:
        file_path: path to the JSON file.

    Returns:
        the parsed list.
    """
    with open(file_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{file_path}: invalid JSON: {e}")
    if not isinstance(data, list):
        raise ValueError(f"{file_path}: expected a JSON list at top level")
    return data


def load_prompts(file_path: str) -> list[PromptItem]:
    """Load and validate the prompts file.

    Args:
        file_path: path to the prompts JSON file.

    Returns:
        The validated prompts.
    """
    prompts = []
    for i, item in enumerate(read_json_list(file_path)):
        try:
            prompts.append(PromptItem.model_validate(item))
        except ValidationError as e:
            raise ValueError(f"{file_path}: invalid data at index {i}: {e}")
    return prompts


def load_functions(file_path: str) -> list[FunctionDef]:
    """Load and validate the functions definition file.

    Args:
        file_path: path to the functions definition JSON file.

    Returns:
        The validated functions definition.
    """
    functions = []
    for i, item in enumerate(read_json_list(file_path)):
        try:
            functions.append(FunctionDef.model_validate(item))
        except ValidationError as e:
            raise ValueError(f"{file_path}: invalid data at index {i}: {e}")
    if not functions:
        raise ValueError(f"{file_path}: no functions defined")
    return functions
