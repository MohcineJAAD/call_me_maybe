"""Writing the function calls to the output JSON file."""

import json
import os

from .models import FunctionCall


def write_results(file_path: str, calls: list[FunctionCall]) -> None:
    """Write function calls to a JSON file.

    Args:
        file_path: Path to the output JSON file.
        calls: The function calls to write.
    """
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    data = [call.model_dump() for call in calls]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
