"""Turn prompts into function calls."""

from .models import FunctionCall, FunctionDef, PromptItem


def decode(prompt: PromptItem, functions: list[FunctionDef]) -> FunctionCall:
    """Return a placeholder call (no LLM yet)."""
    return FunctionCall(
        prompt=prompt.prompt,
        name=functions[0].name,
        parameters={}
    )
