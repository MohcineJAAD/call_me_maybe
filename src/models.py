"""Data models for the input and output JSON files."""

from pydantic import BaseModel


class TypeSpec(BaseModel):
    """Type of a parameter or return value, e.g. {"type": "number"}."""
    type: str


class FunctionDef(BaseModel):
    """Definition of a function."""
    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class PromptItem(BaseModel):
    """One prompt from function_calling_tests.json."""
    prompt: str


class FunctionCall(BaseModel):
    """One result: a call to a function with its arguments."""
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
