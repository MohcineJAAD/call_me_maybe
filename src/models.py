"""Data module for the input JSON files."""

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
