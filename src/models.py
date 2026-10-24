"""Data models for the input and output JSON files."""
import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_PROMPT_LENGTH = 1
MAX_PROMPT_LENGTH = 1000
MAX_NUMBER = 9223372036854775807


class TypeSpec(BaseModel):
    """Type of a parameter or return value, e.g. {"type": "number"}."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["number", "string", "boolean", "integer"]


class FunctionDef(BaseModel):
    """Definition of a function."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str
    description: str = Field(min_length=1)
    parameters: dict[str, TypeSpec]
    returns: TypeSpec

    @field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        """Refuse names that are empty or not valid identifiers, like '0fn'.

        Args:
            name: The function name from the definition.

        Returns:
            The same name, if it is valid.

        Raises:
            ValueError: If the name is not a valid identifier.
        """
        if not name.isidentifier():
            raise ValueError(f"invalid function name: {name!r}")
        return name

    @field_validator("parameters")
    @classmethod
    def check_parameter_names(
            cls,
            parameters: dict[str, TypeSpec]
    ) -> dict[str, TypeSpec]:
        """Refuse parameter names that are not valid identifiers.

        Args:
            parameters: Parameter name to type, from the definition.

        Returns:
            The same parameters, if every name is valid.

        Raises:
            ValueError: If a parameter name is not a valid identifier.
        """
        for key in parameters:
            if not key.isidentifier():
                raise ValueError(f"invalid parameter name: {key!r}")
        return parameters


class PromptItem(BaseModel):
    """One prompt from function_calling_tests.json."""
    model_config = ConfigDict(extra="forbid")
    prompt: str

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, prompt: str) -> str:
        """Refuse empty, too long, or prompts with too-large numbers.

        Args:
            prompt: The prompt text from the input file.

        Returns:
            The same prompt, unchanged (spaces are kept).

        Raises:
            ValueError: If the prompt is empty or only spaces, longer
                than MAX_PROMPT_LENGTH, or contains a number larger
                than MAX_NUMBER.
        """
        if len(prompt.strip()) < MIN_PROMPT_LENGTH:
            raise ValueError("prompt is empty")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"prompt is longer than {MAX_PROMPT_LENGTH} characters"
            )
        for number in re.findall(r"\d+(?:\.\d+)?", prompt):
            if Decimal(number) > MAX_NUMBER:
                raise ValueError(f"number too large: {number[:20]}...")
        return prompt


class FunctionCall(BaseModel):
    """One result: a call to a function with its arguments."""
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
