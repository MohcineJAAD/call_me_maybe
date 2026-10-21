"""Command-line arguments parsing."""

import argparse

from pydantic import BaseModel


class Config(BaseModel):
    """Configuration model for command-line arguments.

    Attributes:
        functions_definition (str): Path to the functions definition JSON file.
        input (str): Path to the input JSON file containing the prompts.
        output (str): Path to the output JSON file where results will be saved.
    """
    functions_definition: str
    input: str
    output: str


def parse_args() -> Config:
    """Parse command-line arguments.

    Returns:
        Config holding the three file paths
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file containing the prompts"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file where results will be saved"
    )
    args = parser.parse_args()
    return Config(
        functions_definition=args.functions_definition,
        input=args.input,
        output=args.output
    )
