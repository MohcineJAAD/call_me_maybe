"""Command-line arguments parsing."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        parsed arguments
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
    return parser.parse_args()
