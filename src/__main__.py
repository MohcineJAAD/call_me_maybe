import sys


from .cli import parse_args
from .loader import load_prompts, load_functions

try:
    config = parse_args()
    prompts = load_prompts(config.input)
    functions = load_functions(config.functions_definition)
    print(prompts)
    print(functions)
except OSError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
