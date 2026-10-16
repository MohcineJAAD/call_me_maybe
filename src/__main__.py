import sys


from .cli import parse_args
from .loader import load_prompts, load_functions
from .decoder import decode
from .writer import write_results

try:
    config = parse_args()
    prompts = load_prompts(config.input)
    functions = load_functions(config.functions_definition)
    calls = []
    for prompt in prompts:
        calls.append(decode(prompt, functions))
    write_results(config.output, calls)
except OSError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
