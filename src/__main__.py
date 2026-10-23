import sys
import time

from llm_sdk import Small_LLM_Model

from .cli import parse_args
from .decoder import load_decoder
from .loader import load_prompts, load_functions
from .writer import write_results

try:
    config = parse_args()
    prompts = load_prompts(config.input)
    functions = load_functions(config.functions_definition)
    decoder = load_decoder(Small_LLM_Model())
    calls = []
    for i, prompt in enumerate(prompts, 1):
        start = time.time()
        calls.append(decoder.decode(prompt, functions))
        print(
            f"[{i}/{len(prompts)}] {time.time() -start:.1f}s "
            f"{prompt.prompt}", file=sys.stderr
        )
    write_results(config.output, calls)
except OSError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
except KeyboardInterrupt:
    print("\nInterrupted by user.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}", file=sys.stderr)
    sys.exit(1)
