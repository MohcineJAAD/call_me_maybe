import json

from pydantic import ValidationError

from .cli import parse_args
from .loader import load_prompts

try:
    config = parse_args()
    prompts = load_prompts(config.input)
    print(prompts)
except OSError as e:
    print(f"Error: {e}")
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON: {e}")
except ValidationError as e:
    print(f"Error: invalid data: {e}")
except ValueError as e:
    print(f"Error: {e}")
