*This project has been created as part of the 42 curriculum by mjaad.*

# call me maybe

Function calling with a small language model, using constrained decoding.

## Description

Large language models are good at understanding natural language, but small ones are unreliable at producing structured output: asked to answer in JSON, Qwen3-0.6B often returns broken or incomplete JSON, invents values, or keeps writing after the answer.

This project translates natural-language requests into **function calls** with typed arguments. Given the request `"What is the sum of 2 and 3?"` and a list of available functions, the program does not answer `5`: it produces

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The model chooses the function and the values, but it never writes the JSON freely. A **constrained decoder** filters every token the model is allowed to produce, so the output is always valid JSON that matches the schema of `functions_definition.json`: an existing function name, every parameter present, every value of the declared type.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- About 2 GB of free RAM. On the first run, the model (Qwen/Qwen3-0.6B, about 1.2 GB) is downloaded from Hugging Face.

### Installation

```bash
make install        # runs: uv sync
```

### Running

```bash
make run            # runs: uv run python -m src
```

With custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three options are optional; the values above are the defaults. The output folder is created if it does not exist. Progress (one line per prompt, with its time) is printed on stderr.

### Other Makefile rules

| Rule | Action |
|------|--------|
| `make debug` | runs the program under `pdb` |
| `make lint` | `flake8` + `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8` + `mypy --strict` (passes) |
| `make clean` | removes `__pycache__` and `.mypy_cache` |

## Algorithm explanation

### 1. Generation, one token at a time

The model reads a list of token IDs and returns a score (logit) for each of the 151,936 possible next tokens. Normal generation takes the best token, appends it, and repeats.

**Constrained decoding** adds one step before choosing: every token that would break the expected structure gets the score `-inf`, so it can never be chosen. The model still picks the best token, but only among the valid ones.

### 2. Fixed pieces and blanks

An answer always has the same shape:

```
{"name": " | fn_add_numbers | ", "parameters": {"a": | 2 | , "b": | 3 | }}
     C     |       M        |          C          | M |   C    | M | C
```

- **C (fixed)**: the structure, and the parameter names taken from the chosen function's definition. At these positions the schema allows exactly one token sequence, so the decoder writes it directly. This gives the same result as masking every other token (the model would have no choice), without wasting model calls.
- **M (model)**: the function name and every value. Here the model chooses, under a mask.

### 3. The rules of each blank

| Blank | Allowed tokens | How it ends |
|-------|---------------|-------------|
| function name | tokens that keep the text a **prefix** of an existing function name | `"`, allowed only when the text is a complete name |
| number | 0-9, `-` only at the start, `.` at most once | `,` (more parameters follow) or `}` (last one) |
| integer | same as number, without `.` | `,` or `}` |
| string | any token without a quote or a newline | a token starting with `"` |
| boolean | prefix of `true` or `false` | `,` or `}` |

The number rules depend on what has been written so far:

| Written so far | Allowed next |
|----------------|-------------|
| nothing | digit, `-` |
| `-`, or ends with `.` | digit |
| contains `.` | digit, end |
| digits only | digit, `.`, end |

The model decides **when** a value ends, by choosing the end token. The decoder then writes the next fixed piece.

### 4. Choosing a token (`Decoder.pick`)

1. Count the allowed tokens in the mask.
2. **0 allowed**: a bug in the rules, so an error is raised (never pick a random token).
3. **1 allowed**: return it without calling the model (forced token, no cost).
4. **Otherwise**: call the model, set the forbidden scores to `-inf` with `numpy.where`, and return the `argmax`.

### 5. Full walkthrough

```
build_prompt      → chat-format prompt ending with  {"name": "
generate_choice   → f / n / _add / _numbers / "      (the model writes the name)
fixed piece       → , "parameters": {
generate_value(a) → fixed  "a":   then the model writes 2 and picks ,
generate_value(b) → fixed  "b":   then the model writes 3 and picks }
→ FunctionCall(name="fn_add_numbers", parameters={"a": 2.0, "b": 3.0})
```

Values are converted to their Python type (`float`, `int`, `bool`, `str`) and validated by pydantic before being written.

## Design decisions

- **Reading the token table from `vocab.json`.** The vocabulary stores tokens in GPT-2's byte-level form: every byte is written as a visible character, and invisible bytes get a stand-in (a space is written `Ġ`, a newline `Ċ`). `byte_decoder()` rebuilds that mapping and reverses it, so every token is known as its **real bytes**. All checks (is it a digit, does it contain a quote) run on real bytes, and accented characters come out correctly.
- **A pydantic `Decoder` class.** Everything that never changes between prompts (the model, the token table, the masks, the IDs of `"`, `,`, `}`, `-`, `.`) is built once by `load_decoder()` and kept in the decoder. Functions are used for everything that has nothing to remember. All classes use pydantic, as required.
- **numpy masks.** Each blank uses a boolean array of one value per token. Masks that never change (`digit_mask`, `string_mask`) are built once; per-step masks are copies with a few positions switched on.
- **Validation at the boundary.** Input files are validated with pydantic models as soon as they are read: unsupported parameter types (anything other than number, integer, string, boolean) are rejected with a clear message before the model is even loaded. The rest of the program only handles valid, typed data.
- **Prompt in the model's chat format.** Qwen3 is a chat model. The functions (with parameter types and descriptions) and a few generic rules go in the system turn, the request in the user turn, and the answer starts in the assistant turn after an empty think block. This raised the number of fully correct answers from 9/11 to 10/11.
- **Normal JSON spacing, no trailing space.** Fixed pieces are written like ordinary JSON (`"a": -4, "b": 2.5`) but never end with a space, because a space at the end of the input becomes its own token and confuses the model.
- **Safety limits on every value.** Numbers are capped at 20 tokens, strings at the request's length plus 10 tokens, so a model that never ends a value cannot block the program.
- **Errors.** File errors, invalid JSON, invalid data, Ctrl+C and unexpected errors all end with a one-line message on stderr and a non-zero exit code, never a traceback. Messages name the file and, for invalid items, their index.

## Performance analysis

Measured on the 11 example prompts.

| Metric | Result |
|--------|--------|
| Valid JSON, schema respected | **100%** (guaranteed by the decoder) |
| Function chosen correctly | **11/11** |
| Fully correct answers (function and every argument) | **10/11 (90.9%)** |
| Total time (11 prompts) | **about 2 minutes** on a 42 school machine (CPU only) |
| Time per prompt | 4-10 s for simple prompts, 20-30 s for prompts with long string values |

The remaining miss: for "Replace all vowels in 'Programming is fun' with asterisks", the model writes the replacement `*****` (one asterisk per vowel) instead of `*`, because it computes the result itself instead of describing one replacement. The regex it writes is correct.

Speed: each model call costs most of the time, and the SDK recomputes the whole sequence at each call. The decoder avoids model calls wherever the next token is forced (fixed pieces, single-option steps such as closing a complete function name), and only checks the tokens that can appear in a function name when building the name mask.

## Challenges faced

- **The model never stopped some values.** Some string values were repeated until the token limit (`([0-9]+)\s([0-9]+)\s...`), or continued with text copied from the prompt after a new line. Fixes: a repetition guard that stops a value when its last tokens repeat the ones just before them, a stop when the model starts a new line, and a token limit based on the request's length.
- **The repetition guard cut real values.** `233` is written as the tokens `2`, `3`, `3`, so a naive guard cut it to `23`; `blablabla` would have been cut too. Final rule: one token three times in a row, or a block of 2 to 12 tokens twice in a row, and only if the value written so far does **not** appear in the request. Copying a repetition from the request is legitimate; inventing one is not.
- **A trailing space changed the answer.** With the prompt ending on `"a": ` (with a space), the model wrote `2.5` instead of `-4` for "the sum of -4 and 2.5". Ending on `"a":` fixed it. With a squeezed format (`,"parameters":{"a":`), the model wrote `-4.0000000...` forever; normal JSON spacing fixed it.
- **The model escaped its own values.** Writing inside a JSON string, the model sometimes wrote a regex as `\\d` (escaped). `unescape()` removes that escaping once, so the value stored and then written by `json.dump` is the real regex.
- **The model's natural way to end a string was forbidden.** Tokens like `",` (quote + comma) were blocked, so the model kept writing. Any token starting with a quote now ends a string; only the plain quote is written.
- **The first token of a string started with a space** (`" *"` instead of `"*"`), because of how text is tokenized. The first token of a string value can no longer start with a space.
- **Understanding byte-level tokenization** (why a space is `Ġ` and `é` is `Ã©` in `vocab.json`) took several hand-made experiments before `byte_decoder()` made sense.

## Testing strategy

- **Example files**: the 11 prompts of `data/input`, checked by hand against the expected function and values, and by running each function call to compare its result with the expected one.
- **Other types and functions**: a custom functions file with an `integer` parameter and a `boolean` parameter (`times: 3`, `on: true` / `on: false` came out with the right types).
- **Error cases**, each checked for a clean message, exit code 1 and no traceback: missing file, a folder instead of a file, a non-JSON file, invalid JSON, a top-level object instead of a list, an item with a missing field, an unsupported parameter type, an empty functions list, an output path that cannot be written, and Ctrl+C. An empty prompts list produces `[]`.
- **Fresh clone**: the repository cloned into a new folder, then `uv sync`, `make lint` and `make run`.
- **Static checks**: `make lint` and `make lint-strict` both pass.
- **Unit experiments** during development: each part of the decoder (prefix check, number rules, string mask, repetition guard) was first tested alone on small inputs before being added to the program.

## Example usage

```bash
$ make run
[1/11] 5.9s What is the sum of 2 and 3?
[2/11] 10.5s What is the sum of 265 and 345?
...
[11/11] 19.7s Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'

$ cat data/output/function_calling_results.json
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  },
  ...
]
```

Errors:

```bash
$ uv run python -m src --input missing.json
Error: [Errno 2] No such file or directory: 'missing.json'

$ uv run python -m src --functions_definition bad.json
Error: bad.json: invalid data at index 0: 1 validation error for FunctionDef
parameters.a.type
  Input should be 'number', 'string', 'boolean' or 'integer' ...
```

## Resources

- [Neural Networks and Deep Learning, chapter 1 (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/chap1.html): neurons, weights, bias, sigmoid, and how a network outputs one score per possible answer.
- [Neural networks series (3Blue1Brown)](https://www.3blue1brown.com/topics/neural-networks): neural networks, transformers and attention, explained visually.
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/): embeddings, attention, and the transformer block.
- [How Transformer LLMs Work: A Deep Dive into Their Architecture and Evolution (Medium)](https://medium.com/@tharika082003/how-transformer-llms-work-a-deep-dive-into-their-architecture-and-evolution-5a493473833c): how a transformer LLM turns tokens into a prediction of the next token.
- [Efficient Guided Generation for Large Language Models (Willard and Louf, 2023)](https://arxiv.org/abs/2307.09702): constrained decoding, masking the tokens that would break the expected format.

### How AI was used

An AI assistant was used as a tutor throughout the project:

- **Explanations and exercises**: tokenization, logits, `-inf` masking, the state machine of the decoder, byte-level encoding, Unicode and UTF-8, mypy and `Any`, pydantic configuration. Concepts were learned through small exercises run by hand (predict the output, then run it) before writing project code.
- **Design**: discussing the structure of the decoder (fixed pieces and blanks, the `Decoder` class, numpy masks) and the plan of the project.
- **Code review and debugging**: reviewing each piece of code after it was written, and analysing failing runs (endless values, the trailing-space effect, the repetition guard cutting `233`).
- **Documentation**: drafting this README from the project's measured results, then reviewed and edited.
