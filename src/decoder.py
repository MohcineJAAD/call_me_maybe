"""Turn prompts into function calls."""

import json
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model

from .models import FunctionCall, FunctionDef, PromptItem

MAX_NUMBER_TOKENS = 20
STRING_TOKEN_MARGIN = 10


def build_prompt(request: str, functions: list[FunctionDef]) -> str:
    """Write the request in the model's chat format, then start the answer.

    Qwen3 is a chat model: it follows instructions best when they come
    in the format it was trained on (system / user / assistant turns).

    Args:
        request: The natural-language prompt.
        functions: The available functions.

    Returns:
        The text given to the model, ending where the function name starts.
    """
    lines = ["Choose the function that answers the request.", "Functions:"]
    for fn in functions:
        parts = []
        for name, spec in fn.parameters.items():
            parts.append(f"{name}: {spec.type}")
        params = ", ".join(parts)
        lines.append(f"- {fn.name}({params}): {fn.description}")
    lines.append("Rules:")
    lines.append("- Copy text values exactly as written in the request.")
    lines.append("- A regex is a general pattern that matches every part "
                 "to replace (for example a character class), "
                 "not a list of examples, and not the spaces around it.")
    lines.append("- The function applies the replacement to each match "
                 "separately, so the replacement is the text for one "
                 "match only.")
    system = "\n".join(lines)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        '{"name": "'
    )


def byte_decoder() -> dict[str, int]:
    """Map each character written in vocab.json back to its real byte.

    Visible bytes are written as themselves. The 68 invisible bytes
    are written as character 256 + their rank (a space becomes 'Ġ').

    Returns:
        Written character to real byte, 256 entries

    """
    real_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    written_as = real_bytes[:]
    extra = 0
    for byte in range(256):
        if byte not in real_bytes:
            real_bytes.append(byte)
            written_as.append(256 + extra)
            extra += 1
    return {
        chr(written_as[i]): real_bytes[i] for i in range(len(real_bytes))
    }


def unescape(raw: str) -> str:
    """Interpret JSON escapes the model may write inside a string.

    The model writes values inside a JSON string, so it may escape
    them (a regex \\d becomes \\\\d). json.loads removes that escaping
    once; if the text is not valid JSON string content, keep it as is.

    Args:
        raw: The text between the quotes, as written by the model.

    Returns:
        The real value.
    """
    try:
        value = json.loads('"' + raw + '"')
    except json.JSONDecodeError:
        return raw
    if isinstance(value, str):
        return value
    return raw


def repeated_block(tokens: list[int]) -> int:
    """Return how many tokens to remove if the last ones are a repeat.

    One token three times in a row, or a block of 2 to 12 tokens twice
    in a row, counts as a repeat.

    Args:
        tokens: The tokens written so far.

    Returns:
        The number of repeated tokens to remove, or 0 if none.
    """
    if len(tokens) >= 3 and tokens[-1] == tokens[-2] == tokens[-3]:
        return 2
    for n in range(2, 13):
        if len(tokens) >= 2 * n and tokens[-n:] == tokens[-2 * n:-n]:
            return n
    return 0


class Decoder(BaseModel):
    """Constrained decoder wrapping the language model.

    It holds everything that never changes between prompts, so it is
    built once and reused for every prompt.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    token_bytes: list[bytes]
    digit_mask: NDArray[np.bool_]
    string_mask: NDArray[np.bool_]
    space_mask: NDArray[np.bool_]
    quote_start_mask: NDArray[np.bool_]
    quote_id: int
    comma_id: int
    close_id: int
    minus_id: int
    dot_id: int

    def encode(self, text: str) -> list[int]:
        """Encode a fixed piece of text into token IDs.

        Args:
            text: The fixed text.

        Returns:
            Its token IDs.
        """
        ids = self.llm.encode(text)[0].tolist()
        return ids

    def pick(self, ids: list[int], mask: NDArray[np.bool_]) -> int:
        """Return the best allowed token, calling the model only if needed.

        Args:
            ids: Everything written so far.
            mask: True for each allowed token.

        Returns:
            The chosen token ID.

        Raises:
            RuntimeError: If no token is allowed (a bug in the rules).
        """
        allowed = np.flatnonzero(mask)
        len_allowed = len(allowed)
        if len_allowed == 0:
            raise RuntimeError("decoder bug: no token allowed")
        if len_allowed == 1:
            return int(allowed[0])
        logits = self.llm.get_logits_from_input_ids(ids)
        size_mask = len(mask)
        scores = np.full(len(logits), -np.inf)
        scores[:size_mask] = np.where(mask, logits[:size_mask], -np.inf)
        return int(np.argmax(scores))

    def generate_choice(
            self,
            ids: list[int],
            options: list[str],
            end_id: int
    ) -> str:
        """Let the model write one of the options, then the end token.

        Args:
            ids: Everything written so far (extended in place).
            options: The allowed words (function names, true/false).
            end_id: The token that closes the choice.

        Returns:
            The chosen option.
        """
        targets = [option.encode() for option in options]
        candidates = [
            token_id for token_id, token in enumerate(self.token_bytes)
            if token and any(token in t for t in targets)
        ]
        written = b""
        while True:
            mask = np.zeros(len(self.token_bytes), dtype=bool)
            for token_id in candidates:
                attempt = written + self.token_bytes[token_id]
                if any(t.startswith(attempt) for t in targets):
                    mask[token_id] = True
            if written in targets:
                mask[end_id] = True
            best = self.pick(ids, mask)
            ids.append(best)
            if best == end_id:
                return written.decode()
            written += self.token_bytes[best]

    def number_mask(self, written: str, end_id: int) -> NDArray[np.bool_]:
        """Return the tokens allowed next in a number.

        Args:
            written: The number written so far.
            end_id: The token that ends the number ("," or "}").

        Returns:
            The mask of allowed tokens.
        """
        mask = self.digit_mask.copy()
        if written == "":
            mask[self.minus_id] = True
        elif written == "-" or written.endswith("."):
            return mask
        elif "." in written:
            mask[end_id] = True
        else:
            mask[end_id] = True
            mask[self.dot_id] = True
        return mask

    def generate_number(self, ids: list[int], end_id: int) -> str:
        """Let the model write a number; it ends by picking end_id.

        Args:
            ids: Everything written so far (extended in place).
            end_id: The token that ends the number ("," or "}").

        Returns:
            The number as text, e.g. "-4" or "2.5".
        """
        written = ""
        for _ in range(MAX_NUMBER_TOKENS):
            best = self.pick(ids, self.number_mask(written, end_id))
            ids.append(best)
            if best == end_id:
                return written
            written += self.token_bytes[best].decode()
        ids.append(end_id)
        return written

    def generate_string(self, ids: list[int], request: str) -> str:
        """Let the model write a string value; it ends by picking '"'.

        Stops early if the model starts a new line, or repeats text that
        is not in the request. Capped at the request's length plus
        STRING_TOKEN_MARGIN tokens, so it can never loop forever.

        Args:
            ids: Everything written so far (extended in place).
            request: The user's request, to tell copying from looping.

        Returns:
            The string value.
        """
        mask = self.string_mask | self.quote_start_mask
        request_bytes = request.encode()
        cap = len(self.encode(request)) + STRING_TOKEN_MARGIN
        chosen: list[int] = []
        repeated = False
        for _ in range(cap):
            if chosen:
                best = self.pick(ids, mask)
            else:
                best = self.pick(ids, mask & ~self.space_mask)
            if self.quote_start_mask[best]:
                break
            chosen.append(best)
            ids.append(best)
            so_far = b"".join(self.token_bytes[t] for t in chosen)
            if b"\\n" in so_far:
                break
            n = repeated_block(chosen)
            if n and so_far not in request_bytes:
                del chosen[-n:]
                del ids[-n:]
                repeated = True
                break
        ids.append(self.quote_id)
        written = b"".join(self.token_bytes[t] for t in chosen)
        written = written.split(b"\\n")[0]
        value = unescape(written.decode("utf-8", errors="replace"))
        if repeated:
            value = value.rstrip("|, ")
        return value

    def generate_value(
            self,
            ids: list[int],
            key: str,
            kind: str,
            end_id: int,
            first: bool,
            request: str
    ) -> str | float | int | bool:
        """Write the key, then let the model fill a value of this type.

        Args:
            ids: Everything written so far (extended in place).
            key: The parameter name, from the definition.
            kind: The parameter type, from the definition.
            end_id: "," if more parameters follow, "}" if last.
            first: True for the first parameter (no space before it).
            request: The user's request, passed to generate_string().

        Returns:
            The value, converted to its Python type.
        """
        space = "" if first else " "
        if kind == "string":
            ids += self.encode(f'{space}"{key}": "')
            value = self.generate_string(ids, request)
            ids.append(end_id)
            return value
        ids += self.encode(f'{space}"{key}":')
        if kind == "boolean":
            choice = self.generate_choice(ids, ["true", "false"], end_id)
            return choice == "true"
        text = self.generate_number(ids, end_id)
        if kind == "integer":
            return int(float(text))
        return float(text)

    def decode(
            self,
            item: PromptItem,
            functions: list[FunctionDef]
    ) -> FunctionCall:
        """Turn one prompt into a valid function call.

        Args:
            item: The prompt.
            functions: The available functions.

        Returns:
            The function call, with typed arguments.
        """
        ids = self.encode(build_prompt(item.prompt, functions))
        names = [fn.name for fn in functions]
        name = self.generate_choice(ids, names, self.quote_id)
        fn = functions[names.index(name)]

        ids += self.encode(', "parameters": {')
        parameters: dict[str, str | float | int | bool] = {}
        keys = list(fn.parameters)
        for index, key in enumerate(keys):
            last = index == len(keys) - 1
            end_id = self.close_id if last else self.comma_id
            kind = fn.parameters[key].type
            parameters[key] = self.generate_value(
                ids, key, kind, end_id, index == 0, item.prompt
            )
        return FunctionCall(
            prompt=item.prompt,
            name=name,
            parameters=parameters
        )


def load_decoder(llm: Small_LLM_Model) -> Decoder:
    """Build every lookup table once and return a ready decoder.

    Args:
        llm: The loaded language model.

    Returns:
        The decoder, with the token table, masks and special IDs.
    """
    with open(llm.get_path_to_vocab_file(), encoding="utf-8") as f:
        vocab = json.load(f)

    to_byte = byte_decoder()
    token_bytes = [b""] * (max(vocab.values()) + 1)
    for written, token_id in vocab.items():
        token_bytes[token_id] = bytes([to_byte[c] for c in written])
    digit_mask = np.array([token.isdigit() for token in token_bytes])
    space_mask = np.array([token.startswith(b" ") for token in token_bytes])
    quote_start_mask = np.array(
        [token.startswith(b'"') for token in token_bytes]
    )
    allowed_in_string = []
    for token in token_bytes:
        if token == b"":
            allowed_in_string.append(False)
        elif b'"' in token:
            allowed_in_string.append(False)
        elif b"\n" in token or b"\r" in token:
            allowed_in_string.append(False)
        else:
            allowed_in_string.append(True)
    string_mask = np.array(allowed_in_string)
    quote_id = vocab['"']
    comma_id = vocab[","]
    close_id = vocab["}"]
    minus_id = vocab["-"]
    dot_id = vocab["."]
    return Decoder(
        llm=llm,
        token_bytes=token_bytes,
        digit_mask=digit_mask,
        string_mask=string_mask,
        space_mask=space_mask,
        quote_start_mask=quote_start_mask,
        quote_id=quote_id,
        comma_id=comma_id,
        close_id=close_id,
        minus_id=minus_id,
        dot_id=dot_id
    )
