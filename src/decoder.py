"""Turn prompts into function calls."""

import json
import numpy as np
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model

from .models import FunctionCall, FunctionDef, PromptItem


def build_prompt(request: str, functions: list[FunctionDef]) -> str:
    """Describe the functions, then start the JSON answer.

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
    lines.append(f"Request: {request}")
    lines.append('Answer: {"name": "')
    return "\n".join(lines)


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


class Decoder(BaseModel):
    """Constrained decoder wrapping the language model.

    It holds everything that never changes between prompts, so it is
    built once and reused for every prompt.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    token_bytes: list[bytes]
    digit_mask: np.ndarray
    string_mask: np.ndarray
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

    def pick(self, ids: list[int], mask: np.ndarray) -> int:
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
        written = b""
        while True:
            mask = np.zeros(len(self.token_bytes), dtype=bool)
            for token_id, token in enumerate(self.token_bytes):
                attempt = written + token
                if token and any(t.startswith(attempt) for t in targets):
                    mask[token_id] = True
            if written in targets:
                mask[end_id] = True
            best = self.pick(ids, mask)
            ids.append(best)
            if best == end_id:
                return written.decode()
            written += self.token_bytes[best]


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
        quote_id=quote_id,
        comma_id=comma_id,
        close_id=close_id,
        minus_id=minus_id,
        dot_id=dot_id
    )


def decode(prompt: PromptItem, functions: list[FunctionDef]) -> FunctionCall:
    """Return a placeholder call (no LLM yet)."""
    return FunctionCall(
        prompt=prompt.prompt,
        name=functions[0].name,
        parameters={}
    )
