"""Turn prompts into function calls."""

from .models import FunctionCall, FunctionDef, PromptItem


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


def decode(prompt: PromptItem, functions: list[FunctionDef]) -> FunctionCall:
    """Return a placeholder call (no LLM yet)."""
    return FunctionCall(
        prompt=prompt.prompt,
        name=functions[0].name,
        parameters={}
    )
