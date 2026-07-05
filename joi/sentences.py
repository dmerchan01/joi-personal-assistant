"""Incremental sentence splitting for streamed LLM text.

Single responsibility: accumulate text deltas and emit complete sentences as
soon as they are available, so TTS can start speaking sentence 1 while the
LLM is still generating sentence 2.
"""
import re

# End of sentence: ., !, ?, or … followed by whitespace. Good enough for
# short spoken replies; abbreviations are rare in Joi's persona style.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


class SentenceBuffer:
    """Feed text deltas in, get finished sentences out."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        """Add a delta; return any sentences completed by it."""
        self._buf += delta
        parts = _SENTENCE_END.split(self._buf)
        self._buf = parts[-1]  # last piece may still be growing
        return [p.strip() for p in parts[:-1] if p.strip()]

    def flush(self) -> str | None:
        """Return whatever remains (the final, unterminated sentence)."""
        rest, self._buf = self._buf.strip(), ""
        return rest or None
